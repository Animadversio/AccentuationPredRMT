"""
Validate the three-term RMT formula at large d (1000–10000) using GPU acceleration.

Strategy for large d:
- Kernel/dual form of ridge estimator: O(n²d) instead of O(nd²)
- Randomized SVD for Van Hateren covariance eigendecomposition
- Only validate top-K PCs (full eigenvector matrix too large to store)
- All heavy compute on CUDA A100

Usage examples:
  # Isotropic d=2000
  python validate_large_d.py --spectrum isotropic --d 2000 --n 1000 --lam 0.05

  # Power-law d=5000
  python validate_large_d.py --spectrum powerlaw --d 5000 --n 2000 --lam 0.05 --alpha 1.0

  # Van Hateren 32x32 patches (d=1024)
  python validate_large_d.py --spectrum vanhateren --patch_hw 32 --n 2000 --lam 0.05

  # Van Hateren 64x64 patches (d=4096)
  python validate_large_d.py --spectrum vanhateren --patch_hw 64 --n 3000 --lam 0.05 --top_k 200

  # Van Hateren 100x100 patches (d=10000)
  python validate_large_d.py --spectrum vanhateren --patch_hw 100 --n 5000 --lam 0.05 --top_k 200
"""
import argparse
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rmt_core import get_spectrum, SpectrumKappa, ridge_error_per_pc_theory
from rmt_core.gpu_simulation_lib import run_monte_carlo_gpu, get_device


FIGDIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spectrum', choices=['isotropic', 'powerlaw', 'vanhateren'],
                        default='vanhateren')
    parser.add_argument('--d', type=int, default=1024,
                        help='Dimension for isotropic/powerlaw')
    parser.add_argument('--n', type=int, default=2000)
    parser.add_argument('--lam', type=float, default=0.05)
    parser.add_argument('--sigma', type=float, default=0.5)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--patch_hw', type=int, default=32,
                        help='Patch size for vanhateren (d = patch_hw²)')
    parser.add_argument('--n_patches', type=int, default=500,
                        help='Patches per image for vanhateren')
    parser.add_argument('--n_images', type=int, default=100)
    parser.add_argument('--top_k', type=int, default=None,
                        help='Validate only top-K PCs (default: min(d, 300))')
    parser.add_argument('--n_trials', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no_gpu', action='store_true')
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    device = get_device(use_gpu=not args.no_gpu)
    print(f"Device: {device}")

    # ── Load spectrum ──
    print(f"Loading spectrum: {args.spectrum} ...")
    t0 = time.time()
    top_k_req = args.top_k or 300

    if args.spectrum == 'vanhateren':
        from rmt_core.spectrum_lib import vanhateren_spectrum
        d_full = args.patch_hw ** 2
        eigenvalues_all, eigenvectors_topk, _ = vanhateren_spectrum(
            patch_hw=args.patch_hw,
            n_patches_per_image=args.n_patches,
            n_images=args.n_images,
            use_randomized_svd=True,
            top_k=top_k_req,
            rng=rng,
        )
        # residual variance: mean of eigenvalues beyond top-K
        # Since we only have top-K from SVD, estimate residual from trace
        # trace(Σ) = sum of all eigenvalues ≈ mean pixel var × d
        # Approximate: residual_var ≈ 0 (top-K captures most variance)
        residual_var = 0.0
        eigenvalues = eigenvalues_all  # top-K only
    else:
        # isotropic/powerlaw: diagonal covariance, identity eigenvectors
        # Use identity eigenvectors (canonical basis = top-K coords)
        from rmt_core.spectrum_lib import isotropic_spectrum, powerlaw_spectrum
        d_full = args.d
        if args.spectrum == 'isotropic':
            eigenvalues_all, _ = isotropic_spectrum(d_full)
        else:
            eigenvalues_all, _ = powerlaw_spectrum(d_full, alpha=args.alpha)
        top_k_req = min(top_k_req, d_full)
        eigenvectors_topk = np.eye(d_full, top_k_req)  # (d, K) canonical basis
        eigenvalues = eigenvalues_all  # full spectrum for κ computation
        residual_var = 0.0

    print(f"  Spectrum loaded in {time.time()-t0:.1f}s")

    K = eigenvectors_topk.shape[1]
    top_k = K
    eigenvalues_topk = eigenvalues[:top_k]

    n, lam, sigma = args.n, args.lam, args.sigma
    # For vanhateren low-rank case: effective γ = K/n (rank of data / n)
    # For isotropic/powerlaw diagonal: γ = d/n (full dimensional)
    if args.spectrum == 'vanhateren':
        gamma = top_k / n  # rank-K Gaussian approximation
        print(f"  d={d_full}, n={n}, K={top_k}, γ_eff=K/n={gamma:.3f}")
    else:
        gamma = d_full / n
        print(f"  d={d_full}, n={n}, γ=d/n={gamma:.3f}, K (PCs validated)={top_k}")
    print(f"  Top-5 eigenvalues: {eigenvalues[:5]}")

    # True coefficient (full d-dimensional)
    beta_star = rng.standard_normal(d_full)
    beta_proj_topk = eigenvectors_topk.T @ beta_star   # (top_k,)
    # Full projections for accurate df₂ and C_sig in theory formula
    if args.spectrum in ('isotropic', 'powerlaw'):
        beta_proj_full = beta_star  # diagonal: projections = coordinates
    else:
        beta_proj_full = beta_proj_topk  # vanhateren: only top-K available

    # ── κ(λ) ──  use full spectrum for accurate κ
    print("Solving κ(λ) ...")
    kappa_fn = SpectrumKappa(eigenvalues, gamma)
    kappa = kappa_fn(lam)
    print(f"  κ({lam}) = {kappa:.5f}")

    # ── Theory (top-K PCs, but use full spectrum for global quantities) ──
    err_theory, t1, t2, t3 = ridge_error_per_pc_theory(
        eigenvalues_topk, beta_proj_topk, kappa, sigma, n,
        eigenvalues_full=eigenvalues, beta_proj_full=beta_proj_full)

    # ── Monte Carlo (GPU) ──
    print(f"Running {args.n_trials} MC trials on {device} ...")
    t0 = time.time()
    # For diagonal spectra: pass full eigenvalues for data generation, eigenvectors_topk only for projection
    if args.spectrum in ('isotropic', 'powerlaw'):
        mc = run_monte_carlo_gpu(
            n, eigenvalues, eigenvectors_topk, beta_star, sigma, lam,
            n_trials=args.n_trials, diagonal=True,
            device=device, rng=rng, verbose=True)
    else:
        mc = run_monte_carlo_gpu(
            n, eigenvalues_topk, eigenvectors_topk, beta_star, sigma, lam,
            n_trials=args.n_trials, residual_var=residual_var,
            device=device, rng=rng, verbose=True)
    print(f"  MC done in {time.time()-t0:.1f}s")
    err_sim = mc['error_per_pc']

    print(f"Total gen error (top-K)  — theory: {np.sum(eigenvalues_topk*err_theory):.5f}  "
          f"MC: {mc['total_gen_error']:.5f}")
    print(f"Accentuation  — alignment: {mc['acc_alignment']:.4f}  error: {mc['acc_error']:.4f}")

    # ── Figure ──
    os.makedirs(FIGDIR, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle(
        f'spectrum={args.spectrum}  d={d_full}, n={n}, λ={lam}, σ={sigma}  '
        f'(top {top_k} PCs)',
        fontsize=11)

    ax = axes[0]
    ax.semilogy(eigenvalues, '-', lw=1)
    ax.set_xlabel('PC index k'); ax.set_ylabel('λ_k')
    ax.set_title('Data covariance spectrum')

    ax = axes[1]
    ax.semilogy(err_sim, 'o', ms=3, alpha=0.6, label='MC')
    ax.semilogy(err_theory, '-', lw=1.5, label='Theory total')
    ax.semilogy(t1, '--', lw=1, label='Term1: overshrinkage')
    ax.semilogy(t2, '--', lw=1, label='Term2: finite-n signal')
    ax.semilogy(t3, '--', lw=1, label='Term3: finite-n noise')
    ax.set_xlabel('PC index k (top-K only)')
    ax.set_ylabel('E[(uₖᵀΔβ)²]')
    ax.set_title(f'Per-PC error  d={d_full}')
    ax.legend(fontsize=7)

    ax = axes[2]
    ax.loglog(err_theory, err_sim, 'o', ms=3, alpha=0.6)
    lims = [min(err_theory.min(), err_sim.min()), max(err_theory.max(), err_sim.max())]
    ax.plot(lims, lims, 'k--', lw=1, label='y=x')
    ax.set_xlabel('RMT theory'); ax.set_ylabel('MC')
    ax.set_title('Theory vs MC scatter')
    ax.legend()

    plt.tight_layout()
    tag = f'{args.spectrum}_d{d_full}_n{n}_lam{lam}'
    if args.spectrum == 'vanhateren':
        tag += f'_patch{args.patch_hw}'
    figpath = os.path.join(FIGDIR, f'ridge_error_large_{tag}.png')
    plt.savefig(figpath, dpi=150, bbox_inches='tight')
    print(f"Saved: {figpath}")
    plt.close()


if __name__ == '__main__':
    main()

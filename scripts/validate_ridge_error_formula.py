"""
Validate the three-term RMT formula for per-PC ridge regression error.

E[(uₖᵀ(β̂-β*))²] ≍  Term1 (overshrinkage) + Term2 (finite-sample signal) + Term3 (finite-sample noise)

Supports three data spectra: isotropic, powerlaw, vanhateren.

Usage examples:
  python validate_ridge_error_formula.py --spectrum isotropic --d 64 --n 200
  python validate_ridge_error_formula.py --spectrum powerlaw --d 100 --n 200 --alpha 1.0
  python validate_ridge_error_formula.py --spectrum vanhateren --n 500 --patch_hw 8
"""
import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rmt_core import (
    get_spectrum,
    SpectrumKappa,
    ridge_error_per_pc_theory,
    run_monte_carlo,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spectrum', choices=['isotropic', 'powerlaw', 'vanhateren'],
                        default='powerlaw')
    parser.add_argument('--d', type=int, default=64,
                        help='Dimension (ignored for vanhateren, set by patch_hw)')
    parser.add_argument('--n', type=int, default=200, help='Number of samples')
    parser.add_argument('--lam', type=float, default=0.1, help='Ridge penalty')
    parser.add_argument('--sigma', type=float, default=0.5, help='Noise std')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Power-law exponent (powerlaw only)')
    parser.add_argument('--patch_hw', type=int, default=8,
                        help='Patch size for vanhateren (d = patch_hw²)')
    parser.add_argument('--n_patches', type=int, default=200,
                        help='Patches per image for vanhateren')
    parser.add_argument('--n_images', type=int, default=50,
                        help='Number of images for vanhateren')
    parser.add_argument('--n_trials', type=int, default=1000, help='MC trials')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # Load spectrum
    print(f"Loading spectrum: {args.spectrum} ...")
    vh_kwargs = dict(patch_hw=args.patch_hw,
                     n_patches_per_image=args.n_patches,
                     n_images=args.n_images)
    eigenvalues, eigenvectors = get_spectrum(
        args.spectrum, d=args.d, alpha=args.alpha, rng=rng,
        **(vh_kwargs if args.spectrum == 'vanhateren' else {}))

    d = len(eigenvalues)
    n, lam, sigma = args.n, args.lam, args.sigma
    gamma = d / n
    print(f"  d={d}, n={n}, γ={gamma:.3f}, λ={lam}, σ={sigma}")
    print(f"  Top-5 eigenvalues: {eigenvalues[:5]}")

    # True coefficient
    beta_star = rng.standard_normal(d)
    beta_proj = eigenvectors.T @ beta_star

    # κ(λ)
    kappa_fn = SpectrumKappa(eigenvalues, gamma)
    kappa = kappa_fn(lam)
    print(f"  κ({lam}) = {kappa:.4f}")

    # Theory
    err_theory, t1, t2, t3 = ridge_error_per_pc_theory(
        eigenvalues, beta_proj, kappa, sigma, n)

    # Monte Carlo
    print(f"Running {args.n_trials} MC trials ...")
    mc = run_monte_carlo(n, eigenvalues, eigenvectors, beta_star, sigma, lam,
                         n_trials=args.n_trials, rng=rng)
    err_sim = mc['error_per_pc']

    print(f"Total gen error  — theory: {np.sum(eigenvalues*err_theory):.5f}  "
          f"MC: {mc['total_gen_error']:.5f}")
    print(f"Accentuation     — alignment: {mc['acc_alignment']:.4f}  "
          f"error: {mc['acc_error']:.5f}")

    # ── Figure ──
    os.makedirs(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures'),
                exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle(f'spectrum={args.spectrum}  d={d}, n={n}, λ={lam}, σ={sigma}',
                 fontsize=11)

    # (a) Eigenspectrum
    ax = axes[0]
    ax.semilogy(eigenvalues, '-o', ms=2, lw=1)
    ax.set_xlabel('PC index k')
    ax.set_ylabel('Eigenvalue λ_k')
    ax.set_title('Data covariance spectrum')

    # (b) Per-PC error: theory vs MC
    ax = axes[1]
    ax.semilogy(err_sim, 'o', ms=3, alpha=0.6, label='MC simulation')
    ax.semilogy(err_theory, '-', lw=1.5, label='RMT theory (total)')
    ax.semilogy(t1, '--', lw=1, label='Term1: overshrinkage')
    ax.semilogy(t2, '--', lw=1, label='Term2: finite-n signal')
    ax.semilogy(t3, '--', lw=1, label='Term3: finite-n noise')
    ax.set_xlabel('PC index k')
    ax.set_ylabel('E[(uₖᵀΔβ)²]')
    ax.set_title('Per-PC error')
    ax.legend(fontsize=7)

    # (c) Theory vs MC scatter
    ax = axes[2]
    ax.loglog(err_theory, err_sim, 'o', ms=3, alpha=0.6)
    lims = [min(err_theory.min(), err_sim.min()),
            max(err_theory.max(), err_sim.max())]
    ax.plot(lims, lims, 'k--', lw=1, label='y=x')
    ax.set_xlabel('RMT theory')
    ax.set_ylabel('MC simulation')
    ax.set_title('Theory vs MC scatter')
    ax.legend()

    plt.tight_layout()
    figdir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures')
    tag = f'{args.spectrum}_d{d}_n{n}_lam{lam}_sigma{sigma}'
    figpath = os.path.join(figdir, f'ridge_error_{tag}.png')
    plt.savefig(figpath, dpi=150, bbox_inches='tight')
    print(f"Saved: {figpath}")
    plt.close()


if __name__ == '__main__':
    main()

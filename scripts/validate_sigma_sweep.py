"""
Sweep noise level σ and compare theory vs MC for ridge error.

Records total generalization error and accentuation error as a function of σ.
Uses d=256, various spectra.

Usage:
  python validate_sigma_sweep.py --spectrum powerlaw --d 256 --n 512 --lam 0.05
  python validate_sigma_sweep.py --spectrum isotropic --d 256 --n 512 --lam 0.05
  python validate_sigma_sweep.py --spectrum vanhateren --n 512 --patch_hw 16
"""
import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rmt_core import (
    get_spectrum, SpectrumKappa,
    ridge_error_per_pc_theory, ridge_error_total_theory,
    accentuation_error_theory,
)
from rmt_core.gpu_simulation_lib import run_monte_carlo_gpu, get_device

FIGDIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures')
TABDIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tables')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spectrum', choices=['isotropic', 'powerlaw', 'vanhateren'],
                        default='powerlaw')
    parser.add_argument('--d', type=int, default=256)
    parser.add_argument('--n', type=int, default=512)
    parser.add_argument('--lam', type=float, default=0.05)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--patch_hw', type=int, default=16,
                        help='Patch size for vanhateren (d=patch_hw²=256)')
    parser.add_argument('--n_patches', type=int, default=300)
    parser.add_argument('--n_images', type=int, default=50)
    parser.add_argument('--sigma_min', type=float, default=0.05)
    parser.add_argument('--sigma_max', type=float, default=3.0)
    parser.add_argument('--n_sigma', type=int, default=15)
    parser.add_argument('--n_trials', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no_gpu', action='store_true')
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    device = get_device(use_gpu=not args.no_gpu)
    print(f"Device: {device}")

    # ── Load spectrum ──
    print(f"Loading spectrum: {args.spectrum} ...")
    if args.spectrum == 'vanhateren':
        from rmt_core.spectrum_lib import vanhateren_spectrum
        d_full = args.patch_hw ** 2
        eigenvalues, eigenvectors, _ = vanhateren_spectrum(
            patch_hw=args.patch_hw,
            n_patches_per_image=args.n_patches,
            n_images=args.n_images,
            use_randomized_svd=False,  # d=256 is small enough for full eigh
            rng=rng)
        diagonal = False
        residual_var = 0.0
    else:
        from rmt_core.spectrum_lib import isotropic_spectrum, powerlaw_spectrum
        d_full = args.d
        if args.spectrum == 'isotropic':
            eigenvalues, _ = isotropic_spectrum(d_full)
        else:
            eigenvalues, _ = powerlaw_spectrum(d_full, alpha=args.alpha)
        eigenvectors = np.eye(d_full)  # canonical basis
        diagonal = False  # use full eigenvectors since d=256 is small
        residual_var = 0.0

    d = len(eigenvalues)
    n, lam = args.n, args.lam
    gamma = d / n
    print(f"  d={d}, n={n}, γ={gamma:.3f}, λ={lam}")
    print(f"  Top-5 eigenvalues: {eigenvalues[:5]}")

    # κ(λ) — fixed across sigma sweep
    kappa_fn = SpectrumKappa(eigenvalues, gamma)
    kappa = kappa_fn(lam)
    print(f"  κ({lam}) = {kappa:.4f}")

    # True coefficient
    beta_star = rng.standard_normal(d)
    beta_proj = eigenvectors.T @ beta_star

    # ── Sigma sweep ──
    sigma_values = np.geomspace(args.sigma_min, args.sigma_max, args.n_sigma)

    theory_gen_errors = []
    mc_gen_errors = []
    theory_acc_errors = []
    mc_acc_errors = []
    theory_R_vals = []
    mc_R_vals = []

    # Theory per-PC error (noise-free part is sigma-independent)
    err_bias, t1, t2, _ = ridge_error_per_pc_theory(
        eigenvalues, beta_proj, kappa, sigma_noise=0.0, n=n)

    for sigma in sigma_values:
        print(f"\nσ={sigma:.3f} ...")

        # Theory
        err_theory, *_ = ridge_error_per_pc_theory(eigenvalues, beta_proj, kappa, sigma, n)
        gen_err_th = float(np.sum(eigenvalues * err_theory))
        acc_err_th, R_th, _ = accentuation_error_theory(eigenvalues, beta_proj, kappa, sigma, n)

        # MC
        mc = run_monte_carlo_gpu(
            n, eigenvalues, eigenvectors, beta_star, sigma, lam,
            n_trials=args.n_trials, residual_var=residual_var, diagonal=diagonal,
            device=device, rng=rng, verbose=False)

        print(f"  Gen error   theory={gen_err_th:.4f}  MC={mc['total_gen_error']:.4f}")
        print(f"  Acc error   theory={acc_err_th:.4f}  MC={mc['acc_error']:.4f}")
        print(f"  Alignment   theory={R_th:.4f}        MC={mc['acc_alignment']:.4f}")

        theory_gen_errors.append(gen_err_th)
        mc_gen_errors.append(mc['total_gen_error'])
        theory_acc_errors.append(acc_err_th)
        mc_acc_errors.append(mc['acc_error'])
        theory_R_vals.append(R_th)
        mc_R_vals.append(mc['acc_alignment'])

    # ── Save table ──
    os.makedirs(TABDIR, exist_ok=True)
    tag = f'{args.spectrum}_d{d}_n{n}_lam{lam}'
    np.savez(os.path.join(TABDIR, f'sigma_sweep_{tag}.npz'),
             sigma=sigma_values,
             theory_gen=np.array(theory_gen_errors),
             mc_gen=np.array(mc_gen_errors),
             theory_acc=np.array(theory_acc_errors),
             mc_acc=np.array(mc_acc_errors),
             theory_R=np.array(theory_R_vals),
             mc_R=np.array(mc_R_vals))

    # ── Figure ──
    os.makedirs(FIGDIR, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(
        f'Noise sweep  spectrum={args.spectrum}, d={d}, n={n}, λ={lam}',
        fontsize=11)

    ax = axes[0]
    ax.plot(sigma_values, theory_gen_errors, '-o', lw=1.5, label='Theory')
    ax.plot(sigma_values, mc_gen_errors, '--s', lw=1.5, label='MC')
    ax.set_xlabel('σ (noise std)'); ax.set_ylabel('Total gen error  β*ᵀΣ(β̂-β*)²')
    ax.set_title('In-distribution generalization error')
    ax.legend()

    ax = axes[1]
    ax.plot(sigma_values, theory_acc_errors, '-o', lw=1.5, label='Theory')
    ax.plot(sigma_values, mc_acc_errors, '--s', lw=1.5, label='MC')
    ax.set_xlabel('σ (noise std)'); ax.set_ylabel('Accentuation error')
    ax.set_title('Accentuation error  (β*ᵀΣβ*)·(1-R)²')
    ax.legend()

    ax = axes[2]
    ax.plot(sigma_values, theory_R_vals, '-o', lw=1.5, label='R_det theory')
    ax.plot(sigma_values, mc_R_vals, '--s', lw=1.5, label='R MC')
    ax.axhline(1.0, color='gray', lw=0.8, linestyle=':')
    ax.set_xlabel('σ (noise std)'); ax.set_ylabel('Alignment R = β̂ᵀβ*/β̂ᵀβ̂')
    ax.set_title('Accentuation alignment vs σ')
    ax.legend()

    plt.tight_layout()
    figpath = os.path.join(FIGDIR, f'sigma_sweep_{tag}.png')
    plt.savefig(figpath, dpi=150, bbox_inches='tight')
    print(f"\nSaved figure: {figpath}")
    plt.close()


if __name__ == '__main__':
    main()

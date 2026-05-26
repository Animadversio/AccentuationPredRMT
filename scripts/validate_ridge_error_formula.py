"""
Validate the three-term RMT formula for per-PC ridge regression error.

E[(uₖᵀ(β̂-β*))²] ≍  Term1 (overshrinkage) + Term2 (finite-sample signal) + Term3 (finite-sample noise)

Sweeps over:
- aspect ratio γ = d/n
- ridge penalty λ
- PC index k

Saves figures to figures/ and results to tables/.
"""
import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rmt_core import (
    SpectrumKappa,
    ridge_error_per_pc_theory,
    run_monte_carlo,
)


def power_law_spectrum(d, alpha=1.0):
    """Power-law eigenvalues: λ_k = k^{-alpha}, k=1..d, normalized to sum=d."""
    k = np.arange(1, d + 1, dtype=float)
    lam = k ** (-alpha)
    lam = lam / lam.mean()  # normalize mean to 1
    return lam


def make_eigenvectors(d, rng):
    """Random orthogonal matrix via QR decomposition."""
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    return Q


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--d', type=int, default=100, help='Dimension')
    parser.add_argument('--n', type=int, default=200, help='Number of samples')
    parser.add_argument('--lam', type=float, default=0.1, help='Ridge penalty')
    parser.add_argument('--sigma', type=float, default=0.5, help='Noise std')
    parser.add_argument('--alpha', type=float, default=1.0, help='Spectrum power-law exponent')
    parser.add_argument('--n_trials', type=int, default=1000, help='MC trials')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    d, n, lam, sigma = args.d, args.n, args.lam, args.sigma
    gamma = d / n

    # Spectrum and geometry
    eigenvalues = power_law_spectrum(d, args.alpha)
    eigenvectors = make_eigenvectors(d, rng)

    # True coefficient: signal spread across top PCs
    beta_star = rng.standard_normal(d)
    beta_proj = eigenvectors.T @ beta_star  # projections onto PCs

    # κ(λ) via fixed-point solver
    kappa_solver = SpectrumKappa(eigenvalues, gamma)
    kappa = kappa_solver(lam)
    print(f"κ({lam:.3f}) = {kappa:.4f}  (γ={gamma:.2f}, d={d}, n={n})")

    # Theory prediction
    err_theory, t1, t2, t3 = ridge_error_per_pc_theory(
        eigenvalues, beta_proj, kappa, sigma, n)

    # Monte Carlo
    print(f"Running {args.n_trials} MC trials...")
    mc = run_monte_carlo(n, eigenvalues, eigenvectors, beta_star, sigma, lam,
                         n_trials=args.n_trials, rng=rng)
    err_sim = mc['error_per_pc']

    print(f"Total gen error  — theory: {np.sum(eigenvalues*err_theory):.4f}  "
          f"MC: {mc['total_gen_error']:.4f}")
    print(f"Acc alignment    — MC: {mc['acc_alignment']:.4f}")
    print(f"Acc error        — MC: {mc['acc_error']:.4f}")

    # ── Figure: theory vs simulation per PC ──
    os.makedirs('figures', exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: per-PC error comparison
    ax = axes[0]
    ax.semilogy(err_sim, 'o', ms=3, alpha=0.6, label='MC simulation')
    ax.semilogy(err_theory, '-', lw=1.5, label='RMT theory (total)')
    ax.semilogy(t1, '--', lw=1, label='Term1: overshrinkage')
    ax.semilogy(t2, '--', lw=1, label='Term2: finite-n signal')
    ax.semilogy(t3, '--', lw=1, label='Term3: finite-n noise')
    ax.set_xlabel('PC index k')
    ax.set_ylabel('E[(uₖᵀ(β̂-β*))²]')
    ax.set_title(f'Per-PC error  d={d}, n={n}, λ={lam}, σ={sigma}')
    ax.legend(fontsize=7)

    # Right: scatter theory vs MC
    ax = axes[1]
    ax.loglog(err_theory, err_sim, 'o', ms=3, alpha=0.6)
    lims = [min(err_theory.min(), err_sim.min()),
            max(err_theory.max(), err_sim.max())]
    ax.plot(lims, lims, 'k--', lw=1, label='y=x')
    ax.set_xlabel('RMT theory')
    ax.set_ylabel('MC simulation')
    ax.set_title('Theory vs MC (per-PC error)')
    ax.legend()

    plt.tight_layout()
    figpath = f'figures/ridge_error_d{d}_n{n}_lam{lam}_sigma{sigma}.png'
    plt.savefig(figpath, dpi=150, bbox_inches='tight')
    print(f"Saved: {figpath}")
    plt.close()


if __name__ == '__main__':
    main()

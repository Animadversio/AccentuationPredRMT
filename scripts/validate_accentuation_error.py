"""
Validate the accentuation error theory:
    E_acc ≍ (β*ᵀΣβ*) · (1 - R_det)²

Sweeps over n (samples) for fixed d and λ, comparing theory vs MC.
"""
import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rmt_core import (
    SpectrumKappa,
    accentuation_error_theory,
    run_monte_carlo,
)


def power_law_spectrum(d, alpha=1.0):
    k = np.arange(1, d + 1, dtype=float)
    lam = k ** (-alpha)
    return lam / lam.mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--d', type=int, default=50)
    parser.add_argument('--lam', type=float, default=0.05)
    parser.add_argument('--sigma', type=float, default=0.5)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--n_trials', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    d = args.d
    lam, sigma = args.lam, args.sigma

    eigenvalues = power_law_spectrum(d, args.alpha)
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    beta_star = rng.standard_normal(d)
    beta_proj = Q.T @ beta_star

    n_values = np.array([60, 80, 100, 150, 200, 300, 500, 1000])
    theory_errors, mc_errors = [], []
    theory_R, mc_R = [], []

    for n in n_values:
        gamma = d / n
        kappa_solver = SpectrumKappa(eigenvalues, gamma)
        kappa = kappa_solver(lam)

        err_th, R_th, _ = accentuation_error_theory(
            eigenvalues, beta_proj, kappa, sigma, n)
        mc = run_monte_carlo(n, eigenvalues, Q, beta_star, sigma, lam,
                             n_trials=args.n_trials, rng=rng)

        theory_errors.append(err_th)
        mc_errors.append(mc['acc_error'])
        theory_R.append(R_th)
        mc_R.append(mc['acc_alignment'])
        print(f"n={n:4d}  κ={kappa:.3f}  "
              f"R_th={R_th:.3f}  R_mc={mc['acc_alignment']:.3f}  "
              f"Err_th={err_th:.4f}  Err_mc={mc['acc_error']:.4f}")

    figdir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 'figures', 'accentuation')
    os.makedirs(figdir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.plot(n_values, theory_R, '-o', label='R_det theory')
    ax.plot(n_values, mc_R, '--s', label='R MC')
    ax.set_xlabel('n (samples)')
    ax.set_ylabel('Alignment R')
    ax.set_title(f'Accentuation alignment  d={d}, λ={lam}')
    ax.legend()

    ax = axes[1]
    ax.plot(n_values, theory_errors, '-o', label='Theory')
    ax.plot(n_values, mc_errors, '--s', label='MC simulation')
    ax.set_xlabel('n (samples)')
    ax.set_ylabel('Accentuation error')
    ax.set_title(f'Accentuation error vs n  d={d}, λ={lam}, σ={sigma}')
    ax.legend()

    plt.tight_layout()
    figpath = os.path.join(
        figdir, f'accentuation_error_d{d}_lam{lam}_sigma{sigma}.png')
    plt.savefig(figpath, dpi=150, bbox_inches='tight')
    print(f"Saved: {figpath}")
    plt.close()


if __name__ == '__main__':
    main()

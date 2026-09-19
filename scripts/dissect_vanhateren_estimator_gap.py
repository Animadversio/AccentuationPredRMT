"""Read-only audit of cached, paired natural-image MC (no new ridge fits).

Separate slope bias/variance and identify noise coefficients from the seven
shared-alpha, shared-design, shared-standard-noise observations per trial.
"""
import argparse
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/accentuationpredrmt-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path(
        'notebooks/outputs/pixel_ridge/vanhateren_selection_estimation/gap_audit'))
    args = parser.parse_args()
    table = pd.read_csv(Path(__file__).resolve().parents[1] /
        'notebooks/outputs/pixel_ridge/vanhateren_selection_estimation/comparison.csv')
    raw = pd.read_csv(args.raw)
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for row in table.itertuples():
        q = raw[(raw.policy == row.policy) & np.isclose(
            raw.sigma, row.sigma, rtol=1e-10, atol=0)]
        assert len(q) == 100 and q.trial.nunique() == 100
        bias2 = (1-q.slope_acc.mean())**2
        variance = q.slope_acc.var(ddof=0)
        assert np.isclose(bias2+variance, q.E_acc.mean(), rtol=1e-8, atol=1e-16)
        rows.append(dict(policy=row.policy, sigma=row.sigma, ratio=row.ratio,
            lam=row.lam, acc_mc=q.E_acc.mean(), acc_bias2=bias2,
            acc_variance=variance, gen_mc=row.E_gen_mc, gen_de=row.E_gen,
            acc_leading=row.E_acc, acc_gaussian=row.acc_distribution))
    decomposition = pd.DataFrame(rows)
    decomposition.to_csv(args.output/'risk_decomposition.csv', index=False)

    low = raw[(raw.policy == 'DE_gen_CV') & (raw.lam < 1.01e-8)]
    coefficients = []
    for trial, q in low.groupby('trial'):
        q = q.sort_values('sigma')
        assert q.alpha.nunique() == 1
        s, R = q.sigma.to_numpy(), q.slope_acc.to_numpy()
        g = np.linalg.lstsq(np.array([s*0+1, s, s*s]).T, q.E_gen, rcond=None)[0]
        # R(s)=(a+b*s)/(1+c*s+d*s**2), normalize by noiseless D.
        c = np.linalg.lstsq(np.array([s*0+1, s, -R*s, -R*s*s]).T, R, rcond=None)[0]
        err_r = np.max(abs((c[0]+c[1]*s)/(1+c[2]*s+c[3]*s*s)-R))
        err_g = np.max(abs(g[0]+g[1]*s+g[2]*s*s-q.E_gen))
        assert err_r < 1e-10 and err_g < 1e-10
        coefficients.append(dict(trial=trial, gen_baseline=g[0], gen_linear=g[1],
            gen_noise=g[2], N0_over_D0=c[0], N1_over_D0=c[1],
            D1_over_D0=c[2], D2_over_D0=c[3], slope_fit_error=err_r,
            gen_fit_error=err_g))
    coefficients = pd.DataFrame(coefficients)
    coefficients.to_csv(args.output/'paired_noise_coefficients.csv', index=False)
    de = table[(table.policy == 'DE_gen_CV') & (table.lam < 1.01e-8)]
    s, R = de.sigma.to_numpy(), de.slope_acc.to_numpy()
    g = np.linalg.lstsq(np.array([s*0+1, s*s]).T, de.E_gen, rcond=None)[0]
    c = np.linalg.lstsq(np.array([s*0+1, -R*s*s]).T, R, rcond=None)[0]
    comparison = pd.DataFrame(dict(quantity=['gen_baseline', 'gen_noise', 'D2_over_D0'],
        DE=[g[0], g[1], c[1]], MC=[coefficients.gen_baseline.mean(),
        coefficients.gen_noise.mean(), coefficients.D2_over_D0.mean()]))
    comparison['MC_over_DE'] = comparison.MC/comparison.DE
    comparison.to_csv(args.output/'noise_amplification.csv', index=False)
    print(comparison.to_string(index=False))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    for j, policy in enumerate(['DE_gen_CV', 'Gaussian_acc_oracle']):
        q = decomposition[decomposition.policy == policy].sort_values('sigma')
        ax = axes[0, j]
        ax.plot(q.sigma, q.gen_mc/q.gen_de, 'o-', label='MC / DE generalization risk')
        ax.axhline(1, color='gray', ls='--')
        ax.set_title(policy)
        ax.set_ylabel('Generalization risk ratio')
        ax.legend(fontsize=9)
        ax = axes[1, j]
        ax.plot(q.sigma, q.acc_mc, 'ko-', ms=3, label='MC mean total')
        ax.plot(q.sigma, q.acc_bias2, color='C0', label='MC squared mean-slope bias')
        ax.plot(q.sigma, q.acc_variance, color='C2', label='MC slope variance')
        ax.plot(q.sigma, q.acc_leading, '--', color='C1', label='Leading DE')
        ax.plot(q.sigma, q.acc_gaussian, ':', color='C3', label='Diagonal Gaussian surrogate')
        ax.set_yscale('log')
        ax.set_ylabel('Accentuation error / signal variance')
        ax.legend(fontsize=8)
        for ax in axes[:, j]:
            ax.set_xscale('symlog', linthresh=.03)
            ax.set_xlim(0, q.sigma.max()*1.05)
            ax.set_xticks([0, .1, 1, 10, 100])
            ax.set_xticklabels(['0', '0.1', '1', '10', '100'])
            ax.set_xlabel(r'Response noise $\sigma$')
            ax.grid(alpha=.2)
            secondary = ax.twiny()
            secondary.set_xscale('symlog', linthresh=.03)
            secondary.set_xlim(ax.get_xlim())
            ticks = [0, 1e-6, 1e-4, 1e-2, 1, 10]
            secondary.set_xticks(np.sqrt(np.asarray(ticks)*float(table.S.iloc[0])))
            secondary.set_xticklabels([f'{x:g}' for x in ticks])
            secondary.set_xlabel(r'Noise / signal variance $\sigma^2/S$', fontsize=9)
    fig.suptitle('Van Hateren: two distinct sources of approximation error')
    fig.savefig(args.output/'gap_dissection.png', dpi=160, bbox_inches='tight')
    fig.savefig(args.output/'gap_dissection.pdf', bbox_inches='tight')
    print('Saved:', args.output)


if __name__ == '__main__':
    main()

"""Dissect the FFHQ RidgeCV accentuation mismatch in spectral coordinates.

The script reuses the cached FFHQ population spectrum and trial-level dense-grid
RidgeCV runs.  It compares two deterministic-equivalent paths:

1. the experiment-matched path (LOOCV alpha grid, followed by an n-sample
   refit), evaluated with the direct sigma/alpha formula for z;
2. a same-n, continuous population-oracle path, on which the stationarity
   identity

       z = kappa * df_12 / B_11 * (B_23 / df_23 - B_12 / df_12)

   is exact at every interior optimum.

Plot-ready dynamic and static spectral tables are cached independently of the
rendered figure.
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import time
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/accentuationpredrmt-matplotlib')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp/accentuationpredrmt-xdg-cache')

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq


REPO_ROOT = Path(__file__).resolve().parents[1]
SPECTRUM_PATH = REPO_ROOT / 'tables' / 'ffhq_disk_teacher_spectrum.npz'
SUMMARY_PATH = REPO_ROOT / 'tables' / 'ffhq_disk_teacher_de_summary.csv'
DYNAMIC_TABLE_PATH = REPO_ROOT / 'tables' / 'ffhq_cv_spectral_mismatch.csv'
PROFILE_TABLE_PATH = REPO_ROOT / 'tables' / 'ffhq_spectral_teacher_profile.csv'
FIGURE_PATH = REPO_ROOT / 'figures' / 'ffhq_cv_spectral_mismatch.png'
LOG_PATH = REPO_ROOT / 'logs' / 'ffhq_cv_spectral_mismatch.log'


def configure_logging(path: Path) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger('ffhq_cv_spectral_mismatch')
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(path, mode='w')
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def read_summary(path: Path) -> list[dict[str, str]]:
    with path.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    return sorted(rows, key=lambda row: float(row['sigma']))


def spectral_terms(eigenvalues: np.ndarray, beta_sq: np.ndarray,
                   kappa: float, n: int) -> dict[str, float]:
    denominator = eigenvalues + kappa
    t = eigenvalues / denominator
    q = eigenvalues / denominator ** 2
    df_11 = float(np.sum(t))
    df_22 = float(np.sum(t ** 2))
    df_12 = float(np.sum(q))
    b_11 = float(np.sum(t * beta_sq))
    b_12 = float(np.sum(q * beta_sq))
    df_23 = float(np.sum(q * t))
    b_23 = float(np.sum(q * t * beta_sq))
    h = float(n - df_22)
    return {
        'df_11': df_11,
        'df_22': df_22,
        'df_12': df_12,
        'B_11': b_11,
        'B_12': b_12,
        'df_23': df_23,
        'B_23': b_23,
        'h': h,
        'limbo_count': kappa * df_12,
        'b2_acc_average': b_12 / df_12,
        'b2_gen_average': b_23 / df_23,
    }


def lambda_from_kappa(terms: dict[str, float], kappa: float,
                      n: int) -> tuple[float, float]:
    alpha = kappa * (n - terms['df_11'])
    return alpha / n, alpha


def generalization_risk(terms: dict[str, float], kappa: float,
                        sigma: float, n: int) -> float:
    return (
        n * kappa ** 2 * terms['B_12']
        + sigma ** 2 * terms['df_22']
    ) / terms['h']


def stationarity_residual(terms: dict[str, float], kappa: float,
                          sigma: float) -> float:
    return (
        sigma ** 2 + kappa ** 2 * terms['B_12']
        - kappa * terms['h'] * terms['b2_gen_average']
    )


def z_metrics(terms: dict[str, float], kappa: float, sigma: float,
              alpha: float) -> dict[str, float]:
    direct = (
        sigma ** 2 * terms['df_12'] - alpha * terms['B_12']
    ) / (terms['h'] * terms['B_11'])
    simplified = (
        kappa * terms['df_12'] / terms['B_11']
        * (terms['b2_gen_average'] - terms['b2_acc_average'])
    )
    residual = stationarity_residual(terms, kappa, sigma)
    reconstructed = (
        simplified
        + terms['df_12'] / (terms['h'] * terms['B_11']) * residual
    )
    return {
        'z_direct': direct,
        'z_simplified': simplified,
        'stationarity_residual': residual,
        'z_reconstructed': reconstructed,
        'slope_acc_leading': 1.0 / (1.0 + direct),
        'r2_acc_leading': 1.0 - direct ** 2,
    }


def ridgeless_kappa(eigenvalues: np.ndarray, n: int) -> float:
    if len(eigenvalues) <= n:
        return 0.0

    def equation(kappa: float) -> float:
        return float(np.sum(eigenvalues / (eigenvalues + kappa)) - n)

    lower = np.finfo(float).tiny
    upper = float(np.max(eigenvalues))
    while equation(upper) > 0:
        upper *= 10.0
    return float(brentq(equation, lower, upper, xtol=1e-14, rtol=1e-13))


def oracle_kappa(
        eigenvalues: np.ndarray, beta_sq: np.ndarray, sigma: float, n: int,
        kappa_grid: np.ndarray, grid_terms: list[dict[str, float]],
        kappa_min: float) -> tuple[float, bool, float]:
    phi = np.asarray([
        kappa * terms['h'] * terms['B_23']
        - terms['df_23'] * (kappa ** 2 * terms['B_12'] + sigma ** 2)
        for kappa, terms in zip(kappa_grid, grid_terms)
    ])
    candidate_kappas = [kappa_min, float(kappa_grid[-1])]
    for index in np.flatnonzero(phi[:-1] * phi[1:] < 0):
        left = float(kappa_grid[index])
        right = float(kappa_grid[index + 1])

        def root_equation(kappa: float) -> float:
            terms = spectral_terms(eigenvalues, beta_sq, kappa, n)
            return (
                kappa * terms['h'] * terms['B_23']
                - terms['df_23']
                * (kappa ** 2 * terms['B_12'] + sigma ** 2)
            )

        candidate_kappas.append(float(brentq(
            root_equation, left, right, xtol=1e-12, rtol=1e-11)))

    candidate_risks = []
    for kappa in candidate_kappas:
        terms = spectral_terms(eigenvalues, beta_sq, kappa, n)
        candidate_risks.append(generalization_risk(terms, kappa, sigma, n))
    best_index = int(np.argmin(candidate_risks))
    best_kappa = candidate_kappas[best_index]
    is_interior = best_index not in (0, 1)
    return best_kappa, is_interior, float(candidate_risks[best_index])


def solve_kappa_for_alpha(eigenvalues: np.ndarray, alpha: float, n: int,
                          kappa_min: float) -> float:
    def equation(kappa: float) -> float:
        df_11 = float(np.sum(eigenvalues / (eigenvalues + kappa)))
        return kappa * (n - df_11) - alpha

    lower = kappa_min * (1.0 + 1e-12)
    upper = max(float(np.max(eigenvalues)), lower * 2.0)
    while equation(upper) < 0:
        upper *= 10.0
    return float(brentq(equation, lower, upper, xtol=1e-12, rtol=1e-11))


def trial_metrics(case_path: Path) -> dict[str, float]:
    with np.load(case_path) as case:
        slopes = np.asarray(case['trial_slope_acc'], dtype=float)
        r2 = np.asarray(case['trial_r2_acc'], dtype=float)
    z = 1.0 / slopes - 1.0
    return {
        'mc_z_mean': float(np.mean(z)),
        'mc_z_median': float(np.median(z)),
        'mc_z_q25': float(np.quantile(z, 0.25)),
        'mc_z_q75': float(np.quantile(z, 0.75)),
        'mc_r2_mean': float(np.mean(r2)),
        'mc_r2_median': float(np.median(r2)),
        'mc_r2_q25': float(np.quantile(r2, 0.25)),
        'mc_r2_q75': float(np.quantile(r2, 0.75)),
    }


def write_rows(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def spectral_profile(eigenvalues: np.ndarray, beta_sq: np.ndarray,
                     n_bins: int = 64) -> list[dict[str, float]]:
    edges = np.logspace(
        np.log10(float(np.min(eigenvalues))),
        np.log10(float(np.max(eigenvalues))), n_bins + 1)
    bin_index = np.clip(np.digitize(eigenvalues, edges) - 1, 0, n_bins - 1)
    rows = []
    for index in range(n_bins):
        selected = bin_index == index
        if not np.any(selected):
            continue
        power = beta_sq[selected]
        rows.append({
            's_left': float(edges[index]),
            's_right': float(edges[index + 1]),
            's_center': float(np.sqrt(edges[index] * edges[index + 1])),
            'mode_count': int(np.sum(selected)),
            'mean_teacher_power': float(np.mean(power)),
            'total_teacher_power': float(np.sum(power)),
        })
    return rows


def add_noise_ratio_axis(ax: plt.Axes, signal_power: float,
                         max_sigma: float) -> None:
    ratio_ticks = np.asarray([0.0, 1e-6, 1e-4, 1e-2, 0.1, 1.0, 10.0])
    sigma_ticks = np.sqrt(ratio_ticks * signal_power)
    visible = sigma_ticks <= max_sigma * 1.001
    ratio_ticks = ratio_ticks[visible]
    sigma_ticks = sigma_ticks[visible]
    labels = [
        '0' if value == 0 else rf'$10^{{{int(np.log10(value))}}}$'
        if np.isclose(np.log10(value), round(np.log10(value))) else f'{value:g}'
        for value in ratio_ticks
    ]
    top = ax.twiny()
    top.set_xscale('symlog', linthresh=0.1)
    top.set_xlim(ax.get_xlim())
    top.set_xticks(sigma_ticks)
    top.set_xticklabels(labels)
    top.set_xlabel(r'noise variance / signal variance  $\sigma^2/S$')
    top.tick_params(axis='x', labelsize=8, pad=1)


def plot_results(dynamic_rows: list[dict[str, float]],
                 profile_rows: list[dict[str, float]],
                 signal_power: float) -> None:
    sigma = np.asarray([row['sigma'] for row in dynamic_rows])
    max_sigma = float(np.max(sigma))
    profile_s = np.asarray([row['s_center'] for row in profile_rows])
    profile_teacher = np.asarray([
        row['mean_teacher_power'] for row in profile_rows])
    profile_count = np.asarray([row['mode_count'] for row in profile_rows])

    fig, axes = plt.subplots(4, 2, figsize=(14.2, 17.2))
    axes = axes.ravel()

    ax = axes[0]
    ax.plot(profile_s, profile_teacher, color='C4', lw=2.1, marker='o', ms=3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'population eigenvalue $s_k$')
    ax.set_ylabel(r'bin mean teacher power $\langle b_k^2\rangle$')
    ax.set_title('Disk-teacher power along the FFHQ spectrum')

    ax_density = axes[1]
    ax_density.step(profile_s, profile_count, where='mid', color='0.25', lw=2)
    ax_density.fill_between(
        profile_s, profile_count, step='mid', color='0.5', alpha=0.18)
    ax_density.set_xscale('log')
    ax_density.set_yscale('log')
    ax_density.set_xlabel(r'population eigenvalue $s_k$')
    ax_density.set_ylabel('modes per log-spectral bin')
    ax_density.set_title('FFHQ spectral density')

    selected_ratios = np.asarray([0.01, 0.1, 1.0, 10.0])
    colors = ['C0', 'C1', 'C2', 'C3']
    nsr = np.asarray([row['noise_signal_ratio'] for row in dynamic_rows])
    for ratio, color in zip(selected_ratios, colors):
        index = int(np.argmin(np.abs(np.log10(np.maximum(nsr, 1e-20)) - np.log10(ratio))))
        kappa = dynamic_rows[index]['oracle_kappa']
        label = rf'$\sigma^2/S\approx{ratio:g}$'
        for static_ax in (axes[0], axes[1]):
            static_ax.axvline(kappa, color=color, lw=1.4, alpha=0.8,
                              label=label if static_ax is axes[1] else None)
    axes[1].legend(fontsize=8, loc='best')

    oracle_kappa_values = np.asarray([row['oracle_kappa'] for row in dynamic_rows])
    grid_kappa = np.asarray([row['grid_kappa_final'] for row in dynamic_rows])
    mc_kappa = np.asarray([row['mc_kappa_median'] for row in dynamic_rows])
    mc_kappa_q25 = np.asarray([row['mc_kappa_q25'] for row in dynamic_rows])
    mc_kappa_q75 = np.asarray([row['mc_kappa_q75'] for row in dynamic_rows])
    ax = axes[2]
    ax.plot(sigma, oracle_kappa_values, color='C0', lw=2.4,
            label='same-n continuous DE oracle')
    ax.plot(sigma, grid_kappa, color='C1', lw=2, ls='--',
            label='LOOCV-grid DE, final refit')
    ax.plot(sigma, mc_kappa, color='k', marker='o', ms=3.5, lw=1.4,
            label='MC RidgeCV median')
    ax.fill_between(sigma, mc_kappa_q25, mc_kappa_q75,
                    color='k', alpha=0.12, linewidth=0)
    ax.set_yscale('log')
    ax.set_ylabel(r'effective regularization $\kappa$')
    ax.set_title(r'Moving ridge cutoff $\kappa(\sigma)$')
    ax.legend(fontsize=8)

    ax = axes[3]
    bgen = np.asarray([row['oracle_b2_gen_average'] for row in dynamic_rows])
    bacc = np.asarray([row['oracle_b2_acc_average'] for row in dynamic_rows])
    ax.plot(sigma, bgen, color='C0', lw=2.4,
            label=r'$B_{2,3}/\mathrm{df}_{2,3}$: gen-weighted')
    ax.plot(sigma, bacc, color='C1', lw=2.4,
            label=r'$B_{1,2}/\mathrm{df}_{1,2}$: acc-weighted')
    ax.fill_between(sigma, bacc, bgen, color='0.4', alpha=0.12)
    ax.set_yscale('log')
    ax.set_ylabel('moving-window mean teacher power')
    ax.set_title('Two moving spectral averages')
    ax.legend(fontsize=8)

    ax = axes[4]
    recoverable = np.asarray([row['oracle_B_11'] for row in dynamic_rows])
    recoverable_grid = np.asarray([row['grid_B_11'] for row in dynamic_rows])
    ax.plot(sigma, recoverable, color='C2', lw=2.4,
            label='same-n continuous DE oracle')
    ax.plot(sigma, recoverable_grid, color='C2', lw=1.8, ls='--',
            label='LOOCV-grid DE, final refit')
    ax.set_ylabel(r'recoverable teacher power $B_{1,1}$')
    ax.set_title(r'Retained alignment $B_{1,1}(\kappa)$')
    ax.legend(fontsize=8)

    ax = axes[5]
    limbo = np.asarray([row['oracle_limbo_count'] for row in dynamic_rows])
    limbo_grid = np.asarray([row['grid_limbo_count'] for row in dynamic_rows])
    ax.plot(sigma, limbo, color='C3', lw=2.4,
            label=r'oracle $\kappa\,\mathrm{df}_{1,2}$')
    ax.plot(sigma, limbo_grid, color='C3', lw=1.8, ls='--',
            label='LOOCV-grid DE, final refit')
    ax.set_ylabel(r'limbo-mode soft count $\kappa\,\mathrm{df}_{1,2}$')
    ax.set_title('Number of modes near the ridge transition')
    ax.legend(fontsize=8)

    ax = axes[6]
    oracle_interior = np.asarray([
        row['oracle_interior'] > 0.5 for row in dynamic_rows])
    oracle_z_simplified = np.asarray([
        row['oracle_z_simplified'] for row in dynamic_rows])
    oracle_z_direct = np.asarray([
        row['oracle_z_direct'] for row in dynamic_rows])
    # The covariance simplification uses the interior stationarity condition.
    # At the ridgeless low-noise boundary, retain the direct sigma/alpha form.
    oracle_z = np.where(
        oracle_interior, oracle_z_simplified, oracle_z_direct)
    grid_z = np.asarray([row['grid_z_direct'] for row in dynamic_rows])
    mc_z = np.asarray([row['mc_z_median'] for row in dynamic_rows])
    mc_z_q25 = np.asarray([row['mc_z_q25'] for row in dynamic_rows])
    mc_z_q75 = np.asarray([row['mc_z_q75'] for row in dynamic_rows])
    ax.plot(sigma, oracle_z, color='C0', lw=2.4,
            label='same-n oracle DE (simplified at interior)')
    ax.plot(sigma, grid_z, color='C1', lw=2.1, ls='--',
            label=r'direct grid/final DE $(\sigma^2,\alpha)$')
    ax.plot(sigma, mc_z, color='k', marker='o', ms=3.5, lw=1.3,
            label='natural-image RidgeCV median')
    ax.fill_between(sigma, mc_z_q25, mc_z_q75,
                    color='k', alpha=0.12, linewidth=0)
    ax.axhline(0, color='0.4', lw=1)
    ax.axhline(1, color='0.4', lw=1, ls=':')
    ax.axhline(-1, color='0.4', lw=1, ls=':')
    ax.set_ylabel(r'$z_{\rm acc}=\mu_D/\mu_N-1$')
    ax.set_title('Accentuation calibration gap')
    ax.legend(fontsize=8)

    ax = axes[7]
    oracle_r2 = 1.0 - oracle_z ** 2
    grid_r2 = 1.0 - grid_z ** 2
    mc_r2_median = np.asarray([row['mc_r2_median'] for row in dynamic_rows])
    mc_r2_q25 = np.asarray([row['mc_r2_q25'] for row in dynamic_rows])
    mc_r2_q75 = np.asarray([row['mc_r2_q75'] for row in dynamic_rows])
    mc_r2_mean = np.asarray([row['mc_r2_mean'] for row in dynamic_rows])
    ax.plot(sigma, oracle_r2, color='C0', lw=2.4,
            label='same-n oracle DE (simplified at interior)')
    ax.plot(sigma, grid_r2, color='C1', lw=2.1, ls='--',
            label='direct grid/final DE')
    ax.plot(sigma, mc_r2_median, color='k', marker='o', ms=3.5, lw=1.3,
            label='MC median / IQR')
    ax.fill_between(sigma, mc_r2_q25, mc_r2_q75,
                    color='k', alpha=0.12, linewidth=0)
    ax.plot(sigma, mc_r2_mean, color='0.35', marker='x', ms=4, lw=1,
            ls=':', label='MC mean')
    ax.axhline(0, color='0.4', lw=1)
    ax.set_ylabel(r'$R^2_{\rm acc}=1-z^2$')
    ax.set_title(r'Leading DE vs natural-image RidgeCV $R^2_{\rm acc}$')
    ax.legend(fontsize=8)

    for dynamic_ax in axes[2:]:
        dynamic_ax.set_xscale('symlog', linthresh=0.1)
        dynamic_ax.set_xlim(0, max_sigma)
        dynamic_ax.set_xlabel(r'response noise $\sigma$')
        dynamic_ax.grid(alpha=0.18)
    add_noise_ratio_axis(axes[2], signal_power, max_sigma)
    add_noise_ratio_axis(axes[3], signal_power, max_sigma)
    for static_ax in axes[:2]:
        static_ax.grid(alpha=0.18)

    fig.suptitle(
        'FFHQ disk teacher: spectral mechanism of RidgeCV accentuation mismatch',
        fontsize=16, y=0.998)
    fig.tight_layout(rect=(0, 0, 1, 0.987))
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=190, bbox_inches='tight')
    FIGURE_PATH.chmod(0o644)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--kappa-grid-size', type=int, default=800)
    parser.add_argument('--force', action='store_true',
                        help='Recompute tables even if cached outputs exist.')
    args = parser.parse_args()
    logger = configure_logging(LOG_PATH)
    start = time.perf_counter()

    with np.load(SPECTRUM_PATH) as spectrum:
        eigenvalues = np.asarray(spectrum['eigenvalues'], dtype=float)
        beta_proj = np.asarray(spectrum['beta_proj'], dtype=float)
    beta_sq = beta_proj ** 2
    summary_rows = read_summary(SUMMARY_PATH)
    n = int(float(summary_rows[0]['n']))
    signal_power = float(summary_rows[0]['theory_signal_power'])
    logger.info('Loaded d=%d FFHQ spectrum and %d cached sigma conditions',
                len(eigenvalues), len(summary_rows))

    profile_rows = spectral_profile(eigenvalues, beta_sq)
    write_rows(PROFILE_TABLE_PATH, profile_rows)

    kappa_min = ridgeless_kappa(eigenvalues, n)
    kappa_grid = np.logspace(
        np.log10(kappa_min * (1.0 + 1e-10)), 5.0, args.kappa_grid_size)
    grid_terms = [
        spectral_terms(eigenvalues, beta_sq, kappa, n)
        for kappa in kappa_grid
    ]

    benchmark_start = time.perf_counter()
    oracle_kappa(
        eigenvalues, beta_sq, float(summary_rows[-1]['sigma']), n,
        kappa_grid, grid_terms, kappa_min)
    benchmark_seconds = time.perf_counter() - benchmark_start
    logger.info(
        'Single-condition oracle benchmark %.3fs; projected %.2fs for %d conditions',
        benchmark_seconds, benchmark_seconds * len(summary_rows),
        len(summary_rows))

    dynamic_rows: list[dict[str, float]] = []
    for index, summary in enumerate(summary_rows, start=1):
        sigma = float(summary['sigma'])
        grid_alpha = float(summary['theory_alpha_cv'])
        grid_kappa_final = float(summary['theory_kappa_final'])
        grid_values = spectral_terms(
            eigenvalues, beta_sq, grid_kappa_final, n)
        grid_z = z_metrics(
            grid_values, grid_kappa_final, sigma, grid_alpha)
        if not np.isclose(
                grid_z['z_direct'], grid_z['z_reconstructed'],
                rtol=2e-8, atol=2e-10):
            raise AssertionError('Grid/final residual reconstruction failed')

        optimum_kappa, optimum_interior, optimum_risk = oracle_kappa(
            eigenvalues, beta_sq, sigma, n,
            kappa_grid, grid_terms, kappa_min)
        oracle_values = spectral_terms(
            eigenvalues, beta_sq, optimum_kappa, n)
        _, oracle_alpha = lambda_from_kappa(
            oracle_values, optimum_kappa, n)
        oracle_z = z_metrics(
            oracle_values, optimum_kappa, sigma, oracle_alpha)
        if optimum_interior and not np.isclose(
                oracle_z['z_direct'], oracle_z['z_simplified'],
                rtol=2e-7, atol=2e-9):
            raise AssertionError(
                f'Oracle formula mismatch at sigma={sigma}: '
                f'{oracle_z["z_direct"]} vs {oracle_z["z_simplified"]}')

        alpha_median = float(summary['mc_alpha_cv_median'])
        alpha_q25 = float(summary['mc_alpha_cv_q25'])
        alpha_q75 = float(summary['mc_alpha_cv_q75'])
        mc_kappa_median = solve_kappa_for_alpha(
            eigenvalues, alpha_median, n, kappa_min)
        mc_kappa_q25 = solve_kappa_for_alpha(
            eigenvalues, alpha_q25, n, kappa_min)
        mc_kappa_q75 = solve_kappa_for_alpha(
            eigenvalues, alpha_q75, n, kappa_min)
        mc_values = trial_metrics(Path(summary['case_path']))

        row: dict[str, float] = {
            'sigma': sigma,
            'noise_signal_ratio': sigma ** 2 / signal_power,
            'signal_power': signal_power,
            'grid_alpha_cv': grid_alpha,
            'grid_kappa_final': grid_kappa_final,
            'mc_kappa_median': mc_kappa_median,
            'mc_kappa_q25': mc_kappa_q25,
            'mc_kappa_q75': mc_kappa_q75,
            'oracle_kappa': optimum_kappa,
            'oracle_alpha': oracle_alpha,
            'oracle_interior': float(optimum_interior),
            'oracle_gen_error': optimum_risk,
        }
        for prefix, values, z_values in (
                ('grid', grid_values, grid_z),
                ('oracle', oracle_values, oracle_z)):
            row.update({f'{prefix}_{key}': value
                        for key, value in values.items()})
            row.update({f'{prefix}_{key}': value
                        for key, value in z_values.items()})
        row.update(mc_values)
        dynamic_rows.append(row)
        oracle_z_for_log = (
            oracle_z['z_simplified'] if optimum_interior
            else oracle_z['z_direct'])
        logger.info(
            '[%d/%d] sigma=%g | kappa oracle/grid/MC %.4g / %.4g / %.4g '
            '| z oracle/grid/MC %.4g / %.4g / %.4g',
            index, len(summary_rows), sigma,
            optimum_kappa, grid_kappa_final, mc_kappa_median,
            oracle_z_for_log, grid_z['z_direct'],
            mc_values['mc_z_median'])

    write_rows(DYNAMIC_TABLE_PATH, dynamic_rows)
    plot_results(dynamic_rows, profile_rows, signal_power)

    interior_differences = [
        abs(row['oracle_z_direct'] - row['oracle_z_simplified'])
        for row in dynamic_rows if row['oracle_interior'] > 0.5
    ]
    logger.info('Maximum interior oracle |z_direct-z_simplified|: %.3g',
                max(interior_differences) if interior_differences else np.nan)
    logger.info('Wrote %s', DYNAMIC_TABLE_PATH)
    logger.info('Wrote %s', PROFILE_TABLE_PATH)
    logger.info('Wrote %s', FIGURE_PATH)
    logger.info('Completed in %.2fs', time.perf_counter() - start)


if __name__ == '__main__':
    main()

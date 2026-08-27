"""Plot RidgeCV spectral selection and fixed-lambda z decompositions.

This script consumes only compact population spectra, cached summary CSVs,
and cached trial metrics.  It does not load the natural-image datasets or run
new ridge fits.

Outputs
-------
1. Van Hateren RidgeCV spectral mechanism, parallel to the FFHQ analysis.
2. FFHQ fixed-lambda decomposition

       z = sigma^2 Q / (h A) - alpha C / (h A),

   with Q=df_12, C=B_12, h=n-df_22, and A=B_11.
3. The corresponding Van Hateren fixed-lambda decomposition.

Every rendered figure has a separate plot-ready CSV cache.
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/accentuationpredrmt-matplotlib')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp/accentuationpredrmt-xdg-cache')

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_ffhq_cv_spectral_mismatch import (
    add_noise_ratio_axis,
    generalization_risk,
    lambda_from_kappa,
    oracle_kappa,
    ridgeless_kappa,
    solve_kappa_for_alpha,
    spectral_profile,
    spectral_terms,
    write_rows,
    z_metrics,
)
from scripts.compare_ffhq_fixed_vs_cv import fixed_case_path


LOG_PATH = REPO_ROOT / 'logs' / 'natural_image_z_breakdowns.log'

DATASETS = {
    'ffhq': {
        'label': 'FFHQ',
        'spectrum': REPO_ROOT / 'tables' / 'ffhq_disk_teacher_spectrum.npz',
        'summary': REPO_ROOT / 'tables' / 'ffhq_fixed_vs_cv_summary.csv',
        'n': 1000,
        'fixed_table': (
            REPO_ROOT / 'tables' / 'ffhq_fixed_lambda_z_breakdown.csv'),
        'fixed_figure': (
            REPO_ROOT / 'figures' / 'ffhq_fixed_lambda_z_breakdown.png'),
    },
    'vanhateren': {
        'label': 'Van Hateren',
        'spectrum': (
            REPO_ROOT / 'tables' / 'vanhateren_disk_teacher_spectrum.npz'),
        'summary': (
            REPO_ROOT / 'tables' / 'vanhateren_fixed_vs_cv_summary.csv'),
        'n': 1000,
        'fixed_table': (
            REPO_ROOT / 'tables' /
            'vanhateren_fixed_lambda_z_breakdown.csv'),
        'fixed_figure': (
            REPO_ROOT / 'figures' /
            'vanhateren_fixed_lambda_z_breakdown.png'),
        'cv_table': (
            REPO_ROOT / 'tables' /
            'vanhateren_cv_spectral_mismatch.csv'),
        'profile_table': (
            REPO_ROOT / 'tables' /
            'vanhateren_spectral_teacher_profile.csv'),
        'cv_figure': (
            REPO_ROOT / 'figures' /
            'vanhateren_cv_spectral_mismatch.png'),
    },
}


def configure_logging(path: Path) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger('natural_image_z_breakdowns')
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


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def summary_stats(values: np.ndarray, prefix: str) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    std = float(np.std(values, ddof=1)) if len(values) > 1 else np.nan
    return {
        f'{prefix}_mean': float(np.mean(values)),
        f'{prefix}_se': std / np.sqrt(len(values)),
        f'{prefix}_median': float(np.median(values)),
        f'{prefix}_q25': float(np.quantile(values, 0.25)),
        f'{prefix}_q75': float(np.quantile(values, 0.75)),
        f'{prefix}_min': float(np.min(values)),
        f'{prefix}_max': float(np.max(values)),
    }


def load_spectrum(dataset: str) -> tuple[np.ndarray, np.ndarray]:
    with np.load(DATASETS[dataset]['spectrum']) as spectrum:
        eigenvalues = np.asarray(spectrum['eigenvalues'], dtype=float)
        beta_proj = np.asarray(spectrum['beta_proj'], dtype=float)
    return eigenvalues, beta_proj


def reconstruct_ffhq_fixed_z(
        acc_error: np.ndarray, r2_acc: np.ndarray,
        signal_power: float) -> np.ndarray:
    """Recover signed z from the two exact path metrics in old FFHQ caches.

    The fixed-alpha FFHQ cache predates storing trial slopes.  For positive
    R=beta_hat^T beta_star / ||beta_hat||^2,

        E_acc/S = (1-R)^2,       1-R2_acc = (1/R-1)^2.

    Their ratio gives R^2, including whether R is above or below one.
    """
    acc_magnitude = np.sqrt(np.maximum(acc_error / signal_power, 0.0))
    z_magnitude = np.sqrt(np.maximum(1.0 - r2_acc, 0.0))
    slope = np.divide(
        acc_magnitude, z_magnitude,
        out=np.ones_like(acc_magnitude), where=z_magnitude > 1e-12)
    if np.any(slope <= 0):
        raise AssertionError('Expected positive fixed-ridge accentuation slope')
    return 1.0 / slope - 1.0


def fixed_trial_metrics(
        dataset: str, summary: dict[str, str],
        signal_power: float) -> tuple[np.ndarray, np.ndarray]:
    if dataset == 'ffhq':
        path = fixed_case_path(
            float(summary['alpha']), float(summary['sigma']),
            int(float(summary['n_trials'])), 20260814)
        with np.load(path, allow_pickle=False) as case:
            r2 = np.asarray(case['trial_r2_acc'], dtype=float)
            z = reconstruct_ffhq_fixed_z(
                np.asarray(case['trial_acc_error'], dtype=float),
                r2, signal_power)
    else:
        path = Path(summary['case_path'])
        with np.load(path, allow_pickle=False) as case:
            slope = np.asarray(case['fixed_trial_slope_acc'], dtype=float)
            r2 = np.asarray(case['fixed_trial_r2_acc'], dtype=float)
        z = 1.0 / slope - 1.0
    identity_error = float(np.max(
        np.abs(r2 - (1.0 - z ** 2)) / np.maximum(1.0, np.abs(r2))))
    if identity_error > 2e-5:
        raise AssertionError(
            f'{dataset} cached trial relative identity error '
            f'{identity_error:g}')
    return z, r2


def compute_fixed_breakdown(
        dataset: str, logger: logging.Logger) -> list[dict[str, float]]:
    config = DATASETS[dataset]
    source_rows = [
        row for row in read_rows(config['summary'])
        if row['policy'] == 'fixed'
    ]
    source_rows.sort(key=lambda row: float(row['sigma']))
    eigenvalues, beta_proj = load_spectrum(dataset)
    beta_sq = beta_proj ** 2
    signal_power = float(np.sum(eigenvalues * beta_sq))
    n = int(config['n'])
    alpha_values = {float(row['alpha']) for row in source_rows}
    if len(alpha_values) != 1:
        raise AssertionError(f'Expected one fixed alpha, got {alpha_values}')
    alpha = alpha_values.pop()
    kappa_min = ridgeless_kappa(eigenvalues, n)
    kappa = solve_kappa_for_alpha(eigenvalues, alpha, n, kappa_min)
    terms = spectral_terms(eigenvalues, beta_sq, kappa, n)
    denominator = terms['h'] * terms['B_11']
    shrink_numerator = alpha * terms['B_12']
    sigma_z1 = np.sqrt((denominator + shrink_numerator) / terms['df_12'])

    output: list[dict[str, float]] = []
    for index, source in enumerate(source_rows, start=1):
        sigma = float(source['sigma'])
        noise_numerator = sigma ** 2 * terms['df_12']
        net_numerator = noise_numerator - shrink_numerator
        z_noise = noise_numerator / denominator
        z_regularization = -shrink_numerator / denominator
        z_de = z_noise + z_regularization
        z_from_cached_theory = (
            1.0 / float(source['theory_acc_ratio_leading_mean']) - 1.0)
        if not np.isclose(z_de, z_from_cached_theory, rtol=2e-9, atol=2e-10):
            raise AssertionError(
                f'{dataset} fixed-z mismatch at sigma={sigma:g}: '
                f'{z_de:g} vs {z_from_cached_theory:g}')
        z_trial, r2_trial = fixed_trial_metrics(
            dataset, source, signal_power)
        row: dict[str, float] = {
            'sigma': sigma,
            'noise_signal_ratio': sigma ** 2 / signal_power,
            'signal_power': signal_power,
            'n': float(n),
            'alpha': alpha,
            'lambda': alpha / n,
            'kappa': kappa,
            'df_12_Q': terms['df_12'],
            'B_12_C': terms['B_12'],
            'h_n_minus_df_22': terms['h'],
            'B_11_A': terms['B_11'],
            'noise_numerator_sigma2_Q': noise_numerator,
            'shrink_numerator_alpha_C': shrink_numerator,
            'net_numerator': net_numerator,
            'denominator_h_A': denominator,
            'sigma_z1': sigma_z1,
            'noise_signal_ratio_z1': sigma_z1 ** 2 / signal_power,
            'z_noise': z_noise,
            'z_regularization': z_regularization,
            'z_de': z_de,
            'r2_de_leading': 1.0 - z_de ** 2,
            'r2_de_second_order': float(
                source.get('theory_r2_acc_second_order',
                           source['theory_r2_acc'])),
        }
        row.update(summary_stats(z_trial, 'mc_z'))
        row.update(summary_stats(r2_trial, 'mc_r2'))
        output.append(row)
        logger.info(
            '%s fixed [%d/%d] sigma=%g | z DE/MC median %.4g/%.4g '
            '| R2 DE/MC median %.4g/%.4g',
            config['label'], index, len(source_rows), sigma,
            z_de, row['mc_z_median'], row['r2_de_leading'],
            row['mc_r2_median'])

    write_rows(config['fixed_table'], output)
    return output


def set_noise_xaxis(ax: plt.Axes, max_sigma: float) -> None:
    ax.set_xscale('symlog', linthresh=0.1)
    ax.set_xlim(0, max_sigma)
    ax.set_xlabel(r'response noise $\sigma$')
    ax.grid(alpha=0.18)


def plot_fixed_breakdown(
        dataset: str, rows: list[dict[str, float]]) -> None:
    config = DATASETS[dataset]
    sigma = np.asarray([row['sigma'] for row in rows])
    max_sigma = float(np.max(sigma))
    signal_power = rows[0]['signal_power']
    alpha = rows[0]['alpha']
    lam = rows[0]['lambda']
    kappa = rows[0]['kappa']
    sigma_z1 = rows[0]['sigma_z1']

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.5))
    axes = axes.ravel()

    z_noise = np.asarray([row['z_noise'] for row in rows])
    z_reg = np.asarray([row['z_regularization'] for row in rows])
    z_de = np.asarray([row['z_de'] for row in rows])
    ax = axes[0]
    ax.plot(sigma, z_noise, color='C0', lw=2.4,
            label=r'noise $+\sigma^2Q/(hA)$')
    ax.plot(sigma, z_reg, color='C3', lw=2.4,
            label=r'shrinkage $-\alpha C/(hA)$')
    ax.plot(sigma, z_de, color='k', lw=2.8,
            label=r'total $z_{\rm DE}$')
    ax.axhline(0, color='0.45', lw=1)
    ax.axhline(1, color='0.35', lw=1.2, ls=':', label=r'$z=1$ failure')
    ax.axvline(sigma_z1, color='C4', lw=1.3, ls='--',
               label=rf'$z=1$ at $\sigma={sigma_z1:.2g}$')
    ax.set_yscale('symlog', linthresh=0.1)
    ax.set_ylabel('normalized contribution to z')
    ax.set_title(r'$z$ decomposition: noise grows, ridge term is fixed')
    ax.legend(fontsize=8, loc='best')

    noise_num = np.asarray([
        row['noise_numerator_sigma2_Q'] for row in rows])
    shrink_num = np.asarray([
        row['shrink_numerator_alpha_C'] for row in rows])
    net_num = np.asarray([row['net_numerator'] for row in rows])
    denom = np.asarray([row['denominator_h_A'] for row in rows])
    ax = axes[1]
    ax.plot(sigma, noise_num, color='C0', lw=2.4,
            label=r'noise numerator $\sigma^2Q$')
    ax.plot(sigma, shrink_num, color='C3', lw=2.2,
            label=r'shrinkage magnitude $\alpha C$')
    ax.plot(sigma, net_num, color='k', lw=2.6,
            label=r'net $\sigma^2Q-\alpha C$')
    ax.plot(sigma, denom, color='C2', lw=2.2, ls='--',
            label=r'failure threshold $hA$')
    ax.axvline(sigma_z1, color='C4', lw=1.3, ls='--')
    ax.set_yscale('symlog', linthresh=max(float(denom[0]) * 0.03, 1e-8))
    ax.set_ylabel('raw numerator / denominator terms')
    ax.set_title(r'$z=1$ exactly when $\sigma^2Q-\alpha C=hA$')
    ax.legend(fontsize=8, loc='best')

    mc_z_median = np.asarray([row['mc_z_median'] for row in rows])
    mc_z_q25 = np.asarray([row['mc_z_q25'] for row in rows])
    mc_z_q75 = np.asarray([row['mc_z_q75'] for row in rows])
    mc_z_mean = np.asarray([row['mc_z_mean'] for row in rows])
    ax = axes[2]
    ax.plot(sigma, z_de, color='C0', lw=2.6, label='leading DE')
    ax.plot(sigma, mc_z_median, color='k', marker='o', ms=3.6, lw=1.4,
            label='natural-image MC median / IQR')
    ax.fill_between(sigma, mc_z_q25, mc_z_q75,
                    color='k', alpha=0.12, linewidth=0)
    ax.plot(sigma, mc_z_mean, color='0.4', marker='x', ms=4, lw=1,
            ls=':', label='MC mean')
    ax.axhline(0, color='0.45', lw=1)
    ax.axhline(1, color='0.35', lw=1.2, ls=':')
    ax.axhline(-1, color='0.35', lw=1.2, ls=':')
    ax.axvline(sigma_z1, color='C4', lw=1.3, ls='--')
    ax.set_yscale('symlog', linthresh=0.1)
    ax.set_ylabel(r'$z_{\rm acc}=\mu_D/\mu_N-1$')
    ax.set_title(r'Fixed-$\lambda$ calibration gap: DE versus experiment')
    ax.legend(fontsize=8, loc='best')

    r2_leading = np.asarray([row['r2_de_leading'] for row in rows])
    r2_second = np.asarray([row['r2_de_second_order'] for row in rows])
    mc_r2_median = np.asarray([row['mc_r2_median'] for row in rows])
    mc_r2_q25 = np.asarray([row['mc_r2_q25'] for row in rows])
    mc_r2_q75 = np.asarray([row['mc_r2_q75'] for row in rows])
    mc_r2_mean = np.asarray([row['mc_r2_mean'] for row in rows])
    ax = axes[3]
    ax.plot(sigma, r2_leading, color='C0', lw=2.6,
            label=r'leading DE $1-z_{\rm DE}^2$')
    ax.plot(sigma, r2_second, color='C1', lw=2, ls='--',
            label='current response-noise second order')
    ax.plot(sigma, mc_r2_median, color='k', marker='o', ms=3.6, lw=1.4,
            label='natural-image MC median / IQR')
    ax.fill_between(sigma, mc_r2_q25, mc_r2_q75,
                    color='k', alpha=0.12, linewidth=0)
    ax.plot(sigma, mc_r2_mean, color='0.4', marker='x', ms=4, lw=1,
            ls=':', label='MC mean')
    ax.axhline(0, color='0.45', lw=1)
    ax.axvline(sigma_z1, color='C4', lw=1.3, ls='--')
    ax.set_yscale('symlog', linthresh=1.0)
    ax.set_ylabel(r'$R^2_{\rm acc}$')
    ax.set_title(r'Fixed-$\lambda$ accentuation $R^2$: DE versus experiment')
    ax.legend(fontsize=8, loc='best')

    for ax in axes:
        set_noise_xaxis(ax, max_sigma)
    add_noise_ratio_axis(axes[0], signal_power, max_sigma)
    add_noise_ratio_axis(axes[1], signal_power, max_sigma)
    fig.suptitle(
        f"{config['label']} disk teacher: fixed ridge "
        rf'$\alpha={alpha:g}$, $\lambda={lam:g}$, $\kappa={kappa:.3g}$',
        fontsize=15, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    config['fixed_figure'].parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(config['fixed_figure'], dpi=190, bbox_inches='tight')
    config['fixed_figure'].chmod(0o644)
    plt.close(fig)


def vanhateren_cv_trial_metrics(path: Path) -> dict[str, float]:
    with np.load(path, allow_pickle=False) as case:
        slope = np.asarray(case['cv_trial_slope_acc'], dtype=float)
        r2 = np.asarray(case['cv_trial_r2_acc'], dtype=float)
    z = 1.0 / slope - 1.0
    values = {}
    values.update(summary_stats(z, 'mc_z'))
    values.update(summary_stats(r2, 'mc_r2'))
    return values


def compute_vanhateren_cv(
        kappa_grid_size: int, logger: logging.Logger
        ) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    config = DATASETS['vanhateren']
    summary_rows = [
        row for row in read_rows(config['summary']) if row['policy'] == 'cv'
    ]
    summary_rows.sort(key=lambda row: float(row['sigma']))
    eigenvalues, beta_proj = load_spectrum('vanhateren')
    beta_sq = beta_proj ** 2
    signal_power = float(np.sum(eigenvalues * beta_sq))
    n = int(float(summary_rows[0]['n']))
    profile_rows = spectral_profile(eigenvalues, beta_sq)

    kappa_min = ridgeless_kappa(eigenvalues, n)
    kappa_grid = np.logspace(
        np.log10(kappa_min * (1.0 + 1e-10)), 5.0, kappa_grid_size)
    grid_terms = [
        spectral_terms(eigenvalues, beta_sq, kappa, n)
        for kappa in kappa_grid
    ]

    probe_start = time.perf_counter()
    oracle_kappa(
        eigenvalues, beta_sq, float(summary_rows[-1]['sigma']), n,
        kappa_grid, grid_terms, kappa_min)
    probe_seconds = time.perf_counter() - probe_start
    logger.info(
        'Van Hateren CV oracle probe %.3fs; projected %.2fs for %d levels',
        probe_seconds, probe_seconds * len(summary_rows), len(summary_rows))

    output: list[dict[str, float]] = []
    for index, summary in enumerate(summary_rows, start=1):
        sigma = float(summary['sigma'])
        grid_alpha = float(summary['alpha'])
        grid_kappa = solve_kappa_for_alpha(
            eigenvalues, grid_alpha, n, kappa_min)
        grid_values = spectral_terms(eigenvalues, beta_sq, grid_kappa, n)
        grid_z = z_metrics(grid_values, grid_kappa, sigma, grid_alpha)

        optimum_kappa, optimum_interior, optimum_risk = oracle_kappa(
            eigenvalues, beta_sq, sigma, n,
            kappa_grid, grid_terms, kappa_min)
        oracle_values = spectral_terms(
            eigenvalues, beta_sq, optimum_kappa, n)
        oracle_lambda, oracle_alpha = lambda_from_kappa(
            oracle_values, optimum_kappa, n)
        if abs(oracle_lambda) < 1e-12:
            oracle_lambda = 0.0
            oracle_alpha = 0.0
        oracle_z = z_metrics(
            oracle_values, optimum_kappa, sigma, oracle_alpha)
        if optimum_interior and not np.isclose(
                oracle_z['z_direct'], oracle_z['z_simplified'],
                rtol=2e-7, atol=2e-9):
            raise AssertionError(
                f'Van Hateren oracle mismatch at sigma={sigma:g}')

        alpha_median = float(summary['mc_alpha_cv_median'])
        alpha_q25 = float(summary['mc_alpha_cv_q25'])
        alpha_q75 = float(summary['mc_alpha_cv_q75'])
        mc_kappa_median = solve_kappa_for_alpha(
            eigenvalues, alpha_median, n, kappa_min)
        mc_kappa_q25 = solve_kappa_for_alpha(
            eigenvalues, alpha_q25, n, kappa_min)
        mc_kappa_q75 = solve_kappa_for_alpha(
            eigenvalues, alpha_q75, n, kappa_min)

        row: dict[str, float] = {
            'sigma': sigma,
            'noise_signal_ratio': sigma ** 2 / signal_power,
            'signal_power': signal_power,
            'grid_alpha_cv': grid_alpha,
            'grid_lambda': grid_alpha / n,
            'grid_kappa_final': grid_kappa,
            'mc_alpha_median': alpha_median,
            'mc_alpha_q25': alpha_q25,
            'mc_alpha_q75': alpha_q75,
            'mc_lambda_median': alpha_median / n,
            'mc_lambda_q25': alpha_q25 / n,
            'mc_lambda_q75': alpha_q75 / n,
            'mc_kappa_median': mc_kappa_median,
            'mc_kappa_q25': mc_kappa_q25,
            'mc_kappa_q75': mc_kappa_q75,
            'oracle_kappa': optimum_kappa,
            'oracle_alpha': oracle_alpha,
            'oracle_lambda': oracle_lambda,
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
        row.update(vanhateren_cv_trial_metrics(Path(summary['case_path'])))
        output.append(row)
        oracle_z_plot = (
            oracle_z['z_simplified'] if optimum_interior
            else oracle_z['z_direct'])
        logger.info(
            'Van Hateren CV [%d/%d] sigma=%g | lambda oracle/grid/MC '
            '%.4g/%.4g/%.4g | z oracle/grid/MC %.4g/%.4g/%.4g',
            index, len(summary_rows), sigma,
            oracle_lambda, grid_alpha / n, alpha_median / n,
            oracle_z_plot, grid_z['z_direct'], row['mc_z_median'])

    write_rows(config['cv_table'], output)
    write_rows(config['profile_table'], profile_rows)
    return output, profile_rows


def plot_vanhateren_cv(
        rows: list[dict[str, float]],
        profile_rows: list[dict[str, float]]) -> None:
    config = DATASETS['vanhateren']
    sigma = np.asarray([row['sigma'] for row in rows])
    max_sigma = float(np.max(sigma))
    signal_power = rows[0]['signal_power']
    nsr = np.asarray([row['noise_signal_ratio'] for row in rows])
    profile_s = np.asarray([row['s_center'] for row in profile_rows])
    profile_teacher = np.asarray([
        row['mean_teacher_power'] for row in profile_rows])
    profile_count = np.asarray([row['mode_count'] for row in profile_rows])

    fig, axes = plt.subplots(3, 3, figsize=(17.2, 13.8))
    axes = axes.ravel()

    axes[0].plot(
        profile_s, profile_teacher, color='C4', lw=2.1, marker='o', ms=3)
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    axes[0].set_xlabel(r'population eigenvalue $s_k$')
    axes[0].set_ylabel(r'bin mean teacher power $\langle b_k^2\rangle$')
    axes[0].set_title('Disk-teacher power along the spectrum')

    axes[1].step(
        profile_s, profile_count, where='mid', color='0.25', lw=2)
    axes[1].fill_between(
        profile_s, profile_count, step='mid', color='0.5', alpha=0.18)
    axes[1].set_xscale('log')
    axes[1].set_yscale('log')
    axes[1].set_xlabel(r'population eigenvalue $s_k$')
    axes[1].set_ylabel('modes per log-spectral bin')
    axes[1].set_title('Van Hateren spectral density')

    selected_ratios = np.asarray([0.01, 0.1, 1.0, 10.0])
    for ratio, color in zip(selected_ratios, ['C0', 'C1', 'C2', 'C3']):
        index = int(np.argmin(np.abs(
            np.log10(np.maximum(nsr, 1e-20)) - np.log10(ratio))))
        kappa = rows[index]['oracle_kappa']
        label = rf'$\sigma^2/S\approx{ratio:g}$'
        for static_ax in axes[:2]:
            static_ax.axvline(
                kappa, color=color, lw=1.4, alpha=0.8,
                label=label if static_ax is axes[1] else None)
    axes[1].legend(fontsize=8, loc='best')

    oracle_lambda = np.asarray([row['oracle_lambda'] for row in rows])
    grid_lambda = np.asarray([row['grid_lambda'] for row in rows])
    mc_lambda = np.asarray([row['mc_lambda_median'] for row in rows])
    mc_lambda_q25 = np.asarray([row['mc_lambda_q25'] for row in rows])
    mc_lambda_q75 = np.asarray([row['mc_lambda_q75'] for row in rows])
    ax = axes[2]
    ax.plot(sigma, oracle_lambda, color='C0', lw=2.4,
            label='same-n continuous DE oracle')
    ax.plot(sigma, grid_lambda, color='C1', lw=2, ls='--',
            label='LOOCV-grid DE, final refit')
    ax.plot(sigma, mc_lambda, color='k', marker='o', ms=3.5, lw=1.4,
            label='MC RidgeCV median / IQR')
    ax.fill_between(sigma, mc_lambda_q25, mc_lambda_q75,
                    color='k', alpha=0.12, linewidth=0)
    ax.set_yscale('symlog', linthresh=1e-6)
    ax.set_ylabel(r'selected $\lambda=\alpha/n$')
    ax.set_title(r'RidgeCV penalty $\lambda(\sigma)$')
    ax.legend(fontsize=8, loc='best')

    oracle_k = np.asarray([row['oracle_kappa'] for row in rows])
    grid_k = np.asarray([row['grid_kappa_final'] for row in rows])
    mc_k = np.asarray([row['mc_kappa_median'] for row in rows])
    mc_k_q25 = np.asarray([row['mc_kappa_q25'] for row in rows])
    mc_k_q75 = np.asarray([row['mc_kappa_q75'] for row in rows])
    ax = axes[3]
    ax.plot(sigma, oracle_k, color='C0', lw=2.4,
            label='same-n continuous DE oracle')
    ax.plot(sigma, grid_k, color='C1', lw=2, ls='--',
            label='LOOCV-grid DE, final refit')
    ax.plot(sigma, mc_k, color='k', marker='o', ms=3.5, lw=1.4,
            label='MC RidgeCV median / IQR')
    ax.fill_between(sigma, mc_k_q25, mc_k_q75,
                    color='k', alpha=0.12, linewidth=0)
    ax.set_yscale('log')
    ax.set_ylabel(r'effective cutoff $\kappa$')
    ax.set_title(r'Induced moving spectral cutoff $\kappa(\sigma)$')
    ax.legend(fontsize=8, loc='best')

    bgen = np.asarray([row['oracle_b2_gen_average'] for row in rows])
    bacc = np.asarray([row['oracle_b2_acc_average'] for row in rows])
    ax = axes[4]
    ax.plot(sigma, bgen, color='C0', lw=2.4,
            label=r'$B_{2,3}/\mathrm{df}_{2,3}$: gen-weighted')
    ax.plot(sigma, bacc, color='C1', lw=2.4,
            label=r'$B_{1,2}/\mathrm{df}_{1,2}$: acc-weighted')
    ax.fill_between(sigma, bacc, bgen, color='0.4', alpha=0.12)
    ax.set_yscale('log')
    ax.set_ylabel('moving-window mean teacher power')
    ax.set_title('Two moving spectral averages')
    ax.legend(fontsize=8, loc='best')

    recoverable = np.asarray([row['oracle_B_11'] for row in rows])
    recoverable_grid = np.asarray([row['grid_B_11'] for row in rows])
    ax = axes[5]
    ax.plot(sigma, recoverable, color='C2', lw=2.4,
            label='same-n continuous DE oracle')
    ax.plot(sigma, recoverable_grid, color='C2', lw=1.8, ls='--',
            label='LOOCV-grid DE, final refit')
    ax.set_ylabel(r'recoverable teacher power $B_{1,1}$')
    ax.set_title(r'Retained alignment $B_{1,1}(\kappa)$')
    ax.legend(fontsize=8, loc='best')

    limbo = np.asarray([row['oracle_limbo_count'] for row in rows])
    limbo_grid = np.asarray([row['grid_limbo_count'] for row in rows])
    ax = axes[6]
    ax.plot(sigma, limbo, color='C3', lw=2.4,
            label=r'oracle $\kappa\,\mathrm{df}_{1,2}$')
    ax.plot(sigma, limbo_grid, color='C3', lw=1.8, ls='--',
            label='LOOCV-grid DE, final refit')
    ax.set_ylabel(r'limbo-mode count $\kappa\,\mathrm{df}_{1,2}$')
    ax.set_title('Number of modes near the ridge transition')
    ax.legend(fontsize=8, loc='best')

    interior = np.asarray([row['oracle_interior'] > 0.5 for row in rows])
    oracle_z = np.where(
        interior,
        np.asarray([row['oracle_z_simplified'] for row in rows]),
        np.asarray([row['oracle_z_direct'] for row in rows]))
    grid_z = np.asarray([row['grid_z_direct'] for row in rows])
    mc_z = np.asarray([row['mc_z_median'] for row in rows])
    mc_z_q25 = np.asarray([row['mc_z_q25'] for row in rows])
    mc_z_q75 = np.asarray([row['mc_z_q75'] for row in rows])
    ax = axes[7]
    ax.plot(sigma, oracle_z, color='C0', lw=2.4,
            label='same-n oracle DE')
    ax.plot(sigma, grid_z, color='C1', lw=2.1, ls='--',
            label=r'direct grid/final DE $(\sigma^2,\alpha)$')
    ax.plot(sigma, mc_z, color='k', marker='o', ms=3.5, lw=1.3,
            label='natural-image RidgeCV median / IQR')
    ax.fill_between(sigma, mc_z_q25, mc_z_q75,
                    color='k', alpha=0.12, linewidth=0)
    ax.axhline(0, color='0.4', lw=1)
    ax.set_yscale('symlog', linthresh=1.0)
    ax.axhline(1, color='0.4', lw=1, ls=':')
    ax.axhline(-1, color='0.4', lw=1, ls=':')
    ax.set_ylabel(r'$z_{\rm acc}=\mu_D/\mu_N-1$')
    ax.set_title('Accentuation calibration gap')
    ax.legend(fontsize=8, loc='best')

    oracle_r2 = 1.0 - oracle_z ** 2
    grid_r2 = 1.0 - grid_z ** 2
    mc_r2 = np.asarray([row['mc_r2_median'] for row in rows])
    mc_r2_q25 = np.asarray([row['mc_r2_q25'] for row in rows])
    mc_r2_q75 = np.asarray([row['mc_r2_q75'] for row in rows])
    mc_r2_mean = np.asarray([row['mc_r2_mean'] for row in rows])
    ax = axes[8]
    ax.plot(sigma, oracle_r2, color='C0', lw=2.4,
            label='same-n oracle DE')
    ax.plot(sigma, grid_r2, color='C1', lw=2.1, ls='--',
            label='direct grid/final DE')
    ax.plot(sigma, mc_r2, color='k', marker='o', ms=3.5, lw=1.3,
            label='MC median / IQR')
    ax.fill_between(sigma, mc_r2_q25, mc_r2_q75,
                    color='k', alpha=0.12, linewidth=0)
    ax.plot(sigma, mc_r2_mean, color='0.35', marker='x', ms=4, lw=1,
            ls=':', label='MC mean')
    ax.axhline(0, color='0.4', lw=1)
    ax.set_yscale('symlog', linthresh=1.0)
    ax.set_ylabel(r'$R^2_{\rm acc}=1-z^2$')
    ax.set_title(r'Leading DE versus natural-image RidgeCV $R^2_{\rm acc}$')
    ax.legend(fontsize=8, loc='best')

    for static_ax in axes[:2]:
        static_ax.grid(alpha=0.18)
    for dynamic_ax in axes[2:]:
        set_noise_xaxis(dynamic_ax, max_sigma)
    add_noise_ratio_axis(axes[2], signal_power, max_sigma)
    fig.suptitle(
        'Van Hateren disk teacher: RidgeCV selection and spectral mismatch',
        fontsize=16, y=0.998)
    fig.tight_layout(rect=(0, 0, 1, 0.982))
    config['cv_figure'].parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(config['cv_figure'], dpi=190, bbox_inches='tight')
    config['cv_figure'].chmod(0o644)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--kappa-grid-size', type=int, default=800)
    args = parser.parse_args()
    logger = configure_logging(LOG_PATH)
    start = time.perf_counter()

    benchmark_start = time.perf_counter()
    ffhq_rows = compute_fixed_breakdown('ffhq', logger)
    benchmark_seconds = time.perf_counter() - benchmark_start
    logger.info(
        'Fixed-breakdown probe/FFHQ completed in %.2fs; '
        'projected %.2fs for two datasets',
        benchmark_seconds, 2 * benchmark_seconds)
    plot_fixed_breakdown('ffhq', ffhq_rows)

    vanhateren_fixed_rows = compute_fixed_breakdown('vanhateren', logger)
    plot_fixed_breakdown('vanhateren', vanhateren_fixed_rows)

    vanhateren_cv_rows, profile_rows = compute_vanhateren_cv(
        args.kappa_grid_size, logger)
    plot_vanhateren_cv(vanhateren_cv_rows, profile_rows)

    for dataset in ('ffhq', 'vanhateren'):
        config = DATASETS[dataset]
        logger.info('Wrote %s', config['fixed_table'])
        logger.info('Wrote %s', config['fixed_figure'])
    logger.info('Wrote %s', DATASETS['vanhateren']['cv_table'])
    logger.info('Wrote %s', DATASETS['vanhateren']['profile_table'])
    logger.info('Wrote %s', DATASETS['vanhateren']['cv_figure'])
    logger.info('Completed all cached analyses in %.2fs',
                time.perf_counter() - start)


if __name__ == '__main__':
    main()

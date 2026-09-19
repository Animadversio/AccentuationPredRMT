"""Diagnose the DE--MC gap along the Van Hateren null Givens rotation.

This script consumes the raw cache written by
``validate_vanhateren_null_rotations.py``.  It does not refit ridge models.
Besides the leading deterministic equivalent (DE), which substitutes the
ratio of expected accentuation numerator and denominator, it evaluates a
distributional Gaussian DE:

    theta_hat ~ N(mean_DE, covariance_DE).

The Gaussian surrogate propagates the DE estimator distribution through the
nonlinear per-fit accentuation statistics.  This separates errors in the DE
first two moments from errors caused by replacing an expectation of a ratio by
a ratio of expectations.

Plot-ready summaries are cached independently of the rendered figures.
"""
from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/accentuationpredrmt-matplotlib')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp/accentuationpredrmt-xdg-cache')

import matplotlib.pyplot as plt
import numpy as np

from rmt_core import spectral_feature_de_metrics
from scripts.validate_ffhq_disk_teacher import configure_logger
from scripts.validate_vanhateren_null_rotations import (
    RAW_PATH,
    SUMMARY_PATH,
    common_de_settings,
    read_rows,
    write_rows,
)
from scripts.storage_paths import require_bulk_path


REPO_ROOT = Path(__file__).resolve().parents[1]
DIAGNOSTIC_SUMMARY_PATH = (
    REPO_ROOT / 'tables' / 'vanhateren_null_rotation_gap_summary.csv')
OVERVIEW_FIGURE_PATH = (
    REPO_ROOT / 'figures' / 'null_rotations' /
    'vanhateren_null_givens_full_metrics.png')
GAP_FIGURE_PATH = (
    REPO_ROOT / 'figures' / 'null_rotations' /
    'vanhateren_null_givens_de_gap.png')
DEFAULT_LOG_PATH = REPO_ROOT / 'logs' / 'vanhateren_null_rotation_gap.log'


def gaussian_quadratic_moments(
        mean: np.ndarray, variance: float, metric: np.ndarray,
        teacher_overlap: np.ndarray) -> dict[str, float]:
    """Moments of N=q^T theta and D=theta^T G theta for isotropic Gaussian theta."""
    mean = np.asarray(mean, dtype=float)
    metric = np.asarray(metric, dtype=float)
    teacher_overlap = np.asarray(teacher_overlap, dtype=float)
    metric_sq = metric @ metric
    mean_numerator = float(teacher_overlap @ mean)
    mean_denominator = float(
        mean @ metric @ mean + variance * np.trace(metric))
    variance_numerator = float(variance * teacher_overlap @ teacher_overlap)
    variance_denominator = float(
        2.0 * variance ** 2 * np.trace(metric_sq)
        + 4.0 * variance * mean @ metric_sq @ mean)
    covariance = float(2.0 * variance * teacher_overlap @ metric @ mean)
    return {
        'mean_numerator': mean_numerator,
        'mean_denominator': mean_denominator,
        'variance_numerator': variance_numerator,
        'variance_denominator': variance_denominator,
        'covariance_numerator_denominator': covariance,
        'cv_denominator': np.sqrt(variance_denominator) / mean_denominator,
    }


def ratio_of_moments_metrics(
        mean_numerator: float, mean_denominator: float) -> dict[str, float]:
    """Leading-DE accentuation statistics formed from E[N] and E[D]."""
    slope = mean_numerator / mean_denominator
    return {
        'slope_acc': slope,
        'acc_error_normalized': (1.0 - slope) ** 2,
        'r2_acc': 1.0 - (1.0 - mean_denominator / mean_numerator) ** 2,
    }


def delta_method_metrics(moments: dict[str, float]) -> dict[str, float]:
    """Second-order delta approximation for nonlinear accentuation metrics.

    The approximation is included as a diagnostic, not as a reliable formula
    when the coefficient of variation of the denominator is order one.
    """
    mu_n = moments['mean_numerator']
    mu_d = moments['mean_denominator']
    var_n = moments['variance_numerator']
    var_d = moments['variance_denominator']
    cov_nd = moments['covariance_numerator_denominator']

    mean_ratio = (
        mu_n / mu_d - cov_nd / mu_d ** 2 + mu_n * var_d / mu_d ** 3)
    var_ratio = (
        var_n / mu_d ** 2 + mu_n ** 2 * var_d / mu_d ** 4
        - 2.0 * mu_n * cov_nd / mu_d ** 3)
    mean_inverse_ratio = (
        mu_d / mu_n - cov_nd / mu_n ** 2 + mu_d * var_n / mu_n ** 3)
    var_inverse_ratio = (
        var_d / mu_n ** 2 + mu_d ** 2 * var_n / mu_n ** 4
        - 2.0 * mu_d * cov_nd / mu_n ** 3)
    return {
        'slope_acc': mean_ratio,
        'acc_error_normalized': (1.0 - mean_ratio) ** 2 + var_ratio,
        'r2_acc_mean': (
            1.0 - (1.0 - mean_inverse_ratio) ** 2 - var_inverse_ratio),
    }


def nonlinear_metrics(
        numerator: np.ndarray, denominator: np.ndarray
        ) -> dict[str, np.ndarray]:
    """Evaluate per-fit accentuation statistics."""
    slope = numerator / denominator
    inverse_slope = denominator / numerator
    return {
        'slope_acc': slope,
        'acc_error_normalized': (1.0 - slope) ** 2,
        'r2_acc': 1.0 - (1.0 - inverse_slope) ** 2,
    }


def bootstrap_median_interval(
        values: np.ndarray, indices: np.ndarray,
        quantiles: tuple[float, float] = (0.025, 0.975)
        ) -> tuple[float, float]:
    """Percentile bootstrap interval using pre-generated resample indices."""
    medians = np.median(np.asarray(values)[indices], axis=1)
    low, high = np.quantile(medians, quantiles)
    return float(low), float(high)


def _mean_and_se(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    return float(np.mean(values)), float(np.std(values, ddof=1) / np.sqrt(len(values)))


def _surrogate_summaries(
        samples: np.ndarray, maps: np.ndarray, teacher: np.ndarray
        ) -> list[dict[str, float]]:
    """Propagate Gaussian estimator samples through each pixel metric."""
    summaries: list[dict[str, float]] = []
    for metric in maps:
        overlap = metric @ teacher
        numerator = samples @ overlap
        denominator = np.einsum(
            'bi,ij,bj->b', samples, metric, samples, optimize=True)
        nonlinear = nonlinear_metrics(numerator, denominator)
        summaries.append({
            'slope_acc': float(np.mean(nonlinear['slope_acc'])),
            'acc_error_normalized': float(np.mean(
                nonlinear['acc_error_normalized'])),
            'r2_acc_median': float(np.median(nonlinear['r2_acc'])),
            'r2_acc_q25': float(np.quantile(nonlinear['r2_acc'], 0.25)),
            'r2_acc_q75': float(np.quantile(nonlinear['r2_acc'], 0.75)),
        })
    return summaries


def build_diagnostic_rows(
        raw: np.lib.npyio.NpzFile, n: int, n_surrogate: int,
        bootstrap_reps: int, seed: int, logger: logging.Logger,
        map_limit: int | None = None, ratio_limit: int | None = None,
        surrogate_limit: int | None = None) -> list[dict[str, object]]:
    """Build plot-ready theory, MC, and distributional-DE summaries."""
    maps = np.asarray(raw['givens_maps'], dtype=float)
    theta = np.asarray(raw['plane_theta'], dtype=float)
    ratios = np.asarray(raw['ratios'], dtype=float)
    alphas = np.asarray(raw['alphas'], dtype=float)
    signal = float(raw['signal'])
    if map_limit is not None:
        maps = maps[:map_limit]
        theta = theta[:map_limit]
    if ratio_limit is not None:
        ratios = ratios[:ratio_limit]
    sample_count = n_surrogate if surrogate_limit is None else surrogate_limit
    n_maps, p, _ = maps.shape
    n_trials = raw['prediction_r2_gen'].shape[1]
    teacher = np.zeros(p)
    teacher[0] = np.sqrt(signal)
    settings = common_de_settings(p, signal, ratios, n, alphas)
    rng = np.random.default_rng(seed)
    bootstrap_indices = rng.integers(
        0, n_trials, size=(bootstrap_reps, n_trials))
    rows: list[dict[str, object]] = []

    trace_g = np.trace(maps, axis1=1, axis2=2)
    trace_g2 = np.einsum('mij,mji->m', maps, maps, optimize=True)
    effective_rank = trace_g ** 2 / trace_g2
    max_eigenvalue = np.linalg.eigvalsh(maps)[:, -1]

    for ratio_index, (ratio, setting) in enumerate(zip(ratios, settings)):
        base_metrics = spectral_feature_de_metrics(
            np.ones(p), teacher, np.ones(p), 0.0, signal,
            setting['kappa'], setting['sigma'], n)
        shrink = 1.0 / (1.0 + setting['kappa'])
        mean = shrink * teacher
        variance = (
            base_metrics['variance_scale'] / (1.0 + setting['kappa']) ** 2)
        samples = mean + np.sqrt(variance) * rng.standard_normal(
            (sample_count, p))
        surrogate = _surrogate_summaries(samples, maps, teacher)

        pred_e_mean, pred_e_se = _mean_and_se(
            raw['prediction_gen_error_normalized'][ratio_index])
        pred_r2_mean, pred_r2_se = _mean_and_se(
            raw['prediction_r2_gen'][ratio_index])
        alpha_values = raw['prediction_alpha_cv'][ratio_index]
        logger.info(
            'Distributional DE %d/%d: sigma^2/S=%g, alpha_DE=%.5g, '
            'MC alpha median/IQR=%.5g [%.5g, %.5g]',
            ratio_index + 1, len(ratios), ratio, setting['alpha_cv'],
            np.median(alpha_values), np.quantile(alpha_values, 0.25),
            np.quantile(alpha_values, 0.75))

        for map_index, metric in enumerate(maps):
            numerator = raw['givens_mc_acc_numerator'][map_index, ratio_index]
            denominator = raw['givens_mc_acc_denominator'][map_index, ratio_index]
            nonlinear = nonlinear_metrics(numerator, denominator)
            mc_slope_mean, mc_slope_se = _mean_and_se(
                nonlinear['slope_acc'])
            mc_error_mean, mc_error_se = _mean_and_se(
                nonlinear['acc_error_normalized'])
            mc_r2 = nonlinear['r2_acc']
            r2_low, r2_high = bootstrap_median_interval(
                mc_r2, bootstrap_indices)
            mean_numerator = float(np.mean(numerator))
            mean_denominator = float(np.mean(denominator))
            empirical_ratio = ratio_of_moments_metrics(
                mean_numerator, mean_denominator)

            overlap = metric @ teacher
            moments = gaussian_quadratic_moments(
                mean, variance, metric, overlap)
            leading = ratio_of_moments_metrics(
                moments['mean_numerator'], moments['mean_denominator'])
            delta = delta_method_metrics(moments)
            row: dict[str, object] = {
                'map_index': map_index,
                'theta': theta[map_index],
                'low_fraction': np.sin(theta[map_index]) ** 2,
                'trace_g': trace_g[map_index],
                'trace_g2': trace_g2[map_index],
                'effective_rank_g': effective_rank[map_index],
                'max_eigenvalue_g': max_eigenvalue[map_index],
                'noise_signal_ratio': ratio,
                'sigma': setting['sigma'],
                'de_alpha_cv': setting['alpha_cv'],
                'de_kappa': setting['kappa'],
                'mc_alpha_cv_mean': np.mean(alpha_values),
                'mc_alpha_cv_median': np.median(alpha_values),
                'mc_alpha_cv_q25': np.quantile(alpha_values, 0.25),
                'mc_alpha_cv_q75': np.quantile(alpha_values, 0.75),
                'de_gen_error_normalized': raw[
                    'givens_de_gen_error_normalized'][map_index, ratio_index],
                'de_r2_gen': raw['givens_de_r2_gen'][map_index, ratio_index],
                'mc_gen_error_normalized': pred_e_mean,
                'mc_gen_error_normalized_se': pred_e_se,
                'mc_r2_gen': pred_r2_mean,
                'mc_r2_gen_se': pred_r2_se,
                'de_slope_acc': leading['slope_acc'],
                'de_acc_error_normalized': leading['acc_error_normalized'],
                'de_r2_acc': leading['r2_acc'],
                'mc_moment_slope_acc': empirical_ratio['slope_acc'],
                'mc_moment_acc_error_normalized': empirical_ratio[
                    'acc_error_normalized'],
                'mc_moment_r2_acc': empirical_ratio['r2_acc'],
                'mc_slope_acc': mc_slope_mean,
                'mc_slope_acc_se': mc_slope_se,
                'mc_acc_error_normalized': mc_error_mean,
                'mc_acc_error_normalized_se': mc_error_se,
                'mc_r2_acc_median': np.median(mc_r2),
                'mc_r2_acc_q25': np.quantile(mc_r2, 0.25),
                'mc_r2_acc_q75': np.quantile(mc_r2, 0.75),
                'mc_r2_acc_median_ci_low': r2_low,
                'mc_r2_acc_median_ci_high': r2_high,
                'mc_denominator_mean': mean_denominator,
                'mc_denominator_cv': np.std(denominator, ddof=1) / mean_denominator,
                'gaussian_slope_acc': surrogate[map_index]['slope_acc'],
                'gaussian_acc_error_normalized': surrogate[map_index][
                    'acc_error_normalized'],
                'gaussian_r2_acc_median': surrogate[map_index][
                    'r2_acc_median'],
                'gaussian_r2_acc_q25': surrogate[map_index]['r2_acc_q25'],
                'gaussian_r2_acc_q75': surrogate[map_index]['r2_acc_q75'],
                'gaussian_denominator_mean': moments['mean_denominator'],
                'gaussian_denominator_cv': moments['cv_denominator'],
                'gaussian_variance_scale_per_coordinate': variance,
                'delta2_slope_acc': delta['slope_acc'],
                'delta2_acc_error_normalized': delta[
                    'acc_error_normalized'],
                'delta2_r2_acc_mean': delta['r2_acc_mean'],
            }
            rows.append(row)
    return rows


def _rows_for_ratio(
        rows: list[dict[str, object]], ratio: float
        ) -> list[dict[str, object]]:
    return sorted(
        [row for row in rows if np.isclose(
            float(row['noise_signal_ratio']), ratio)],
        key=lambda row: float(row['map_index']))


def _array(rows: list[dict[str, object]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows])


def _style_axis(ax: plt.Axes) -> None:
    ax.grid(alpha=0.18, which='both')
    ax.spines[['top', 'right']].set_alpha(0.35)


def plot_overview(rows: list[dict[str, object]], path: Path) -> None:
    """Plot prediction invariance and accentuation variation in aligned panels."""
    ratios = sorted({float(row['noise_signal_ratio']) for row in rows})
    colors = plt.get_cmap('viridis')(np.linspace(0.08, 0.90, len(ratios)))
    base = _rows_for_ratio(rows, ratios[0])
    x = _array(base, 'low_fraction')

    fig, axes = plt.subplots(2, 3, figsize=(15.2, 8.2), sharex=True)
    geometry = axes[0, 0]
    geometry.plot(x, _array(base, 'trace_g'), color='0.18', lw=2.4)
    geometry.set_yscale('log')
    geometry.set_ylabel(r'$\operatorname{tr}(FF^\top)$')
    geometry.set_title('Control geometry changes')
    geometry_rank = geometry.twinx()
    geometry_rank.plot(
        x, _array(base, 'effective_rank_g'), color='0.55', lw=1.8,
        ls='--')
    geometry_rank.set_ylabel(r'effective rank of $FF^\top$', color='0.42')
    geometry_rank.tick_params(axis='y', colors='0.42')

    prediction_specs = (
        (axes[0, 1], 'gen_error_normalized', r'$E_{\rm gen}/S$', 'log'),
        (axes[0, 2], 'r2_gen', r'$R^2_{\rm gen}$', 'linear'),
    )
    accentuation_specs = (
        (axes[1, 0], 'acc_error_normalized', r'$E_{\rm acc}/S$', 'log'),
        (axes[1, 1], 'r2_acc', r'$R^2_{\rm acc}$', 'symlog'),
        (axes[1, 2], 'slope_acc', r'slope$_{\rm acc}$', 'log'),
    )

    for color, ratio in zip(colors, ratios):
        selected = _rows_for_ratio(rows, ratio)
        for ax, metric, ylabel, scale in prediction_specs:
            de = _array(selected, f'de_{metric}')
            mc = _array(selected, f'mc_{metric}')
            se = _array(selected, f'mc_{metric}_se')
            ax.plot(x, de, color=color, lw=2.0)
            ax.fill_between(x, mc - 2 * se, mc + 2 * se,
                            color=color, alpha=0.10, linewidth=0)
            markers = np.arange(0, len(x), 5)
            ax.plot(x[markers], mc[markers], 'o', color=color, ms=3.8,
                    mec='white', mew=0.45,
                    label=rf'$\sigma^2/S={ratio:g}$')
            ax.set_ylabel(ylabel)
            ax.set_yscale(scale)

        for ax, metric, ylabel, scale in accentuation_specs:
            de = _array(selected, f'de_{metric}')
            gaussian_key = (
                'gaussian_r2_acc_median' if metric == 'r2_acc'
                else f'gaussian_{metric}')
            mc_key = (
                'mc_r2_acc_median' if metric == 'r2_acc'
                else f'mc_{metric}')
            gaussian = _array(selected, gaussian_key)
            mc = _array(selected, mc_key)
            ax.plot(x, de, color=color, lw=1.4, ls=':', alpha=0.9)
            ax.plot(x, gaussian, color=color, lw=2.2)
            markers = np.arange(0, len(x), 5)
            ax.plot(x[markers], mc[markers], 'o', color=color, ms=3.8,
                    mec='black', mew=0.35)
            ax.set_ylabel(ylabel)
            if scale == 'symlog':
                ax.set_yscale('symlog', linthresh=1.0)
                ax.axhline(0.0, color='0.55', lw=0.7, ls='--')
            else:
                ax.set_yscale(scale)
            if metric == 'slope_acc':
                ax.axhline(1.0, color='0.55', lw=0.7, ls='--')

    axes[0, 1].set_title('Prediction error is invariant')
    axes[0, 2].set_title('Prediction $R^2$ is invariant')
    axes[1, 0].set_title('Accentuation error changes')
    axes[1, 1].set_title('Accentuation $R^2$ can collapse')
    axes[1, 2].set_title('Accentuation slope shrinks')
    axes[0, 1].legend(fontsize=8, ncol=2, loc='best')
    prediction_handles = [
        plt.Line2D([0], [0], color='0.25', lw=2.0,
                   label='DE prediction'),
        plt.Line2D([0], [0], marker='o', color='0.25', lw=0, ms=4,
                   label=r'MC mean $\pm,2$ SE'),
    ]
    axes[0, 2].legend(
        handles=prediction_handles, fontsize=7.8, loc='lower left')

    method_handles = [
        plt.Line2D([0], [0], color='0.2', lw=1.6, ls=':',
                   label='leading DE (ratio of moments)'),
        plt.Line2D([0], [0], color='0.2', lw=2.2,
                   label='Gaussian distributional DE'),
        plt.Line2D([0], [0], marker='o', color='0.2', lw=0, ms=4,
                   label='500-run MC per-fit statistic'),
    ]
    axes[1, 0].legend(handles=method_handles, fontsize=7.8, loc='best')
    for ax in axes.ravel():
        _style_axis(ax)
    for ax in axes[1]:
        ax.set_xlabel(r'low-variance fraction $\sin^2\theta$')
    geometry.set_xlabel(r'low-variance fraction $\sin^2\theta$')

    fig.suptitle(
        'Van Hateren disk teacher: prediction invariance, control dissociation',
        fontsize=15)
    fig.text(
        0.5, 0.012,
        r'All rotations keep $F\Sigma F^\top=I$, '
        r'$F\Sigma\beta^\star=\sqrt{S}e_1$, and $F\beta^\star$ fixed; '
        r'only $FF^\top$ changes.',
        ha='center', fontsize=9, color='0.28')
    fig.tight_layout(rect=(0, 0.045, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches='tight')
    path.chmod(0o644)
    plt.close(fig)


def plot_gap_diagnostic(
        rows: list[dict[str, object]], path: Path, ratio: float = 0.01) -> None:
    """Show that denominator non-concentration explains the leading-DE gap."""
    selected = _rows_for_ratio(rows, ratio)
    x = _array(selected, 'low_fraction')
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.2), sharex=True)

    geometry = axes[0, 0]
    geometry.plot(
        x, _array(selected, 'trace_g'), color='0.15', lw=2.2,
        label=r'$\operatorname{tr}(FF^\top)$')
    geometry.set_yscale('log')
    geometry.set_ylabel(r'$\operatorname{tr}(FF^\top)$')
    geometry.set_title('A single low-variance mode dominates')
    concentration = geometry.twinx()
    concentration.plot(
        x, _array(selected, 'gaussian_denominator_cv'), color='C3', lw=2.0,
        label='Gaussian DE')
    concentration.plot(
        x, _array(selected, 'mc_denominator_cv'), color='C3', lw=0,
        marker='o', ms=3.4, markevery=4, alpha=0.75, label='MC')
    concentration.axhline(1.0, color='C3', lw=0.8, ls='--')
    concentration.set_ylabel(r'$\operatorname{SD}(D)/\mathbb{E}D$', color='C3')
    concentration.tick_params(axis='y', colors='C3')
    geometry.text(
        0.04, 0.06,
        'effective rank\n'
        f"{_array(selected, 'effective_rank_g')[0]:.1f}"
        r'$\;\longrightarrow\;$'
        f"{_array(selected, 'effective_rank_g')[-1]:.2f}",
        transform=geometry.transAxes, fontsize=9,
        bbox={'facecolor': 'white', 'edgecolor': '0.8', 'alpha': 0.85})

    specs = (
        (axes[0, 1], 'slope_acc', r'slope$_{\rm acc}$', 'linear'),
        (axes[1, 0], 'acc_error_normalized', r'$E_{\rm acc}/S$', 'linear'),
        (axes[1, 1], 'r2_acc', r'$R^2_{\rm acc}$', 'symlog'),
    )
    for ax, metric, ylabel, scale in specs:
        is_r2 = metric == 'r2_acc'
        de = _array(selected, f'de_{metric}')
        moments = _array(selected, f'mc_moment_{metric}')
        gaussian = _array(
            selected,
            'gaussian_r2_acc_median' if is_r2 else f'gaussian_{metric}')
        mc = _array(
            selected, 'mc_r2_acc_median' if is_r2 else f'mc_{metric}')
        ax.plot(x, de, color='0.2', lw=1.8, ls=':',
                label='leading DE')
        ax.plot(x, moments, color='0.55', lw=0, marker='x', ms=4.0,
                markevery=3, label='MC ratio of moments')
        ax.plot(x, gaussian, color='C0', lw=2.3,
                label='Gaussian distributional DE')
        if is_r2:
            low = _array(selected, 'mc_r2_acc_median_ci_low')
            high = _array(selected, 'mc_r2_acc_median_ci_high')
        else:
            se = _array(selected, f'mc_{metric}_se')
            low, high = mc - 2 * se, mc + 2 * se
        ax.fill_between(x, low, high, color='C3', alpha=0.13, linewidth=0)
        ax.plot(x, mc, color='C3', lw=0, marker='o', ms=3.8,
                markevery=3, label='500-run MC per-fit statistic')
        ax.set_ylabel(ylabel)
        ax.set_title({
            'slope_acc': r'$\mathbb{E}[N/D]\ne\mathbb{E}N/\mathbb{E}D$',
            'acc_error_normalized': 'Expected nonlinear control error',
            'r2_acc': 'Median nonlinear control $R^2$',
        }[metric])
        if scale == 'symlog':
            ax.set_yscale('symlog', linthresh=1.0)
            ax.axhline(0.0, color='0.6', lw=0.7, ls='--')
        _style_axis(ax)

    axes[0, 1].legend(fontsize=7.5, loc='best')
    for ax in axes[1]:
        ax.set_xlabel(r'low-variance fraction $\sin^2\theta$')
    geometry.set_xlabel(r'low-variance fraction $\sin^2\theta$')
    fig.suptitle(
        rf'Why leading DE and per-fit MC differ ($\sigma^2/S={ratio:g}$)',
        fontsize=15)
    fig.text(
        0.5, 0.012,
        r'The leading DE correctly tracks the MC ratio of moments.  When '
        r'$D=\hat w^\top FF^\top\hat w$ does not concentrate, the per-fit '
        'nonlinear statistic requires distributional propagation.',
        ha='center', fontsize=9, color='0.28')
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches='tight')
    path.chmod(0o644)
    plt.close(fig)


def report_endpoint(rows: list[dict[str, object]], logger: logging.Logger) -> None:
    """Log compact endpoint comparisons and distributional gap closure."""
    for ratio in sorted({float(row['noise_signal_ratio']) for row in rows}):
        endpoint = _rows_for_ratio(rows, ratio)[-1]
        logger.info(
            'Endpoint sigma^2/S=%g: effective rank %.3f, CV(D) DE/MC '
            '%.3f/%.3f', ratio, endpoint['effective_rank_g'],
            endpoint['gaussian_denominator_cv'], endpoint['mc_denominator_cv'])
        for metric, mc_key, gaussian_key in (
                ('slope_acc', 'mc_slope_acc', 'gaussian_slope_acc'),
                ('acc_error_normalized', 'mc_acc_error_normalized',
                 'gaussian_acc_error_normalized'),
                ('r2_acc', 'mc_r2_acc_median', 'gaussian_r2_acc_median')):
            leading = float(endpoint[f'de_{metric}'])
            observed = float(endpoint[mc_key])
            propagated = float(endpoint[gaussian_key])
            raw_gap = abs(leading - observed)
            closure = (
                1.0 - abs(propagated - observed) / raw_gap
                if raw_gap > 0 else np.nan)
            logger.info(
                '  %s: leading %.6g, MC %.6g, distributional %.6g, '
                'gap closed %.1f%%', metric, leading, observed, propagated,
                100.0 * closure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-path', type=Path, default=RAW_PATH)
    parser.add_argument('--source-summary-path', type=Path, default=SUMMARY_PATH)
    parser.add_argument('--summary-path', type=Path,
                        default=DIAGNOSTIC_SUMMARY_PATH)
    parser.add_argument('--overview-figure-path', type=Path,
                        default=OVERVIEW_FIGURE_PATH)
    parser.add_argument('--gap-figure-path', type=Path, default=GAP_FIGURE_PATH)
    parser.add_argument('--n-surrogate', type=int, default=200_000)
    parser.add_argument('--bootstrap-reps', type=int, default=1000)
    parser.add_argument('--benchmark-samples', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=20260829)
    parser.add_argument('--plot-only', action='store_true')
    parser.add_argument('--benchmark-only', action='store_true')
    parser.add_argument('--log-file', type=Path, default=DEFAULT_LOG_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = configure_logger(args.log_file)
    if args.plot_only:
        rows = read_rows(args.summary_path)
        plot_overview(rows, args.overview_figure_path)
        plot_gap_diagnostic(rows, args.gap_figure_path)
        logger.info('Replotted diagnostics from %s', args.summary_path)
        return

    raw_path = require_bulk_path(
        args.raw_path, 'Van Hateren null-rotation raw cache')
    source_rows = read_rows(args.source_summary_path)
    n = int(float(source_rows[0]['n']))
    with np.load(raw_path) as raw:
        start = time.perf_counter()
        build_diagnostic_rows(
            raw, n=n, n_surrogate=args.n_surrogate,
            bootstrap_reps=min(100, args.bootstrap_reps), seed=args.seed,
            logger=logger, map_limit=5, ratio_limit=1,
            surrogate_limit=min(args.benchmark_samples, args.n_surrogate))
        pilot = time.perf_counter() - start
        n_maps = len(raw['givens_maps'])
        n_ratios = len(raw['ratios'])
        projected = (
            pilot * (args.n_surrogate / min(
                args.benchmark_samples, args.n_surrogate))
            * (n_maps / min(5, n_maps)) * n_ratios)
        logger.info(
            'Small-scale timing: %d samples, 5 maps, 1 noise in %.2fs; '
            'conservative projected runtime %.1fs (%.1fmin)',
            min(args.benchmark_samples, args.n_surrogate), pilot,
            projected, projected / 60.0)
        if args.benchmark_only:
            return

        start = time.perf_counter()
        rows = build_diagnostic_rows(
            raw, n=n, n_surrogate=args.n_surrogate,
            bootstrap_reps=args.bootstrap_reps, seed=args.seed + 1,
            logger=logger)
        elapsed = time.perf_counter() - start
    logger.info('Full cached-data diagnostic completed in %.1fs', elapsed)
    write_rows(args.summary_path, rows)
    plot_overview(rows, args.overview_figure_path)
    plot_gap_diagnostic(rows, args.gap_figure_path)
    report_endpoint(rows, logger)
    logger.info('Plot-ready diagnostic table: %s', args.summary_path)
    logger.info('Figures: %s, %s', args.overview_figure_path,
                args.gap_figure_path)


if __name__ == '__main__':
    main()

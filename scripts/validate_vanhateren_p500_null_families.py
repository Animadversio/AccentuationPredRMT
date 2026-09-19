"""Continuous and random prediction-null feature families at p=500.

The measured Van Hateren spectrum and disk teacher are represented in
whitened input coordinates.  Each feature row is unit norm and orthogonal to
both b=Sigma^{1/2} beta* and Sigma^{-1}b.  Consequently every map satisfies

    F Sigma F^T = I,
    F Sigma beta* = sqrt(S)e_1,
    F beta* = g_0 sqrt(S)e_1,

so prediction, the DE-CV ridge penalty, and teacher overlap are invariant.
The p=500 feature space contains 300 rotatable rows and 199 fixed rows, plus
the teacher row.  Continuous rotation moves all 300 rows together from a
high-variance band into the extreme low-variance tail.  Random maps use
independent rowwise rotations, random high--low pairings, and a random global
tail propensity; they remain in the same joint null space.

Finite-n Monte Carlo uses the exact real-Wishart/orthogonal-invariance ridge
sampler from ``validate_vanhateren_null_rotation_dimension_scaling.py``.
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
from scripts.analyze_vanhateren_null_rotation_gap import nonlinear_metrics
from scripts.validate_ffhq_disk_teacher import configure_logger
from scripts.validate_vanhateren_null_rotation_dimension_scaling import (
    real_wishart_eigenvalues,
)
from scripts.validate_vanhateren_null_rotations import (
    RAW_PATH as BASE_RAW_PATH,
    common_de_settings,
    read_rows,
    write_rows,
)
from scripts.storage_paths import configured_bulk_path, require_bulk_path


REPO_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = REPO_ROOT / 'tables' / 'vanhateren_p500_null_family_summary.csv'
RAW_PATH = configured_bulk_path('tables/vanhateren_p500_null_family_cases.npz')
CONTINUOUS_FIGURE_PATH = (
    REPO_ROOT / 'figures' / 'null_rotations' /
    'vanhateren_p500_continuous_null_rotation.png')
RANDOM_FIGURE_PATH = (
    REPO_ROOT / 'figures' / 'null_rotations' /
    'vanhateren_p500_random_null_diversity.png')
DEFAULT_LOG_PATH = REPO_ROOT / 'logs' / 'vanhateren_p500_null_families.log'

DEFAULT_RATIOS = np.asarray([0.01, 0.1, 1.0, 10.0])


def localized_joint_null_weights(
        eigenvalues: np.ndarray, beta: np.ndarray, indices: np.ndarray,
        n_rows: int, seed: int) -> tuple[np.ndarray, dict[str, float]]:
    """Construct disjoint three-PC rows null to b and Sigma^{-1}b."""
    s = np.asarray(eigenvalues, dtype=float)
    beta = np.asarray(beta, dtype=float)
    indices = np.asarray(indices, dtype=int)
    if len(indices) < 3 * n_rows:
        raise ValueError('Need at least three disjoint coordinates per null row')
    b = np.sqrt(s) * beta
    c = b / s
    inverse = 1.0 / s
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(indices)
    weights = np.empty(n_rows)
    max_b_residual = 0.0
    max_c_residual = 0.0
    for row in range(n_rows):
        support = permutation[3 * row:3 * row + 3]
        _, _, right = np.linalg.svd(
            np.vstack((b[support], c[support])), full_matrices=True)
        direction = right[-1]
        direction /= np.linalg.norm(direction)
        weights[row] = np.sum(inverse[support] * direction ** 2)
        max_b_residual = max(
            max_b_residual, float(abs(direction @ b[support])))
        max_c_residual = max(
            max_c_residual, float(abs(direction @ c[support])))
    return weights, {
        'max_b_residual': max_b_residual,
        'max_c_residual': max_c_residual,
    }


def stratified_fixed_null_weights(
        eigenvalues: np.ndarray, beta: np.ndarray, n_rows: int, seed: int
        ) -> tuple[np.ndarray, dict[str, float]]:
    """Construct fixed rows using one PC from each of three middle bands."""
    s = np.asarray(eigenvalues, dtype=float)
    beta = np.asarray(beta, dtype=float)
    d = len(s)
    b = np.sqrt(s) * beta
    c = b / s
    inverse = 1.0 / s
    bands = (
        np.arange(int(0.15 * d), int(0.30 * d)),
        np.arange(int(0.30 * d), int(0.45 * d)),
        np.arange(int(0.45 * d), int(0.60 * d)),
    )
    if any(len(band) < n_rows for band in bands):
        raise ValueError('Middle bands are too small for fixed rows')
    rng = np.random.default_rng(seed)
    permutations = [rng.permutation(band) for band in bands]
    weights = np.empty(n_rows)
    max_b_residual = 0.0
    max_c_residual = 0.0
    for row in range(n_rows):
        support = np.asarray([
            permutations[band_index][row] for band_index in range(3)])
        _, _, right = np.linalg.svd(
            np.vstack((b[support], c[support])), full_matrices=True)
        direction = right[-1]
        direction /= np.linalg.norm(direction)
        weights[row] = np.sum(inverse[support] * direction ** 2)
        max_b_residual = max(
            max_b_residual, float(abs(direction @ b[support])))
        max_c_residual = max(
            max_c_residual, float(abs(direction @ c[support])))
    return weights, {
        'max_b_residual': max_b_residual,
        'max_c_residual': max_c_residual,
    }


def build_p500_null_families(
        eigenvalues: np.ndarray, beta: np.ndarray, n_angles: int,
        n_random: int, n_rotating: int, seed: int
        ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, float]]:
    """Return diagonal G weights for continuous and random joint-null maps."""
    p = 500
    n_fixed = p - 1 - n_rotating
    if not 1 <= n_rotating <= 400 or n_fixed < 1:
        raise ValueError('Require 1 <= n_rotating <= 400 and at least one fixed row')
    d = len(eigenvalues)
    high, high_diag = localized_joint_null_weights(
        eigenvalues, beta, np.arange(int(0.002 * d), int(0.15 * d)),
        n_rotating, seed + 1)
    low, low_diag = localized_joint_null_weights(
        eigenvalues, beta, np.arange(int(0.60 * d), d),
        n_rotating, seed + 2)
    fixed, fixed_diag = stratified_fixed_null_weights(
        eigenvalues, beta, n_fixed, seed + 3)
    # Quantile pairing makes the global continuous rotation smooth and avoids
    # arbitrary changes due only to row ordering.
    high = np.sort(high)
    low = np.sort(low)

    s = np.asarray(eigenvalues, dtype=float)
    b = np.sqrt(s) * np.asarray(beta, dtype=float)
    signal = float(b @ b)
    g0 = float(np.sum((b / np.sqrt(signal)) ** 2 / s))
    tail_fraction = np.linspace(0.0, 1.0, n_angles)
    continuous = np.empty((n_angles, p))
    for index, fraction in enumerate(tail_fraction):
        rotating = (1.0 - fraction) * high + fraction * low
        continuous[index] = np.concatenate(([g0], rotating, fixed))

    rng = np.random.default_rng(seed + 4)
    random = np.empty((n_random, p))
    random_mean_tail = np.empty(n_random)
    random_tail_propensity = np.empty(n_random)
    random_tail_sd = np.empty(n_random)
    concentration = 12.0
    for map_index in range(n_random):
        propensity = rng.random()
        row_fraction = rng.beta(
            1.0 + concentration * propensity,
            1.0 + concentration * (1.0 - propensity),
            size=n_rotating)
        pairing = rng.permutation(n_rotating)
        rotating = (
            (1.0 - row_fraction) * high + row_fraction * low[pairing])
        random[map_index] = np.concatenate(([g0], rotating, fixed))
        random_mean_tail[map_index] = np.mean(row_fraction)
        random_tail_propensity[map_index] = propensity
        random_tail_sd[map_index] = np.std(row_fraction)

    family_metadata = {
        'continuous_tail_fraction': tail_fraction,
        'random_mean_tail_fraction': random_mean_tail,
        'random_tail_propensity': random_tail_propensity,
        'random_tail_fraction_sd': random_tail_sd,
    }
    diagnostics = {
        'g0': g0,
        'signal': signal,
        'n_rotating': float(n_rotating),
        'n_fixed': float(n_fixed),
        'high_weight_min': float(np.min(high)),
        'high_weight_median': float(np.median(high)),
        'high_weight_max': float(np.max(high)),
        'low_weight_min': float(np.min(low)),
        'low_weight_median': float(np.median(low)),
        'low_weight_max': float(np.max(low)),
        'fixed_weight_min': float(np.min(fixed)),
        'fixed_weight_median': float(np.median(fixed)),
        'fixed_weight_max': float(np.max(fixed)),
        'max_b_null_residual': max(
            high_diag['max_b_residual'], low_diag['max_b_residual'],
            fixed_diag['max_b_residual']),
        'max_c_null_residual': max(
            high_diag['max_c_residual'], low_diag['max_c_residual'],
            fixed_diag['max_c_residual']),
    }
    return {'continuous': continuous, 'random': random}, family_metadata, diagnostics


def simulate_isotropic_ridge_sufficient(
        p: int, n: int, signal: float, settings: list[dict[str, float]],
        n_trials: int, seed: int, progress: bool
        ) -> dict[str, np.ndarray]:
    """Sample exact finite-n ridge sufficient statistics for every G map."""
    n_ratios = len(settings)
    output = {
        'teacher_axis': np.empty((n_ratios, n_trials)),
        'norm_sq': np.empty((n_ratios, n_trials)),
        'null_norm_sq': np.empty((n_ratios, n_trials)),
        'null_direction_sq': np.empty((n_ratios, n_trials, p - 1)),
        'gen_error_normalized': np.empty((n_ratios, n_trials)),
        'r2_gen': np.empty((n_ratios, n_trials)),
        'slope_gen': np.empty((n_ratios, n_trials)),
    }
    teacher_norm = np.sqrt(signal)
    rng = np.random.default_rng(seed)
    iterator = range(n_trials)
    if progress:
        try:
            from tqdm.auto import tqdm
            iterator = tqdm(iterator, desc='exact ridge p=500', unit='fit')
        except ImportError:
            pass
    for trial in iterator:
        sample_spectrum = real_wishart_eigenvalues(n, p, rng)
        teacher_pc = rng.standard_normal(p)
        teacher_pc /= np.linalg.norm(teacher_pc)
        for ratio_index, setting in enumerate(settings):
            ridge_lambda = setting['alpha_cv'] / n
            shrink = sample_spectrum / (sample_spectrum + ridge_lambda)
            noise_pc = (
                setting['sigma'] / np.sqrt(n)
                * np.sqrt(sample_spectrum) / (sample_spectrum + ridge_lambda))
            fitted_pc = (
                teacher_norm * shrink * teacher_pc
                + noise_pc * rng.standard_normal(p))
            teacher_axis = float(teacher_pc @ fitted_pc)
            norm_sq = float(fitted_pc @ fitted_pc)
            null_norm_sq = max(norm_sq - teacher_axis ** 2, 0.0)
            null_direction = rng.standard_normal(p - 1)
            null_direction /= np.linalg.norm(null_direction)
            gen_error = norm_sq - 2.0 * teacher_norm * teacher_axis + signal
            output['teacher_axis'][ratio_index, trial] = teacher_axis
            output['norm_sq'][ratio_index, trial] = norm_sq
            output['null_norm_sq'][ratio_index, trial] = null_norm_sq
            output['null_direction_sq'][ratio_index, trial] = (
                null_direction ** 2)
            output['gen_error_normalized'][ratio_index, trial] = (
                gen_error / signal)
            output['r2_gen'][ratio_index, trial] = 1.0 - gen_error / signal
            output['slope_gen'][ratio_index, trial] = (
                teacher_norm * teacher_axis / norm_sq)
    return output


def evaluate_maps(
        weights: np.ndarray, sufficient: dict[str, np.ndarray], signal: float,
        g0: float) -> dict[str, np.ndarray]:
    """Evaluate all diagonal pixel metrics against shared exact ridge draws."""
    n_maps = len(weights)
    n_ratios, n_trials = sufficient['teacher_axis'].shape
    output = {
        key: np.empty((n_maps, n_ratios, n_trials))
        for key in ('acc_error_normalized', 'r2_acc', 'slope_acc',
                    'acc_numerator', 'acc_denominator')}
    teacher_norm = np.sqrt(signal)
    null_weights = weights[:, 1:]
    for ratio_index in range(n_ratios):
        teacher_axis = sufficient['teacher_axis'][ratio_index]
        null_norm_sq = sufficient['null_norm_sq'][ratio_index]
        quadratic_null = (
            null_weights
            @ sufficient['null_direction_sq'][ratio_index].T)
        numerator = g0 * teacher_norm * teacher_axis
        denominator = (
            g0 * teacher_axis[None, :] ** 2
            + null_norm_sq[None, :] * quadratic_null)
        nonlinear = nonlinear_metrics(numerator[None, :], denominator)
        output['acc_error_normalized'][:, ratio_index] = (
            nonlinear['acc_error_normalized'])
        output['r2_acc'][:, ratio_index] = nonlinear['r2_acc']
        output['slope_acc'][:, ratio_index] = nonlinear['slope_acc']
        output['acc_numerator'][:, ratio_index] = numerator[None, :]
        output['acc_denominator'][:, ratio_index] = denominator
    return output


def theory_for_maps(
        weights: np.ndarray, signal: float, settings: list[dict[str, float]],
        n: int) -> dict[str, np.ndarray]:
    p = weights.shape[1]
    teacher = np.zeros(p)
    teacher[0] = np.sqrt(signal)
    names = ('gen_error_normalized', 'r2_gen', 'slope_gen',
             'acc_error_normalized', 'r2_acc', 'slope_acc',
             'acc_numerator', 'acc_denominator')
    output = {
        name: np.empty((len(weights), len(settings))) for name in names}
    for map_index, g in enumerate(weights):
        for ratio_index, setting in enumerate(settings):
            metrics = spectral_feature_de_metrics(
                np.ones(p), teacher, g, 0.0, signal,
                setting['kappa'], setting['sigma'], n)
            for name in names:
                output[name][map_index, ratio_index] = metrics[name]
    return output


def _summary(values):
    values = np.asarray(values, dtype=float)
    return {
        'mean': float(np.mean(values)),
        'se': float(np.std(values, ddof=1) / np.sqrt(len(values))),
        'median': float(np.median(values)),
        'q25': float(np.quantile(values, 0.25)),
        'q75': float(np.quantile(values, 0.75)),
    }


def build_summary_rows(
        families: dict[str, np.ndarray], family_metadata: dict[str, np.ndarray],
        settings, de_by_family, mc_by_family, sufficient, n: int,
        n_trials: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for family, weights in families.items():
        tail = (
            family_metadata['continuous_tail_fraction']
            if family == 'continuous'
            else family_metadata['random_mean_tail_fraction'])
        for map_index, g in enumerate(weights):
            trace = float(np.sum(g))
            trace2 = float(g @ g)
            for ratio_index, setting in enumerate(settings):
                row: dict[str, object] = {
                    'family': family,
                    'map_index': map_index,
                    'p': weights.shape[1],
                    'n': n,
                    'n_trials': n_trials,
                    'tail_fraction': tail[map_index],
                    'trace_g': trace,
                    'trace_g2': trace2,
                    'effective_rank_g': trace ** 2 / trace2,
                    'max_weight_fraction_g': np.max(g) / trace,
                    'noise_signal_ratio': setting['noise_signal_ratio'],
                    'sigma': setting['sigma'],
                    'de_alpha_cv': setting['alpha_cv'],
                    'de_kappa': setting['kappa'],
                }
                if family == 'random':
                    row['tail_propensity'] = family_metadata[
                        'random_tail_propensity'][map_index]
                    row['tail_fraction_sd'] = family_metadata[
                        'random_tail_fraction_sd'][map_index]
                for name, values in de_by_family[family].items():
                    row[f'de_{name}'] = values[map_index, ratio_index]
                for name in ('gen_error_normalized', 'r2_gen', 'slope_gen'):
                    stats = _summary(sufficient[name][ratio_index])
                    for statistic, value in stats.items():
                        row[f'mc_{name}_{statistic}'] = value
                for name, values in mc_by_family[family].items():
                    stats = _summary(values[map_index, ratio_index])
                    for statistic, value in stats.items():
                        row[f'mc_{name}_{statistic}'] = value
                rows.append(row)
    return rows


def _select(rows, family, ratio):
    return sorted(
        [row for row in rows if row['family'] == family and np.isclose(
            float(row['noise_signal_ratio']), ratio)],
        key=lambda row: float(row['map_index']))


def _values(rows, key):
    return np.asarray([float(row[key]) for row in rows])


def _style(ax):
    ax.grid(alpha=0.18, which='both')
    ax.spines[['top', 'right']].set_alpha(0.35)


def plot_continuous(rows: list[dict[str, object]], path: Path) -> None:
    ratios = sorted({
        float(row['noise_signal_ratio']) for row in rows
        if row['family'] == 'continuous'})
    colors = plt.get_cmap('viridis')(
        np.linspace(0.06, 0.92, len(ratios)))
    base = _select(rows, 'continuous', ratios[0])
    x = _values(base, 'tail_fraction')
    fig, axes = plt.subplots(2, 3, figsize=(15.2, 8.2), sharex=True)

    geometry = axes[0, 0]
    geometry.plot(x, _values(base, 'trace_g'), color='0.18', lw=2.3)
    geometry.set_yscale('log')
    geometry.set_ylabel(r'$\operatorname{tr}(FF^\top)$')
    rank_axis = geometry.twinx()
    rank_axis.plot(
        x, _values(base, 'effective_rank_g'), color='0.55', lw=1.8,
        ls='--')
    rank_axis.set_ylabel('effective rank', color='0.45')
    rank_axis.tick_params(axis='y', colors='0.45')
    geometry.set_title('300 rows rotate into the spectral tail')

    for color, ratio in zip(colors, ratios):
        selected = _select(rows, 'continuous', ratio)
        sigma = float(selected[0]['sigma'])
        noise_label = (
            rf'$\sigma={sigma:.3g}$ '
            rf'($\sigma^2/S={ratio:g}$)')
        for ax, metric, ylabel, scale in (
                (axes[0, 1], 'gen_error_normalized', r'$E_{\rm gen}/S$', 'log'),
                (axes[0, 2], 'r2_gen', r'$R^2_{\rm gen}$', 'linear')):
            de = _values(selected, f'de_{metric}')
            mc = _values(selected, f'mc_{metric}_mean')
            se = _values(selected, f'mc_{metric}_se')
            ax.plot(x, de, color=color, lw=2.0,
                    label=noise_label)
            ax.fill_between(x, mc - 2 * se, mc + 2 * se,
                            color=color, alpha=0.10, linewidth=0)
            ax.plot(x[::5], mc[::5], 'o', color=color, ms=3.7,
                    mec='white', mew=0.4)
            ax.set_ylabel(ylabel)
            ax.set_yscale(scale)
        for ax, metric, ylabel, scale in (
                (axes[1, 0], 'acc_error_normalized', r'$E_{\rm acc}/S$', 'log'),
                (axes[1, 1], 'r2_acc', r'$R^2_{\rm acc}$', 'symlog'),
                (axes[1, 2], 'slope_acc', r'slope$_{\rm acc}$', 'log')):
            de = _values(selected, f'de_{metric}')
            mc = _values(
                selected, f'mc_{metric}_median'
                if metric == 'r2_acc' else f'mc_{metric}_mean')
            ax.plot(x, de, color=color, lw=2.0)
            ax.plot(x[::5], mc[::5], 'o', color=color, ms=3.7,
                    mec='black', mew=0.35)
            ax.set_ylabel(ylabel)
            if scale == 'symlog':
                ax.set_yscale('symlog', linthresh=1.0)
                ax.axhline(0.0, color='0.55', lw=0.7, ls='--')
            else:
                ax.set_yscale(scale)

    axes[0, 1].set_title('Generalization error is unchanged')
    axes[0, 2].set_title('Generalization $R^2$ is unchanged')
    axes[1, 0].set_title('Accentuation error changes')
    axes[1, 1].set_title('Accentuation $R^2$ changes dramatically')
    axes[1, 2].set_title('Accentuation slope changes dramatically')
    axes[0, 1].legend(fontsize=8, ncol=2, loc='best')
    axes[1, 0].legend(handles=[
        plt.Line2D([0], [0], color='0.2', lw=2, label='leading DE'),
        plt.Line2D([0], [0], color='0.2', marker='o', lw=0, ms=4,
                   label='exact finite-$n$ MC'),
    ], fontsize=8, loc='best')
    for ax in axes.ravel():
        _style(ax)
    for ax in axes[1]:
        ax.set_xlabel('shared tail-loading fraction')
    geometry.set_xlabel('shared tail-loading fraction')
    fig.suptitle(
        r'$p=500$: continuous prediction-null rotation with noise sweep',
        fontsize=15)
    fig.text(
        0.5, 0.012,
        r'Every point has the same $F\Sigma F^\top$, '
        r'$F\Sigma\beta^\star$, and $F\beta^\star$; only $FF^\top$ changes.',
        ha='center', fontsize=9, color='0.28')
    fig.tight_layout(rect=(0, 0.045, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches='tight')
    path.chmod(0o644)
    plt.close(fig)


def plot_random(rows: list[dict[str, object]], path: Path) -> None:
    ratios = sorted({
        float(row['noise_signal_ratio']) for row in rows
        if row['family'] == 'random'})
    colors = plt.get_cmap('viridis')(
        np.linspace(0.06, 0.92, len(ratios)))
    base = _select(rows, 'random', ratios[0])
    trace = _values(base, 'trace_g')
    order = np.argsort(trace)
    # The geometry panel uses tail loading on x, whereas every metric panel
    # uses tr(G).  Do not link their x axes.
    fig, axes = plt.subplots(2, 3, figsize=(15.2, 8.2), sharex=False)

    scatter = axes[0, 0].scatter(
        _values(base, 'tail_fraction'), trace,
        c=_values(base, 'effective_rank_g'), cmap='magma', s=22,
        alpha=0.72, edgecolor='none')
    axes[0, 0].set_yscale('log')
    axes[0, 0].set_xlabel('mean random tail loading')
    axes[0, 0].set_ylabel(r'$\operatorname{tr}(FF^\top)$')
    axes[0, 0].set_title('Random joint-null maps span control geometries')
    colorbar = fig.colorbar(scatter, ax=axes[0, 0], pad=0.02)
    colorbar.set_label('effective rank')

    for color, ratio in zip(colors, ratios):
        selected = _select(rows, 'random', ratio)
        sigma = float(selected[0]['sigma'])
        noise_label = (
            rf'$\sigma={sigma:.3g}$ '
            rf'($\sigma^2/S={ratio:g}$)')
        for ax, metric, ylabel in (
                (axes[0, 1], 'gen_error_normalized', r'$E_{\rm gen}/S$'),
                (axes[0, 2], 'r2_gen', r'$R^2_{\rm gen}$')):
            de = _values(selected, f'de_{metric}')[order]
            mc = _values(selected, f'mc_{metric}_mean')[order]
            ax.plot(trace[order], de, color=color, lw=1.7,
                    label=noise_label)
            ax.scatter(trace[order][::8], mc[::8], color=color, s=15,
                       edgecolor='white', linewidth=0.35)
            ax.set_ylabel(ylabel)
        for ax, metric, ylabel, scale in (
                (axes[1, 0], 'acc_error_normalized', r'$E_{\rm acc}/S$', 'log'),
                (axes[1, 1], 'r2_acc', r'$R^2_{\rm acc}$', 'symlog'),
                (axes[1, 2], 'slope_acc', r'slope$_{\rm acc}$', 'log')):
            de = _values(selected, f'de_{metric}')
            mc = _values(
                selected, f'mc_{metric}_median'
                if metric == 'r2_acc' else f'mc_{metric}_mean')
            ax.scatter(trace, de, color=color, s=15, alpha=0.45)
            ax.scatter(trace, mc, facecolors='none', edgecolors=color,
                       s=19, linewidths=0.7, alpha=0.70)
            ax.set_ylabel(ylabel)
            if scale == 'symlog':
                ax.set_yscale('symlog', linthresh=1.0)
                ax.axhline(0.0, color='0.55', lw=0.7, ls='--')
            else:
                ax.set_yscale(scale)

    axes[0, 1].set_title('Generalization error is identical')
    axes[0, 2].set_title('Generalization $R^2$ is identical')
    axes[1, 0].set_title('Diverse accentuation errors')
    axes[1, 1].set_title('Diverse accentuation $R^2$')
    axes[1, 2].set_title('Diverse accentuation slopes')
    axes[0, 1].set_xscale('log')
    axes[0, 2].set_xscale('log')
    axes[0, 1].set_xlabel(r'$\operatorname{tr}(FF^\top)$')
    axes[0, 2].set_xlabel(r'$\operatorname{tr}(FF^\top)$')
    for ax in axes[1]:
        ax.set_xscale('log')
        ax.set_xlabel(r'$\operatorname{tr}(FF^\top)$')
    axes[0, 1].legend(fontsize=8, ncol=2, loc='best')
    axes[1, 0].legend(handles=[
        plt.Line2D([0], [0], marker='o', color='0.25', lw=0, ms=4,
                   label='leading DE'),
        plt.Line2D([0], [0], marker='o', markerfacecolor='none',
                   markeredgecolor='0.25', lw=0, ms=4,
                   label='exact finite-$n$ MC'),
    ], fontsize=8, loc='best')
    for ax in axes.ravel():
        _style(ax)
    fig.suptitle(
        r'$p=500$: random sampling in the prediction-null space', fontsize=15)
    fig.text(
        0.5, 0.012,
        'Each map independently randomizes 300 rowwise high-to-tail rotations '
        'and their pairings; prediction and teacher overlap remain fixed.',
        ha='center', fontsize=9, color='0.28')
    fig.tight_layout(rect=(0, 0.045, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches='tight')
    path.chmod(0o644)
    plt.close(fig)


def render(rows, continuous_path, random_path):
    plot_continuous(rows, continuous_path)
    plot_random(rows, random_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=1000)
    parser.add_argument('--n-trials', type=int, default=2000)
    parser.add_argument('--n-angles', type=int, default=41)
    parser.add_argument('--n-random', type=int, default=300)
    parser.add_argument('--n-rotating', type=int, default=300)
    parser.add_argument('--ratios', type=float, nargs='+',
                        default=DEFAULT_RATIOS.tolist())
    parser.add_argument('--benchmark-trials', type=int, default=20)
    parser.add_argument('--seed', type=int, default=20260831)
    parser.add_argument('--base-raw-path', type=Path, default=BASE_RAW_PATH)
    parser.add_argument('--summary-path', type=Path, default=SUMMARY_PATH)
    parser.add_argument('--raw-path', type=Path, default=RAW_PATH)
    parser.add_argument('--continuous-figure-path', type=Path,
                        default=CONTINUOUS_FIGURE_PATH)
    parser.add_argument('--random-figure-path', type=Path,
                        default=RANDOM_FIGURE_PATH)
    parser.add_argument('--log-file', type=Path, default=DEFAULT_LOG_PATH)
    parser.add_argument('--plot-only', action='store_true')
    parser.add_argument('--benchmark-only', action='store_true')
    parser.add_argument('--no-progress', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    logger = configure_logger(args.log_file)
    if args.plot_only:
        rows = read_rows(args.summary_path)
        render(rows, args.continuous_figure_path, args.random_figure_path)
        logger.info('Replotted p=500 null families from %s', args.summary_path)
        return

    base_raw_path = require_bulk_path(
        args.base_raw_path, 'base Van Hateren null-rotation raw cache')
    raw_path = require_bulk_path(
        args.raw_path, 'Van Hateren p=500 null-family raw cache')
    ratios = np.asarray(args.ratios, dtype=float)
    with np.load(base_raw_path) as base:
        eigenvalues = np.asarray(base['eigenvalues'], dtype=float)
        beta = np.asarray(base['beta_proj'], dtype=float)
        alphas = np.asarray(base['alphas'], dtype=float)
    families, family_metadata, diagnostics = build_p500_null_families(
        eigenvalues, beta, args.n_angles, args.n_random,
        args.n_rotating, args.seed + 11)
    signal = diagnostics['signal']
    settings = common_de_settings(500, signal, ratios, args.n, alphas)
    logger.info(
        'Joint-null invariance residuals: b %.2e, Sigma^-1 b %.2e; '
        '%d rotating and %d fixed rows',
        diagnostics['max_b_null_residual'],
        diagnostics['max_c_null_residual'],
        int(diagnostics['n_rotating']), int(diagnostics['n_fixed']))
    logger.info(
        'G weights high %.3g/%.3g/%.3g, tail %.3g/%.3g/%.3g '
        '(min/median/max)',
        diagnostics['high_weight_min'], diagnostics['high_weight_median'],
        diagnostics['high_weight_max'], diagnostics['low_weight_min'],
        diagnostics['low_weight_median'], diagnostics['low_weight_max'])

    de_by_family = {
        family: theory_for_maps(weights, signal, settings, args.n)
        for family, weights in families.items()}
    start = time.perf_counter()
    simulate_isotropic_ridge_sufficient(
        500, args.n, signal, settings, args.benchmark_trials,
        args.seed + 101, progress=False)
    pilot = time.perf_counter() - start
    projected = pilot * args.n_trials / args.benchmark_trials
    logger.info(
        'Small-scale timing: %d p=500 exact ridge fits in %.2fs; '
        'projected %.1fs (%.1fmin) for %d trials and %d noises',
        args.benchmark_trials, pilot, projected, projected / 60.0,
        args.n_trials, len(ratios))
    if args.benchmark_only:
        return

    start = time.perf_counter()
    sufficient = simulate_isotropic_ridge_sufficient(
        500, args.n, signal, settings, args.n_trials,
        args.seed + 201, progress=not args.no_progress)
    mc_by_family = {
        family: evaluate_maps(
            weights, sufficient, signal, diagnostics['g0'])
        for family, weights in families.items()}
    elapsed = time.perf_counter() - start
    logger.info('Exact ridge fits plus all-map evaluation completed in %.1fs', elapsed)

    rows = build_summary_rows(
        families, family_metadata, settings, de_by_family, mc_by_family,
        sufficient, args.n, args.n_trials)
    write_rows(args.summary_path, rows)
    raw_values: dict[str, np.ndarray | float] = {
        'ratios': ratios,
        'signal': signal,
        **{f'diagnostics_{key}': value for key, value in diagnostics.items()},
        **{f'metadata_{key}': value for key, value in family_metadata.items()},
        **{f'{family}_weights': value for family, value in families.items()},
        **{f'ridge_{key}': value for key, value in sufficient.items()},
        **{
            f'{family}_mc_{key}': value
            for family, metrics in mc_by_family.items()
            for key, value in metrics.items()},
        **{
            f'{family}_de_{key}': value
            for family, metrics in de_by_family.items()
            for key, value in metrics.items()},
    }
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(raw_path, **raw_values)
    raw_path.chmod(0o644)
    render(rows, args.continuous_figure_path, args.random_figure_path)

    for family in ('continuous', 'random'):
        selected = [
            row for row in rows if row['family'] == family
            and np.isclose(float(row['noise_signal_ratio']), ratios[0])]
        logger.info(
            '%s family: trace(G) %.3g--%.3g, effective rank %.1f--%.1f, '
            'Egen span DE/MC %.2e/%.2e',
            family, min(float(row['trace_g']) for row in selected),
            max(float(row['trace_g']) for row in selected),
            min(float(row['effective_rank_g']) for row in selected),
            max(float(row['effective_rank_g']) for row in selected),
            np.ptp([float(row['de_gen_error_normalized']) for row in selected]),
            np.ptp([float(row['mc_gen_error_normalized_mean']) for row in selected]))
    logger.info('Plot-ready summary: %s', args.summary_path)
    logger.info('Raw trial/map cache: %s', raw_path)
    logger.info('Figures: %s, %s', args.continuous_figure_path,
                args.random_figure_path)


if __name__ == '__main__':
    main()

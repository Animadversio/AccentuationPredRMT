"""Scale the Van Hateren prediction-null rotation from p=16 to p=1000.

The p=16 control metric is exactly the metric from the original Givens
experiment.  Larger feature spaces append localized null directions whose
supports are disjoint from the original high-, middle-, and low-rank bands.
Every appended direction is orthogonal in whitened coordinates to both

    b = Sigma^{1/2} beta*  and  Sigma^{-1} b,

so C=I, the realizable teacher, q=F beta*, and the entire prediction/RidgeCV
problem remain unchanged within each p.  Only G=F F^T changes along the
Givens rotation.

Finite-n ridge Monte Carlo is exact for the isotropic feature model but avoids
dense p-by-p solves.  Real-Wishart eigenvalues are sampled from the
Dumitriu--Edelman bidiagonal model; orthogonal invariance then reduces a fit to
its teacher-axis coefficient, null-space norm, and a uniform null direction.
The selected penalty is the fixed DE-CV optimum for each (p, noise) setting.
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
from scipy.linalg import eigh_tridiagonal

from rmt_core import spectral_feature_de_metrics
from scripts.analyze_vanhateren_null_rotation_gap import (
    bootstrap_median_interval,
    nonlinear_metrics,
    ratio_of_moments_metrics,
)
from scripts.validate_ffhq_disk_teacher import configure_logger
from scripts.validate_vanhateren_null_rotations import (
    RAW_PATH as BASE_RAW_PATH,
    common_de_settings,
    read_rows,
    write_rows,
)
from scripts.storage_paths import configured_bulk_path, require_bulk_path


REPO_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = (
    REPO_ROOT / 'tables' / 'vanhateren_null_rotation_dimension_summary.csv')
RAW_PATH = configured_bulk_path(
    'tables/vanhateren_null_rotation_dimension_cases.npz')
CURVE_FIGURE_PATH = (
    REPO_ROOT / 'figures' / 'null_rotations' /
    'vanhateren_null_dimension_curves.png')
ADEQUACY_FIGURE_PATH = (
    REPO_ROOT / 'figures' / 'null_rotations' /
    'vanhateren_null_leading_de_adequacy.png')
DEFAULT_LOG_PATH = (
    REPO_ROOT / 'logs' / 'vanhateren_null_rotation_dimension.log')

DEFAULT_DIMENSIONS = (16, 128, 500, 1000)
DEFAULT_RATIOS = (0.01, 0.1)


def real_wishart_eigenvalues(
        n: int, p: int, rng: np.random.Generator) -> np.ndarray:
    """Sample eigenvalues of X^T X/n for real iid-Gaussian X, n >= p.

    This is the beta=1 Laguerre bidiagonal model.  Only a symmetric
    tridiagonal eigensolve is required.
    """
    if not 1 <= p <= n:
        raise ValueError('The current exact sampler requires 1 <= p <= n')
    diagonal_bidiagonal = np.sqrt(rng.chisquare(
        np.arange(n, n - p, -1, dtype=float)))
    subdiagonal_bidiagonal = np.sqrt(rng.chisquare(
        np.arange(p - 1, 0, -1, dtype=float)))
    diagonal = diagonal_bidiagonal ** 2
    diagonal[1:] += subdiagonal_bidiagonal ** 2
    off_diagonal = (
        diagonal_bidiagonal[:-1] * subdiagonal_bidiagonal)
    return eigh_tridiagonal(
        diagonal, off_diagonal, eigvals_only=True,
        check_finite=False) / n


def isotropic_ridge_axis_draw(
        eigenvalues: np.ndarray, teacher_norm: float,
        sigma_over_sqrt_n: float,
        ridge_lambda: float, rng: np.random.Generator
        ) -> tuple[float, float]:
    """Draw the teacher-axis coefficient and squared norm of an exact fit.

    ``sigma_over_sqrt_n`` is the response-noise standard deviation divided by
    the square root of the sample size.
    """
    p = len(eigenvalues)
    teacher_eigenbasis = rng.standard_normal(p)
    teacher_eigenbasis /= np.linalg.norm(teacher_eigenbasis)
    noise = rng.standard_normal(p)
    shrink = eigenvalues / (eigenvalues + ridge_lambda)
    noise_pc = sigma_over_sqrt_n * np.sqrt(eigenvalues) / (
        eigenvalues + ridge_lambda)
    coefficient_eigenbasis = (
        teacher_norm * shrink * teacher_eigenbasis + noise_pc * noise)
    teacher_axis = float(teacher_eigenbasis @ coefficient_eigenbasis)
    norm_sq = float(coefficient_eigenbasis @ coefficient_eigenbasis)
    return teacher_axis, norm_sq


def build_nested_control_spectra(
        base_raw: np.lib.npyio.NpzFile, dimensions: tuple[int, ...], seed: int
        ) -> tuple[dict[int, np.ndarray], np.ndarray, dict[str, float]]:
    """Append disjoint, localized prediction-null features to the p=16 map."""
    dimensions = tuple(sorted(dimensions))
    if dimensions[0] < 16:
        raise ValueError('Dimensions must be at least the p=16 base construction')
    eigenvalues = np.asarray(base_raw['eigenvalues'], dtype=float)
    beta = np.asarray(base_raw['beta_proj'], dtype=float)
    base_maps = np.asarray(base_raw['givens_maps'], dtype=float)
    theta = np.asarray(base_raw['plane_theta'], dtype=float)
    d = len(eigenvalues)
    if dimensions[-1] - 16 > min(
            int(0.15 * d), int(0.15 * d), int(0.15 * d)):
        raise ValueError('Input spectrum is too small for disjoint null triples')

    b = np.sqrt(eigenvalues) * beta
    c = b / eigenvalues
    inverse = 1.0 / eigenvalues
    g0 = float(base_maps[0, 0, 0])
    g_high = float(base_maps[0, 1, 1])
    g_low = float(base_maps[-1, 1, 1])
    base_fixed = np.linalg.eigvalsh(base_maps[0, 2:, 2:])

    # The original fixed block is supported on ranks 301--1500.  The three
    # new bands use ranks 1501--6000, while the Givens low direction begins at
    # rank 6001.  Thus all appended rows have mutually disjoint supports and
    # are disjoint from the original construction.
    bands = (
        np.arange(int(0.15 * d), int(0.30 * d)),
        np.arange(int(0.30 * d), int(0.45 * d)),
        np.arange(int(0.45 * d), int(0.60 * d)),
    )
    rng = np.random.default_rng(seed)
    permutations = [rng.permutation(band) for band in bands]
    n_extra = dimensions[-1] - 16
    extra_weights = np.empty(n_extra, dtype=float)
    max_b_residual = 0.0
    max_c_residual = 0.0
    for row in range(n_extra):
        indices = np.asarray([
            permutations[band_index][row] for band_index in range(3)])
        constraints = np.vstack((b[indices], c[indices]))
        _, _, right = np.linalg.svd(constraints, full_matrices=True)
        direction = right[-1]
        direction /= np.linalg.norm(direction)
        max_b_residual = max(
            max_b_residual, float(abs(direction @ b[indices])))
        max_c_residual = max(
            max_c_residual, float(abs(direction @ c[indices])))
        extra_weights[row] = float(np.sum(
            inverse[indices] * direction ** 2))

    fixed_by_dimension: dict[int, np.ndarray] = {}
    for p in dimensions:
        fixed_by_dimension[p] = np.concatenate((
            base_fixed, extra_weights[:p - 16]))
        if len(fixed_by_dimension[p]) != p - 2:
            raise RuntimeError('Incorrect nested fixed-spectrum size')
    metadata = {
        'g0': g0,
        'g_high': g_high,
        'g_low': g_low,
        'max_b_null_residual': max_b_residual,
        'max_c_null_residual': max_c_residual,
        'extra_weight_min': float(np.min(extra_weights)) if n_extra else np.nan,
        'extra_weight_median': (
            float(np.median(extra_weights)) if n_extra else np.nan),
        'extra_weight_max': float(np.max(extra_weights)) if n_extra else np.nan,
    }
    return fixed_by_dimension, theta, metadata


def simulate_exact_fixed_de_cv_ridge(
        p: int, n: int, signal: float, ratios: np.ndarray,
        settings: list[dict[str, float]], g0: float, g_rotation: np.ndarray,
        g_fixed: np.ndarray, n_trials: int, seed: int,
        progress: bool = True) -> dict[str, np.ndarray]:
    """Exact finite-n fixed-alpha ridge MC using orthogonal invariance."""
    n_angles = len(g_rotation)
    n_ratios = len(ratios)
    output = {
        'gen_error_normalized': np.empty((n_ratios, n_trials)),
        'r2_gen': np.empty((n_ratios, n_trials)),
        'slope_gen': np.empty((n_ratios, n_trials)),
        'acc_error_normalized': np.empty((n_angles, n_ratios, n_trials)),
        'r2_acc': np.empty((n_angles, n_ratios, n_trials)),
        'slope_acc': np.empty((n_angles, n_ratios, n_trials)),
        'acc_numerator': np.empty((n_angles, n_ratios, n_trials)),
        'acc_denominator': np.empty((n_angles, n_ratios, n_trials)),
    }
    teacher_norm = np.sqrt(signal)
    rng = np.random.default_rng(seed)
    iterator = range(n_trials)
    if progress:
        try:
            from tqdm.auto import tqdm
            iterator = tqdm(iterator, desc=f'exact ridge p={p}', unit='fit')
        except ImportError:
            pass

    for trial in iterator:
        sample_spectrum = real_wishart_eigenvalues(n, p, rng)
        teacher_eigenbasis = rng.standard_normal(p)
        teacher_eigenbasis /= np.linalg.norm(teacher_eigenbasis)
        for ratio_index, setting in enumerate(settings):
            ridge_lambda = setting['alpha_cv'] / n
            response_noise = setting['sigma']
            noise = rng.standard_normal(p)
            shrink = sample_spectrum / (sample_spectrum + ridge_lambda)
            noise_pc = (
                response_noise / np.sqrt(n)
                * np.sqrt(sample_spectrum) / (sample_spectrum + ridge_lambda))
            coefficient_eigenbasis = (
                teacher_norm * shrink * teacher_eigenbasis
                + noise_pc * noise)
            teacher_axis = float(
                teacher_eigenbasis @ coefficient_eigenbasis)
            norm_sq = float(
                coefficient_eigenbasis @ coefficient_eigenbasis)
            null_norm_sq = max(norm_sq - teacher_axis ** 2, 0.0)

            gen_error = (
                norm_sq - 2.0 * teacher_norm * teacher_axis + signal)
            gen_numerator = teacher_norm * teacher_axis
            output['gen_error_normalized'][ratio_index, trial] = (
                gen_error / signal)
            output['r2_gen'][ratio_index, trial] = 1.0 - gen_error / signal
            output['slope_gen'][ratio_index, trial] = (
                gen_numerator / norm_sq)

            null_direction = rng.standard_normal(p - 1)
            null_direction /= np.linalg.norm(null_direction)
            rotating_energy = null_direction[0] ** 2
            fixed_energy = float(
                g_fixed @ (null_direction[1:] ** 2))
            numerator = g0 * teacher_norm * teacher_axis
            denominator = (
                g0 * teacher_axis ** 2
                + null_norm_sq * (
                    g_rotation * rotating_energy + fixed_energy))
            nonlinear = nonlinear_metrics(
                np.full(n_angles, numerator), denominator)
            output['acc_error_normalized'][:, ratio_index, trial] = (
                nonlinear['acc_error_normalized'])
            output['r2_acc'][:, ratio_index, trial] = nonlinear['r2_acc']
            output['slope_acc'][:, ratio_index, trial] = (
                nonlinear['slope_acc'])
            output['acc_numerator'][:, ratio_index, trial] = numerator
            output['acc_denominator'][:, ratio_index, trial] = denominator
    return output


def gaussian_distributional_summaries(
        signal: float, mean0: float, variance: float, g0: float,
        g_rotation: np.ndarray, g_fixed: np.ndarray, n_samples: int,
        seed: int, sample_chunk: int = 5000
        ) -> dict[str, np.ndarray]:
    """Propagate the Gaussian DE through all nonlinear accentuation ratios."""
    rng = np.random.default_rng(seed)
    teacher_norm = np.sqrt(signal)
    teacher_axis = mean0 + np.sqrt(variance) * rng.standard_normal(n_samples)
    rotating_standard = rng.standard_normal(n_samples)
    fixed_quadratic = np.zeros(n_samples)
    for start in range(0, n_samples, sample_chunk):
        stop = min(start + sample_chunk, n_samples)
        fixed_standard = rng.standard_normal((stop - start, len(g_fixed)))
        fixed_quadratic[start:stop] = (
            variance * (fixed_standard ** 2 @ g_fixed))
    numerator = g0 * teacher_norm * teacher_axis
    base_denominator = g0 * teacher_axis ** 2 + fixed_quadratic
    result = {
        'slope_acc': np.empty(len(g_rotation)),
        'acc_error_normalized': np.empty(len(g_rotation)),
        'r2_acc_median': np.empty(len(g_rotation)),
        'r2_acc_q25': np.empty(len(g_rotation)),
        'r2_acc_q75': np.empty(len(g_rotation)),
    }
    for index, weight in enumerate(g_rotation):
        denominator = (
            base_denominator
            + variance * weight * rotating_standard ** 2)
        nonlinear = nonlinear_metrics(numerator, denominator)
        result['slope_acc'][index] = np.mean(nonlinear['slope_acc'])
        result['acc_error_normalized'][index] = np.mean(
            nonlinear['acc_error_normalized'])
        result['r2_acc_median'][index] = np.median(nonlinear['r2_acc'])
        result['r2_acc_q25'][index] = np.quantile(
            nonlinear['r2_acc'], 0.25)
        result['r2_acc_q75'][index] = np.quantile(
            nonlinear['r2_acc'], 0.75)
    return result


def _mean_se(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    return (
        float(np.mean(values)),
        float(np.std(values, ddof=1) / np.sqrt(len(values))))


def build_summary_rows(
        dimensions: tuple[int, ...], ratios: np.ndarray, theta: np.ndarray,
        signal: float, n: int, fixed_by_dimension: dict[int, np.ndarray],
        metadata: dict[str, float], settings_by_dimension,
        de_by_dimension, gaussian_by_dimension, mc_by_dimension,
        bootstrap_reps: int, seed: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(seed)
    n_trials = next(iter(mc_by_dimension.values()))['r2_gen'].shape[1]
    bootstrap_indices = rng.integers(
        0, n_trials, size=(bootstrap_reps, n_trials))
    g_rotation = (
        metadata['g_high'] * np.cos(theta) ** 2
        + metadata['g_low'] * np.sin(theta) ** 2)
    for p in dimensions:
        g_fixed = fixed_by_dimension[p]
        for ratio_index, ratio in enumerate(ratios):
            gen_error_mean, gen_error_se = _mean_se(
                mc_by_dimension[p]['gen_error_normalized'][ratio_index])
            r2_gen_mean, r2_gen_se = _mean_se(
                mc_by_dimension[p]['r2_gen'][ratio_index])
            for angle_index, angle in enumerate(theta):
                g = np.concatenate((
                    [metadata['g0'], g_rotation[angle_index]], g_fixed))
                denominator = mc_by_dimension[p][
                    'acc_denominator'][angle_index, ratio_index]
                numerator = mc_by_dimension[p][
                    'acc_numerator'][angle_index, ratio_index]
                mc_error = mc_by_dimension[p][
                    'acc_error_normalized'][angle_index, ratio_index]
                mc_slope = mc_by_dimension[p][
                    'slope_acc'][angle_index, ratio_index]
                mc_r2 = mc_by_dimension[p][
                    'r2_acc'][angle_index, ratio_index]
                error_mean, error_se = _mean_se(mc_error)
                slope_mean, slope_se = _mean_se(mc_slope)
                ratio_metrics = ratio_of_moments_metrics(
                    float(np.mean(numerator)), float(np.mean(denominator)))
                if angle_index == len(theta) - 1:
                    r2_ci_low, r2_ci_high = bootstrap_median_interval(
                        mc_r2, bootstrap_indices)
                else:
                    r2_ci_low = r2_ci_high = np.nan
                rows.append({
                    'p': p,
                    'n': n,
                    'gamma': p / n,
                    'n_trials': n_trials,
                    'n_surrogate': gaussian_by_dimension[p][ratio_index][
                        'n_samples'],
                    'map_index': angle_index,
                    'theta': angle,
                    'low_fraction': np.sin(angle) ** 2,
                    'noise_signal_ratio': ratio,
                    'sigma': settings_by_dimension[p][ratio_index]['sigma'],
                    'de_alpha_cv': settings_by_dimension[p][ratio_index][
                        'alpha_cv'],
                    'de_kappa': settings_by_dimension[p][ratio_index]['kappa'],
                    'trace_g': np.sum(g),
                    'trace_g2': g @ g,
                    'effective_rank_g': np.sum(g) ** 2 / (g @ g),
                    'max_weight_fraction_g': np.max(g) / np.sum(g),
                    'de_denominator_cv': de_by_dimension[p][ratio_index][
                        'denominator_cv'][angle_index],
                    'mc_denominator_cv': (
                        np.std(denominator, ddof=1) / np.mean(denominator)),
                    'de_gen_error_normalized': de_by_dimension[p][ratio_index][
                        'gen_error_normalized'][angle_index],
                    'de_r2_gen': de_by_dimension[p][ratio_index][
                        'r2_gen'][angle_index],
                    'mc_gen_error_normalized': gen_error_mean,
                    'mc_gen_error_normalized_se': gen_error_se,
                    'mc_r2_gen': r2_gen_mean,
                    'mc_r2_gen_se': r2_gen_se,
                    'de_acc_error_normalized': de_by_dimension[p][ratio_index][
                        'acc_error_normalized'][angle_index],
                    'de_r2_acc': de_by_dimension[p][ratio_index][
                        'r2_acc'][angle_index],
                    'de_slope_acc': de_by_dimension[p][ratio_index][
                        'slope_acc'][angle_index],
                    'gaussian_acc_error_normalized': (
                        gaussian_by_dimension[p][ratio_index][
                            'acc_error_normalized'][angle_index]),
                    'gaussian_r2_acc_median': gaussian_by_dimension[p][
                        ratio_index]['r2_acc_median'][angle_index],
                    'gaussian_slope_acc': gaussian_by_dimension[p][ratio_index][
                        'slope_acc'][angle_index],
                    'mc_acc_error_normalized': error_mean,
                    'mc_acc_error_normalized_se': error_se,
                    'mc_r2_acc_median': np.median(mc_r2),
                    'mc_r2_acc_q25': np.quantile(mc_r2, 0.25),
                    'mc_r2_acc_q75': np.quantile(mc_r2, 0.75),
                    'mc_r2_acc_median_ci_low': r2_ci_low,
                    'mc_r2_acc_median_ci_high': r2_ci_high,
                    'mc_slope_acc': slope_mean,
                    'mc_slope_acc_se': slope_se,
                    'mc_moment_acc_error_normalized': ratio_metrics[
                        'acc_error_normalized'],
                    'mc_moment_r2_acc': ratio_metrics['r2_acc'],
                    'mc_moment_slope_acc': ratio_metrics['slope_acc'],
                })
    return rows


def compute_theory(
        dimensions: tuple[int, ...], ratios: np.ndarray, theta: np.ndarray,
        signal: float, n: int, alphas: np.ndarray,
        fixed_by_dimension: dict[int, np.ndarray], metadata: dict[str, float],
        n_surrogate: int, seed: int, logger: logging.Logger):
    settings_by_dimension = {}
    de_by_dimension = {}
    gaussian_by_dimension = {}
    g_rotation = (
        metadata['g_high'] * np.cos(theta) ** 2
        + metadata['g_low'] * np.sin(theta) ** 2)
    for dimension_index, p in enumerate(dimensions):
        teacher = np.zeros(p)
        teacher[0] = np.sqrt(signal)
        settings = common_de_settings(p, signal, ratios, n, alphas)
        settings_by_dimension[p] = settings
        de_by_dimension[p] = []
        gaussian_by_dimension[p] = []
        for ratio_index, setting in enumerate(settings):
            result = {
                key: np.empty(len(theta)) for key in (
                    'gen_error_normalized', 'r2_gen',
                    'acc_error_normalized', 'r2_acc', 'slope_acc',
                    'denominator_cv')}
            base = spectral_feature_de_metrics(
                np.ones(p), teacher, np.ones(p), 0.0, signal,
                setting['kappa'], setting['sigma'], n)
            mean0 = np.sqrt(signal) / (1.0 + setting['kappa'])
            variance = (
                base['variance_scale'] / (1.0 + setting['kappa']) ** 2)
            for angle_index, rotation_weight in enumerate(g_rotation):
                g = np.concatenate((
                    [metadata['g0'], rotation_weight],
                    fixed_by_dimension[p]))
                metrics = spectral_feature_de_metrics(
                    np.ones(p), teacher, g, 0.0, signal,
                    setting['kappa'], setting['sigma'], n)
                for key in ('gen_error_normalized', 'r2_gen',
                            'acc_error_normalized', 'r2_acc', 'slope_acc'):
                    result[key][angle_index] = metrics[key]
                mean_denominator = (
                    metadata['g0'] * mean0 ** 2 + variance * np.sum(g))
                variance_denominator = (
                    2.0 * variance ** 2 * (g @ g)
                    + 4.0 * variance * metadata['g0'] ** 2 * mean0 ** 2)
                result['denominator_cv'][angle_index] = (
                    np.sqrt(variance_denominator) / mean_denominator)
            de_by_dimension[p].append(result)
            surrogate = gaussian_distributional_summaries(
                signal, mean0, variance, metadata['g0'], g_rotation,
                fixed_by_dimension[p], n_surrogate,
                seed + 1000 * dimension_index + ratio_index)
            surrogate['n_samples'] = n_surrogate
            gaussian_by_dimension[p].append(surrogate)
            logger.info(
                'Theory p=%d, sigma^2/S=%g: alpha_DE=%.5g, kappa=%.5g, '
                'endpoint rank_eff=%.2f, CV(D)=%.3f',
                p, ratios[ratio_index], setting['alpha_cv'],
                setting['kappa'],
                (metadata['g0'] + g_rotation[-1]
                 + np.sum(fixed_by_dimension[p])) ** 2
                / (metadata['g0'] ** 2 + g_rotation[-1] ** 2
                   + fixed_by_dimension[p] @ fixed_by_dimension[p]),
                result['denominator_cv'][-1])
    return settings_by_dimension, de_by_dimension, gaussian_by_dimension


def _select(rows, p: int, ratio: float):
    return sorted(
        [row for row in rows if int(float(row['p'])) == p and np.isclose(
            float(row['noise_signal_ratio']), ratio)],
        key=lambda row: float(row['map_index']))


def _values(rows, key):
    return np.asarray([float(row[key]) for row in rows])


def _style(ax):
    ax.grid(alpha=0.18, which='both')
    ax.spines[['top', 'right']].set_alpha(0.35)


def plot_dimension_curves(
        rows: list[dict[str, object]], path: Path, ratio: float = 0.01) -> None:
    dimensions = sorted({int(float(row['p'])) for row in rows})
    colors = plt.get_cmap('viridis')(
        np.linspace(0.08, 0.90, len(dimensions)))
    base = _select(rows, dimensions[0], ratio)
    x = _values(base, 'low_fraction')
    fig, axes = plt.subplots(2, 3, figsize=(15.2, 8.2), sharex=True)

    for color, p in zip(colors, dimensions):
        selected = _select(rows, p, ratio)
        axes[0, 0].plot(
            x, _values(selected, 'effective_rank_g'), color=color, lw=2.0,
            label=rf'$p={p}$')
        for ax, metric, ylabel, scale in (
                (axes[0, 1], 'gen_error_normalized', r'$E_{\rm gen}/S$', 'log'),
                (axes[0, 2], 'r2_gen', r'$R^2_{\rm gen}$', 'linear')):
            de = _values(selected, f'de_{metric}')
            mc = _values(selected, f'mc_{metric}')
            se = _values(selected, f'mc_{metric}_se')
            ax.plot(x, de, color=color, lw=2.0)
            ax.fill_between(x, mc - 2 * se, mc + 2 * se,
                            color=color, alpha=0.10, linewidth=0)
            ax.plot(x[::5], mc[::5], 'o', color=color, ms=3.5,
                    mec='white', mew=0.35)
            ax.set_ylabel(ylabel)
            ax.set_yscale(scale)
        for ax, metric, ylabel, scale in (
                (axes[1, 0], 'acc_error_normalized', r'$E_{\rm acc}/S$', 'log'),
                (axes[1, 1], 'r2_acc', r'$R^2_{\rm acc}$', 'symlog'),
                (axes[1, 2], 'slope_acc', r'slope$_{\rm acc}$', 'log')):
            de = _values(selected, f'de_{metric}')
            gaussian = _values(
                selected, 'gaussian_r2_acc_median'
                if metric == 'r2_acc' else f'gaussian_{metric}')
            mc = _values(
                selected, 'mc_r2_acc_median'
                if metric == 'r2_acc' else f'mc_{metric}')
            ax.plot(x, de, color=color, lw=1.4, ls=':')
            ax.plot(x, gaussian, color=color, lw=2.2)
            ax.plot(x[::5], mc[::5], 'o', color=color, ms=3.5,
                    mec='black', mew=0.3)
            ax.set_ylabel(ylabel)
            if scale == 'symlog':
                ax.set_yscale('symlog', linthresh=1.0)
                ax.axhline(0.0, color='0.55', lw=0.7, ls='--')
            else:
                ax.set_yscale(scale)

    axes[0, 0].set_yscale('log')
    axes[0, 0].set_ylabel(r'effective rank of $FF^\top$')
    axes[0, 0].set_title('More features restore concentration')
    axes[0, 0].legend(fontsize=8, ncol=2)
    axes[0, 1].set_title('Prediction error remains invariant')
    axes[0, 2].set_title('Prediction $R^2$ remains invariant')
    axes[1, 0].set_title('Accentuation error')
    axes[1, 1].set_title('Accentuation $R^2$')
    axes[1, 2].set_title('Accentuation slope')
    method_handles = [
        plt.Line2D([0], [0], color='0.2', lw=1.5, ls=':',
                   label='leading DE'),
        plt.Line2D([0], [0], color='0.2', lw=2.2,
                   label='Gaussian distributional DE'),
        plt.Line2D([0], [0], color='0.2', marker='o', lw=0, ms=4,
                   label='exact finite-$n$ ridge MC'),
    ]
    axes[1, 0].legend(handles=method_handles, fontsize=7.7, loc='best')
    for ax in axes.ravel():
        _style(ax)
    for ax in axes[1]:
        ax.set_xlabel(r'low-variance fraction $\sin^2\theta$')
    axes[0, 0].set_xlabel(r'low-variance fraction $\sin^2\theta$')
    fig.suptitle(
        rf'Van Hateren null rotation across feature dimension '
        rf'($n=1000$, $\sigma^2/S={ratio:g}$)', fontsize=15)
    fig.text(
        0.5, 0.012,
        'The p=16 metric is unchanged from the original experiment; larger p '
        'appends disjoint prediction-null features.  MC uses the fixed DE-CV '
        'ridge penalty.', ha='center', fontsize=8.8, color='0.28')
    fig.tight_layout(rect=(0, 0.045, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches='tight')
    path.chmod(0o644)
    plt.close(fig)


def plot_leading_adequacy(rows: list[dict[str, object]], path: Path) -> None:
    dimensions = sorted({int(float(row['p'])) for row in rows})
    ratios = sorted({float(row['noise_signal_ratio']) for row in rows})
    colors = plt.get_cmap('viridis')(
        np.linspace(0.12, 0.82, len(ratios)))
    endpoint = {
        (p, ratio): _select(rows, p, ratio)[-1]
        for p in dimensions for ratio in ratios}
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.2), sharex=True)

    ax = axes[0, 0]
    ranks = np.asarray([
        float(endpoint[(p, ratios[0])]['effective_rank_g'])
        for p in dimensions])
    ax.plot(dimensions, ranks, color='0.18', lw=2.2, marker='o')
    ax.set_yscale('log')
    ax.set_ylabel(r'effective rank of $FF^\top$')
    concentration = ax.twinx()
    for color, ratio in zip(colors, ratios):
        concentration.plot(
            dimensions,
            [float(endpoint[(p, ratio)]['de_denominator_cv'])
             for p in dimensions], color=color, lw=1.8, ls='--', marker='s',
            label=rf'$\sigma^2/S={ratio:g}$')
    concentration.axhline(1.0, color='0.55', lw=0.7, ls=':')
    concentration.set_yscale('log')
    concentration.set_ylabel(r'$\operatorname{SD}(D)/\mathbb{E}D$')
    concentration.legend(fontsize=8, loc='center right')
    ax.set_title('The denominator concentrates as $p$ grows')

    for target_ax, metric, ylabel, scale in (
            (axes[0, 1], 'slope_acc', r'slope$_{\rm acc}$', 'log'),
            (axes[1, 0], 'acc_error_normalized', r'$E_{\rm acc}/S$', 'linear'),
            (axes[1, 1], 'r2_acc', r'$R^2_{\rm acc}$', 'symlog')):
        for color, ratio in zip(colors, ratios):
            selected = [endpoint[(p, ratio)] for p in dimensions]
            de = _values(selected, f'de_{metric}')
            gaussian = _values(
                selected, 'gaussian_r2_acc_median'
                if metric == 'r2_acc' else f'gaussian_{metric}')
            mc = _values(
                selected, 'mc_r2_acc_median'
                if metric == 'r2_acc' else f'mc_{metric}')
            target_ax.plot(dimensions, de, color=color, lw=1.5, ls=':')
            target_ax.plot(dimensions, gaussian, color=color, lw=2.2)
            target_ax.plot(dimensions, mc, color=color, lw=0, marker='o',
                           ms=5, mec='black', mew=0.35)
        target_ax.set_ylabel(ylabel)
        target_ax.set_title({
            'slope_acc': 'Leading slope becomes sufficient',
            'acc_error_normalized': 'Leading error becomes sufficient',
            'r2_acc': 'Leading $R^2$ becomes sufficient',
        }[metric])
        if scale == 'symlog':
            target_ax.set_yscale('symlog', linthresh=1.0)
            target_ax.axhline(0.0, color='0.55', lw=0.7, ls='--')
        else:
            target_ax.set_yscale(scale)
        _style(target_ax)

    axes[0, 1].legend(handles=[
        plt.Line2D([0], [0], color='0.25', lw=1.5, ls=':',
                   label='leading DE'),
        plt.Line2D([0], [0], color='0.25', lw=2.2,
                   label='Gaussian distributional DE'),
        plt.Line2D([0], [0], color='0.25', lw=0, marker='o', ms=5,
                   label='exact finite-$n$ MC'),
    ], fontsize=7.8, loc='best')
    for axis in axes.ravel():
        axis.set_xscale('log')
        axis.set_xticks(dimensions, labels=[str(p) for p in dimensions])
        axis.set_xlabel('feature dimension $p$')
    _style(axes[0, 0])
    fig.suptitle(
        'When does the leading accentuation DE become sufficient?', fontsize=15)
    fig.text(
        0.5, 0.012,
        'Low-variance endpoint of the same prediction-equivalent rotation.  '
        'Agreement improves with effective rank, even while control itself '
        'becomes more inaccurate.', ha='center', fontsize=8.8, color='0.28')
    fig.tight_layout(rect=(0, 0.045, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches='tight')
    path.chmod(0o644)
    plt.close(fig)


def report_endpoints(rows, logger):
    for ratio in sorted({float(row['noise_signal_ratio']) for row in rows}):
        logger.info('Low-variance endpoint, sigma^2/S=%g', ratio)
        for p in sorted({int(float(row['p'])) for row in rows}):
            row = _select(rows, p, ratio)[-1]
            gaps = []
            for metric, mc_key, gaussian_key in (
                    ('slope_acc', 'mc_slope_acc', 'gaussian_slope_acc'),
                    ('acc_error_normalized', 'mc_acc_error_normalized',
                     'gaussian_acc_error_normalized'),
                    ('r2_acc', 'mc_r2_acc_median',
                     'gaussian_r2_acc_median')):
                leading = float(row[f'de_{metric}'])
                mc = float(row[mc_key])
                gaussian = float(row[gaussian_key])
                scale = max(abs(mc), 1e-12)
                gaps.append(
                    f'{metric}: lead/MC/G={leading:.4g}/{mc:.4g}/{gaussian:.4g}, '
                    f'rel-gap={abs(leading-mc)/scale:.2%}')
            logger.info(
                '  p=%d rank_eff=%.2f CV(D)=%.3f; %s', p,
                float(row['effective_rank_g']),
                float(row['de_denominator_cv']), '; '.join(gaps))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dimensions', type=int, nargs='+',
                        default=list(DEFAULT_DIMENSIONS))
    parser.add_argument('--ratios', type=float, nargs='+',
                        default=list(DEFAULT_RATIOS))
    parser.add_argument('--n', type=int, default=1000)
    parser.add_argument('--n-trials', type=int, default=2000)
    parser.add_argument('--n-surrogate', type=int, default=100_000)
    parser.add_argument('--n-angles', type=int, default=41)
    parser.add_argument('--bootstrap-reps', type=int, default=1000)
    parser.add_argument('--benchmark-trials', type=int, default=10)
    parser.add_argument('--seed', type=int, default=20260830)
    parser.add_argument('--base-raw-path', type=Path, default=BASE_RAW_PATH)
    parser.add_argument('--summary-path', type=Path, default=SUMMARY_PATH)
    parser.add_argument('--raw-path', type=Path, default=RAW_PATH)
    parser.add_argument('--curve-figure-path', type=Path,
                        default=CURVE_FIGURE_PATH)
    parser.add_argument('--adequacy-figure-path', type=Path,
                        default=ADEQUACY_FIGURE_PATH)
    parser.add_argument('--log-file', type=Path, default=DEFAULT_LOG_PATH)
    parser.add_argument('--plot-only', action='store_true')
    parser.add_argument('--benchmark-only', action='store_true')
    parser.add_argument('--no-progress', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    logger = configure_logger(args.log_file)
    dimensions = tuple(sorted(set(args.dimensions)))
    ratios = np.asarray(args.ratios, dtype=float)
    if dimensions[-1] > args.n:
        raise ValueError('The exact Wishart sampler currently requires p <= n')
    if args.plot_only:
        rows = read_rows(args.summary_path)
        plot_dimension_curves(rows, args.curve_figure_path)
        plot_leading_adequacy(rows, args.adequacy_figure_path)
        logger.info('Replotted dimension scaling from %s', args.summary_path)
        return

    base_raw_path = require_bulk_path(
        args.base_raw_path, 'base Van Hateren null-rotation raw cache')
    raw_path = require_bulk_path(
        args.raw_path, 'Van Hateren dimension-scaling raw cache')
    with np.load(base_raw_path) as base_raw:
        signal = float(base_raw['signal'])
        alphas = np.asarray(base_raw['alphas'], dtype=float)
        fixed_by_dimension, base_theta, metadata = build_nested_control_spectra(
            base_raw, dimensions, args.seed + 11)
    theta = np.linspace(base_theta[0], base_theta[-1], args.n_angles)
    g_rotation = (
        metadata['g_high'] * np.cos(theta) ** 2
        + metadata['g_low'] * np.sin(theta) ** 2)
    logger.info(
        'Nested feature map: max b-null residual %.2e, c-null residual %.2e; '
        'new G weights %.3g / %.3g / %.3g (min/median/max)',
        metadata['max_b_null_residual'], metadata['max_c_null_residual'],
        metadata['extra_weight_min'], metadata['extra_weight_median'],
        metadata['extra_weight_max'])

    settings_by_dimension, de_by_dimension, gaussian_by_dimension = (
        compute_theory(
            dimensions, ratios, theta, signal, args.n, alphas,
            fixed_by_dimension, metadata, args.n_surrogate,
            args.seed + 101, logger))

    largest = dimensions[-1]
    start = time.perf_counter()
    simulate_exact_fixed_de_cv_ridge(
        largest, args.n, signal, ratios, settings_by_dimension[largest],
        metadata['g0'], g_rotation, fixed_by_dimension[largest],
        args.benchmark_trials, args.seed + 1001, progress=False)
    pilot = time.perf_counter() - start
    projected = pilot / args.benchmark_trials * args.n_trials * sum(
        (p / largest) ** 2 for p in dimensions)
    logger.info(
        'Small-scale timing: p=%d, %d trials in %.2fs; projected exact-MC '
        'runtime %.1fs (%.1fmin) for dimensions %s',
        largest, args.benchmark_trials, pilot, projected, projected / 60.0,
        dimensions)
    if args.benchmark_only:
        return

    mc_by_dimension = {}
    raw_values: dict[str, np.ndarray | float] = {
        'dimensions': np.asarray(dimensions),
        'ratios': ratios,
        'theta': theta,
        'signal': signal,
        **{f'metadata_{key}': value for key, value in metadata.items()},
    }
    start = time.perf_counter()
    for dimension_index, p in enumerate(dimensions):
        mc = simulate_exact_fixed_de_cv_ridge(
            p, args.n, signal, ratios, settings_by_dimension[p],
            metadata['g0'], g_rotation, fixed_by_dimension[p],
            args.n_trials, args.seed + 2001 + dimension_index,
            progress=not args.no_progress)
        mc_by_dimension[p] = mc
        raw_values[f'g_fixed_p{p}'] = fixed_by_dimension[p]
        for key, value in mc.items():
            raw_values[f'p{p}_mc_{key}'] = value
    elapsed = time.perf_counter() - start
    logger.info('Exact finite-n ridge Monte Carlo completed in %.1fs', elapsed)

    rows = build_summary_rows(
        dimensions, ratios, theta, signal, args.n, fixed_by_dimension,
        metadata, settings_by_dimension, de_by_dimension,
        gaussian_by_dimension, mc_by_dimension,
        args.bootstrap_reps, args.seed + 3001)
    write_rows(args.summary_path, rows)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(raw_path, **raw_values)
    raw_path.chmod(0o644)
    plot_dimension_curves(rows, args.curve_figure_path)
    plot_leading_adequacy(rows, args.adequacy_figure_path)
    report_endpoints(rows, logger)
    logger.info('Plot-ready summary: %s', args.summary_path)
    logger.info('Raw trial cache: %s', raw_path)
    logger.info('Figures: %s, %s', args.curve_figure_path,
                args.adequacy_figure_path)


if __name__ == '__main__':
    main()

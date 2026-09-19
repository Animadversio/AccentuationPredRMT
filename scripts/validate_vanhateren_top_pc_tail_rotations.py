"""Rotate an unwhitened top-PC projector into prediction-null tail features.

For p in {100, 500}, the left endpoint is exactly

    F(0) = [e_1^T; ...; e_p^T]

in the Van Hateren population-PC basis: ordinary, unwhitened top-p PCA
features.  Write A = F Sigma^{1/2}, b = Sigma^{1/2} beta*, and
c = Sigma^{-1/2} beta*.  Decompose the top-PC endpoint as

    A_0 = P + R,

where P is the rowwise projection onto span{b,c}.  Construct a tail matrix V
with

    V V^T = R R^T,  V R^T = V P^T = 0,  V b = V c = 0,

and rotate

    A(t) = P + sqrt(1-t) R + sqrt(t) V,  t in [0,1].

Consequently A(t)A(t)^T, A(t)b, and A(t)c are exactly invariant.  Thus the
complete feature-space ridge/CV problem, natural-distribution prediction, and
teacher overlap are unchanged.  Only G(t)=F(t)F(t)^T changes, from G(0)=I to
a low-variance-amplifying pixel backpropagation metric.

The script caches plot-ready summaries separately from the finite-n draws and
renders p=100 and p=500 metric figures plus a feature-spectrum visualization.
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

from rmt_core import SpectrumKappa, select_spectral_feature_de_alpha
from rmt_core.ridge_theory_lib import compute_df2
from scripts.analyze_vanhateren_null_rotation_gap import nonlinear_metrics
from scripts.validate_ffhq_disk_teacher import configure_logger
from scripts.validate_vanhateren_disk_teacher import SPECTRUM_PATH
from scripts.storage_paths import configured_bulk_path, require_bulk_path


REPO_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = (
    REPO_ROOT / 'tables' / 'vanhateren_top_pc_tail_rotation_summary.csv')
RAW_PATH = configured_bulk_path(
    'tables/vanhateren_top_pc_tail_rotation_cases.npz')
FIGURE_TEMPLATE = 'vanhateren_top{p}_to_tail_rotation.png'
FEATURE_FIGURE_PATH = (
    REPO_ROOT / 'figures' / 'null_rotations' /
    'vanhateren_top_pc_tail_feature_spectra.png')
DEFAULT_LOG_PATH = (
    REPO_ROOT / 'logs' / 'vanhateren_top_pc_tail_rotations.log')

DEFAULT_DIMENSIONS = (100, 500)
DEFAULT_RATIOS = np.asarray([0.01, 0.1, 1.0, 10.0])


def constraint_basis(*vectors: np.ndarray) -> np.ndarray:
    """Return an orthonormal basis for the supplied constraint columns."""
    matrix = np.column_stack([
        np.asarray(vector, dtype=float) / np.linalg.norm(vector)
        for vector in vectors])
    basis, triangular = np.linalg.qr(matrix, mode='reduced')
    keep = np.abs(np.diag(triangular)) > 1e-11
    return basis[:, keep]


def symmetric_root(matrix: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(matrix)
    tolerance = 1e-10 * max(float(np.max(values)), 1.0)
    if float(np.min(values)) < -tolerance:
        raise RuntimeError(
            f'Gram complement is not positive semidefinite: {values.min()}')
    return (vectors * np.sqrt(np.maximum(values, 0.0))) @ vectors.T


def _sym(matrix: np.ndarray) -> np.ndarray:
    return matrix + matrix.T


def build_top_pc_tail_family(
        eigenvalues: np.ndarray, beta: np.ndarray, p: int,
        tail_fraction: np.ndarray, low_start_fraction: float, seed: int
        ) -> dict[str, np.ndarray | float | int]:
    """Build the invariant top-p-to-tail family through compact Gram blocks."""
    s = np.asarray(eigenvalues, dtype=float)
    beta = np.asarray(beta, dtype=float)
    tail_fraction = np.asarray(tail_fraction, dtype=float)
    d = len(s)
    low_start = int(np.rint(low_start_fraction * d))
    if not 2 <= p < low_start or d - low_start < p + 2:
        raise ValueError('Need p top PCs and at least p+2 tail coordinates')
    if np.any((tail_fraction < 0) | (tail_fraction > 1)):
        raise ValueError('tail fractions must lie in [0,1]')

    sqrt_s = np.sqrt(s)
    b = sqrt_s * beta
    c_constraint = beta / sqrt_s
    constraints = constraint_basis(b, c_constraint)
    rank = constraints.shape[1]

    # A0 has A0[j,j]=sqrt(s_j).  P=A0 QQ^T is represented by left_factor Q^T.
    left_factor = sqrt_s[:p, None] * constraints[:p]
    covariance = np.diag(s[:p])
    gram_p = left_factor @ left_factor.T
    gram_r = covariance - gram_p
    root_r = symmetric_root(gram_r)

    low_indices = np.arange(low_start, d)
    low_constraints = constraints[low_indices]
    low_null_constraints = constraint_basis(
        b[low_indices], c_constraint[low_indices])
    rng = np.random.default_rng(seed)
    random_columns = rng.standard_normal((len(low_indices), p))
    random_columns -= low_null_constraints @ (
        low_null_constraints.T @ random_columns)
    low_columns, triangular = np.linalg.qr(random_columns, mode='reduced')
    if np.min(np.abs(np.diag(triangular))) < 1e-9:
        raise RuntimeError('Degenerate tail Stiefel draw')
    w_low = low_columns[:, :p].T

    inverse = 1.0 / s
    inverse_low = inverse[low_indices]
    constraint_inverse_gram = (
        constraints.T @ (inverse[:, None] * constraints))
    g_pp = left_factor @ constraint_inverse_gram @ left_factor.T
    p_d_a0 = left_factor @ (
        constraints[:p].T / sqrt_s[:p][None, :])
    g_pr = p_d_a0 - g_pp
    g_rr = np.eye(p) - g_pp - _sym(g_pr)

    b_d_w = low_constraints.T @ (
        inverse_low[:, None] * w_low.T)
    g_pv = left_factor @ b_d_w @ root_r.T
    w_d_w = (w_low * inverse_low[None, :]) @ w_low.T
    g_vv = root_r @ w_d_w @ root_r.T

    # R=-P on the tail support, hence G_RV=-G_PV exactly.
    blocks = {
        'g_pp': g_pp,
        'g_rr': g_rr,
        'g_pr_sym': _sym(g_pr),
        'g_vv': g_vv,
        'g_pv_sym': _sym(g_pv),
    }
    g_matrices = np.empty((len(tail_fraction), p, p), dtype=float)
    trace_g = np.empty(len(tail_fraction))
    trace_g2 = np.empty(len(tail_fraction))
    for index, fraction in enumerate(tail_fraction):
        sine = np.sqrt(fraction)
        cosine = np.sqrt(1.0 - fraction)
        g = (
            g_pp + cosine ** 2 * g_rr + fraction * g_vv
            + cosine * blocks['g_pr_sym']
            + sine * (1.0 - cosine) * blocks['g_pv_sym'])
        g = 0.5 * (g + g.T)
        g_matrices[index] = g
        trace_g[index] = np.trace(g)
        trace_g2[index] = np.sum(g * g)

    signal = float(np.sum(s * beta ** 2))
    represented_signal = float(np.sum(s[:p] * beta[:p] ** 2))
    residual_signal = max(signal - represented_signal, 0.0)
    target_covariance = covariance
    target_h = s[:p] * beta[:p]
    target_q = beta[:p]
    tail_b_residual = float(np.max(np.abs(w_low @ b[low_indices])))
    tail_c_residual = float(np.max(np.abs(
        w_low @ c_constraint[low_indices])))

    return {
        'p': p,
        'd': d,
        'tail_fraction': tail_fraction,
        'low_start': low_start,
        'feature_eigenvalues': s[:p],
        'feature_teacher': beta[:p],
        'teacher_overlap': target_q,
        'signal': signal,
        'represented_signal': represented_signal,
        'residual_signal': residual_signal,
        'constraint_basis': constraints,
        'constraint_rank': rank,
        'left_factor': left_factor,
        'root_r': root_r,
        'w_low': w_low,
        **blocks,
        'g_matrices': g_matrices,
        'trace_g': trace_g,
        'trace_g2': trace_g2,
        'effective_rank_g': trace_g ** 2 / trace_g2,
        'target_covariance': target_covariance,
        'target_h': target_h,
        'target_q': target_q,
        'tail_b_residual': tail_b_residual,
        'tail_c_residual': tail_c_residual,
        'g0_identity_error': float(np.max(np.abs(
            g_matrices[0] - np.eye(p)))),
    }


def feature_rows_pc(
        family: dict[str, np.ndarray | float | int],
        eigenvalues: np.ndarray, row_indices: np.ndarray,
        fractions: np.ndarray) -> np.ndarray:
    """Materialize selected feature rows in the population-PC basis."""
    s = np.asarray(eigenvalues, dtype=float)
    sqrt_s = np.sqrt(s)
    p = int(family['p'])
    constraints = np.asarray(family['constraint_basis'])
    left_factor = np.asarray(family['left_factor'])
    root_r = np.asarray(family['root_r'])
    w_low = np.asarray(family['w_low'])
    low_start = int(family['low_start'])
    rows = np.asarray(row_indices, dtype=int)
    output = np.empty((len(rows), len(fractions), len(s)), dtype=float)
    for output_row, row in enumerate(rows):
        if not 0 <= row < p:
            raise ValueError(f'feature row {row} is outside p={p}')
        projection = left_factor[row] @ constraints.T
        residual = -projection
        residual = residual.copy()
        residual[row] += sqrt_s[row]
        tail = np.zeros(len(s), dtype=float)
        tail[low_start:] = root_r[row] @ w_low
        for fraction_index, fraction in enumerate(fractions):
            a_row = (
                projection + np.sqrt(1.0 - fraction) * residual
                + np.sqrt(fraction) * tail)
            output[output_row, fraction_index] = a_row / sqrt_s
    return output


def select_settings(
        family: dict[str, np.ndarray | float | int], ratios: np.ndarray,
        n: int, alphas: np.ndarray) -> list[dict[str, float]]:
    c = np.asarray(family['feature_eigenvalues'])
    theta = np.asarray(family['feature_teacher'])
    signal = float(family['signal'])
    residual = float(family['residual_signal'])
    solver = SpectrumKappa(c, len(c) / n)
    settings = []
    for ratio in ratios:
        sigma = float(np.sqrt(ratio * signal))
        alpha, risks = select_spectral_feature_de_alpha(
            c, theta, np.ones_like(c), residual, signal,
            sigma, n, alphas)
        kappa = float(solver(alpha / n))
        settings.append({
            'noise_signal_ratio': float(ratio),
            'sigma': sigma,
            'alpha_cv': alpha,
            'kappa': kappa,
            'cv_risk_min': float(np.min(risks)),
        })
    return settings


def prediction_de_components(
        family: dict[str, np.ndarray | float | int], setting: dict[str, float],
        n: int) -> dict[str, np.ndarray | float]:
    c = np.asarray(family['feature_eigenvalues'])
    theta = np.asarray(family['feature_teacher'])
    residual = float(family['residual_signal'])
    signal = float(family['signal'])
    kappa = setting['kappa']
    df2 = compute_df2(c, kappa)
    denominator = n - df2
    shrink = c / (c + kappa)
    mean = shrink * theta
    c_sig = np.sum(c / (c + kappa) ** 2 * theta ** 2)
    variance_scale = (
        kappa ** 2 * c_sig + setting['sigma'] ** 2 + residual
        ) / denominator
    variance = variance_scale * c / (c + kappa) ** 2
    error = (1.0 - shrink) ** 2 * theta ** 2 + variance
    gen_error = residual + float(c @ error)
    gen_numerator = float(np.sum(c * theta * mean))
    gen_denominator = float(np.sum(c * (mean ** 2 + variance)))
    return {
        'mean': mean,
        'variance': variance,
        'gen_error_normalized': gen_error / signal,
        'r2_gen': 1.0 - gen_error / signal,
        'slope_gen': gen_numerator / gen_denominator,
        'df2': float(df2),
        'variance_scale': float(variance_scale),
    }


def theory_metrics(
        family: dict[str, np.ndarray | float | int],
        settings: list[dict[str, float]], n: int) -> dict[str, np.ndarray]:
    g_matrices = np.asarray(family['g_matrices'])
    q = np.asarray(family['teacher_overlap'])
    signal = float(family['signal'])
    output = {
        key: np.empty((len(g_matrices), len(settings)))
        for key in ('gen_error_normalized', 'r2_gen', 'slope_gen',
                    'acc_error_normalized', 'r2_acc', 'slope_acc',
                    'acc_numerator', 'acc_denominator')}
    for ratio_index, setting in enumerate(settings):
        components = prediction_de_components(family, setting, n)
        mean = np.asarray(components['mean'])
        variance = np.asarray(components['variance'])
        numerator = float(q @ mean)
        for angle_index, g in enumerate(g_matrices):
            denominator = float(
                mean @ g @ mean + np.diag(g) @ variance)
            slope = numerator / denominator
            output['gen_error_normalized'][angle_index, ratio_index] = (
                components['gen_error_normalized'])
            output['r2_gen'][angle_index, ratio_index] = components['r2_gen']
            output['slope_gen'][angle_index, ratio_index] = (
                components['slope_gen'])
            output['acc_error_normalized'][angle_index, ratio_index] = (
                1.0 - slope) ** 2
            output['r2_acc'][angle_index, ratio_index] = (
                1.0 - (1.0 - denominator / numerator) ** 2)
            output['slope_acc'][angle_index, ratio_index] = slope
            output['acc_numerator'][angle_index, ratio_index] = numerator
            output['acc_denominator'][angle_index, ratio_index] = denominator
    return output


def simulate_shared_ridge(
        family: dict[str, np.ndarray | float | int],
        settings: list[dict[str, float]], n: int, n_trials: int,
        seed: int, progress: bool) -> dict[str, np.ndarray]:
    """Exact Gaussian feature-design ridge fits shared across the rotation."""
    c = np.asarray(family['feature_eigenvalues'])
    theta = np.asarray(family['feature_teacher'])
    q = np.asarray(family['teacher_overlap'])
    signal = float(family['signal'])
    residual = float(family['residual_signal'])
    g_matrices = np.asarray(family['g_matrices'])
    n_ratios = len(settings)
    n_angles = len(g_matrices)
    p = len(c)
    output = {
        key: np.empty((n_ratios, n_trials))
        for key in ('gen_error_normalized', 'r2_gen', 'slope_gen')}
    output.update({
        key: np.empty((n_angles, n_ratios, n_trials))
        for key in ('acc_error_normalized', 'r2_acc', 'slope_acc',
                    'acc_numerator', 'acc_denominator')})
    estimates = np.empty((n_ratios, n_trials, p), dtype=float)
    rng = np.random.default_rng(seed)
    iterator = range(n_trials)
    if progress:
        try:
            from tqdm.auto import tqdm
            iterator = tqdm(
                iterator, desc=f'exact ridge top-PC p={p}', unit='fit')
        except ImportError:
            pass
    sqrt_c = np.sqrt(c)
    effective_sigmas = np.asarray([
        np.sqrt(setting['sigma'] ** 2 + residual) for setting in settings])
    for trial in iterator:
        design = rng.standard_normal((n, p)) * sqrt_c[None, :]
        gram = design.T @ design
        sample_values, sample_vectors = np.linalg.eigh(gram)
        sample_values = np.maximum(sample_values, 0.0)
        signal_rhs = gram @ theta
        noise = rng.standard_normal((n, n_ratios))
        noise_rhs = design.T @ noise
        for ratio_index, setting in enumerate(settings):
            rhs = (
                signal_rhs
                + effective_sigmas[ratio_index] * noise_rhs[:, ratio_index])
            estimate = sample_vectors @ (
                (sample_vectors.T @ rhs)
                / (sample_values + setting['alpha_cv']))
            estimates[ratio_index, trial] = estimate
            delta = estimate - theta
            gen_error = residual + float(c @ (delta ** 2))
            gen_numerator = float(np.sum(c * theta * estimate))
            gen_denominator = float(np.sum(c * estimate ** 2))
            output['gen_error_normalized'][ratio_index, trial] = (
                gen_error / signal)
            output['r2_gen'][ratio_index, trial] = 1.0 - gen_error / signal
            output['slope_gen'][ratio_index, trial] = (
                gen_numerator / gen_denominator)

    for ratio_index in range(n_ratios):
        fitted = estimates[ratio_index]
        numerator = fitted @ q
        # G(t) is a five-block trigonometric polynomial.  Evaluating those
        # five quadratic forms is much cheaper than contracting every dense
        # p-by-p G(t) separately, especially at p=500.
        block_quadratics = {
            key: np.einsum(
                'ti,ij,tj->t', fitted, np.asarray(family[key]), fitted,
                optimize=True)
            for key in ('g_pp', 'g_rr', 'g_pr_sym', 'g_vv', 'g_pv_sym')}
        fractions = np.asarray(family['tail_fraction'])
        sine = np.sqrt(fractions)[:, None]
        cosine = np.sqrt(1.0 - fractions)[:, None]
        denominator = (
            block_quadratics['g_pp'][None, :]
            + cosine ** 2 * block_quadratics['g_rr'][None, :]
            + cosine * block_quadratics['g_pr_sym'][None, :]
            + fractions[:, None] * block_quadratics['g_vv'][None, :]
            + sine * (1.0 - cosine)
            * block_quadratics['g_pv_sym'][None, :])
        nonlinear = nonlinear_metrics(numerator[None, :], denominator)
        output['acc_error_normalized'][:, ratio_index] = (
            nonlinear['acc_error_normalized'])
        output['r2_acc'][:, ratio_index] = nonlinear['r2_acc']
        output['slope_acc'][:, ratio_index] = nonlinear['slope_acc']
        output['acc_numerator'][:, ratio_index] = numerator[None, :]
        output['acc_denominator'][:, ratio_index] = denominator
    output['estimates'] = estimates
    return output


def _summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    return {
        'mean': float(np.mean(values)),
        'se': float(np.std(values, ddof=1) / np.sqrt(len(values))),
        'median': float(np.median(values)),
        'q25': float(np.quantile(values, 0.25)),
        'q75': float(np.quantile(values, 0.75)),
    }


def build_summary_rows(
        families, settings_by_p, theory_by_p, mc_by_p, n: int,
        trials_by_p: dict[int, int]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for p, family in families.items():
        fractions = np.asarray(family['tail_fraction'])
        trace = np.asarray(family['trace_g'])
        effective_rank = np.asarray(family['effective_rank_g'])
        for angle_index, fraction in enumerate(fractions):
            for ratio_index, setting in enumerate(settings_by_p[p]):
                row: dict[str, object] = {
                    'p': p,
                    'n': n,
                    'n_trials': trials_by_p[p],
                    'map_index': angle_index,
                    'tail_fraction': fraction,
                    'trace_g': trace[angle_index],
                    'effective_rank_g': effective_rank[angle_index],
                    'represented_signal_fraction': (
                        float(family['represented_signal'])
                        / float(family['signal'])),
                    'noise_signal_ratio': setting['noise_signal_ratio'],
                    'sigma': setting['sigma'],
                    'de_alpha_cv': setting['alpha_cv'],
                    'de_kappa': setting['kappa'],
                }
                for metric, values in theory_by_p[p].items():
                    row[f'de_{metric}'] = values[angle_index, ratio_index]
                for metric in ('gen_error_normalized', 'r2_gen', 'slope_gen'):
                    statistics = _summary(mc_by_p[p][metric][ratio_index])
                    for name, value in statistics.items():
                        row[f'mc_{metric}_{name}'] = value
                for metric in ('acc_error_normalized', 'r2_acc', 'slope_acc',
                               'acc_numerator', 'acc_denominator'):
                    statistics = _summary(
                        mc_by_p[p][metric][angle_index, ratio_index])
                    for name, value in statistics.items():
                        row[f'mc_{metric}_{name}'] = value
                rows.append(row)
    return rows


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    path.chmod(0o644)


def read_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(newline='') as handle:
        for source in csv.DictReader(handle):
            rows.append({
                key: (value if key == '' else float(value))
                for key, value in source.items()})
    return rows


def _select(rows, p, ratio):
    return sorted([
        row for row in rows
        if int(float(row['p'])) == p and np.isclose(
            float(row['noise_signal_ratio']), ratio)],
        key=lambda row: float(row['map_index']))


def _values(rows, key):
    return np.asarray([float(row[key]) for row in rows])


def _style_rotation_axis(ax):
    ax.set_xscale('symlog', linthresh=1e-6, linscale=0.6)
    ax.set_xlim(-2e-7, 1.15)
    ax.grid(alpha=0.18, which='both')
    ax.spines[['top', 'right']].set_alpha(0.35)


def plot_dimension(rows: list[dict[str, object]], p: int, path: Path) -> None:
    ratios = sorted({
        float(row['noise_signal_ratio']) for row in rows
        if int(float(row['p'])) == p})
    colors = plt.get_cmap('viridis')(
        np.linspace(0.06, 0.92, len(ratios)))
    base = _select(rows, p, ratios[0])
    x = _values(base, 'tail_fraction')
    fig, axes = plt.subplots(2, 3, figsize=(15.3, 8.3), sharex=True)

    geometry = axes[0, 0]
    geometry.plot(x, _values(base, 'trace_g'), color='0.18', lw=2.2)
    geometry.set_yscale('log')
    geometry.set_ylabel(r'$\operatorname{tr}(FF^\top)$')
    rank_axis = geometry.twinx()
    rank_axis.plot(
        x, _values(base, 'effective_rank_g'), color='0.55',
        lw=1.7, ls='--')
    rank_axis.set_ylabel('effective rank', color='0.45')
    rank_axis.tick_params(axis='y', colors='0.45')
    geometry.set_title(f'Top-{p} projector rotates into tail PCs')

    for color, ratio in zip(colors, ratios):
        selected = _select(rows, p, ratio)
        sigma = float(selected[0]['sigma'])
        label = rf'$\sigma={sigma:.3g}$ ($\sigma^2/S={ratio:g}$)'
        for ax, metric, ylabel, yscale in (
                (axes[0, 1], 'gen_error_normalized', r'$E_{\rm gen}/S$', 'log'),
                (axes[0, 2], 'r2_gen', r'$R^2_{\rm gen}$', 'linear')):
            de = _values(selected, f'de_{metric}')
            mc = _values(selected, f'mc_{metric}_mean')
            ax.plot(x, de, color=color, lw=2.0, label=label)
            ax.plot(x[::8], mc[::8], 'o', color=color, ms=3.6,
                    mec='white', mew=0.4)
            ax.set_ylabel(ylabel)
            ax.set_yscale(yscale)
        for ax, metric, ylabel, yscale in (
                (axes[1, 0], 'acc_error_normalized', r'$E_{\rm acc}/S$', 'log'),
                (axes[1, 1], 'r2_acc', r'$R^2_{\rm acc}$', 'symlog'),
                (axes[1, 2], 'slope_acc', r'slope$_{\rm acc}$', 'log')):
            de = _values(selected, f'de_{metric}')
            statistic = 'median' if metric == 'r2_acc' else 'mean'
            mc = _values(selected, f'mc_{metric}_{statistic}')
            ax.plot(x, de, color=color, lw=2.0)
            ax.plot(x[::8], mc[::8], 'o', color=color, ms=3.6,
                    mec='0.15', mew=0.35)
            ax.set_ylabel(ylabel)
            if yscale == 'symlog':
                ax.set_yscale('symlog', linthresh=1.0)
                ax.axhline(0.0, color='0.55', lw=0.7, ls='--')
                ax.axhline(1.0, color='0.65', lw=0.7, ls=':')
            else:
                ax.set_yscale(yscale)

    axes[0, 1].set_title('Generalization error is invariant')
    axes[0, 2].set_title('Generalization $R^2$ is invariant')
    axes[1, 0].set_title('Accentuation error leaves the good endpoint')
    axes[1, 1].set_title(r'$R^2_{\rm acc}$: near 1 to very negative')
    axes[1, 2].set_title('Accentuation slope collapses')
    axes[0, 1].legend(fontsize=8, ncol=2, loc='best')
    axes[1, 0].legend(handles=[
        plt.Line2D([0], [0], color='0.2', lw=2, label='leading DE'),
        plt.Line2D([0], [0], color='0.2', marker='o', lw=0, ms=4,
                   label='exact finite-$n$ MC'),
    ], fontsize=8, loc='best')
    for ax in axes.ravel():
        _style_rotation_axis(ax)
    for ax in axes[1]:
        ax.set_xlabel(r'tail loading $t=\sin^2\theta$')
    geometry.set_xlabel(r'tail loading $t=\sin^2\theta$')
    represented = float(base[0]['represented_signal_fraction'])
    fig.suptitle(
        rf'Van Hateren disk teacher: top-{p} PCs $\rightarrow$ '
        rf'prediction-null spectral tail', fontsize=15)
    fig.text(
        0.5, 0.012,
        rf'At $t=0$, $F$ is the unwhitened top-{p} PC projector '
        rf'({100 * represented:.3f}% of signal); '
        r'$F\Sigma F^\top$, $F\Sigma\beta^\star$, and '
        r'$F\beta^\star$ stay fixed.',
        ha='center', fontsize=9, color='0.28')
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches='tight')
    path.chmod(0o644)
    plt.close(fig)


def _log_energy_bins(values: np.ndarray, n_bins: int = 72):
    d = len(values)
    edges = np.unique(np.rint(np.geomspace(1, d + 1, n_bins + 1)).astype(int))
    edges[0] = 1
    edges[-1] = d + 1
    energy = values ** 2
    total = np.sum(energy)
    binned = np.asarray([
        np.sum(energy[start - 1:stop - 1]) / total
        for start, stop in zip(edges[:-1], edges[1:])])
    centers = np.sqrt(edges[:-1] * (edges[1:] - 1))
    return centers, binned


def plot_feature_spectra(
        families, eigenvalues: np.ndarray, path: Path) -> None:
    fractions = np.asarray([0.0, 1e-4, 1e-2, 1.0])
    dimensions = sorted(families)
    fig, axes = plt.subplots(
        len(dimensions), len(fractions), figsize=(14.8, 6.3),
        sharex=True, sharey=False, squeeze=False)
    colors = plt.get_cmap('magma')(
        np.linspace(0.18, 0.82, 3))
    for dimension_index, p in enumerate(dimensions):
        row_indices = np.asarray([0, p // 2, p - 1])
        feature_rows = feature_rows_pc(
            families[p], eigenvalues, row_indices, fractions)
        for fraction_index, fraction in enumerate(fractions):
            ax = axes[dimension_index, fraction_index]
            for color, feature_index, values in zip(
                    colors, row_indices, feature_rows[:, fraction_index]):
                rank, energy = _log_energy_bins(values)
                ax.plot(
                    rank, energy, color=color, lw=1.6,
                    label=f'row {feature_index + 1}')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_ylim(1e-9, 1.2)
            ax.grid(alpha=0.16, which='both')
            ax.spines[['top', 'right']].set_alpha(0.35)
            ax.set_title(
                ('top-PC endpoint' if fraction == 0
                 else rf'$t={fraction:g}$'))
            if fraction_index == 0:
                ax.set_ylabel(
                    rf'$p={p}$' + '\nnormalized feature energy')
            if dimension_index == len(dimensions) - 1:
                ax.set_xlabel('population PC rank')
            if dimension_index == 0 and fraction_index == 0:
                ax.legend(fontsize=8, loc='lower left')
    fig.suptitle(
        'How individual feature rows move from top PCs into the spectral tail',
        fontsize=15)
    fig.text(
        0.5, 0.012,
        'Each curve is normalized to expose its spectral location; '
        r'the pixel-space norm growth is quantified by $\operatorname{tr}(FF^\top)$ '
        'in the metric figures.',
        ha='center', fontsize=9, color='0.28')
    fig.tight_layout(rect=(0, 0.055, 1, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches='tight')
    path.chmod(0o644)
    plt.close(fig)


def render(rows, families, eigenvalues, figure_dir: Path, feature_path: Path):
    for p in sorted(families):
        plot_dimension(
            rows, p, figure_dir / FIGURE_TEMPLATE.format(p=p))
    plot_feature_spectra(families, eigenvalues, feature_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dimensions', type=int, nargs='+',
                        default=list(DEFAULT_DIMENSIONS))
    parser.add_argument('--n', type=int, default=1000)
    parser.add_argument('--n-trials-100', type=int, default=1000)
    parser.add_argument('--n-trials-500', type=int, default=300)
    parser.add_argument('--ratios', type=float, nargs='+',
                        default=DEFAULT_RATIOS.tolist())
    parser.add_argument('--n-log-tail', type=int, default=61)
    parser.add_argument('--min-tail', type=float, default=1e-6)
    parser.add_argument('--low-start-fraction', type=float, default=0.60)
    parser.add_argument('--alpha-grid-size', type=int, default=281)
    parser.add_argument('--alpha-min-exp', type=float, default=-4.0)
    parser.add_argument('--alpha-max-exp', type=float, default=5.0)
    parser.add_argument('--benchmark-trials', type=int, default=4)
    parser.add_argument('--seed', type=int, default=20260901)
    parser.add_argument('--summary-path', type=Path, default=SUMMARY_PATH)
    parser.add_argument('--raw-path', type=Path, default=RAW_PATH)
    parser.add_argument('--figure-dir', type=Path,
                        default=REPO_ROOT / 'figures' / 'null_rotations')
    parser.add_argument('--feature-figure-path', type=Path,
                        default=FEATURE_FIGURE_PATH)
    parser.add_argument('--log-file', type=Path, default=DEFAULT_LOG_PATH)
    parser.add_argument('--plot-only', action='store_true')
    parser.add_argument('--benchmark-only', action='store_true')
    parser.add_argument('--no-progress', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    logger = configure_logger(args.log_file)
    with np.load(SPECTRUM_PATH) as spectrum:
        eigenvalues = np.asarray(spectrum['eigenvalues'], dtype=float)
        beta = np.asarray(spectrum['beta_proj'], dtype=float)
    dimensions = tuple(sorted(set(args.dimensions)))
    fractions = np.concatenate((
        [0.0], np.geomspace(args.min_tail, 1.0, args.n_log_tail)))
    families = {
        p: build_top_pc_tail_family(
            eigenvalues, beta, p, fractions, args.low_start_fraction,
            args.seed + p)
        for p in dimensions}
    if args.plot_only:
        rows = read_rows(args.summary_path)
        render(
            rows, families, eigenvalues, args.figure_dir,
            args.feature_figure_path)
        logger.info('Replotted top-PC tail rotations from %s', args.summary_path)
        return

    ratios = np.asarray(args.ratios, dtype=float)
    alphas = np.logspace(
        args.alpha_min_exp, args.alpha_max_exp, args.alpha_grid_size)
    settings_by_p = {
        p: select_settings(family, ratios, args.n, alphas)
        for p, family in families.items()}
    theory_by_p = {
        p: theory_metrics(family, settings_by_p[p], args.n)
        for p, family in families.items()}
    trials_by_p = {
        p: (args.n_trials_100 if p <= 100 else args.n_trials_500)
        for p in dimensions}

    for p, family in families.items():
        logger.info(
            'p=%d: represented signal %.6f, G trace %.3g -> %.3g, '
            'identity error %.2e, tail null residuals b/c %.2e/%.2e',
            p, float(family['represented_signal']) / float(family['signal']),
            float(np.asarray(family['trace_g'])[0]),
            float(np.asarray(family['trace_g'])[-1]),
            float(family['g0_identity_error']),
            float(family['tail_b_residual']),
            float(family['tail_c_residual']))

    projected_total = 0.0
    for p, family in families.items():
        start = time.perf_counter()
        simulate_shared_ridge(
            family, settings_by_p[p], args.n, args.benchmark_trials,
            args.seed + 10_000 + p, progress=False)
        elapsed = time.perf_counter() - start
        projected = elapsed * trials_by_p[p] / args.benchmark_trials
        projected_total += projected
        logger.info(
            'Pilot p=%d: %d exact fits in %.2fs; projected %.1fs (%.1fmin) '
            'for %d trials', p, args.benchmark_trials, elapsed,
            projected, projected / 60.0, trials_by_p[p])
    logger.info(
        'Projected total exact-fit time %.1fs (%.1fmin)',
        projected_total, projected_total / 60.0)
    if args.benchmark_only:
        return

    mc_by_p = {}
    for p, family in families.items():
        start = time.perf_counter()
        mc_by_p[p] = simulate_shared_ridge(
            family, settings_by_p[p], args.n, trials_by_p[p],
            args.seed + 20_000 + p, progress=not args.no_progress)
        logger.info('p=%d exact simulation completed in %.1fs',
                    p, time.perf_counter() - start)

    rows = build_summary_rows(
        families, settings_by_p, theory_by_p, mc_by_p,
        args.n, trials_by_p)
    write_rows(args.summary_path, rows)
    raw: dict[str, np.ndarray | float | int] = {
        'eigenvalues': eigenvalues,
        'beta': beta,
        'ratios': ratios,
        'tail_fraction': fractions,
    }
    for p, family in families.items():
        for key in ('feature_eigenvalues', 'feature_teacher',
                    'teacher_overlap', 'trace_g', 'trace_g2',
                    'effective_rank_g', 'g_matrices'):
            raw[f'p{p}_{key}'] = np.asarray(family[key])
        for key, values in theory_by_p[p].items():
            raw[f'p{p}_de_{key}'] = values
        for key, values in mc_by_p[p].items():
            raw[f'p{p}_mc_{key}'] = values
        for key in ('signal', 'represented_signal', 'residual_signal',
                    'low_start', 'g0_identity_error', 'tail_b_residual',
                    'tail_c_residual'):
            raw[f'p{p}_{key}'] = family[key]
    raw_path = require_bulk_path(
        args.raw_path, 'Van Hateren top-PC-to-tail raw cache')
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(raw_path, **raw)
    raw_path.chmod(0o644)
    render(
        rows, families, eigenvalues, args.figure_dir,
        args.feature_figure_path)

    for p in dimensions:
        selected = _select(rows, p, ratios[0])
        logger.info(
            'p=%d lowest noise: R2gen %.6f invariant; R2acc DE %.6g -> %.6g, '
            'MC median %.6g -> %.6g',
            p, float(selected[0]['de_r2_gen']),
            float(selected[0]['de_r2_acc']),
            float(selected[-1]['de_r2_acc']),
            float(selected[0]['mc_r2_acc_median']),
            float(selected[-1]['mc_r2_acc_median']))
    logger.info('Plot-ready summary: %s', args.summary_path)
    logger.info('Raw cache: %s', raw_path)
    logger.info('Feature visualization: %s', args.feature_figure_path)


if __name__ == '__main__':
    main()

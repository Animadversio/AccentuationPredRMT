"""Prediction-equivalent null rotations for the Van Hateren disk teacher.

The measured Van Hateren covariance spectrum and circular-teacher alignment are
loaded from ``tables/vanhateren_disk_teacher_spectrum.npz``.  In whitened input
coordinates, let

    b = Sigma^{1/2} beta*,        A = F Sigma^{1/2}.

Every map constructed here satisfies

    A A^T = I_p,                 A b = sqrt(S) e_1,

so ``Sigma_F``, the realizable teacher, the complete DE-CV risk curve, and all
natural-distribution prediction metrics are identical.  Only the pixel metric

    G = F F^T = A Sigma^{-1} A^T

changes.  Two families are evaluated:

1. ``givens`` rotates one null row between robust high- and low-variance PC
   bands while remaining orthogonal to both ``b`` and ``Sigma^{-1} b``.  Thus
   the teacher overlap is fixed and the experiment isolates metric inflation.
2. ``random`` samples prediction-null Stiefel frames from a stratified spectral
   pool.  These rotations preserve prediction but allow both ``G`` and the
   teacher-overlap vector to vary.

The same Gaussian feature designs, response noise, RidgeCV selections, and
feature-space fits are reused for every map.  Plot-ready CSV and raw trial/map
arrays are cached separately from figures.
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

from rmt_core import (
    SpectrumKappa,
    select_spectral_feature_de_alpha,
    spectral_feature_de_metrics,
)
from scripts.validate_ffhq_disk_teacher import configure_logger
from scripts.storage_paths import configured_bulk_path, require_bulk_path
from scripts.validate_vanhateren_disk_teacher import SPECTRUM_PATH


REPO_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = REPO_ROOT / 'tables' / 'vanhateren_null_rotation_summary.csv'
RAW_PATH = configured_bulk_path('tables/vanhateren_null_rotation_cases.npz')
PLANE_FIGURE_PATH = (
    REPO_ROOT / 'figures' / 'null_rotations' /
    'vanhateren_null_rotation_plane.png')
GIVENS_FIGURE_PATH = (
    REPO_ROOT / 'figures' / 'null_rotations' /
    'vanhateren_null_givens_metrics.png')
RANDOM_FIGURE_PATH = (
    REPO_ROOT / 'figures' / 'null_rotations' /
    'vanhateren_null_random_metrics.png')
DEFAULT_LOG_PATH = REPO_ROOT / 'logs' / 'vanhateren_null_rotations.log'

DEFAULT_RATIOS = np.asarray([0.01, 0.1, 1.0, 10.0])


def _constraint_basis(*vectors: np.ndarray) -> np.ndarray:
    """Return an orthonormal basis for the supplied constraint columns."""
    columns = []
    for vector in vectors:
        value = np.asarray(vector, dtype=float)
        norm = np.linalg.norm(value)
        if norm > 1e-14:
            columns.append(value / norm)
    if not columns:
        return np.empty((len(vectors[0]), 0), dtype=float)
    matrix = np.column_stack(columns)
    basis, triangular = np.linalg.qr(matrix, mode='reduced')
    keep = np.abs(np.diag(triangular)) > 1e-10
    return basis[:, keep]


def _project_off(matrix: np.ndarray, constraints: np.ndarray) -> np.ndarray:
    if constraints.shape[1] == 0:
        return matrix
    return matrix - constraints @ (constraints.T @ matrix)


def projected_coordinate_extreme(
        inverse_eigenvalues: np.ndarray, b: np.ndarray, c: np.ndarray,
        indices: np.ndarray, maximize: bool
        ) -> tuple[np.ndarray, dict[str, float]]:
    """Find a stable high/low inverse-variance direction in a PC band.

    Candidate coordinate vectors are projected into the intersection of the
    null spaces of ``b`` and ``c = Sigma^{-1} b``.  The projected coordinate
    with the extremal inverse-covariance Rayleigh quotient is returned.
    """
    indices = np.asarray(indices, dtype=int)
    local_inverse = inverse_eigenvalues[indices]
    constraints = _constraint_basis(b[indices], c[indices])
    best_value = -np.inf if maximize else np.inf
    best_local = None
    best_coordinate = -1
    for coordinate in range(len(indices)):
        candidate = np.zeros(len(indices), dtype=float)
        candidate[coordinate] = 1.0
        candidate = _project_off(candidate[:, None], constraints)[:, 0]
        norm = np.linalg.norm(candidate)
        if norm < 1e-6:
            continue
        candidate /= norm
        value = float(np.sum(local_inverse * candidate ** 2))
        better = value > best_value if maximize else value < best_value
        if better:
            best_value = value
            best_local = candidate
            best_coordinate = coordinate
    if best_local is None:
        raise RuntimeError('No nonzero null direction found in spectral band')
    full = np.zeros_like(b)
    full[indices] = best_local
    metadata = {
        'inverse_variance_rayleigh': best_value,
        'effective_input_eigenvalue': 1.0 / best_value,
        'dominant_pc_rank': float(indices[best_coordinate] + 1),
        'band_start_rank': float(indices[0] + 1),
        'band_stop_rank': float(indices[-1] + 1),
    }
    return full, metadata


def orthonormal_rows_in_band(
        b: np.ndarray, c: np.ndarray, indices: np.ndarray, n_rows: int,
        rng: np.random.Generator) -> np.ndarray:
    """Sample rows supported on one band and null to both b and c."""
    indices = np.asarray(indices, dtype=int)
    constraints = _constraint_basis(b[indices], c[indices])
    random_columns = rng.standard_normal((len(indices), n_rows))
    random_columns = _project_off(random_columns, constraints)
    columns, triangular = np.linalg.qr(random_columns, mode='reduced')
    if np.min(np.abs(np.diag(triangular))) < 1e-8:
        raise RuntimeError('Degenerate fixed-null-frame draw')
    rows = np.zeros((n_rows, len(b)), dtype=float)
    rows[:, indices] = columns[:, :n_rows].T
    return rows


def spectral_bands(d: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return robust high, middle, and low PC bands by rank fraction."""
    edges = np.asarray([0.002, 0.03, 0.15, 0.60, 0.95])
    ranks = np.clip(np.rint(edges * d).astype(int), 1, d)
    high = np.arange(ranks[0], max(ranks[1], ranks[0] + 4))
    middle = np.arange(ranks[1], max(ranks[2], ranks[1] + 8))
    low = np.arange(ranks[3], max(ranks[4], ranks[3] + 4))
    if len(high) < 3 or len(middle) < 3 or len(low) < 3:
        raise ValueError('Spectrum is too small for the default rank bands')
    return high, middle, low


def build_givens_family(
        eigenvalues: np.ndarray, beta: np.ndarray, p: int,
        n_angles: int, seed: int
        ) -> tuple[np.ndarray, list[dict[str, float]], dict[str, np.ndarray | float]]:
    """Construct a metric-only high-to-low null Givens rotation."""
    if p < 2:
        raise ValueError('p must be at least 2')
    s = np.asarray(eigenvalues, dtype=float)
    beta = np.asarray(beta, dtype=float)
    inverse = 1.0 / s
    b = np.sqrt(s) * beta
    c = inverse * b
    signal = float(b @ b)
    a0 = b / np.sqrt(signal)
    high_band, middle_band, low_band = spectral_bands(len(s))
    u_high, high_meta = projected_coordinate_extreme(
        inverse, b, c, high_band, maximize=False)
    u_low, low_meta = projected_coordinate_extreme(
        inverse, b, c, low_band, maximize=True)
    rng = np.random.default_rng(seed)
    fixed = orthonormal_rows_in_band(
        b, c, middle_band, p - 2, rng) if p > 2 else np.empty((0, len(s)))

    theta = np.linspace(0.0, 0.5 * np.pi, n_angles)
    maps = np.empty((n_angles, p, p), dtype=float)
    rows: list[dict[str, float]] = []
    target_h = np.zeros(p)
    target_h[0] = np.sqrt(signal)
    max_c_error = 0.0
    max_h_error = 0.0
    max_q_error = 0.0
    q_reference = None
    for index, angle in enumerate(theta):
        rotating = np.cos(angle) * u_high + np.sin(angle) * u_low
        A = np.vstack((a0, rotating[None, :], fixed))
        covariance = A @ A.T
        h = A @ b
        G = (A * inverse[None, :]) @ A.T
        q = A @ c
        if q_reference is None:
            q_reference = q.copy()
        maps[index] = G
        max_c_error = max(max_c_error, float(np.max(np.abs(
            covariance - np.eye(p)))))
        max_h_error = max(max_h_error, float(np.max(np.abs(h - target_h))))
        max_q_error = max(max_q_error, float(np.max(np.abs(q - q_reference))))
        eig_g = np.linalg.eigvalsh(G)
        rows.append({
            'map_index': float(index),
            'theta': float(angle),
            'low_fraction': float(np.sin(angle) ** 2),
            'trace_g': float(np.trace(G)),
            'max_eigenvalue_g': float(eig_g[-1]),
            'condition_g': float(eig_g[-1] / eig_g[0]),
            'q_norm': float(np.linalg.norm(q)),
            'q_null_norm': float(np.linalg.norm(q[1:])),
        })
    plane = {
        'u_high': u_high,
        'u_low': u_low,
        'theta': theta,
        'max_c_invariance_error': max_c_error,
        'max_h_invariance_error': max_h_error,
        'max_q_invariance_error': max_q_error,
        **{f'high_{key}': value for key, value in high_meta.items()},
        **{f'low_{key}': value for key, value in low_meta.items()},
    }
    return maps, rows, plane


def stratified_pool_indices(
        d: int, pool_size: int, rng: np.random.Generator) -> np.ndarray:
    """Sample equal counts from four separated spectral rank bands."""
    if pool_size < 8:
        raise ValueError('pool_size must be at least 8')
    edges = np.asarray([0.002, 0.03, 0.15, 0.60, 0.95])
    bounds = np.clip(np.rint(edges * d).astype(int), 1, d)
    counts = np.full(4, pool_size // 4, dtype=int)
    counts[:pool_size % 4] += 1
    selected = []
    for start, stop, count in zip(bounds[:-1], bounds[1:], counts):
        candidates = np.arange(start, max(stop, start + count))
        if count > len(candidates):
            raise ValueError('pool_size is too large for a spectral band')
        selected.extend(rng.choice(candidates, size=count, replace=False))
    return np.asarray(sorted(selected), dtype=int)


def build_random_family(
        eigenvalues: np.ndarray, beta: np.ndarray, p: int,
        n_maps: int, pool_size: int, seed: int
        ) -> tuple[np.ndarray, list[dict[str, float]], np.ndarray, dict[str, float]]:
    """Sample prediction-null Stiefel frames in a stratified PC pool."""
    if p < 2 or pool_size < p:
        raise ValueError('Require 2 <= p <= pool_size')
    s = np.asarray(eigenvalues, dtype=float)
    beta = np.asarray(beta, dtype=float)
    inverse = 1.0 / s
    b = np.sqrt(s) * beta
    signal = float(b @ b)
    a0 = b / np.sqrt(signal)
    g00 = float(np.sum(inverse * a0 ** 2))
    rng = np.random.default_rng(seed)
    pool = stratified_pool_indices(len(s), pool_size, rng)
    pool_b = b[pool]
    pool_inverse = inverse[pool]
    pool_c_over_sqrt_signal = inverse[pool] * b[pool] / np.sqrt(signal)
    b_constraint = _constraint_basis(pool_b)

    maps = np.empty((n_maps, p, p), dtype=float)
    rows: list[dict[str, float]] = []
    max_c_error = 0.0
    max_h_error = 0.0
    target_h = np.zeros(p)
    target_h[0] = np.sqrt(signal)
    for index in range(n_maps):
        random_columns = rng.standard_normal((pool_size, p - 1))
        random_columns = _project_off(random_columns, b_constraint)
        columns, triangular = np.linalg.qr(random_columns, mode='reduced')
        if np.min(np.abs(np.diag(triangular))) < 1e-8:
            raise RuntimeError('Degenerate random Stiefel draw')
        q_rows = columns[:, :p - 1].T
        cross = q_rows @ pool_c_over_sqrt_signal
        null_block = (
            q_rows * pool_inverse[None, :]) @ q_rows.T
        G = np.empty((p, p), dtype=float)
        G[0, 0] = g00
        G[0, 1:] = cross
        G[1:, 0] = cross
        G[1:, 1:] = null_block
        maps[index] = G

        # C and h are calculated in the reduced representation for diagnostics.
        local_c = q_rows @ q_rows.T
        local_h = q_rows @ pool_b
        covariance_error = max(
            float(np.max(np.abs(local_c - np.eye(p - 1)))),
            float(np.max(np.abs(local_h))) / np.sqrt(signal))
        h_error = float(np.max(np.abs(local_h)))
        max_c_error = max(max_c_error, covariance_error)
        max_h_error = max(max_h_error, h_error)
        q = G @ target_h
        eig_g = np.linalg.eigvalsh(G)
        rows.append({
            'map_index': float(index),
            'theta': np.nan,
            'low_fraction': np.nan,
            'trace_g': float(np.trace(G)),
            'max_eigenvalue_g': float(eig_g[-1]),
            'condition_g': float(eig_g[-1] / eig_g[0]),
            'q_norm': float(np.linalg.norm(q)),
            'q_null_norm': float(np.linalg.norm(q[1:])),
        })
    diagnostics = {
        'max_c_invariance_error': max_c_error,
        'max_h_invariance_error': max_h_error,
    }
    return maps, rows, pool, diagnostics


def common_de_settings(
        p: int, signal: float, ratios: np.ndarray, n: int,
        alphas: np.ndarray) -> list[dict[str, float]]:
    """Select the common DE-CV alpha/kappa for every noise level."""
    c = np.ones(p)
    wstar = np.zeros(p)
    wstar[0] = np.sqrt(signal)
    solver = SpectrumKappa(c, p / n)
    settings = []
    for ratio in ratios:
        sigma = float(np.sqrt(ratio * signal))
        alpha, _ = select_spectral_feature_de_alpha(
            c, wstar, np.ones(p), 0.0, signal, sigma, n, alphas)
        kappa = float(solver(alpha / n))
        settings.append({
            'noise_signal_ratio': float(ratio),
            'sigma': sigma,
            'alpha_cv': alpha,
            'kappa': kappa,
        })
    return settings


def de_metrics_for_maps(
        maps: np.ndarray, signal: float, settings: list[dict[str, float]],
        n: int) -> dict[str, np.ndarray]:
    """Evaluate non-diagonal G by diagonalizing it; C=I commutes with G."""
    n_maps, p, _ = maps.shape
    wstar = np.zeros(p)
    wstar[0] = np.sqrt(signal)
    names = (
        'gen_error_normalized', 'r2_gen', 'slope_gen',
        'acc_error', 'acc_error_normalized', 'r2_acc', 'slope_acc',
        'acc_numerator', 'acc_denominator')
    output = {
        name: np.empty((n_maps, len(settings)), dtype=float)
        for name in names}
    for map_index, G in enumerate(maps):
        g, rotation = np.linalg.eigh(G)
        if np.min(g) <= 0:
            raise RuntimeError('Feature-map metric is not positive definite')
        teacher = rotation.T @ wstar
        for ratio_index, setting in enumerate(settings):
            metrics = spectral_feature_de_metrics(
                np.ones(p), teacher, g, 0.0, signal,
                setting['kappa'], setting['sigma'], n)
            for name in names:
                output[name][map_index, ratio_index] = metrics[name]
    return output


def _common_prediction_metrics(
        estimate: np.ndarray, wstar: np.ndarray, signal: float
        ) -> dict[str, float]:
    delta = estimate - wstar
    gen_error_normalized = float(delta @ delta / signal)
    numerator = float(estimate @ wstar)
    denominator = float(estimate @ estimate)
    return {
        'gen_error_normalized': gen_error_normalized,
        'r2_gen': 1.0 - gen_error_normalized,
        'slope_gen': numerator / denominator,
    }


def simulate_common_ridgecv(
        maps_by_family: dict[str, np.ndarray], signal: float,
        ratios: np.ndarray, alphas: np.ndarray, n: int, n_trials: int,
        seed: int, progress: bool
        ) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, np.ndarray], float]:
    """Fit shared RidgeCV paths and evaluate every feature map per fit."""
    p = next(iter(maps_by_family.values())).shape[1]
    if any(maps.shape[1:] != (p, p) for maps in maps_by_family.values()):
        raise ValueError('All map families must have the same feature dimension')
    wstar = np.zeros(p)
    wstar[0] = np.sqrt(signal)
    q_by_family = {
        family: maps @ wstar for family, maps in maps_by_family.items()}
    metric_names = (
        'acc_error', 'acc_error_normalized', 'r2_acc', 'slope_acc',
        'acc_numerator', 'acc_denominator')
    family_metrics = {
        family: {
            name: np.empty((len(maps), len(ratios), n_trials), dtype=float)
            for name in metric_names}
        for family, maps in maps_by_family.items()}
    prediction = {
        name: np.empty((len(ratios), n_trials), dtype=float)
        for name in ('alpha_cv', 'gen_error_normalized', 'r2_gen', 'slope_gen')}

    rng = np.random.default_rng(seed)
    iterator = range(n_trials)
    if progress:
        try:
            from tqdm.auto import tqdm
            iterator = tqdm(iterator, desc='shared RidgeCV', unit='design')
        except ImportError:
            pass
    start = time.perf_counter()
    for trial in iterator:
        X = rng.standard_normal((n, p))
        Xc = X - X.mean(axis=0, keepdims=True)
        gram = Xc.T @ Xc
        eigenvalues, eigenvectors = np.linalg.eigh(gram)
        eigenvalues = np.maximum(eigenvalues, 0.0)
        rotated_design = Xc @ eigenvectors
        inverse_path = 1.0 / (eigenvalues[:, None] + alphas[None, :])
        leverage = (
            1.0 / n + rotated_design ** 2 @ inverse_path)
        denominator_loo = np.maximum(1.0 - leverage, 1e-10)
        noiseless = X @ wstar

        for ratio_index, ratio in enumerate(ratios):
            sigma = np.sqrt(ratio * signal)
            y = noiseless + rng.standard_normal(n) * sigma
            yc = y - y.mean()
            rhs = rotated_design.T @ yc
            coefficients_eigen = rhs[:, None] * inverse_path
            residual_path = yc[:, None] - rotated_design @ coefficients_eigen
            loo_risk = np.mean((residual_path / denominator_loo) ** 2, axis=0)
            selected = int(np.argmin(loo_risk))
            estimate = eigenvectors @ coefficients_eigen[:, selected]
            prediction['alpha_cv'][ratio_index, trial] = alphas[selected]
            pred_metrics = _common_prediction_metrics(estimate, wstar, signal)
            for name, value in pred_metrics.items():
                prediction[name][ratio_index, trial] = value

            for family, maps in maps_by_family.items():
                numerator = q_by_family[family] @ estimate
                denominator = np.einsum(
                    'i,mij,j->m', estimate, maps, estimate, optimize=True)
                slope = np.divide(
                    numerator, denominator,
                    out=np.full_like(numerator, np.nan),
                    where=np.abs(denominator) > 0)
                r2 = np.divide(
                    denominator, numerator,
                    out=np.full_like(numerator, np.nan),
                    where=np.abs(numerator) > 1e-300)
                r2 = 1.0 - (1.0 - r2) ** 2
                values = {
                    'acc_error': signal * (1.0 - slope) ** 2,
                    'acc_error_normalized': (1.0 - slope) ** 2,
                    'r2_acc': r2,
                    'slope_acc': slope,
                    'acc_numerator': numerator,
                    'acc_denominator': denominator,
                }
                for name, value in values.items():
                    family_metrics[family][name][:, ratio_index, trial] = value
    elapsed = time.perf_counter() - start
    return family_metrics, prediction, elapsed


def _summaries(values: np.ndarray) -> dict[str, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return {key: np.nan for key in (
            'mean', 'std', 'se', 'median', 'q25', 'q75')}
    std = float(np.std(finite, ddof=1)) if len(finite) > 1 else np.nan
    return {
        'mean': float(np.mean(finite)),
        'std': std,
        'se': std / np.sqrt(len(finite)) if len(finite) > 1 else np.nan,
        'median': float(np.median(finite)),
        'q25': float(np.quantile(finite, 0.25)),
        'q75': float(np.quantile(finite, 0.75)),
    }


def build_summary_rows(
        geometry: dict[str, list[dict[str, float]]],
        de: dict[str, dict[str, np.ndarray]],
        mc: dict[str, dict[str, np.ndarray]], prediction: dict[str, np.ndarray],
        settings: list[dict[str, float]], n: int, n_trials: int
        ) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for family, map_rows in geometry.items():
        for map_index, map_row in enumerate(map_rows):
            for ratio_index, setting in enumerate(settings):
                row: dict[str, object] = {
                    'family': family,
                    **map_row,
                    'n': n,
                    'n_trials': n_trials,
                    'noise_signal_ratio': setting['noise_signal_ratio'],
                    'sigma': setting['sigma'],
                    'de_alpha_cv': setting['alpha_cv'],
                    'de_kappa': setting['kappa'],
                }
                for name, array in de[family].items():
                    row[f'de_{name}'] = float(array[map_index, ratio_index])
                for name, array in mc[family].items():
                    summary = _summaries(array[map_index, ratio_index])
                    for statistic, value in summary.items():
                        key = f'mc_{name}' if statistic == 'mean' else f'mc_{name}_{statistic}'
                        row[key] = value
                for name, array in prediction.items():
                    summary = _summaries(array[ratio_index])
                    for statistic, value in summary.items():
                        key = f'mc_{name}' if statistic == 'mean' else f'mc_{name}_{statistic}'
                        row[key] = value
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
        for raw in csv.DictReader(handle):
            row: dict[str, object] = {}
            for key, value in raw.items():
                try:
                    row[key] = float(value)
                except (TypeError, ValueError):
                    row[key] = value
            rows.append(row)
    return rows


def plot_plane(
        eigenvalues: np.ndarray, plane: dict[str, np.ndarray | float]) -> None:
    ranks = np.arange(1, len(eigenvalues) + 1)
    u_high = np.asarray(plane['u_high'])
    u_low = np.asarray(plane['u_low'])
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.2))
    axes[0].loglog(ranks, eigenvalues, color='0.22', lw=1.5)
    axes[0].axvspan(
        float(plane['high_band_start_rank']),
        float(plane['high_band_stop_rank']), color='C0', alpha=0.16,
        label='high-variance search band')
    axes[0].axvspan(
        float(plane['low_band_start_rank']),
        float(plane['low_band_stop_rank']), color='C1', alpha=0.16,
        label='low-variance search band')
    axes[0].scatter(
        [plane['high_dominant_pc_rank'], plane['low_dominant_pc_rank']],
        [plane['high_effective_input_eigenvalue'],
         plane['low_effective_input_eigenvalue']],
        color=['C0', 'C1'], edgecolor='black', linewidth=0.5, zorder=4)
    axes[0].set_xlabel('Van Hateren PC rank')
    axes[0].set_ylabel(r'input eigenvalue $s_j$')
    axes[0].set_title('Rotation-plane selection')
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.16, which='both')

    for vector, color, label in (
            (u_high, 'C0', 'high-variance null direction'),
            (u_low, 'C1', 'low-variance null direction')):
        active = np.flatnonzero(vector ** 2 > 1e-10)
        axes[1].scatter(
            active + 1, vector[active] ** 2, s=12, color=color,
            alpha=0.75, label=label)
    axes[1].set_xscale('log')
    axes[1].set_yscale('log')
    axes[1].set_xlabel('PC rank')
    axes[1].set_ylabel('squared plane-vector loading')
    axes[1].set_title(r'Both directions lie in $b^\perp\cap(\Sigma^{-1}b)^\perp$')
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.16, which='both')
    fig.suptitle('Van Hateren circular teacher: null Givens plane', fontsize=14)
    fig.tight_layout()
    PLANE_FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLANE_FIGURE_PATH, dpi=190, bbox_inches='tight')
    PLANE_FIGURE_PATH.chmod(0o644)
    plt.close(fig)


def _selected(rows: list[dict[str, object]], family: str, ratio: float):
    return sorted(
        [row for row in rows
         if row['family'] == family and np.isclose(
             float(row['noise_signal_ratio']), ratio)],
        key=lambda row: float(row['map_index']))


def plot_givens(rows: list[dict[str, object]]) -> None:
    ratios = sorted({
        float(row['noise_signal_ratio']) for row in rows
        if row['family'] == 'givens'})
    colors = plt.get_cmap('viridis')(
        np.linspace(0.08, 0.90, len(ratios)))
    base = _selected(rows, 'givens', ratios[0])
    x = np.asarray([float(row['low_fraction']) for row in base])
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.0), sharex=True)
    axes[0, 0].plot(
        x, [float(row['trace_g']) for row in base], color='0.15', lw=2.2)
    axes[0, 0].set_yscale('log')
    axes[0, 0].set_ylabel(r'$\operatorname{tr}(FF^\top)$')
    axes[0, 0].set_title('Pixel-metric inflation')

    specs = (
        (axes[0, 1], 'acc_error_normalized', r'$E_{\rm acc}/S$', 'log'),
        (axes[1, 0], 'r2_acc', r'$R^2_{\rm acc}$', 'symlog'),
        (axes[1, 1], 'slope_acc', r'slope$_{\rm acc}$', 'symlog'),
    )
    for color, ratio in zip(colors, ratios):
        selected = _selected(rows, 'givens', ratio)
        for ax, metric, ylabel, scale in specs:
            de = np.asarray([float(row[f'de_{metric}']) for row in selected])
            mc = np.asarray([
                float(row[f'mc_{metric}_median']
                      if metric == 'r2_acc' else row[f'mc_{metric}'])
                for row in selected])
            ax.plot(x, de, color=color, lw=2.0, label=rf'$\sigma^2/S={ratio:g}$')
            marker_indices = np.arange(0, len(x), max(1, len(x) // 10))
            ax.plot(
                x[marker_indices], mc[marker_indices], 'o', color=color,
                mec='black', mew=0.35, ms=3.6, alpha=0.72)
            ax.set_ylabel(ylabel)
            if scale == 'log':
                ax.set_yscale('log')
            else:
                ax.set_yscale('symlog', linthresh=0.1)
                ax.axhline(0, color='0.55', lw=0.7, ls='--')
            if metric == 'slope_acc':
                ax.axhline(1, color='0.55', lw=0.7, ls=':')
    axes[0, 1].legend(fontsize=8, ncol=2)
    for ax in axes.ravel():
        ax.grid(alpha=0.16)
    for ax in axes[1]:
        ax.set_xlabel(r'low-variance fraction $\sin^2\theta$')
    fig.suptitle(
        'Van Hateren disk teacher: prediction-equivalent Givens rotation',
        fontsize=14)
    fig.text(
        0.5, 0.012,
        'lines: deterministic equivalent; markers: shared-design Monte Carlo; '
        r'$F\Sigma F^\top$, $F\Sigma\beta^\star$, and $F\beta^\star$ fixed',
        ha='center', fontsize=8.5, color='0.3')
    fig.tight_layout(rect=(0, 0.045, 1, 0.96))
    GIVENS_FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(GIVENS_FIGURE_PATH, dpi=190, bbox_inches='tight')
    GIVENS_FIGURE_PATH.chmod(0o644)
    plt.close(fig)


def plot_random(rows: list[dict[str, object]]) -> None:
    ratios = sorted({
        float(row['noise_signal_ratio']) for row in rows
        if row['family'] == 'random'})
    colors = plt.get_cmap('viridis')(
        np.linspace(0.08, 0.90, len(ratios)))
    base = _selected(rows, 'random', ratios[0])
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.0), sharex=True)
    trace = np.asarray([float(row['trace_g']) for row in base])
    axes[0, 0].scatter(
        trace, [float(row['q_null_norm']) for row in base],
        c='0.22', s=18, alpha=0.65)
    axes[0, 0].set_ylabel(r'$\|(F\beta^\star)_{\rm null}\|$')
    axes[0, 0].set_title('Random rotations also vary teacher overlap')
    specs = (
        (axes[0, 1], 'acc_error_normalized', r'$E_{\rm acc}/S$', 'log'),
        (axes[1, 0], 'r2_acc', r'$R^2_{\rm acc}$', 'symlog'),
        (axes[1, 1], 'slope_acc', r'slope$_{\rm acc}$', 'symlog'),
    )
    for color, ratio in zip(colors, ratios):
        selected = _selected(rows, 'random', ratio)
        x = np.asarray([float(row['trace_g']) for row in selected])
        for ax, metric, ylabel, scale in specs:
            de = np.asarray([float(row[f'de_{metric}']) for row in selected])
            mc = np.asarray([
                float(row[f'mc_{metric}_median']
                      if metric == 'r2_acc' else row[f'mc_{metric}'])
                for row in selected])
            ax.scatter(
                x, de, color=color, s=18, alpha=0.48,
                label=rf'$\sigma^2/S={ratio:g}$')
            ax.scatter(
                x, mc, facecolors='none', edgecolors=color,
                marker='o', s=18, linewidths=0.7, alpha=0.58)
            ax.set_ylabel(ylabel)
            if scale == 'log':
                ax.set_yscale('log')
            else:
                ax.set_yscale('symlog', linthresh=0.1)
                ax.axhline(0, color='0.55', lw=0.7, ls='--')
            if metric == 'slope_acc':
                ax.axhline(1, color='0.55', lw=0.7, ls=':')
    axes[0, 1].legend(fontsize=8, ncol=2)
    for ax in axes.ravel():
        ax.set_xscale('log')
        ax.grid(alpha=0.16)
    for ax in axes[1]:
        ax.set_xlabel(r'$\operatorname{tr}(FF^\top)$')
    axes[0, 0].set_xlabel(r'$\operatorname{tr}(FF^\top)$')
    fig.suptitle(
        'Van Hateren disk teacher: random prediction-null feature rotations',
        fontsize=14)
    fig.text(
        0.5, 0.012,
        'filled: deterministic equivalent; open: shared-design Monte Carlo; '
        r'all maps have identical $R^2_{\rm gen}$ and RidgeCV path',
        ha='center', fontsize=8.5, color='0.3')
    fig.tight_layout(rect=(0, 0.045, 1, 0.96))
    RANDOM_FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(RANDOM_FIGURE_PATH, dpi=190, bbox_inches='tight')
    RANDOM_FIGURE_PATH.chmod(0o644)
    plt.close(fig)


def render_all(rows: list[dict[str, object]], eigenvalues: np.ndarray,
               plane: dict[str, np.ndarray | float]) -> None:
    plot_plane(eigenvalues, plane)
    plot_givens(rows)
    plot_random(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=1000)
    parser.add_argument('--p', type=int, default=16)
    parser.add_argument('--n-angles', type=int, default=41)
    parser.add_argument('--n-random', type=int, default=160)
    parser.add_argument('--random-pool-size', type=int, default=256)
    parser.add_argument('--n-trials', type=int, default=100)
    parser.add_argument('--seed', type=int, default=20260828)
    parser.add_argument('--alpha-min-exp', type=float, default=-6.0)
    parser.add_argument('--alpha-max-exp', type=float, default=8.0)
    parser.add_argument('--alpha-grid-size', type=int, default=281)
    parser.add_argument('--benchmark-trials', type=int, default=2)
    parser.add_argument('--benchmark-only', action='store_true')
    parser.add_argument('--plot-only', action='store_true')
    parser.add_argument('--raw-path', type=Path, default=RAW_PATH)
    parser.add_argument('--no-progress', action='store_true')
    parser.add_argument('--log-file', type=Path, default=DEFAULT_LOG_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = configure_logger(args.log_file)
    raw_path = require_bulk_path(
        args.raw_path, 'Van Hateren null-rotation raw cache')
    with np.load(SPECTRUM_PATH) as data:
        eigenvalues = np.asarray(data['eigenvalues'], dtype=float)
        beta = np.asarray(data['beta_proj'], dtype=float)
    signal = float(np.sum(eigenvalues * beta ** 2))
    ratios = DEFAULT_RATIOS.copy()
    alphas = np.logspace(
        args.alpha_min_exp, args.alpha_max_exp, args.alpha_grid_size)

    if args.plot_only:
        rows = read_rows(SUMMARY_PATH)
        with np.load(raw_path) as raw:
            plane = {
                key.removeprefix('plane_'): np.asarray(raw[key])
                for key in raw.files if key.startswith('plane_')}
        render_all(rows, eigenvalues, plane)
        logger.info('Replotted null-rotation figures from cached summaries')
        return

    givens_maps, givens_geometry, plane = build_givens_family(
        eigenvalues, beta, args.p, args.n_angles, args.seed + 11)
    random_maps, random_geometry, random_pool, random_diagnostics = (
        build_random_family(
            eigenvalues, beta, args.p, args.n_random,
            args.random_pool_size, args.seed + 29))
    logger.info(
        'Givens plane: high rank %.0f lambda_eff=%.4g; '
        'low rank %.0f lambda_eff=%.4g; inverse-variance ratio %.3g',
        plane['high_dominant_pc_rank'],
        plane['high_effective_input_eigenvalue'],
        plane['low_dominant_pc_rank'],
        plane['low_effective_input_eigenvalue'],
        float(plane['low_inverse_variance_rayleigh']) /
        float(plane['high_inverse_variance_rayleigh']))
    logger.info(
        'Invariance errors: Givens C %.2e, h %.2e, q %.2e; random C %.2e, h %.2e',
        plane['max_c_invariance_error'], plane['max_h_invariance_error'],
        plane['max_q_invariance_error'],
        random_diagnostics['max_c_invariance_error'],
        random_diagnostics['max_h_invariance_error'])

    maps = {'givens': givens_maps, 'random': random_maps}
    pilot_trials = max(1, min(args.benchmark_trials, args.n_trials))
    _, _, pilot_elapsed = simulate_common_ridgecv(
        {key: value[:min(8, len(value))] for key, value in maps.items()},
        signal, ratios, alphas, args.n, pilot_trials, args.seed + 101,
        progress=False)
    projected = pilot_elapsed * args.n_trials / pilot_trials
    logger.info(
        'Small-scale timing: %d shared designs in %.2fs; projected %.1fs '
        '(%.1fmin) for %d trials, %d Givens maps, %d random maps, %d noises',
        pilot_trials, pilot_elapsed, projected, projected / 60,
        args.n_trials, args.n_angles, args.n_random, len(ratios))
    if args.benchmark_only:
        return

    settings = common_de_settings(args.p, signal, ratios, args.n, alphas)
    de = {
        family: de_metrics_for_maps(value, signal, settings, args.n)
        for family, value in maps.items()}
    mc, prediction, elapsed = simulate_common_ridgecv(
        maps, signal, ratios, alphas, args.n, args.n_trials,
        args.seed, progress=not args.no_progress)
    logger.info('Full shared-design Monte Carlo completed in %.1fs', elapsed)

    geometry = {'givens': givens_geometry, 'random': random_geometry}
    rows = build_summary_rows(
        geometry, de, mc, prediction, settings, args.n, args.n_trials)
    write_rows(SUMMARY_PATH, rows)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        raw_path,
        eigenvalues=eigenvalues,
        beta_proj=beta,
        signal=signal,
        ratios=ratios,
        alphas=alphas,
        random_pool=random_pool,
        givens_maps=givens_maps,
        random_maps=random_maps,
        **{f'plane_{key}': value for key, value in plane.items()},
        **{f'prediction_{key}': value for key, value in prediction.items()},
        **{
            f'{family}_mc_{metric}': value
            for family, metrics in mc.items()
            for metric, value in metrics.items()},
        **{
            f'{family}_de_{metric}': value
            for family, metrics in de.items()
            for metric, value in metrics.items()},
    )
    raw_path.chmod(0o644)
    render_all(rows, eigenvalues, plane)
    logger.info('Plot-ready summary: %s', SUMMARY_PATH)
    logger.info('Raw map/trial cache: %s', raw_path)
    logger.info(
        'Figures: %s, %s, %s',
        PLANE_FIGURE_PATH, GIVENS_FIGURE_PATH, RANDOM_FIGURE_PATH)


if __name__ == '__main__':
    main()

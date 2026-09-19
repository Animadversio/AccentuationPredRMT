"""Sweep teacher alignment through a power-law population eigenspectrum.

The localized teachers control the per-PC natural signal allocation
``lambda_k * beta_k**2`` and keep its sum equal to one. Ridge penalties are
selected independently by actual K-fold CV in Monte Carlo and by the
fold-training-size deterministic equivalent in theory.

The script caches raw cases, plot-ready metric tables, and plot-ready
coefficient summaries. Use ``--plot-only`` to change figure styling without
repeating simulation.
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
sys.path.insert(0, str(REPO_ROOT))

from rmt_core import (  # noqa: E402
    SpectrumKappa,
    make_spectral_teacher,
    powerlaw_spectrum,
    run_paired_cv_monte_carlo,
)
from scripts.validate_cv_selected_r2 import (  # noqa: E402
    final_r2_theory,
    lambda_distribution_summary,
    select_de_lambda,
)
from scripts.validate_r2_peer_review import (  # noqa: E402
    configure_logging,
    float_tag,
)
from scripts.storage_paths import configured_bulk_path, require_bulk_path


SUMMARY_PATH = REPO_ROOT / 'tables' / 'powerlaw_teacher_alignment_summary.csv'
COEFFICIENT_PATH = REPO_ROOT / 'tables' / 'powerlaw_teacher_weight_summary.csv'
CASE_DIR = configured_bulk_path('tables/powerlaw_teacher_alignment_cases')
FIGURE_DIR = REPO_ROOT / 'figures' / 'teacher_alignment'
R2_FIGURE_PATH = FIGURE_DIR / 'powerlaw_teacher_alignment_r2.png'
LAMBDA_FIGURE_PATH = FIGURE_DIR / 'powerlaw_teacher_alignment_lambda.png'
PROFILE_FIGURE_PATH = FIGURE_DIR / 'powerlaw_teacher_signal_profiles.png'
WEIGHT_FIGURE_PATH = FIGURE_DIR / 'powerlaw_teacher_weights_eigenbasis.png'
SCALED_WEIGHT_FIGURE_PATH = (
    FIGURE_DIR / 'powerlaw_teacher_response_weights_eigenbasis.png')
DEFAULT_LOG_PATH = REPO_ROOT / 'logs' / 'powerlaw_teacher_alignment.log'


def profile_specs(centers: list[float]) -> list[dict[str, object]]:
    specs = [
        {
            'profile_id': f'localized_{center:.3f}',
            'profile_type': 'localized',
            'requested_center': float(center),
            'profile_label': f'localized {center:.2f}',
        }
        for center in centers
    ]
    specs.append({
        'profile_id': 'random',
        'profile_type': 'random',
        'requested_center': np.nan,
        'profile_label': 'i.i.d. random',
    })
    return specs


def case_path(args: argparse.Namespace, profile_id: str, sigma: float,
              mc_seed: int) -> Path:
    tag = (
        f'powerlaw{float_tag(args.alpha)}_{profile_id}_bw{float_tag(args.bandwidth)}'
        f'_d{args.d}_n{args.n}_folds{args.n_folds}_sigma{float_tag(sigma)}'
        f'_grid{args.n_lambda}_{float_tag(args.lambda_min)}-'
        f'{float_tag(args.lambda_max)}_trials{args.n_trials}_seed{mc_seed}.npz'
    )
    return require_bulk_path(
        CASE_DIR, 'power-law teacher-alignment case caches') / tag


def save_case(path: Path, row: dict[str, object],
              trials: dict[str, np.ndarray], coefficients: np.ndarray,
              beta_proj: np.ndarray, signal_allocation: np.ndarray,
              eigenvalues: np.ndarray, lambda_grid: np.ndarray,
              cv_risk: np.ndarray, full_risk: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {f'row_{key}': np.asarray(value) for key, value in row.items()}
    payload.update({f'trial_{key}': value for key, value in trials.items()})
    payload.update({
        'coefficient_trials': coefficients,
        'beta_proj': beta_proj,
        'signal_allocation': signal_allocation,
        'eigenvalues': eigenvalues,
        'lambda_grid': lambda_grid,
        'theory_cv_risk_path': cv_risk,
        'theory_full_risk_path': full_risk,
    })
    np.savez_compressed(path, **payload)


def load_case(path: Path) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as data:
        row = {
            key.removeprefix('row_'): (
                data[key].item() if data[key].ndim == 0 else data[key].tolist())
            for key in data.files if key.startswith('row_')
        }
        arrays = {
            key: np.asarray(data[key]) for key in (
                'coefficient_trials', 'beta_proj', 'signal_allocation',
                'eigenvalues')
        }
    return row, arrays


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    path.chmod(0o644)


def read_csv(path: Path) -> list[dict[str, object]]:
    with path.open(newline='') as handle:
        rows: list[dict[str, object]] = list(csv.DictReader(handle))
    string_keys = {'profile_id', 'profile_type', 'profile_label'}
    for row in rows:
        for key, value in list(row.items()):
            if key in string_keys:
                continue
            try:
                row[key] = float(value)
            except (TypeError, ValueError):
                pass
    return rows


def coefficient_rows(row: dict[str, object], arrays: dict[str, np.ndarray],
                     eigenvectors: np.ndarray) -> list[dict[str, object]]:
    coefficients = arrays['coefficient_trials']
    pooled = coefficients.reshape(-1, coefficients.shape[-1]) @ eigenvectors
    beta_proj = arrays['beta_proj']
    eigenvalues = arrays['eigenvalues']
    signs = np.where(beta_proj >= 0, 1.0, -1.0)
    aligned = pooled * signs[None, :]

    kappa = SpectrumKappa(
        eigenvalues, len(eigenvalues) / float(row['n']))(
            float(row['theory_lambda_cv']))
    de_mean = eigenvalues / (eigenvalues + kappa) * np.abs(beta_proj)
    mean = np.mean(aligned, axis=0)
    q10, q90 = np.quantile(aligned, [0.1, 0.9], axis=0)
    std = np.std(aligned, axis=0, ddof=1)

    rows = []
    for index in range(len(eigenvalues)):
        rows.append({
            'profile_id': row['profile_id'],
            'profile_type': row['profile_type'],
            'profile_label': row['profile_label'],
            'requested_center': row['requested_center'],
            'signal_rank_centroid': row['signal_rank_centroid'],
            'sigma': row['sigma'],
            'pc_index': index + 1,
            'rank_fraction': (index + 0.5) / len(eigenvalues),
            'eigenvalue': eigenvalues[index],
            'teacher_coefficient': beta_proj[index],
            'teacher_aligned': abs(beta_proj[index]),
            'signal_allocation': arrays['signal_allocation'][index],
            'mc_mean_aligned': mean[index],
            'mc_std_aligned': std[index],
            'mc_q10_aligned': q10[index],
            'mc_q90_aligned': q90[index],
            'theory_mean_aligned': de_mean[index],
            'theory_lambda_cv': row['theory_lambda_cv'],
            'mc_lambda_median': row['mc_lambda_median'],
        })
    return rows


def _localized_and_random(rows: list[dict[str, object]], sigma: float):
    subset = [row for row in rows if np.isclose(float(row['sigma']), sigma)]
    localized = sorted(
        (row for row in subset if row['profile_type'] == 'localized'),
        key=lambda row: float(row['signal_rank_centroid']))
    random = next(row for row in subset if row['profile_type'] == 'random')
    return localized, random


def plot_r2(rows: list[dict[str, object]]) -> None:
    sigmas = sorted({float(row['sigma']) for row in rows})
    metrics = [
        ('r2_gen', 'theory_r2_gen', None, r'$R^2_{gen}$'),
        ('r2_acc', 'theory_r2_acc', 'theory_r2_acc_delta', r'$R^2_{acc}$'),
        ('r2_peer', 'theory_r2_peer', 'theory_r2_peer_delta', r'$R^2_{peer}$'),
    ]
    fig, axes = plt.subplots(
        3, len(sigmas), figsize=(4.1 * len(sigmas), 9.5), squeeze=False,
        sharex=True)
    for column, sigma in enumerate(sigmas):
        localized, random = _localized_and_random(rows, sigma)
        x = np.asarray([float(row['signal_rank_centroid']) for row in localized])
        for row_index, (mc_key, theory_key, corrected_key, ylabel) in enumerate(metrics):
            ax = axes[row_index, column]
            mc = np.asarray([float(row[f'mc_{mc_key}']) for row in localized])
            se = np.asarray([float(row[f'mc_{mc_key}_se']) for row in localized])
            theory_plot_key = corrected_key or theory_key
            theory = np.asarray([float(row[theory_plot_key]) for row in localized])
            ax.plot(x, theory, '-', color='C0', lw=1.8,
                    label='DE at DE-CV lambda')
            ax.errorbar(x, mc, yerr=2 * se, fmt='o', ms=4, color='black',
                        capsize=2, label='actual CV, mean +/- 2 SE')
            xr = float(random['signal_rank_centroid'])
            ax.plot(xr, float(random[theory_plot_key]), marker='D', ms=6,
                    markerfacecolor='none', markeredgecolor='C3', ls='none',
                    label='random teacher, DE')
            ax.plot(xr, float(random[f'mc_{mc_key}']), marker='x', ms=7,
                    color='C3', ls='none', label='random teacher, actual CV')
            ax.axhline(1.0, color='0.82', ls=':', lw=0.8)
            ax.axhline(0.0, color='0.88', ls=':', lw=0.8)
            ax.grid(alpha=0.2)
            ax.set_xlim(0, 1)
            if row_index == 0:
                ax.set_title(rf'noise $\sigma={sigma:g}$')
            if column == 0:
                ax.set_ylabel(ylabel)
            if row_index == 2:
                ax.set_xlabel('teacher signal centroid in PC rank\n(top 0 -> bottom 1)')
            if row_index == 0 and column == 0:
                ax.legend(fontsize=7, loc='best')
    fig.suptitle('Power-law data: CV ridge performance across teacher spectral alignment')
    fig.tight_layout()
    R2_FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(R2_FIGURE_PATH, dpi=180, bbox_inches='tight')
    R2_FIGURE_PATH.chmod(0o644)
    plt.close(fig)


def plot_lambdas(rows: list[dict[str, object]]) -> None:
    sigmas = sorted({float(row['sigma']) for row in rows})
    fig, axes = plt.subplots(
        1, len(sigmas), figsize=(4.1 * len(sigmas), 3.8), squeeze=False,
        sharex=True, sharey=True)
    for column, sigma in enumerate(sigmas):
        ax = axes[0, column]
        localized, random = _localized_and_random(rows, sigma)
        x = np.asarray([float(row['signal_rank_centroid']) for row in localized])
        median = np.asarray([float(row['mc_lambda_median']) for row in localized])
        q25 = np.asarray([float(row['mc_lambda_q25']) for row in localized])
        q75 = np.asarray([float(row['mc_lambda_q75']) for row in localized])
        de = np.asarray([float(row['theory_lambda_cv']) for row in localized])
        ax.fill_between(x, q25, q75, color='black', alpha=0.14,
                        label='actual CV IQR')
        ax.plot(x, median, 'o-', color='black', ms=4,
                label='actual CV median')
        ax.plot(x, de, 's-', color='C0', ms=4, label='DE-CV lambda')
        xr = float(random['signal_rank_centroid'])
        ax.plot(xr, float(random['mc_lambda_median']), 'x', color='C3', ms=7,
                label='random teacher actual')
        ax.plot(xr, float(random['theory_lambda_cv']), marker='D', ls='none',
                markerfacecolor='none', markeredgecolor='C3', ms=6,
                label='random teacher DE')
        ax.set_yscale('log')
        ax.set_xlim(0, 1)
        ax.grid(alpha=0.2)
        ax.set_title(rf'noise $\sigma={sigma:g}$')
        ax.set_xlabel('teacher signal centroid\n(top 0 -> bottom 1)')
        if column == 0:
            ax.set_ylabel(r'cross-validated $\lambda$')
            ax.legend(fontsize=7)
    fig.suptitle('CV regularization changes with teacher spectral alignment')
    fig.tight_layout()
    LAMBDA_FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(LAMBDA_FIGURE_PATH, dpi=180, bbox_inches='tight')
    LAMBDA_FIGURE_PATH.chmod(0o644)
    plt.close(fig)


def _representative_profiles(rows: list[dict[str, object]]) -> list[tuple[str, str]]:
    one_sigma = min(float(row['sigma']) for row in rows)
    localized, random = _localized_and_random(rows, one_sigma)
    targets = [('Top', 0.05), ('Middle', 0.5), ('Bottom', 0.95)]
    selected = []
    for label, target in targets:
        row = min(localized, key=lambda item: abs(
            float(item['signal_rank_centroid']) - target))
        selected.append((label, str(row['profile_id'])))
    selected.append(('Random', str(random['profile_id'])))
    return selected


def plot_profiles(coefficient_data: list[dict[str, object]]) -> None:
    profiles = _representative_profiles(coefficient_data)
    sigma = min(float(row['sigma']) for row in coefficient_data)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.1))
    for label, profile_id in profiles:
        subset = sorted(
            (row for row in coefficient_data
             if row['profile_id'] == profile_id
             and np.isclose(float(row['sigma']), sigma)),
            key=lambda row: float(row['pc_index']))
        x = np.asarray([float(row['rank_fraction']) for row in subset])
        allocation = np.asarray([float(row['signal_allocation']) for row in subset])
        teacher = np.asarray([float(row['teacher_aligned']) for row in subset])
        axes[0].plot(x, allocation, lw=1.6, label=label)
        axes[1].plot(x, teacher, lw=1.3, label=label)
    axes[0].set_ylabel(r'natural signal allocation $\lambda_k(\beta_k^*)^2$')
    axes[1].set_ylabel(r'teacher coefficient magnitude $|\beta_k^*|$')
    axes[1].set_yscale('log')
    for ax in axes:
        ax.set_xlabel('population PC rank (top 0 -> bottom 1)')
        ax.set_xlim(0, 1)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    axes[0].set_title('Controlled response-variance placement')
    axes[1].set_title('Raw weights needed for unit signal power')
    fig.tight_layout()
    PROFILE_FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PROFILE_FIGURE_PATH, dpi=180, bbox_inches='tight')
    PROFILE_FIGURE_PATH.chmod(0o644)
    plt.close(fig)


def plot_weights(coefficient_data: list[dict[str, object]], *,
                 response_scaled: bool) -> None:
    profiles = _representative_profiles(coefficient_data)
    sigmas = sorted({float(row['sigma']) for row in coefficient_data})
    fig, axes = plt.subplots(
        len(profiles), len(sigmas),
        figsize=(4.0 * len(sigmas), 2.75 * len(profiles)), squeeze=False,
        sharex=True)
    for row_index, (label, profile_id) in enumerate(profiles):
        for column, sigma in enumerate(sigmas):
            ax = axes[row_index, column]
            subset = sorted(
                (row for row in coefficient_data
                 if row['profile_id'] == profile_id
                 and np.isclose(float(row['sigma']), sigma)),
                key=lambda row: float(row['pc_index']))
            x = np.asarray([float(row['rank_fraction']) for row in subset])
            scale = (np.sqrt(np.asarray(
                [float(row['eigenvalue']) for row in subset]))
                     if response_scaled else np.ones(len(subset)))
            teacher = scale * np.asarray(
                [float(row['teacher_aligned']) for row in subset])
            mc = scale * np.asarray(
                [float(row['mc_mean_aligned']) for row in subset])
            q10 = scale * np.asarray(
                [float(row['mc_q10_aligned']) for row in subset])
            q90 = scale * np.asarray(
                [float(row['mc_q90_aligned']) for row in subset])
            theory = scale * np.asarray(
                [float(row['theory_mean_aligned']) for row in subset])
            ax.fill_between(x, q10, q90, color='C0', alpha=0.16,
                            label='actual CV 10-90%')
            ax.plot(x, teacher, color='black', lw=1.3, label='teacher')
            ax.plot(x, mc, color='C0', lw=1.3, label='actual CV mean')
            ax.plot(x, theory, '--', color='C3', lw=1.2, label='DE mean')
            max_teacher = max(float(np.max(np.abs(teacher))), 1e-8)
            ax.set_yscale('symlog', linthresh=max_teacher * 1e-3)
            ax.grid(alpha=0.18)
            ax.set_xlim(0, 1)
            lambda_mc = float(subset[0]['mc_lambda_median'])
            lambda_de = float(subset[0]['theory_lambda_cv'])
            ax.text(0.98, 0.96,
                    rf'$\lambda_{{CV}}$ MC/DE {lambda_mc:.3g}/{lambda_de:.3g}',
                    transform=ax.transAxes, ha='right', va='top', fontsize=7)
            if row_index == 0:
                ax.set_title(rf'noise $\sigma={sigma:g}$')
            if column == 0:
                suffix = r'$\sqrt{\lambda_k}\,\beta_k$' if response_scaled else r'$\beta_k$'
                ax.set_ylabel(f'{label}\nsign-aligned {suffix}')
            if row_index == len(profiles) - 1:
                ax.set_xlabel('PC rank (top 0 -> bottom 1)')
            if row_index == 0 and column == 0:
                ax.legend(fontsize=7, loc='lower left')
    title = (
        'Teacher and CV-ridge weights in the data eigenbasis '
        '(scaled by input standard deviation)'
        if response_scaled else
        'Teacher and CV-ridge raw weights in the data eigenbasis')
    fig.suptitle(title)
    fig.tight_layout()
    path = SCALED_WEIGHT_FIGURE_PATH if response_scaled else WEIGHT_FIGURE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches='tight')
    path.chmod(0o644)
    plt.close(fig)


def render_all(rows: list[dict[str, object]],
               coefficient_data: list[dict[str, object]]) -> None:
    plot_r2(rows)
    plot_lambdas(rows)
    plot_profiles(coefficient_data)
    plot_weights(coefficient_data, response_scaled=False)
    plot_weights(coefficient_data, response_scaled=True)


def benchmark(args: argparse.Namespace, logger: logging.Logger,
              lambda_grid: np.ndarray,
              specs: list[dict[str, object]]) -> float:
    eigenvalues, eigenvectors = powerlaw_spectrum(args.d, alpha=args.alpha)
    spec = specs[0]
    beta_star, _, _ = make_spectral_teacher(
        eigenvalues, eigenvectors, profile=str(spec['profile_type']),
        center=float(spec['requested_center']), bandwidth=args.bandwidth,
        seed=args.seed)
    result = run_paired_cv_monte_carlo(
        args.n, eigenvalues, eigenvectors, beta_star, args.sigmas[0],
        lambda_grid, n_folds=args.n_folds,
        n_trials=args.benchmark_trials, rng=np.random.default_rng(args.seed + 1))
    seconds_per_trial = result['elapsed_seconds'] / args.benchmark_trials
    total_cases = len(specs) * len(args.sigmas)
    projected = seconds_per_trial * args.n_trials * total_cases
    logger.info(
        'Benchmark: %.5f s/paired CV trial; projected %.1fs (%.1fmin) for %d cases x %d trials',
        seconds_per_trial, projected, projected / 60, total_cases, args.n_trials)
    return projected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--d', type=int, default=128)
    parser.add_argument('--n', type=int, default=256)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--centers', type=float, nargs='+',
                        default=[0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95])
    parser.add_argument('--bandwidth', type=float, default=0.10)
    parser.add_argument('--sigmas', type=float, nargs='+',
                        default=[0.1, 0.3, 0.6, 1.0])
    parser.add_argument('--lambda-min', type=float, default=1e-4)
    parser.add_argument('--lambda-max', type=float, default=10.0)
    parser.add_argument('--n-lambda', type=int, default=61)
    parser.add_argument('--n-trials', type=int, default=200)
    parser.add_argument('--benchmark-trials', type=int, default=3)
    parser.add_argument('--seed', type=int, default=20260814)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--plot-only', action='store_true')
    parser.add_argument('--no-progress', action='store_true')
    parser.add_argument('--log-file', type=Path, default=DEFAULT_LOG_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = configure_logging(args.log_file)
    if args.plot_only:
        rows = read_csv(SUMMARY_PATH)
        coefficient_data = read_csv(COEFFICIENT_PATH)
        render_all(rows, coefficient_data)
        logger.info('Replotted five figures from %s and %s',
                    SUMMARY_PATH, COEFFICIENT_PATH)
        return

    specs = profile_specs(args.centers)
    lambda_grid = np.geomspace(args.lambda_min, args.lambda_max, args.n_lambda)
    projected = benchmark(args, logger, lambda_grid, specs)
    if projected > 15 * 60:
        logger.warning('Projected runtime exceeds 15 minutes; profile before scaling.')

    eigenvalues, eigenvectors = powerlaw_spectrum(args.d, alpha=args.alpha)
    n_cv_train = args.n * (args.n_folds - 1) / args.n_folds
    rows: list[dict[str, object]] = []
    all_coefficient_rows: list[dict[str, object]] = []
    total_cases = len(specs) * len(args.sigmas)
    case_index = 0
    start = time.perf_counter()

    for profile_index, spec in enumerate(specs):
        requested_center = float(spec['requested_center'])
        beta_star, beta_proj, signal_allocation = make_spectral_teacher(
            eigenvalues, eigenvectors, profile=str(spec['profile_type']),
            center=(requested_center if np.isfinite(requested_center) else 0.5),
            bandwidth=args.bandwidth, seed=args.seed)
        rank_fraction = (np.arange(args.d, dtype=float) + 0.5) / args.d
        signal_centroid = float(np.sum(rank_fraction * signal_allocation))

        for sigma_index, sigma in enumerate(args.sigmas):
            case_index += 1
            mc_seed = args.seed + 1000 * profile_index + 10 * sigma_index + 1
            cache = case_path(args, str(spec['profile_id']), sigma, mc_seed)
            if cache.exists() and not args.force:
                logger.info('[%d/%d] Loading %s', case_index, total_cases, cache)
                row, arrays = load_case(cache)
                rows.append(row)
                all_coefficient_rows.extend(
                    coefficient_rows(row, arrays, eigenvectors))
                continue

            logger.info(
                '[%d/%d] %s sigma=%.3g: %d paired %d-fold CV trials',
                case_index, total_cases, spec['profile_label'], sigma,
                args.n_trials, args.n_folds)
            lambda_de_cv, cv_risk = select_de_lambda(
                eigenvalues, beta_proj, sigma, n_cv_train, lambda_grid)
            lambda_de_full, full_risk = select_de_lambda(
                eigenvalues, beta_proj, sigma, args.n, lambda_grid)
            row: dict[str, object] = {
                'profile_id': spec['profile_id'],
                'profile_type': spec['profile_type'],
                'profile_label': spec['profile_label'],
                'requested_center': requested_center,
                'signal_rank_centroid': signal_centroid,
                'bandwidth': args.bandwidth,
                'alpha': args.alpha,
                'd': args.d,
                'n': args.n,
                'n_folds': args.n_folds,
                'n_cv_train': n_cv_train,
                'sigma': sigma,
                'n_trials': args.n_trials,
                'teacher_seed': args.seed,
                'mc_seed': mc_seed,
                'lambda_min': args.lambda_min,
                'lambda_max': args.lambda_max,
                'n_lambda': args.n_lambda,
                'theory_lambda_cv': lambda_de_cv,
                'theory_lambda_full': lambda_de_full,
            }
            row.update(final_r2_theory(
                eigenvalues, beta_proj, sigma, args.n, lambda_de_cv))

            result = run_paired_cv_monte_carlo(
                args.n, eigenvalues, eigenvectors, beta_star, sigma,
                lambda_grid, n_folds=args.n_folds, n_trials=args.n_trials,
                rng=np.random.default_rng(mc_seed),
                progress=not args.no_progress, return_trials=True,
                return_coefficients=True)
            pooled_lambdas = np.concatenate([
                result['trials']['lambda_cv_a'],
                result['trials']['lambda_cv_b'],
            ])
            row.update(lambda_distribution_summary(pooled_lambdas, lambda_grid))
            row['mc_elapsed_seconds'] = result['elapsed_seconds']
            for key, value in result.items():
                if key in {'trials', 'coefficient_trials', 'elapsed_seconds',
                           'n_trials'}:
                    continue
                row[f'mc_{key}'] = value

            arrays = {
                'coefficient_trials': result['coefficient_trials'],
                'beta_proj': beta_proj,
                'signal_allocation': signal_allocation,
                'eigenvalues': eigenvalues,
            }
            save_case(
                cache, row, result['trials'], result['coefficient_trials'],
                beta_proj, signal_allocation, eigenvalues, lambda_grid,
                cv_risk, full_risk)
            rows.append(row)
            all_coefficient_rows.extend(
                coefficient_rows(row, arrays, eigenvectors))
            logger.info(
                '[%d/%d] done %.2fs | centroid %.3f | lambda DE/MC %.4g/%.4g | R2 gen %.3f/%.3f acc %.3f/%.3f peer %.3f/%.3f',
                case_index, total_cases, result['elapsed_seconds'],
                signal_centroid, lambda_de_cv, row['mc_lambda_median'],
                row['theory_r2_gen'], row['mc_r2_gen'],
                row['theory_r2_acc_delta'], row['mc_r2_acc'],
                row['theory_r2_peer_delta'], row['mc_r2_peer'])

    rows.sort(key=lambda row: (
        float(row['sigma']), str(row['profile_type']),
        float(row['signal_rank_centroid'])))
    all_coefficient_rows.sort(key=lambda row: (
        float(row['sigma']), str(row['profile_type']),
        float(row['signal_rank_centroid']), float(row['pc_index'])))
    write_csv(SUMMARY_PATH, rows)
    write_csv(COEFFICIENT_PATH, all_coefficient_rows)
    render_all(rows, all_coefficient_rows)
    logger.info('Finished in %.1fs', time.perf_counter() - start)
    logger.info('Progress log: %s', args.log_file)
    logger.info('Summary tables: %s and %s', SUMMARY_PATH, COEFFICIENT_PATH)
    logger.info(
        'Raw case caches: %s',
        require_bulk_path(
            CASE_DIR, 'power-law teacher-alignment case caches'))
    logger.info('Figures: %s, %s, %s, %s, %s',
                R2_FIGURE_PATH, LAMBDA_FIGURE_PATH, PROFILE_FIGURE_PATH,
                WEIGHT_FIGURE_PATH, SCALED_WEIGHT_FIGURE_PATH)


if __name__ == '__main__':
    main()

"""Validate DE predictions when ridge lambda is selected by K-fold CV.

For K-fold CV, the DE-selected penalty minimizes predicted generalization risk
at the fold training size n*(K-1)/K. The final R² predictions use that penalty
after refitting on all n samples. Actual Monte Carlo models independently select
lambda by K-fold validation MSE from the same candidate grid.
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
    accentuation_r2_theory,
    generalization_r2_theory,
    peer_review_r2_theory,
    ridge_error_total_theory,
    run_paired_cv_monte_carlo,
)
from scripts.validate_r2_peer_review import (  # noqa: E402
    configure_logging,
    float_tag,
    make_beta_star,
    spectrum_from_name,
)


SUMMARY_PATH = REPO_ROOT / 'tables' / 'cv_selected_r2_summary.csv'
CASE_DIR = REPO_ROOT / 'tables' / 'cv_selected_r2_cases'
R2_FIGURE_PATH = REPO_ROOT / 'figures' / 'model_selection' / 'cv_selected_r2.png'
LAMBDA_FIGURE_PATH = (
    REPO_ROOT / 'figures' / 'model_selection' / 'cv_selected_lambda.png')
DEFAULT_LOG_PATH = REPO_ROOT / 'logs' / 'cv_selected_r2.log'


def case_path(args: argparse.Namespace, spectrum: str, sigma: float,
              mc_seed: int) -> Path:
    tag = (
        f'{spectrum}_d{args.d}_n{args.n}_folds{args.n_folds}'
        f'_sigma{float_tag(sigma)}_grid{args.n_lambda}'
        f'_{float_tag(args.lambda_min)}-{float_tag(args.lambda_max)}'
        f'_trials{args.n_trials}_seed{mc_seed}.npz'
    )
    return CASE_DIR / tag


def select_de_lambda(eigenvalues: np.ndarray, beta_star: np.ndarray,
                     sigma: float, n_fit: float,
                     lambda_grid: np.ndarray) -> tuple[float, np.ndarray]:
    """Minimize the DE population risk on the supplied candidate grid."""
    gamma = len(eigenvalues) / n_fit
    kappa_solver = SpectrumKappa(eigenvalues, gamma)
    risks = np.empty(len(lambda_grid), dtype=float)
    for index, lam in enumerate(lambda_grid):
        kappa = kappa_solver(float(lam))
        risks[index] = ridge_error_total_theory(
            eigenvalues, beta_star, kappa, sigma, n_fit)
    best_index = int(np.argmin(risks))
    return float(lambda_grid[best_index]), risks


def final_r2_theory(eigenvalues: np.ndarray, beta_star: np.ndarray,
                    sigma: float, n: int, lam: float) -> dict[str, float]:
    """Predict final-refit R² values at a selected lambda."""
    kappa = SpectrumKappa(eigenvalues, len(eigenvalues) / n)(lam)
    r2_gen, _, _ = generalization_r2_theory(
        eigenvalues, beta_star, kappa, sigma, n)
    r2_acc, alignment, _ = accentuation_r2_theory(
        eigenvalues, beta_star, kappa, sigma, n)
    r2_acc_delta, _, _ = accentuation_r2_theory(
        eigenvalues, beta_star, kappa, sigma, n,
        include_delta_correction=True)
    r2_peer, peer_gain, _, _ = peer_review_r2_theory(
        eigenvalues, beta_star, kappa, sigma, n)
    r2_peer_delta, peer_gain_delta, _, _ = peer_review_r2_theory(
        eigenvalues, beta_star, kappa, sigma, n,
        include_delta_correction=True)
    return {
        'theory_kappa_final': kappa,
        'theory_r2_gen': r2_gen,
        'theory_r2_acc': r2_acc,
        'theory_r2_acc_delta': r2_acc_delta,
        'theory_acc_alignment': alignment,
        'theory_r2_peer': r2_peer,
        'theory_r2_peer_delta': r2_peer_delta,
        'theory_peer_gain': peer_gain,
        'theory_peer_gain_delta': peer_gain_delta,
    }


def lambda_distribution_summary(values: np.ndarray,
                                lambda_grid: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    unique, counts = np.unique(values, return_counts=True)
    mode = float(unique[np.argmax(counts)])
    return {
        'mc_lambda_mean': float(np.mean(values)),
        'mc_lambda_log_mean': float(np.exp(np.mean(np.log(values)))),
        'mc_lambda_median': float(np.median(values)),
        'mc_lambda_q25': float(np.quantile(values, 0.25)),
        'mc_lambda_q75': float(np.quantile(values, 0.75)),
        'mc_lambda_mode': mode,
        'mc_lambda_at_lower_boundary_fraction': float(
            np.mean(values == lambda_grid[0])),
        'mc_lambda_at_upper_boundary_fraction': float(
            np.mean(values == lambda_grid[-1])),
    }


def save_case(path: Path, row: dict[str, object],
              trials: dict[str, np.ndarray], lambda_grid: np.ndarray,
              cv_risk: np.ndarray, full_risk: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: np.asarray(value) for key, value in row.items()}
    payload.update({f'trial_{key}': value for key, value in trials.items()})
    payload['lambda_grid'] = lambda_grid
    payload['theory_cv_risk_path'] = cv_risk
    payload['theory_full_risk_path'] = full_risk
    np.savez_compressed(path, **payload)


def load_case(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as data:
        row = {}
        for key in data.files:
            if (key.startswith('trial_') or key in {
                    'lambda_grid', 'theory_cv_risk_path',
                    'theory_full_risk_path'}):
                continue
            value = data[key]
            row[key] = value.item() if value.ndim == 0 else value.tolist()
    return row


def write_summary(rows: list[dict[str, object]]) -> None:
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_PATH.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    SUMMARY_PATH.chmod(0o644)


def read_summary() -> list[dict[str, object]]:
    with SUMMARY_PATH.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key, value in list(row.items()):
            if key == 'spectrum':
                continue
            try:
                row[key] = float(value)
            except (TypeError, ValueError):
                pass
    return rows


def plot_r2(rows: list[dict[str, object]]) -> None:
    spectra = list(dict.fromkeys(str(row['spectrum']) for row in rows))
    metrics = [
        ('r2_gen', 'theory_r2_gen', None, r'$R^2_{gen}$'),
        ('r2_acc', 'theory_r2_acc', 'theory_r2_acc_delta', r'$R^2_{acc}$'),
        ('r2_peer', 'theory_r2_peer', 'theory_r2_peer_delta', r'$R^2_{peer}$'),
    ]
    fig, axes = plt.subplots(
        len(spectra), 3, figsize=(13, 3.5 * len(spectra)), squeeze=False)
    for row_index, spectrum in enumerate(spectra):
        subset = sorted(
            (row for row in rows if row['spectrum'] == spectrum),
            key=lambda row: float(row['sigma']))
        sigma = np.asarray([float(row['sigma']) for row in subset])
        for column, (mc_key, leading_key, corrected_key, title) in enumerate(metrics):
            ax = axes[row_index, column]
            mc = np.asarray([float(row[f'mc_{mc_key}']) for row in subset])
            se = np.asarray([float(row[f'mc_{mc_key}_se']) for row in subset])
            leading = np.asarray([float(row[leading_key]) for row in subset])
            ax.errorbar(sigma, mc, yerr=2 * se, fmt='o', color='black',
                        capsize=3, label='actual 5-fold CV, mean ± 2 SE')
            ax.plot(sigma, leading, '--s', color='C0', label='leading DE')
            if corrected_key:
                corrected = np.asarray(
                    [float(row[corrected_key]) for row in subset])
                ax.plot(sigma, corrected, '-^', color='C3',
                        label='DE + fluctuation correction')
            ax.axhline(1, color='0.8', ls=':', lw=0.8)
            ax.axhline(0, color='0.85', ls=':', lw=0.8)
            ax.set_title(f'{spectrum}: {title}')
            ax.set_xlabel(r'noise $σ$ (signal variance = 1)')
            ax.set_ylabel(title)
            ax.grid(alpha=0.2)
            if row_index == 0:
                ax.legend(fontsize=7)
    fig.suptitle(
        r'R² after refitting at the cross-validated $lambda$', y=1.01)
    fig.tight_layout()
    R2_FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(R2_FIGURE_PATH, dpi=170, bbox_inches='tight')
    R2_FIGURE_PATH.chmod(0o644)
    plt.close(fig)


def plot_lambdas(rows: list[dict[str, object]]) -> None:
    spectra = list(dict.fromkeys(str(row['spectrum']) for row in rows))
    fig, axes = plt.subplots(1, len(spectra), figsize=(4.5 * len(spectra), 4),
                             squeeze=False)
    for column, spectrum in enumerate(spectra):
        ax = axes[0, column]
        subset = sorted(
            (row for row in rows if row['spectrum'] == spectrum),
            key=lambda row: float(row['sigma']))
        sigma = np.asarray([float(row['sigma']) for row in subset])
        median = np.asarray([float(row['mc_lambda_median']) for row in subset])
        q25 = np.asarray([float(row['mc_lambda_q25']) for row in subset])
        q75 = np.asarray([float(row['mc_lambda_q75']) for row in subset])
        de_cv = np.asarray([float(row['theory_lambda_cv']) for row in subset])
        de_full = np.asarray([float(row['theory_lambda_full']) for row in subset])
        ax.fill_between(sigma, q25, q75, alpha=0.2, color='black',
                        label='actual CV IQR')
        ax.plot(sigma, median, 'o-', color='black', label='actual CV median')
        ax.plot(sigma, de_cv, 's-', color='C0', label='DE for fold train size')
        ax.plot(sigma, de_full, '^--', color='C2', label='DE full-n optimum')
        ax.set_yscale('log')
        ax.set_title(spectrum)
        ax.set_xlabel(r'noise $σ$')
        ax.set_ylabel(r'selected $lambda$')
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7)
    fig.suptitle(r'Actual and deterministic-equivalent CV penalty selection')
    fig.tight_layout()
    LAMBDA_FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(LAMBDA_FIGURE_PATH, dpi=170, bbox_inches='tight')
    LAMBDA_FIGURE_PATH.chmod(0o644)
    plt.close(fig)


def benchmark(args: argparse.Namespace, logger: logging.Logger,
              lambda_grid: np.ndarray) -> float:
    eigenvalues, eigenvectors = spectrum_from_name(args.spectra[0], args.d)
    beta_star = make_beta_star(eigenvalues, args.seed)
    result = run_paired_cv_monte_carlo(
        args.n, eigenvalues, eigenvectors, beta_star, args.sigmas[0],
        lambda_grid, n_folds=args.n_folds, n_trials=args.benchmark_trials,
        rng=np.random.default_rng(args.seed + 1), progress=False)
    seconds_per_trial = result['elapsed_seconds'] / args.benchmark_trials
    total_cases = len(args.spectra) * len(args.sigmas)
    projected = seconds_per_trial * args.n_trials * total_cases
    logger.info(
        'Benchmark: %.5f s/paired CV trial; projected %.1fs (%.1fmin) for %d cases × %d trials',
        seconds_per_trial, projected, projected / 60, total_cases, args.n_trials)
    return projected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--d', type=int, default=128)
    parser.add_argument('--n', type=int, default=256)
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--sigmas', type=float, nargs='+',
                        default=[0.1, 0.3, 0.6, 1.0])
    parser.add_argument(
        '--spectra', nargs='+',
        choices=['isotropic', 'powerlaw_1', 'powerlaw_1p5'],
        default=['isotropic', 'powerlaw_1', 'powerlaw_1p5'])
    parser.add_argument('--lambda-min', type=float, default=1e-4)
    parser.add_argument('--lambda-max', type=float, default=10.0)
    parser.add_argument('--n-lambda', type=int, default=61)
    parser.add_argument('--n-trials', type=int, default=300)
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
        rows = read_summary()
        plot_r2(rows)
        plot_lambdas(rows)
        logger.info('Replotted %s and %s from %s',
                    R2_FIGURE_PATH, LAMBDA_FIGURE_PATH, SUMMARY_PATH)
        return

    lambda_grid = np.geomspace(args.lambda_min, args.lambda_max, args.n_lambda)
    projected = benchmark(args, logger, lambda_grid)
    if projected > 15 * 60:
        logger.warning('Projected runtime exceeds 15 minutes; profile before scaling.')

    rows: list[dict[str, object]] = []
    total_cases = len(args.spectra) * len(args.sigmas)
    case_index = 0
    start = time.perf_counter()
    n_cv_train = args.n * (args.n_folds - 1) / args.n_folds

    for spectrum_index, spectrum in enumerate(args.spectra):
        eigenvalues, eigenvectors = spectrum_from_name(spectrum, args.d)
        teacher_seed = args.seed + 1000 * spectrum_index
        beta_star = make_beta_star(eigenvalues, teacher_seed)
        for sigma_index, sigma in enumerate(args.sigmas):
            case_index += 1
            mc_seed = args.seed + 1000 * spectrum_index + 10 * sigma_index + 1
            cache = case_path(args, spectrum, sigma, mc_seed)
            if cache.exists() and not args.force:
                logger.info('[%d/%d] Loading %s', case_index, total_cases, cache)
                rows.append(load_case(cache))
                continue

            logger.info(
                '[%d/%d] %s sigma=%.3g: %d paired %d-fold CV trials',
                case_index, total_cases, spectrum, sigma, args.n_trials,
                args.n_folds)
            lambda_de_cv, cv_risk = select_de_lambda(
                eigenvalues, beta_star, sigma, n_cv_train, lambda_grid)
            lambda_de_full, full_risk = select_de_lambda(
                eigenvalues, beta_star, sigma, args.n, lambda_grid)
            row: dict[str, object] = {
                'spectrum': spectrum,
                'd': args.d,
                'n': args.n,
                'n_folds': args.n_folds,
                'n_cv_train': n_cv_train,
                'sigma': sigma,
                'n_trials': args.n_trials,
                'teacher_seed': teacher_seed,
                'mc_seed': mc_seed,
                'lambda_min': args.lambda_min,
                'lambda_max': args.lambda_max,
                'n_lambda': args.n_lambda,
                'theory_lambda_cv': lambda_de_cv,
                'theory_lambda_full': lambda_de_full,
            }
            row.update(final_r2_theory(
                eigenvalues, beta_star, sigma, args.n, lambda_de_cv))

            result = run_paired_cv_monte_carlo(
                args.n, eigenvalues, eigenvectors, beta_star, sigma,
                lambda_grid, n_folds=args.n_folds, n_trials=args.n_trials,
                rng=np.random.default_rng(mc_seed),
                progress=not args.no_progress, return_trials=True)
            pooled_lambdas = np.concatenate([
                result['trials']['lambda_cv_a'],
                result['trials']['lambda_cv_b'],
            ])
            row.update(lambda_distribution_summary(pooled_lambdas, lambda_grid))
            row['mc_elapsed_seconds'] = result['elapsed_seconds']
            for key, value in result.items():
                if key in {'trials', 'elapsed_seconds', 'n_trials'}:
                    continue
                row[f'mc_{key}'] = value
            save_case(cache, row, result['trials'], lambda_grid, cv_risk, full_risk)
            rows.append(row)
            logger.info(
                '[%d/%d] done %.2fs | lambda DE-CV/MC median %.4g/%.4g | R2 gen %.4f/%.4f acc %.4f/%.4f peer %.4f/%.4f',
                case_index, total_cases, result['elapsed_seconds'],
                lambda_de_cv, row['mc_lambda_median'],
                row['theory_r2_gen'], row['mc_r2_gen'],
                row['theory_r2_acc_delta'], row['mc_r2_acc'],
                row['theory_r2_peer_delta'], row['mc_r2_peer'])

    rows.sort(key=lambda row: (str(row['spectrum']), float(row['sigma'])))
    write_summary(rows)
    plot_r2(rows)
    plot_lambdas(rows)
    logger.info('Finished in %.1fs', time.perf_counter() - start)
    logger.info('Summary: %s', SUMMARY_PATH)
    logger.info('Case caches: %s', CASE_DIR)
    logger.info('Figures: %s and %s', R2_FIGURE_PATH, LAMBDA_FIGURE_PATH)


if __name__ == '__main__':
    main()

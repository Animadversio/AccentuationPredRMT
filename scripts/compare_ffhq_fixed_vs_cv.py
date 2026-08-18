"""Compare fixed ridge regularization with RidgeCV on the FFHQ disk teacher.

The comparison reuses the dense-grid RidgeCV summary and runs a paired
fixed-alpha fit on the same natural-image subsets and response-noise draws.
Scikit-learn alpha and normalized lambda obey alpha = n * lambda.
"""
from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/accentuationpredrmt-matplotlib')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp/accentuationpredrmt-xdg-cache')

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D

from scripts.validate_ffhq_disk_teacher import (
    FFHQ_ZIP,
    SPECTRUM_PATH,
    add_noise_ratio_axis,
    aggregate,
    compute_population_spectrum,
    configure_logger,
    create_disk_teacher,
    fit_metrics,
    read_summary as read_cv_summary,
    stage_ffhq_uint8,
    theory_metrics,
)
from scripts.validate_r2_peer_review import float_tag


REPO_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = REPO_ROOT / 'tables' / 'ffhq_fixed_vs_cv_summary.csv'
CASE_DIR = REPO_ROOT / 'tables' / 'ffhq_fixed_alpha_cases'
FIGURE_PATH = REPO_ROOT / 'figures' / 'ffhq_fixed_vs_cv_comparison.png'
DEFAULT_LOG_PATH = REPO_ROOT / 'logs' / 'ffhq_fixed_vs_cv.log'


def fixed_case_path(alpha: float, sigma: float, n_trials: int, seed: int) -> Path:
    return CASE_DIR / (
        f'ffhq_fixed_alpha{float_tag(alpha)}_sigma{float_tag(sigma)}'
        f'_trials{n_trials}_seed{seed}.npz')


def ridge_fixed_fit(
        X: torch.Tensor, beta: torch.Tensor, sigma: float, alpha: float,
        generator: torch.Generator) -> torch.Tensor:
    """Fit intercept-aware ridge at a fixed scikit-learn alpha."""
    n = X.shape[0]
    Xc = X - X.mean(dim=0, keepdim=True)
    noise = torch.randn(n, device=X.device, generator=generator) * sigma
    yc = Xc @ beta + noise
    yc = yc - yc.mean()
    gram = Xc @ Xc.T
    sample_eval, sample_evec = torch.linalg.eigh(gram)
    sample_eval.clamp_min_(0)
    response_proj = sample_evec.T @ yc
    dual = sample_evec @ (response_proj / (sample_eval + alpha))
    return Xc.T @ dual


def aggregate_trials(
        sigma: float, alpha: float, n: int, signal: float,
        trials: dict[str, np.ndarray], elapsed: float,
        case_path: Path) -> dict[str, object]:
    row: dict[str, object] = {
        'policy': 'fixed',
        'sigma': sigma,
        'noise_signal_ratio': sigma ** 2 / signal,
        'alpha': alpha,
        'lambda': alpha / n,
        'n_trials': len(next(iter(trials.values()))),
        'mc_elapsed_seconds': elapsed,
        'case_path': str(case_path),
    }
    for name, values in trials.items():
        mean, std, se = aggregate(values)
        row[f'mc_{name}'] = mean
        row[f'mc_{name}_std'] = std
        row[f'mc_{name}_se'] = se
        row[f'mc_{name}_median'] = float(np.median(values))
        row[f'mc_{name}_q25'] = float(np.quantile(values, 0.25))
        row[f'mc_{name}_q75'] = float(np.quantile(values, 0.75))
    return row


def save_fixed_case(
        path: Path, row: dict[str, object],
        trials: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {f'row_{key}': np.asarray(value) for key, value in row.items()}
    payload.update({f'trial_{key}': value for key, value in trials.items()})
    np.savez_compressed(path, **payload)


def load_fixed_case(path: Path) -> tuple[dict[str, object],
                                         dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as data:
        row = {
            key.removeprefix('row_'): (
                data[key].item() if data[key].ndim == 0 else data[key].tolist())
            for key in data.files if key.startswith('row_')
        }
        trials = {
            key.removeprefix('trial_'): data[key]
            for key in data.files if key.startswith('trial_')
        }
    return row, trials


def normalized_policy_row(
        policy: str, base: dict[str, object], signal: float
        ) -> dict[str, object]:
    row = {
        'policy': policy,
        'sigma': float(base['sigma']),
        'noise_signal_ratio': float(base['noise_signal_ratio']),
        'alpha': (
            float(base['theory_alpha_cv']) if policy == 'cv'
            else float(base['alpha'])),
        'lambda': (
            float(base['theory_lambda_final']) if policy == 'cv'
            else float(base['lambda'])),
        'n_trials': int(float(base['n_trials'])),
    }
    row['theory_gen_error_normalized'] = (
        float(base['theory_gen_error']) / signal)
    row['theory_acc_error_normalized'] = (
        float(base['theory_acc_error']) / signal)
    row['theory_r2_gen'] = float(base['theory_r2_gen'])
    row['theory_r2_acc'] = float(base['theory_r2_acc'])
    for metric in ('gen_error', 'acc_error'):
        row[f'mc_{metric}_normalized'] = float(base[f'mc_{metric}']) / signal
        row[f'mc_{metric}_normalized_se'] = (
            float(base[f'mc_{metric}_se']) / signal)
    for metric in ('r2_gen', 'r2_acc'):
        for suffix in ('', '_se', '_median', '_q25', '_q75'):
            key = f'mc_{metric}{suffix}'
            if key in base:
                row[key] = float(base[key])
    if policy == 'cv':
        for suffix in ('median', 'q25', 'q75'):
            row[f'mc_alpha_cv_{suffix}'] = float(
                base[f'mc_alpha_cv_{suffix}'])
    return row


def write_summary(rows: list[dict[str, object]]) -> None:
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with SUMMARY_PATH.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    SUMMARY_PATH.chmod(0o644)


def read_summary() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with SUMMARY_PATH.open(newline='') as handle:
        for source in csv.DictReader(handle):
            row: dict[str, object] = {}
            for key, value in source.items():
                if value in ('', None):
                    row[key] = np.nan
                elif key == 'policy':
                    row[key] = value
                else:
                    row[key] = float(value)
            rows.append(row)
    return rows


def metric_arrays(
        rows: list[dict[str, object]], policy: str, metric: str
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    selected = sorted(
        [row for row in rows if row['policy'] == policy],
        key=lambda item: float(item['sigma']))
    sigma = np.asarray([float(row['sigma']) for row in selected])
    theory = np.asarray([float(row[f'theory_{metric}']) for row in selected])
    mc = np.asarray([float(row[f'mc_{metric}']) for row in selected])
    if metric == 'r2_acc':
        mc = np.asarray([float(row['mc_r2_acc_median']) for row in selected])
        q25 = np.asarray([float(row['mc_r2_acc_q25']) for row in selected])
        q75 = np.asarray([float(row['mc_r2_acc_q75']) for row in selected])
        error = np.vstack((mc - q25, q75 - mc))
    else:
        error = 2 * np.asarray(
            [float(row[f'mc_{metric}_se']) for row in selected])
    return sigma, theory, mc, error


def plot_comparison(
        rows: list[dict[str, object]], fixed_alpha: float,
        signal: float, n: int) -> None:
    fig = plt.figure(figsize=(13.2, 10.5))
    grid = fig.add_gridspec(
        3, 2, height_ratios=(0.72, 1, 1), hspace=0.34, wspace=0.22)
    alpha_ax = fig.add_subplot(grid[0, :])
    axes = {
        'gen_error_normalized': fig.add_subplot(grid[1, 0]),
        'acc_error_normalized': fig.add_subplot(grid[1, 1]),
        'r2_gen': fig.add_subplot(grid[2, 0]),
        'r2_acc': fig.add_subplot(grid[2, 1]),
    }
    colors = {'cv': 'C0', 'fixed': 'C1'}
    labels = {
        'cv': 'RidgeCV',
        'fixed': rf'fixed $\alpha={fixed_alpha:g}$ '
                 rf'($\lambda={fixed_alpha / n:g}$)',
    }
    cv_rows = sorted(
        [row for row in rows if row['policy'] == 'cv'],
        key=lambda item: float(item['sigma']))
    cv_sigma = np.asarray([float(row['sigma']) for row in cv_rows])
    cv_alpha = np.asarray([float(row['alpha']) for row in cv_rows])
    alpha_ax.plot(cv_sigma, cv_alpha, color=colors['cv'], lw=2.1)
    alpha_ax.axhline(fixed_alpha, color=colors['fixed'], lw=2.1)
    alpha_ax.scatter(
        cv_sigma, [float(row['mc_alpha_cv_median']) for row in cv_rows],
        color=colors['cv'], edgecolor='black', linewidth=0.4, s=23, zorder=4)
    alpha_ax.fill_between(
        cv_sigma,
        [float(row['mc_alpha_cv_q25']) for row in cv_rows],
        [float(row['mc_alpha_cv_q75']) for row in cv_rows],
        color=colors['cv'], alpha=0.16, linewidth=0)
    anchor_index = int(np.argmin(np.abs(cv_alpha - fixed_alpha)))
    alpha_ax.annotate(
        rf'anchor: RidgeCV selects $\alpha={fixed_alpha:g}$',
        xy=(cv_sigma[anchor_index], cv_alpha[anchor_index]),
        xytext=(18, 24), textcoords='offset points', fontsize=8,
        arrowprops={'arrowstyle': '->', 'lw': 0.8, 'color': '0.35'})
    alpha_ax.set_yscale('log')
    alpha_ax.set_ylabel(r'regularization $\alpha=n\lambda$')
    alpha_ax.set_title('Regularization policy')

    metric_specs = {
        'gen_error_normalized': (r'$E_{\rm gen}/S$', True),
        'acc_error_normalized': (r'$E_{\rm acc}/S$', True),
        'r2_gen': (r'$R^2_{\rm gen}$', False),
        'r2_acc': (r'$R^2_{\rm acc}$', False),
    }
    for metric, (ylabel, log_y) in metric_specs.items():
        ax = axes[metric]
        for policy in ('cv', 'fixed'):
            sigma, theory, mc, error = metric_arrays(rows, policy, metric)
            color = colors[policy]
            ax.plot(sigma, theory, color=color, lw=2.0)
            marker = (
                'd' if metric == 'r2_acc'
                else ('o' if policy == 'cv' else 's'))
            ax.errorbar(
                sigma, mc, yerr=error, fmt=marker, ls='none', color=color,
                mec='black', mew=0.4, ms=4.4, capsize=2.2, zorder=4)
        if log_y:
            ax.set_yscale('log')
        if metric == 'r2_acc':
            ax.set_yscale('symlog', linthresh=0.1)
        if metric.startswith('r2_'):
            ax.axhline(1, color='0.5', lw=0.8, ls=':')
            ax.axhline(0, color='0.5', lw=0.8, ls='--')
        ax.set_ylabel(ylabel)

    max_sigma = max(float(row['sigma']) for row in rows) * 1.15
    for ax in [alpha_ax, *axes.values()]:
        ax.set_xscale('symlog', linthresh=0.01)
        ax.set_xlim(0, max_sigma)
        ax.grid(alpha=0.18)
    add_noise_ratio_axis(alpha_ax, signal, max_sigma)
    for ax in (axes['r2_gen'], axes['r2_acc']):
        ax.set_xlabel(r'response noise $\sigma$')
    for ax in (axes['gen_error_normalized'], axes['acc_error_normalized']):
        ax.tick_params(labelbottom=False)
    alpha_ax.tick_params(labelbottom=False)

    legend_handles = [
        Line2D([0], [0], color=colors['cv'], lw=2, marker='o',
               markeredgecolor='black', markeredgewidth=0.4,
               label=labels['cv']),
        Line2D([0], [0], color=colors['fixed'], lw=2, marker='s',
               markeredgecolor='black', markeredgewidth=0.4,
               label=labels['fixed']),
        Line2D([0], [0], color='0.25', lw=2, label='DE curve'),
        Line2D([0], [0], color='0.25', marker='o', ls='none',
               label=r'MC mean $\pm$ 2 SE'),
        Line2D([0], [0], color='0.25', marker='d', ls='none',
               label=r'$R^2_{\rm acc}$ MC median / IQR'),
    ]
    fig.legend(
        handles=legend_handles, loc='lower center', ncol=3, frameon=False,
        fontsize=8.5, bbox_to_anchor=(0.5, 0.005))
    fig.suptitle(
        'FFHQ disk teacher: fixed regularization versus cross-validation',
        y=0.995)
    fig.subplots_adjust(
        left=0.075, right=0.985, bottom=0.105, top=0.90,
        hspace=0.38, wspace=0.22)
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=190, bbox_inches='tight')
    FIGURE_PATH.chmod(0o644)
    plt.close(fig)


def merge_rows(
        cv_rows: list[dict[str, object]],
        fixed_rows: list[dict[str, object]], signal: float
        ) -> list[dict[str, object]]:
    merged = [
        normalized_policy_row('cv', row, signal) for row in cv_rows
    ] + [
        normalized_policy_row('fixed', row, signal) for row in fixed_rows
    ]
    return sorted(
        merged, key=lambda row: (
            float(row['sigma']), 0 if row['policy'] == 'cv' else 1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--fixed-alpha', type=float, default=100.0)
    parser.add_argument('--n', type=int, default=1000)
    parser.add_argument('--n-trials', type=int, default=100)
    parser.add_argument('--benchmark-trials', type=int, default=3)
    parser.add_argument('--seed', type=int, default=20260814)
    parser.add_argument('--img-size', type=int, default=100)
    parser.add_argument('--radius', type=float, default=0.3)
    parser.add_argument('--train-pool-size', type=int, default=10000)
    parser.add_argument('--population-size', type=int, default=20000)
    parser.add_argument('--zip-path', type=Path, default=FFHQ_ZIP)
    parser.add_argument('--scratch-dir', type=Path,
                        default=Path('/tmp/ffhq_disk_teacher_de'))
    parser.add_argument('--plot-only', action='store_true')
    parser.add_argument('--benchmark-only', action='store_true')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--no-progress', action='store_true')
    parser.add_argument('--log-file', type=Path, default=DEFAULT_LOG_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.fixed_alpha <= 0:
        raise ValueError('--fixed-alpha must be positive')
    logger = configure_logger(args.log_file)
    with np.load(SPECTRUM_PATH) as spectrum:
        compact_eigenvalues = np.asarray(spectrum['eigenvalues'])
        compact_beta_proj = np.asarray(spectrum['beta_proj'])
    signal = float(np.sum(compact_eigenvalues * compact_beta_proj ** 2))
    cv_rows = sorted(read_cv_summary(), key=lambda row: float(row['sigma']))
    if args.plot_only:
        rows = read_summary()
        cached_fixed = sorted({
            float(row['alpha']) for row in rows if row['policy'] == 'fixed'})
        if len(cached_fixed) != 1:
            raise RuntimeError(
                f'Expected one cached fixed alpha, found {cached_fixed}')
        plot_comparison(rows, cached_fixed[0], signal, args.n)
        logger.info('Replotted %s from %s', FIGURE_PATH, SUMMARY_PATH)
        return
    if not torch.cuda.is_available():
        raise RuntimeError('Fixed-alpha natural-image Monte Carlo requires CUDA.')

    device = torch.device('cuda')
    logger.info('GPU: %s', torch.cuda.get_device_name(device))
    args.scratch_dir.mkdir(parents=True, exist_ok=True)
    total_images = args.train_pool_size + args.population_size
    staged_path = args.scratch_dir / (
        f'ffhq_gray_{args.img_size}px_{total_images}_uint8.npy')
    staged = stage_ffhq_uint8(
        args.zip_path, staged_path, total_images, args.img_size, logger)
    beta_np = create_disk_teacher(args.img_size, args.radius)
    beta = torch.from_numpy(beta_np).to(device)
    eigval, eigvec, _, eigenvalues, beta_proj, _ = compute_population_spectrum(
        staged, args.train_pool_size, args.population_size, beta_np,
        args.scratch_dir, device, logger)
    beta_proj_t = torch.from_numpy(beta_proj.astype(np.float32)).to(device)
    train_pool = torch.from_numpy(
        np.asarray(staged[:args.train_pool_size], dtype=np.float32)).to(device)
    train_pool.div_(255.0)

    benchmark_generator = torch.Generator(device=device).manual_seed(args.seed)
    start = time.perf_counter()
    for _ in range(args.benchmark_trials):
        indices = torch.randperm(
            args.train_pool_size, generator=benchmark_generator,
            device=device)[:args.n]
        ridge_fixed_fit(
            train_pool[indices], beta, float(cv_rows[0]['sigma']),
            args.fixed_alpha, benchmark_generator)
    torch.cuda.synchronize()
    seconds_per_trial = (
        time.perf_counter() - start) / args.benchmark_trials
    total_trials = len(cv_rows) * args.n_trials
    logger.info(
        'Benchmark %.4fs/trial; projected %.1fs (%.1fmin) for %d fits',
        seconds_per_trial, seconds_per_trial * total_trials,
        seconds_per_trial * total_trials / 60, total_trials)
    if args.benchmark_only:
        return

    fixed_rows: list[dict[str, object]] = []
    metric_names = ('gen_error', 'r2_gen', 'acc_error', 'r2_acc')
    for case_index, cv_row in enumerate(cv_rows, start=1):
        sigma = float(cv_row['sigma'])
        path = fixed_case_path(
            args.fixed_alpha, sigma, args.n_trials, args.seed)
        if path.exists() and not args.force:
            logger.info('[%d/%d] Loading %s', case_index, len(cv_rows), path)
            row, _ = load_fixed_case(path)
            fixed_rows.append(row)
            continue
        theoretical, _ = theory_metrics(
            eigenvalues, beta_proj, sigma, args.n, args.fixed_alpha)
        trials = {
            name: np.empty(args.n_trials, dtype=float)
            for name in metric_names
        }
        generator = torch.Generator(device=device).manual_seed(
            args.seed + 1000 * case_index)
        iterator = range(args.n_trials)
        if not args.no_progress:
            try:
                from tqdm.auto import tqdm
                iterator = tqdm(
                    iterator, desc=f'fixed alpha sigma={sigma:g}', unit='trial')
            except ImportError:
                pass
        case_start = time.perf_counter()
        for trial_index in iterator:
            indices = torch.randperm(
                args.train_pool_size, generator=generator,
                device=device)[:args.n]
            weight = ridge_fixed_fit(
                train_pool[indices], beta, sigma, args.fixed_alpha, generator)
            metrics, _ = fit_metrics(
                weight, beta, eigval, eigvec, beta_proj_t, signal)
            for name in metric_names:
                trials[name][trial_index] = metrics[name]
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - case_start
        row = aggregate_trials(
            sigma, args.fixed_alpha, args.n, signal, trials, elapsed, path)
        row.update(theoretical)
        save_fixed_case(path, row, trials)
        fixed_rows.append(row)
        logger.info(
            '[%d/%d] sigma=%.3g done %.2fs | Egen/S %.4g | '
            'Eacc/S %.4g | R2acc median %.4g',
            case_index, len(cv_rows), sigma, elapsed,
            float(row['mc_gen_error']) / signal,
            float(row['mc_acc_error']) / signal,
            float(row['mc_r2_acc_median']))

    rows = merge_rows(cv_rows, fixed_rows, signal)
    write_summary(rows)
    plot_comparison(rows, args.fixed_alpha, signal, args.n)
    logger.info('Summary: %s', SUMMARY_PATH)
    logger.info('Figure: %s', FIGURE_PATH)


if __name__ == '__main__':
    main()

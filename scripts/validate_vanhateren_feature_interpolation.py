"""Validate two linear feature-space families on Van Hateren statistics.

The measured 100 x 100 Van Hateren covariance spectrum and disk-teacher
alignment are held fixed.  We compare

    coupled: a_j(t)^2 = s_j^{-(1-t)} [1 + (s_100 / s_j)^2]^{-t}
    power:   a_j(t)^2 = s_j^{-(1-t)}

from whitening (t=0) to either a soft top-100 filter or full PCA (t=1).
Ridge regularization is selected independently at every response-noise level
using deterministic-equivalent CV or exact analytic LOOCV in Monte Carlo.
Plot-ready summaries and raw trials are cached separately.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from scripts.validate_ffhq_disk_teacher import configure_logger
from scripts.validate_ffhq_feature_interpolation import (
    PATH_VALUES,
    VALIDATION_RATIOS,
    de_sweep,
    feature_name,
    make_feature_problem,
    merge_mc_rows,
    plot_feature_profiles,
    plot_gen_acc_facets,
    run_mc_variant,
)
from scripts.validate_ffhq_linear_features import read_rows, write_rows
from scripts.validate_vanhateren_disk_teacher import (
    DEFAULT_NOISE_RATIOS,
    SPECTRUM_PATH,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = (
    REPO_ROOT / 'tables' / 'vanhateren_feature_interpolation_summary.csv')
CASE_DIR = REPO_ROOT / 'tables' / 'vanhateren_feature_interpolation_cases'
DEFAULT_LOG_PATH = (
    REPO_ROOT / 'logs' / 'vanhateren_feature_interpolation_validation.log')

COUPLED_COLUMN_FIGURE_PATH = (
    REPO_ROOT / 'figures' /
    'vanhateren_feature_interpolation_gen_acc_by_feature.png')
COUPLED_TRANSPOSED_FIGURE_PATH = (
    REPO_ROOT / 'figures' /
    'vanhateren_feature_interpolation_gen_acc_transposed.png')
POWER_COLUMN_FIGURE_PATH = (
    REPO_ROOT / 'figures' /
    'vanhateren_power_interpolation_gen_acc_by_feature.png')
POWER_TRANSPOSED_FIGURE_PATH = (
    REPO_ROOT / 'figures' /
    'vanhateren_power_interpolation_gen_acc_transposed.png')
COUPLED_PROFILE_FIGURE_PATH = (
    REPO_ROOT / 'figures' /
    'vanhateren_feature_interpolation_profiles.png')
POWER_PROFILE_FIGURE_PATH = (
    REPO_ROOT / 'figures' /
    'vanhateren_power_interpolation_profiles.png')

COUPLED_MC_PATH_VALUES = np.asarray([0.4, 0.6, 0.8, 1.0])
POWER_MC_PATH_VALUES = np.asarray([0.2, 0.4, 0.6, 0.8, 1.0])
DATASET_TITLE = 'Van Hateren disk teacher'


def build_problems(
        eigenvalues: np.ndarray, beta: np.ndarray, cutoff: int,
        taper_power: float
        ) -> dict[str, dict[str, np.ndarray | float | int | str]]:
    problems: dict[str, dict[str, np.ndarray | float | int | str]] = {}
    for family in ('power', 'coupled'):
        for t in PATH_VALUES:
            name = feature_name(family, float(t))
            problems[name] = make_feature_problem(
                family, float(t), eigenvalues, beta, cutoff, taper_power)
    problems['hard_top100'] = make_feature_problem(
        'hard', 1.0, eigenvalues, beta, cutoff, taper_power)
    return problems


def render_all(
        rows: list[dict[str, object]], eigenvalues: np.ndarray,
        cutoff: int, taper_power: float) -> None:
    plot_gen_acc_facets(
        rows, transpose=False, family='coupled',
        dataset_title=DATASET_TITLE,
        output_path=COUPLED_COLUMN_FIGURE_PATH)
    plot_gen_acc_facets(
        rows, transpose=True, family='coupled',
        dataset_title=DATASET_TITLE,
        output_path=COUPLED_TRANSPOSED_FIGURE_PATH)
    plot_gen_acc_facets(
        rows, transpose=False, family='power',
        dataset_title=DATASET_TITLE,
        output_path=POWER_COLUMN_FIGURE_PATH)
    plot_gen_acc_facets(
        rows, transpose=True, family='power',
        dataset_title=DATASET_TITLE,
        output_path=POWER_TRANSPOSED_FIGURE_PATH)
    plot_feature_profiles(
        eigenvalues, cutoff, taper_power, family='coupled',
        dataset_title='Van Hateren',
        output_path=COUPLED_PROFILE_FIGURE_PATH)
    plot_feature_profiles(
        eigenvalues, cutoff, taper_power, family='power',
        dataset_title='Van Hateren',
        output_path=POWER_PROFILE_FIGURE_PATH)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=1000)
    parser.add_argument('--n-trials', type=int, default=25)
    parser.add_argument('--seed', type=int, default=20260828)
    parser.add_argument('--cutoff', type=int, default=100)
    parser.add_argument('--taper-power', type=float, default=2.0)
    parser.add_argument('--alpha-min-exp', type=float, default=-6.0)
    parser.add_argument('--alpha-max-exp', type=float, default=9.0)
    parser.add_argument('--alpha-grid-size', type=int, default=301)
    parser.add_argument('--cpu-threads', type=int, default=8)
    parser.add_argument('--de-only', action='store_true')
    parser.add_argument('--plot-only', action='store_true')
    parser.add_argument('--benchmark-only', action='store_true')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--no-progress', action='store_true')
    parser.add_argument('--log-file', type=Path, default=DEFAULT_LOG_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = configure_logger(args.log_file)
    with np.load(SPECTRUM_PATH) as data:
        eigenvalues = np.asarray(data['eigenvalues'], dtype=float)
        beta = np.asarray(data['beta_proj'], dtype=float)

    if args.plot_only:
        rows = read_rows(SUMMARY_PATH)
        render_all(rows, eigenvalues, args.cutoff, args.taper_power)
        logger.info('Replotted all Van Hateren feature-family figures')
        return

    signal = float(np.sum(eigenvalues * beta ** 2))
    sigmas = np.sqrt(DEFAULT_NOISE_RATIOS * signal)
    alphas = np.logspace(
        args.alpha_min_exp, args.alpha_max_exp, args.alpha_grid_size)
    problems = build_problems(
        eigenvalues, beta, args.cutoff, args.taper_power)

    de_start = time.perf_counter()
    rows: list[dict[str, object]] = []
    for index, (name, problem) in enumerate(problems.items(), start=1):
        logger.info('[DE %d/%d] %s', index, len(problems), name)
        feature_rows = de_sweep(name, problem, sigmas, args.n, alphas)
        for row in feature_rows:
            row['dataset'] = 'vanhateren'
        rows.extend(feature_rows)
    logger.info(
        'Van Hateren DE family sweep completed in %.1fs',
        time.perf_counter() - de_start)

    if not args.de_only:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device.type == 'cpu':
            torch.set_num_threads(args.cpu_threads)
        logger.info(
            'Monte Carlo device: %s%s', device,
            (f' ({torch.cuda.get_device_name(device)})'
             if device.type == 'cuda' else
             f' with {torch.get_num_threads()} threads'))

        benchmark_name = feature_name('power', 0.6)
        benchmark_start = time.perf_counter()
        benchmark_rows = run_mc_variant(
            benchmark_name, problems[benchmark_name], VALIDATION_RATIOS,
            alphas, args.n, 1, args.seed + 991, device, True, False, logger,
            case_dir=CASE_DIR, dataset='vanhateren')
        benchmark_seconds = time.perf_counter() - benchmark_start
        n_unique_variants = (
            1 + len(COUPLED_MC_PATH_VALUES)
            + len(POWER_MC_PATH_VALUES) + 1)
        projected = benchmark_seconds * n_unique_variants * args.n_trials
        logger.info(
            'Small-scale benchmark %.2fs/design; projected %.1fs (%.1fmin) '
            'for %d designs across %d feature maps; each serves %d noises',
            benchmark_seconds, projected, projected / 60,
            n_unique_variants * args.n_trials, n_unique_variants,
            len(VALIDATION_RATIOS))
        benchmark_path = Path(str(benchmark_rows[0]['case_path']))
        if benchmark_path.exists():
            benchmark_path.unlink()
        if args.benchmark_only:
            return

        mc_rows: list[dict[str, object]] = []
        whitening_name = feature_name('power', 0.0)
        whitening_rows = run_mc_variant(
            whitening_name, problems[whitening_name], VALIDATION_RATIOS,
            alphas, args.n, args.n_trials, args.seed, device, args.force,
            not args.no_progress, logger,
            case_dir=CASE_DIR, dataset='vanhateren')
        mc_rows.extend(whitening_rows)
        for source in whitening_rows:
            duplicate = dict(source)
            duplicate['feature'] = feature_name('coupled', 0.0)
            mc_rows.append(duplicate)

        for family, path_values in (
                ('coupled', COUPLED_MC_PATH_VALUES),
                ('power', POWER_MC_PATH_VALUES)):
            for t in path_values:
                name = feature_name(family, float(t))
                mc_rows.extend(run_mc_variant(
                    name, problems[name], VALIDATION_RATIOS,
                    alphas, args.n, args.n_trials, args.seed, device,
                    args.force, not args.no_progress, logger,
                    case_dir=CASE_DIR, dataset='vanhateren'))
        hard_name = 'hard_top100'
        mc_rows.extend(run_mc_variant(
            hard_name, problems[hard_name], VALIDATION_RATIOS,
            alphas, args.n, args.n_trials, args.seed, device, args.force,
            not args.no_progress, logger,
            case_dir=CASE_DIR, dataset='vanhateren'))
        merge_mc_rows(rows, mc_rows)

    rows.sort(key=lambda row: (
        str(row['family']), float(row['t']), float(row['sigma'])))
    write_rows(SUMMARY_PATH, rows)
    render_all(rows, eigenvalues, args.cutoff, args.taper_power)
    logger.info('Plot-ready summary: %s', SUMMARY_PATH)
    logger.info('Raw MC cases: %s', CASE_DIR)
    logger.info(
        'Figures: %s, %s, %s, %s, %s, %s',
        COUPLED_COLUMN_FIGURE_PATH, COUPLED_TRANSPOSED_FIGURE_PATH,
        POWER_COLUMN_FIGURE_PATH, POWER_TRANSPOSED_FIGURE_PATH,
        COUPLED_PROFILE_FIGURE_PATH, POWER_PROFILE_FIGURE_PATH)


if __name__ == '__main__':
    main()

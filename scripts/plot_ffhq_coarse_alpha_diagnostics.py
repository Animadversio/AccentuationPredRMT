"""Plot vertically aligned diagnostics for the original coarse FFHQ alpha grid.

This script reads the preserved ten-candidate RidgeCV case caches, writes a
plot-ready CSV, and aligns regularization stages with generalization and
accentuation errors, R2 values, and fitted-response slopes.
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/accentuationpredrmt-matplotlib')

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.validate_ffhq_disk_teacher import (  # noqa: E402
    CASE_DIR,
    add_noise_ratio_axis,
    load_case,
)
from scripts.storage_paths import require_bulk_path


FIGURE_PATH = (
    REPO_ROOT / 'figures' / 'model_selection' /
    'ffhq_disk_teacher_coarse_alpha_aligned.png')
TABLE_PATH = (
    REPO_ROOT / 'tables' / 'ffhq_disk_teacher_coarse_alpha_aligned.csv')


def coarse_case_paths() -> list[Path]:
    """Return only the original ten-alpha caches, excluding dense-grid cases."""
    pattern = (
        'ffhq_disk_d10000_n1000_sigma*_trials100_pop20000_'
        'seed20260814.npz')
    case_dir = require_bulk_path(
        CASE_DIR, 'FFHQ disk-teacher coarse-grid case caches')
    paths = list(case_dir.glob(pattern))
    if not paths:
        raise FileNotFoundError(
            f'No preserved coarse-grid cases matching {case_dir / pattern}')
    return paths


def quantiles(values: np.ndarray) -> tuple[float, float, float]:
    return tuple(float(np.quantile(values, q)) for q in (0.25, 0.5, 0.75))


def read_coarse_rows() -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for path in coarse_case_paths():
        row, arrays = load_case(path)
        signal = float(row['theory_signal_power'])
        alpha_q25, alpha_median, alpha_q75 = quantiles(
            arrays['trial_alpha_cv'])
        r2_q25, r2_median, r2_q75 = quantiles(arrays['trial_r2_acc'])
        slope_acc_q25, slope_acc_median, slope_acc_q75 = quantiles(
            arrays['trial_slope_acc'])
        rows.append({
            'sigma': float(row['sigma']),
            'noise_signal_ratio': float(row['sigma']) ** 2 / signal,
            'signal_power': signal,
            'de_alpha': float(row['theory_alpha_cv']),
            'mc_alpha_median': alpha_median,
            'mc_alpha_q25': alpha_q25,
            'mc_alpha_q75': alpha_q75,
            'de_gen_error_normalized': float(row['theory_gen_error']) / signal,
            'mc_gen_error_normalized': float(row['mc_gen_error']) / signal,
            'mc_gen_error_normalized_se': float(row['mc_gen_error_se']) / signal,
            'de_acc_error_normalized': float(row['theory_acc_error']) / signal,
            'mc_acc_error_normalized': float(row['mc_acc_error']) / signal,
            'mc_acc_error_normalized_se': float(row['mc_acc_error_se']) / signal,
            'de_r2_gen': float(row['theory_r2_gen']),
            'mc_r2_gen': float(row['mc_r2_gen']),
            'mc_r2_gen_se': float(row['mc_r2_gen_se']),
            'de_r2_acc': float(row['theory_r2_acc']),
            'mc_r2_acc': float(row['mc_r2_acc']),
            'mc_r2_acc_se': float(row['mc_r2_acc_se']),
            'mc_r2_acc_median': r2_median,
            'mc_r2_acc_q25': r2_q25,
            'mc_r2_acc_q75': r2_q75,
            'de_slope_gen': float(row['theory_slope_gen']),
            'mc_slope_gen': float(row['mc_slope_gen']),
            'mc_slope_gen_se': float(row['mc_slope_gen_se']),
            'de_slope_acc': float(row['theory_slope_acc']),
            'mc_slope_acc': float(row['mc_slope_acc']),
            'mc_slope_acc_se': float(row['mc_slope_acc_se']),
            'mc_slope_acc_median': slope_acc_median,
            'mc_slope_acc_q25': slope_acc_q25,
            'mc_slope_acc_q75': slope_acc_q75,
        })
    return sorted(rows, key=lambda item: item['sigma'])


def write_table(rows: list[dict[str, float]]) -> None:
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TABLE_PATH.open('w', newline='') as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    TABLE_PATH.chmod(0o644)


def transition_boundaries(
        sigma: np.ndarray, alpha: np.ndarray) -> tuple[np.ndarray, list[slice]]:
    change_indices = np.flatnonzero(alpha[1:] != alpha[:-1]) + 1
    boundaries = []
    for index in change_indices:
        left, right = sigma[index - 1], sigma[index]
        boundaries.append(
            np.sqrt(left * right) if left > 0 else right / 2.0)
    starts = np.concatenate(([0], change_indices))
    stops = np.concatenate((change_indices, [len(alpha)]))
    stages = [slice(int(start), int(stop)) for start, stop in zip(starts, stops)]
    return np.asarray(boundaries), stages


def plot(rows: list[dict[str, float]]) -> None:
    sigma = np.asarray([row['sigma'] for row in rows])
    signal = rows[0]['signal_power']
    de_alpha = np.asarray([row['de_alpha'] for row in rows])
    mc_alpha = np.asarray([row['mc_alpha_median'] for row in rows])
    mc_alpha_q25 = np.asarray([row['mc_alpha_q25'] for row in rows])
    mc_alpha_q75 = np.asarray([row['mc_alpha_q75'] for row in rows])
    boundaries, stages = transition_boundaries(sigma, de_alpha)
    xmax = float(max(sigma) * 1.15)

    fig, axes = plt.subplots(
        4, 1, figsize=(10.8, 13.5), sharex=True,
        gridspec_kw={'height_ratios': [0.9, 1.0, 1.25, 1.0]})

    stage_edges = np.concatenate(([0.0], boundaries, [xmax]))
    for ax in axes:
        for stage_index in range(len(stage_edges) - 1):
            if stage_index % 2:
                ax.axvspan(
                    stage_edges[stage_index], stage_edges[stage_index + 1],
                    color='0.5', alpha=0.055, zorder=-3)
        for boundary in boundaries:
            ax.axvline(boundary, color='0.35', lw=0.8, ls=':', zorder=-1)
        ax.grid(alpha=0.18)
        ax.set_xscale('symlog', linthresh=0.01)
        ax.set_xlim(0, xmax)

    ax = axes[0]
    ax.errorbar(
        sigma, mc_alpha,
        yerr=np.vstack((mc_alpha - mc_alpha_q25,
                        mc_alpha_q75 - mc_alpha)),
        fmt='o-', color='black', capsize=3, label='MC median / IQR')
    ax.step(sigma, de_alpha, where='mid', color='C0', lw=1.6, ls='--',
            label='DE-selected alpha')
    ax.plot(sigma, de_alpha, 's', color='C0', ms=4)
    ax.set_yscale('log')
    ax.set_ylabel(r'RidgeCV $\alpha=n\lambda$')
    ax.set_title('Regularization stages')
    ax.legend(loc='upper left', fontsize=8, ncol=2)
    for stage in stages:
        stage_sigma = sigma[stage]
        positive = stage_sigma[stage_sigma > 0]
        center = (np.sqrt(positive[0] * positive[-1]) if len(positive) > 1
                  else float(stage_sigma[len(stage_sigma) // 2]))
        if center == 0:
            center = 0.002
        ax.text(
            center, 0.96, rf'$\alpha={de_alpha[stage.start]:g}$',
            transform=ax.get_xaxis_transform(), ha='center', va='top',
            fontsize=7, color='0.25')
    add_noise_ratio_axis(ax, signal, xmax)

    path_colors = {'gen': 'C0', 'acc': 'C1'}
    source_handles = [
        Line2D([0], [0], color='0.25', marker='o', lw=1.5,
               label='MC mean +/- 2 SE'),
        Line2D([0], [0], color='0.25', marker='s', lw=1.5, ls='--',
               label='DE'),
    ]
    path_handles = [
        Line2D([0], [0], color='C0', lw=2, label='generalization'),
        Line2D([0], [0], color='C1', lw=2, label='accentuation'),
    ]

    ax = axes[1]
    for path_name in ('gen', 'acc'):
        color = path_colors[path_name]
        mc = np.asarray(
            [row[f'mc_{path_name}_error_normalized'] for row in rows])
        se = np.asarray(
            [row[f'mc_{path_name}_error_normalized_se'] for row in rows])
        de = np.asarray(
            [row[f'de_{path_name}_error_normalized'] for row in rows])
        ax.errorbar(
            sigma, mc, yerr=2 * se, fmt='o-', color=color, capsize=2)
        ax.plot(sigma, de, 's--', color=color, ms=4)
    ax.set_yscale('log')
    ax.set_ylabel(r'normalized error $E/S$')
    ax.set_title('Response errors')
    ax.legend(handles=path_handles + source_handles, loc='upper left',
              fontsize=8, ncol=2)

    ax = axes[2]
    for path_name in ('gen', 'acc'):
        color = path_colors[path_name]
        mc = np.asarray([row[f'mc_r2_{path_name}'] for row in rows])
        se = np.asarray([row[f'mc_r2_{path_name}_se'] for row in rows])
        de = np.asarray([row[f'de_r2_{path_name}'] for row in rows])
        ax.errorbar(
            sigma, mc, yerr=2 * se, fmt='o-', color=color, capsize=2)
        ax.plot(sigma, de, 's--', color=color, ms=4)
    r2_median = np.asarray([row['mc_r2_acc_median'] for row in rows])
    r2_q25 = np.asarray([row['mc_r2_acc_q25'] for row in rows])
    r2_q75 = np.asarray([row['mc_r2_acc_q75'] for row in rows])
    ax.fill_between(sigma, r2_q25, r2_q75, color='C1', alpha=0.10)
    ax.plot(sigma, r2_median, '^:', color='C1', ms=4,
            label=r'MC median $R^2_{acc}$ / IQR')
    negative = np.asarray([row['mc_r2_acc'] < 0 for row in rows])
    ax.scatter(
        sigma[negative],
        np.asarray([row['mc_r2_acc'] for row in rows])[negative],
        s=72, facecolors='none', edgecolors='C3', linewidths=1.2,
        zorder=6, label=r'negative MC mean $R^2_{acc}$')
    ax.axhline(0, color='0.4', lw=0.8)
    ax.axhline(1, color='0.5', lw=0.7, ls=':')
    ax.set_ylabel(r'$R^2$')
    ax.set_title(r'Path-wise $R^2$')
    ax.legend(loc='lower left', fontsize=8, ncol=2)

    ax = axes[3]
    for path_name in ('gen', 'acc'):
        color = path_colors[path_name]
        mc = np.asarray([row[f'mc_slope_{path_name}'] for row in rows])
        se = np.asarray([row[f'mc_slope_{path_name}_se'] for row in rows])
        de = np.asarray([row[f'de_slope_{path_name}'] for row in rows])
        ax.errorbar(
            sigma, mc, yerr=2 * se, fmt='o-', color=color, capsize=2)
        ax.plot(sigma, de, 's--', color=color, ms=4)
    ax.axhline(1, color='0.5', lw=0.7, ls=':')
    ax.set_ylabel('true-on-fitted slope')
    ax.set_title('Response slopes')
    ax.set_xlabel(r'response noise $\sigma$')

    fig.suptitle(
        'FFHQ disk teacher: coarse-alpha stages aligned with prediction metrics')
    fig.text(
        0.5, 0.008,
        'Vertical dotted lines and alternating bands mark DE-selected alpha transitions.',
        ha='center', fontsize=8, color='0.35')
    fig.tight_layout(rect=(0, 0.025, 1, 0.965), h_pad=1.35)
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=190, bbox_inches='tight')
    FIGURE_PATH.chmod(0o644)
    plt.close(fig)


def main() -> None:
    rows = read_coarse_rows()
    write_table(rows)
    plot(rows)
    print(f'Wrote {TABLE_PATH}')
    print(f'Wrote {FIGURE_PATH}')


if __name__ == '__main__':
    main()

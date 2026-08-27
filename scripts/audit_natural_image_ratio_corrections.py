"""Compare ratio-of-expectations and second-order accentuation theories.

This is a plot/cache-only audit.  It reuses the stored FFHQ and Van Hateren
spectra, selected ridge strengths, and Monte Carlo summaries; no image loading
or regression refitting is required.  The leading and second-order curves use
the same cached alpha at every noise level, including the DE-CV alpha.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.compare_ffhq_fixed_vs_cv import (
    GAP_FIGURE_PATH as FFHQ_FIGURE_PATH,
    SPECTRUM_PATH as FFHQ_SPECTRUM_PATH,
    SUMMARY_PATH as FFHQ_SUMMARY_PATH,
    add_theory_order_columns,
    plot_policy_gap_comparison,
    read_summary as read_ffhq_summary,
    write_summary as write_ffhq_summary,
)
from scripts.validate_vanhateren_disk_teacher import (
    FIGURE_PATH as VANHATEREN_FIGURE_PATH,
    SPECTRUM_PATH as VANHATEREN_SPECTRUM_PATH,
    SUMMARY_PATH as VANHATEREN_SUMMARY_PATH,
    read_summary as read_vanhateren_summary,
    write_summary as write_vanhateren_summary,
)


ERROR_SUMMARY_PATH = (
    REPO_ROOT / 'tables' / 'natural_image_ratio_theory_errors.csv')


def load_spectrum(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as spectrum:
        return (
            np.asarray(spectrum['eigenvalues']),
            np.asarray(spectrum['beta_proj']),
        )


def summarize_errors(
        dataset: str, rows: list[dict[str, object]]
        ) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    specs = (
        (
            'acc_error_normalized', 'mc_mean',
            'mc_acc_error_normalized',
            'theory_acc_error_ratio_normalized',
            'theory_acc_error_second_order_normalized',
        ),
        (
            'r2_acc', 'mc_mean', 'mc_r2_acc',
            'theory_r2_acc_ratio', 'theory_r2_acc_second_order',
        ),
        (
            'r2_acc', 'mc_median', 'mc_r2_acc_median',
            'theory_r2_acc_ratio', 'theory_r2_acc_second_order',
        ),
    )
    for policy in ('fixed', 'cv'):
        policy_rows = [row for row in rows if row['policy'] == policy]
        for metric, target_name, target_key, leading_key, corrected_key in specs:
            for approximation, theory_key in (
                    ('ratio_of_expectations', leading_key),
                    ('response_noise_second_order', corrected_key)):
                for max_noise_ratio in (1.0, 10.0):
                    selected = [
                        row for row in policy_rows
                        if float(row['noise_signal_ratio'])
                        <= max_noise_ratio * (1.0 + 1e-12)
                    ]
                    target = np.asarray(
                        [float(row[target_key]) for row in selected])
                    theory = np.asarray(
                        [float(row[theory_key]) for row in selected])
                    error = theory - target
                    output.append({
                        'dataset': dataset,
                        'policy': policy,
                        'metric': metric,
                        'empirical_target': target_name,
                        'theory_approximation': approximation,
                        'max_noise_signal_ratio': max_noise_ratio,
                        'n_noise_levels': len(selected),
                        'mean_absolute_error': float(
                            np.mean(np.abs(error))),
                        'median_absolute_error': float(
                            np.median(np.abs(error))),
                        'max_absolute_error': float(
                            np.max(np.abs(error))),
                        'mean_signed_error': float(np.mean(error)),
                    })
    return output


def write_error_summary(rows: list[dict[str, object]]) -> None:
    ERROR_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with ERROR_SUMMARY_PATH.open('w', newline='') as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    ERROR_SUMMARY_PATH.chmod(0o644)


def main() -> None:
    error_rows: list[dict[str, object]] = []
    jobs = (
        (
            'ffhq', read_ffhq_summary, write_ffhq_summary,
            FFHQ_SUMMARY_PATH, FFHQ_SPECTRUM_PATH,
            FFHQ_FIGURE_PATH, 'FFHQ disk teacher', 1000,
        ),
        (
            'vanhateren', read_vanhateren_summary,
            write_vanhateren_summary, VANHATEREN_SUMMARY_PATH,
            VANHATEREN_SPECTRUM_PATH, VANHATEREN_FIGURE_PATH,
            r'Van Hateren $100\times100$ disk teacher', None,
        ),
    )
    for (
            dataset, reader, writer, summary_path, spectrum_path,
            figure_path, title, default_n) in jobs:
        rows = reader()
        eigenvalues, beta_proj = load_spectrum(spectrum_path)
        n = int(float(rows[0].get('n', default_n)))
        add_theory_order_columns(rows, eigenvalues, beta_proj, n)
        writer(rows)
        fixed = sorted({
            float(row['alpha']) for row in rows if row['policy'] == 'fixed'})
        if len(fixed) != 1:
            raise RuntimeError(
                f'{dataset}: expected one fixed alpha, found {fixed}')
        signal = float(np.sum(eigenvalues * beta_proj ** 2))
        plot_policy_gap_comparison(
            rows, fixed[0], signal, n, dataset_title=title,
            figure_path=figure_path)
        error_rows.extend(summarize_errors(dataset, rows))
        print(
            f'{dataset}: updated {summary_path} and {figure_path} '
            f'with {len(rows)} cached rows')
    write_error_summary(error_rows)
    print(f'Error audit: {ERROR_SUMMARY_PATH} ({len(error_rows)} rows)')


if __name__ == '__main__':
    main()

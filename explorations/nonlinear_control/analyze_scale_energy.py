"""Aggregate the 25-site x 10-model scale run into model-level energy plots."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm


MODEL_ORDER = [
    'AlexNet_training_seed_01', 'resnet50', 'resnet50_robust',
    'resnet50_clip', 'resnet50_dino', 'regnety_640', 'clipag_vitb32',
    'dinov2_vitb14_reg', 'siglip2_vitb16', 'radio_v2.5-b',
]
MODEL_LABELS = {
    'AlexNet_training_seed_01': 'AlexNet',
    'resnet50': 'RN50',
    'resnet50_robust': 'Robust\nRN50',
    'resnet50_clip': 'CLIP\nRN50†',
    'resnet50_dino': 'DINO\nRN50',
    'regnety_640': 'RegNetY\n640',
    'clipag_vitb32': 'CLIPAG',
    'dinov2_vitb14_reg': 'DINOv2',
    'siglip2_vitb16': 'SigLIP2',
    'radio_v2.5-b': 'RADIO',
}
MODEL_COLORS = {
    'AlexNet_training_seed_01': '#8c8c8c',
    'resnet50': '#4477aa',
    'resnet50_robust': '#228833',
    'resnet50_clip': '#ee7733',
    'resnet50_dino': '#66ccee',
    'regnety_640': '#aa3377',
    'clipag_vitb32': '#cc3311',
    'dinov2_vitb14_reg': '#009988',
    'siglip2_vitb16': '#aa4499',
    'radio_v2.5-b': '#997700',
}
DF2_FRACTIONS = (.10, .25, .50, .75, .90)
FINITE_METRICS = ('smooth', 'neighborhood', 'variance', 'step')


def df2_key(fraction):
    return f'{fraction:.2f}'.replace('.', '')


def kappa_at_df2(spectrum, target):
    """Solve sum_k [s_k / (s_k + kappa)]^2 = target."""
    spectrum = np.asarray(spectrum, dtype=np.float64)
    if not 0 < target < len(spectrum):
        raise ValueError('target df2 must be strictly between zero and rank')
    lo, hi = 0., float(spectrum.max())
    while np.square(spectrum / (spectrum + hi)).sum() > target:
        hi *= 2
    for _ in range(100):
        midpoint = (lo + hi) / 2
        if np.square(spectrum / (spectrum + midpoint)).sum() > target:
            lo = midpoint
        else:
            hi = midpoint
    return (lo + hi) / 2


def model_subject_summary(frame, value_column, metric_label):
    subjects = (
        frame.groupby(['model', 'subject'], as_index=False)[value_column]
        .mean().rename(columns={value_column: 'value'})
    )
    rows = []
    for model, group in subjects.groupby('model'):
        rows.append(dict(
            model=model, metric=metric_label, mean=group.value.mean(),
            subject_sd=group.value.std(ddof=1),
            subject_sem=group.value.sem(ddof=1),
            subject_min=group.value.min(), subject_max=group.value.max(),
            n_subjects=len(group), n_sites=int(
                frame.loc[frame.model == model, ['subject', 'unit']].drop_duplicates().shape[0]
            ),
        ))
    return pd.DataFrame(rows), subjects.assign(metric=metric_label)


def style_axis(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color='#dddddd', linewidth=.6, alpha=.7)
    ax.set_axisbelow(True)


def plot_exact_bars(exact, summary, subjects, output):
    metrics = [
        ('raw_energy', r'Raw local energy  $\sum_k q_k$'),
        ('trace_df2_050', r'Control trace  $T(\kappa)$, matched $df_2=375$'),
        ('unregularized_trace', r'Unregularized trace  $\sum_k q_k/s_k$'),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4))
    positions = np.arange(len(MODEL_ORDER))
    colors = [MODEL_COLORS[model] for model in MODEL_ORDER]
    labels = [MODEL_LABELS[model] for model in MODEL_ORDER]
    rng = np.random.default_rng(20260919)
    for ax, (metric, title) in zip(axes, metrics):
        model_summary = summary[summary.metric == metric].set_index('model').loc[MODEL_ORDER]
        ax.bar(
            positions, model_summary['mean'], yerr=2 * model_summary.subject_sem,
            color=colors, edgecolor='white', linewidth=.7, capsize=3,
        )
        subject_values = subjects[subjects.metric == metric]
        for index, model in enumerate(MODEL_ORDER):
            values = subject_values.loc[subject_values.model == model, 'value'].to_numpy()
            jitter = rng.uniform(-.13, .13, len(values))
            ax.scatter(
                index + jitter, values, s=17, facecolor='white', edgecolor='#222222',
                linewidth=.6, zorder=3,
            )
        ax.set_yscale('log')
        ax.set_xticks(positions, labels, rotation=40, ha='right')
        ax.set_title(title, fontsize=11)
        ax.set_ylabel('RGB-coordinate energy (log scale)')
        style_axis(ax)
    fig.suptitle(
        'Exact local feature-space energy across 10 encoding models\n'
        'Bars: mean of 5 subject means; dots: subjects; error bars: ±2 subject-level SE',
        fontsize=13,
    )
    fig.text(
        .99, .01,
        'Each subject mean averages 5 selected sites × 10 fixed seed images.  '
        '† CLIP RN50 finite-difference QC is marginal; exact VJP shown here is unaffected.',
        ha='right', va='bottom', fontsize=8,
    )
    fig.tight_layout(rect=(0, .075, 1, .91))
    for extension in ('png', 'pdf'):
        fig.savefig(output / f'model_exact_energy_bars.{extension}', dpi=180)
    plt.close(fig)


def plot_finite_bars(summary, output, metrics, filename, title, note):
    taus = sorted(summary.noise_255.unique())
    fig, axes = plt.subplots(2, 2, figsize=(18, 9), sharex=True)
    positions = np.arange(len(MODEL_ORDER))
    width = .72 / len(metrics)
    palette = {
        'smooth': '#228833', 'neighborhood': '#777777',
        'variance': '#4477aa', 'step': '#ee7733',
    }
    names = {
        'smooth': 'Smoothed Jacobian', 'neighborhood': 'Neighborhood energy',
        'variance': 'Response covariance', 'step': 'Total response change',
    }
    for ax, tau in zip(axes.flat, taus):
        part = summary[summary.noise_255 == tau]
        for metric_index, metric in enumerate(metrics):
            values = part[part.metric == metric].set_index('model').loc[MODEL_ORDER]
            offset = (metric_index - (len(metrics) - 1) / 2) * width
            bars = ax.bar(
                positions + offset, values['mean_ratio'], width,
                yerr=2 * values['subject_sem_ratio'], color=palette[metric],
                edgecolor='white', linewidth=.5, capsize=2, label=names[metric],
            )
            if metric in {'smooth', 'neighborhood'}:
                bars[MODEL_ORDER.index('resnet50_clip')].set_hatch('///')
        ax.axhline(1, color='black', linestyle='--', linewidth=1)
        ax.set_yscale('log')
        ax.set_title(rf'Noise SD $\tau\times255={tau:g}$')
        ax.set_ylabel('Energy / exact local control trace')
        ax.set_xticks(positions, [MODEL_LABELS[model] for model in MODEL_ORDER], rotation=40, ha='right')
        style_axis(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc='upper center', bbox_to_anchor=(.5, .955),
        ncol=len(metrics), frameon=False,
    )
    fig.suptitle(title, fontsize=13, y=.995)
    fig.text(.99, .01, note, ha='right', va='bottom', fontsize=8)
    fig.tight_layout(rect=(0, .045, 1, .86))
    for extension in ('png', 'pdf'):
        fig.savefig(output / f'{filename}.{extension}', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--site-manifest', type=Path, required=True)
    parser.add_argument('--qc', type=Path, required=True)
    parser.add_argument('--tables', type=Path, required=True)
    parser.add_argument('--figures', type=Path, required=True)
    args = parser.parse_args()
    args.tables.mkdir(parents=True, exist_ok=True)
    args.figures.mkdir(parents=True, exist_ok=True)
    site_manifest = pd.read_csv(args.site_manifest)
    qc = pd.read_csv(args.qc)[['geometry_id', 'fd_quality']]
    site_manifest = site_manifest.merge(qc, on='geometry_id', validate='many_to_one')
    if set(site_manifest.model) != set(MODEL_ORDER) or len(site_manifest) != 250:
        raise ValueError('Expected exactly 250 rows across the ten registered models')

    geometry_cache = {}
    for geometry_id in tqdm(site_manifest.geometry_id.unique(), desc='Loading geometry summaries'):
        path = args.input / 'geometry' / geometry_id / 'summary.npz'
        with np.load(path) as values:
            spectrum = values['spectrum'].astype(np.float64)
            exact = values['exact_by_seed'].astype(np.float64)
            record = dict(
                spectrum=spectrum, exact=exact, tau=values['tau'].astype(np.float64),
                raw_energy=exact.sum(1),
                unregularized_trace=(exact / spectrum[None, :]).sum(1),
            )
            for fraction in DF2_FRACTIONS:
                kappa = kappa_at_df2(spectrum, fraction * len(spectrum))
                weight = spectrum / np.square(spectrum + kappa)
                key = df2_key(fraction)
                record[f'kappa_df2_{key}'] = kappa
                record[f'trace_df2_{key}'] = exact @ weight
                if fraction == .50:
                    for metric in FINITE_METRICS:
                        record[metric] = np.einsum(
                            'itk,k->it', values[f'{metric}_mean_by_seed'], weight,
                            dtype=np.float64,
                        )
        geometry_cache[geometry_id] = record

    exact_rows, finite_rows = [], []
    for site in tqdm(site_manifest.itertuples(index=False), total=len(site_manifest), desc='Expanding sites'):
        record = geometry_cache[site.geometry_id]
        for seed_index in range(record['exact'].shape[0]):
            exact_row = dict(
                model=site.model, subject=site.subject, unit=site.unit,
                geometry_id=site.geometry_id, layer=site.layer,
                fd_quality=site.fd_quality, seed=seed_index + 1,
                raw_energy=record['raw_energy'][seed_index],
                unregularized_trace=record['unregularized_trace'][seed_index],
            )
            for fraction in DF2_FRACTIONS:
                key = df2_key(fraction)
                exact_row[f'kappa_df2_{key}'] = record[f'kappa_df2_{key}']
                exact_row[f'trace_df2_{key}'] = record[f'trace_df2_{key}'][seed_index]
            exact_rows.append(exact_row)
            exact_reference = record['trace_df2_050'][seed_index]
            for tau_index, tau in enumerate(record['tau']):
                for metric in FINITE_METRICS:
                    value = record[metric][seed_index, tau_index]
                    finite_rows.append(dict(
                        model=site.model, subject=site.subject, unit=site.unit,
                        geometry_id=site.geometry_id, layer=site.layer,
                        fd_quality=site.fd_quality, seed=seed_index + 1,
                        metric=metric, tau=tau, noise_255=tau * 255,
                        energy=value, exact_energy=exact_reference,
                    ))
    exact = pd.DataFrame(exact_rows)
    finite = pd.DataFrame(finite_rows)
    exact.to_csv(args.tables / 'site_seed_exact_energy.csv.gz', index=False, compression='gzip')
    finite.to_csv(args.tables / 'site_seed_finite_energy.csv.gz', index=False, compression='gzip')

    exact_summaries, exact_subjects = [], []
    exact_metrics = ['raw_energy', 'unregularized_trace'] + [
        f'trace_df2_{df2_key(fraction)}' for fraction in DF2_FRACTIONS
    ]
    for metric in exact_metrics:
        summary, subjects = model_subject_summary(exact, metric, metric)
        exact_summaries.append(summary)
        exact_subjects.append(subjects)
    exact_summary = pd.concat(exact_summaries, ignore_index=True)
    subject_exact = pd.concat(exact_subjects, ignore_index=True)
    exact_summary.to_csv(args.tables / 'model_exact_energy_summary.csv', index=False)
    subject_exact.to_csv(args.tables / 'subject_exact_energy.csv', index=False)

    subject_finite = (
        finite.groupby(['model', 'subject', 'metric', 'noise_255'], as_index=False)
        [['energy', 'exact_energy']].mean()
    )
    subject_finite['ratio'] = subject_finite.energy / subject_finite.exact_energy
    finite_summary = (
        subject_finite.groupby(['model', 'metric', 'noise_255'])
        .agg(
            mean_ratio=('ratio', 'mean'), subject_sd_ratio=('ratio', 'std'),
            subject_sem_ratio=('ratio', 'sem'), subject_min_ratio=('ratio', 'min'),
            subject_max_ratio=('ratio', 'max'), n_subjects=('ratio', 'count'),
        ).reset_index()
    )
    subject_finite.to_csv(args.tables / 'subject_finite_energy.csv', index=False)
    finite_summary.to_csv(args.tables / 'model_finite_energy_summary.csv', index=False)

    plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42, 'font.size': 9})
    plot_exact_bars(exact, exact_summary, subject_exact, args.figures)
    plot_finite_bars(
        finite_summary, args.figures, ('variance', 'step'),
        'model_finite_response_bars',
        r'Forward-only finite-noise response energy at matched $df_2=375$',
        'Bars: mean over 5 subject-level ratios; error bars: ±2 SE.  '
        'Both estimators are independent of the finite-difference step h.',
    )
    plot_finite_bars(
        finite_summary, args.figures, ('smooth', 'neighborhood'),
        'model_smoothed_gradient_bars',
        r'Smoothed and neighborhood gradient energy at matched $df_2=375$',
        'Smoothed = ||E[J]||²; neighborhood = E[||J||²]. Hatched CLIP RN50 bars use a marginal h diagnostic.',
    )
    print(exact_summary[exact_summary.metric == 'trace_df2_050'].sort_values('mean', ascending=False).to_string(index=False))
    print(finite_summary.to_string(index=False))


if __name__ == '__main__':
    main()

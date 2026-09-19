"""Matched-seed geometry comparison, with explicit opt-in for different layers."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def kappa_at_df2(s, target):
    lo, hi = 0., float(s.max())
    while np.sum((s/(s+hi))**2) > target:
        hi *= 2
    for _ in range(100):
        mid = (lo+hi)/2
        if np.sum((s/(s+mid))**2) > target:
            lo = mid
        else:
            hi = mid
    return (lo+hi)/2


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--standard', '--first', dest='standard', type=Path, required=True)
    ap.add_argument('--robust', '--second', dest='robust', type=Path, required=True)
    ap.add_argument('--first-label', default='Standard RN50')
    ap.add_argument('--second-label', default='Robust RN50')
    ap.add_argument('--allow-different-layers', action='store_true')
    ap.add_argument('--filename-base', default='rn50_comparison')
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    paths = [args.standard, args.robust]
    audits = [json.loads((p/'audit.json').read_text()) for p in paths]
    images = [json.loads((p/'evaluation_images.json').read_text()) for p in paths]
    if not args.allow_different_layers:
        assert audits[0]['layer'] == audits[1]['layer'], 'Different layers require explicit opt-in'
    assert audits[0]['n_train'] == audits[1]['n_train']
    assert images[0] == images[1], 'Must compare the same seed images in the same order'
    plt.rcParams['pdf.fonttype'] = 42
    fig, axs = plt.subplots(2, 2, figsize=(11, 8))
    rows, seed_traces = [], []
    labels = [args.first_label, args.second_label]
    for path, label, color in zip(paths, labels, ['#c04a36', '#2378ad']):
        d = np.genfromtxt(path/'pc_summary.csv', names=True, delimiter=',')
        s, q, rank = d['variance'], d['mean_power'], d['pc']
        assert len(s) == 750
        axs[0, 0].loglog(rank, s/s[0], label=label, color=color)
        axs[0, 1].loglog(rank, q/s, label=label, color=color)
        fractions = np.linspace(.01, .99, 120)
        kappas = np.array([kappa_at_df2(s, f*len(s)) for f in fractions])
        traces = ((q/s)[None]*(s[None]/(s[None]+kappas[:, None]))**2).sum(1)
        axs[1, 0].semilogy(fractions, traces, label=label, color=color)
        raw = np.genfromtxt(path/'gradient_power.csv', names=True, delimiter=',')
        power = np.stack([raw[raw['dataframe_row']==img['row']]['rgb_gradient_power'] for img in images[0]])
        for f in [.1, .25, .5, .75, .9, .99]:
            k = kappa_at_df2(s, f*len(s))
            weights = s/(s+k)**2
            per_seed = power @ weights
            c = q*weights
            rows.append(dict(model=label, df2_fraction=f, kappa=k,
                             kappa_over_s1=k/s[0], trace=c.sum(),
                             trace_over_df2=c.sum()/(f*len(s)),
                             tail250_fraction=c[500:].sum()/c.sum(),
                             seed_trace_min=per_seed.min(), seed_trace_max=per_seed.max()))
            if f == .5:
                seed_traces.append(per_seed)
                axs[1, 1].semilogy(np.arange(1, len(per_seed)+1), per_seed, 'o-', label=label, color=color)
        rows.append(dict(model=label, df2_fraction=1., kappa=0., kappa_over_s1=0.,
                         trace=(q/s).sum(), trace_over_df2=(q/s).mean(),
                         tail250_fraction=(q/s)[500:].sum()/(q/s).sum(),
                         seed_trace_min=(power/s).sum(1).min(), seed_trace_max=(power/s).sum(1).max()))
    axs[0, 0].set(xlabel='PC rank', ylabel='Training variance / leading variance')
    axs[0, 1].set(xlabel='PC rank', ylabel='q_k / s_k, RGB coordinates')
    axs[1, 0].set(xlabel='Matched df₂ / 750 = mean shrinkage²', ylabel='Seed-mean trace T(κ)')
    axs[1, 1].set(xlabel='Experiment seed index', ylabel='Per-seed T(κ), matched df₂ / 750 = 0.5')
    for ax in axs.flat:
        ax.legend(fontsize=8)
        ax.spines[['top', 'right']].set_visible(False)
    layer_title = audits[0]['layer'] if audits[0]['layer']==audits[1]['layer'] else 'different architectures / selected layers'
    fig.suptitle(f"{labels[0]} vs {labels[1]} | {len(images[0])} fixed seeds\n"
                 f'{layer_title}; separate training PCA750; RGB metric')
    fig.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(args.output/f'{args.filename_base}.{ext}', dpi=150)
    with (args.output/'matched_df2.csv').open('w') as file:
        writer = csv.DictWriter(
            file, fieldnames=list(rows[0]), lineterminator='\n'
        )
        writer.writeheader(); writer.writerows(rows)
    ratios = seed_traces[0]/seed_traces[1]
    np.savetxt(args.output/'paired_seed_traces.csv',
               np.column_stack((np.arange(1, len(images[0])+1), seed_traces[0], seed_traces[1], ratios)),
               delimiter=',', header='seed_index,first_trace,second_trace,first_over_second', comments='')
    (args.output/'comparison_config.json').write_text(json.dumps(dict(
        first=str(paths[0]), second=str(paths[1]), labels=labels, audits=audits,
        different_layers=audits[0]['layer']!=audits[1]['layer']), indent=2))
    print(f'Per-seed {labels[0]}/{labels[1]} T ratios at matched df2/750=0.5:', ratios)
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()

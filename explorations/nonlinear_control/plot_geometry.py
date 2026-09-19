"""Replot cached PC statistics without model inference or Jacobian computation."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from geometry import trace_contributions


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('input', type=Path)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    d = np.genfromtxt(args.input/'pc_summary.csv', delimiter=',', names=True)
    audit = json.loads((args.input/'audit.json').read_text())
    raw = np.genfromtxt(args.input/'gradient_power.csv', delimiter=',', names=True)
    n_images = len(np.unique(raw['dataframe_row']))
    rank, s, q = d['pc'], d['variance'], d['mean_power']
    kappas = s.max()*np.array([1e-4, 1e-3, 1e-2, .1, 1.])
    contributions = trace_contributions(s, q, kappas)
    plt.rcParams['pdf.fonttype'] = 42
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes[0, 0].loglog(rank, s)
    axes[0, 0].set(ylabel='Training feature variance s_k', xlabel='PC rank')
    for row in np.unique(raw['dataframe_row']):
        image_q = raw[raw['dataframe_row']==row]['rgb_gradient_power']
        if len(image_q) == len(rank):
            axes[0, 1].loglog(rank, image_q, color='gray', alpha=.12, linewidth=.5)
            axes[1, 0].loglog(rank, image_q/s, color='gray', alpha=.12, linewidth=.5)
    axes[0, 1].loglog(rank, q)
    axes[0, 1].set(ylabel='Mean RGB gradient power q_k', xlabel='PC rank')
    axes[1, 0].loglog(rank, q/s)
    axes[1, 0].set(ylabel='q_k / s_k (zero-kappa weight)', xlabel='PC rank')
    for kappa, c in zip(kappas, contributions):
        axes[1, 1].plot(rank, c.cumsum()/c.sum(), label=f'κ/s₁={kappa/s.max():g}')
    axes[1, 1].set(xlabel='Retained PC rank', ylabel='Cumulative fraction of retained trace', ylim=(0, 1.02))
    axes[1, 1].legend(fontsize=8)
    for ax in axes.flat:
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle(f"{audit['model']} {audit['layer']} | {audit['n_train']} training images\n"
                 f"Geometry pilot: {n_images} {audit.get('evaluation', 'training')} images, {len(rank)} PCs; empirical spectrum")
    fig.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(args.output/f'pc_geometry.{ext}', dpi=150)
    np.savetxt(args.output/'cumulative_trace.csv',
               np.column_stack([rank]+[c.cumsum()/c.sum() for c in contributions]),
               delimiter=',', header='pc,'+','.join(f'kappa_{k:g}' for k in kappas), comments='')
    print(args.output/'pc_geometry.png')


if __name__ == '__main__':
    main()

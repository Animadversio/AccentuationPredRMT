"""Replay paired fits to cache missing slopes; extend the three-policy figure."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/accentuationpredrmt-matplotlib')
import argparse
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from scripts import pixel_ridge_notebook_utils as u
from scripts.run_vanhateren_selection_estimation import GaussianRisk
from scripts.validate_ffhq_disk_teacher import create_disk_teacher


def compute(work):
    root = Path(__file__).resolve().parents[1]
    out = root/'notebooks/outputs/pixel_ridge/vanhateren_selection_estimation'
    table = pd.read_csv(out/'comparison.csv')
    raw_old = pd.read_csv(work/'mc_raw.csv')
    torch.set_num_threads(4)
    s, b = u.load_problem(root, 'vanhateren')
    S = float(s@b**2)
    gauss = GaussianRisk(s, b, 1000, 8192, 20260917)
    start = time.perf_counter()
    for idx, r in tqdm(table.iterrows(), total=len(table), desc='Gaussian slope and R2'):
        k = r.kappa
        mean = s/(s+k)*b
        v = (r.E_gen*S+r.sigma**2)/1000*s/(s+k)**2
        a = torch.as_tensor(mean, device='cuda')
        sd = torch.as_tensor(np.sqrt(v), device='cuda')
        N = (a*gauss.b).sum()+gauss.z@(sd*gauss.b)
        D = a.square().sum()+2*(gauss.z@(a*sd))+gauss.z2@(sd.square())
        for name, values in [('slope_acc', N/D), ('R2_acc', 1-(D/N-1).square())]:
            table.loc[idx, name+'_distribution'] = values.mean().item()
            table.loc[idx, name+'_distribution_se'] = values.std().item()/np.sqrt(len(values))
        table.loc[idx, 'gaussian_min_abs_N'] = N.abs().min().item()
        table.loc[idx, 'slope_gen'] = (s@ (mean*b))/(s@(mean**2+v))
        if idx == 0:
            print(f'Gaussian pilot ETA {(time.perf_counter()-start)*len(table):.1f}s', flush=True)
    del gauss
    data = np.load(work/'vanhateren_log_gray_100px_22000_uint8.npy', mmap_mode='r')
    train = torch.as_tensor(np.array(data[:2000], dtype=np.float64), device='cuda')/255
    pop = torch.as_tensor(np.array(data[2000:], dtype=np.float64), device='cuda')/255
    pop -= pop.mean(0, keepdim=True)
    beta = torch.as_tensor(create_disk_teacher(100, .3), device='cuda', dtype=torch.float64)
    target = pop@beta
    alpha = torch.as_tensor(table.alpha.to_numpy(), device='cuda')
    sigma = torch.as_tensor(table.sigma.to_numpy(), device='cuda')
    frames = []
    for i in tqdm(range(100), desc='Replay same MC seeds, add generalization slope'):
        path = work/f'extended_trial_{i:04d}.csv'
        if path.exists():
            frame = pd.read_csv(path)
        else:
            torch.cuda.synchronize(); start = time.perf_counter()
            g = torch.Generator(device='cuda').manual_seed(20260918+i)
            ids = torch.randperm(len(train), device='cuda', generator=g)[:1000]
            X = train[ids].clone(); X -= X.mean(0, keepdim=True)
            eps = torch.randn(1000, device='cuda', generator=g, dtype=torch.float64)
            eps -= eps.mean()
            eig, Q = torch.linalg.eigh(X@X.T); eig = eig.clamp_min(0)
            y = X@beta
            response = (Q.T@y)[:, None]+(Q.T@eps)[:, None]*sigma[None, :]
            w = X.T@(Q@(response/(eig[:, None]+alpha[None, :])))
            prediction = pop@w
            slope = (target@prediction)/prediction.square().sum(0)
            Eg = (prediction-target[:, None]).square().mean(0)/S
            R = (beta@w)/w.square().sum(0)
            frame = table[['policy', 'ratio', 'sigma', 'alpha']].copy()
            frame['trial'] = i
            frame['slope_gen'] = slope.cpu().numpy()
            frame['slope_acc'] = R.cpu().numpy()
            frame['R2_gen'] = 1-Eg.cpu().numpy()
            frame['R2_acc'] = (1-(1/R-1).square()).cpu().numpy()
            old = raw_old[raw_old.trial == i].reset_index(drop=True)
            assert list(old.policy) == list(frame.policy)
            np.testing.assert_allclose(Eg.cpu().numpy(), old.E_gen, rtol=1e-6, atol=1e-12)
            np.testing.assert_allclose(frame.slope_acc, old.slope_acc, rtol=1e-6, atol=1e-10)
            frame.to_csv(path, index=False)
            torch.cuda.synchronize()
            if i == 0:
                elapsed = time.perf_counter()-start
                print(f'MC pilot {elapsed:.2f}s; total ETA {100*elapsed:.1f}s', flush=True)
                if elapsed*100 > 1800:
                    raise RuntimeError('Pilot exceeds 30 minutes; profile before scaling')
        frames.append(frame)
    raw = pd.concat(frames, ignore_index=True)
    raw.to_csv(work/'extended_mc_raw.csv', index=False)
    for idx, r in table.iterrows():
        q = raw[(raw.policy == r.policy)&np.isclose(raw.ratio, r.ratio, rtol=1e-10, atol=0)]
        assert len(q) == 100
        for name in ['R2_gen', 'R2_acc', 'slope_gen', 'slope_acc']:
            for suffix, value in [('', q[name].mean()), ('_se', q[name].sem()),
                                  ('_q10', q[name].quantile(.1)), ('_q90', q[name].quantile(.9))]:
                table.loc[idx, name+'_mc'+suffix] = value
    table.to_csv(out/'comparison_extended.csv', index=False)
    return table


def plot(table, sklearn_cv=None, show_sklearn_cv=False, stationary=None):
    """Compare empirical and DE generalization selection in the middle column only."""
    import matplotlib.pyplot as plt
    # Embed TrueType fonts as searchable vector text, not Type-3 glyphs.
    plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'svg.fonttype': 'none', 'font.family': 'DejaVu Sans',
                         'mathtext.fontset': 'dejavusans'})
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    fig, axes = plt.subplots(4, 3, figsize=(17, 15), sharex='col', sharey='row',
                             gridspec_kw={'height_ratios': [1.4, 1, 1, .65]})
    titles = ['Fixed regularization', r'$E_{gen}$ selection (DE)',
              r'$E_{acc}$ selection (Gaussian oracle)']
    colors = ['#15579b', '#489fce', '#bb4a0a', '#ec9a25', '#ee6633']
    extrapolated = stationary is not None and 'extrapolated' in stationary and stationary.extrapolated.any()
    if show_sklearn_cv:
        if sklearn_cv is None:
            raise ValueError('Pass cached sklearn_cv summary when enabling overlay')
        sklearn_cv = sklearn_cv.sort_values('sigma')
    for j, policy in enumerate(['fixed', 'DE_gen_CV', 'Gaussian_acc_oracle']):
        overlay_cv = show_sklearn_cv and policy == 'DE_gen_CV'
        p = table[table.policy == policy].sort_values('sigma')
        x = p.sigma.to_numpy()
        if overlay_cv:
            np.testing.assert_allclose(sklearn_cv.sigma, x, rtol=1e-10, atol=0)
        for i, (gen, acc, dist) in enumerate([
                ('E_gen', 'E_acc', 'acc_distribution'),
                ('R2_gen', 'R2_acc', 'R2_acc_distribution'),
                ('slope_gen', 'slope_acc', 'slope_acc_distribution')]):
            ax = axes[i, j]
            ax.plot(x, p[gen], color=colors[0], lw=2)
            ax.plot(x, p[acc], color=colors[2], ls='--', lw=2)
            ax.plot(x, p[dist], color=colors[3], ls='-.', lw=2)
            for name, color, marker in [(gen, colors[1], 'o'), (acc, colors[4], 'D')]:
                ax.fill_between(x, p[name+'_mc_q10'], p[name+'_mc_q90'], color=color, alpha=.18)
                ax.errorbar(x, p[name+'_mc'], yerr=2*p[name+'_mc_se'], fmt=marker,
                            mfc='white', color=color, ms=4, capsize=2, zorder=5)
            if overlay_cv:
                for name, color, marker in [(gen, colors[0], '^'), (acc, colors[2], 'v')]:
                    q = sklearn_cv
                    ax.fill_between(x, q[name+'_mc_q10'], q[name+'_mc_q90'],
                                    color=color, alpha=.08, hatch='//', linewidth=0)
                    ax.errorbar(x, q[name+'_mc'], yerr=2*q[name+'_mc_se'],
                                color=color, ls=':', marker=marker, ms=4, capsize=2,
                                lw=1.3, zorder=6)
            if stationary is not None and policy == 'DE_gen_CV':
                q = stationary.sort_values('sigma')
                np.testing.assert_allclose(q.sigma, x)
                ax.plot(x, q[acc], color='#873c16', ls=(0, (8, 3, 2, 3)),
                        marker='x', markevery=3, ms=5, lw=1.8, zorder=8)
                if extrapolated:
                    mask = q.extrapolated.to_numpy(dtype=bool)
                    ax.plot(x[mask], q.loc[mask, acc], ls='none', marker='s',
                            mfc='white', color='#873c16', ms=5, zorder=9)
            if i == 0:
                ax.set_yscale('log')
            elif i == 1:
                ax.set_yscale('symlog', linthresh=1)
                ax.axhline(0, color='gray', ls=':', lw=.8)
                ax.axhline(1, color='gray', ls=':', lw=.8)
            else:
                ax.axhline(1, color='gray', ls=':', lw=.8)
        axes[3, j].plot(x, p.kappa, color='#644191', label=r'$\kappa$')
        axes[3, j].plot(x, p.lam, '--', color='#777777', label=r'$\lambda$')
        axes[3, j].set_yscale('log')
        if stationary is not None and policy == 'DE_gen_CV':
            q = stationary.sort_values('sigma')
            axes[3, j].plot(x, q.kappa, color='#644191', ls=(0, (8, 3, 2, 3)),
                           marker='x', markevery=3, ms=4, label=r'Paper $\kappa^*$')
            axes[3, j].plot(x, q.lam, color='#777777', ls=(0, (8, 3, 2, 3)),
                           marker='x', markevery=3, ms=4, label=r'Paper $\lambda^*$')
        if overlay_cv:
            q = sklearn_cv
            axes[3, j].fill_between(x, q.lam_mc_q10, q.lam_mc_q90,
                                   color='#269676', alpha=.2, label='sklearn λ 10–90%')
            axes[3, j].plot(x, q.lam_mc_median, ':', marker='s', ms=3,
                           color='#16785c', label='sklearn λ median')
        axes[3, j].legend(fontsize=9)
        for ax in axes[:, j]:
            ax.set_xscale('symlog', linthresh=x[x > 0].min())
            ax.set_xlim(0, x.max()*1.05)
            ax.set_xticks([0, .1, 1, 10, 100])
            ax.set_xticklabels(['0', '0.1', '1', '10', '100'])
            ax.grid(alpha=.18)
        top = axes[0, j].twiny()
        top.set_xscale('symlog', linthresh=x[x > 0].min())
        top.set_xlim(axes[0, j].get_xlim())
        ticks = [0, 1e-6, 1e-4, 1e-2, 1, 10]
        top.set_xticks(np.sqrt(np.asarray(ticks)*p.S.iloc[0]))
        top.set_xticklabels([f'{v:g}' for v in ticks])
        top.set_xlabel(r'Noise / signal variance $\sigma^2/S$')
        title = r'$E_{gen}$ selection: DE vs sklearn CV' if overlay_cv else titles[j]
        top.set_title(title, pad=14)
        axes[3, j].set_xlabel(r'Response noise $\sigma$')
    for ax, label in zip(axes[:, 0], ['Error / signal variance: E/S',
                                    r'$R^2$ (symlog)', 'True-on-fitted slope', 'Regularization']):
        ax.set_ylabel(label)
    handles = [Line2D([], [], color=colors[0], label='Gen: DE'),
               Line2D([], [], color=colors[1], marker='o', mfc='white', ls='', label='Gen: MC mean'),
               Line2D([], [], color=colors[2], ls='--', label='Acc: leading DE'),
               Line2D([], [], color=colors[3], ls='-.', label='Acc: Gaussian surrogate*'),
               Line2D([], [], color=colors[4], marker='D', mfc='white', ls='', label='Acc: MC mean')]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .963), ncol=5)
    extra = [Patch(color=colors[1], alpha=.18, label='Gen: MC 10–90% trials'),
             Patch(color=colors[4], alpha=.18, label='Acc: MC 10–90% trials')]
    if show_sklearn_cv:
        extra += [Line2D([], [], color=colors[0], ls=':', marker='^', label='Gen: sklearn CV MC (middle only)'),
                  Line2D([], [], color=colors[2], ls=':', marker='v', label='Acc: sklearn CV MC (middle only)')]
    fig.legend(handles=extra,
               loc='upper center', bbox_to_anchor=(.5, .936), ncol=4 if show_sklearn_cv else 2)
    if stationary is not None:
        fig.legend(handles=[Line2D([], [], color='#873c16', ls=(0, (8, 3, 2, 3)),
                    marker='x', label=('Acc: paper formula + boundary substitution (open squares)' if extrapolated
                                      else r'Acc: paper interior stationary formula ($n=1000$; middle only)'))],
                   loc='upper center', bbox_to_anchor=(.5, .910))
    fig.suptitle('Van Hateren disk teacher: full 10,000 pixels, n=1,000', y=.99, fontsize=18)
    fig.text(.5, .032, 'MC: arithmetic mean of 100 trials; error bars ±2 SE; shading: trial 10–90%, not confidence intervals.', ha='center')
    fig.text(.5, .016, '*Gaussian surrogate: 8,192 draws; R² acc uses per-draw inverse slopes (not transformed mean slopes).', ha='center')
    if show_sklearn_cv:
        fig.text(.5, .002, 'Middle column only: triangles/dotted lines = empirical sklearn LOOCV; hatched bands = its 10–90% trials.', ha='center')
    if stationary is not None:
        note = ('Open squares: direct formula at boundary κ₀ (not stationary); λ*=0 omitted on log axis.'
                if extrapolated else 'Paper curve omitted where no feasible interior minimum exists; DE selection uses n−1 LOOCV then refits at n.')
        fig.text(.5, -.012, note, ha='center')
    fig.subplots_adjust(top=.825 if stationary is not None else .855, bottom=.075, hspace=.17, wspace=.12)
    return fig


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--work', type=Path)
    parser.add_argument('--plot-only', action='store_true')
    parser.add_argument('--show-sklearn-cv', action='store_true')
    parser.add_argument('--show-stationary', action='store_true')
    parser.add_argument('--extrapolate-stationary', action='store_true')
    args = parser.parse_args()
    out = Path(__file__).resolve().parents[1]/'notebooks/outputs/pixel_ridge/vanhateren_selection_estimation'
    table = pd.read_csv(out/'comparison_extended.csv') if args.plot_only else compute(args.work)
    cv = pd.read_csv(out/'sklearn_cv_summary.csv') if args.show_sklearn_cv else None
    stationary = pd.read_csv(out/'stationary_cv.csv') if args.show_stationary else None
    if args.extrapolate_stationary:
        stationary = pd.read_csv(out/'stationary_cv_extrapolated.csv')
    fig = plot(table, sklearn_cv=cv, show_sklearn_cv=args.show_sklearn_cv, stationary=stationary)
    suffix = '_with_sklearn_cv' if args.show_sklearn_cv else ''
    if args.show_stationary:
        suffix += '_stationary'
    if args.extrapolate_stationary:
        suffix += '_extrapolated'
    for ext in ['png', 'pdf']:
        fig.savefig(out/f'selection_vs_estimation_extended{suffix}.{ext}', dpi=160, bbox_inches='tight')

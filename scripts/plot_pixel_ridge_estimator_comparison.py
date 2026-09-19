"""Separate selection policies (columns) from risk estimators (line styles)."""
from pathlib import Path
import time
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from scripts import pixel_ridge_notebook_utils as u


def prepare_tables(summary, gaussian, raw_mc, s, beta, n, draws=8192, seed=20260917):
    """Reuse actual ridge fits; independently integrate Gaussian risk per policy.

    Only use noise levels shared with the Gaussian-selected policy. Every
    estimator in one column uses exactly the same selected regularization.
    """
    selected = []
    for row in gaussian.itertuples():
        for policy in ['fixed', 'DE_gen_CV', 'Gaussian_acc_oracle']:
            source = gaussian if policy == 'Gaussian_acc_oracle' else summary
            match = source[(source.policy == policy) &
                           np.isclose(source.ratio, row.ratio, rtol=1e-10, atol=0)]
            if len(match) != 1:
                raise ValueError(f'Expected one selected setting: {policy}, {row.ratio}')
            selected.append(match.iloc[0].to_dict())
    normal = np.random.default_rng(seed).normal(size=(draws, len(s)))
    first = selected[0]
    start = time.perf_counter()
    u.gaussian_acc_risk(s,beta,n,first['sigma'],first['alpha'],normal)
    print(f'Independent distribution evaluation ETA: {(time.perf_counter()-start)*len(selected):.2f}s',flush=True)
    rows = []
    for r in tqdm(selected, desc='Evaluate all estimators at selected alpha'):
        sample = raw_mc[(raw_mc.policy == r['policy']) &
                        np.isclose(raw_mc.ratio, r['ratio'],rtol=1e-10,atol=0)]
        if len(sample) < 2 or not np.allclose(sample.alpha,r['alpha']):
            raise ValueError('Missing or mismatched MC fits; do not compare different alphas')
        value,se = u.gaussian_acc_risk(s,beta,n,r['sigma'],r['alpha'],normal)
        r.update(acc_distribution=value, acc_distribution_se=se, trials=len(sample))
        for metric in ['E_gen','E_acc']:
            r[metric+'_mc'] = sample[metric].mean()
            r[metric+'_mc_se'] = sample[metric].sem()
        rows.append(r)
    return pd.DataFrame(rows)


def add_mc_variability(table, raw_mc):
    """Attach empirical trial spread; keep arithmetic mean as the plotted center."""
    table=table.copy()
    for index,r in table.iterrows():
        sample=raw_mc[(raw_mc.policy==r.policy)&
                      np.isclose(raw_mc.ratio,r.ratio,rtol=1e-10,atol=0)]
        if len(sample)<2 or sample.trial.duplicated().any():
            raise ValueError('Need distinct repeated trials for each setting')
        if not np.allclose(sample.alpha,r.alpha,rtol=1e-9,atol=0):
            raise ValueError('MC alpha differs from selected alpha')
        table.loc[index,'trials']=len(sample)
        for metric in ['E_gen','E_acc']:
            np.testing.assert_allclose(sample[metric].mean(),r[metric+'_mc'],rtol=1e-9,atol=1e-18)
            for suffix,value in [('std',sample[metric].std()),
                                 ('median',sample[metric].median()),
                                 ('q10',sample[metric].quantile(.1)),
                                 ('q90',sample[metric].quantile(.9))]:
                table.loc[index,metric+'_mc_'+suffix]=value
    return table


def plot_comparison(table):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    show_spread=all(m+'_mc_q10' in table and m+'_mc_q90' in table for m in ['E_gen','E_acc'])
    policies = ['fixed','DE_gen_CV','Gaussian_acc_oracle']
    titles = ['Fixed regularization',r'$E_{\rm gen}$ selection (DE)',
              r'$E_{\rm acc}$ selection (distributional oracle)']
    blue, lightblue = '#15579b','#489fce'
    orange, gold, darkorange = '#bb4a0a','#ec9a25','#ee6633'
    fig,axes = plt.subplots(2,3,figsize=(16,8),sharex='col',sharey='row',
                            gridspec_kw={'height_ratios':[2.3,1]})
    for j,(policy,title) in enumerate(zip(policies,titles)):
        p=table[table.policy==policy].sort_values('sigma')
        x=p.sigma.to_numpy()
        ax=axes[0,j]
        if show_spread:
            for metric,color in [('E_gen',lightblue),('E_acc',darkorange)]:
                ax.fill_between(x,np.maximum(p[metric+'_mc_q10'],1e-16),
                                np.maximum(p[metric+'_mc_q90'],1e-16),
                                color=color,alpha=.2,zorder=1,linewidth=0)
        ax.plot(x,p.E_gen,color=blue,lw=2.3)
        ax.errorbar(x,p.E_gen_mc,yerr=2*p.E_gen_mc_se,fmt='o',mfc='white',
                    mec=lightblue,ecolor=lightblue,ms=7,capsize=3,zorder=6)
        ax.plot(x,np.maximum(p.E_acc,1e-16),color=orange,lw=2,ls='--')
        ax.plot(x,p.acc_distribution,color=gold,lw=2.3,ls='-.')
        ax.errorbar(x,p.acc_distribution,yerr=2*p.acc_distribution_se,
                    fmt='none',ecolor=gold,alpha=.45,capsize=2)
        ax.errorbar(x,p.E_acc_mc,yerr=2*p.E_acc_mc_se,fmt='D',mfc='white',
                    mec=darkorange,ecolor=darkorange,ms=6,capsize=3,zorder=7)
        ax.set_title(title,pad=12,fontsize=12)
        ax.set_yscale('log')
        axes[1,j].plot(x,p.kappa,'-',color='#644191',lw=2,label=r'$\kappa$')
        axes[1,j].plot(x,p.lam,'--',color='#777777',lw=2,label=r'$\lambda$')
        axes[1,j].set_yscale('log')
        axes[1,j].legend(loc='best',fontsize=9)
        for a in axes[:,j]:
            if np.any(x == 0):
                a.set_xscale('symlog',linthresh=x[x>0].min())
                a.set_xlim(0,x.max()*1.05)
                powers=np.arange(np.ceil(np.log10(x[x>0].min())),np.floor(np.log10(x.max()))+1)
                a.set_xticks(np.r_[0.,10.**powers])
                a.set_xticklabels(['0']+[rf'$10^{{{int(p)}}}$' for p in powers])
            else:
                a.set_xscale('log')
            a.grid(alpha=.18,which='major')
        axes[1,j].set_xlabel(r'Response noise $\sigma$')
        S=float(p.S.iloc[0])
        if np.any(x == 0):
            top=ax.twiny()
            top.set_xscale('symlog',linthresh=x[x>0].min())
            top.set_xlim(ax.get_xlim())
            ticks=[0,1e-6,1e-4,1e-2,1,10]
            top.set_xticks(np.sqrt(np.asarray(ticks)*S))
            top.set_xticklabels([f'{t:g}' for t in ticks])
        else:
            top=ax.secondary_xaxis('top',functions=(
                lambda x,S=S: np.asarray(x)**2/S,
                lambda x,S=S: np.sqrt(np.maximum(x,0)*S)))
        top.set_xlabel(r'Noise / signal variance $\sigma^2/S$',labelpad=7)
    axes[0,0].set_ylabel(r'Error / signal variance: $E/S$')
    axes[1,0].set_ylabel('Selected regularization')
    handles=[Line2D([],[],color=blue,lw=2,label=r'$E_{\rm gen}$: DE'),
             Line2D([],[],color=lightblue,marker='o',mfc='white',ls='none',label=r'$E_{\rm gen}$: MC mean'),
             Line2D([],[],color=orange,ls='--',lw=2,label=r'$E_{\rm acc}$: leading DE'),
             Line2D([],[],color=gold,ls='-.',lw=2,label=r'$E_{\rm acc}$: distributional DE*'),
             Line2D([],[],color=darkorange,marker='D',mfc='white',ls='none',label=r'$E_{\rm acc}$: MC mean')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.965),ncol=5,frameon=False)
    fig.suptitle('Pixel ridge: selection policy vs risk estimator',y=1.01,fontsize=17)
    if show_spread:
        fig.legend(handles=[Patch(facecolor=lightblue,alpha=.2,label=r'$E_{\rm gen}$: MC 10–90% trials'),
                            Patch(facecolor=darkorange,alpha=.2,label=r'$E_{\rm acc}$: MC 10–90% trials')],
                   loc='upper center',bbox_to_anchor=(.5,.923),ncol=2,frameon=False)
    counts='/'.join(str(int(v)) for v in sorted(table.trials.unique()))
    fig.text(.5,.032,f'MC markers: arithmetic mean ({counts} trials); error bars: mean ±2 SE (uncertainty of mean).',
             ha='center',fontsize=10)
    fig.text(.5,.009,('Shading: empirical 10–90% trial range, not a confidence interval.  ' if show_spread else '')+
             '*Distributional DE: moment-matched Gaussian surrogate.',ha='center',fontsize=10)
    fig.subplots_adjust(top=.765,bottom=.13,hspace=.2,wspace=.12)
    return fig,axes

"""Aggregate cached forward-noise MC with errors over directions, not PCs."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LABELS={'resnet50':'Standard RN50','resnet50_robust':'Robust RN50',
        'clipag_vitb32':'CLIPAG','dinov2_vitb14_reg':'DINOv2'}
METRICS=['smooth','variance','neighborhood','step','odd','even','drift']


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',type=Path,required=True)
    ap.add_argument('--tables',type=Path,required=True)
    ap.add_argument('--figures',type=Path,required=True)
    args=ap.parse_args()
    args.tables.mkdir(parents=True,exist_ok=True);args.figures.mkdir(parents=True,exist_ok=True)
    records=[]
    for model in LABELS:
        folder=args.input/model
        meta=json.loads((folder/'config.json').read_text())
        d=pd.read_csv(folder/'seed_trace_summary.csv')
        if d.seed_index.nunique()!=len(meta['images']):
            raise ValueError(f'Incomplete run: {model}')
        for (metric,tau),group in d.groupby(['metric','tau']):
            n=len(group);mean=group.trace_mean.mean()
            mcse=np.sqrt(np.sum(group.trace_mc_se**2))/n
            exact=group.exact_trace.mean()
            records.append(dict(model=model,metric=metric,tau=tau,noise_255=tau*255,
                                trace=mean,mc_se=mcse,seed_sd=group.trace_mean.std(ddof=1),
                                seed_min=group.trace_mean.min(),seed_max=group.trace_mean.max(),
                                exact_trace=exact,relative_to_exact=mean/exact,
                                relative_mc_se=mcse/exact,outside_rgb_fraction=group.outside_rgb_fraction.mean(),
                                n_seeds=n,directions=meta['directions'],centers=meta['centers'],h=meta['h']))
        # Small plot-ready PC means + MC errors, independent of rendered figures.
        packs=[np.load(folder/f'seed_{i:02d}.npz') for i in range(1,len(meta['images'])+1)]
        compact=dict(spectrum=packs[0]['spectrum'],tau=packs[0]['tau'],
                     exact=np.mean([p['exact'] for p in packs],axis=0))
        for metric in METRICS:
            means=[];errors=[]
            for k in range(len(meta['tau_rgb'])):
                values=[p[f'{metric}_{k}'].astype('float64') for p in packs]
                means.append(np.mean([v.mean(0) for v in values],axis=0))
                errors.append(np.sqrt(np.sum([v.var(0,ddof=1)/len(v) for v in values],axis=0))/len(values))
            compact[f'{metric}_mean']=np.stack(means);compact[f'{metric}_mc_se']=np.stack(errors)
        np.savez_compressed(args.tables/f'{model}_pc_summary.npz',**compact)
        for p in packs:p.close()
        (args.tables/f'{model}_config.json').write_text(json.dumps(meta,indent=2))
    result=pd.DataFrame(records)
    result.to_csv(args.tables/'trace_summary.csv',index=False)
    plt.rcParams['pdf.fonttype']=42
    colors={'smooth':'#228833','variance':'#4477aa','neighborhood':'#777777','step':'#ee7733'}
    names={'smooth':'Smoothed Jacobian','variance':'Noise covariance','neighborhood':'Mean local energy','step':'Total response change'}
    fig,axs=plt.subplots(2,2,figsize=(12,8))
    for ax,(model,label) in zip(axs.flat,LABELS.items()):
        part=result[result.model==model]
        ax.axhline(1,color='black',linestyle='--',linewidth=1,label='Exact local Jacobian')
        local=part[part.metric=='local_forward'].iloc[0]
        ax.errorbar([0],[local.relative_to_exact],yerr=[2*local.relative_mc_se],fmt='ko',markersize=4,label='Forward local check')
        for metric in colors:
            q=part[part.metric==metric].sort_values('tau')
            ax.errorbar(q.noise_255,q.relative_to_exact,yerr=2*q.relative_mc_se,
                        color=colors[metric],fmt='o-',markersize=4,capsize=2,label=names[metric])
        ax.set_xscale('symlog',linthresh=.5)
        ax.set_yscale('symlog',linthresh=.01)
        ax.set_xticks([0,.5,2,8,16],['0','0.5','2','8','16'])
        ax.set(title=label,xlabel='Gaussian RGB noise SD × 255',ylabel='T / exact local T')
        ax.spines[['top','right']].set_visible(False)
    handles,labels=axs[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=3,fontsize=9)
    fig.suptitle('Forward noise geometry | 10 fixed seeds/model, PCA750, matched df₂=375\n'
                 'Error bars: ±2 MC SE; no pixel clipping; original training spectrum fixed')
    fig.tight_layout(rect=(0,.09,1,.93))
    for ext in ['png','pdf']:fig.savefig(args.figures/f'noise_geometry.{ext}',dpi=150)
    fig2,axs=plt.subplots(1,3,figsize=(15,4.8))
    modelcolors=['#c04a36','#2378ad','#228833','#9955bb']
    for ax,metric in zip(axs,['smooth','variance','step']):
        for (model,label),color in zip(LABELS.items(),modelcolors):
            q=result[(result.model==model)&(result.metric==metric)].sort_values('tau')
            ax.errorbar(np.r_[0,q.noise_255],np.r_[q.exact_trace.iloc[0],q.trace],
                        yerr=np.r_[0,2*q.mc_se],fmt='o-',color=color,label=label,markersize=4,capsize=2)
        ax.set_xscale('symlog',linthresh=.5);ax.set_yscale('symlog',linthresh=.1)
        ax.set_xticks([0,.5,2,8,16],['0','0.5','2','8','16'])
        ax.set(title=names[metric],xlabel='Gaussian RGB noise SD × 255',ylabel='T at matched df₂=375')
        ax.spines[['top','right']].set_visible(False)
    axs[0].legend(fontsize=8)
    fig2.suptitle('Same training spectra and fixed seed images; finite-scale targets differ')
    fig2.tight_layout()
    for ext in ['png','pdf']:fig2.savefig(args.figures/f'model_comparison.{ext}',dpi=150)
    print(result[result.metric.isin(['smooth','variance','step','local_forward'])][
        ['model','noise_255','metric','trace','mc_se','relative_to_exact']].to_string(index=False))


if __name__=='__main__':main()

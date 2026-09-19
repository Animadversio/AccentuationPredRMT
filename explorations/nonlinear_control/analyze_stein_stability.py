"""Diagnose Stein U-statistic MC instability and positive-support selection bias."""
from pathlib import Path
import os
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm,spearmanr
from tqdm import tqdm

REPO=Path(__file__).resolve().parents[2]
TABLE=REPO/'tables/nonlinear_control/biological_validation'
FIGURE=REPO/'figures/nonlinear_control/biological_validation'
TAUS=np.array([.5,2.,8.,16.])
R_CURRENT=64
MODELS=['AlexNet_training_seed_01','resnet50','resnet50_robust','resnet50_clip',
        'resnet50_dino','regnety_640','clipag_vitb32','dinov2_vitb14_reg',
        'siglip2_vitb16','radio_v2.5-b']


def trace_mean_se(pseudovalues,weight):
    traces=np.asarray(pseudovalues,dtype=np.float64)@weight
    means=traces.mean(1)
    seed_mc_var=traces.var(1,ddof=1)/traces.shape[1]
    return float(means.mean()),float(np.sqrt(seed_mc_var.sum())/len(means)),traces


def build_mc_table(data):
    store=os.environ.get('STORE_DIR')
    if not store: raise ValueError('STORE_DIR is not configured')
    root=Path(store)/'Projects/AccentuationPredRMT/nonlinear_control/mass_v1/geometry'
    rows=[];started=time.monotonic()
    groups=list(data.groupby('geometry_id',sort=False))
    for geometry_id,aliases in tqdm(groups,desc='Stein raw-archive diagnostics'):
        stein=[];smooth=[];spectrum=None
        for seed in range(1,11):
            with np.load(root/geometry_id/f'seed_{seed:02d}.npz') as values:
                stein.append(values['stein']);smooth.append(values['smooth'])
                if spectrum is None: spectrum=np.asarray(values['spectrum'],float)
        stein=np.asarray(stein);smooth=np.asarray(smooth) # seed,tau,R,PC
        for row in aliases.itertuples(index=False):
            weight=spectrum/(spectrum+row.ridge_kappa)**2
            for li,tau in enumerate(TAUS):
                sm,se_s,tr_s=trace_mean_se(stein[:,li],weight)
                ref,se_ref,tr_ref=trace_mean_se(smooth[:,li],weight)
                diff=tr_s-tr_ref
                diff_se=float(np.sqrt((diff.var(1,ddof=1)/diff.shape[1]).sum())/len(diff))
                out=dict(subject=row.subject,monkey=row.monkey,unit=row.unit,site_id=row.site_id,
                    model=row.model,geometry_id=geometry_id,robust_model=row.robust_model,tau_255=tau,
                    directions=R_CURRENT,seeds=10,stein_trace=sm,smooth_trace=ref,
                    stein_mc_se=se_s,smooth_mc_se=se_ref,difference_mc_se=diff_se,
                    stein_positive=sm>0,stein_z=sm/se_s if se_s>0 else np.nan,
                    stein_minus_smooth_z=(sm-ref)/diff_se if diff_se>0 else np.nan,
                    stein_over_smooth=sm/ref if ref!=0 else np.nan)
                for r_new in [64,128,256,512,1024,2048]:
                    scaled_se=se_s*np.sqrt(R_CURRENT/r_new)
                    out[f'pred_negative_prob_R{r_new}']=float(norm.cdf(-ref/scaled_se)) if scaled_se>0 else 0.
                out['directions_for_50pct_relative_se']=(R_CURRENT*(se_s/(.5*abs(ref)))**2
                    if se_s>0 and ref!=0 else np.nan)
                out['directions_for_25pct_relative_se']=(R_CURRENT*(se_s/(.25*abs(ref)))**2
                    if se_s>0 and ref!=0 else np.nan)
                rows.append(out)
    result=pd.DataFrame(rows)
    print(f'Loaded {len(groups)*10} seed archives in {time.monotonic()-started:.1f}s',flush=True)
    return result


def within_site_spearman(data,predictor,outcome):
    valid=(data[predictor]>0)&np.isfinite(data[predictor])&np.isfinite(data[outcome])
    d=data.loc[valid,['site_id',predictor,outcome]].copy()
    raw_n=len(d);raw_n_sites=d.site_id.nunique()
    counts=d.groupby('site_id').size();d=d[d.site_id.isin(counts[counts>=2].index)]
    x=np.log10(d[predictor]);x=x-x.groupby(d.site_id).transform('mean')
    y=d[outcome]-d.groupby('site_id')[outcome].transform('mean')
    return spearmanr(x,y).statistic,len(d),d.site_id.nunique(),raw_n,raw_n_sites


def common_support_table(data):
    methods={'stein':'geom_stein_tau255_{tau}',
             'smooth':'geom_smooth_tau255_{tau}',
             'neighborhood':'geom_neighborhood_tau255_{tau}',
             'variance':'geom_variance_tau255_{tau}'}
    subsets=[('all_models',data),('without_robust',data[~data.robust_model]),
             ('seven_models',data[(~data.robust_model)&(data.model!='AlexNet_training_seed_01')])]
    rows=[]
    for subset,d0 in subsets:
        for tau in TAUS:
            token=str(int(tau)) if tau.is_integer() else '0p5'
            stein=f'geom_stein_tau255_{token}__V_control_session_site_train_anchor_affine_mean'
            positive=d0[np.isfinite(d0[stein])&(d0[stein]>0)].copy()
            supports=[('stein_positive',positive,methods)]
            # Non-Stein estimators are defined for all rows. Their full-support
            # association reveals how much selecting on Stein positivity moves rho.
            supports.append(('all_available',d0,{k:v for k,v in methods.items() if k!='stein'}))
            for support_name,support,support_methods in supports:
                for method,template in support_methods.items():
                    predictor=f'{template.format(tau=token)}__V_control_session_site_train_anchor_affine_mean'
                    for outcome in ['control_anchor_affine_slope','control_anchor_affine_mse']:
                        rho,n,n_sites,raw_n,raw_n_sites=within_site_spearman(support,predictor,outcome)
                        rows.append(dict(subset=subset,tau_255=tau,support=support_name,
                            method=method,outcome=outcome,raw_n=raw_n,raw_n_sites=raw_n_sites,
                            n=n,n_sites=n_sites,spearman=rho))
    return pd.DataFrame(rows)


def summary_table(mc):
    rows=[]
    for tau,g in mc.groupby('tau_255'):
        row=dict(tau_255=tau,n=len(g),observed_negative_fraction=1-g.stein_positive.mean(),
            median_stein_over_smooth=g.stein_over_smooth.median(),
            median_abs_stein_minus_smooth_z=g.stein_minus_smooth_z.abs().median(),
            fraction_abs_stein_minus_smooth_z_below_1=(g.stein_minus_smooth_z.abs()<1).mean(),
            median_abs_stein_z=g.stein_z.abs().median(),
            spearman_stein_vs_smooth=g[['stein_trace','smooth_trace']].corr(method='spearman').iloc[0,1],
            median_directions_for_50pct_relative_se=g.directions_for_50pct_relative_se.median(),
            median_directions_for_25pct_relative_se=g.directions_for_25pct_relative_se.median())
        for r_new in [64,128,256,512,1024,2048]:
            row[f'mean_pred_negative_fraction_R{r_new}']=g[f'pred_negative_prob_R{r_new}'].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def model_support_table(mc):
    result=(mc.groupby(['model','tau_255'],as_index=False)
        .agg(n=('stein_positive','size'),n_positive=('stein_positive','sum')))
    result['positive_fraction']=result.n_positive/result.n
    return result


def plot(mc,common):
    fig,axs=plt.subplots(2,2,figsize=(13,9))
    positive=(mc.groupby(['model','tau_255']).stein_positive.mean().unstack())
    positive=positive.reindex(MODELS)
    image=axs[0,0].imshow(positive,cmap='viridis',vmin=0,vmax=1,aspect='auto')
    axs[0,0].set_xticks(range(4),[.5,2,8,16]);axs[0,0].set_yticks(range(len(MODELS)),MODELS,fontsize=8)
    axs[0,0].set(xlabel='Noise SD × 255',title='Fraction of positive Stein traces')
    fig.colorbar(image,ax=axs[0,0],fraction=.046)

    finite=[mc.loc[(mc.tau_255==tau)&np.isfinite(mc.stein_over_smooth),'stein_over_smooth'] for tau in TAUS]
    axs[0,1].boxplot(finite,positions=np.arange(4),showfliers=False)
    axs[0,1].axhline(0,color='.5',lw=.8);axs[0,1].axhline(1,color='.5',lw=.8,ls='--')
    axs[0,1].set_xticks(range(4),[.5,2,8,16]);axs[0,1].set_yscale('symlog',linthresh=1)
    axs[0,1].set(xlabel='Noise SD × 255',ylabel='Stein trace / smooth trace',
                 title='Signed estimator dispersion (all 250 rows)')

    r_grid=[64,128,256,512,1024,2048]
    for tau in TAUS:
        g=mc[mc.tau_255==tau]
        predicted=[g[f'pred_negative_prob_R{r}'].mean() for r in r_grid]
        axs[1,0].plot(r_grid,predicted,'o-',label=f'{tau:g}/255')
        axs[1,0].scatter([64],[1-g.stein_positive.mean()],marker='x',s=55,color=axs[1,0].lines[-1].get_color())
    axs[1,0].set_xscale('log',base=2);axs[1,0].set_xticks(r_grid,r_grid)
    axs[1,0].set(xlabel='Gaussian directions per seed image',ylabel='Negative estimate fraction',
                 title='Projected sign failures; × = observed at R=64')
    axs[1,0].legend(fontsize=8,title='Noise SD')

    c=common[(common.subset=='without_robust')&(common.outcome=='control_anchor_affine_mse')]
    colors={'stein':'#d55e00','smooth':'#cc79a7','neighborhood':'#009e73','variance':'#e69f00'}
    for method,color in colors.items():
        g=c[(c.method==method)&(c.support=='stein_positive')]
        axs[1,1].plot(g.tau_255,g.spearman,'o-',label=f'{method}: Stein-positive',color=color)
        if method!='stein':
            full=c[(c.method==method)&(c.support=='all_available')]
            axs[1,1].plot(full.tau_255,full.spearman,'--',lw=1.4,label=f'{method}: all rows',color=color,alpha=.75)
    axs[1,1].axhline(0,color='.5',lw=.8);axs[1,1].set_xscale('log',base=2)
    axs[1,1].set_xticks(TAUS,TAUS);axs[1,1].set(xlabel='Noise SD × 255',
        ylabel='Site-residual Spearman with calibrated MSE',
        title='High-noise correlation changes with selected support')
    axs[1,1].legend(fontsize=7,ncol=2)
    for ax in axs.flat: ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Stein forward-only estimator: Monte Carlo stability and selection bias')
    fig.tight_layout();fig.savefig(FIGURE/'stein_stability_diagnostics.png',dpi=200);plt.close(fig)


def main():
    data=pd.read_parquet(TABLE/'biological_validation_synopsis_v1.parquet')
    cache=TABLE/'stein_mc_stability.csv.gz'
    mc=pd.read_csv(cache) if cache.exists() else build_mc_table(data)
    common=common_support_table(data);summary=summary_table(mc);model_support=model_support_table(mc)
    mc.to_csv(TABLE/'stein_mc_stability.csv.gz',index=False,compression={'method':'gzip','compresslevel':6,'mtime':0})
    common.to_csv(TABLE/'stein_common_support_correlations.csv',index=False)
    summary.to_csv(TABLE/'stein_stability_summary.csv',index=False)
    model_support.to_csv(TABLE/'stein_stability_model_support.csv',index=False)
    plot(mc,common)
    print(summary.to_string(index=False))
    print('\nCommon support, without robust, MSE:\n',common[(common.subset=='without_robust')&
        (common.outcome=='control_anchor_affine_mse')].to_string(index=False))


if __name__=='__main__': main()

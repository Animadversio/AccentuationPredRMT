"""Analyze nested-R one-sided and antithetic Stein GPU pilot outputs."""
import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

REPO=Path(__file__).resolve().parents[2]
TABLE=REPO/'tables/nonlinear_control/biological_validation'
FIGURE=REPO/'figures/nonlinear_control/biological_validation'


def trace_pseudovalues(delta,gram,tau,weights):
    """Project PC Stein pseudovalues to every alias without storing R x PC x alias."""
    count=len(delta)
    cross=delta*(gram@delta)/tau**2
    cross_trace=cross@weights.T
    total=cross_trace.sum(0)
    estimate=total/(count*(count-1))
    loo=(total[None,:]-2*cross_trace)/((count-1)*(count-2))
    return count*estimate[None,:]-(count-1)*loo


def collect(root,synopsis):
    rows=[]
    geometries=sorted(path.parent for path in root.glob('geometry/*/DONE.json'))
    for geometry in tqdm(geometries,desc='Nested Stein geometries'):
        geometry_id=geometry.name
        aliases=synopsis[synopsis.geometry_id==geometry_id].drop_duplicates('site_id')
        if aliases.empty: continue
        seed_paths=sorted(geometry.glob('seed_*.npz'))
        per={}
        for seed_path in seed_paths:
            with np.load(seed_path) as z:
                tau=z['tau'];prefixes=z['prefixes'].astype(int);spectrum=z['spectrum']
                weights=np.stack([spectrum/(spectrum+k)**2 for k in aliases.ridge_kappa])
                for li,level in enumerate(tau):
                    for prefix in prefixes:
                        gram=np.asarray(z['direction_gram'][:prefix,:prefix],dtype=np.float64).copy()
                        np.fill_diagonal(gram,0.)
                        for method,key in [('one_sided','one_sided_delta'),('antithetic','antithetic_delta')]:
                            delta=np.asarray(z[key][li,:prefix],dtype=np.float64)
                            pseudo=trace_pseudovalues(delta,gram,float(level),weights)
                            per.setdefault((method,float(level*255),int(prefix)),[]).append(pseudo)
        for (method,level,prefix),seed_pseudo in per.items():
            means=np.stack([p.mean(0) for p in seed_pseudo])
            variances=np.stack([p.var(0,ddof=1)/len(p) for p in seed_pseudo])
            estimate=means.mean(0);se=np.sqrt(variances.sum(0))/len(seed_pseudo)
            token='0p5' if np.isclose(level,.5) else str(int(round(level)))
            # The nested estimator here is a geometry trace. V additionally
            # multiplies this trace by held-out MSE / n_train.
            smooth_col=f'geom_smooth_tau255_{token}__trace_mean'
            old_col=f'geom_stein_tau255_{token}__trace_mean'
            for ai,alias in enumerate(aliases.itertuples(index=False)):
                ref=getattr(alias,smooth_col);old=getattr(alias,old_col)
                rows.append(dict(geometry_id=geometry_id,site_id=alias.site_id,model=alias.model,
                    subject=alias.subject,unit=alias.unit,robust_model=alias.robust_model,
                    method=method,tau_255=level,directions=prefix,seeds=len(seed_pseudo),
                    estimate=estimate[ai],mc_se=se[ai],positive=estimate[ai]>0,
                    smooth_reference=ref,old_one_sided_R64=old,
                    relative_error_to_smooth=(estimate[ai]-ref)/ref,
                    mc_se_over_smooth=se[ai]/abs(ref),
                    V_control_anchor_affine=estimate[ai]*alias.control_session_site_train_anchor_affine_gen_test_mse/alias.n_train,
                    control_anchor_affine_slope=alias.control_anchor_affine_slope,
                    control_anchor_affine_mse=alias.control_anchor_affine_mse))
    return pd.DataFrame(rows)


def summarize(data):
    return (data.groupby(['method','tau_255','directions'],as_index=False)
        .agg(n=('estimate','size'),n_geometries=('geometry_id','nunique'),
             n_sites=('site_id','nunique'),positive_fraction=('positive','mean'),
             median_mc_se_over_smooth=('mc_se_over_smooth','median'),
             median_abs_relative_error=('relative_error_to_smooth',lambda x:np.median(np.abs(x))),
             median_signed_relative_error=('relative_error_to_smooth','median')))


def biological_correlations(data):
    subsets={'all_models':lambda d:d,
        'without_robust':lambda d:d[~d.robust_model],
        'seven_models':lambda d:d[(~d.robust_model)&(d.model!='AlexNet_training_seed_01')]}
    rows=[]
    for (method,tau,directions),base in data.groupby(['method','tau_255','directions']):
        for subset,select in subsets.items():
            selected=select(base)
            for outcome in ['control_anchor_affine_slope','control_anchor_affine_mse']:
                valid=selected.positive&np.isfinite(selected[outcome])
                d=selected.loc[valid,['site_id','V_control_anchor_affine',outcome]].copy()
                raw_n=len(d);counts=d.groupby('site_id').size();d=d[d.site_id.isin(counts[counts>=2].index)]
                x=np.log10(d.V_control_anchor_affine);x-=x.groupby(d.site_id).transform('mean')
                y=d[outcome]-d.groupby('site_id')[outcome].transform('mean')
                rho=pd.Series(x).corr(pd.Series(y),method='spearman')
                rows.append(dict(method=method,tau_255=tau,directions=directions,subset=subset,
                    outcome=outcome,raw_n=raw_n,n=len(d),n_sites=d.site_id.nunique(),spearman=rho))
    return pd.DataFrame(rows)


def plot(summary):
    fig,axs=plt.subplots(1,3,figsize=(14,4.2));colors={'one_sided':'#d55e00','antithetic':'#0072b2'}
    metrics=[('positive_fraction','Positive trace fraction'),
             ('median_mc_se_over_smooth','Median MC SE / smooth trace'),
             ('median_abs_relative_error','Median |Stein - smooth| / smooth')]
    for ax,(metric,title) in zip(axs,metrics):
        for (method,tau),g in summary.groupby(['method','tau_255']):
            ax.plot(g.directions,g[metric],marker='o',color=colors[method],
                ls={.5:'-',2:'--',8:'-.',16:':'}[tau],label=f'{method.replace("_","-")}, {tau:g}/255')
        ticks=sorted(summary.directions.unique())
        ax.set_xscale('log',base=2);ax.set_xticks(ticks,ticks)
        ax.set_xlabel('Nested Gaussian directions R');ax.set_title(title);ax.spines[['top','right']].set_visible(False)
        if metric!='positive_fraction': ax.set_yscale('log')
    axs[0].set_ylim(0,1.03);axs[-1].legend(fontsize=7,ncol=2,bbox_to_anchor=(1.02,1),loc='upper left')
    full=summary.n.max()>=250
    fig.suptitle(('Full 250-row nested-direction validation' if full else 'Nested-direction pilot')+
        ': one-sided versus antithetic Stein')
    fig.tight_layout();FIGURE.mkdir(parents=True,exist_ok=True)
    filename='stein_nested_full_validation.png' if full else 'stein_nested_pilot.png'
    fig.savefig(FIGURE/filename,dpi=200,bbox_inches='tight');plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path)
    args=parser.parse_args()
    store=os.environ.get('STORE_DIR')
    if args.input is None:
        if not store: raise ValueError('STORE_DIR is not configured')
        args.input=Path(store)/'Projects/AccentuationPredRMT/nonlinear_control/stein_nested_v1'
    synopsis=pd.read_parquet(TABLE/'biological_validation_synopsis_v1.parquet')
    data=collect(args.input,synopsis);summary=summarize(data);correlations=biological_correlations(data)
    data.to_csv(TABLE/'stein_nested_site_estimates.csv.gz',index=False,
        compression={'method':'gzip','compresslevel':6,'mtime':0})
    summary.to_csv(TABLE/'stein_nested_summary.csv',index=False)
    correlations.to_csv(TABLE/'stein_nested_biological_correlations.csv',index=False)
    plot(summary)
    print(summary.to_string(index=False))
    check=data[(data.method=='one_sided')&(data.directions==64)]
    difference=check.estimate-check.old_one_sided_R64
    print('\nR=64 reproduction relative L2:',np.linalg.norm(difference)/np.linalg.norm(check.old_one_sided_R64))
    print('\nAntithetic R=max biological correlations:\n',correlations[(correlations.method=='antithetic')&
        (correlations.directions==correlations.directions.max())].to_string(index=False))


if __name__=='__main__': main()

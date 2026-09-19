"""Summarize leakage-free encoding-to-control anchor affine calibration."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO=Path(__file__).resolve().parents[2]
TABLE=REPO/'tables/nonlinear_control/biological_validation'
FIGURE=REPO/'figures/nonlinear_control/biological_validation'
MONKEY_COLORS={'red':'#cc3311','paul':'#4477aa','venus':'#009988','leap':'#aa4499','three0':'#997700'}

ENDPOINTS={
    'identity':('Identity reference (previous)','control_mse','control_slope'),
    'site_all':('Site neural map\nall anchors','control_site_anchor_affine_mse','control_site_anchor_affine_slope'),
    'site_train':('Site neural map\ntrain anchors','control_site_train_anchor_affine_mse','control_site_train_anchor_affine_slope'),
    'outcome_refit':('Accentuated-cloud refit\n(leaky lower bound)','control_refit_mse',None),
}
GEOMETRY=[('Exact local','geom_exact'),('Smooth 8/255','geom_smooth_tau255_8'),
          ('Smooth 16/255','geom_smooth_tau255_16'),
          ('Neighborhood 8/255','geom_neighborhood_tau255_8'),
          ('Neighborhood 16/255','geom_neighborhood_tau255_16'),
          ('Variance 16/255','geom_variance_tau255_16')]


def site_residual_spearman(data,predictor,outcome):
    valid=(data[predictor]>0)&np.isfinite(data[predictor])&np.isfinite(data[outcome])
    d=data.loc[valid,['site_id',predictor,outcome]].copy()
    d['x']=np.log10(d[predictor])-np.log10(d[predictor]).groupby(d.site_id).transform('mean')
    d['y']=d[outcome]-d.groupby('site_id')[outcome].transform('mean')
    return spearmanr(d.x,d.y).statistic,len(d)


def summary_tables(data):
    rows=[]
    for endpoint,(label,column,slope_column) in ENDPOINTS.items():
        for group,d in [('all',data)]+[(mk,g) for mk,g in data.groupby('monkey')]:
            ratio=d[column]/d.control_mse
            rows.append(dict(endpoint=endpoint,label=label.replace('\n',' '),group=group,n=len(d),
                mse_mean=d[column].mean(),mse_median=d[column].median(),
                control_slope_mean=d[slope_column].mean() if slope_column else 1.,
                control_slope_median=d[slope_column].median() if slope_column else 1.,
                ratio_to_identity_mean=ratio.mean(),ratio_to_identity_median=ratio.median(),
                fraction_better_than_identity=(ratio<1).mean()))
    pd.DataFrame(rows).to_csv(TABLE/'crosssession_affine_mse_summary.csv',index=False)
    fit=[]
    for scope in ['site']:
        for subset in ['anchor','train_anchor','heldout_anchor']:
            base=f'crosssession_{scope}_{subset}_fit'
            for group,d in [('all',data)]+[(mk,g) for mk,g in data.groupby('monkey')]:
                # The one site mapping is repeated over ten model rows.
                d=d.drop_duplicates('site_id')
                fit.append(dict(scope=scope,anchor_subset=subset,group=group,n=len(d),
                    n_anchors_min=int(d[f'{base}_n'].min()),n_anchors_max=int(d[f'{base}_n'].max()),
                    gain_median=d[f'{base}_slope'].median(),gain_min=d[f'{base}_slope'].min(),
                    gain_max=d[f'{base}_slope'].max(),nonpositive_gain=int((d[f'{base}_slope']<=0).sum()),
                    anchor_r_median=d[f'{base}_r'].median()))
    pd.DataFrame(fit).to_csv(TABLE/'crosssession_affine_fit_summary.csv',index=False)


def geometry_table(data):
    d=data[~data.robust_model]
    endpoint_specs=[
        ('identity','control_mse','V_control_session'),
        ('site_train','control_site_train_anchor_affine_mse','V_control_session_site_train_anchor_affine')]
    rows=[]
    for endpoint,outcome,vname in endpoint_specs:
        for label,base in GEOMETRY:
            rho,n=site_residual_spearman(d,f'{base}__{vname}_mean',outcome)
            rows.append(dict(endpoint=endpoint,outcome=outcome,geometry=label,predictor=f'{base}__{vname}_mean',
                             n=n,site_residual_spearman=rho))
    out=pd.DataFrame(rows)
    out.to_csv(TABLE/'crosssession_affine_geometry_correlations.csv',index=False)
    return out


def plot(data,geometry):
    fig,axs=plt.subplots(1,3,figsize=(16,4.8))
    keys=['site_all','site_train','outcome_refit']
    positions=np.arange(len(keys))
    for mk,g in data.groupby('monkey'):
        med=[np.median(g[ENDPOINTS[k][1]]/g.control_mse) for k in keys]
        axs[0].plot(positions,med,'o-',color=MONKEY_COLORS[mk],label=mk,alpha=.85)
    axs[0].axhline(1,color='.4',lw=1);axs[0].set_yscale('log')
    axs[0].set_xticks(positions,[ENDPOINTS[k][0] for k in keys],rotation=35,ha='right',fontsize=8)
    axs[0].set_ylabel('Median MSE / identity MSE');axs[0].set_title('Frozen affine effect by monkey')
    axs[0].legend(fontsize=7,ncol=2)

    sites=data.drop_duplicates('site_id')
    for mk,g in sites.groupby('monkey'):
        axs[1].scatter(g.crosssession_site_train_anchor_fit_r,
            g.crosssession_site_train_anchor_fit_slope,s=42,color=MONKEY_COLORS[mk],label=mk)
    axs[1].axhline(1,color='.6',lw=.8);axs[1].set(xlabel='Train-anchor cross-session r',
        ylabel='Frozen gain b',title='Site-level neural alignment')

    width=.24;x=np.arange(len(GEOMETRY))
    for j,(endpoint,label,color) in enumerate([('identity','Identity','#777777'),
                                               ('site_train','Site neural map','#009e73')]):
        q=geometry[geometry.endpoint==endpoint].set_index('geometry')
        axs[2].bar(x+(j-.5)*width,[q.loc[g[0],'site_residual_spearman'] for g in GEOMETRY],
                   width,label=label,color=color)
    axs[2].axhline(0,color='.5',lw=.8);axs[2].set_xticks(x,[g[0] for g in GEOMETRY],rotation=35,ha='right',fontsize=8)
    axs[2].set_ylabel('Site-residual Spearman with matched V')
    axs[2].set_title('Geometry association after calibration')
    axs[2].legend(fontsize=7)
    for ax in axs: ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Encoding-to-control affine calibration from natural anchors only')
    fig.tight_layout();fig.savefig(FIGURE/'crosssession_anchor_affine_validation.png',dpi=200);plt.close(fig)


def main():
    data=pd.read_parquet(TABLE/'biological_validation_synopsis_v1.parquet')
    summary_tables(data);geometry=geometry_table(data);plot(data,geometry)
    print(pd.read_csv(TABLE/'crosssession_affine_mse_summary.csv').query("group == 'all'").to_string(index=False))
    print('\n',geometry.to_string(index=False))


if __name__=='__main__': main()

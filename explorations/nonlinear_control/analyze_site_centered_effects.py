"""Site-within associations and direct biological control-MSE figures."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr,spearmanr
import statsmodels.api as sm

REPO=Path(__file__).resolve().parents[2]
TABLE=REPO/'tables/nonlinear_control/biological_validation'
FIGURE=REPO/'figures/nonlinear_control/biological_validation'
TAUS=[.5,2.,8.,16.]
ENDPOINTS=['control_session','encoding_session_matched','encoding_session']
ENDPOINT_LABELS={'control_session':'Control session, matched images',
                 'encoding_session_matched':'Encoding session, matched images',
                 'encoding_session':'Encoding session, all 195 images'}
MONKEY_COLORS={'red':'#cc3311','paul':'#4477aa','venus':'#009988','leap':'#aa4499','three0':'#997700'}


def center_within_site(data,column):
    return data[column]-data.groupby('site_id')[column].transform('mean')


def association(data,predictor,outcome):
    valid=(data[predictor]>0)&np.isfinite(data[predictor])&np.isfinite(data[outcome])
    d=data.loc[valid,['site_id',predictor,outcome]].copy()
    d['logV']=np.log10(d[predictor])
    d['x']=center_within_site(d,'logV');d['y']=center_within_site(d,outcome)
    x=d.x.to_numpy();y=d.y.to_numpy()
    fit=sm.OLS(y,sm.add_constant(x)).fit(cov_type='cluster',cov_kwds={'groups':d.site_id})
    xs=x/x.std(ddof=0);ys=y/y.std(ddof=0)
    standardized=sm.OLS(ys,sm.add_constant(xs)).fit(cov_type='cluster',cov_kwds={'groups':d.site_id})
    return dict(n=len(d),n_sites=d.site_id.nunique(),pearson=pearsonr(x,y).statistic,
                spearman=spearmanr(x,y).statistic,beta_per_log10_V=fit.params[1],
                cluster_se=fit.bse[1],cluster_p=fit.pvalues[1],standardized_beta=standardized.params[1])


def build_table(data):
    configs=[('exact',0.,'geom_exact')]
    configs += [(method,tau,f'geom_{method}_tau255_{str(int(tau)) if tau.is_integer() else str(tau).replace(".","p")}')
                for method in ['smooth','neighborhood'] for tau in TAUS]
    rows=[]
    for subset,d in [('all_models',data),('without_robust',data[~data.robust_model])]:
        for method,tau,base in configs:
            for endpoint in ENDPOINTS:
                predictor=f'{base}__V_{endpoint}_mean'
                for outcome in ['control_slope','control_slope_within_seed','control_mse']:
                    rows.append(dict(subset=subset,method=method,tau_255=tau,endpoint=endpoint,
                                     predictor=predictor,outcome=outcome,**association(d,predictor,outcome)))
    return pd.DataFrame(rows)


def smooth_level_plot(results):
    fig,axs=plt.subplots(1,2,figsize=(12,4.4),sharex=True)
    colors={'control_session':'#cc6677','encoding_session_matched':'#ddaa33','encoding_session':'#4477aa'}
    for ax,outcome in zip(axs,['control_slope','control_mse']):
        for endpoint in ENDPOINTS:
            for subset,style,alpha in [('all_models','--',.6),('without_robust','-',1.)]:
                d=results[(results.method=='smooth')&(results.endpoint==endpoint)&
                          (results.outcome==outcome)&(results.subset==subset)]
                label=ENDPOINT_LABELS[endpoint]+(' (all models)' if subset=='all_models' else ' (no robust)')
                ax.plot(d.tau_255,d.spearman,'o'+style,color=colors[endpoint],alpha=alpha,label=label)
        ax.axhline(0,color='.6',linewidth=.8);ax.set_xscale('log',base=2)
        ax.set_xticks(TAUS,TAUS);ax.set_xlabel('Noise SD × 255')
        ax.set_title('Biological control slope' if outcome=='control_slope' else 'Direct control MSE')
        ax.spines[['top','right']].set_visible(False)
    axs[0].set_ylabel('Site-centered Spearman correlation')
    axs[1].legend(fontsize=7,loc='best')
    fig.suptitle('Within-site smooth-V associations\nBoth log10 V and outcome have their site mean removed')
    fig.tight_layout();fig.savefig(FIGURE/'site_centered_smooth_correlations.png',dpi=180);plt.close(fig)


def control_mse_plot(data,results):
    data=data[~data.robust_model].copy()
    predictor='geom_smooth_tau255_2__V_control_session_mean'
    data['logV']=np.log10(data[predictor])
    data['logV_centered']=center_within_site(data,'logV')
    data['mse_centered']=center_within_site(data,'control_mse')
    fig,axs=plt.subplots(1,2,figsize=(11,4.7))
    for monkey,g in data.groupby('monkey'):
        axs[0].scatter(g.logV,g.control_mse,s=24,alpha=.65,color=MONKEY_COLORS[monkey],label=monkey)
        axs[1].scatter(g.logV_centered,g.mse_centered,s=24,alpha=.65,color=MONKEY_COLORS[monkey])
    for ax,x,y in [(axs[0],data.logV,data.control_mse),(axs[1],data.logV_centered,data.mse_centered)]:
        coef=np.polyfit(x,y,1);grid=np.linspace(x.min(),x.max(),100);ax.plot(grid,np.polyval(coef,grid),color='black')
        ax.spines[['top','right']].set_visible(False)
    raw_s=spearmanr(data.logV,data.control_mse).statistic
    raw_p=pearsonr(data.logV,data.control_mse).statistic
    within=results[(results.subset=='without_robust')&(results.method=='smooth')&(results.tau_255==2)&
                   (results.endpoint=='control_session')&(results.outcome=='control_mse')].iloc[0]
    axs[0].set(xlabel=r'$\log_{10} V_{control\ session}$',ylabel='Direct biological control MSE',
               title=f'Pooled: Pearson={raw_p:.3f}, Spearman={raw_s:.3f}')
    axs[1].set(xlabel=r'$\log_{10}V-\overline{\log_{10}V}_{site}$',ylabel=r'$MSE-\overline{MSE}_{site}$',
               title=f'Within site: Pearson={within.pearson:.3f}, Spearman={within.spearman:.3f}')
    axs[0].legend(fontsize=8)
    fig.suptitle(r'$MSE_{control}=N_{acc}^{-1}\sum_i(y_i^{control}-\hat y_i^{posthoc,floor})^2$'
                 '\nSmooth V at noise SD × 255 = 2; CLIPAG and robust RN50 excluded')
    fig.tight_layout();fig.savefig(FIGURE/'control_mse_V_control_session_correlation.png',dpi=180);plt.close(fig)


def main():
    data=pd.read_parquet(TABLE/'biological_validation_synopsis_v1.parquet')
    results=build_table(data)
    results.to_csv(TABLE/'site_centered_associations.csv',index=False)
    smooth_level_plot(results);control_mse_plot(data,results)
    print(results[(results.subset=='without_robust')&(results.method=='smooth')&
                  results.outcome.isin(['control_slope','control_mse'])].to_string(index=False))


if __name__=='__main__':
    main()

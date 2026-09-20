"""Site-within associations and direct biological control-MSE figures."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr,spearmanr
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

REPO=Path(__file__).resolve().parents[2]
TABLE=REPO/'tables/nonlinear_control/biological_validation'
FIGURE=REPO/'figures/nonlinear_control/biological_validation'
TAUS=[.5,2.,8.,16.]
ENDPOINTS=['control_session','encoding_session_matched','encoding_session']
ENDPOINT_LABELS={'control_session':'Control session, matched images',
                 'encoding_session_matched':'Encoding session, matched images',
                 'encoding_session':'Encoding session, all 195 images'}
BENCHMARK_SPECS=[
    dict(endpoint='control_session_anchor_affine',geometry_endpoint='control_session_site_train_anchor_affine',
         gen_predictor='control_session_site_train_anchor_affine_gen_test_mse',
         slope_outcome='control_anchor_affine_slope',mse_outcome='control_anchor_affine_mse',
         endpoint_label='Control session; one train-anchor site affine shared across all models'),
    dict(endpoint='control_session_identity_reference',geometry_endpoint='control_session',
         gen_predictor='control_session_gen_test_mse',slope_outcome='control_slope',
         mse_outcome='control_mse',endpoint_label='Control session; identity-scale reference'),
    dict(endpoint='encoding_session_matched',geometry_endpoint='encoding_session_matched',
         gen_predictor='encoding_session_matched_gen_test_mse',slope_outcome='control_slope',
         mse_outcome='control_mse',endpoint_label='Encoding session, matched images; identity reference'),
    dict(endpoint='encoding_session',geometry_endpoint='encoding_session',
         gen_predictor='encoding_session_gen_test_mse',slope_outcome='control_slope',
         mse_outcome='control_mse',endpoint_label='Encoding session, all 195 images; identity reference')]
MONKEY_COLORS={'red':'#cc3311','paul':'#4477aa','venus':'#009988','leap':'#aa4499','three0':'#997700'}
METHOD_COLORS={'baseline':'#666666','local':'#0072b2','smooth':'#cc79a7',
               'neighborhood':'#009e73','variance':'#e69f00','step':'#56b4e9',
               'stein':'#d55e00','stein_antithetic':'#332288'}


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


def tau_tag(tau):
    return str(int(tau)) if float(tau).is_integer() else str(tau).replace('.','p')


def geometry_configs():
    configs=[dict(predictor_id='exact_trace',label=r'Exact trace $T(\kappa)$',group='baseline',base='geom_exact',quantity='trace'),
             dict(predictor_id='exact_V',label='Exact local V',group='local',base='geom_exact',quantity='V'),
             dict(predictor_id='local_mc_V',label='Local MC V',group='local',base='geom_local_mc',quantity='V')]
    labels={'smooth':'Smooth','neighborhood':'Neighborhood','variance':'Variance',
            'step':'Finite step','stein':'Stein'}
    for method in labels:
        for tau in TAUS:
            configs.append(dict(predictor_id=f'{method}_{tau_tag(tau)}',
                                label=f'{labels[method]}  {tau:g}/255',group=method,
                                base=f'geom_{method}_tau255_{tau_tag(tau)}',quantity='V',tau_255=tau))
    for tau in TAUS:
        configs.append(dict(predictor_id=f'stein_antithetic_{tau_tag(tau)}',
            label=f'Antithetic Stein  {tau:g}/255',group='stein_antithetic',
            base=f'geom_stein_antithetic_R512_tau255_{tau_tag(tau)}',quantity='V',tau_255=tau))
    return configs


def build_benchmark_table(data):
    """One plot-ready row per endpoint × predictor × outcome × model subset."""
    rows=[]
    for spec in BENCHMARK_SPECS:
        endpoint=spec['endpoint'];geometry_endpoint=spec['geometry_endpoint']
        configs=[dict(predictor_id='generalization_mse',label='Held-out generalization MSE',
                      group='baseline',predictor=spec['gen_predictor'])]
        for config in geometry_configs():
            config=config.copy()
            suffix='trace_mean' if config.pop('quantity')=='trace' else f'V_{geometry_endpoint}_mean'
            config['predictor']=f"{config.pop('base')}__{suffix}"
            configs.append(config)
        benchmark_subsets=[
            ('all_models',data),
            ('without_robust',data[~data.robust_model]),
            ('without_robust_and_untrained',
             data[(~data.robust_model)&(data.model!='AlexNet_training_seed_01')])]
        for subset,d in benchmark_subsets:
            for order,config in enumerate(configs):
                for outcome_role,outcome,direction in [
                        ('control_slope',spec['slope_outcome'],-1),
                        ('control_mse',spec['mse_outcome'],1)]:
                    stats=association(d,config['predictor'],outcome)
                    rows.append(dict(endpoint=endpoint,endpoint_label=spec['endpoint_label'],
                                     geometry_endpoint=geometry_endpoint,subset=subset,
                                     outcome_role=outcome_role,outcome=outcome,
                                     display_order=order,direction_sign=direction,**config,**stats))
    result=pd.DataFrame(rows)
    result['direction_aligned_pearson']=result.direction_sign*result.pearson
    result['direction_aligned_spearman']=result.direction_sign*result.spearman
    result['direction_aligned_standardized_beta']=result.direction_sign*result.standardized_beta
    result['cluster_q_bh']=np.nan
    for _,idx in result.groupby(['endpoint','subset','outcome_role']).groups.items():
        idx=list(idx)
        result.loc[idx,'cluster_q_bh']=multipletests(result.loc[idx,'cluster_p'],method='fdr_bh')[1]
    return result


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


def level_plot(results,method,filename):
    fig,axs=plt.subplots(1,2,figsize=(12,4.4),sharex=True)
    colors={'control_session':'#cc6677','encoding_session_matched':'#ddaa33','encoding_session':'#4477aa'}
    for ax,outcome in zip(axs,['control_slope','control_mse']):
        for endpoint in ENDPOINTS:
            for subset,style,alpha in [('all_models','--',.6),('without_robust','-',1.)]:
                d=results[(results.method==method)&(results.endpoint==endpoint)&
                          (results.outcome==outcome)&(results.subset==subset)]
                label=ENDPOINT_LABELS[endpoint]+(' (all models)' if subset=='all_models' else ' (no robust)')
                ax.plot(d.tau_255,d.spearman,'o'+style,color=colors[endpoint],alpha=alpha,label=label)
        ax.axhline(0,color='.6',linewidth=.8);ax.set_xscale('log',base=2)
        ax.set_xticks(TAUS,TAUS);ax.set_xlabel('Noise SD × 255')
        ax.set_title('Biological control slope' if outcome=='control_slope' else 'Direct control MSE')
        ax.spines[['top','right']].set_visible(False)
    axs[0].set_ylabel('Site-centered Spearman correlation')
    axs[1].legend(fontsize=7,loc='best')
    fig.suptitle(f'Within-site {method}-V associations\nBoth log10 V and outcome have their site mean removed')
    fig.tight_layout();fig.savefig(FIGURE/filename,dpi=180);plt.close(fig)


def predictor_benchmark_plot(results,endpoint,filename,significance='raw'):
    assert significance in {'raw','fdr'}
    d=results[results.endpoint==endpoint].copy()
    significance_column='cluster_p' if significance=='raw' else 'cluster_q_bh'
    significance_label=('Raw site-clustered p < 0.05' if significance=='raw'
                        else f'BH-FDR q < 0.05 across {d.predictor_id.nunique()} predictors')
    meta=(d[['predictor_id','label','group','display_order']].drop_duplicates()
          .sort_values('display_order'))
    y=np.arange(len(meta))[::-1]
    ymap=dict(zip(meta.predictor_id,y))
    fig,axs=plt.subplots(1,2,figsize=(12.8,11.2),sharey=True)
    outcomes=[('control_slope','Control slope',r'$-\rho$'),
              ('control_mse','Direct control MSE',r'$+\rho$')]
    for ax,(outcome,title,sign) in zip(axs,outcomes):
        panel=d[d.outcome_role==outcome]
        for _,row in meta.iterrows():
            pair=panel[panel.predictor_id==row.predictor_id].set_index('subset')
            required={'all_models','without_robust','without_robust_and_untrained'}
            if not required.issubset(pair.index):
                continue
            xa=pair.loc['all_models','direction_aligned_spearman']
            xr=pair.loc['without_robust','direction_aligned_spearman']
            xt=pair.loc['without_robust_and_untrained','direction_aligned_spearman']
            yi=ymap[row.predictor_id];color=METHOD_COLORS[row.group]
            ax.plot([xa,xr,xt],[yi,yi,yi],color='.76',lw=2,zorder=1)
            ax.scatter(xa,yi,s=62,color=color,edgecolor=color,zorder=3)
            ax.scatter(xr,yi,s=62,facecolor='white',edgecolor=color,lw=2,zorder=4)
            ax.scatter(xt,yi,s=62,marker='D',facecolor='white',edgecolor=color,lw=2,zorder=5)
            # Mark significance separately for each model subset. A star sits
            # directly above the marker whose clustered linear effect survives
            # BH correction within this endpoint/outcome/subset panel.
            for xvalue,subset in [(xa,'all_models'),(xr,'without_robust'),
                                  (xt,'without_robust_and_untrained')]:
                if pair.loc[subset,significance_column] < .05:
                    ax.text(xvalue,yi+.20,'★',color=color,ha='center',va='bottom',
                            fontsize=10,zorder=6)
            conventional=pair.loc['without_robust_and_untrained']
            if conventional.n < 175:
                ax.text(max(xa,xr,xt)+.018,yi,f"n={int(conventional.n)}",color='.35',
                        va='center',fontsize=7)
        ax.axvline(0,color='.55',lw=.9)
        ax.grid(axis='x',color='.91',lw=.8)
        ax.set_xlabel(f'Direction-aligned site-residual Spearman {sign}')
        ax.set_title(title)
        ax.spines[['top','right','left']].set_visible(False)
        ax.tick_params(axis='y',length=0)
    axs[0].set_yticks(y,[row.label for _,row in meta.iterrows()],fontsize=9)
    handles=[plt.Line2D([],[],marker='o',linestyle='',color='.25',label='All 10 models'),
             plt.Line2D([],[],marker='o',linestyle='',markerfacecolor='white',markeredgewidth=2,
                        color='.25',label='Without CLIPAG + robust RN50'),
             plt.Line2D([],[],marker='D',linestyle='',markerfacecolor='white',markeredgewidth=2,
                        color='.25',label='Also without untrained AlexNet'),
             plt.Line2D([],[],marker='*',linestyle='',color='.25',markersize=9,
                        label=significance_label+' above that marker')]
    group_handles=[plt.Line2D([],[],color=METHOD_COLORS[g],lw=5,label=l) for g,l in
                   [('baseline','Baseline'),('local','Local'),('smooth','Smooth'),
                    ('neighborhood','Neighborhood'),('variance','Variance'),
                    ('step','Finite step'),('stein','Stein'),
                    ('stein_antithetic','Antithetic Stein')]]
    fig.legend(handles=handles+group_handles,ncol=6,loc='lower center',frameon=False,fontsize=8)
    fig.suptitle('Benchmark of control-geometry predictors of biological outcomes\n'
                 f'{d.endpoint_label.iloc[0]}; site means removed; stars: {significance_label}',y=.985)
    fig.subplots_adjust(left=.23,right=.98,top=.91,bottom=.12,wspace=.08)
    fig.savefig(FIGURE/filename,dpi=200)
    plt.close(fig)


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
    level_plot(results,'smooth','site_centered_smooth_correlations.png')
    level_plot(results,'neighborhood','site_centered_neighborhood_correlations.png')
    control_mse_plot(data,results)
    benchmark=build_benchmark_table(data)
    benchmark.to_csv(TABLE/'predictor_benchmark_site_centered.csv',index=False)
    predictor_benchmark_plot(benchmark,'control_session_anchor_affine',
                             'site_centered_predictor_benchmark_control_session.png',significance='raw')
    predictor_benchmark_plot(benchmark,'control_session_anchor_affine',
                             'site_centered_predictor_benchmark_control_session_fdr.png',significance='fdr')
    predictor_benchmark_plot(benchmark,'control_session_identity_reference',
                             'site_centered_predictor_benchmark_control_session_identity_reference.png',
                             significance='raw')
    predictor_benchmark_plot(benchmark,'encoding_session',
                             'site_centered_predictor_benchmark_encoding_session.png',significance='raw')
    print(results[(results.subset=='without_robust')&(results.method=='smooth')&
                  results.outcome.isin(['control_slope','control_mse'])].to_string(index=False))


if __name__=='__main__':
    main()

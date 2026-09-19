"""Build and validate the maintained 250-row biological-validation synopsis.

One row is one subject x unit x encoding-model combination. The long-format
variance_predictors tables remain canonical for adding new geometry estimators;
this wide table is the fast analysis interface.
"""
from pathlib import Path
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
TABLE = REPO / 'tables/nonlinear_control/biological_validation'
FIGURE = REPO / 'figures/nonlinear_control/biological_validation'
KEY = ['subject', 'monkey', 'unit', 'model']
ROBUST_MODELS = {'clipag_vitb32', 'resnet50_robust'}
VERSION = '1.0.0'

sys.path.insert(0, str(HERE))
from validate_biology import upstream_modules  # noqa: E402


def regression_metrics(predicted, measured):
    """Direct-prediction and OLS-refit metrics, with population-variance convention."""
    x = np.asarray(predicted, float)
    y = np.asarray(measured, float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    out = {'n': len(x)}
    if len(x) < 2:
        return out
    error = y - x
    sy = float(np.var(y, ddof=0))
    sx = float(np.var(x, ddof=0))
    out.update(
        mse=float(np.mean(error**2)), rmse=float(np.sqrt(np.mean(error**2))),
        mae=float(np.mean(np.abs(error))), bias_measured_minus_predicted=float(error.mean()),
        measured_mean=float(y.mean()), predicted_mean=float(x.mean()),
        measured_variance=sy, predicted_variance=sx,
        r=float(np.corrcoef(x, y)[0, 1]) if sx > 0 and sy > 0 else np.nan)
    if sx > 0:
        slope = float(np.mean((x-x.mean())*(y-y.mean()))/sx)
        intercept = float(y.mean()-slope*x.mean())
        residual = y-(intercept+slope*x)
        out.update(slope=slope, intercept=intercept,
                   refit_mse=float(np.mean(residual**2)))
    else:
        out.update(slope=np.nan, intercept=np.nan, refit_mse=np.nan)
    out['r2_identity'] = float(1-out['mse']/sy) if sy > 0 else np.nan
    out['r2_refit'] = out['r']**2 if np.isfinite(out['r']) else np.nan
    return out


def prefix(values, name):
    return {f'{name}_{key}': value for key, value in values.items()}


def trial_mean_noise(trial_stim, trial_values, requested_names):
    """Mean estimated variance of each stimulus mean from its repeated trials."""
    names=np.asarray(trial_stim).astype(str)
    values=np.asarray(trial_values,float)
    estimates=[]
    for name in requested_names:
        y=values[names==str(name)]
        y=y[np.isfinite(y)]
        if len(y)>=2:
            estimates.append(float(np.var(y,ddof=1)/len(y)))
    return (float(np.mean(estimates)) if estimates else np.nan,
            len(estimates), len(requested_names))


def biological_metrics(manifest):
    _, loader, _ = upstream_modules()
    control = pd.read_csv(TABLE/'control_clouds.csv.gz')
    control_groups = {k:g for k,g in control.groupby(['subject','unit','model'], sort=False)}
    rows = []
    for row in manifest.itertuples(index=False):
        monkey = row.monkey
        brain = loader.load_brain(monkey)
        ui = list(brain['units']).index(row.unit)
        ceilings = brain['ceilings']
        ci = np.where((ceilings['unit']==row.unit) & (ceilings['model']==row.model))[0]
        assert len(ci)==1
        ci = ci[0]
        x_test, y_test = loader.encoding_cloud(monkey, row.unit, row.model, split='test')
        x_anchor, y_anchor = loader.anchor_cloud(monkey, row.unit, row.model)
        cg = control_groups[(row.subject, row.unit, row.model)]
        gen = regression_metrics(x_test, y_test)
        cross = regression_metrics(x_anchor, y_anchor)
        ctl = regression_metrics(cg.predicted, cg.measured)
        centered = cg[['predicted','measured']]-cg.groupby('seed')[['predicted','measured']].transform('mean')
        within_denom = float(np.square(centered.predicted).sum())
        control_slope_within_seed = (float((centered.predicted*centered.measured).sum()/within_denom)
                                     if within_denom > 0 else np.nan)
        s_nat = gen['measured_variance']
        s_control = ctl['measured_variance']
        test_names=set(loader._test_names(monkey))
        calibration=brain['calibration']
        heldout_names=[n for n in calibration['stim'] if n in test_names]
        gen_noise,n_gen_noise,n_gen_total=trial_mean_noise(
            calibration['trial_stim'],calibration['trial_z'][:,ui],heldout_names)
        control_data=brain['control']
        control_noise,n_control_noise,n_control_total=trial_mean_noise(
            control_data['trial_stim'],control_data['trial_z'][:,ui],cg.stimulus.tolist())
        # Across M fixed stimuli, centering removes mean(v_i)/M from the expected
        # population variance. With incomplete repeat coverage, mean(v_i) is a plug-in.
        s_nat_corrected=s_nat-(1-1/len(y_test))*gen_noise if np.isfinite(gen_noise) else np.nan
        control_mse_corrected=ctl['mse']-control_noise if np.isfinite(control_noise) else np.nan
        signal_var = float(ceilings['signal_var'][ci])
        noise_var = float(ceilings['noise_var'][ci])
        values = dict(subject=row.subject, monkey=monkey, unit=row.unit, model=row.model,
            region=str(brain['region'][ui]), robust_model=row.model in ROBUST_MODELS,
            site_id=f'{row.subject}:unit{row.unit}', geometry_id=row.geometry_id,
            biology_reliability=float(brain['reliability'][ui]),
            firing_floor=float(brain['firing_floor'][ui]),
            **prefix(gen, 'gen_test'), **prefix(cross, 'crossphase_anchor'), **prefix(ctl, 'control'))
        values.update(
            gen_test_error_over_S_nat_observed=gen['mse']/s_nat if s_nat > 0 else np.nan,
            gen_test_trialmean_noise_variance=gen_noise,
            gen_test_trialmean_noise_n_stim=n_gen_noise,
            gen_test_trialmean_noise_fraction=n_gen_noise/n_gen_total,
            gen_test_S_nat_noise_corrected=s_nat_corrected,
            control_error_over_S_nat_observed=ctl['mse']/s_nat if s_nat > 0 else np.nan,
            control_error_over_S_control_observed=ctl['mse']/s_control if s_control > 0 else np.nan,
            control_trialmean_noise_variance=control_noise,
            control_trialmean_noise_n_stim=n_control_noise,
            control_trialmean_noise_fraction=n_control_noise/n_control_total,
            control_trialmean_noise_high_coverage=n_control_noise/n_control_total >= .8,
            control_mse_noise_corrected=control_mse_corrected,
            control_error_over_S_nat_noise_corrected=(ctl['mse']/s_nat_corrected
                if np.isfinite(s_nat_corrected) and s_nat_corrected > 0 else np.nan),
            control_error_noise_corrected_over_S_nat_noise_corrected=(control_mse_corrected/s_nat_corrected
                if np.isfinite(control_mse_corrected) and np.isfinite(s_nat_corrected) and s_nat_corrected > 0 else np.nan),
            control_slope_within_seed=control_slope_within_seed,
            control_noise_ceiling_r=float(ceilings['nc_r'][ci]),
            control_noise_ceiling_signal_variance=signal_var,
            control_noise_ceiling_noise_variance=noise_var,
            control_noise_ceiling_n_stim=int(ceilings['n_stim'][ci]),
            control_error_over_ceiling_signal_variance=(ctl['mse']/signal_var
                if np.isfinite(signal_var) and signal_var > 0 else np.nan))
        rows.append(values)
    return pd.DataFrame(rows)


def geometry_wide():
    means = pd.read_csv(TABLE/'variance_predictors.csv')
    seeds = pd.read_csv(TABLE/'variance_predictors_by_seed.csv.gz')
    metric_cols = ['raw_energy', 'trace', 'V']
    seed_stats = seeds.groupby(KEY+['method','tau_255'])[metric_cols].agg(['mean','std'])
    seed_stats.columns = [f'{metric}_{stat}' for metric,stat in seed_stats.columns]
    seed_stats = seed_stats.reset_index()
    seed_stats['tau_token'] = seed_stats.tau_255.map(
        lambda x: str(int(x)) if float(x).is_integer() else str(x).replace('.','p'))
    seed_stats['geometry_label'] = np.where(
        seed_stats.method.isin(['exact','local_mc']), seed_stats.method,
        seed_stats.method+'_tau255_'+seed_stats.tau_token)
    assert not seed_stats.duplicated(KEY+['geometry_label']).any()
    wide = seed_stats.pivot(index=KEY, columns='geometry_label',
                            values=[f'{m}_{s}' for m in metric_cols for s in ['mean','std']])
    wide.columns = [f'geom_{label}__{metric}' for metric,label in wide.columns]
    return wide.reset_index()


def ridge_and_qc(manifest):
    ridge = pd.read_csv(TABLE/'ridge_metadata.csv')
    values = pd.read_csv(TABLE/'variance_predictors.csv')
    fixed = values[values.method=='exact'].copy()
    fixed = fixed[KEY+['lambda_normalized','kappa','kappa_residual','df2','fd_error']]
    fixed = fixed.rename(columns={c:f'ridge_{c}' for c in
        ['lambda_normalized','kappa','kappa_residual','df2']}).rename(columns={'fd_error':'geometry_fd_error'})
    ridge = ridge.rename(columns={c:f'ridge_{c}' for c in ridge.columns if c not in ['subject','model','unit']})
    out = manifest.merge(ridge,on=['subject','unit','model'],validate='one_to_one')
    return out.merge(fixed,on=KEY,validate='one_to_one')


def schema_for(frame):
    descriptions = {
        'subject':'Recording/encoding subject identifier.', 'monkey':'Short animal identifier.',
        'unit':'Absolute recording channel/unit index.', 'model':'Encoding feature model identifier.',
        'site_id':'Stable subject-unit identifier.', 'geometry_id':'Shared feature/layer/split geometry identifier.',
        'region':'Recorded visual area.', 'robust_model':'Upstream robust-model group: CLIPAG or robust RN50.',
        'biology_reliability':'Source neural reliability metadata.',
        'control_error_over_S_nat_observed':'Control identity MSE divided by held-out natural measured-response variance; closest available empirical proxy to E_acc/S, but its denominator includes measurement noise.',
        'control_error_over_S_control_observed':'Control identity MSE divided by control measured-response variance; exactly 1-control_r2_identity and therefore descriptive/tautological.',
        'gen_test_S_nat_noise_corrected':'Held-out natural measured-response variance minus the repeat-estimated contribution of trial-mean measurement noise; empirical proxy for latent natural teacher signal power S.',
        'control_mse_noise_corrected':'Control identity MSE minus repeat-estimated trial-mean response noise; may be negative and is not clipped.',
        'control_error_over_S_nat_noise_corrected':'Raw control identity MSE divided by repeat-noise-corrected natural signal variance.',
        'control_error_noise_corrected_over_S_nat_noise_corrected':'Repeat-noise-corrected control identity MSE divided by repeat-noise-corrected natural signal variance; sensitivity metric because control repeat coverage varies and can be sparse.',
        'control_trialmean_noise_high_coverage':'True when at least 80% of matched control stimuli have at least two trials for noise estimation.',
        'control_error_over_ceiling_signal_variance':'Control identity MSE divided by upstream control noise-ceiling signal variance; unavailable for Leap and Three0 and is path-specific, not natural S.',
    }
    rows=[]
    for col,dtype in frame.dtypes.items():
        if col.startswith('geom_'):
            group='geometry'
            desc='Across 10 image seeds: '+('mean' if col.endswith('_mean') else 'sample SD')+' of '+col.split('__',1)[1].rsplit('_',1)[0]+'.'
        elif col.startswith('gen_test_'): group,desc='generalization','Held-out natural-image '+col.removeprefix('gen_test_').replace('_',' ')+'.'
        elif col.startswith('crossphase_'): group,desc='crossphase','Control-session anchor '+col.removeprefix('crossphase_anchor_').replace('_',' ')+'.'
        elif col.startswith('control_'): group,desc='control','Biological control '+col.removeprefix('control_').replace('_',' ')+'.'
        elif col.startswith('ridge_'): group,desc='ridge','Ridge/DE '+col.removeprefix('ridge_').replace('_',' ')+'.'
        else: group,desc='identity',descriptions.get(col,col.replace('_',' ')+'.')
        rows.append(dict(column=col,group=group,dtype=str(dtype),description=descriptions.get(col,desc)))
    return pd.DataFrame(rows)


def correlation_table(synopsis):
    configurations=[('exact',0.),('local_mc',0.)]
    configurations += [(method,tau) for method in ['smooth','neighborhood','variance','step','stein']
                       for tau in [.5,2.,8.,16.]]
    rows=[]
    for method,tau in configurations:
        label=method if method in ['exact','local_mc'] else f'{method}_tau255_{str(int(tau)) if tau.is_integer() else str(tau).replace(".","p")}'
        for quantity in ['trace','V']:
            predictor=f'geom_{label}__{quantity}_mean'
            for subset,data in [('all_250',synopsis),
                                ('without_CLIPAG_and_robust_RN50',synopsis[~synopsis.robust_model])]:
                for outcome in ['control_slope','control_slope_within_seed','control_mse',
                        'control_error_over_S_nat_observed','control_error_over_S_nat_noise_corrected',
                        'control_error_noise_corrected_over_S_nat_noise_corrected',
                        'control_r2_identity']:
                    valid=np.isfinite(data[predictor])&np.isfinite(data[outcome])
                    d=data.loc[valid,[predictor,outcome,'site_id']].copy()
                    x=d[predictor].to_numpy(); y=d[outcome].to_numpy()
                    row=dict(method=method,tau_255=tau,quantity=quantity,predictor=predictor,
                        subset=subset,outcome=outcome,n=len(d),pearson_raw=np.nan,
                        pearson_log10=np.nan,spearman=np.nan,
                        site_demeaned_spearman_raw=np.nan,site_demeaned_spearman_log10=np.nan)
                    if len(d)>2 and np.std(x)>0 and np.std(y)>0:
                        row['pearson_raw']=pearsonr(x,y).statistic
                        row['spearman']=spearmanr(x,y).statistic
                        if np.all(x>0): row['pearson_log10']=pearsonr(np.log10(x),y).statistic
                        centered=d[[predictor,outcome]]-d.groupby('site_id')[[predictor,outcome]].transform('mean')
                        row['site_demeaned_spearman_raw']=spearmanr(centered[predictor],centered[outcome]).statistic
                        if np.all(x>0):
                            logged=d.copy(); logged[predictor]=np.log10(logged[predictor])
                            centered_log=logged[[predictor,outcome]]-logged.groupby('site_id')[[predictor,outcome]].transform('mean')
                            row['site_demeaned_spearman_log10']=spearmanr(centered_log[predictor],centered_log[outcome]).statistic
                    rows.append(row)
    return pd.DataFrame(rows)


def plot_smoothing(corr):
    colors={'all_250':'#3366aa','without_CLIPAG_and_robust_RN50':'#aa3377'}
    fig,axs=plt.subplots(1,2,figsize=(11,4.3),sharey=True)
    for ax,method in zip(axs,['smooth','neighborhood']):
        for subset,color in colors.items():
            d=corr[(corr.method==method)&(corr.quantity=='V')&(corr.outcome=='control_slope')&(corr.subset==subset)]
            ax.plot(d.tau_255,d.spearman,'o-',color=color,label=subset.replace('_',' '))
            ax.plot(d.tau_255,d.pearson_log10,'s--',color=color,alpha=.75)
        ax.axhline(0,color='.6',linewidth=.8); ax.set_xscale('log',base=2)
        ax.set_xticks([.5,2,8,16],[.5,2,8,16]); ax.set_xlabel('Noise SD × 255'); ax.set_title(method)
        ax.spines[['top','right']].set_visible(False)
    axs[0].set_ylabel('Correlation with biological control slope')
    axs[0].legend(fontsize=8)
    fig.suptitle('250-row synopsis: circles = Spearman; squares = Pearson(log10 V)')
    fig.tight_layout(); FIGURE.mkdir(parents=True,exist_ok=True)
    fig.savefig(FIGURE/'synopsis_smoothing_level_correlations.png',dpi=180)
    plt.close(fig)


def validate(synopsis, schema):
    assert len(synopsis)==250 and not synopsis.duplicated(KEY).any()
    assert synopsis.groupby(['subject','unit']).size().eq(10).all()
    assert synopsis.groupby('model').size().eq(25).all()
    assert len(schema)==len(synopsis.columns) and schema.column.is_unique
    assert synopsis.filter(regex=r'^geom_exact__').shape[1]==6
    assert synopsis.filter(regex=r'^geom_smooth_tau255_').shape[1]==24
    numeric=synopsis.select_dtypes(include=[np.number])
    assert not np.isinf(numeric.to_numpy()).any()
    for v_col in synopsis.filter(regex=r'^geom_.*__V_mean$').columns:
        trace_col=v_col.replace('__V_mean','__trace_mean')
        np.testing.assert_allclose(synopsis[v_col],synopsis.gen_test_mse*synopsis[trace_col]/synopsis.n_train,
                                   rtol=2e-7,atol=1e-12)
    np.testing.assert_allclose(synopsis.control_error_over_S_control_observed,
                               1-synopsis.control_r2_identity,rtol=1e-10,atol=1e-10)
    np.testing.assert_allclose(synopsis.gen_test_error_over_S_nat_observed,
                               1-synopsis.gen_test_r2_identity,rtol=1e-10,atol=1e-10)
    assert synopsis.gen_test_trialmean_noise_fraction.min()==1
    assert synopsis.control_trialmean_noise_fraction.between(0,1).all()


def main():
    manifest=pd.read_csv(REPO/'tables/nonlinear_control/scale_manifest/site_model_manifest.csv')
    manifest['monkey']=manifest.subject.str.split('_').str[0]
    manifest=manifest[KEY+['geometry_id','layer','n_total','n_pcs','n_train']]
    base=ridge_and_qc(manifest)
    bio=biological_metrics(manifest)
    synopsis=base.merge(bio,on=KEY+['geometry_id'],validate='one_to_one').merge(geometry_wide(),on=KEY,validate='one_to_one')
    synopsis=synopsis.sort_values(KEY).reset_index(drop=True)
    schema=schema_for(synopsis)
    validate(synopsis,schema)
    synopsis.to_csv(TABLE/'biological_validation_synopsis_v1.csv.gz',index=False,compression={'method':'gzip','compresslevel':6,'mtime':0})
    try:
        synopsis.to_parquet(TABLE/'biological_validation_synopsis_v1.parquet',index=False)
        parquet=True
    except ImportError:
        parquet=False
    schema.to_csv(TABLE/'biological_validation_synopsis_v1_schema.csv',index=False)
    corr=correlation_table(synopsis)
    corr.to_csv(TABLE/'synopsis_correlations.csv',index=False)
    plot_smoothing(corr)
    metadata=dict(version=VERSION,rows=len(synopsis),columns=len(synopsis.columns),key=KEY,
        parquet_written=parquet,canonical_geometry_source='variance_predictors_by_seed.csv.gz',
        normalization_note='Theory S is latent natural teacher signal variance. gen_test_S_nat_noise_corrected is the primary empirical denominator (100% repeat coverage). Control numerator noise correction is sensitivity-only because repeat coverage varies; raw control MSE over corrected natural S is the primary normalized control error.',
        robust_models=sorted(ROBUST_MODELS))
    (TABLE/'biological_validation_synopsis_v1_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata,indent=2))


if __name__=='__main__':
    main()

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
VERSION = '1.3.0'

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


def control_session_test_cloud(loader, monkey, unit, model):
    """Encoding-held-out natural images re-presented during the control session."""
    brain=loader.load_brain(monkey)
    encoding=loader.load_encoding(monkey)
    ui=list(brain['units']).index(unit)
    mi=list(encoding['models']).index(model)
    predicted=dict(zip(encoding['stim'].tolist(),encoding['pred'][ui,mi]))
    test_names=set(loader._test_names(monkey))
    control=brain['control']
    x,y,names=[],[],[]
    for i,(name,kind) in enumerate(zip(control['stim'],control['kind'])):
        if kind=='calibration' and name in test_names and name in predicted:
            x.append(predicted[name]); y.append(control['resp_z'][i,ui]); names.append(name)
    return np.asarray(x,float),np.asarray(y,float),names


def cross_session_neural_anchor_cloud(loader,monkey,unit,subset='all'):
    """Matched natural anchors: encoding response and control response.

    Control responses have already received the official within-control-day
    anchorDay normalization. This cloud estimates the additional frozen affine
    mapping from encoding-session units to control-session units.
    """
    brain=loader.load_brain(monkey)
    ui=list(brain['units']).index(unit)
    encoding_measured=dict(zip(brain['calibration']['stim'].astype(str),
                               brain['calibration']['resp_z'][:,ui]))
    assert subset in {'all','train','heldout'}
    heldout=loader._test_names(monkey)
    yenc,yctrl,names=[],[],[]
    control=brain['control']
    for name,kind,response in zip(control['stim'],control['kind'],control['resp_z'][:,ui]):
        name=str(name)
        selected=(subset=='all' or (subset=='heldout' and name in heldout) or
                  (subset=='train' and name not in heldout))
        if kind=='calibration' and name in encoding_measured and selected:
            yenc.append(encoding_measured[name])
            yctrl.append(response);names.append(name)
    return tuple(np.asarray(x,float) for x in [yenc,yctrl]),names


def affine_map(source,target):
    metrics=regression_metrics(source,target)
    assert metrics['n']>=2 and np.isfinite(metrics['slope']) and np.isfinite(metrics['intercept'])
    return metrics['intercept'],metrics['slope'],metrics


def biological_metrics(manifest):
    _, loader, _ = upstream_modules()
    control = pd.read_csv(TABLE/'control_clouds.csv.gz')
    control_groups = {k:g for k,g in control.groupby(['subject','unit','model'], sort=False)}
    rows = [];affine_cloud_rows=[];site_affine_cache={}
    for row in manifest.itertuples(index=False):
        monkey = row.monkey
        brain = loader.load_brain(monkey)
        ui = list(brain['units']).index(row.unit)
        ceilings = brain['ceilings']
        ci = np.where((ceilings['unit']==row.unit) & (ceilings['model']==row.model))[0]
        assert len(ci)==1
        ci = ci[0]
        x_encoding_test, y_encoding_test = loader.encoding_cloud(monkey, row.unit, row.model, split='test')
        x_control_test, y_control_test, control_test_names = control_session_test_cloud(
            loader,monkey,row.unit,row.model)
        x_anchor, y_anchor = loader.anchor_cloud(monkey, row.unit, row.model)
        cg = control_groups[(row.subject, row.unit, row.model)]
        encoding_gen = regression_metrics(x_encoding_test, y_encoding_test)
        control_gen = regression_metrics(x_control_test, y_control_test)
        cross = regression_metrics(x_anchor, y_anchor)
        ctl = regression_metrics(cg.predicted, cg.measured)
        # Leakage-free cross-session calibration: fit on natural anchors only,
        # freeze the map, then evaluate it on the disjoint accentuated stimuli.
        site_key=(monkey,row.unit)
        if site_key not in site_affine_cache:
            site_affine_cache[site_key]={}
            for anchor_label,anchor_subset in [('anchor','all'),('train_anchor','train'),
                                               ('heldout_anchor','heldout')]:
                (anchor_enc,anchor_ctrl),anchor_names=cross_session_neural_anchor_cloud(
                    loader,monkey,row.unit,subset=anchor_subset)
                site_a,site_b,site_fit=affine_map(anchor_enc,anchor_ctrl)
                site_affine_cache[site_key][anchor_label]=dict(
                    site_fit=site_fit,site_a=site_a,site_b=site_b,anchor_names=anchor_names)
        anchor_variants={}
        for anchor_label,cached_map in site_affine_cache[site_key].items():
            site_a,site_b=cached_map['site_a'],cached_map['site_b']
            site_prediction=site_a+site_b*cg.predicted.to_numpy()
            anchor_variants[anchor_label]=dict(cached_map,
                site_control=regression_metrics(site_prediction,cg.measured))
        all_map=anchor_variants['anchor'];train_map=anchor_variants['train_anchor']
        for stimulus,seed,predicted,measured in zip(cg.stimulus,cg.seed,cg.predicted,cg.measured):
            affine_cloud_rows.append(dict(subject=row.subject,monkey=monkey,unit=row.unit,
                model=row.model,stimulus=stimulus,seed=seed,predicted_identity=predicted,
                predicted_site_anchor_affine=all_map['site_a']+all_map['site_b']*predicted,
                predicted_site_train_anchor_affine=train_map['site_a']+train_map['site_b']*predicted,
                measured=measured))
        calibration=brain['calibration']
        encoding_measured=dict(zip(calibration['stim'].astype(str),calibration['resp_z'][:,ui]))
        y_encoding_matched=np.asarray([encoding_measured[str(name)] for name in control_test_names],float)
        encoding_matched_gen=regression_metrics(x_control_test,y_encoding_matched)
        drift=regression_metrics(y_encoding_matched,y_control_test)
        centered = cg[['predicted','measured']]-cg.groupby('seed')[['predicted','measured']].transform('mean')
        within_denom = float(np.square(centered.predicted).sum())
        control_slope_within_seed = (float((centered.predicted*centered.measured).sum()/within_denom)
                                     if within_denom > 0 else np.nan)
        s_nat_encoding = encoding_gen['measured_variance']
        s_nat_control = control_gen['measured_variance']
        s_control = ctl['measured_variance']
        test_names=set(loader._test_names(monkey))
        heldout_names=[n for n in calibration['stim'] if n in test_names]
        encoding_gen_noise,n_encoding_gen_noise,n_encoding_gen_total=trial_mean_noise(
            calibration['trial_stim'],calibration['trial_z'][:,ui],heldout_names)
        control_data=brain['control']
        control_gen_noise,n_control_gen_noise,n_control_gen_total=trial_mean_noise(
            control_data['trial_stim'],control_data['trial_z'][:,ui],control_test_names)
        control_noise,n_control_noise,n_control_total=trial_mean_noise(
            control_data['trial_stim'],control_data['trial_z'][:,ui],cg.stimulus.tolist())
        # Across M fixed stimuli, centering removes mean(v_i)/M from the expected
        # population variance. With incomplete repeat coverage, mean(v_i) is a plug-in.
        s_nat_encoding_corrected=(s_nat_encoding-(1-1/len(y_encoding_test))*encoding_gen_noise
                                  if np.isfinite(encoding_gen_noise) else np.nan)
        s_nat_control_corrected=(s_nat_control-(1-1/len(y_control_test))*control_gen_noise
                                 if np.isfinite(control_gen_noise) else np.nan)
        control_mse_corrected=ctl['mse']-control_noise if np.isfinite(control_noise) else np.nan
        signal_var = float(ceilings['signal_var'][ci])
        noise_var = float(ceilings['noise_var'][ci])
        values = dict(subject=row.subject, monkey=monkey, unit=row.unit, model=row.model,
            region=str(brain['region'][ui]), robust_model=row.model in ROBUST_MODELS,
            site_id=f'{row.subject}:unit{row.unit}', geometry_id=row.geometry_id,
            biology_reliability=float(brain['reliability'][ui]),
            firing_floor=float(brain['firing_floor'][ui]),
            **prefix(encoding_gen, 'encoding_session_gen_test'),
            **prefix(encoding_matched_gen, 'encoding_session_matched_gen_test'),
            **prefix(control_gen, 'control_session_gen_test'),
            **prefix(drift, 'session_drift_response'),
            **prefix(cross, 'crossphase_anchor'), **prefix(ctl, 'control'))
        for anchor_label,result in anchor_variants.items():
            values.update(**prefix(result['site_fit'],f'crosssession_site_{anchor_label}_fit'),
                          **prefix(result['site_control'],f'control_site_{anchor_label}_affine'))
            # Held-out natural generalization after freezing the same affine.
            site_gen_prediction=result['site_a']+result['site_b']*x_control_test
            values.update(**prefix(regression_metrics(site_gen_prediction,y_control_test),
                                   f'control_session_site_{anchor_label}_affine_gen_test'))
        # One neuron/site mapping fitted on encoding-training anchors and
        # shared unchanged across all ten model predictions.
        for metric,value in anchor_variants['train_anchor']['site_control'].items():
            values[f'control_anchor_affine_{metric}']=value
        values.update(
            session_drift_gen_mse_control_minus_encoding=control_gen['mse']-encoding_gen['mse'],
            session_drift_gen_mse_control_over_encoding=control_gen['mse']/encoding_gen['mse'],
            encoding_session_gen_test_error_over_S_nat_observed=(encoding_gen['mse']/s_nat_encoding
                if s_nat_encoding > 0 else np.nan),
            encoding_session_gen_test_trialmean_noise_variance=encoding_gen_noise,
            encoding_session_gen_test_trialmean_noise_n_stim=n_encoding_gen_noise,
            encoding_session_gen_test_trialmean_noise_fraction=n_encoding_gen_noise/n_encoding_gen_total,
            encoding_session_gen_test_S_nat_noise_corrected=s_nat_encoding_corrected,
            control_session_gen_test_error_over_S_nat_observed=(control_gen['mse']/s_nat_control
                if s_nat_control > 0 else np.nan),
            control_session_gen_test_trialmean_noise_variance=control_gen_noise,
            control_session_gen_test_trialmean_noise_n_stim=n_control_gen_noise,
            control_session_gen_test_trialmean_noise_fraction=n_control_gen_noise/n_control_gen_total,
            control_session_gen_test_trialmean_noise_high_coverage=n_control_gen_noise/n_control_gen_total >= .8,
            control_session_gen_test_S_nat_noise_corrected=s_nat_control_corrected,
            control_error_over_S_nat_encoding_session_observed=(ctl['mse']/s_nat_encoding
                if s_nat_encoding > 0 else np.nan),
            control_error_over_S_nat_control_session_observed=(ctl['mse']/s_nat_control
                if s_nat_control > 0 else np.nan),
            control_error_over_S_control_observed=ctl['mse']/s_control if s_control > 0 else np.nan,
            control_trialmean_noise_variance=control_noise,
            control_trialmean_noise_n_stim=n_control_noise,
            control_trialmean_noise_fraction=n_control_noise/n_control_total,
            control_trialmean_noise_high_coverage=n_control_noise/n_control_total >= .8,
            control_mse_noise_corrected=control_mse_corrected,
            control_error_over_S_nat_encoding_session_noise_corrected=(ctl['mse']/s_nat_encoding_corrected
                if np.isfinite(s_nat_encoding_corrected) and s_nat_encoding_corrected > 0 else np.nan),
            control_error_over_S_nat_control_session_noise_corrected=(ctl['mse']/s_nat_control_corrected
                if np.isfinite(s_nat_control_corrected) and s_nat_control_corrected > 0 else np.nan),
            control_error_noise_corrected_over_S_nat_control_session_noise_corrected=(control_mse_corrected/s_nat_control_corrected
                if np.isfinite(control_mse_corrected) and np.isfinite(s_nat_control_corrected) and s_nat_control_corrected > 0 else np.nan),
            control_slope_within_seed=control_slope_within_seed,
            control_noise_ceiling_r=float(ceilings['nc_r'][ci]),
            control_noise_ceiling_signal_variance=signal_var,
            control_noise_ceiling_noise_variance=noise_var,
            control_noise_ceiling_n_stim=int(ceilings['n_stim'][ci]),
            control_error_over_ceiling_signal_variance=(ctl['mse']/signal_var
                if np.isfinite(signal_var) and signal_var > 0 else np.nan))
        rows.append(values)
    pd.DataFrame(affine_cloud_rows).to_csv(TABLE/'control_anchor_affine_clouds.csv.gz',index=False,
        compression={'method':'gzip','compresslevel':6,'mtime':0})
    return pd.DataFrame(rows)


def geometry_wide():
    seeds = pd.read_csv(TABLE/'variance_predictors_by_seed.csv.gz')
    metric_cols = ['raw_energy', 'trace']
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


def add_session_specific_V(synopsis):
    """Attach both generalization-error choices to every geometry trace."""
    trace_columns=synopsis.filter(regex=r'^geom_.*__trace_(mean|std)$').columns
    endpoints={
        'control_session':'control_session_gen_test_mse',
        'control_session_site_train_anchor_affine':'control_session_site_train_anchor_affine_gen_test_mse',
        'encoding_session':'encoding_session_gen_test_mse',
        'encoding_session_matched':'encoding_session_matched_gen_test_mse',
    }
    additions={}
    for trace_col in trace_columns:
        root_name,stat=trace_col.rsplit('__trace_',1)
        for endpoint,mse_col in endpoints.items():
            additions[f'{root_name}__V_{endpoint}_{stat}']=(
                synopsis[mse_col]*synopsis[trace_col]/synopsis.n_train).to_numpy()
    return pd.concat([synopsis,pd.DataFrame(additions,index=synopsis.index)],axis=1)


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
        'control_error_over_S_nat_control_session_observed':'Primary empirical E_acc/S proxy: control identity MSE divided by held-out natural response variance measured in the control session.',
        'control_error_over_S_nat_control_session_noise_corrected':'Sensitivity E_acc/S proxy using repeat-noise correction in the control-session natural response denominator; correction coverage is low for Leap and Three0.',
        'control_error_over_S_control_observed':'Control identity MSE divided by control measured-response variance; exactly 1-control_r2_identity and therefore descriptive/tautological.',
        'control_mse_noise_corrected':'Control identity MSE minus repeat-estimated trial-mean response noise; may be negative and is not clipped.',
        'control_error_noise_corrected_over_S_nat_control_session_noise_corrected':'Repeat-noise-corrected control identity MSE divided by control-session repeat-noise-corrected natural signal variance; sensitivity metric because accentuated-stimulus repeat coverage varies and can be sparse.',
        'control_trialmean_noise_high_coverage':'True when at least 80% of matched control stimuli have at least two trials for noise estimation.',
        'control_error_over_ceiling_signal_variance':'Control identity MSE divided by upstream control noise-ceiling signal variance; unavailable for Leap and Three0 and is path-specific, not natural S.',
        'control_anchor_affine_mse':'Default cross-session-calibrated MSE: one site/neuron affine fitted on encoding-training natural anchors and shared unchanged across all models.',
    }
    rows=[]
    for col,dtype in frame.dtypes.items():
        if col.startswith('geom_'):
            group='geometry'
            desc='Across 10 image seeds: '+('mean' if col.endswith('_mean') else 'sample SD')+' of '+col.split('__',1)[1].rsplit('_',1)[0]+'.'
        elif col.startswith('encoding_session_gen_test_'): group,desc='generalization_encoding_session','Encoding-session held-out natural-image '+col.removeprefix('encoding_session_gen_test_').replace('_',' ')+'.'
        elif col.startswith('encoding_session_matched_gen_test_'): group,desc='generalization_encoding_session_matched','Encoding-session response on the exact held-out image subset re-presented during control '+col.removeprefix('encoding_session_matched_gen_test_').replace('_',' ')+'.'
        elif col.startswith('control_session_gen_test_'): group,desc='generalization_control_session','Control-session response on encoding-held-out natural-image '+col.removeprefix('control_session_gen_test_').replace('_',' ')+'.'
        elif col.startswith('session_drift_'): group,desc='session_drift','Encoding-to-control recording-session drift '+col.removeprefix('session_drift_').replace('_',' ')+'.'
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
        for quantity in ['trace','V_control_session',
                         'V_control_session_site_train_anchor_affine',
                         'V_encoding_session','V_encoding_session_matched']:
            predictor=f'geom_{label}__{quantity}_mean'
            subsets=[('all_250',synopsis),
                     ('without_CLIPAG_and_robust_RN50',synopsis[~synopsis.robust_model]),
                     ('only_n50_monkeys',synopsis[synopsis.monkey.isin(['red','paul','venus'])]),
                     ('without_robust_only_n50_monkeys',synopsis[(~synopsis.robust_model)&synopsis.monkey.isin(['red','paul','venus'])])]
            subsets += [(f'without_robust_leaveout_{monkey}',synopsis[(~synopsis.robust_model)&(synopsis.monkey!=monkey)])
                        for monkey in sorted(synopsis.monkey.unique())]
            for subset,data in subsets:
                for outcome in ['control_slope','control_slope_within_seed','control_mse',
                        'control_error_over_S_nat_control_session_observed',
                        'control_error_over_S_nat_control_session_noise_corrected',
                        'control_error_over_S_nat_encoding_session_noise_corrected',
                        'control_error_noise_corrected_over_S_nat_control_session_noise_corrected',
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
    endpoints={
        'control_session': ('V_control_session',
            'Control-session generalization V\nheld-out n = 50 (red/paul/venus), 24 (Leap), 22 (Three0)'),
        'encoding_session': ('V_encoding_session',
            'Encoding-session generalization V\nheld-out n = 195 for every site/model'),
    }
    FIGURE.mkdir(parents=True,exist_ok=True)
    for endpoint,(quantity,title) in endpoints.items():
        fig,axs=plt.subplots(1,2,figsize=(11,4.5),sharey=True)
        for ax,method in zip(axs,['smooth','neighborhood']):
            for subset,color in colors.items():
                d=corr[(corr.method==method)&(corr.quantity==quantity)&
                       (corr.outcome=='control_slope')&(corr.subset==subset)]
                ax.plot(d.tau_255,d.spearman,'o-',color=color,label=subset.replace('_',' '))
                ax.plot(d.tau_255,d.pearson_log10,'s--',color=color,alpha=.75)
            ax.axhline(0,color='.6',linewidth=.8); ax.set_xscale('log',base=2)
            ax.set_xticks([.5,2,8,16],[.5,2,8,16]); ax.set_xlabel('Noise SD × 255'); ax.set_title(method)
            ax.spines[['top','right']].set_visible(False)
        axs[0].set_ylabel('Correlation with biological control slope')
        axs[0].legend(fontsize=8)
        fig.suptitle(title+'\nCircles = Spearman; squares = Pearson(log10 V)')
        fig.tight_layout()
        fig.savefig(FIGURE/f'synopsis_smoothing_level_correlations_{endpoint}.png',dpi=180)
        plt.close(fig)


def validate(synopsis, schema):
    assert len(synopsis)==250 and not synopsis.duplicated(KEY).any()
    assert synopsis.groupby(['subject','unit']).size().eq(10).all()
    assert synopsis.groupby('model').size().eq(25).all()
    assert len(schema)==len(synopsis.columns) and schema.column.is_unique
    assert synopsis.filter(regex=r'^geom_exact__').shape[1]==12
    assert synopsis.filter(regex=r'^geom_smooth_tau255_').shape[1]==48
    numeric=synopsis.select_dtypes(include=[np.number])
    assert not np.isinf(numeric.to_numpy()).any()
    endpoints={'control_session':'control_session_gen_test_mse',
               'control_session_site_train_anchor_affine':'control_session_site_train_anchor_affine_gen_test_mse',
               'encoding_session':'encoding_session_gen_test_mse',
               'encoding_session_matched':'encoding_session_matched_gen_test_mse'}
    for endpoint,mse_col in endpoints.items():
        mse=synopsis[mse_col]
        for v_col in synopsis.filter(regex=rf'^geom_.*__V_{endpoint}_mean$').columns:
            trace_col=v_col.replace(f'__V_{endpoint}_mean','__trace_mean')
            np.testing.assert_allclose(synopsis[v_col],mse*synopsis[trace_col]/synopsis.n_train,
                                       rtol=2e-7,atol=1e-12)
    np.testing.assert_allclose(synopsis.control_error_over_S_control_observed,
                               1-synopsis.control_r2_identity,rtol=1e-10,atol=1e-10)
    for metric in ['n','mse','rmse','mae','bias_measured_minus_predicted','measured_mean',
                   'predicted_mean','measured_variance','predicted_variance','r','slope',
                   'intercept','refit_mse','r2_identity','r2_refit']:
        np.testing.assert_allclose(synopsis[f'control_anchor_affine_{metric}'],
            synopsis[f'control_site_train_anchor_affine_{metric}'],rtol=0,atol=0)
    for session in ['control','encoding']:
        np.testing.assert_allclose(synopsis[f'{session}_session_gen_test_error_over_S_nat_observed'],
                                   1-synopsis[f'{session}_session_gen_test_r2_identity'],rtol=1e-10,atol=1e-10)
    assert synopsis.encoding_session_gen_test_trialmean_noise_fraction.min()==1
    assert synopsis.control_session_gen_test_trialmean_noise_fraction.between(0,1).all()
    assert synopsis.control_trialmean_noise_fraction.between(0,1).all()


def main():
    manifest=pd.read_csv(REPO/'tables/nonlinear_control/scale_manifest/site_model_manifest.csv')
    manifest['monkey']=manifest.subject.str.split('_').str[0]
    manifest=manifest[KEY+['geometry_id','layer','n_total','n_pcs','n_train']]
    base=ridge_and_qc(manifest)
    bio=biological_metrics(manifest)
    synopsis=base.merge(bio,on=KEY+['geometry_id'],validate='one_to_one').merge(geometry_wide(),on=KEY,validate='one_to_one')
    synopsis=add_session_specific_V(synopsis)
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
        primary_generalization='control_session_site_train_anchor_affine_gen_test: encoding-heldout natural predictions mapped by the site-level affine fitted on encoding-training anchors',
        identity_generalization_reference='control_session_gen_test: encoding-heldout natural predictions compared directly with control-session responses',
        secondary_generalization='encoding_session_gen_test: original encoding-session responses; retained to quantify session drift',
        matched_generalization='encoding_session_matched_gen_test: encoding-session responses restricted to the exact control-session held-out image subset',
        primary_geometry_V='V_control_session_site_train_anchor_affine = control_session_site_train_anchor_affine_gen_test_mse * trace / n_train',
        identity_geometry_V_reference='V_control_session = control_session_gen_test_mse * trace / n_train',
        control_response_scale='Official anchorDay normalization is applied within each control day before stimulus averaging. The unqualified control_* metrics already use this response scale.',
        anchor_affine_sensitivity='control_anchor_affine_* is one site/neuron encoding-response to control-response affine fitted on encoding-training natural anchors and shared unchanged across all ten models.',
        recommended_theory_control_endpoint='control_site_train_anchor_affine_mse: site-level encoding-neural to control-neural affine fitted on encoding-training natural anchors, then frozen on accentuated stimuli',
        recommended_theory_generalization='control_session_site_train_anchor_affine_gen_test_mse: the same train-anchor affine evaluated on held-out natural anchors',
        normalization_note='Theory S is latent natural teacher signal variance. Control-session observed held-out natural-response variance is primary because it matches the control recording session. Its repeat-noise correction is sensitivity-only: repeat coverage is complete for red/paul/venus but low for Leap/Three0. Encoding-session observed and fully repeat-noise-corrected versions are retained.',
        robust_models=sorted(ROBUST_MODELS))
    (TABLE/'biological_validation_synopsis_v1_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata,indent=2))


if __name__=='__main__':
    main()

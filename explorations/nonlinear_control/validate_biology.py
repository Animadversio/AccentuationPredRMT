"""Basic biological validation; CPU-only, cached, pinned upstream preprocessing.

V is a variance contribution proxy, NOT a quantitative prediction of control slope.
Run setup, preprocess (one monkey timing pilot first), then analyze.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import spearmanr
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[2]
LAB = Path('/n/holylabs/LABS/alvarez_lab/Lab/VVS_Accentuation')
SHA = '9b6fd95228d1b77146f8b279c7c228914ca34bb8'
TABLE = REPO / 'tables/nonlinear_control/biological_validation'
MONKEYS = ['red', 'paul', 'venus', 'leap', 'three0']


def root():
    return Path(os.environ['STORE_DIR']) / 'Projects/AccentuationPredRMT/nonlinear_control/biological_validation_v1'


def setup(upstream):
    r = root()
    r.mkdir(parents=True, exist_ok=True)
    dest = r / 'upstream'
    if not dest.exists():
        shutil.copytree(upstream, dest)
    assert subprocess.check_output(['git', '-C', str(dest), 'rev-parse', 'HEAD'], text=True).strip() == SHA
    source = r / 'source_data'
    source.mkdir(exist_ok=True)
    for name, target in [('image_pca_projections', LAB / 'Encoding_models'),
                         ('model_predictions', LAB / 'Encoding_models'),
                         ('brain_data_control', LAB / 'Ephys_Data')]:
        p = source / name
        if not p.exists():
            p.symlink_to(target, target_is_directory=True)
    enc = source / 'brain_data_encoding'
    enc.mkdir(exist_ok=True)
    files = list((LAB / 'Ephys_Data').glob('*vvs-encodingstimuli*.h5'))
    files += list((LAB / 'NeuralData_raw').glob('*sessdata.pkl'))
    for f in files:
        p = enc / f.name
        if not p.exists():
            p.symlink_to(f)
    (r / 'cache').mkdir(exist_ok=True)
    (r / 'logs').mkdir(exist_ok=True)
    (r / 'provenance.json').write_text(json.dumps(dict(upstream_sha=SHA, lab=str(LAB),
        source_files=[str(f) for f in files], scope='basic variance contribution only'), indent=2))
    print(f'Source adapter ready: {r}; {len(files)} encoding/session sources', flush=True)


def upstream_modules():
    r = root()
    os.environ['PNC_SOURCE_DATA'] = str(r / 'source_data')
    os.environ['PNC_PREPROCESSED_DATA'] = str(r / 'cache')
    os.environ['PNC_OUTPUT'] = str(r / 'outputs')
    sys.path.insert(0, str(r / 'upstream'))
    spec = importlib.util.spec_from_file_location('pnc_run_preproc', r / 'upstream/scripts/preprocessing/run_preproc.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from pnc.preproc import loader, encoding
    return module, loader, encoding


def preprocess(monkeys):
    p, _, _ = upstream_modules()
    cfg = dict(p.CONFIG, monkeys=monkeys, experiments=['main'])
    times = []
    for monkey in tqdm(monkeys, desc='Official biological preprocessing'):
        start = time.monotonic()
        for name, fun in [('brain', p.build_brain_main), ('encoding', p.build_encoding),
                          ('predictions', p.build_predictions)]:
            if not (root() / 'cache' / f'{name}_{monkey}.pkl').exists():
                print(f'Start {name}_{monkey}',flush=True)
                fun(monkey, cfg)
        times.append(dict(monkey=monkey, seconds=time.monotonic()-start))
        print(json.dumps(times[-1]), flush=True)
    available = [m for m in MONKEYS if (root() / 'cache' / f'brain_{m}.pkl').exists()]
    try:
        p.build_stimuli(dict(cfg, monkeys=available))
    except KeyError as exc:
        # Upstream writes a valid main-experiment cache, then its final print
        # indexes an absent optional controversial table. Validate before accepting.
        with open(root()/'cache/stimuli.pkl','rb') as f:
            st=pickle.load(f)
        if exc.args != ('stimulus_name',) or st['controversial'] or not st['calibration']:
            raise
        print('Main stimulus cache complete; optional controversial experiment absent.',flush=True)
    pd.DataFrame(times).to_csv(root() / f'timing_{"_".join(monkeys)}.csv', index=False)


def leap_audit():
    """Reconstruct the missing frozen pixel audit from original saved RGB images.

    Uses all flattened RGB channels, no resizing. This is a reconstructed audit,
    not a claim of byte-identical reproduction of the unavailable frozen file.
    """
    from PIL import Image
    subject = 'leap_250426-250501'
    path = LAB/'Encoding_models'/subject/'posthoc_model_predict'/f'accentuated_stim_info_w_pred_resp_{subject}.pkl'
    df = pd.read_pickle(path)
    pairs = [g for _,g in df.groupby(['model_name','unit_id','img_id','level']) if len(g)>1]
    assert len(pairs)==277 and all(len(g)==2 for g in pairs)
    records=[]
    start=time.monotonic()
    for i,g in enumerate(tqdm(pairs,desc='Leap duplicate-image pixel audit')):
        paths=[Path(p) for p in g.filepath]
        imgs=[np.asarray(Image.open(p).convert('RGB'),dtype=float) for p in paths]
        assert imgs[0].shape==imgs[1].shape
        r=float(np.corrcoef(imgs[0].ravel(),imgs[1].ravel())[0,1])
        records.append(dict(stim_a=paths[0].name,stim_b=paths[1].name,pixel_r=r))
        if i==4:
            print(f'5-pair pilot {time.monotonic()-start:.2f}s; projected {len(pairs)/5*(time.monotonic()-start):.1f}s',flush=True)
    (root()/'cache/leap_variant_similarity.json').write_text(json.dumps(records,indent=2))
    (root()/'cache/leap_variant_audit_provenance.json').write_text(json.dumps(dict(
        method='flattened RGB Pearson, native image size, reconstructed audit',pairs=len(records),
        merges=sum(r['pixel_r']>=.9 for r in records),seconds=time.monotonic()-start),indent=2))
    print(f'{len(records)} pairs, {sum(r["pixel_r"]>=.9 for r in records)} pixel r >= .90',flush=True)


def kappa_from_alpha(s, alpha, n):
    lam = alpha / n
    # Positive stable branch; n > retained rank for these experiments.
    assert lam > 0 and n > len(s) and np.all(s >= 0)
    f = lambda k: k * (1 - np.sum(s/(s+k))/n) - lam
    hi = max(float(s.max()), lam) * 2
    while f(hi) < 0:
        hi *= 2
    k = brentq(f, lam, hi, xtol=max(1e-12, lam*1e-12))
    return lam, k, f(k)


def extract_readouts(manifest):
    import torch
    from pnc.preproc.encoding import _flag_to_bool
    records = []
    cache_dir = root() / 'ridge_metadata'
    cache_dir.mkdir(exist_ok=True)
    for (subject, model), group in tqdm(list(manifest.groupby(['subject', 'model'])), desc='Restore alpha and verify readouts'):
        cache = cache_dir / f'{subject}_{model}.csv'
        if cache.exists():
            records.extend(pd.read_csv(cache).to_dict('records'))
            continue
        start = time.monotonic()
        path = LAB / 'Encoding_models' / subject / 'model_outputs_pca4all' / f'{subject}_{model}_sweep_regressors_layers_fitmodels_RidgeCV.pth'
        fits = torch.load(path, weights_only=False, map_location='cpu')
        rows = []
        for row in group.itertuples():
            fit = fits[((row.layer, 'pca750'), 'RidgeCV')]
            with open(row.cache_path, 'rb') as f:
                obj = pickle.load(f)
            w = np.asarray(obj['readout_vec']).ravel().astype(float)
            bias = float(np.asarray(obj['readout_bias']).ravel()[0])
            coef = np.asarray(fit.coef_[row.unit], float)
            rel = np.linalg.norm(w-coef)/max(np.linalg.norm(coef), 1e-30)
            assert rel < 1e-5, (subject, model, row.unit, rel)
            assert np.isclose(bias, fit.intercept_[row.unit], rtol=1e-5, atol=1e-5)
            train = _flag_to_bool(obj['df']['is_train'])
            assert train.sum() == row.n_train
            pred = np.asarray(obj['PCA_resp']) @ w + bias
            discrepancy = float(np.max(np.abs(pred-np.asarray(obj['target_unit_resp']).ravel())))
            assert discrepancy < 1e-3, (subject, model, discrepancy)
            rows.append(dict(subject=subject, model=model, unit=row.unit,
                alpha=float(np.asarray(fit.alpha_)[row.unit]), coef_rel_error=rel,
                prediction_max_error=discrepancy, n_train=int(train.sum()), n_test=int((~train).sum())))
        pd.DataFrame(rows).to_csv(cache, index=False)
        records.extend(rows)
        del fits
        print(f'{subject} {model}: {time.monotonic()-start:.1f}s', flush=True)
    return pd.DataFrame(records)


def outcome_table(manifest, L):
    rows, clouds = [], []
    preprocessing_audit=[]
    for monkey in MONKEYS:
        b=L.load_brain(monkey)
        c=b['control']
        for day in np.unique(c['trial_day']):
            sel=c['trial_day']==day
            n_anchor=int(np.sum(sel & (c['trial_kind']=='calibration')))
            preprocessing_audit.append(dict(monkey=monkey,day=day,n_trials=int(sel.sum()),
                n_anchor=n_anchor,allday_fallback=n_anchor<5,n_dropped_monkey=b['n_trials_dropped']))
    pd.DataFrame(preprocessing_audit).to_csv(TABLE/'preprocessing_audit.csv',index=False)
    for row in tqdm(list(manifest.itertuples()), desc='Biological outcomes'):
        monkey = row.subject.split('_')[0]
        x, y = L.encoding_cloud(monkey, row.unit, row.model, split='test')
        assert len(x) == row.n_total-row.n_train, (monkey, len(x))
        assert np.isfinite(x).all() and np.isfinite(y).all()
        xc, yc = L.control_cloud(monkey, row.unit, row.model)
        r, slope, r2 = L._measures(xc, yc)
        P = L.load_predictions(monkey)
        mi = list(P['pred_models']).index(row.model)
        ui = list(P['target_units']).index(row.unit)
        resp = L._acc_response(monkey, row.unit)
        sel = (P['gen_model']==row.model) & (P['gen_unit']==row.unit)
        local = []
        for name, pred, seed in zip(P['stim'][sel], P['pred'][sel,mi,ui], P['gen_seed'][sel]):
            if name in resp:
                local.append(dict(subject=row.subject, monkey=monkey, unit=row.unit, model=row.model,
                    stimulus=name, seed=int(seed), predicted_raw=float(pred),
                    predicted=max(float(pred),L.firing_floor(monkey,row.unit)), measured=resp[name]))
        c = pd.DataFrame(local)
        if len(c):
            centered = c[['predicted','measured']] - c.groupby('seed')[['predicted','measured']].transform('mean')
            denom = np.square(centered.predicted).sum()
            within = (centered.predicted*centered.measured).sum()/denom if denom>0 else np.nan
        else:
            within = np.nan
        clouds.extend(local)
        rows.append(dict(subject=row.subject, monkey=monkey, unit=row.unit, model=row.model,
            mse_test=float(np.mean((x-y)**2)), variance_test=float(np.var(y)),
            n_test_observed=len(x), n_control=len(xc), control_slope=slope, control_r=r, control_R2=r2,
            control_mse=float(np.mean((xc-yc)**2)), control_Eacc_observed_variance=float(np.mean((xc-yc)**2)/np.var(y)),
            control_slope_within_seed=within,
            leap_merge_audited=(root()/'cache/leap_variant_similarity.json').exists() if monkey=='leap' else True))
    pd.DataFrame(clouds).to_csv(TABLE/'control_clouds.csv.gz', index=False)
    return pd.DataFrame(rows)


def analyze():
    _, L, _ = upstream_modules()
    TABLE.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(REPO/'tables/nonlinear_control/scale_manifest/site_model_manifest.csv')
    alpha = extract_readouts(manifest)
    outcomes = outcome_table(manifest, L)
    alpha.to_csv(TABLE/'ridge_metadata.csv', index=False)
    outcomes.to_csv(TABLE/'biological_outcomes.csv', index=False)
    base = manifest.merge(alpha, on=['subject','model','unit','n_train'], validate='one_to_one').merge(
        outcomes, on=['subject','model','unit'], validate='one_to_one')
    assert len(base)==250 and base['leap_merge_audited'].all()
    geometry_root = root().parent/'mass_v1/geometry'
    qc=pd.read_csv(REPO/'tables/nonlinear_control/scale_manifest/production_deep_qc.csv').set_index('geometry_id')
    spectra = {}
    rows = []
    for row in tqdm(list(base.itertuples()), desc='Integrate saved spectral masses'):
        if row.geometry_id not in spectra:
            with np.load(geometry_root/row.geometry_id/'summary.npz') as z:
                spectra[row.geometry_id] = dict(z)
        z = spectra[row.geometry_id]
        s = z['spectrum']
        lam, k, residual = kappa_from_alpha(s, row.alpha, row.n_train)
        weight = s/(s+k)**2
        common = dict(subject=row.subject, monkey=row.monkey, unit=row.unit, model=row.model,
            geometry_id=row.geometry_id, alpha=row.alpha, n_train=row.n_train, lambda_normalized=lam,
            kappa=k, kappa_residual=residual, df2=float(np.sum((s/(s+k))**2)),
            mse_test=row.mse_test, control_slope=row.control_slope,
            control_slope_within_seed=row.control_slope_within_seed,control_R2=row.control_R2,
            control_mse=row.control_mse,n_control=row.n_control, leap_merge_audited=row.leap_merge_audited,
            fd_error=float(qc.loc[row.geometry_id,'fd_error']))
        masses = [('exact',0.,z['exact_by_seed']),('local_mc',0.,z['local_mean_by_seed'])]
        for method in ['smooth','neighborhood','stein','variance','step']:
            masses.extend((method,float(tau),z[f'{method}_mean_by_seed'][:,j,:]) for j,tau in enumerate(z['tau']))
        for method,tau,q in masses:
            for seed_idx, qs in enumerate(q):
                trace = float(qs@weight)
                rows.append(dict(common,method=method,tau=tau,tau_255=tau*255,seed_index=seed_idx+1,
                    trace=trace,raw_energy=float(qs.sum()),V=row.mse_test/row.n_train*trace))
    long = pd.DataFrame(rows)
    assert len(long)==250*10*22
    assert np.isfinite(long[['mse_test','kappa','trace','V']].to_numpy()).all()
    long.to_csv(TABLE/'variance_predictors_by_seed.csv.gz', index=False)
    keys = ['subject','monkey','unit','model','geometry_id','method','tau','tau_255']
    means = long.groupby(keys,as_index=False).agg({c:'mean' for c in long.columns if c not in keys+['seed_index']})
    means.to_csv(TABLE/'variance_predictors.csv', index=False)
    write_provenance(base,means,long)
    plots_and_stats(means)


def write_provenance(base,means,long):
    assert len(base)==250 and base.leap_merge_audited.all()
    assert len(means)==250*22 and len(long)==250*22*10
    assert np.isfinite(long[['mse_test','kappa','trace','V']].to_numpy()).all()
    np.testing.assert_allclose(long.V,long.mse_test*long.trace/long.n_train,rtol=1e-10)
    assert means.groupby(['method','tau_255']).size().eq(250).all()
    sensitivity=[]
    for (method,tau),tab in means.groupby(['method','tau_255']):
        for subset,t in [('all',tab),('without_robust_pair',tab[~tab.model.isin(['clipag_vitb32','resnet50_robust'])]),
                         ('without_three0',tab[tab.monkey!='three0'])]:
            t=t[(t.V>0)&np.isfinite(t.control_slope)].copy()
            t['logV']=np.log10(t.V)
            centered=t[['logV','control_slope']]-t.groupby(['subject','unit'])[['logV','control_slope']].transform('mean')
            sensitivity.append(dict(method=method,tau_255=tau,subset=subset,n=len(t),
                spearman=spearmanr(t.logV,t.control_slope).statistic,
                site_demeaned_spearman=spearmanr(centered.logV,centered.control_slope).statistic))
    pd.DataFrame(sensitivity).to_csv(TABLE/'robust_pair_and_normalization_sensitivity.csv',index=False)
    (TABLE/'analysis_provenance.json').write_text(json.dumps(dict(upstream_sha=SHA,
        spectrum='training PCA spectrum, ddof=1; population-spectrum proxy',
        regularization='lambda=alpha/n; kappa*(1-sum(s/(s+kappa))/n)=lambda',
        estimand='V=MSE_test*T/n, variance contribution proxy, not full predicted slope',
        test_noise='MSE against trial-averaged measured responses; no extra sigma squared added',
        leap_audit='reconstructed flattened native RGB correlation, threshold 0.90',
        fd_qc='CLIP RN50 smooth/neighborhood/local_mc provisional; marginal directional FD QC',
        signed_masses='No clipping; nonpositive aggregate predictors omitted only from log associations/CV',
        counts=dict(site_models=len(base),predictor_rows=len(means),seed_rows=len(long)),
        negative_site_traces=int((means.trace<0).sum()),
        missing_control_slopes=int(base.control_slope.isna().sum())),indent=2))
    print('Output validation passed: 250 outcomes, 5,500 site-method rows, 55,000 seed-method rows.',flush=True)


def plots_and_stats(df):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from analyze_scale_energy import MODEL_ORDER, MODEL_LABELS, MODEL_COLORS
    out = REPO/'figures/nonlinear_control/biological_validation'
    out.mkdir(parents=True,exist_ok=True)
    stats, cv = [], []
    for (method,tau), tab in df.groupby(['method','tau_255']):
        sm = tab.groupby(['monkey','model'])[['mse_test','trace','V','control_slope']].mean().reset_index()
        fig, axs = plt.subplots(1,4,figsize=(22,5))
        for ax,col,title in zip(axs,['mse_test','trace','V','control_slope'],['Held-out MSE',r'$T(\kappa)$',r'$V=\mathrm{MSE}\,T/n$','Biological control slope']):
            for j,m in enumerate(MODEL_ORDER):
                vals = sm.loc[sm.model==m,col].to_numpy()
                ax.bar(j,vals.mean(),color=MODEL_COLORS[m])
                ax.errorbar(j,vals.mean(),yerr=2*vals.std(ddof=1)/np.sqrt(len(vals)),color='k',capsize=3)
                ax.scatter(j+np.linspace(-.12,.12,len(vals)),vals,s=18,facecolor='white',edgecolor='.3',zorder=3)
            if col!='control_slope':
                if (sm[col]>0).all(): ax.set_yscale('log')
                else: ax.set_yscale('symlog',linthresh=1e-2)
            ax.set_xticks(range(10),[MODEL_LABELS[m] for m in MODEL_ORDER],rotation=45,ha='right')
            ax.set_title(title)
            ax.spines[['top','right']].set_visible(False)
        fig.suptitle(f'{method}, noise SD × 255 = {tau:g}; actual ridge alpha → κ (training-spectrum proxy)\nBars: mean of 5 subject means; dots: subjects; error bars: ±2 subject-level SE')
        fig.tight_layout()
        fig.savefig(out/f'biology_bars_{method}_{tau:g}.png',dpi=150)
        plt.close(fig)
        if method=='exact' or np.isclose(tau,2):
            fig,axs=plt.subplots(1,3,figsize=(15,4.6))
            for ax,col in zip(axs,['mse_test','trace','V']):
                for model in MODEL_ORDER:
                    t=tab[(tab.model==model)&(tab[col]>0)]
                    ax.scatter(t[col],t.control_slope,s=22,alpha=.55,color=MODEL_COLORS[model],label=MODEL_LABELS[model].replace('\n',' '))
                ax.set_xscale('log'); ax.set_xlabel(col); ax.set_ylabel('Biological control slope')
                ax.axhline(1,color='.7',linestyle=':'); ax.spines[['top','right']].set_visible(False)
            axs[-1].legend(fontsize=7,loc='best')
            fig.suptitle(f'{method}, noise SD × 255 = {tau:g}; each dot = one site × model; descriptive association')
            fig.tight_layout(); fig.savefig(out/f'biology_scatter_{method}_{tau:g}.png',dpi=150); plt.close(fig)
        for exclusion in ['all','without_leap','without_clip_rn50']:
            t=tab.copy()
            if exclusion=='without_leap': t=t[t.monkey!='leap']
            if exclusion=='without_clip_rn50': t=t[t.model!='resnet50_clip']
            for outcome in ['control_slope','control_slope_within_seed','control_R2']:
                for predictor in ['mse_test','trace','V']:
                    valid = (t[predictor]>0)&np.isfinite(t[outcome])
                    v=t.loc[valid].copy(); v['log_predictor']=np.log10(v[predictor])
                    rho=spearmanr(v.log_predictor,v[outcome]).statistic
                    centered=v[['log_predictor',outcome]]-v.groupby(['subject','unit'])[['log_predictor',outcome]].transform('mean')
                    rho_within=spearmanr(centered.log_predictor,centered[outcome]).statistic
                    stats.append(dict(method=method,tau_255=tau,subset=exclusion,outcome=outcome,predictor=predictor,n=len(v),
                        spearman=rho,site_demeaned_spearman=rho_within))
        # Fixed small linear baselines; all transforms fit only on each training fold.
        # No noise-level or predictor selection based on these held-out results.
        for group in ['monkey','model']:
            # Compare every candidate on exactly the same held-out observations.
            # Stein U-statistics can be negative; keep those in the source tables,
            # but log-linear comparisons require a shared positive subset.
            valid=np.isfinite(tab.control_slope)&(tab[['mse_test','trace','V']]>0).all(axis=1)
            for columns in [[],['mse_test'],['trace'],['V'],['mse_test','trace']]:
                t=tab[valid].copy()
                for heldout in sorted(t[group].unique()):
                    train=t[group]!=heldout
                    if not columns:
                        pred=np.repeat(t.loc[train,'control_slope'].mean(),(~train).sum())
                    else:
                        x=np.log10(t[columns].to_numpy())
                        mu=x[train].mean(0); sd=x[train].std(0); sd[sd==0]=1
                        x=np.column_stack([np.ones(len(x)),(x-mu)/sd])
                        coef=np.linalg.lstsq(x[train],t.loc[train,'control_slope'],rcond=None)[0]
                        pred=x[~train]@coef
                    for (_,row),estimate in zip(t.loc[~train].iterrows(),pred):
                        cv.append(dict(method=method,tau_255=tau,group=group,heldout=heldout,
                            predictor='+'.join(columns) or 'intercept',subject=row.subject,unit=row.unit,model=row.model,
                            observed=row.control_slope,predicted=float(estimate)))
    pd.DataFrame(stats).to_csv(TABLE/'descriptive_associations.csv',index=False)
    cv=pd.DataFrame(cv)
    cv['squared_error']=(cv.observed-cv.predicted)**2
    cv.to_csv(TABLE/'heldout_slope_predictions.csv.gz',index=False)
    cv.groupby(['method','tau_255','group','predictor'],as_index=False).agg(
        heldout_mse=('squared_error','mean'),n=('squared_error','size')).to_csv(TABLE/'heldout_slope_comparison.csv',index=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['setup','preprocess','leap-audit','readouts','analyze','validate','plots'])
    parser.add_argument('--upstream',default='/tmp/parametric-neural-control-jacob')
    parser.add_argument('--monkeys',nargs='+',default=MONKEYS,choices=MONKEYS)
    args=parser.parse_args()
    if args.stage=='setup': setup(args.upstream)
    elif args.stage=='preprocess': preprocess(args.monkeys)
    elif args.stage=='leap-audit': leap_audit()
    elif args.stage=='readouts':
        upstream_modules()
        manifest=pd.read_csv(REPO/'tables/nonlinear_control/scale_manifest/site_model_manifest.csv')
        extract_readouts(manifest)
    elif args.stage=='validate':
        write_provenance(pd.read_csv(TABLE/'biological_outcomes.csv'),pd.read_csv(TABLE/'variance_predictors.csv'),
                         pd.read_csv(TABLE/'variance_predictors_by_seed.csv.gz'))
    elif args.stage=='plots': plots_and_stats(pd.read_csv(TABLE/'variance_predictors.csv'))
    else: analyze()

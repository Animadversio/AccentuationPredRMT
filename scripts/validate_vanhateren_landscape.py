"""Validate grid integrity and agreement with the previous natural-image run."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.vanhateren_landscape import evaluate, METRICS


def load(path):
    with np.load(path) as z: return {k:z[k] for k in z.files}


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--previous',type=Path,required=True); parser.add_argument('--input',type=Path,required=True)
    args=parser.parse_args(); c=load(args.data/'coordinates.npz')
    shape=(len(c['sigma']),len(c['alpha']))
    mc=load(args.data/'mc_summary.npz'); count=int(mc['trials']); assert count==100
    report=dict(grid_shape=list(shape),trials=count)
    for filename in ['theory.npz','gaussian_evaluation.npz','gaussian_selection.npz','mc_summary.npz']:
        for key,v in load(args.data/filename).items():
            if np.ndim(v)==2:
                assert v.shape==shape,(filename,key,v.shape)
                assert np.isfinite(v).all(),(filename,key,'nonfinite')
    for name in METRICS:
        raw=load(args.data/f'mc_raw_{name}.npz')['values']
        assert raw.shape==(count,*shape)
        np.testing.assert_allclose(raw.mean(0),mc[name+'_mean'],rtol=1e-12,atol=1e-12)
        if name.startswith('R2'): assert raw.max()<=1+1e-10
        if name.startswith('E_'): assert raw.min()>=-1e-12
    # Compare first trial at the exact old noise coordinates (no interpolation).
    old=pd.read_csv(args.previous/'trial_0000.csv')
    coef=load(args.data/'coefficients_0000.npz')
    errors={}
    for r in old.itertuples():
        j=int(np.argmin(abs(np.log(c['alpha']/r.alpha))))
        np.testing.assert_allclose(c['alpha'][j],r.alpha,rtol=1e-10)
        vals=evaluate({k:v[j] for k,v in coef.items()},r.sigma,float(c['S']),float(np.sum(c['beta_proj']**2)))
        for metric in ['E_gen','E_acc','slope_acc']:
            expected=getattr(r,metric)
            np.testing.assert_allclose(vals[metric],expected,rtol=2e-5,atol=1e-8)
            errors[metric]=max(errors.get(metric,0.),abs(vals[metric]-expected))
    report['previous_trial_0_max_absolute_error']=errors
    hasher=hashlib.sha256()
    with args.input.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): hasher.update(block)
    report['input_sha256']=hasher.hexdigest(); report['input_path']=str(args.input)
    report['all_grid_arrays_finite']=True; report['raw_summary_consistent']=True
    (args.data/'integrity_validation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__': main()

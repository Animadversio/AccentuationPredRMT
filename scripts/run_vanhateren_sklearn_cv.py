"""Actual sklearn RidgeCV, one independent LOOCV alpha per noise target.

Multi-output alpha_per_target=True reuses the same design decomposition;
it is not joint alpha selection across noise levels. Paired seeds match MC.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
import argparse
import inspect
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import RidgeCV
from threadpoolctl import threadpool_limits
import torch
from tqdm.auto import tqdm
from scripts.validate_ffhq_disk_teacher import create_disk_teacher


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--trials', type=int, default=100)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = root/'notebooks/outputs/pixel_ridge/vanhateren_selection_estimation'
    config = json.loads((output/'config.json').read_text())
    selected = pd.read_csv(output/'comparison.csv')
    selected = selected[selected.policy == 'fixed'].sort_values('sigma')
    sigmas = selected.sigma.to_numpy()
    S = selected.S.iloc[0]
    alphas = np.array(config['alphas'])
    cache = args.work/'sklearn_ridgecv'
    cache.mkdir(exist_ok=True)
    metadata = dict(sklearn_version=sklearn.__version__, n=1000, d=10000,
                    alphas=alphas.tolist(), sigma=sigmas.tolist(),
                    cv=None, scoring=None, fit_intercept=True,
                    alpha_per_target=True, gcv_mode='eigen',
                    seed=20260918, evaluation='centered population; intercept excluded as in original',
                    signal_variance=float(S))
    meta = cache/'config.json'
    if meta.exists():
        assert json.loads(meta.read_text()) == metadata, 'Incompatible cache'
    else:
        meta.write_text(json.dumps(metadata, indent=2))
    torch.set_num_threads(4)
    start = time.perf_counter()
    data = np.load(args.work/'vanhateren_log_gray_100px_22000_uint8.npy', mmap_mode='r')
    train = np.array(data[:2000], dtype=np.float64)/255
    pop = torch.as_tensor(np.array(data[2000:], dtype=np.float64), device='cuda')/255
    pop -= pop.mean(0, keepdim=True)
    beta = np.asarray(create_disk_teacher(100, .3), dtype=np.float64)
    bt = torch.as_tensor(beta, device='cuda')
    target = pop@bt
    print(f'Input staging {time.perf_counter()-start:.2f}s; sklearn {sklearn.__version__}', flush=True)
    store_kw = ('store_cv_results' if 'store_cv_results' in inspect.signature(RidgeCV).parameters
                else 'store_cv_values')
    frames = []
    for trial in tqdm(range(args.trials), desc='sklearn RidgeCV trials'):
        path = cache/f'trial_{trial:04d}.csv'
        if path.exists():
            frames.append(pd.read_csv(path))
            continue
        start = time.perf_counter()
        g = torch.Generator(device='cuda').manual_seed(20260918+trial)
        ids = torch.randperm(2000, device='cuda', generator=g)[:1000].cpu().numpy()
        eps = torch.randn(1000, device='cuda', generator=g, dtype=torch.float64)
        eps = (eps-eps.mean()).cpu().numpy()
        X = train[ids].copy()
        X -= X.mean(0, keepdims=True)
        y = (X@beta)[:, None]+eps[:, None]*sigmas[None, :]
        with threadpool_limits(limits=4):
            model = RidgeCV(alphas=alphas, cv=None, scoring=None,
                            fit_intercept=True, gcv_mode='eigen', alpha_per_target=True,
                            **{store_kw: True}).fit(X, y)
        fit_seconds = time.perf_counter()-start
        values = getattr(model, 'cv_results_', None)
        if values is None:
            values = model.cv_values_
        cv_path = values.mean(axis=0)
        np.savez_compressed(cache/f'cv_path_{trial:04d}.npz', alphas=alphas,
                            sigma=sigmas, mse=cv_path)
        # Check batched per-target alpha selection against direct single-target calls.
        if trial == 0:
            with threadpool_limits(limits=4):
                for j in [0, len(sigmas)//2, len(sigmas)-1]:
                    check = RidgeCV(alphas=alphas, cv=None, scoring=None,
                                    fit_intercept=True, gcv_mode='eigen').fit(X, y[:, j])
                    np.testing.assert_allclose(model.alpha_[j], check.alpha_, rtol=1e-12)
                    np.testing.assert_allclose(model.coef_[j], check.coef_, rtol=1e-6, atol=1e-7)
            print('Validated 3 targets against separate sklearn RidgeCV fits.', flush=True)
        w = torch.as_tensor(model.coef_.T.copy(), device='cuda')
        prediction = pop@w
        Eg = (prediction-target[:, None]).square().mean(0)/S
        R = (bt@w)/w.square().sum(0)
        frame = pd.DataFrame(dict(trial=trial, sigma=sigmas, ratio=selected.ratio.to_numpy(),
            alpha=model.alpha_, lam=model.alpha_/1000, cv_mse=-model.best_score_,
            E_gen=Eg.cpu().numpy(), E_acc=(1-R).square().cpu().numpy(),
            R2_gen=(1-Eg).cpu().numpy(), R2_acc=(1-(1/R-1).square()).cpu().numpy(),
            slope_gen=((target@prediction)/prediction.square().sum(0)).cpu().numpy(),
            slope_acc=R.cpu().numpy(), intercept=model.intercept_))
        frame['alpha_at_boundary'] = np.isin(model.alpha_, [alphas[0], alphas[-1]])
        frame.to_csv(path, index=False)
        frames.append(frame)
        elapsed = time.perf_counter()-start
        print(f'Trial {trial+1}/{args.trials}: fit {fit_seconds:.2f}s total {elapsed:.2f}s; '
              f'remaining ETA {fit_seconds*(args.trials-trial-1):.1f}s', flush=True)
        if trial == 0 and fit_seconds*100 > 1800:
            raise RuntimeError('Projected full run >30min: profile before proceeding')
    raw = pd.concat(frames, ignore_index=True)
    raw.to_csv(cache/'raw.csv', index=False)
    rows = []
    for sigma, q in raw.groupby('sigma', sort=True):
        row = dict(sigma=sigma, ratio=q.ratio.iloc[0], trials=len(q),
                   boundary_fraction=q.alpha_at_boundary.mean())
        for metric in ['E_gen', 'E_acc', 'R2_gen', 'R2_acc', 'slope_gen', 'slope_acc', 'lam']:
            for suffix, value in [('_mc', q[metric].mean()), ('_mc_se', q[metric].sem()),
                                  ('_mc_median', q[metric].median()),
                                  ('_mc_q10', q[metric].quantile(.1)),
                                  ('_mc_q90', q[metric].quantile(.9))]:
                row[metric+suffix] = value
        rows.append(row)
    pd.DataFrame(rows).to_csv(cache/'summary.csv', index=False)
    if args.trials == 100:
        pd.DataFrame(rows).to_csv(output/'sklearn_cv_summary.csv', index=False)
    print(f'COMPLETE: {cache}', flush=True)


if __name__ == '__main__':
    main()

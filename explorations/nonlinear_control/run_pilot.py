"""Training-spectrum audit and exact PC-VJP pilot using the experimental model.

Default: spectrum audit only. --gradients runs a bounded timing pilot.
--all-pcs evaluates all retained PCs after reporting the measured ETA.
"""
import argparse
import json
import os
from pathlib import Path
import pickle
import sys
import time

import numpy as np
import torch
from geometry import pc_gradient_power, batched_pc_gradient_power, trace_contributions

PROJECT = Path('/n/home12/binxuwang/Github/Closed-loop-visual-insilico')
ROOT = Path('/n/holylabs/LABS/alvarez_lab/Lab/VVS_Accentuation/Encoding_models/red_20250428-20250430')
DEFAULT_CACHE = ROOT / 'posthoc_model_predict_PCA_popul_unit/posthoc_prediction_NSDencimg_PCA_pop_unit_red_20250428-20250430_unit2_resnet50_robust.pkl'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cache', type=Path, default=DEFAULT_CACHE)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--gradients', action='store_true')
    ap.add_argument('--all-pcs', action='store_true')
    ap.add_argument('--images', type=int, default=1)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--evaluation', choices=['training', 'seeds'], default='training')
    ap.add_argument('--seed-root', type=Path, default=Path('/n/holylabs/LABS/alvarez_lab/Lab/VVS_Accentuation/Stimuli'))
    ap.add_argument('--chunk-size', type=int, default=1)
    ap.add_argument('--benchmark-chunks', action='store_true')
    ap.add_argument('--disable-xformers', action='store_true', help='Use native DINO attention for batched VJP compatibility')
    args = ap.parse_args()
    if args.images < 1 or args.chunk_size < 1 or (args.all_pcs and not args.gradients):
        ap.error('Require images >= 1 and --gradients with --all-pcs')
    torch.set_num_threads(args.threads)
    # Keep Jacobian comparisons out of TF32's reduced-mantissa arithmetic.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.output.mkdir(parents=True, exist_ok=True)
    # Trusted project cache only: pickle contains tensors, config and dataframe.
    with args.cache.open('rb') as f:
        data = pickle.load(f)
    config, df = data['config'], data['df']
    z = np.asarray(data['PCA_resp'].cpu(), dtype=np.float64)
    mask = df['is_train'].to_numpy(dtype=bool)
    train = z[mask]
    centered = train - train.mean(0)
    cov = centered.T @ centered / (len(train) - 1)
    spectrum = np.diag(cov).copy()
    offdiag = np.linalg.norm(cov - np.diag(spectrum)) / np.linalg.norm(cov)
    # The trace weights must use eigenvalues in the SAME basis as the VJPs.
    # Fail rather than silently substitute eigenvalues for old-coordinate variances.
    audit = dict(cache=str(args.cache), model=config['model_name'],
                 layer=config['layer_name'], n_train=len(train), n_total=len(z),
                 pcs=z.shape[1], covariance_ddof=1, offdiag_relative_frobenius=float(offdiag),
                 training_mean_norm=float(np.linalg.norm(train.mean(0))),
                 spectrum_source='training rows of cached unwhitened PCA scores',
                 population_status='empirical plug-in, not population truth')
    audit['evaluation'] = args.evaluation
    np.savetxt(args.output / 'training_spectrum.csv',
               np.column_stack((np.arange(1, len(spectrum)+1), spectrum)),
               delimiter=',', header='pc,training_variance', comments='')
    (args.output / 'audit.json').write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2), flush=True)
    if not args.gradients:
        return
    if offdiag > 1e-3:
        raise ValueError('Cached training coordinates are not diagonal; reconcile PCA provenance first')
    if args.all_pcs:
        store = os.environ.get('STORE_DIR')
        if not store:
            raise ValueError('STORE_DIR must be configured for full runs')
        if not args.output.resolve().is_relative_to((Path(store)/'Projects/AccentuationPredRMT').resolve()):
            raise ValueError('Full-run output must be under STORE_DIR/Projects/AccentuationPredRMT')
    sys.path.insert(0, str(PROJECT))
    if args.disable_xformers:
        os.environ['XFORMERS_DISABLED'] = '1'
    from core.model_load_utils import load_model_transform
    from circuit_toolkit.layer_hook_utils import featureFetcher
    from PIL import Image
    from tqdm.auto import tqdm
    model, transform = load_model_transform(config['model_name'], device=args.device)
    model.eval().to(args.device).requires_grad_(False)
    fetcher = featureFetcher(model, input_size=(3, 224, 224), device=args.device, print_module=False)
    fetcher.record(config['layer_name'], ingraph=True, store_device=args.device)
    pca = torch.jit.load(config['xtransform_path'], map_location=args.device).eval()
    norms = [t for t in transform.transforms if type(t).__name__ == 'Normalize']
    if len(norms) != 1 or transform.transforms[-1] is not norms[0]:
        raise ValueError('Explicit preprocessing derivative adapter needed for this model')
    std = norms[0].std
    pilot_pcs = np.unique(np.linspace(0, len(spectrum)-1, 32 if args.benchmark_chunks else 8, dtype=int))
    # Fixed RNG; evaluation distribution is training images for this first pilot.
    candidates = np.flatnonzero(mask)
    rows = np.random.default_rng(20260919).permutation(candidates)[:args.images]
    paths = [df.iloc[row]['image_fps'] for row in rows]
    if args.evaluation == 'seeds':
        paths = [args.seed_root / p for p in config['seed_image_paths']][:args.images]
        rows = []
        for path in paths:
            matches = np.flatnonzero(df['image_fps'].map(lambda p: Path(p).stem == path.stem))
            if len(matches) != 1 or not path.is_file():
                raise ValueError(f'Seed missing or ambiguous cache match: {path}')
            rows.append(int(matches[0]))
    (args.output/'evaluation_images.json').write_text(json.dumps([
        dict(row=int(row), path=str(path), is_train=bool(mask[row]))
        for row, path in zip(rows, paths)], indent=2))
    records = []
    chunk_size = args.chunk_size
    for i, row in enumerate(tqdm(rows, desc='Images')):
        # Verify the cache's own image/preprocessing independently: seed PNGs
        # can differ slightly from training-cache JPEGs of the same stimulus.
        if i == 0:
            with torch.no_grad():
                reference = transform(Image.open(df.iloc[row]['image_fps']).convert('RGB')).unsqueeze(0).to(args.device)
                model(reference)
                check = pca(fetcher[config['layer_name']]).cpu().numpy()[0]
                # Standardize by feature SD: relative error to an individual
                # score is ill-conditioned when that image's score is near zero.
                error = check-z[row]
                scaled = error/np.sqrt(spectrum)
                verification = dict(relative_l2=float(np.linalg.norm(error)/np.linalg.norm(z[row])),
                                    max_sd_error=float(np.abs(scaled).max()),
                                    rms_sd_error=float(np.sqrt(np.mean(scaled**2))),
                                    max_absolute_error=float(np.abs(error).max()),
                                    tf32=False, xformers_disabled=args.disable_xformers)
                (args.output/'score_verification.json').write_text(json.dumps(verification, indent=2))
                print(f'Cache reproduction: {verification}', flush=True)
                if verification['relative_l2'] > 1e-3 or verification['max_sd_error'] > 5e-3:
                    raise ValueError('Model/PCA/cache disagreement exceeds 0.1% norm or 0.5% feature SD')
                del reference
        x = transform(Image.open(paths[i]).convert('RGB')).unsqueeze(0).to(args.device)
        x.requires_grad_(True)
        start = time.perf_counter()
        model(x)
        scores = pca(fetcher[config['layer_name']])
        if args.device.startswith('cuda'):
            torch.cuda.synchronize()
        forward_seconds = time.perf_counter()-start
        if i == 0:
            pc_gradient_power(scores, x, pilot_pcs[:2], std)  # warm up backward kernels
        start = time.perf_counter()
        pilot = pc_gradient_power(scores, x, pilot_pcs, std)
        sec_per_pc = (time.perf_counter()-start)/len(pilot_pcs)
        if i == 0 and args.benchmark_chunks:
            benchmarks = [dict(chunk_size=1, seconds_per_pc=sec_per_pc)]
            for chunk in [4, 16, 32]:
                batched_pc_gradient_power(scores, x, pilot_pcs[:chunk], std)  # warm up
                if args.device.startswith('cuda'):
                    torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                batched = np.concatenate([batched_pc_gradient_power(scores, x, pilot_pcs[j:j+chunk], std)
                                          for j in range(0, len(pilot_pcs), chunk)])
                seconds = (time.perf_counter()-start)/len(pilot_pcs)
                np.testing.assert_allclose(batched, pilot, rtol=2e-4, atol=1e-7)
                benchmarks.append(dict(chunk_size=chunk, seconds_per_pc=seconds,
                                       max_relative_difference=float(np.max(np.abs(batched-pilot)/np.maximum(pilot, 1e-20))),
                                       peak_memory_bytes=torch.cuda.max_memory_allocated() if args.device.startswith('cuda') else None))
            winner = min(benchmarks, key=lambda b: b['seconds_per_pc'])
            chunk_size, sec_per_pc = winner['chunk_size'], winner['seconds_per_pc']
            (args.output/'chunk_benchmark.json').write_text(json.dumps(benchmarks, indent=2))
            print(f'Batched VJP benchmark: {benchmarks}; selected chunk={chunk_size}', flush=True)
        if i == 0:
            timing = dict(forward_seconds=forward_seconds, seconds_per_pc=sec_per_pc,
                          projected_seconds=len(rows)*(forward_seconds+len(spectrum)*sec_per_pc),
                          requested_images=len(rows), requested_pcs=len(spectrum),
                          coordinate='RGB [0,1], after spatial preprocessing', channel_std=list(std))
            timing['chunk_size'] = chunk_size
            (args.output/'timing.json').write_text(json.dumps(timing, indent=2))
            print(json.dumps(timing), flush=True)
        values = dict(zip(pilot_pcs, pilot))
        if args.all_pcs:
            remaining = [k for k in range(len(spectrum)) if k not in values]
            for j in tqdm(range(0, len(remaining), chunk_size), desc=f'PC VJP blocks image {i}'):
                block = remaining[j:j+chunk_size]
                method = pc_gradient_power if chunk_size == 1 else batched_pc_gradient_power
                values.update(zip(block, method(scores, x, block, std)))
        indices = sorted(values)
        records.extend((int(row), int(k+1), float(values[k])) for k in indices)
        np.savetxt(args.output/'gradient_power.csv', records, delimiter=',',
                   header='dataframe_row,pc,rgb_gradient_power', comments='')
        del scores, x
    powers = np.array([r[2] for r in records]).reshape(len(rows), -1)
    idx = np.asarray(indices)
    mean = powers.mean(0)
    se = powers.std(0, ddof=1)/np.sqrt(len(rows)) if len(rows)>1 else np.full_like(mean, np.nan)
    np.savetxt(args.output/'pc_summary.csv', np.column_stack((idx+1, spectrum[idx], mean, se, mean/spectrum[idx])),
               delimiter=',', header='pc,variance,mean_power,power_se,power_over_variance', comments='')
    kappas = np.geomspace(spectrum.max()*1e-6, spectrum.max()*10, 100)
    contributions = trace_contributions(spectrum[idx], mean, kappas)
    np.savetxt(args.output/'trace.csv', np.column_stack((kappas, contributions.sum(1))),
               delimiter=',', header='kappa,trace_over_evaluated_pcs_only', comments='')
    print(f'Saved {len(rows)} images x {len(indices)} PCs to {args.output}', flush=True)


if __name__ == '__main__':
    main()

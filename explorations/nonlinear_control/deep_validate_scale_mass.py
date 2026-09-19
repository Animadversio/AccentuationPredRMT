"""Parallel deep validation of every production seed archive and checksum."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = (
    'smooth', 'neighborhood', 'stein', 'odd', 'step', 'even',
    'variance', 'drift',
)


def file_hash(path, block_bytes=8 << 20):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        while block := handle.read(block_bytes):
            digest.update(block)
    return digest.hexdigest()


def validate_geometry(record, root):
    directory = Path(root) / 'geometry' / record['geometry_id']
    config = json.loads((directory / 'config.json').read_text())
    done = json.loads((directory / 'DONE.json').read_text())
    directions = int(config['directions'])
    levels = len(config['levels'])
    images = int(config['images'])
    pcs = int(record['n_pcs'])
    required = {'tau', 'spectrum', 'exact', 'local', 'outside_rgb_fraction', *METRICS}
    max_variance_identity_error = 0.
    max_step_identity_error = 0.
    max_variance_identity_relative = 0.
    max_step_identity_relative = 0.
    for index in range(1, images + 1):
        path = directory / f'seed_{index:02d}.npz'
        if file_hash(path) != done['checksums'][path.name]:
            raise ValueError(f'checksum mismatch: {path}')
        with np.load(path) as values:
            if not required.issubset(values.files):
                raise ValueError(f'missing arrays: {path}')
            if values['exact'].shape != (pcs,) or values['local'].shape != (directions, pcs):
                raise ValueError(f'local/exact shape: {path}')
            if values['tau'].shape != (levels,) or values['spectrum'].shape != (pcs,):
                raise ValueError(f'spectrum/tau shape: {path}')
            for key in required:
                if not np.isfinite(values[key]).all():
                    raise ValueError(f'nonfinite {key}: {path}')
            for metric in METRICS:
                if values[metric].shape != (levels, directions, pcs):
                    raise ValueError(f'{metric} shape: {path}')
            if np.any(values['exact'] < 0) or np.any(values['local'] < 0):
                raise ValueError(f'negative exact/local power: {path}')
            variance_residual = values['variance'] + values['drift'] - values['step']
            step_residual = values['odd'] + values['even'] - values['step']
            max_variance_identity_error = max(
                max_variance_identity_error, float(np.max(np.abs(variance_residual)))
            )
            max_step_identity_error = max(
                max_step_identity_error, float(np.max(np.abs(step_residual)))
            )
            variance_scale = (
                np.abs(values['variance']) + np.abs(values['drift']) + np.abs(values['step'])
            )
            step_scale = np.abs(values['odd']) + np.abs(values['even']) + np.abs(values['step'])
            variance_relative = float(np.max(
                np.abs(variance_residual) / np.maximum(variance_scale, 1e-30)
            ))
            step_relative = float(np.max(
                np.abs(step_residual) / np.maximum(step_scale, 1e-30)
            ))
            max_variance_identity_relative = max(max_variance_identity_relative, variance_relative)
            max_step_identity_relative = max(max_step_identity_relative, step_relative)
            if variance_relative > 2e-6:
                raise ValueError(f'variance+drift identity: {path}')
            if step_relative > 2e-6:
                raise ValueError(f'odd+even identity: {path}')
    summary = directory / 'summary.npz'
    if file_hash(summary) != done['checksums']['summary.npz']:
        raise ValueError(f'checksum mismatch: {summary}')
    with np.load(summary) as values:
        if values['exact_by_seed'].shape != (images, pcs):
            raise ValueError(f'exact summary shape: {summary}')
        if values['smooth_mean_by_seed'].shape != (images, levels, pcs):
            raise ValueError(f'smooth summary shape: {summary}')
    candidates = config.get('finite_difference_candidates', [])
    fd_error = config.get('finite_difference_relative_error')
    if fd_error is None and candidates:
        fd_error = min(item['relative_directional_error'] for item in candidates)
    if fd_error is None:
        raise ValueError(f'missing finite-difference audit: {directory}')
    quality = 'pass' if fd_error <= .20 else ('marginal' if fd_error <= .50 else 'poor')
    alias_hashes = {item['sha256'] for item in config['pca_alias_audit']}
    if alias_hashes != {config['pca_buffer_sha256']}:
        raise ValueError(f'PCA alias audit mismatch: {directory}')
    return dict(
        geometry_id=record['geometry_id'], model=record['model'], layer=record['layer'],
        fd_error=float(fd_error), fd_quality=quality,
        cache_relative_l2=float(config['score_verification']['relative_l2']),
        cache_p99_sd_error=float(config['score_verification'].get('p99_sd_error', np.nan)),
        cache_max_sd_error=float(config['score_verification']['max_sd_error']),
        alias_count=int(record['alias_count']), seed_archives=images,
        max_variance_identity_error=max_variance_identity_error,
        max_step_identity_error=max_step_identity_error,
        max_variance_identity_relative=max_variance_identity_relative,
        max_step_identity_relative=max_step_identity_relative,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    manifest = pd.read_csv(args.manifest)
    records = manifest.to_dict('records')
    results, errors = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(validate_geometry, record, args.output): record for record in records}
        for future in as_completed(futures):
            record = futures[future]
            try:
                results.append(future.result())
            except Exception as error:
                errors.append(dict(geometry_id=record['geometry_id'], error=repr(error)))
                print(f'FAIL {record["geometry_id"]}: {error}', flush=True)
            else:
                print(f'PASS {record["geometry_id"]}', flush=True)
    frame = pd.DataFrame(results).sort_values(['model', 'layer'])
    frame.to_csv(args.output / 'deep_qc.csv', index=False)
    report = dict(
        expected=len(records), passed=len(results), failed=len(errors),
        seed_archives=int(frame.seed_archives.sum()) if len(frame) else 0,
        fd_quality=frame.fd_quality.value_counts().to_dict() if len(frame) else {},
        max_cache_relative_l2=float(frame.cache_relative_l2.max()) if len(frame) else None,
        max_cache_p99_sd_error=float(frame.cache_p99_sd_error.max()) if len(frame) else None,
        max_cache_sd_error=float(frame.cache_max_sd_error.max()) if len(frame) else None,
        max_variance_identity_error=float(frame.max_variance_identity_error.max()) if len(frame) else None,
        max_step_identity_error=float(frame.max_step_identity_error.max()) if len(frame) else None,
        max_variance_identity_relative=float(frame.max_variance_identity_relative.max()) if len(frame) else None,
        max_step_identity_relative=float(frame.max_step_identity_relative.max()) if len(frame) else None,
        errors=errors,
    )
    (args.output / 'deep_validation.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

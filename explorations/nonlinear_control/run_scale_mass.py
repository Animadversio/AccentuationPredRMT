"""Resumable model-grouped production runner for nonlinear control spectra."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import time

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from forward_noise import (
    covariance_pseudovalues,
    inner_statistics,
    step_statistics,
)
from geometry import batched_pc_gradient_power


PROJECT = Path('/n/home12/binxuwang/Github/Closed-loop-visual-insilico')
SEED_ROOT = Path('/n/holylabs/LABS/alvarez_lab/Lab/VVS_Accentuation/Stimuli')
METRICS = (
    'smooth', 'neighborhood', 'stein', 'odd', 'step', 'even',
    'variance', 'drift',
)


def json_hash(value):
    payload = json.dumps(value, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(payload).hexdigest()


def file_hash(path, block_bytes=8 << 20):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        while block := handle.read(block_bytes):
            digest.update(block)
    return digest.hexdigest()


def pca_buffer_hash(path):
    """Hash numerical PCA state, ignoring TorchScript archive metadata."""
    module = torch.jit.load(str(path), map_location='cpu').eval()
    digest = hashlib.sha256()
    shapes = {}
    for name, tensor in sorted(module.state_dict().items()):
        array = tensor.detach().contiguous().numpy()
        shapes[name] = list(array.shape)
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(memoryview(array))
    del module
    gc.collect()
    return digest.hexdigest(), shapes


def audit_pca_aliases(geometry_id, site_manifest):
    rows = site_manifest[site_manifest.geometry_id == geometry_id]
    records = []
    for path in rows.xtransform_path:
        digest, shapes = pca_buffer_hash(path)
        records.append(dict(path=path, sha256=digest, shapes=shapes))
    hashes = {record['sha256'] for record in records}
    if len(hashes) != 1:
        raise ValueError(f'PCA buffer mismatch among aliases for {geometry_id}')
    return records


def preprocessing_adapter(transform, device):
    normalizers = [
        op for op in getattr(transform, 'transforms', [])
        if type(op).__name__ == 'Normalize'
    ]
    if not normalizers:
        mean, std = (0., 0., 0.), (1., 1., 1.)
    elif len(normalizers) == 1 and transform.transforms[-1] is normalizers[0]:
        mean, std = normalizers[0].mean, normalizers[0].std
    else:
        raise ValueError(f'Unsupported preprocessing derivative: {transform}')
    mean_tensor = torch.as_tensor(mean, device=device).view(1, 3, 1, 1)
    std_tensor = torch.as_tensor(std, device=device).view(1, 3, 1, 1)
    mean_values = [float(value) for value in mean_tensor.flatten().cpu()]
    std_values = [float(value) for value in std_tensor.flatten().cpu()]
    return mean_tensor, std_tensor, mean_values, std_values


def stein_pseudovalues_from_delta(delta, directions_gram, tau):
    """Jackknife pseudovalues for ||E[delta(z) z/tau]||^2."""
    delta = np.asarray(delta, dtype=np.float64)
    gram = np.asarray(directions_gram, dtype=np.float64).copy()
    np.fill_diagonal(gram, 0.)
    count = len(delta)
    if count < 3:
        raise ValueError('Need at least three independent perturbations')
    cross = delta * (gram @ delta) / tau**2
    total = cross.sum(0)
    estimate = total / (count * (count - 1))
    leave_one_out = (total[None, :] - 2 * cross) / ((count - 1) * (count - 2))
    return count * estimate[None, :] - (count - 1) * leave_one_out


def stein_pseudovalues(plus, baseline, directions_gram, tau):
    """One-sided Stein estimate using f(x+tau*z)-f(x)."""
    delta = np.asarray(plus, dtype=np.float64) - np.asarray(baseline, dtype=np.float64)
    return stein_pseudovalues_from_delta(delta, directions_gram, tau)


def antithetic_stein_pseudovalues(plus, minus, directions_gram, tau):
    """Antithetic Stein estimate using [f(x+tau*z)-f(x-tau*z)]/2."""
    delta = (np.asarray(plus, dtype=np.float64)-np.asarray(minus, dtype=np.float64))/2
    return stein_pseudovalues_from_delta(delta, directions_gram, tau)


def validate_seed_archive(path, directions, levels, pcs):
    required = {'tau', 'spectrum', 'exact', 'local', 'outside_rgb_fraction', *METRICS}
    try:
        with np.load(path) as values:
            if not required.issubset(values.files):
                return False
            if values['exact'].shape != (pcs,) or values['local'].shape != (directions, pcs):
                return False
            if values['tau'].shape != (levels,) or values['spectrum'].shape != (pcs,):
                return False
            for metric in METRICS:
                if values[metric].shape != (levels, directions, pcs):
                    return False
            return all(np.isfinite(values[key]).all() for key in required)
    except (OSError, ValueError, EOFError):
        return False


def atomic_npz(path, arrays):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp.npz')
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def write_summary(output, seed_paths, levels, directions, pcs):
    exact = []
    local_mean = []
    metric_mean = {metric: [] for metric in METRICS}
    metric_mc_se = {metric: [] for metric in METRICS}
    for path in seed_paths:
        with np.load(path) as values:
            exact.append(values['exact'])
            local_mean.append(values['local'].mean(0))
            for metric in METRICS:
                metric_mean[metric].append(values[metric].mean(1))
                metric_mc_se[metric].append(
                    values[metric].std(1, ddof=1) / np.sqrt(directions)
                )
            spectrum = values['spectrum']
            tau = values['tau']
    arrays = dict(
        spectrum=spectrum, tau=tau, exact_by_seed=np.asarray(exact),
        local_mean_by_seed=np.asarray(local_mean),
    )
    for metric in METRICS:
        arrays[f'{metric}_mean_by_seed'] = np.asarray(metric_mean[metric])
        arrays[f'{metric}_mc_se_by_seed'] = np.asarray(metric_mc_se[metric])
    summary_path = output / 'summary.npz'
    atomic_npz(summary_path, arrays)
    with np.load(summary_path) as check:
        if check['exact_by_seed'].shape != (len(seed_paths), pcs):
            raise ValueError(f'Invalid summary {summary_path}')
        if check['smooth_mean_by_seed'].shape != (len(seed_paths), levels, pcs):
            raise ValueError(f'Invalid summary {summary_path}')
    return summary_path


def run_geometry(row, site_manifest, model, transform, args, model_output):
    from PIL import Image
    from circuit_toolkit.layer_hook_utils import featureFetcher

    output = model_output.parent / row.geometry_id
    output.mkdir(parents=True, exist_ok=True)
    seed_paths = [output / f'seed_{index:02d}.npz' for index in range(1, args.images + 1)]
    run_parameters = dict(
        model=row.model, layer=row.layer, directions=args.directions,
        centers=args.centers, images=args.images, levels=args.levels,
        rng_seed=args.seed, pca_score_hash=row.score_hash,
        split_hash=row.split_hash, seed_hash=row.seed_hash,
        coordinate='post-resize RGB [0,1]; unbounded Gaussian noise, no clipping',
        tf32=False, xformers_disabled=True,
    )
    signature = json_hash(run_parameters)
    done_path = output / 'DONE.json'
    if done_path.exists():
        done = json.loads(done_path.read_text())
        if done.get('run_signature') == signature and all(
            validate_seed_archive(path, args.directions, len(args.levels), int(row.n_pcs))
            for path in seed_paths
        ):
            print(f'SKIP complete {row.geometry_id}', flush=True)
            return
        raise ValueError(f'Existing DONE record is incompatible or corrupt: {done_path}')

    started_geometry = time.perf_counter()
    alias_audit = audit_pca_aliases(row.geometry_id, site_manifest)
    pca_hash = alias_audit[0]['sha256']
    with open(row.representative_cache, 'rb') as handle:
        cached = pickle.load(handle)
    config, frame = cached['config'], cached['df']
    responses = np.asarray(cached['PCA_resp'].detach().cpu(), dtype=np.float64)
    train = responses[frame.is_train.to_numpy(dtype=bool)]
    spectrum = train.var(0, ddof=1)
    if len(spectrum) != int(row.n_pcs) or np.any(spectrum < 0) or not np.isfinite(spectrum).all():
        raise ValueError(f'Invalid spectrum for {row.geometry_id}')

    fetcher = featureFetcher(model, input_size=(3, 224, 224), device='cuda', print_module=False)
    fetcher.record(row.layer, ingraph=True, store_device='cuda')
    pca = torch.jit.load(row.representative_xtransform, map_location='cuda').eval()
    mean, std, mean_list, std_list = preprocessing_adapter(transform, 'cuda')
    image_paths = [SEED_ROOT / relative for relative in config['seed_image_paths'][:args.images]]
    inputs = [transform(Image.open(path).convert('RGB')).unsqueeze(0).cuda() for path in image_paths]
    dataframe_rows = []
    stimulus_names = frame.stimulus_name.to_numpy()
    for image_path in image_paths:
        matches = np.flatnonzero(stimulus_names == image_path.name)
        if len(matches) != 1:
            raise ValueError(f'Could not uniquely match seed image in cache: {image_path}')
        dataframe_rows.append(int(matches[0]))
    runtime_batch = [args.batch]

    @torch.no_grad()
    def predict(batch):
        while True:
            outputs = []
            try:
                for begin in range(0, len(batch), runtime_batch[0]):
                    model(batch[begin:begin + runtime_batch[0]])
                    outputs.append(pca(fetcher[row.layer]).detach())
                return torch.cat(outputs).double().cpu().numpy()
            except torch.cuda.OutOfMemoryError:
                del outputs
                if runtime_batch[0] == 1:
                    raise
                runtime_batch[0] = max(1, runtime_batch[0] // 2)
                torch.cuda.empty_cache()
                print(f'{row.geometry_id}: reducing forward batch to {runtime_batch[0]}', flush=True)

    baseline0 = predict(inputs[0])[0]
    cache_error = baseline0 - responses[dataframe_rows[0]]
    scaled_cache_error = np.abs(cache_error) / np.sqrt(np.maximum(spectrum, 1e-30))
    score_check = dict(
        dataframe_row=dataframe_rows[0],
        relative_l2=float(np.linalg.norm(cache_error) / max(np.linalg.norm(responses[dataframe_rows[0]]), 1e-30)),
        p99_sd_error=float(np.quantile(scaled_cache_error, .99)),
        max_sd_error=float(np.max(scaled_cache_error)),
    )
    if score_check['relative_l2'] > 5e-3 or score_check['max_sd_error'] > 1e-1:
        raise ValueError(f'Cache mismatch for {row.geometry_id}: {score_check}')

    config_path = output / 'config.json'
    selected_h = None
    if config_path.exists():
        previous = json.loads(config_path.read_text())
        if previous.get('run_signature') != signature:
            raise ValueError(f'Run signature mismatch in {config_path}')
        selected_h = previous.get('selected_h')

    metadata = dict(
        **run_parameters, run_signature=signature,
        geometry_id=row.geometry_id, representative_cache=row.representative_cache,
        representative_xtransform=row.representative_xtransform,
        alias_count=int(row.alias_count), pca_buffer_sha256=pca_hash,
        pca_alias_audit=alias_audit, preprocessing=repr(transform),
        channel_mean=mean_list, channel_std=std_list, score_verification=score_check,
        selected_h=selected_h,
    )

    for image_index, (image_path, x, seed_path) in enumerate(
        tqdm(list(zip(image_paths, inputs, seed_paths)), desc=row.geometry_id, leave=False)
    ):
        if seed_path.exists() and validate_seed_archive(
            seed_path, args.directions, len(args.levels), len(spectrum)
        ):
            print(f'SKIP valid {seed_path}', flush=True)
            continue
        if seed_path.exists():
            quarantine = seed_path.with_suffix(f'.invalid.{int(time.time())}.npz')
            seed_path.replace(quarantine)
            print(f'Quarantined invalid archive as {quarantine}', flush=True)

        seed_started = time.perf_counter()
        generator = torch.Generator(device='cuda').manual_seed(args.seed + image_index * 10007)
        directions = torch.randn(
            (args.directions, 3, x.shape[-2], x.shape[-1]), generator=generator, device='cuda'
        )
        directions_gram = (
            directions.flatten(1) @ directions.flatten(1).T
        ).double().cpu().numpy()

        inp = x.detach().requires_grad_(True)
        model(inp)
        scores = pca(fetcher[row.layer])
        exact = np.concatenate([
            batched_pc_gradient_power(scores, inp, range(begin, min(begin + args.vjp_block, len(spectrum))), std)
            for begin in range(0, len(spectrum), args.vjp_block)
        ])
        if selected_h is None:
            probe_pcs = np.unique(np.linspace(0, len(spectrum) - 1, 8, dtype=int))
            derivatives = []
            for pc in probe_pcs:
                gradient, = torch.autograd.grad(scores[0, pc], inp, retain_graph=True)
                derivatives.append(
                    ((gradient / std) * directions[:16]).flatten(1).sum(1).detach().cpu().numpy()
                )
            derivatives = np.stack(derivatives, axis=1)
        del scores, inp
        torch.cuda.empty_cache()

        if selected_h is None:
            candidates = []
            for candidate in [1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 2e-3, 3e-3]:
                plus = predict(x + candidate * directions[:16] / std)
                minus = predict(x - candidate * directions[:16] / std)
                relative_error = float(
                    np.linalg.norm((plus - minus)[:, probe_pcs] / (2 * candidate) - derivatives)
                    / max(np.linalg.norm(derivatives), 1e-30)
                )
                candidates.append(dict(h=candidate, relative_directional_error=relative_error))
            best = min(candidates, key=lambda item: item['relative_directional_error'])
            if best['relative_directional_error'] > 1.:
                raise ValueError(f'No reliable finite-difference step for {row.geometry_id}: {candidates}')
            selected_h = best['h']
            metadata['selected_h'] = selected_h
            metadata['finite_difference_candidates'] = candidates
            metadata['finite_difference_relative_error'] = best['relative_directional_error']
            if best['relative_directional_error'] <= .20:
                metadata['finite_difference_quality'] = 'pass'
            elif best['relative_directional_error'] <= .50:
                metadata['finite_difference_quality'] = 'marginal'
            else:
                metadata['finite_difference_quality'] = 'poor'
            config_path.write_text(json.dumps(metadata, indent=2))
            print(f'{row.geometry_id}: selected h={selected_h:g}, error={best["relative_directional_error"]:.3g}', flush=True)
        elif not config_path.exists():
            metadata['selected_h'] = selected_h
            config_path.write_text(json.dumps(metadata, indent=2))

        baseline = predict(x)[0]
        local = (
            (predict(x + selected_h * directions / std) - predict(x - selected_h * directions / std))
            / (2 * selected_h)
        ) ** 2
        rgb = x * std + mean
        level_values = {metric: [] for metric in METRICS}
        outside_fractions = []
        for tau in tqdm(args.levels, desc=f'seed {image_index + 1} tau', leave=False):
            plus = predict(x + tau * directions / std)
            minus = predict(x - tau * directions / std)
            odd, step, even = step_statistics(plus, minus, baseline, tau)
            variance, drift = covariance_pseudovalues(plus, minus, baseline, tau)
            stein = stein_pseudovalues(plus, baseline, directions_gram, tau)
            noise_generator = torch.Generator(device='cuda').manual_seed(
                args.seed + 500003 + image_index * 10007
            )
            inner = []
            outside, pixels = 0, 0
            for begin in range(0, args.directions, args.direction_block):
                end = min(begin + args.direction_block, args.directions)
                noise = torch.randn(
                    (end - begin, args.centers, 3, x.shape[-2], x.shape[-1]),
                    generator=noise_generator, device='cuda',
                )
                center = x[:, None] + tau * noise / std[:, None]
                direction = selected_h * directions[begin:end, None] / std[:, None]
                pp = predict((center + direction).flatten(0, 1)).reshape(end - begin, args.centers, -1)
                mm = predict((center - direction).flatten(0, 1)).reshape(end - begin, args.centers, -1)
                inner.append((pp - mm) / (2 * selected_h))
                perturbed_rgb = rgb[:, None] + tau * noise
                outside += int(((perturbed_rgb < 0) | (perturbed_rgb > 1)).sum().item())
                pixels += perturbed_rgb.numel()
            smooth, neighborhood = inner_statistics(np.concatenate(inner))
            for metric, value in (
                ('smooth', smooth), ('neighborhood', neighborhood), ('stein', stein),
                ('odd', odd), ('step', step), ('even', even),
                ('variance', variance), ('drift', drift),
            ):
                level_values[metric].append(np.asarray(value, dtype=np.float32))
            outside_fractions.append(outside / pixels)

        arrays = dict(
            tau=np.asarray(args.levels, dtype=np.float64), spectrum=spectrum.astype(np.float64),
            exact=exact.astype(np.float64), local=local.astype(np.float32),
            outside_rgb_fraction=np.asarray(outside_fractions),
            selected_h=np.asarray(selected_h), dataframe_row=np.asarray(dataframe_rows[image_index]),
        )
        arrays.update({metric: np.stack(level_values[metric]) for metric in METRICS})
        atomic_npz(seed_path, arrays)
        if not validate_seed_archive(seed_path, args.directions, len(args.levels), len(spectrum)):
            raise ValueError(f'Wrote invalid archive {seed_path}')
        timing = dict(
            seed=image_index + 1, image=str(image_path),
            elapsed_seconds=time.perf_counter() - seed_started,
            forward_batch=runtime_batch[0], peak_gpu_bytes=torch.cuda.max_memory_allocated(),
            archive_bytes=seed_path.stat().st_size,
        )
        (output / f'seed_{image_index + 1:02d}_timing.json').write_text(json.dumps(timing, indent=2))
        print(f'{row.geometry_id} seed {image_index + 1}: {timing["elapsed_seconds"]:.1f}s -> {seed_path}', flush=True)
        del directions
        torch.cuda.empty_cache()

    if not all(validate_seed_archive(path, args.directions, len(args.levels), len(spectrum)) for path in seed_paths):
        raise ValueError(f'Geometry incomplete after loop: {row.geometry_id}')
    summary_path = write_summary(output, seed_paths, len(args.levels), args.directions, len(spectrum))
    checksums = {path.name: file_hash(path) for path in [*seed_paths, summary_path, config_path]}
    done = dict(
        geometry_id=row.geometry_id, model=row.model, layer=row.layer,
        run_signature=signature, seeds=args.images, pcs=len(spectrum),
        elapsed_seconds=time.perf_counter() - started_geometry,
        checksums=checksums, completed_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
    )
    done_path.write_text(json.dumps(done, indent=2))
    print(f'COMPLETE {row.geometry_id} in {done["elapsed_seconds"]:.1f}s', flush=True)
    fetcher.cleanup()
    del cached, responses, train, pca, fetcher, inputs
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--geometry-manifest', type=Path, required=True)
    parser.add_argument('--site-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--directions', type=int, default=64)
    parser.add_argument('--centers', type=int, default=8)
    parser.add_argument('--images', type=int, default=10)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--direction-block', type=int, default=4)
    parser.add_argument('--vjp-block', type=int, default=32)
    parser.add_argument('--levels', type=float, nargs='+', default=[.5/255, 2/255, 8/255, 16/255])
    parser.add_argument('--seed', type=int, default=20260920)
    parser.add_argument('--max-geometries', type=int)
    parser.add_argument('--geometry-index', type=int)
    args = parser.parse_args()
    if args.centers < 2 or args.directions < 16 or not 1 <= args.images <= 10:
        parser.error('Need centers>=2, directions>=16 and images in 1..10')
    store = os.environ.get('STORE_DIR')
    if not store:
        raise ValueError('STORE_DIR is not configured')
    allowed = Path(store) / 'Projects/AccentuationPredRMT'
    if not args.output.resolve().is_relative_to(allowed.resolve()):
        raise ValueError(f'Output must be below {allowed}')
    args.output.mkdir(parents=True, exist_ok=True)
    geometry_manifest = pd.read_csv(args.geometry_manifest)
    site_manifest = pd.read_csv(args.site_manifest)
    rows = geometry_manifest[geometry_manifest.model == args.model]
    if args.geometry_index is not None:
        if not 0 <= args.geometry_index < len(rows):
            parser.error(f'geometry-index must be in 0..{len(rows) - 1}')
        rows = rows.iloc[[args.geometry_index]]
    elif args.max_geometries:
        rows = rows.iloc[:args.max_geometries]
    if rows.empty:
        raise ValueError(f'No geometries found for model {args.model}')

    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    os.environ['XFORMERS_DISABLED'] = '1'
    sys.path.insert(0, str(PROJECT))
    from core.model_load_utils import load_model_transform

    model_started = time.perf_counter()
    model, transform = load_model_transform(args.model, device='cuda')
    model.eval().cuda().requires_grad_(False)
    model_output = args.output / 'geometry' / f'.{args.model}.anchor'
    model_output.parent.mkdir(parents=True, exist_ok=True)
    print(f'Loaded {args.model} in {time.perf_counter() - model_started:.1f}s; {len(rows)} geometries', flush=True)
    for _, row in rows.iterrows():
        run_geometry(row, site_manifest, model, transform, args, model_output)
    print(f'MODEL COMPLETE {args.model}', flush=True)


if __name__ == '__main__':
    main()

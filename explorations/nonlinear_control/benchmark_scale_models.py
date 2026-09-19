"""Benchmark one representative geometry per model for scale-up scheduling."""
import argparse
import gc
import json
import os
from pathlib import Path
import pickle
import sys
import time

import numpy as np
import pandas as pd
import torch

PROJECT = Path('/n/home12/binxuwang/Github/Closed-loop-visual-insilico')
SEED_ROOT = Path('/n/holylabs/LABS/alvarez_lab/Lab/VVS_Accentuation/Stimuli')


def sync():
    torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--models', nargs='*')
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    manifest = pd.read_csv(args.manifest)
    models = args.models or sorted(manifest.model.unique())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(PROJECT))
    os.environ['XFORMERS_DISABLED'] = '1'
    from PIL import Image
    from core.model_load_utils import load_model_transform
    from circuit_toolkit.layer_hook_utils import featureFetcher
    from geometry import batched_pc_gradient_power
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.output.exists():
        results = json.loads(args.output.read_text())
        if not isinstance(results, list):
            raise ValueError(f'Expected a list in {args.output}')
    else:
        results = []
    completed_models = {row['model'] for row in results}
    for name in models:
        if name in completed_models:
            print(f'Skipping completed model {name}', flush=True)
            continue
        row = manifest[manifest.model == name].iloc[0]
        print(f'BENCHMARK {name} {row.layer}', flush=True)
        with open(row.representative_cache, 'rb') as file:
            cached = pickle.load(file)
        config = cached['config']
        started = time.perf_counter()
        model, transform = load_model_transform(name, device='cuda')
        model.eval().cuda().requires_grad_(False)
        sync(); model_load_seconds = time.perf_counter()-started
        fetcher = featureFetcher(model, input_size=(3,224,224), device='cuda', print_module=False)
        fetcher.record(row.layer, ingraph=True, store_device='cuda')
        started = time.perf_counter()
        pca = torch.jit.load(row.representative_xtransform, map_location='cuda').eval()
        sync(); pca_load_seconds = time.perf_counter()-started
        image_path = SEED_ROOT/config['seed_image_paths'][0]
        x = transform(Image.open(image_path).convert('RGB')).unsqueeze(0).cuda()
        normalizers = [
            op for op in getattr(transform, 'transforms', [])
            if type(op).__name__ == 'Normalize'
        ]
        if not normalizers:
            # Some backbones (currently RADIO) consume RGB in [0, 1] directly.
            std = (1.0, 1.0, 1.0)
        elif len(normalizers) == 1 and transform.transforms[-1] is normalizers[0]:
            std = normalizers[0].std
        else:
            raise ValueError(f'Unexpected transform for {name}: {transform}')
        forward = None
        last_error = None
        for batch in [32, 16, 8, 4, 2, 1]:
            try:
                xb = x.expand(batch, -1, -1, -1).contiguous()
                with torch.no_grad():
                    model(xb); pca(fetcher[row.layer]); sync()
                    samples = []
                    torch.cuda.reset_peak_memory_stats()
                    for _ in range(args.repeats):
                        tic = time.perf_counter(); model(xb); pca(fetcher[row.layer]); sync()
                        samples.append(time.perf_counter()-tic)
                forward = dict(batch_size=batch, seconds_per_batch=float(np.median(samples)),
                               forward_peak_gpu_bytes=int(torch.cuda.max_memory_allocated()))
                del xb
                break
            except torch.cuda.OutOfMemoryError as error:
                last_error = str(error); torch.cuda.empty_cache()
        if forward is None:
            raise RuntimeError(f'No forward batch fits for {name}: {last_error}')
        xg = x.detach().requires_grad_(True)
        model(xg); scores = pca(fetcher[row.layer])
        vjp = None
        for block in [32, 16, 8, 4]:
            try:
                indices = list(range(block))
                batched_pc_gradient_power(scores, xg, indices, std); sync()
                torch.cuda.reset_peak_memory_stats(); samples=[]
                for _ in range(args.repeats):
                    tic=time.perf_counter(); batched_pc_gradient_power(scores,xg,indices,std);sync()
                    samples.append(time.perf_counter()-tic)
                vjp = dict(vjp_block=block, seconds_per_vjp_block=float(np.median(samples)),
                           vjp_peak_gpu_bytes=int(torch.cuda.max_memory_allocated()))
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
        if vjp is None:
            raise RuntimeError(f'No batched VJP block fits for {name}')
        # Current R=64,L=8, four-tau design executes 148 forward batches of
        # logical size 32 per seed. Correct for a smaller measured batch size.
        logical_images_per_seed = 64*2 + 4*(64*2 + 64*8*2)
        forward_batches_per_geometry = int(np.ceil(logical_images_per_seed*10/forward['batch_size']))
        forward_seconds_per_geometry = forward_batches_per_geometry*forward['seconds_per_batch']
        exact_blocks_per_geometry = int(np.ceil(750/vjp['vjp_block']))*10
        exact_seconds_per_geometry = exact_blocks_per_geometry*vjp['seconds_per_vjp_block']
        result = dict(model=name, representative_geometry=row.geometry_id, layer=row.layer,
                      model_load_seconds=model_load_seconds, pca_load_seconds=pca_load_seconds,
                      projected_forward_seconds_per_geometry=forward_seconds_per_geometry,
                      projected_exact_seconds_per_geometry=exact_seconds_per_geometry,
                      projected_compute_seconds_per_geometry=forward_seconds_per_geometry+exact_seconds_per_geometry,
                      **forward, **vjp)
        results.append(result)
        args.output.write_text(json.dumps(results, indent=2))
        print(json.dumps(result, indent=2), flush=True)
        fetcher.cleanup(); del cached, model, transform, fetcher, pca, x, xg, scores
        gc.collect(); torch.cuda.empty_cache()
    print(f'Saved {args.output}', flush=True)


if __name__ == '__main__':
    main()

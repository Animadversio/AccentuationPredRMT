"""GPU pilot for nested one-sided and antithetic Stein estimators."""
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
from tqdm.auto import tqdm

from run_scale_mass import (
    PROJECT, SEED_ROOT, atomic_npz, json_hash, preprocessing_adapter,
)


def valid_archive(path, levels, directions, pcs):
    try:
        with np.load(path) as values:
            return (
                values['tau'].shape == (levels,)
                and values['direction_gram'].shape == (directions, directions)
                and values['one_sided_delta'].shape == (levels, directions, pcs)
                and values['antithetic_delta'].shape == (levels, directions, pcs)
                and all(np.isfinite(values[key]).all() for key in values.files)
            )
    except (OSError, ValueError, KeyError, EOFError):
        return False


def run_geometry(row, model, transform, args):
    from PIL import Image
    from circuit_toolkit.layer_hook_utils import featureFetcher

    output=args.output/'geometry'/row.geometry_id
    output.mkdir(parents=True,exist_ok=True)
    seed_paths=[output/f'seed_{i:02d}.npz' for i in range(1,args.images+1)]
    parameters=dict(model=row.model,geometry_id=row.geometry_id,layer=row.layer,
        directions=args.directions,images=args.images,levels=args.levels,seed=args.seed,
        prefixes=args.prefixes,coordinate='post-resize RGB [0,1]; unbounded Gaussian noise, no clipping',
        estimator='one-sided and antithetic Stein response deltas',tf32=False)
    signature=json_hash(parameters)
    config_path=output/'config.json'
    if config_path.exists() and json.loads(config_path.read_text()).get('run_signature')!=signature:
        raise ValueError(f'Incompatible pilot configuration at {config_path}')

    with open(row.representative_cache,'rb') as handle:
        cached=pickle.load(handle)
    config,frame=cached['config'],cached['df']
    responses=np.asarray(cached['PCA_resp'].detach().cpu(),dtype=np.float64)
    train=responses[frame.is_train.to_numpy(dtype=bool)]
    spectrum=train.var(0,ddof=1)
    pcs=len(spectrum)
    fetcher=featureFetcher(model,input_size=(3,224,224),device='cuda',print_module=False)
    fetcher.record(row.layer,ingraph=True,store_device='cuda')
    pca=torch.jit.load(row.representative_xtransform,map_location='cuda').eval()
    _,std,mean_values,std_values=preprocessing_adapter(transform,'cuda')
    image_paths=[SEED_ROOT/relative for relative in config['seed_image_paths'][:args.images]]
    inputs=[transform(Image.open(path).convert('RGB')).unsqueeze(0).cuda() for path in image_paths]
    names=frame.stimulus_name.to_numpy();frame_rows=[]
    for path in image_paths:
        matches=np.flatnonzero(names==path.name)
        if len(matches)!=1: raise ValueError(f'Could not match seed image {path}')
        frame_rows.append(int(matches[0]))
    runtime_batch=[args.batch]

    @torch.no_grad()
    def predict(batch):
        while True:
            outputs=[]
            try:
                for begin in range(0,len(batch),runtime_batch[0]):
                    model(batch[begin:begin+runtime_batch[0]])
                    outputs.append(pca(fetcher[row.layer]).detach().cpu())
                return torch.cat(outputs).numpy()
            except torch.cuda.OutOfMemoryError:
                del outputs
                if runtime_batch[0]==1: raise
                runtime_batch[0]=max(1,runtime_batch[0]//2)
                torch.cuda.empty_cache()
                print(f'{row.geometry_id}: reducing batch to {runtime_batch[0]}',flush=True)

    baseline0=predict(inputs[0])[0].astype(np.float64)
    error=baseline0-responses[frame_rows[0]]
    check=dict(relative_l2=float(np.linalg.norm(error)/max(np.linalg.norm(responses[frame_rows[0]]),1e-30)),
        max_sd_error=float(np.max(np.abs(error)/np.sqrt(np.maximum(spectrum,1e-30)))))
    if check['relative_l2']>5e-3 or check['max_sd_error']>1e-1:
        raise ValueError(f'Cache mismatch: {check}')
    if not config_path.exists():
        config_path.write_text(json.dumps(dict(**parameters,run_signature=signature,
            representative_cache=row.representative_cache,
            representative_xtransform=row.representative_xtransform,
            alias_count=int(row.alias_count),preprocessing=repr(transform),
            channel_mean=mean_values,channel_std=std_values,score_verification=check),indent=2))

    for image_index,(x,path) in enumerate(tqdm(list(zip(inputs,seed_paths)),desc=row.geometry_id)):
        if path.exists() and valid_archive(path,len(args.levels),args.directions,pcs):
            print(f'SKIP {path}',flush=True);continue
        started=time.perf_counter()
        generator=torch.Generator(device='cuda').manual_seed(args.seed+image_index*10007)
        directions=torch.randn((args.directions,3,x.shape[-2],x.shape[-1]),
            generator=generator,device='cuda')
        gram=(directions.flatten(1)@directions.flatten(1).T).cpu().numpy().astype(np.float32)
        baseline=predict(x)[0]
        one=[];anti=[]
        for tau in tqdm(args.levels,desc=f'seed {image_index+1} noise',leave=False):
            plus=predict(x+tau*directions/std)
            minus=predict(x-tau*directions/std)
            one.append((plus-baseline).astype(np.float32))
            anti.append(((plus-minus)/2).astype(np.float32))
        atomic_npz(path,dict(tau=np.asarray(args.levels),prefixes=np.asarray(args.prefixes),
            spectrum=spectrum.astype(np.float64),direction_gram=gram,
            one_sided_delta=np.asarray(one),antithetic_delta=np.asarray(anti),
            dataframe_row=np.asarray(frame_rows[image_index]),rng_seed=np.asarray(args.seed+image_index*10007)))
        if not valid_archive(path,len(args.levels),args.directions,pcs):
            raise ValueError(f'Invalid archive {path}')
        timing=dict(seed=image_index+1,elapsed_seconds=time.perf_counter()-started,
            batch=runtime_batch[0],peak_gpu_bytes=torch.cuda.max_memory_allocated(),archive_bytes=path.stat().st_size)
        (output/f'seed_{image_index+1:02d}_timing.json').write_text(json.dumps(timing,indent=2))
        print(f'{row.geometry_id} seed {image_index+1}: {timing["elapsed_seconds"]:.1f}s',flush=True)
        del directions;torch.cuda.empty_cache()

    done=dict(run_signature=signature,geometry_id=row.geometry_id,model=row.model,
        completed_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    (output/'DONE.json').write_text(json.dumps(done,indent=2))
    fetcher.cleanup();del fetcher,pca,inputs,cached,responses,train
    gc.collect();torch.cuda.empty_cache()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',required=True)
    parser.add_argument('--geometry-manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--directions',type=int,default=1024)
    parser.add_argument('--prefixes',type=int,nargs='+',default=[64,128,256,512,1024])
    parser.add_argument('--images',type=int,default=10)
    parser.add_argument('--batch',type=int,default=32)
    parser.add_argument('--levels',type=float,nargs='+',default=[.5/255,2/255,8/255,16/255])
    parser.add_argument('--seed',type=int,default=20260920)
    parser.add_argument('--max-geometries',type=int,default=1)
    parser.add_argument('--geometry-start',type=int,default=0)
    parser.add_argument('--geometry-stride',type=int,default=1)
    args=parser.parse_args()
    if max(args.prefixes)>args.directions or min(args.prefixes)<3:
        parser.error('Prefixes must be between 3 and directions')
    if sorted(set(args.prefixes))!=args.prefixes:
        parser.error('Prefixes must be unique and increasing')
    store=os.environ.get('STORE_DIR')
    if not store: raise ValueError('STORE_DIR is not configured')
    allowed=Path(store)/'Projects/AccentuationPredRMT'
    if not args.output.resolve().is_relative_to(allowed.resolve()):
        raise ValueError(f'Output must be below {allowed}')
    args.output.mkdir(parents=True,exist_ok=True)
    manifest=pd.read_csv(args.geometry_manifest)
    if args.geometry_start<0 or args.geometry_stride<1:
        parser.error('geometry-start must be nonnegative and geometry-stride must be positive')
    rows=(manifest[manifest.model==args.model].sort_values(['alias_count','geometry_id'],ascending=[False,True])
        .iloc[args.geometry_start::args.geometry_stride].head(args.max_geometries))
    if rows.empty: raise ValueError(f'No geometry for {args.model}')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    os.environ['XFORMERS_DISABLED']='1';sys.path.insert(0,str(PROJECT))
    from core.model_load_utils import load_model_transform
    started=time.perf_counter();model,transform=load_model_transform(args.model,device='cuda')
    model.eval().cuda().requires_grad_(False)
    print(f'Loaded {args.model} in {time.perf_counter()-started:.1f}s',flush=True)
    for _,row in rows.iterrows(): run_geometry(row,model,transform,args)


if __name__=='__main__': main()

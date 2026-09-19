"""Four-model forward-only noise geometry, with numerical and timing pilots."""
import argparse
import gc
import json
import os
from pathlib import Path
import pickle
import sys
import time
import numpy as np
import torch
from tqdm.auto import tqdm
from forward_noise import inner_statistics, step_statistics, covariance_pseudovalues
from compare_resnets import kappa_at_df2

PROJECT = Path('/n/home12/binxuwang/Github/Closed-loop-visual-insilico')
CACHE_ROOT = Path('/n/holylabs/LABS/alvarez_lab/Lab/VVS_Accentuation/Encoding_models/red_20250428-20250430/posthoc_model_predict_PCA_popul_unit')
SEED_ROOT = Path('/n/holylabs/LABS/alvarez_lab/Lab/VVS_Accentuation/Stimuli')
MODELS = {'resnet50': 'standard_resnet50_seed10', 'resnet50_robust': 'robust_resnet50_seed10',
          'clipag_vitb32': 'clipag_seed10', 'dinov2_vitb14_reg': 'dinov2_seed10'}


def run_model(name, args, base):
    from PIL import Image
    from core.model_load_utils import load_model_transform
    from circuit_toolkit.layer_hook_utils import featureFetcher
    output = args.output/name
    output.mkdir(parents=True, exist_ok=True)
    config_path = CACHE_ROOT/f'posthoc_prediction_NSDencimg_PCA_pop_unit_red_20250428-20250430_unit2_{name}.pkl'
    with config_path.open('rb') as file:
        cache = pickle.load(file)
    config, df = cache['config'], cache['df']
    exact_path = base/MODELS[name]
    original = np.genfromtxt(exact_path/'pc_summary.csv', names=True, delimiter=',')
    s = original['variance']
    kappa = kappa_at_df2(s, .5*len(s))
    weights = s/(s+kappa)**2
    exact_raw = np.genfromtxt(exact_path/'gradient_power.csv', names=True, delimiter=',')
    manifest = json.loads((exact_path/'evaluation_images.json').read_text())[:args.images]
    print(f'Loading {name}: {config["layer_name"]}', flush=True)
    model, transform = load_model_transform(name, device='cuda')
    model.eval().cuda().requires_grad_(False)
    fetcher = featureFetcher(model, input_size=(3,224,224), device='cuda', print_module=False)
    fetcher.record(config['layer_name'], ingraph=True, store_device='cuda')
    pca = torch.jit.load(config['xtransform_path'], map_location='cuda').eval()
    norm = transform.transforms[-1]
    if type(norm).__name__ != 'Normalize':
        raise ValueError('Need final Normalize for RGB-coordinate perturbations')
    std = torch.tensor(norm.std, device='cuda')[None,:,None,None]
    mean = torch.tensor(norm.mean, device='cuda')[None,:,None,None]

    @torch.no_grad()
    def predict(x):
        outputs = []
        for start in range(0,len(x),args.batch):
            model(x[start:start+args.batch])
            outputs.append(pca(fetcher[config['layer_name']]).detach())
        return torch.cat(outputs).double().cpu().numpy()

    metadata = dict(model=name, layer=config['layer_name'], images=manifest,
                    tau_rgb=args.levels, directions=args.directions, centers=args.centers,
                    seed=args.seed, kappa=kappa, df2=375., original_spectrum_held_fixed=True,
                    coordinate='post-resize RGB [0,1]; unbounded Gaussian noise, no clipping',
                    preprocessing=repr(transform), tf32=False, xformers_disabled=True)
    (output/'config.json').write_text(json.dumps(metadata, indent=2))
    all_summaries=[]
    for image_index, item in enumerate(tqdm(manifest, desc=name)):
        started = time.perf_counter()
        x = transform(Image.open(item['path']).convert('RGB')).unsqueeze(0).cuda()
        rgb = x*std+mean
        baseline = predict(x)[0]
        if image_index == 0:
            reference = transform(Image.open(df.iloc[item['row']]['image_fps']).convert('RGB')).unsqueeze(0).cuda()
            error = predict(reference)[0]-cache['PCA_resp'][item['row']].cpu().numpy()
            check = dict(relative_l2=float(np.linalg.norm(error)/np.linalg.norm(cache['PCA_resp'][item['row']].cpu().numpy())),
                         max_sd_error=float(np.max(np.abs(error)/np.sqrt(s))))
            if check['relative_l2']>1e-3 or check['max_sd_error']>5e-3:
                raise ValueError(f'Cache mismatch: {check}')
            repeated = predict(x.expand(args.batch,-1,-1,-1).contiguous())
            check['batch_repeat_relative_error'] = float(np.linalg.norm(repeated-baseline)/np.linalg.norm(repeated))
            if check['batch_repeat_relative_error']>1e-3:
                raise ValueError(f'Batch changes outputs: {check}')
            (output/'score_verification.json').write_text(json.dumps(check, indent=2))
        generator = torch.Generator(device='cuda').manual_seed(args.seed+image_index*10007)
        v = torch.randn((args.directions,3,224,224), generator=generator, device='cuda')
        if image_index == 0:
            # Same-probe comparison isolates finite-difference error from MC error.
            inp = x.detach().requires_grad_(True)
            model(inp); scores = pca(fetcher[config['layer_name']])
            pcs = np.unique(np.linspace(0,len(s)-1,8,dtype=int))
            deriv=[]
            for pc in pcs:
                grad, = torch.autograd.grad(scores[0,pc],inp,retain_graph=True)
                deriv.append(((grad/std)*v[:16]).flatten(1).sum(1).detach().cpu().numpy())
            deriv=np.stack(deriv,axis=1)
            del grad,scores,inp
            candidates=[]
            for h in [3e-6,1e-5,3e-5,1e-4,3e-4]:
                plus=predict(x+h*v[:16]/std); minus=predict(x-h*v[:16]/std)
                fd=(plus-minus)/(2*h)
                rel=float(np.linalg.norm(fd[:,pcs]-deriv)/np.linalg.norm(deriv))
                candidates.append(dict(h=h,relative_directional_error=rel))
            selected=min(candidates,key=lambda row:row['relative_directional_error'])
            if selected['relative_directional_error']>.15:
                raise ValueError(f'No reliable forward difference step: {candidates}')
            h=selected['h']*args.h_multiplier
            (output/'finite_difference_check.json').write_text(json.dumps(candidates,indent=2))
            print(f'{name}: derivative pilot {candidates}; h={h}',flush=True)
        # At tau=0 every inner sample is identical; only one plus/minus per v.
        t0=time.perf_counter()
        local=((predict(x+h*v/std)-predict(x-h*v/std))/(2*h))**2
        local_seconds=time.perf_counter()-t0
        exact=exact_raw[exact_raw['dataframe_row']==item['row']]['rgb_gradient_power']
        arrays=dict(local=local.astype('float32'), exact=exact, spectrum=s, weights=weights)
        level_records=[]
        for level_index,tau in enumerate(tqdm(args.levels,desc=f'seed {image_index+1} noise',leave=False)):
            plus=predict(x+tau*v/std); minus=predict(x-tau*v/std)
            odd,step,even=step_statistics(plus,minus,baseline,tau)
            variance,drift=covariance_pseudovalues(plus,minus,baseline,tau)
            # Independent centers per v; same z streams reused across tau/models.
            noise_generator=torch.Generator(device='cuda').manual_seed(args.seed+500003+image_index*10007)
            inner=[]; clipped_count=0; pixel_count=0
            nested_start=time.perf_counter()
            for r in range(0,args.directions,args.direction_block):
                end=min(r+args.direction_block,args.directions)
                z=torch.randn((end-r,args.centers,3,224,224),generator=noise_generator,device='cuda')
                center=x[:,None]+tau*z/std[:,None]
                direction=h*v[r:end,None]/std[:,None]
                pp=predict((center+direction).flatten(0,1)).reshape(end-r,args.centers,-1)
                mm=predict((center-direction).flatten(0,1)).reshape(end-r,args.centers,-1)
                inner.append((pp-mm)/(2*h))
                perturbed_rgb=rgb[:,None]+tau*z
                clipped_count+=int(((perturbed_rgb<0)|(perturbed_rgb>1)).sum().item())
                pixel_count+=perturbed_rgb.numel()
                if image_index==0 and level_index==0 and r==0:
                    pilot_sec=time.perf_counter()-nested_start
                    eta=pilot_sec/(end-r)*args.directions*len(args.levels)*len(manifest)
                    timing=dict(pilot_outer_directions=end-r,pilot_seconds=pilot_sec,
                                projected_nested_seconds=eta,local_seconds=local_seconds,
                                peak_gpu_bytes=torch.cuda.max_memory_allocated())
                    (output/'timing.json').write_text(json.dumps(timing,indent=2))
                    print(f'{name}: projected nested MC {eta:.1f}s; log dir {output}',flush=True)
            inner=np.concatenate(inner)
            smooth,neighborhood=inner_statistics(inner)
            for metric,value in [('smooth',smooth),('neighborhood',neighborhood),('odd',odd),('step',step),('even',even),('variance',variance),('drift',drift)]:
                arrays[f'{metric}_{level_index}']=value.astype('float32')
                trace=value@weights
                level_records.append(dict(seed_index=image_index+1,tau=tau,metric=metric,
                                          trace_mean=float(trace.mean()),trace_mc_se=float(trace.std(ddof=1)/np.sqrt(len(trace))),
                                          exact_trace=float(exact@weights),h=h,
                                          outside_rgb_fraction=clipped_count/pixel_count))
            arrays[f'inner_derivatives_{level_index}']=inner.astype('float32')
        arrays['tau']=np.asarray(args.levels)
        arrays['h']=np.asarray(h)
        np.savez_compressed(output/f'seed_{image_index+1:02d}.npz',**arrays)
        trace=local@weights
        level_records.append(dict(seed_index=image_index+1,tau=0.,metric='local_forward',
                                  trace_mean=float(trace.mean()),trace_mc_se=float(trace.std(ddof=1)/np.sqrt(len(trace))),
                                  exact_trace=float(exact@weights),h=h,outside_rgb_fraction=0.))
        all_summaries.extend(level_records)
        import csv
        with (output/'seed_trace_summary.csv').open('w') as file:
            writer=csv.DictWriter(file,fieldnames=list(all_summaries[0]),lineterminator='\n');writer.writeheader();writer.writerows(all_summaries)
        print(f'{name} seed {image_index+1}: {time.perf_counter()-started:.1f}s; saved powers and inner derivatives',flush=True)
    metadata['h']=h
    (output/'config.json').write_text(json.dumps(metadata,indent=2))
    fetcher.cleanup()
    del model,pca,fetcher,cache,v,x
    gc.collect();torch.cuda.empty_cache()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--models',nargs='+',default=list(MODELS),choices=list(MODELS))
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--directions',type=int,default=64)
    ap.add_argument('--centers',type=int,default=8)
    ap.add_argument('--images',type=int,default=10)
    ap.add_argument('--batch',type=int,default=32)
    ap.add_argument('--direction-block',type=int,default=4)
    ap.add_argument('--levels',type=float,nargs='+',default=[.5/255,2/255,8/255,16/255])
    ap.add_argument('--seed',type=int,default=20260920)
    ap.add_argument('--h-multiplier',type=float,default=1.,help='Paired finite-difference step sensitivity audit')
    args=ap.parse_args()
    if args.centers<2 or args.directions<16 or args.h_multiplier<=0 or not 1<=args.images<=10 or any(t<=0 for t in args.levels):
        ap.error('Need centers>=2, directions>=16, images in 1..10, positive noise scales')
    if not os.environ.get('STORE_DIR'):
        raise ValueError('STORE_DIR not set')
    base=Path(os.environ['STORE_DIR'])/'Projects/AccentuationPredRMT/nonlinear_control'
    if not args.output.resolve().is_relative_to(base.resolve()):
        raise ValueError('Bulk output must be under STORE_DIR project nonlinear_control folder')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    os.environ['XFORMERS_DISABLED']='1'
    sys.path.insert(0,str(PROJECT))
    for name in args.models:
        run_model(name,args,base)


if __name__=='__main__':
    main()

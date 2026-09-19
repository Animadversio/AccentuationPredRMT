"""Full-pixel natural-image rerun; archive under STORE_DIR, optional local workdir.

Run on an allocated CUDA node. Fixed and DE-gen settings exactly reuse the
26-noise Van Hateren notebook configuration. Acc selection is a Gaussian
moment-surrogate oracle, independently integrated for all three policies.
"""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/accentuationpredrmt-matplotlib')
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from scripts import pixel_ridge_notebook_utils as u
from scripts.validate_vanhateren_disk_teacher import stage_vanhateren_uint8, VANHATEREN_DIR
from scripts.validate_ffhq_disk_teacher import create_disk_teacher, configure_logger
from scripts.plot_pixel_ridge_estimator_comparison import plot_comparison, add_mc_variability


class GaussianRisk:
    """Gaussian N,D propagated jointly, evaluated without materializing weights.

    Eacc/S = E[(1-N/D)^2]. Shared draws smooth numerical alpha optimization.
    This diagonal moment surrogate is not a proved full distributional DE.
    """
    def __init__(self,s,beta,n,draws,seed):
        self.s,self.beta,self.n=s,beta,n
        self.S=float(s@beta**2)
        self.b=torch.as_tensor(beta,device='cuda',dtype=torch.float64)
        g=torch.Generator(device='cuda').manual_seed(seed)
        self.z=torch.randn((draws,len(s)),generator=g,device='cuda',dtype=torch.float64)
        self.z2=self.z.square()

    def evaluate(self,sigma,alpha):
        m=u.metrics(self.s,self.beta,self.n,sigma,alpha)
        k=m['kappa']
        mean=self.s/(self.s+k)*self.beta
        v=(m['E_gen']*self.S+sigma**2)/self.n*self.s/(self.s+k)**2
        a=torch.as_tensor(mean,device='cuda')
        v=torch.as_tensor(v,device='cuda')
        sd=v.sqrt()
        N=(a*self.b).sum()+self.z@(sd*self.b)
        D=a.square().sum()+2*(self.z@(a*sd))+self.z2@v
        loss=(1-N/D).square()
        return loss.mean().item(),(loss.std()/np.sqrt(len(loss))).item()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--trials',type=int,default=100)
    parser.add_argument('--max-seconds',type=float,default=1800)
    parser.add_argument('--work-dir',type=Path,default=None,
                        help='Authorized temporary scratch; archive separately after storage recovers')
    parser.add_argument('--source-dir',type=Path,default=VANHATEREN_DIR)
    parser.add_argument('--theory-only',action='store_true',help='No Lustre input needed; cache all selected DE/distributional risks')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    archive=Path(os.environ['STORE_DIR'])/'Projects/AccentuationPredRMT/vanhateren_selection_estimation_v1'
    storage=args.work_dir if args.work_dir is not None else archive
    storage.mkdir(parents=True,exist_ok=True)
    log=configure_logger(storage/'progress.log')
    torch.set_num_threads(4)
    if not torch.cuda.is_available(): raise RuntimeError('CUDA node required')
    device=torch.device('cuda')
    log.info('GPU %s; bulk output %s',torch.cuda.get_device_name(),storage)
    base=root/'notebooks/outputs/pixel_ridge/81ff70222c55d53a'
    config=json.loads((base/'config.json').read_text())
    s,beta_pc=u.load_problem(root,'vanhateren')
    n=config['n']; S=float(s@beta_pc**2)
    assert len(s)==10000 and n==1000
    config.update(trials=args.trials,input_distribution='natural Van Hateren image patches',
                  image_seed=20260818,mc_seed=20260918,selection_draws=2048,
                  evaluation_draws=8192,selection_seed=123,evaluation_seed=20260917,
                  signal_variance=S,bulk_directory=str(storage),archive_directory=str(archive),
                  source_directory=str(args.source_dir),fit_intercept=True)
    (storage/'config.json').write_text(json.dumps(config,indent=2))
    summary=pd.read_csv(base/'summary.csv')
    alphas=np.asarray(config['alphas'])
    selected_path=storage/'selected.csv'
    if selected_path.exists():
        selected=pd.read_csv(selected_path)
    else:
        selected=summary[summary.policy.isin(['fixed','DE_gen_CV'])].copy()
        oracle=GaussianRisk(s,beta_pc,n,2048,123)
        rows=[]
        for ratio in tqdm(config['ratios'],desc='Gaussian oracle selection'):
            sigma=float(np.sqrt(ratio*S))
            start=time.perf_counter()
            alpha,value,boundary,path=u.numerical_minimum(
                alphas,lambda a:oracle.evaluate(sigma,a)[0])
            rows.append(dict(policy='Gaussian_acc_oracle',ratio=ratio,sigma=sigma,S=S,n=n,d=len(s),
                             boundary=boundary,selection_risk=value,**u.metrics(s,beta_pc,n,sigma,alpha)))
            pd.DataFrame(dict(alpha=alphas,risk=path)).to_csv(storage/f'oracle_path_{len(rows):02d}.csv',index=False)
            log.info('Selection %d/%d sigma=%g alpha=%g elapsed=%.2fs ETA=%.1fs',
                     len(rows),len(config['ratios']),sigma,alpha,time.perf_counter()-start,
                     (time.perf_counter()-start)*(len(config['ratios'])-len(rows)))
            if len(rows)==1 and (time.perf_counter()-start)*len(config['ratios'])>args.max_seconds:
                raise RuntimeError('Oracle selection pilot exceeds budget; profile before scaling')
        del oracle
        selected=pd.concat([selected,pd.DataFrame(rows)],ignore_index=True)
        selected.to_csv(selected_path,index=False)
    evaluated_path=storage/'evaluated.csv'
    if evaluated_path.exists():
        selected=pd.read_csv(evaluated_path)
    else:
        evaluator=GaussianRisk(s,beta_pc,n,8192,20260917)
        values=[evaluator.evaluate(r.sigma,r.alpha) for r in tqdm(selected.itertuples(),total=len(selected),desc='Independent distribution evaluation')]
        selected['acc_distribution']=[v[0] for v in values]
        selected['acc_distribution_se']=[v[1] for v in values]
        selected.to_csv(evaluated_path,index=False)
        del evaluator
    torch.cuda.empty_cache()
    if args.theory_only:
        log.info('THEORY COMPLETE: %s; no original image IO attempted; archive pending %s',evaluated_path,archive)
        return
    # Identical image sampling and normalization to the original natural-image run.
    staged=stage_vanhateren_uint8(args.source_dir,storage/'vanhateren_log_gray_100px_22000_uint8.npy',
                                2000,20000,100,1,10,20260818,10,log)
    train=torch.as_tensor(np.asarray(staged[:2000],dtype=np.float64),device=device)/255
    population=torch.as_tensor(np.asarray(staged[2000:],dtype=np.float64),device=device)/255
    population-=population.mean(0,keepdim=True)
    beta=torch.as_tensor(create_disk_teacher(100,.3),device=device,dtype=torch.float64)
    target=population@beta
    direct_S=target.square().mean().item()
    log.info('Spectrum S=%.9g; direct heldout S=%.9g; relative mismatch %.3g',S,direct_S,abs(S-direct_S)/S)
    if not np.isclose(S,direct_S,rtol=1e-4): raise RuntimeError('Population mismatch with original spectrum')
    log.info('Natural-image MC: centered ridge; full n=%d d=%d; %d settings/trial',n,len(s),len(selected))
    sigma=torch.as_tensor(selected.sigma.to_numpy(),device=device)
    alpha=torch.as_tensor(selected.alpha.to_numpy(),device=device)

    def trial(index):
        generator=torch.Generator(device=device).manual_seed(20260918+index)
        indices=torch.randperm(len(train),device=device,generator=generator)[:n]
        X=train[indices].clone();X-=X.mean(0,keepdim=True)
        eps=torch.randn(n,device=device,generator=generator,dtype=torch.float64);eps-=eps.mean()
        gram=X@X.T
        eig,Q=torch.linalg.eigh(gram);eig=eig.clamp_min(0)
        y=X@beta
        response=(Q.T@y)[:,None]+(Q.T@eps)[:,None]*sigma[None,:]
        weights=X.T@(Q@(response/(eig[:,None]+alpha[None,:])))
        D=weights.square().sum(0);N=beta@weights
        # Evaluate on the independent population used to define Sigma, not training images.
        delta=weights-beta[:,None]
        Eg=(population@delta).square().mean(0)/S
        Ea=(1-N/D).square()
        result=selected[['policy','ratio','sigma','alpha','lam']].copy()
        result['trial']=index
        result['E_gen']=Eg.cpu().numpy();result['E_acc']=Ea.cpu().numpy()
        result['slope_acc']=(N/D).cpu().numpy()
        result['R2_acc']=(1-(D/N-1).square()).cpu().numpy()
        return result

    torch.cuda.synchronize();start=time.perf_counter()
    pilot=trial(1000000)
    torch.cuda.synchronize();elapsed=time.perf_counter()-start
    log.info('Full-noise/full-policy MC pilot %.2fs/trial; %d-trial ETA %.1fs',elapsed,args.trials,elapsed*args.trials)
    if elapsed*args.trials>args.max_seconds: raise RuntimeError('MC pilot exceeds budget; profile before scaling')
    all_trials=[]
    for i in tqdm(range(args.trials),desc='Natural-image MC'):
        path=storage/f'trial_{i:04d}.csv'
        if path.exists(): frame=pd.read_csv(path)
        else:
            frame=trial(i);frame.to_csv(path,index=False)
        all_trials.append(frame)
        if i%10==0:log.info('MC %d/%d complete',i+1,args.trials)
    raw=pd.concat(all_trials,ignore_index=True)
    raw.to_csv(storage/'mc_raw.csv',index=False)
    table=selected.copy()
    for metric in ['E_gen','E_acc']:
        means=[];ses=[]
        for r in table.itertuples():
            q=raw[(raw.policy==r.policy)&np.isclose(raw.ratio,r.ratio,rtol=1e-10,atol=0)]
            assert len(q)==args.trials
            means.append(q[metric].mean());ses.append(q[metric].sem())
        table[metric+'_mc']=means;table[metric+'_mc_se']=ses
    table['trials']=args.trials
    table=add_mc_variability(table,raw)
    table.to_csv(storage/'comparison.csv',index=False)
    out=root/'notebooks/outputs/pixel_ridge/vanhateren_selection_estimation'
    out.mkdir(parents=True,exist_ok=True)
    table.to_csv(out/'comparison.csv',index=False)
    (out/'config.json').write_text(json.dumps(config,indent=2))
    fig,_=plot_comparison(table)
    fig.suptitle('Van Hateren disk teacher: full 10,000 pixels, n=1,000',y=1.01,fontsize=17)
    fig.savefig(out/'selection_vs_estimation.png',dpi=180,bbox_inches='tight')
    fig.savefig(out/'selection_vs_estimation.pdf',bbox_inches='tight')
    log.info('DONE figure %s; raw data %s',out,storage)


if __name__=='__main__': main()

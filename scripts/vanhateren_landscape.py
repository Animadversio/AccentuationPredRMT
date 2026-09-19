"""Full noise x ridge landscape, with reusable exact noise-polynomial caches.

All errors E_gen/E_acc are normalized by S; weight_error is unnormalized.
Natural-image MC is distinct from the diagonal Gaussian coefficient surrogate.
"""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/accentuationpredrmt-matplotlib')
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from scripts.pixel_ridge_notebook_utils import load_problem, kappa_for_alpha
from scripts.validate_ffhq_disk_teacher import create_disk_teacher
from scripts.vanhateren_regularization_paths import theory

ROOT = Path(__file__).resolve().parents[1]
METRICS = ['E_gen', 'E_acc', 'R2_gen', 'R2_acc', 'slope_gen', 'slope_acc', 'weight_error']


def evaluate(c, amplitude, S, beta_norm):
    """Exact polynomial evaluation; amplitude may be sigma or sqrt(tau)."""
    a = amplitude
    N = c['N0'] + a*c['N1']
    D = c['D0'] + 2*a*c['D1'] + a*a*c['D2']
    E = c['E0'] + 2*a*c['E1'] + a*a*c['E2']
    Q = c['Q0'] + 2*a*c['Q1'] + a*a*c['Q2']
    C = c['C0'] + a*c['C1']
    with np.errstate(divide='ignore', invalid='ignore'):
        return dict(E_gen=E/S, E_acc=(1-N/D)**2, R2_gen=1-E/S,
                    R2_acc=1-(D/N-1)**2, slope_gen=C/Q, slope_acc=N/D,
                    weight_error=D-2*N+beta_norm)


def grid():
    # Round log coordinates, not floating-point dataframe grouping keys.
    powers=np.unique(np.round(np.r_[np.linspace(-8,1,91),np.linspace(-6,1,25)],12))
    ratios=np.r_[0.,10.**powers]
    old=ROOT/'notebooks/outputs/pixel_ridge/vanhateren_regularization_paths/mc_summary.csv'
    alphas=np.geomspace(1e-5,1e7,401)
    if old.exists(): alphas=np.r_[alphas,pd.read_csv(old).alpha.unique()]
    # Preserve all previous deterministic policy cuts as exact evaluation points.
    prior=pd.read_csv(ROOT/'notebooks/outputs/pixel_ridge/vanhateren_selection_estimation/comparison_extended.csv')
    alphas=np.r_[alphas,prior.alpha.unique(),100.]
    alphas=10.**np.unique(np.round(np.log10(alphas),12))
    return ratios,alphas


def make_coefficients(X, y, eps, pop, target, beta, alphas):
    eig,U=torch.linalg.eigh(X@X.T)
    eig=eig.clamp_min(0)
    at=torch.as_tensor(alphas,device=X.device)
    B=X.T@U
    w0=B@((U.T@y)[:,None]/(eig[:,None]+at))
    w1=B@((U.T@eps)[:,None]/(eig[:,None]+at))
    p0=pop@w0; p1=pop@w1; e0=p0-target[:,None]
    c=dict(N0=beta@w0,N1=beta@w1,
           D0=w0.square().sum(0),D1=(w0*w1).sum(0),D2=w1.square().sum(0),
           E0=e0.square().mean(0),E1=(e0*p1).mean(0),E2=p1.square().mean(0),
           Q0=p0.square().mean(0),Q1=(p0*p1).mean(0),Q2=p1.square().mean(0),
           C0=(target[:,None]*p0).mean(0),C1=(target[:,None]*p1).mean(0))
    # Exact linear-smoother LOOCV including the fitted intercept leverage 1/n.
    leverage=U.square()@(eig[:,None]/(eig[:,None]+at))+1/len(X)
    denominator=1-leverage
    if denominator.min()<=0: raise RuntimeError('Nonpositive LOOCV denominator')
    e_cv0=(y[:,None]-X@w0)/denominator
    e_cv1=(eps[:,None]-X@w1)/denominator
    c.update(CV0=e_cv0.square().mean(0),CV1=(e_cv0*e_cv1).mean(0),CV2=e_cv1.square().mean(0))
    return {k:v.cpu().numpy() for k,v in c.items()},(w0,w1)


def summarize(values):
    return dict(mean=np.mean(values,axis=0),std=np.std(values,axis=0,ddof=1),
                se=np.std(values,axis=0,ddof=1)/np.sqrt(len(values)),
                median=np.median(values,axis=0),q10=np.quantile(values,.1,axis=0),
                q90=np.quantile(values,.9,axis=0))


def gaussian_surface(out,s,b,n,sigmas,alphas,draws,seed,label):
    """Reduce Gaussian draws once per alpha, then reuse across every noise."""
    dest=out/f'gaussian_{label}.npz'
    if dest.exists(): return
    start=time.perf_counter()
    with np.load(out/'theory.npz') as z:
        ks=z['kappa']; eg=z['E_gen']
    S=s@b**2
    g=torch.Generator(device='cuda').manual_seed(seed)
    z=torch.randn((draws,len(s)),device='cuda',generator=g,dtype=torch.float64)
    z2=z.square()
    # Each row holds a draw and each column a regularization value.
    m=s[:,None]/(s[:,None]+ks)*b[:,None]
    f=s[:,None]/(s[:,None]+ks)**2
    mt=torch.as_tensor(m,device='cuda'); ft=torch.as_tensor(f,device='cuda')
    st=torch.as_tensor(s[:,None],device='cuda'); bt=torch.as_tensor(b[:,None],device='cuda')
    sqrtf=ft.sqrt()
    coeff=dict(N0=(bt*mt).sum(0),N1=z@(bt*sqrtf),
        D0=mt.square().sum(0),D1=z@(mt*sqrtf),D2=z2@ft,
        E0=(st*(mt-bt).square()).sum(0),E1=z@(st*(mt-bt)*sqrtf),E2=z2@(st*ft),
        Q0=(st*mt.square()).sum(0),Q1=z@(st*mt*sqrtf),Q2=z2@(st*ft),
        C0=(st*bt*mt).sum(0),C1=z@(st*bt*sqrtf))
    # Cache reduced draw coefficients, not the huge full coefficient samples.
    np.savez_compressed(out/f'gaussian_coefficients_{label}.npz',
                        **{k:v.cpu().numpy() for k,v in coeff.items()})
    result={}
    names=['E_acc'] if label=='selection' else METRICS
    for i,sigma in enumerate(tqdm(sigmas,desc=f'Gaussian {label} noise slices')):
        t=torch.as_tensor(np.sqrt((eg[i]*S+sigma*sigma)/n),device='cuda')
        N=coeff['N0']+t*coeff['N1']
        D=coeff['D0']+2*t*coeff['D1']+t*t*coeff['D2']
        E=coeff['E0']+2*t*coeff['E1']+t*t*coeff['E2']
        Q=coeff['Q0']+2*t*coeff['Q1']+t*t*coeff['Q2']
        C=coeff['C0']+t*coeff['C1']
        vals=dict(E_gen=E/S,E_acc=(1-N/D).square(),R2_gen=1-E/S,
                  R2_acc=1-(D/N-1).square(),slope_gen=C/Q,slope_acc=N/D,
                  weight_error=D-2*N+np.sum(b*b))
        for name in names:
            for suffix,v in [('mean',vals[name].mean(0)),('se',vals[name].std(0)/np.sqrt(draws))]:
                result.setdefault(name+'_'+suffix,[]).append(v.cpu().numpy())
        result.setdefault('N_nonpositive_fraction',[]).append((N<=0).double().mean(0).cpu().numpy())
        result.setdefault('min_abs_N',[]).append(N.abs().min(0).values.cpu().numpy())
        if i==0:
            print(f'Gaussian {label}: setup+first slice {time.perf_counter()-start:.2f}s',flush=True)
    np.savez_compressed(dest,draws=draws,seed=seed,**{k:np.array(v) for k,v in result.items()})
    print(f'Gaussian {label} complete in {time.perf_counter()-start:.1f}s',flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--work',type=Path,required=True)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--trials',type=int,default=100)
    parser.add_argument('--skip-gaussian',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(4)
    out=args.work/'landscape_v1'; out.mkdir(exist_ok=True)
    s,b=load_problem(ROOT,'vanhateren'); S=float(s@b**2); n=1000
    ratios,alphas=grid(); sigmas=np.sqrt(ratios*S)
    ks=np.array([kappa_for_alpha(s,n,a) for a in alphas])
    config=dict(n=n,d=len(s),signal_variance=S,ratios=ratios.tolist(),sigma=sigmas.tolist(),
        alpha=alphas.tolist(),lam=(alphas/n).tolist(),kappa=ks.tolist(),
        axes=['noise','regularization'],mc_seed=20260918,requested_trials=100,
        input_distribution='natural Van Hateren patches',train_pool=2000,population_pool=20000,
        image_preprocessing='100x100 log-gray uint8 / 255; centered fits and population',
        metric_convention='E_gen/E_acc normalized by S; weight_error unnormalized; intercept omitted in population evaluation',
        cv='exact LOOCV with fitted intercept; alpha=n*lambda',
        gaussian='diagonal Gaussian coefficient surrogate, NOT full distributional DE',
        archive=str(args.archive),code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if (out/'config.json').exists():
        old=json.loads((out/'config.json').read_text())
        for key in ['alpha','sigma','n','mc_seed']: assert old[key]==config[key]
    (out/'config.json').write_text(json.dumps(config,indent=2))
    np.savez_compressed(out/'coordinates.npz',sigma=sigmas,ratio=ratios,alpha=alphas,lam=alphas/n,kappa=ks,S=S,n=n,eigenvalues=s,beta_proj=b)
    print(f'Grid: {len(sigmas)} noise x {len(alphas)} alpha = {len(sigmas)*len(alphas)} cells; {args.trials} paired MC trials',flush=True)
    if not (out/'theory.npz').exists():
        start=time.perf_counter(); tab=theory(s,b,n,sigmas,alphas)
        values={name:tab[name].to_numpy().reshape(len(sigmas),len(alphas)) for name in METRICS if name!='weight_error'}
        mean=s[:,None]/(s[:,None]+ks)*b[:,None]
        df12=np.sum(s[:,None]/(s[:,None]+ks)**2,axis=0)
        values['weight_error']=np.sum((mean-b[:,None])**2,axis=0)[None,:]+(values['E_gen']*S+sigmas[:,None]**2)/n*df12
        np.savez_compressed(out/'theory.npz',kappa=ks,**values)
        cvtheory=theory(s,b,n-1,sigmas,alphas)
        np.savez_compressed(out/'theory_cv_nminus1.npz',E_gen=cvtheory.E_gen.to_numpy().reshape(len(sigmas),len(alphas)))
        print(f'DE surface {time.perf_counter()-start:.2f}s',flush=True)
    start=time.perf_counter()
    data=np.load(args.work/'vanhateren_log_gray_100px_22000_uint8.npy',mmap_mode='r')
    train=torch.as_tensor(np.array(data[:2000],dtype=np.float64),device='cuda')/255
    pop=torch.as_tensor(np.array(data[2000:],dtype=np.float64),device='cuda')/255
    pop-=pop.mean(0,keepdim=True)
    beta=torch.as_tensor(create_disk_teacher(100,.3),device='cuda',dtype=torch.float64)
    target=pop@beta
    print(f'Input staging {time.perf_counter()-start:.2f}s',flush=True)
    validation={}
    for trial in tqdm(range(args.trials),desc='MC coefficient fits'):
        path=out/f'coefficients_{trial:04d}.npz'
        if path.exists(): continue
        start=time.perf_counter()
        g=torch.Generator(device='cuda').manual_seed(20260918+trial)
        ids=torch.randperm(len(train),device='cuda',generator=g)[:n]
        X=train[ids].clone(); X-=X.mean(0,keepdim=True)
        eps=torch.randn(n,device='cuda',generator=g,dtype=torch.float64); eps-=eps.mean()
        y=X@beta
        c,(w0,w1)=make_coefficients(X,y,eps,pop,target,beta,alphas)
        torch.cuda.synchronize(); elapsed=time.perf_counter()-start
        np.savez_compressed(path,**c)
        print(f'Trial {trial+1}: fit+evaluation {elapsed:.2f}s; projected 100 trials {100*elapsed:.1f}s',flush=True)
        if trial==0:
            if elapsed*100>1800: raise RuntimeError('Pilot >30min; profile before scaling')
            # Independent direct ridge solve verifies all noise-polynomial metrics.
            sigma=float(np.sqrt(.1*S)); j=int(np.argmin(abs(np.log(alphas/100))))
            w=X.T@torch.linalg.solve(X@X.T+alphas[j]*torch.eye(n,device='cuda',dtype=torch.float64),y+sigma*eps)
            prediction=pop@w; N=beta@w; D=w@w
            direct=dict(E_gen=((prediction-target).square().mean()/S).item(),E_acc=((1-N/D)**2).item(),
                R2_gen=(1-(prediction-target).square().mean()/S).item(),R2_acc=(1-(D/N-1)**2).item(),
                slope_gen=((target@prediction)/prediction.square().sum()).item(),slope_acc=(N/D).item(),weight_error=(w-beta).square().sum().item())
            predicted=evaluate({k:v[j] for k,v in c.items()},sigma,S,float(beta@beta))
            for name in METRICS: np.testing.assert_allclose(predicted[name],direct[name],rtol=1e-6,atol=1e-9)
            validation['direct_ridge_polynomial_max_abs_error']=max(abs(predicted[k]-direct[k]) for k in METRICS)
            from sklearn.linear_model import RidgeCV
            from threadpoolctl import threadpool_limits
            check_sigmas=np.sqrt(S*np.array([.001,.1,10.]))
            with threadpool_limits(limits=4):
                model=RidgeCV(alphas=alphas,fit_intercept=True,cv=None,gcv_mode='eigen',alpha_per_target=True).fit(
                    X.cpu().numpy(),y.cpu().numpy()[:,None]+eps.cpu().numpy()[:,None]*check_sigmas)
            risks=c['CV0']+2*check_sigmas[:,None]*c['CV1']+check_sigmas[:,None]**2*c['CV2']
            np.testing.assert_allclose(alphas[risks.argmin(1)],model.alpha_,rtol=1e-9)
            np.testing.assert_allclose(risks.min(1),-model.best_score_,rtol=1e-6)
            validation['sklearn_RidgeCV_three_targets_matched']=True
            (out/'validation.json').write_text(json.dumps(validation,indent=2))
            print('Direct solve and sklearn RidgeCV validation passed.',flush=True)
    del train,pop,target; torch.cuda.empty_cache()
    # Exact reconstruction and uncertainty on the common rectangular grid.
    start=time.perf_counter()
    coeffs=[]
    for i in range(args.trials):
        with np.load(out/f'coefficients_{i:04d}.npz') as z: coeffs.append({k:z[k] for k in z.files})
    cubes={name:[] for name in METRICS}; cv_choices=[]; cv_curves=[]
    for i,c in enumerate(tqdm(coeffs,desc='Reconstruct noise axis')):
        vals=evaluate(c,sigmas[:,None],S,float(beta@beta))
        cv=c['CV0']+2*sigmas[:,None]*c['CV1']+sigmas[:,None]**2*c['CV2']
        cv_choices.append(cv.argmin(1)); cv_curves.append(cv)
        for name in METRICS: cubes[name].append(vals[name])
    summaries={}
    for name in METRICS:
        v=np.stack(cubes.pop(name))
        np.savez_compressed(out/f'mc_raw_{name}.npz',values=v)
        for suffix,value in summarize(v).items(): summaries[name+'_'+suffix]=value
    summaries['trials']=np.array(args.trials)
    np.savez_compressed(out/'mc_summary.npz',**summaries)
    np.savez_compressed(out/'mc_loocv.npz',risk=np.stack(cv_curves),selected_index=np.stack(cv_choices))
    print(f'Grid reconstruction+cache {time.perf_counter()-start:.2f}s',flush=True)
    if not args.skip_gaussian:
        gaussian_surface(out,s,b,n,sigmas,alphas,2048,123,'selection')
        gaussian_surface(out,s,b,n,sigmas,alphas,8192,20260917,'evaluation')
    print(f'COMPLETE: {out}',flush=True)


if __name__=='__main__': main()

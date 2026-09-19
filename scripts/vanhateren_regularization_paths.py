"""Four-noise full-pixel risk paths; natural-image MC and Gaussian surrogate."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/accentuationpredrmt-matplotlib')
import argparse
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import RidgeCV
from threadpoolctl import threadpool_limits
from tqdm.auto import tqdm
from scripts.pixel_ridge_notebook_utils import load_problem, kappa_for_alpha
from scripts.validate_ffhq_disk_teacher import create_disk_teacher

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'notebooks/outputs/pixel_ridge/vanhateren_regularization_paths'
METRICS = ['E_gen','E_acc','R2_gen','R2_acc','slope_gen','slope_acc']


def theory(s, b, n, sigmas, alphas):
    S = s@b**2
    k = np.array([kappa_for_alpha(s,n,a) for a in alphas])
    r = s[:,None]/(s[:,None]+k)
    means = b[:,None]*r
    d2 = (r*r).sum(0)
    bias = (s[:,None]*(b[:,None]-means)**2).sum(0)
    rows=[]
    for sigma in sigmas:
        eg = n/(n-d2)*(bias+sigma**2)-sigma**2
        v = (eg+sigma**2)[None,:]/n*s[:,None]/(s[:,None]+k)**2
        N = b@means
        D = (means**2+v).sum(0)
        slope = N/D
        rows.append(pd.DataFrame(dict(sigma=sigma,ratio=sigma**2/S,alpha=alphas,
            lam=alphas/n,kappa=k,E_gen=eg/S,E_acc=(1-slope)**2,
            R2_gen=1-eg/S,R2_acc=1-(1/slope-1)**2,slope_acc=slope,
            slope_gen=(s*b)@means/(s[:,None]*(means**2+v)).sum(0))))
    return pd.concat(rows,ignore_index=True)


def distribution(table,s,b,n,draws,seed):
    g=torch.Generator(device='cuda').manual_seed(seed)
    z=torch.randn((draws,len(s)),generator=g,device='cuda',dtype=torch.float64)
    z2=z.square()
    bt=torch.as_tensor(b,device='cuda')
    result=table.copy()
    S=s@b**2
    for start in tqdm(range(0,len(table),128),desc=f'Gaussian surrogate {draws} draws'):
        q=table.iloc[start:start+128]
        k=q.kappa.to_numpy()
        m=s[:,None]/(s[:,None]+k)*b[:,None]
        v=(q.E_gen.to_numpy()*S+q.sigma.to_numpy()**2)[None,:]/n*s[:,None]/(s[:,None]+k)**2
        mt=torch.as_tensor(m,device='cuda'); vt=torch.as_tensor(v,device='cuda'); sd=vt.sqrt()
        N=bt@mt+z@(sd*bt[:,None])
        D=mt.square().sum(0)+2*z@(mt*sd)+z2@vt
        for name,values in [('E_acc',(1-N/D).square()),('R2_acc',1-(D/N-1).square()),('slope_acc',N/D)]:
            result.loc[q.index,name+'_dist']=values.mean(0).cpu().numpy()
            result.loc[q.index,name+'_dist_se']=(values.std(0)/np.sqrt(draws)).cpu().numpy()
    return result


def compute(work,trials):
    torch.set_num_threads(4)
    OUT.mkdir(parents=True,exist_ok=True)
    cache=work/'regularization_paths_v1'; cache.mkdir(exist_ok=True)
    s,b=load_problem(ROOT,'vanhateren'); n=1000; S=s@b**2
    sigmas=np.sqrt(S*np.array([.001,.01,.1,1.]))
    dense=np.geomspace(1e-5,1e7,401)
    if not (OUT/'theory.csv').exists():
        start=time.perf_counter()
        t=theory(s,b,n,sigmas,dense)
        selection=distribution(t,s,b,n,2048,123)
        t=distribution(t,s,b,n,8192,20260917)
        selected=[]
        for sigma in sigmas:
            q=t[t.sigma==sigma]; qs=selection[selection.sigma==sigma]
            # n-1 proxy matches the earlier DE-generalization selection convention.
            cv=theory(s,b,n-1,[sigma],dense)
            for name,idx in [('gen_DE',cv.E_gen.idxmin()),('acc_leading',q.E_acc.idxmin()-q.index[0]),
                             ('acc_distribution',qs.E_acc_dist.idxmin()-qs.index[0])]:
                r=q.iloc[idx]
                selected.append(dict(sigma=sigma,method=name,alpha=r.alpha,lam=r.lam,kappa=r.kappa,
                                     boundary=bool(idx in [0,len(q)-1])))
        pd.DataFrame(selected).to_csv(OUT/'selected.csv',index=False)
        selection.to_csv(cache/'distribution_selection_path.csv',index=False)
        t.to_csv(OUT/'theory.csv',index=False)
        print(f'Theory complete {time.perf_counter()-start:.1f}s',flush=True)
    selected=pd.read_csv(OUT/'selected.csv')
    # Include all theory optima in the shared MC grid in addition to a broad scan.
    alphas=np.unique(np.r_[np.geomspace(1e-5,1e7,49),selected.alpha])
    path=theory(s,b,n,sigmas,alphas)
    data=np.load(work/'vanhateren_log_gray_100px_22000_uint8.npy',mmap_mode='r')
    train=torch.as_tensor(np.array(data[:2000],dtype=np.float64),device='cuda')/255
    pop=torch.as_tensor(np.array(data[2000:],dtype=np.float64),device='cuda')/255
    pop-=pop.mean(0,keepdim=True)
    beta=torch.as_tensor(create_disk_teacher(100,.3),device='cuda',dtype=torch.float64)
    target=pop@beta
    at=torch.as_tensor(path.alpha.to_numpy(),device='cuda')
    st=torch.as_tensor(path.sigma.to_numpy(),device='cuda')
    frames=[]; cvs=[]
    for trial in tqdm(range(trials),desc='Natural-image paired MC paths'):
        dest=cache/f'trial_{trial:04d}.csv'; cvdest=cache/f'cv_{trial:04d}.csv'
        if dest.exists() and cvdest.exists():
            frames.append(pd.read_csv(dest)); cvs.append(pd.read_csv(cvdest)); continue
        start=time.perf_counter()
        g=torch.Generator(device='cuda').manual_seed(20260918+trial)
        ids=torch.randperm(2000,device='cuda',generator=g)[:n]
        X=train[ids].clone(); X-=X.mean(0,keepdim=True)
        eps=torch.randn(n,device='cuda',generator=g,dtype=torch.float64); eps-=eps.mean()
        y=X@beta
        eig,Q=torch.linalg.eigh(X@X.T); eig=eig.clamp_min(0)
        response=(Q.T@y)[:,None]+(Q.T@eps)[:,None]*st
        w=X.T@(Q@(response/(eig[:,None]+at)))
        pred=pop@w
        eg=(pred-target[:,None]).square().mean(0)/S
        slope=(beta@w)/w.square().sum(0)
        frame=path[['sigma','ratio','alpha','lam','kappa']].copy(); frame['trial']=trial
        vals=[eg,(1-slope).square(),1-eg,1-(1/slope-1).square(),target@pred/pred.square().sum(0),slope]
        for name,value in zip(METRICS,vals): frame[name]=value.cpu().numpy()
        with threadpool_limits(limits=4):
            model=RidgeCV(alphas=dense,cv=None,fit_intercept=True,gcv_mode='eigen',alpha_per_target=True).fit(
                X.cpu().numpy(),(y[:,None]+eps[:,None]*torch.as_tensor(sigmas,device='cuda')).cpu().numpy())
        cv=pd.DataFrame(dict(trial=trial,sigma=sigmas,alpha=model.alpha_,lam=model.alpha_/n))
        cv['kappa']=[kappa_for_alpha(s,n,a) for a in cv.alpha]
        frame.to_csv(dest,index=False); cv.to_csv(cvdest,index=False)
        frames.append(frame); cvs.append(cv)
        elapsed=time.perf_counter()-start
        print(f'Trial {trial+1}/{trials}: {elapsed:.2f}s; projected 100 trials {100*elapsed:.1f}s',flush=True)
        if trial==0 and elapsed*100>1800: raise RuntimeError('Profile before scaling beyond 30min')
    # CSV round-trips can change the last bit of float keys. Restore the exact
    # shared grid before grouping cached pilot and newly computed trials.
    for frame in frames:
        np.testing.assert_allclose(frame[['sigma','alpha']],path[['sigma','alpha']],rtol=1e-12)
        for key in ['sigma','ratio','alpha','lam','kappa']:
            frame[key]=path[key].to_numpy()
    raw=pd.concat(frames,ignore_index=True); raw.to_csv(cache/'mc_raw.csv',index=False)
    pd.concat(cvs,ignore_index=True).to_csv(OUT/'cv_choices.csv',index=False)
    rows=[]
    for (sigma,alpha),q in raw.groupby(['sigma','alpha']):
        row=dict(sigma=sigma,alpha=alpha,lam=q.lam.iloc[0],kappa=q.kappa.iloc[0],trials=len(q))
        for name in METRICS:
            for suffix,value in [('mean',q[name].mean()),('se',q[name].sem()),('q10',q[name].quantile(.1)),('q90',q[name].quantile(.9))]: row[name+'_'+suffix]=value
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT/'mc_summary.csv',index=False)


def plot(xname, zoom=False):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,'font.family':'DejaVu Sans','mathtext.fontset':'dejavusans'})
    t=pd.read_csv(OUT/'theory.csv'); mc=pd.read_csv(OUT/'mc_summary.csv')
    choices=pd.read_csv(OUT/'cv_choices.csv'); selected=pd.read_csv(OUT/'selected.csv')
    fig,axes=plt.subplots(4,4,figsize=(20,15),sharex='col',sharey='row',gridspec_kw={'height_ratios':[1.2,1,1,.65]})
    blue='#15579b'; orange='#bb4a0a'; gold='#ec9a25'
    methods=['gen_DE','acc_leading','acc_distribution']; marks=['v','s','*']; colors=['#15579b','#bb4a0a','#ec9a25']
    for col,sigma in enumerate(sorted(t.sigma.unique())):
        q=t[t.sigma==sigma]; m=mc[np.isclose(mc.sigma,sigma)]; sel=selected[np.isclose(selected.sigma,sigma)]
        axes[0,col].set_title(rf'$\sigma={sigma:.3g}$; $\sigma^2/S={q.ratio.iloc[0]:g}$',pad=13)
        for row,(gen,acc) in enumerate([('E_gen','E_acc'),('R2_gen','R2_acc'),('slope_gen','slope_acc')]):
            ax=axes[row,col]
            ax.plot(q[xname],q[gen],color=blue,lw=2)
            ax.plot(q[xname],q[acc],color=orange,ls='--',lw=2)
            ax.plot(q[xname],q[acc+'_dist'],color=gold,ls='-.',lw=2)
            for name,color,marker in [(gen,blue,'o'),(acc,orange,'D')]:
                ax.fill_between(m[xname],m[name+'_q10'],m[name+'_q90'],color=color,alpha=.10)
                ax.errorbar(m[xname],m[name+'_mean'],yerr=2*m[name+'_se'],fmt=marker,ms=3,mfc='white',mec=color,ecolor=color,elinewidth=.6,alpha=.85)
            ax.set_xscale('log'); ax.grid(alpha=.18)
        axes[0,col].set_yscale('log'); axes[0,col].set_ylim(1e-7,1e2)
        axes[1,col].set_yscale('symlog',linthresh=1); axes[1,col].axhline(0,c='gray',lw=.7)
        axes[2,col].set_yscale('log')
        axes[2,col].axhline(1,c='gray',lw=.7)
        ax=axes[3,col]
        cv=choices[np.isclose(choices.sigma,sigma)][xname]
        # A rug of individual selected values plus median and 10--90% range.
        ax.scatter(cv,np.full(len(cv),3),marker='|',color='#238b45',alpha=.3,s=100)
        ax.errorbar(cv.median(),3,xerr=[[cv.median()-cv.quantile(.1)],[cv.quantile(.9)-cv.median()]],fmt='o',color='#238b45',capsize=4)
        for j,(method,mark,color) in enumerate(zip(methods,marks,colors)):
            r=sel[sel.method==method].iloc[0]
            ax.scatter(r[xname],j,marker=mark,color=color,s=80,zorder=4)
            for ar in axes[:3,col]: ar.axvline(r[xname],color=color,lw=.8,alpha=.25)
        ax.set_yticks(range(4),['Gen DE','Acc leading','Acc distr.','sklearn CV'])
        ax.set_ylim(-.6,3.6); ax.set_xscale('log'); ax.grid(axis='x',alpha=.18)
        ax.set_xlabel(r'$\lambda$ ($\alpha=n\lambda$)' if xname=='lam' else r'Effective regularization $\kappa$')
        if zoom:
            visible=q[(q.lam>=1e-4)&(q.lam<=10)]
            ax.set_xlim(visible[xname].min(),visible[xname].max())
    for row,label in enumerate([r'Error / signal variance $E/S$',r'$R^2$',r'True-on-fitted slope']): axes[row,0].set_ylabel(label)
    if zoom:
        axes[1,0].set_ylim(-1e7,1.5)
        axes[2,0].set_ylim(1e-4,5)
    handles=[Line2D([],[],c=blue,label='Gen: DE'),Line2D([],[],c=orange,ls='--',label='Acc: leading DE'),Line2D([],[],c=gold,ls='-.',label='Acc: distributional surrogate'),Line2D([],[],c=blue,marker='o',mfc='white',ls='',label='Gen: MC mean'),Line2D([],[],c=orange,marker='D',mfc='white',ls='',label='Acc: MC mean')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.957),ncol=5,frameon=False)
    fig.suptitle('Van Hateren disk teacher: regularization risk paths | 10,000 pixels, n=1,000',fontsize=20,y=.99)
    fig.text(.5,.015,f'MC: {int(mc.trials.min())} paired trials; mean ±2 SE; shading: trial 10–90%. Bottom: sklearn median / 10–90% + trial rug.\nDistributional DE: diagonal Gaussian surrogate, 8,192 evaluation draws; inverse-N R² may be heavy-tailed. Selection uses independent draws.',ha='center',fontsize=11)
    fig.subplots_adjust(top=.895,bottom=.08,hspace=.14,wspace=.12)
    suffix='_zoom' if zoom else ''
    for ext in ['png','pdf']: fig.savefig(OUT/f'regularization_paths_{xname}{suffix}.{ext}',dpi=160,bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--work',type=Path); p.add_argument('--trials',type=int,default=100); p.add_argument('--plot-only',action='store_true')
    args=p.parse_args()
    if not args.plot_only: compute(args.work,args.trials)
    for xname in ['lam','kappa']:
        plot(xname)
        plot(xname,zoom=True)

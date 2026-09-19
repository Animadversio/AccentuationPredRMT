"""Plot exact 1D cuts through the cached Van Hateren 2D landscape."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/accentuationpredrmt-matplotlib')
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

BLUE='#15579b'; LIGHTBLUE='#489fce'; ORANGE='#bb4a0a'; LIGHTORANGE='#ee6633'; GREEN='#238b45'


def load(path):
    with np.load(path) as z: return {k:z[k] for k in z.files}


def nearest_log(values, targets):
    values=np.asarray(values)
    return np.array([np.argmin(abs(np.log(values/t))) for t in targets])


def add_sigma_axis(ax,S):
    top=ax.secondary_xaxis('top',functions=(lambda r:np.sqrt(np.maximum(r,0)*S),lambda s:s*s/S))
    top.set_xlabel(r'Response noise $\sigma$')


def add_lambda_axis(ax,kappa,lam):
    lk=np.log(kappa); ll=np.log(lam)
    def k_to_l(k): return np.exp(np.interp(np.log(np.maximum(k,kappa[0])),lk,ll))
    def l_to_k(l): return np.exp(np.interp(np.log(np.maximum(l,lam[0])),ll,lk))
    top=ax.secondary_xaxis('top',functions=(k_to_l,l_to_k)); top.set_xlabel(r'Ridge $\lambda$')
    # Tiny lambda values collapse against the positive kappa floor; omit the
    # leftmost label to prevent visually overlapping, nearly identical ticks.
    top.set_xticks([1e-4,1,1e4])


def shared_handles():
    return [Line2D([],[],c=BLUE,lw=2,label='Gen: DE'),
            Line2D([],[],c=LIGHTBLUE,marker='o',mfc='white',label='Gen: MC mean'),
            Line2D([],[],c=ORANGE,ls='--',lw=2,label='Acc: leading DE'),
            Line2D([],[],c='#ec9a25',ls='-.',label='Acc: Gaussian surrogate'),
            Line2D([],[],c=LIGHTORANGE,marker='D',mfc='white',label='Acc: MC mean'),
            Patch(fc=LIGHTBLUE,alpha=.13,label='MC trial 10–90%')]


METRIC_SPECS={
    'error':('E_gen','E_acc',r'Normalized error $E/S$'),
    'r2':('R2_gen','R2_acc',r'$R^2$'),
    'slope':('slope_gen','slope_acc','True-on-fitted slope'),
}


def set_metric_axis(ax,category):
    if category=='error':
        ax.set_yscale('log'); ax.set_ylim(1e-7,30)
    elif category=='r2':
        ax.set_yscale('symlog',linthresh=1); ax.axhline(0,color='gray',lw=.7); ax.axhline(1,color='gray',lw=.7,ls=':')
    else:
        ax.set_yscale('symlog',linthresh=1e-3); ax.axhline(1,color='gray',lw=.7,ls=':')


def metric_curves(ax,x,de,mc,dg,noise_slice,reg_slice,category,markevery=5):
    gen,acc,_=METRIC_SPECS[category]
    sel=(noise_slice,reg_slice)
    ax.plot(x,de[gen][sel],color=BLUE,lw=2,label=f'{gen}: DE')
    ax.plot(x,de[acc][sel],color=ORANGE,lw=2,ls='--',label=f'{acc}: leading DE')
    ax.plot(x,dg[acc+'_mean'][sel],color='#ec9a25',lw=1.6,ls='-.',label=f'{acc}: Gaussian surrogate')
    for name,color,marker in [(gen,LIGHTBLUE,'o'),(acc,LIGHTORANGE,'D')]:
        lo=mc[name+'_q10'][sel]; hi=mc[name+'_q90'][sel]; mean=mc[name+'_mean'][sel]
        ax.fill_between(x,lo,hi,color=color,alpha=.13,lw=0)
        ax.plot(x,mean,color=color,marker=marker,markevery=markevery,ms=3,mfc='white',lw=.8,alpha=.9)
    ax.set_xscale('log'); set_metric_axis(ax,category); ax.grid(alpha=.18)


def fixed_regularization(c,de,mc,dg,out,category):
    targets=[1e-4,1e-2,1e-1,1.]
    js=nearest_log(c['lam'],targets); ratio=c['ratio'][1:]
    fig,axes=plt.subplots(1,4,figsize=(18,5.3),sharex=True,sharey=True)
    rows=[]
    for ax,j in zip(axes,js):
        metric_curves(ax,ratio,de,mc,dg,slice(1,None),j,category,7)
        ax.set_title(rf'$\lambda={c["lam"][j]:.3g}$; $\kappa={c["kappa"][j]:.3g}$')
        ax.set_xlabel(r'Noise / signal variance $\sigma^2/S$'); add_sigma_axis(ax,float(c['S']))
        for i in range(len(c['ratio'])):
            rows.append(dict(cut='fixed_lambda',noise_index=i,regularization_index=j,sigma=c['sigma'][i],ratio=c['ratio'][i],alpha=c['alpha'][j],lam=c['lam'][j],kappa=c['kappa'][j]))
    axes[0].set_ylabel(METRIC_SPECS[category][2])
    fig.legend(handles=shared_handles(),loc='upper center',bbox_to_anchor=(.5,.91),ncol=6,frameon=False)
    fig.suptitle(f'Landscape cuts: fixed regularization, varying response noise — {category}',fontsize=16,y=.995)
    fig.subplots_adjust(top=.69,bottom=.15,left=.06,right=.99,wspace=.08)
    for ext in ['png','pdf']: fig.savefig(out/f'slices_fixed_lambda_{category}.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig); return rows


def fixed_noise(c,de,mc,dg,out,category):
    targets=[1e-3,1e-2,1e-1,1.]
    ii=nearest_log(c['ratio'][1:],targets)+1; kappa=c['kappa']
    fig,axes=plt.subplots(1,4,figsize=(18,5.3),sharex=True,sharey=True)
    rows=[]
    for ax,i in zip(axes,ii):
        metric_curves(ax,kappa,de,mc,dg,i,slice(None),category,24)
        ax.set_title(rf'$\sigma^2/S={c["ratio"][i]:g}$; $\sigma={c["sigma"][i]:.3g}$')
        ax.set_xlabel(r'Effective regularization $\kappa$'); add_lambda_axis(ax,c['kappa'],c['lam'])
        for j in range(len(c['lam'])):
            rows.append(dict(cut='fixed_noise',noise_index=i,regularization_index=j,sigma=c['sigma'][i],ratio=c['ratio'][i],alpha=c['alpha'][j],lam=c['lam'][j],kappa=c['kappa'][j]))
    axes[0].set_ylabel(METRIC_SPECS[category][2])
    fig.legend(handles=shared_handles(),loc='upper center',bbox_to_anchor=(.5,.91),ncol=6,frameon=False)
    fig.suptitle(f'Landscape cuts: fixed noise, varying effective regularization — {category}',fontsize=16,y=.995)
    fig.subplots_adjust(top=.69,bottom=.15,left=.06,right=.99,wspace=.08)
    for ext in ['png','pdf']: fig.savefig(out/f'slices_fixed_noise_{category}.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig); return rows


def prediction_selected(c,de,mc,indices,cv,cv_eval,out,category):
    gen,acc,ylabel=METRIC_SPECS[category]
    ratio=c['ratio'][1:]; grid_idx=indices['gen_DE'][1:].astype(int); i=np.arange(1,len(c['ratio']))
    fig,axes=plt.subplots(2,2,figsize=(13,8),sharex='col',layout='constrained',gridspec_kw={'height_ratios':[1.45,1]})
    ax=axes[0,0]
    for name,color,ls in [(gen,BLUE,'-'),(acc,ORANGE,'--')]:
        ax.plot(ratio,de[name][i,grid_idx],color=color,ls=ls,lw=2,label=f'{name}: DE')
        ax.fill_between(ratio,mc[name+'_q10'][i,grid_idx],mc[name+'_q90'][i,grid_idx],color=color,alpha=.13)
        ax.plot(ratio,mc[name+'_mean'][i,grid_idx],color=color,marker='o' if name=='E_gen' else 'D',markevery=7,ms=3,mfc='white',lw=.8,label=f'{name}: MC mean')
    ax.set_title('DE prediction-optimal grid path'); ax.set_ylabel(ylabel); set_metric_axis(ax,category); ax.grid(alpha=.18); ax.legend(fontsize=8,ncol=2)
    ax=axes[0,1]; cv_index=cv['selected_index'][:,1:].astype(int)
    for name,color,marker in [(gen,BLUE,'o'),(acc,ORANGE,'D')]:
        values=cv_eval[name][:,1:]
        mean=values.mean(0); lo=np.quantile(values,.1,axis=0); hi=np.quantile(values,.9,axis=0)
        ax.fill_between(ratio,lo,hi,color=color,alpha=.13)
        ax.plot(ratio,mean,color=color,marker=marker,markevery=7,ms=3,mfc='white',lw=1.5,label=f'{name}: trialwise MC')
    ax.set_title('Empirical LOOCV / sklearn-equivalent path'); set_metric_axis(ax,category); ax.grid(alpha=.18); ax.legend(fontsize=8)
    ax=axes[1,0]
    ax.plot(ratio,c['kappa'][grid_idx],color='#6941a5',lw=2,label=r'$\kappa_{gen,DE}$')
    ax.plot(ratio,c['lam'][grid_idx],color='gray',ls='--',lw=2,label=r'$\lambda_{gen,DE}$')
    ax.set_ylabel('Selected regularization'); ax.legend(fontsize=8)
    ax=axes[1,1]
    for coord,color,ls,label in [('kappa','#6941a5','-',r'$\kappa$'),('lam','gray','--',r'$\lambda$')]:
        values=c[coord][cv_index]
        ax.fill_between(ratio,np.quantile(values,.1,axis=0),np.quantile(values,.9,axis=0),color=color,alpha=.13)
        ax.plot(ratio,np.median(values,axis=0),color=color,ls=ls,lw=2,label=f'median {label}; 10–90%')
    ax.legend(fontsize=8)
    for col in range(2):
        for row in range(2): axes[row,col].set_xscale('log'); add_sigma_axis(axes[row,col],float(c['S']))
        axes[1,col].set_yscale('log'); axes[1,col].grid(alpha=.18); axes[1,col].set_xlabel(r'Noise / signal variance $\sigma^2/S$')
    fig.suptitle(f'Prediction-selected paths extracted from the same 2D landscape — {category}',fontsize=16)
    for ext in ['png','pdf']: fig.savefig(out/f'slices_prediction_selected_{category}.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    rows=[]
    for n,j in zip(i,grid_idx): rows.append(dict(cut='prediction_DE_optimal',noise_index=n,regularization_index=j,sigma=c['sigma'][n],ratio=c['ratio'][n],alpha=c['alpha'][j],lam=c['lam'][j],kappa=c['kappa'][j]))
    return rows


def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',type=Path,required=True); p.add_argument('--figures',type=Path,required=True); p.add_argument('--summary',type=Path,required=True)
    args=p.parse_args(); args.figures.mkdir(parents=True,exist_ok=True); args.summary.parent.mkdir(parents=True,exist_ok=True)
    c=load(args.data/'coordinates.npz'); de=load(args.data/'theory.npz'); mc=load(args.data/'mc_summary.npz'); dg=load(args.data/'gaussian_evaluation.npz'); indices=load(args.data/'selected_indices.npz'); cv=load(args.data/'mc_loocv.npz'); cv_eval=load(args.data/'cv_selected_evaluations.npz')
    plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,'font.family':'DejaVu Sans','mathtext.fontset':'dejavusans'})
    rows=[]
    for category in METRIC_SPECS:
        new_rows=fixed_regularization(c,de,mc,dg,args.figures,category)+fixed_noise(c,de,mc,dg,args.figures,category)+prediction_selected(c,de,mc,indices,cv,cv_eval,args.figures,category)
        if category=='error': rows=new_rows
    frame=pd.DataFrame(rows)
    for name in ['E_gen','E_acc','R2_gen','R2_acc','slope_gen','slope_acc']:
        frame[name+'_DE']=[de[name][i,j] for i,j in zip(frame.noise_index,frame.regularization_index)]
        frame[name+'_MC_mean']=[mc[name+'_mean'][i,j] for i,j in zip(frame.noise_index,frame.regularization_index)]
        frame[name+'_MC_q10']=[mc[name+'_q10'][i,j] for i,j in zip(frame.noise_index,frame.regularization_index)]
        frame[name+'_MC_q90']=[mc[name+'_q90'][i,j] for i,j in zip(frame.noise_index,frame.regularization_index)]
    frame.to_csv(args.summary,index=False)
    print(f'Saved 9 PNG/PDF slice figures and {len(frame)} plot-ready rows.',flush=True)


if __name__=='__main__': main()

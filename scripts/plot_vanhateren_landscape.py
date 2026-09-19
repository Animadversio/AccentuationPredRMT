"""Cache selected cuts and export an initial DE / MC landscape overview."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/accentuationpredrmt-matplotlib')
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm
from matplotlib.lines import Line2D


def load(path):
    with np.load(path) as z: return {k:z[k] for k in z.files}


def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',type=Path,required=True); p.add_argument('--figures',type=Path,required=True)
    args=p.parse_args(); args.figures.mkdir(parents=True,exist_ok=True)
    c=load(args.data/'coordinates.npz'); de=load(args.data/'theory.npz'); mc=load(args.data/'mc_summary.npz')
    cv=load(args.data/'mc_loocv.npz'); dg=load(args.data/'gaussian_evaluation.npz'); ds=load(args.data/'gaussian_selection.npz')
    cvde=load(args.data/'theory_cv_nminus1.npz')
    indices=dict(gen_DE=de['E_gen'].argmin(1),gen_DE_LOOCV_proxy=cvde['E_gen'].argmin(1),
                 acc_leading=de['E_acc'].argmin(1),acc_distribution=ds['E_acc_mean'].argmin(1),
                 gen_MC_oracle=mc['E_gen_mean'].argmin(1),acc_MC_oracle=mc['E_acc_mean'].argmin(1))
    rows=[]
    for method,idx in indices.items():
        for i,j in enumerate(idx):
            row=dict(method=method,sigma=c['sigma'][i],ratio=c['ratio'][i],alpha=c['alpha'][j],lam=c['lam'][j],kappa=c['kappa'][j],at_boundary=bool(j in [0,len(c['lam'])-1]))
            for metric in ['E_gen','E_acc','R2_gen','R2_acc','slope_gen','slope_acc','weight_error']:
                row[metric+'_DE']=de[metric][i,j]
                row[metric+'_MC_mean']=mc[metric+'_mean'][i,j]
                row[metric+'_distribution']=dg[metric+'_mean'][i,j]
            rows.append(row)
    pd.DataFrame(rows).to_csv(args.data/'selected_cuts.csv',index=False)
    # Paired trial-specific CV evaluations, not evaluation at median alpha.
    cv_evaluations={}; ii=np.arange(len(c['sigma']))[None,:]; tt=np.arange(len(cv['selected_index']))[:,None]
    for metric in ['E_gen','E_acc','R2_gen','R2_acc','slope_gen','slope_acc','weight_error']:
        raw=load(args.data/f'mc_raw_{metric}.npz')['values']
        cv_evaluations[metric]=raw[tt,ii,cv['selected_index']]
    np.savez_compressed(args.data/'cv_selected_evaluations.npz',**cv_evaluations)
    np.savez_compressed(args.data/'selected_indices.npz',**indices,sklearn_equivalent_cv=cv['selected_index'])
    plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,'font.family':'DejaVu Sans','mathtext.fontset':'dejavusans'})
    ratio=c['ratio'][1:]; lam=c['lam']; X,Y=np.meshgrid(ratio,lam)
    for category,names in [('error',['E_gen','E_acc']),('r2',['R2_gen','R2_acc']),('slope',['slope_gen','slope_acc'])]:
        fig,axes=plt.subplots(2,2,figsize=(13,10),sharex=True,sharey=True,layout='constrained')
        norm=LogNorm(1e-7,10) if category=='error' else SymLogNorm(linthresh=1,vmin=-1e5,vmax=1) if category=='r2' else LogNorm(1e-3,10)
        for row,estimator in enumerate(['Leading DE','Natural-image MC mean']):
            for col,name in enumerate(names):
                ax=axes[row,col]
                values=de[name] if row==0 else mc[name+'_mean']
                surface=ax.pcolormesh(X,Y,values[1:].T,shading='auto',norm=norm,cmap='viridis' if category!='r2' else 'coolwarm_r',rasterized=True)
                ax.plot(ratio,lam[indices['gen_DE'][1:]],color='white',lw=1.6)
                ax.plot(ratio,lam[indices['acc_leading'][1:]],color='#ff5bce',lw=1.5,ls='--')
                # Previous figures are vertical constant-noise cuts and horizontal fixed-lambda cuts.
                for noise in [.001,.01,.1,1.]: ax.axvline(noise,color='gray',alpha=.5,lw=.7,ls=':')
                ax.axhline(.1,color='gray',lw=.8,ls=':')
                ax.set_xscale('log'); ax.set_yscale('log'); ax.set_ylim(1e-8,1e4)
                ax.set_title(f'{estimator}: {name}')
                if col==0: ax.set_ylabel(r'Regularization $\lambda$')
                if row==1: ax.set_xlabel(r'Noise / signal variance $\sigma^2/S$')
        fig.colorbar(surface,ax=axes,label='E / S' if category=='error' else category,extend='both',shrink=.8)
        fig.suptitle(f'Van Hateren disk teacher | {len(c["sigma"])} × {len(lam)} grid | {int(mc["trials"])} MC trials\nWhite: prediction DE optimum; pink: accentuation leading-DE grid optimum',fontsize=13)
        for ext in ['png','pdf']: fig.savefig(args.figures/f'landscape_{category}.{ext}',dpi=160,bbox_inches='tight')
        plt.close(fig)
    print('Saved selected cuts, paired CV evaluations, and preview figures.',flush=True)


if __name__=='__main__': main()

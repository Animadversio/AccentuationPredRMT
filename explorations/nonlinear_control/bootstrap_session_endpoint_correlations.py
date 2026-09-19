"""Image-resampling uncertainty for session-specific V--control-slope correlations.

Within each monkey, all site/model rows use the same bootstrap resample of held-out
image identities. This preserves shared-stimulus dependence while quantifying finite
held-out-image uncertainty. It does not bootstrap animals, sites, or neural trials.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
TABLE=REPO/'tables/nonlinear_control/biological_validation'
KEY=['subject','monkey','unit','model']
TAUS=[.5,2.,8.,16.]
ENDPOINTS=['control_session','encoding_session_matched','encoding_session']
N_BOOT=2000

sys.path.insert(0,str(HERE))
from build_biological_synopsis import control_session_test_cloud  # noqa: E402
from validate_biology import upstream_modules  # noqa: E402


def spearman_rows(x,y):
    return np.corrcoef(rankdata(x),rankdata(y))[0,1]


def squared_errors(loader,row):
    x_control,y_control,names=control_session_test_cloud(loader,row.monkey,row.unit,row.model)
    brain=loader.load_brain(row.monkey); ui=list(brain['units']).index(row.unit)
    encoding_measured=dict(zip(brain['calibration']['stim'].astype(str),brain['calibration']['resp_z'][:,ui]))
    y_encoding_matched=np.asarray([encoding_measured[str(name)] for name in names],float)
    x_encoding,y_encoding=loader.encoding_cloud(row.monkey,row.unit,row.model,split='test')
    return dict(control_session=np.square(x_control-y_control),
                encoding_session_matched=np.square(x_control-y_encoding_matched),
                encoding_session=np.square(x_encoding-y_encoding))


def main():
    _,loader,_=upstream_modules()
    synopsis=pd.read_parquet(TABLE/'biological_validation_synopsis_v1.parquet')
    synopsis=synopsis[~synopsis.robust_model].sort_values(KEY).reset_index(drop=True)
    errors={endpoint:{} for endpoint in ENDPOINTS}
    for monkey,group in tqdm(list(synopsis.groupby('monkey')),desc='Load paired squared errors'):
        by_endpoint={endpoint:[] for endpoint in ENDPOINTS}
        for row in group.itertuples():
            values=squared_errors(loader,row)
            for endpoint in ENDPOINTS: by_endpoint[endpoint].append(values[endpoint])
        for endpoint in ENDPOINTS:
            lengths={len(x) for x in by_endpoint[endpoint]}; assert len(lengths)==1
            errors[endpoint][monkey]=np.stack(by_endpoint[endpoint])

    rng=np.random.default_rng(20260919)
    mse_draws={endpoint:np.empty((N_BOOT,len(synopsis))) for endpoint in ENDPOINTS}
    for monkey,idx in synopsis.groupby('monkey').groups.items():
        idx=np.asarray(list(idx)); n_by_endpoint={e:errors[e][monkey].shape[1] for e in ENDPOINTS}
        for endpoint in ENDPOINTS:
            n=n_by_endpoint[endpoint]
            counts=rng.multinomial(n,np.full(n,1/n),size=N_BOOT)
            mse_draws[endpoint][:,idx]=counts@errors[endpoint][monkey].T/n

    records=[]
    subsets={'all_200':np.ones(len(synopsis),bool),
             'n50_monkeys_120':synopsis.monkey.isin(['red','paul','venus']).to_numpy()}
    slope=synopsis.control_slope.to_numpy()
    for endpoint in ENDPOINTS:
        for tau in TAUS:
            token=str(int(tau)) if tau.is_integer() else str(tau).replace('.','p')
            trace=synopsis[f'geom_smooth_tau255_{token}__trace_mean'].to_numpy()
            v=mse_draws[endpoint]*trace[None,:]/synopsis.n_train.to_numpy()[None,:]
            for subset,mask in subsets.items():
                for b in range(N_BOOT):
                    records.append(dict(bootstrap=b,endpoint=endpoint,tau_255=tau,subset=subset,
                                        spearman=spearman_rows(v[b,mask],slope[mask])))
    draws=pd.DataFrame(records)
    draws.to_csv(TABLE/'session_endpoint_image_bootstrap_draws.csv.gz',index=False,
                 compression={'method':'gzip','compresslevel':6,'mtime':0})
    summary=draws.groupby(['endpoint','tau_255','subset']).spearman.agg(
        mean='mean',sd='std',q025=lambda x:x.quantile(.025),median='median',q975=lambda x:x.quantile(.975),
        probability_negative=lambda x:np.mean(x<0)).reset_index()
    summary.to_csv(TABLE/'session_endpoint_image_bootstrap_summary.csv',index=False)
    labels={'control_session':'Control session, matched images',
            'encoding_session_matched':'Encoding session, matched images',
            'encoding_session':'Encoding session, all 195 images'}
    colors={'control_session':'#cc6677','encoding_session_matched':'#ddaa33','encoding_session':'#4477aa'}
    fig,axs=plt.subplots(1,2,figsize=(11,4.4),sharey=True)
    for ax,subset in zip(axs,['all_200','n50_monkeys_120']):
        for endpoint in ENDPOINTS:
            d=summary[(summary.endpoint==endpoint)&(summary.subset==subset)]
            ax.plot(d.tau_255,d['mean'],'o-',label=labels[endpoint],color=colors[endpoint])
            ax.fill_between(d.tau_255,d.q025,d.q975,color=colors[endpoint],alpha=.16)
        ax.axhline(0,color='.5',linewidth=.8);ax.set_xscale('log',base=2)
        ax.set_xticks(TAUS,TAUS);ax.set_xlabel('Noise SD × 255')
        ax.set_title('All five monkeys' if subset=='all_200' else 'Only n=50 monkeys: red, paul, venus')
        ax.spines[['top','right']].set_visible(False)
    axs[0].set_ylabel('Image-bootstrap Spearman(V, control slope)')
    axs[0].legend(fontsize=8)
    fig.suptitle('Finite held-out-image uncertainty; non-robust models only\nShared image resampling within each monkey; bands are 95% intervals')
    fig.tight_layout()
    out=REPO/'figures/nonlinear_control/biological_validation/session_endpoint_image_bootstrap.png'
    fig.savefig(out,dpi=180);plt.close(fig)
    print(summary.to_string(index=False))


if __name__=='__main__':
    main()

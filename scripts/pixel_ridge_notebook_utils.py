"""Inspectable pixel-ridge risk paths, numerical selection, and exact LOOCV.

All coordinates are population PCs of the original input; no feature
projection or whitening is applied. Natural spectra require no eigenvectors
for these rotationally invariant Gaussian-design experiments.
"""
from pathlib import Path
import time
import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize_scalar
from tqdm.auto import tqdm
from rmt_core.feature_theory_lib import spectral_feature_de_metrics
from rmt_core.ridge_theory_lib import accentuation_error_theory


def load_problem(root, dataset='vanhateren', d=128, seed=42):
    if dataset in ('vanhateren', 'ffhq'):
        with np.load(Path(root)/f'tables/{dataset}_disk_teacher_spectrum.npz') as z:
            return z['eigenvalues'].astype(float), z['beta_proj'].astype(float)
    if dataset not in ('powerlaw', 'isotropic'):
        raise ValueError('dataset must be vanhateren, ffhq, powerlaw, or isotropic')
    s = np.ones(d) if dataset == 'isotropic' else np.arange(1, d+1.)**-1.5
    s /= s.mean()
    beta = np.random.default_rng(seed).normal(size=d)
    beta /= np.sqrt(np.sum(s*beta**2))
    return s, beta


def kappa_for_alpha(s, n, alpha):
    """Solve on the positive physical branch; alpha=n*lambda."""
    lam = float(alpha)/n
    if lam <= 0:
        raise ValueError('Use strictly positive alpha')
    def equation(k):
        return k*(1-np.sum(s/(s+k))/n)-lam
    return brentq(equation, lam, lam+s.sum()/n+lam,
                  xtol=max(1e-15, lam*1e-10))


def metrics(s, beta, n, sigma, alpha):
    k = kappa_for_alpha(s, n, alpha)
    signal = float(s @ (beta**2))
    m = spectral_feature_de_metrics(s, beta, np.ones_like(s),
                                   0., signal, k, sigma, n)
    # Existing code's conditional-response-noise ratio correction; not full Var(R).
    corrected, _, _ = accentuation_error_theory(
        s, beta, k, sigma, n, include_var_R=True,
        include_ratio_mean_correction=True)
    return dict(alpha=float(alpha), lam=float(alpha/n), kappa=k,
                E_gen=m['gen_error_normalized'], E_acc=m['acc_error_normalized'],
                E_acc_corrected=corrected/signal, R2_gen=m['r2_gen'],
                R2_acc=m['r2_acc'], slope_acc=m['slope_acc'])


def numerical_minimum(alphas, objective):
    """Dense global scan, refine every sampled local minimum in log(alpha).

    Returns the best solution within the supplied finite range, not a proof
    of an unconstrained global minimum. Boundary flags request range expansion.
    """
    x = np.log(np.asarray(alphas, float))
    if len(x) < 3 or np.any(np.diff(x) <= 0):
        raise ValueError('Need at least three increasing positive alphas')
    y = np.asarray([objective(a) for a in alphas])
    if not np.isfinite(y).all():
        raise ValueError('Nonfinite selection objective')
    candidates = [(y[i], x[i]) for i in (0, int(np.argmin(y)), len(x)-1)]
    for i in range(1, len(x)-1):
        if y[i] <= y[i-1] and y[i] <= y[i+1]:
            opt = minimize_scalar(lambda t: objective(np.exp(t)),
                                  bounds=(x[i-1], x[i+1]), method='bounded',
                                  options={'xatol': 1e-10})
            if opt.success:
                candidates.append((opt.fun, opt.x))
    value, log_alpha = min(candidates)
    boundary = (abs(log_alpha-x[0]) < 1e-6 or abs(log_alpha-x[-1]) < 1e-6)
    return float(np.exp(log_alpha)), float(value), bool(boundary), y


def sweep(s, beta, n, ratios, alphas, fixed_lambda=.1,
          acc_objective='E_acc', progress=True):
    if acc_objective not in ('E_acc', 'E_acc_corrected'):
        raise ValueError('Unknown accentuation objective')
    if n < 3 or fixed_lambda <= 0:
        raise ValueError('Require n>=3 and positive fixed lambda')
    rows, paths = [], []
    S = float(s @ beta**2)
    for ratio in tqdm(ratios, disable=not progress, desc='DE noise sweep'):
        sigma = np.sqrt(ratio*S)
        cache = {}
        def risk(a, count):
            key = (float(a), count)
            if key not in cache:
                cache[key] = metrics(s, beta, count, sigma, a)
            return cache[key]
        # LOOCV fits n-1 observations; refit the chosen alpha at n.
        ag, vg, bg, gen_path = numerical_minimum(
            alphas, lambda a: risk(a, n-1)['E_gen'])
        aa, va, ba, acc_path = numerical_minimum(
            alphas, lambda a: risk(a, n)[acc_objective])
        for policy, alpha, boundary in [('fixed', n*fixed_lambda, False),
                                        ('DE_gen_CV', ag, bg),
                                        ('DE_acc_oracle', aa, ba)]:
            rows.append(dict(policy=policy, ratio=ratio, sigma=sigma, S=S,
                             n=n, d=len(s), boundary=boundary,
                             acc_objective=acc_objective, **risk(alpha, n)))
        for j, alpha in enumerate(alphas):
            paths.append(dict(ratio=ratio, sigma=sigma,
                              cv_E_gen=gen_path[j], **risk(alpha, n)))
    return pd.DataFrame(rows), pd.DataFrame(paths)


def exact_loo_path(X, y, alphas):
    """Exact ridge LOOCV with fit_intercept=False, one SVD for all alphas."""
    U, z, VT = np.linalg.svd(X, full_matrices=False)
    uy = U.T@y
    factors = z[:, None]/(z[:, None]**2+alphas[None, :])
    weights = VT.T @ (factors*uy[:, None])
    shrink = z[:, None]**2/(z[:, None]**2+alphas[None, :])
    residual = y[:, None] - U@(shrink*uy[:, None])
    leverage = (U**2)@shrink
    loo = np.mean((residual/(1-leverage))**2, axis=0)
    return weights, loo


def gaussian_acc_risk(s, beta, n, sigma, alpha, normal_draws):
    """Moment-matched Gaussian surrogate, NOT an exact distributional theorem.

    Diagonal covariance matches DE coordinate-wise second moments, including
    design-induced variance. Off-diagonal covariance/higher cumulants are not
    established here. Form N,D from the SAME weight draws to retain dependence
    and positivity of D. Output is normalized E_acc/S and its sampling SE.
    """
    m = metrics(s, beta, n, sigma, alpha)
    k = m['kappa']
    S = float(s @ beta**2)
    mean = s/(s+k)*beta
    variance = (m['E_gen']*S + sigma**2)/n * s/(s+k)**2
    weights = mean + normal_draws*np.sqrt(variance)
    numerator = weights @ beta
    denominator = np.sum(weights**2, axis=1)
    loss = (1-numerator/denominator)**2
    return float(loss.mean()), float(loss.std(ddof=1)/np.sqrt(len(loss)))


def gaussian_acc_selection(s, beta, n, ratios, alphas, draws=1024,
                           validation_draws=4096, seed=123, progress=True):
    """Common-random-number optimization with independent surrogate evaluation.

    Gaussian integration SE is NOT uncertainty about surrogate accuracy.
    Both selection and evaluation are teacher-dependent oracle operations.
    """
    rng = np.random.default_rng(seed)
    normal = rng.normal(size=(draws, len(s)))
    validation = rng.normal(size=(validation_draws, len(s)))
    S = float(s @ beta**2)
    rows, paths = [], []
    for ratio in tqdm(ratios, disable=not progress, desc='Gaussian oracle sweep'):
        sigma = np.sqrt(ratio*S)
        objective = lambda a: gaussian_acc_risk(s,beta,n,sigma,a,normal)[0]
        alpha, risk, boundary, path = numerical_minimum(alphas, objective)
        value, se = gaussian_acc_risk(s,beta,n,sigma,alpha,validation)
        rows.append(dict(policy='Gaussian_acc_oracle', ratio=ratio, sigma=sigma,
                         S=S, n=n, d=len(s), boundary=boundary,
                         gaussian_selection_risk=risk, gaussian_validation_risk=value,
                         gaussian_validation_se=se, **metrics(s,beta,n,sigma,alpha)))
        paths.extend(dict(ratio=ratio, sigma=sigma, alpha=a,
                          gaussian_selection_risk=v) for a,v in zip(alphas,path))
    return pd.DataFrame(rows), pd.DataFrame(paths)


def monte_carlo(s, beta, n, selected, alphas, trials=10, seed=17, progress=True):
    """Gaussian pixel ridge: three fixed DE policies and empirical LOOCV.

    Acc-oracle alpha comes from DE, never chosen using evaluated MC losses.
    For zero-mean synthetic data intercept is fixed at zero.
    """
    rng = np.random.default_rng(seed)
    S = float(s@beta**2)
    rows = []
    for trial in tqdm(range(trials), disable=not progress, desc='Gaussian ridge trials'):
        X = rng.normal(size=(n, len(s)))*np.sqrt(s)
        eps = rng.normal(size=n)
        # Share design/noise across sigma; factorization reused within each sigma.
        for ratio, group in selected.groupby('ratio', sort=True):
            sigma = np.sqrt(ratio*S)
            grid = np.unique(np.r_[alphas, group.alpha.to_numpy()])
            W, loo = exact_loo_path(X, X@beta+sigma*eps, grid)
            # CV selection is restricted to the user-specified grid.
            allowed = np.searchsorted(grid, alphas)
            chosen = allowed[np.argmin(loo[allowed])]
            policies = [(r.policy, np.searchsorted(grid, r.alpha))
                        for r in group.itertuples()]
            policies.append(('empirical_RidgeCV', chosen))
            for policy, j in policies:
                w = W[:, j]
                N, D = beta@w, w@w
                rows.append(dict(trial=trial, policy=policy, ratio=ratio,
                    sigma=sigma, alpha=grid[j], lam=grid[j]/n,
                    E_gen=np.sum(s*(w-beta)**2)/S,
                    E_acc=(1-N/D)**2, R2_acc=1-(D/N-1)**2,
                    slope_acc=N/D))
    return pd.DataFrame(rows)


def plot_policies(summary):
    import matplotlib.pyplot as plt
    policies = ['fixed', 'DE_gen_CV', 'DE_acc_oracle']
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for j, policy in enumerate(policies):
        part = summary[summary.policy == policy].sort_values('sigma')
        x = part.sigma.to_numpy()
        for metric, style, label in [('E_gen', '-', 'E gen / S'),
                                     ('E_acc', '--', 'E acc / S')]:
            # Clip only for log display; raw zero values stay in tables.
            axes[0,j].plot(x, np.maximum(part[metric], 1e-16), style, label=label)
        axes[0,j].set(yscale='log', title=policy, ylabel='normalized error')
        if 'acc_objective' in part and (part.acc_objective == 'E_acc_corrected').any():
            axes[0,j].plot(x, np.where(part.E_acc_corrected > 0,
                                      part.E_acc_corrected, np.nan), ':',
                            label='E acc / S (partial correction)')
        axes[0,j].legend()
        axes[1,j].plot(x, part.lam, label='lambda')
        axes[1,j].plot(x, part.kappa, '--', label='kappa')
        axes[1,j].set(yscale='log', ylabel='regularization')
        axes[1,j].legend()
        S = float(part.S.iloc[0])
        for ax in axes[:,j]:
            positive = x[x>0]
            ax.set_xscale('symlog', linthresh=positive.min() if len(positive) else 1)
            ax.set_xlim(0, x.max()*1.05 if x.max() > 0 else 1)
            ax.set_xlabel('response noise sigma')
            # An explicit twin shares the exact symlog transform, including
            # its linear zero region (secondary_xaxis does not inherit this).
            top = ax.twiny()
            top.set_xscale('symlog', linthresh=positive.min() if len(positive) else 1)
            top.set_xlim(ax.get_xlim())
            ax.callbacks.connect('xlim_changed', lambda parent, top=top: top.set_xlim(parent.get_xlim()))
            top.set_xlabel('noise / signal variance')
            ticks = [r for r in [0, 1e-6, 1e-4, 1e-2, 1, 10, 100]
                     if r <= x.max()**2/S*1.1]
            top.set_xticks(np.sqrt(np.asarray(ticks)*S))
            top.set_xticklabels([f'{r:g}' for r in ticks])
            ax.grid(alpha=.2)
    fig.tight_layout()
    return fig, axes

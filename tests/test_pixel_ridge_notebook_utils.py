import numpy as np
import pandas as pd
from scripts import pixel_ridge_notebook_utils as u
from scripts.plot_pixel_ridge_estimator_comparison import add_mc_variability


def test_mc_variability_keeps_mean_not_median():
    raw=pd.DataFrame(dict(policy=['fixed']*3,ratio=[0.]*3,alpha=[1.]*3,
                          trial=[0,1,2],E_gen=[1.,2.,9.],E_acc=[0.,1.,8.]))
    table=pd.DataFrame([dict(policy='fixed',ratio=0.,alpha=1.,E_gen_mc=4.,E_acc_mc=3.)])
    result=add_mc_variability(table,raw).iloc[0]
    assert result.E_acc_mc==3 and result.E_acc_mc_median==1
    np.testing.assert_allclose(result.E_acc_mc_std,np.std([0.,1.,8.],ddof=1))
    np.testing.assert_allclose([result.E_acc_mc_q10,result.E_acc_mc_q90],np.quantile([0.,1.,8.],[.1,.9]))
    assert result.trials==3


def test_loo_matches_explicit_refits():
    rng = np.random.default_rng(21)
    X, y = rng.normal(size=(12, 5)), rng.normal(size=12)
    alphas = np.array([.01, 1., 100.])
    weights, loo = u.exact_loo_path(X, y, alphas)
    for j, a in enumerate(alphas):
        expected = np.linalg.solve(X.T@X+a*np.eye(5), X.T@y)
        np.testing.assert_allclose(weights[:, j], expected, atol=1e-12)
        residuals = []
        for i in range(len(y)):
            keep = np.arange(len(y)) != i
            w = np.linalg.solve(X[keep].T@X[keep]+a*np.eye(5), X[keep].T@y[keep])
            residuals.append((y[i]-X[i]@w)**2)
        np.testing.assert_allclose(loo[j], np.mean(residuals), rtol=1e-12)


def test_physical_kappa_and_gaussian_moments():
    s = np.arange(1., 10.)**-1.5
    beta = np.arange(1., 10.)/10
    for n in [5, 20]:
        k = u.kappa_for_alpha(s, n, .3)
        np.testing.assert_allclose(k*(1-np.sum(s/(s+k))/n), .3/n)
        m = u.metrics(s,beta,n,.5,.3)
        S = s@beta**2
        mean = s/(s+k)*beta
        v = (m['E_gen']*S+.5**2)/n*s/(s+k)**2
        np.testing.assert_allclose(beta@mean/(mean@mean+sum(v)),m['slope_acc'])
        draws = np.random.default_rng(2).normal(size=(100, len(s)))
        risk,se = u.gaussian_acc_risk(s,beta,n,.5,.3,draws)
        w = mean+np.sqrt(v)*draws
        expected = (1-w@beta/np.sum(w*w,axis=1))**2
        np.testing.assert_allclose(risk,expected.mean())
        assert se >= 0


def test_refined_minimum_and_boundary():
    alphas = np.logspace(-3,3,31)
    a, val, boundary, _ = u.numerical_minimum(alphas, lambda a:(np.log(a)-.123)**2)
    assert abs(np.log(a)-.123) < 1e-6 and not boundary
    assert val < 1e-12
    _, _, boundary, _ = u.numerical_minimum(alphas, lambda a: a)
    assert boundary

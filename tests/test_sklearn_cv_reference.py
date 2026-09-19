import numpy as np
from sklearn.linear_model import RidgeCV


def test_per_target_loocv_is_independent_and_matches_explicit_loo():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(13, 20))
    y = rng.normal(size=(13, 3))
    alphas = np.array([.01, 1., 100.])
    fit = RidgeCV(alphas=alphas, cv=None, fit_intercept=True,
                  gcv_mode='eigen', alpha_per_target=True).fit(X, y)
    for j in range(3):
        single = RidgeCV(alphas=alphas, cv=None, fit_intercept=True,
                         gcv_mode='eigen').fit(X, y[:, j])
        assert fit.alpha_[j] == single.alpha_
        np.testing.assert_allclose(fit.coef_[j], single.coef_, atol=1e-11)
        scores = []
        for alpha in alphas:
            errors = []
            for i in range(len(X)):
                keep = np.arange(len(X)) != i
                xx, yy = X[keep], y[keep, j]
                xm, ym = xx.mean(0), yy.mean()
                xc = xx-xm
                w = np.linalg.solve(xc.T@xc+alpha*np.eye(X.shape[1]), xc.T@(yy-ym))
                errors.append((y[i, j]-((X[i]-xm)@w+ym))**2)
            scores.append(np.mean(errors))
        assert fit.alpha_[j] == alphas[np.argmin(scores)]
        np.testing.assert_allclose(-fit.best_score_[j], min(scores), rtol=1e-9)

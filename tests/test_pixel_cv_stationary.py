import numpy as np
from scripts.pixel_cv_stationary import stationary_table, moments
from scripts.pixel_ridge_notebook_utils import metrics


def test_stationary_formula_matches_direct_metrics_and_local_minimum():
    s = np.geomspace(3, .01, 40)
    b = np.random.default_rng(3).normal(size=40)
    n = 25
    t = stationary_table(s, b, n, [2., 5.])
    for r in t.itertuples():
        assert r.status == 'interior_minimum'
        assert r.better_than_boundaries
        m = metrics(s, b, n, r.sigma, n*r.lam)
        for key in ['E_acc', 'R2_acc', 'slope_acc', 'E_gen']:
            np.testing.assert_allclose(getattr(r, key), m[key], atol=1e-9)
        def risk(k):
            _, b12, _, _, _, d2, _ = moments(s, b, k, n)
            return n/(n-d2)*(k*k*b12+r.sigma**2)-r.sigma**2
        assert risk(r.kappa) <= risk(r.kappa*.999)
        assert risk(r.kappa) <= risk(r.kappa*1.001)


def test_isotropic_stationary_calibrates_accentuation():
    t = stationary_table(np.ones(10), np.ones(10), 30, [1., 2.])
    np.testing.assert_allclose(t.z_acc, 0, atol=1e-12)
    np.testing.assert_allclose(t.slope_acc, 1, atol=1e-12)


def test_boundary_is_not_filled_with_stationary_formula():
    t = stationary_table(np.ones(10), np.ones(10), 30, [0.])
    assert t.status.iloc[0] == 'no_interior_minimum'
    assert np.isnan(t.E_acc.iloc[0])

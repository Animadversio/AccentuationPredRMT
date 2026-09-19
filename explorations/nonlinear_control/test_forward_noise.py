import numpy as np
from forward_noise import inner_statistics, step_statistics, covariance_pseudovalues


def test_unbiased_inner_square_and_variance_identity():
    d = np.array([[[1., 4.], [2., -1.], [3., 2.]]])
    smooth, neighborhood = inner_statistics(d)
    pair_mean = np.mean([d[:, i]*d[:, j] for i in range(3) for j in range(3) if i!=j], axis=0)
    np.testing.assert_allclose(smooth, pair_mean)
    np.testing.assert_allclose(neighborhood-smooth, d.var(1, ddof=1))


def test_gaussian_polynomial_targets_by_exact_quadrature():
    # Tensor quadrature integrates the finite-degree polynomials exactly.
    from numpy.polynomial.hermite import hermgauss
    nodes, w = hermgauss(10); nodes *= np.sqrt(2); w /= np.sqrt(np.pi)
    x, tau, h = .7, .3, 1e-3
    v, z = nodes[:, None], nodes[None, :]
    for power in [1, 2, 3]:
        d = ((x+tau*z+h*v)**power-(x+tau*z-h*v)**power)/(2*h)
        mean_z = d @ w
        smooth = w @ (mean_z**2)
        neighborhood = w @ ((d*d) @ w)
        if power == 1:
            np.testing.assert_allclose([smooth, neighborhood], [1., 1.])
        elif power == 2:
            np.testing.assert_allclose([smooth, neighborhood], [4*x*x, 4*(x*x+tau*tau)])
        else:
            # Finite-h cubic directional difference includes h²v³.
            a = 3*(x*x+tau*tau)
            np.testing.assert_allclose(smooth, a*a+6*a*h*h+15*h**4)
    plus, minus = (x+tau*nodes)**2, (x-tau*nodes)**2
    odd, step, even = step_statistics(plus, minus, x*x, tau)
    np.testing.assert_allclose([w@odd, w@step, w@even], [4*x*x,4*x*x+3*tau*tau,3*tau*tau])


def test_antithetic_covariance_debiasing_and_jackknife():
    p=np.array([[1.,3.], [2.,4.], [-1.,2.], [4.,1.]])
    m=np.array([[2.,0.], [-1.,2.], [3.,1.], [0.,3.]])
    base=np.array([.2,.4]);tau=.3
    variance,drift=covariance_pseudovalues(p,m,base,tau)
    _,step,_=step_statistics(p,m,base,tau)
    b=(p+m-2*base)/(2*tau)
    expected_drift=np.mean([b[i]*b[j] for i in range(4) for j in range(4) if i!=j],axis=0)
    np.testing.assert_allclose(drift.mean(0),expected_drift)
    np.testing.assert_allclose(variance.mean(0),step.mean(0)-expected_drift)
    np.testing.assert_allclose(variance+drift,step)

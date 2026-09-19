"""Algebraic reference checks for the proposed scientific measurement."""
import numpy as np
import torch
from geometry import pc_gradient_power, batched_pc_gradient_power, trace_contributions


def test_linear_limit_and_nondiagonal_metric():
    rng = np.random.default_rng(7)
    f = rng.normal(size=(4, 6))
    a = rng.normal(size=(6, 6))
    c = f @ (a @ a.T) @ f.T
    s, u = np.linalg.eigh(c)
    x = torch.randn(1, 6, dtype=torch.double, requires_grad=True)
    scores = x @ torch.tensor(f.T @ u)
    q = pc_gradient_power(scores, x, range(4))
    np.testing.assert_allclose(q, np.diag(u.T @ f @ f.T @ u))
    for kappa in [0.1, 1., 10.]:
        inv = np.linalg.inv(c + kappa*np.eye(4))
        reference = np.trace(f @ f.T @ inv @ inv @ c)
        np.testing.assert_allclose(trace_contributions(s, q, [kappa]).sum(), reference)


def test_rgb_chain_rule_and_nonlinear_gradient():
    x = torch.tensor([[[[2.]], [[3.]], [[4.]]]], requires_grad=True)
    scores = torch.stack((x.square().sum(), x.sin().sum()))[None]
    std = np.array([.2, .3, .4])
    q = pc_gradient_power(scores, x, [0, 1], std)
    expected = [(2*np.array([2., 3., 4.])/std)**2,
                (np.cos([2., 3., 4.])/std)**2]
    np.testing.assert_allclose(q, np.sum(expected, axis=1), rtol=1e-6)
    qb = batched_pc_gradient_power(scores, x, [0, 1], std)
    np.testing.assert_allclose(qb, q, rtol=1e-6)
    shaped_std = torch.as_tensor(std).reshape(1, 3, 1, 1)
    np.testing.assert_allclose(
        batched_pc_gradient_power(scores, x, [0, 1], shaped_std), q, rtol=1e-6
    )


def test_matched_df2_comparison_is_feature_scale_invariant():
    from compare_resnets import kappa_at_df2
    s, q = np.array([10., 2., .1]), np.array([5., 4., 3.])
    k = kappa_at_df2(s, 1.5)
    np.testing.assert_allclose(np.sum((s/(s+k))**2), 1.5)
    scale = 49.  # Multiplying features and Jacobians by 7 scales s and q by 49.
    k2 = kappa_at_df2(s*scale, 1.5)
    np.testing.assert_allclose(k2, k*scale)
    np.testing.assert_allclose(trace_contributions(s, q, [k]),
                               trace_contributions(s*scale, q*scale, [k2]))

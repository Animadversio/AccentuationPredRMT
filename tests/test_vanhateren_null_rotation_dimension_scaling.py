import numpy as np

from scripts.validate_vanhateren_null_rotation_dimension_scaling import (
    isotropic_ridge_axis_draw,
    real_wishart_eigenvalues,
)


def test_real_wishart_eigenvalues_have_correct_trace_mean():
    rng = np.random.default_rng(12)
    n, p = 80, 20
    traces = np.asarray([
        np.sum(real_wishart_eigenvalues(n, p, rng)) for _ in range(2000)])
    # E tr(X^T X/n) = p and Var tr(X^T X/n) = 2p/n.
    np.testing.assert_allclose(np.mean(traces), p, atol=0.08)
    np.testing.assert_allclose(np.var(traces), 2 * p / n, rtol=0.12)


def test_axis_reduction_matches_direct_small_ridge_moments():
    rng_axis = np.random.default_rng(31)
    rng_direct = np.random.default_rng(32)
    n, p, n_trials = 50, 8, 5000
    signal = 2.0
    teacher_norm = np.sqrt(signal)
    sigma = np.sqrt(0.1 * signal)
    alpha = 1.0
    ridge_lambda = alpha / n
    axis = np.empty((n_trials, 3))
    direct = np.empty((n_trials, 3))
    for trial in range(n_trials):
        spectrum = real_wishart_eigenvalues(n, p, rng_axis)
        teacher_pc = rng_axis.standard_normal(p)
        teacher_pc /= np.linalg.norm(teacher_pc)
        noise = rng_axis.standard_normal(p)
        shrink = spectrum / (spectrum + ridge_lambda)
        noise_pc = (
            sigma / np.sqrt(n) * np.sqrt(spectrum)
            / (spectrum + ridge_lambda))
        fitted_pc = teacher_norm * shrink * teacher_pc + noise_pc * noise
        w0 = teacher_pc @ fitted_pc
        norm_sq = fitted_pc @ fitted_pc
        axis[trial] = (
            w0, norm_sq,
            (norm_sq - 2 * teacher_norm * w0 + signal) / signal)

        design = rng_direct.standard_normal((n, p))
        response = (
            teacher_norm * design[:, 0]
            + sigma * rng_direct.standard_normal(n))
        fitted = np.linalg.solve(
            design.T @ design + alpha * np.eye(p), design.T @ response)
        direct[trial] = (
            fitted[0], fitted @ fitted,
            ((fitted[0] - teacher_norm) ** 2 + fitted[1:] @ fitted[1:])
            / signal)

    combined_se = np.sqrt(
        np.var(axis, axis=0) / n_trials
        + np.var(direct, axis=0) / n_trials)
    assert np.all(np.abs(np.mean(axis, axis=0) - np.mean(
        direct, axis=0)) < 3.5 * combined_se)


def test_isotropic_axis_helper_uses_pre_scaled_noise():
    rng = np.random.default_rng(3)
    spectrum = np.ones(10)
    w0, norm_sq = isotropic_ridge_axis_draw(
        spectrum, teacher_norm=2.0, sigma_over_sqrt_n=0.0,
        ridge_lambda=0.2, rng=rng)
    np.testing.assert_allclose(w0, 2.0 / 1.2)
    np.testing.assert_allclose(norm_sq, (2.0 / 1.2) ** 2)

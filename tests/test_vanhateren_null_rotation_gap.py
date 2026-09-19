import numpy as np

from scripts.analyze_vanhateren_null_rotation_gap import (
    delta_method_metrics,
    gaussian_quadratic_moments,
    nonlinear_metrics,
    ratio_of_moments_metrics,
)


def test_gaussian_quadratic_moments_match_monte_carlo():
    rng = np.random.default_rng(11)
    mean = np.asarray([0.7, -0.2, 0.4])
    variance = 0.3
    metric = np.asarray([
        [2.0, 0.2, 0.0],
        [0.2, 0.8, -0.1],
        [0.0, -0.1, 1.3],
    ])
    overlap = np.asarray([0.4, -0.8, 0.3])
    moments = gaussian_quadratic_moments(
        mean, variance, metric, overlap)
    samples = mean + np.sqrt(variance) * rng.standard_normal((300_000, 3))
    numerator = samples @ overlap
    denominator = np.einsum('bi,ij,bj->b', samples, metric, samples)
    np.testing.assert_allclose(
        np.mean(numerator), moments['mean_numerator'], rtol=8e-3)
    np.testing.assert_allclose(
        np.mean(denominator), moments['mean_denominator'], rtol=8e-3)
    np.testing.assert_allclose(
        np.var(numerator), moments['variance_numerator'], rtol=1.5e-2)
    np.testing.assert_allclose(
        np.var(denominator), moments['variance_denominator'], rtol=1.5e-2)
    np.testing.assert_allclose(
        np.cov(numerator, denominator, ddof=0)[0, 1],
        moments['covariance_numerator_denominator'], rtol=2e-2)


def test_ratio_of_moments_matches_definition():
    metrics = ratio_of_moments_metrics(3.0, 4.0)
    assert metrics['slope_acc'] == 0.75
    assert metrics['acc_error_normalized'] == 0.25 ** 2
    assert metrics['r2_acc'] == 1.0 - (1.0 - 4.0 / 3.0) ** 2


def test_delta_method_is_accurate_for_concentrated_ratios():
    mean = np.asarray([4.0, 1.0])
    variance = 1e-4
    metric = np.asarray([[1.2, 0.1], [0.1, 0.8]])
    overlap = np.asarray([1.0, 0.2])
    moments = gaussian_quadratic_moments(
        mean, variance, metric, overlap)
    delta = delta_method_metrics(moments)
    rng = np.random.default_rng(7)
    samples = mean + np.sqrt(variance) * rng.standard_normal((250_000, 2))
    numerator = samples @ overlap
    denominator = np.einsum('bi,ij,bj->b', samples, metric, samples)
    observed = nonlinear_metrics(numerator, denominator)
    np.testing.assert_allclose(
        np.mean(observed['slope_acc']), delta['slope_acc'], rtol=2e-4)
    np.testing.assert_allclose(
        np.mean(observed['acc_error_normalized']),
        delta['acc_error_normalized'], rtol=2e-4)

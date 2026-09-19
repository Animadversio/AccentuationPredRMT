import numpy as np

from scripts.validate_vanhateren_p500_null_families import (
    build_p500_null_families,
    evaluate_maps,
)


def _problem(d=3000):
    eigenvalues = np.geomspace(10.0, 1e-5, d)
    rng = np.random.default_rng(4)
    beta = rng.standard_normal(d)
    beta /= np.sqrt(np.sum(eigenvalues * beta ** 2))
    return eigenvalues, beta


def test_p500_families_preserve_prediction_and_vary_control():
    eigenvalues, beta = _problem(10_000)
    families, metadata, diagnostics = build_p500_null_families(
        eigenvalues, beta, n_angles=7, n_random=12,
        n_rotating=300, seed=8)
    assert families['continuous'].shape == (7, 500)
    assert families['random'].shape == (12, 500)
    assert diagnostics['max_b_null_residual'] < 1e-10
    assert diagnostics['max_c_null_residual'] < 1e-8
    assert np.ptp(np.sum(families['continuous'], axis=1)) > 1e4
    assert np.ptp(np.sum(families['random'], axis=1)) > 1e4
    np.testing.assert_allclose(families['continuous'][:, 0], diagnostics['g0'])
    np.testing.assert_allclose(families['random'][:, 0], diagnostics['g0'])
    assert np.ptp(metadata['random_mean_tail_fraction']) > 0.2


def test_map_evaluation_reuses_identical_prediction_draws():
    rng = np.random.default_rng(5)
    n_trials, p = 40, 6
    teacher_axis = 1.0 + 0.1 * rng.standard_normal((2, n_trials))
    norm_sq = teacher_axis ** 2 + 0.3
    null_direction = rng.standard_normal((2, n_trials, p - 1))
    null_direction /= np.linalg.norm(null_direction, axis=2, keepdims=True)
    sufficient = {
        'teacher_axis': teacher_axis,
        'norm_sq': norm_sq,
        'null_norm_sq': norm_sq - teacher_axis ** 2,
        'null_direction_sq': null_direction ** 2,
    }
    weights = np.vstack((np.ones(p), np.arange(1, p + 1)))
    metrics = evaluate_maps(weights, sufficient, signal=1.0, g0=1.0)
    assert metrics['slope_acc'].shape == (2, 2, n_trials)
    assert np.ptp(np.mean(metrics['slope_acc'], axis=2), axis=0).min() > 0

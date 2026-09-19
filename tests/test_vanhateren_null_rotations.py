import numpy as np

from scripts.validate_vanhateren_null_rotations import (
    build_givens_family,
    build_random_family,
    common_de_settings,
    de_metrics_for_maps,
)


def _problem(d=400):
    eigenvalues = np.geomspace(10.0, 1e-4, d)
    rng = np.random.default_rng(3)
    beta = rng.standard_normal(d)
    beta /= np.sqrt(np.sum(eigenvalues * beta ** 2))
    return eigenvalues, beta


def test_givens_family_preserves_prediction_and_teacher_overlap():
    eigenvalues, beta = _problem()
    maps, rows, plane = build_givens_family(
        eigenvalues, beta, p=6, n_angles=7, seed=7)
    assert maps.shape == (7, 6, 6)
    assert plane['max_c_invariance_error'] < 1e-10
    assert plane['max_h_invariance_error'] < 1e-10
    assert plane['max_q_invariance_error'] < 1e-8
    assert rows[-1]['trace_g'] > rows[0]['trace_g']


def test_random_family_preserves_prediction_but_varies_control_geometry():
    eigenvalues, beta = _problem()
    maps, rows, pool, diagnostics = build_random_family(
        eigenvalues, beta, p=6, n_maps=12, pool_size=40, seed=9)
    assert maps.shape == (12, 6, 6)
    assert len(pool) == 40
    assert diagnostics['max_c_invariance_error'] < 1e-10
    assert diagnostics['max_h_invariance_error'] < 1e-10
    assert np.std([row['trace_g'] for row in rows]) > 0
    assert np.std([row['q_null_norm'] for row in rows]) > 0


def test_de_prediction_metrics_are_identical_across_null_rotations():
    eigenvalues, beta = _problem()
    maps, _, _ = build_givens_family(
        eigenvalues, beta, p=6, n_angles=5, seed=11)
    signal = float(np.sum(eigenvalues * beta ** 2))
    settings = common_de_settings(
        p=6, signal=signal, ratios=np.asarray([0.1, 1.0]), n=100,
        alphas=np.logspace(-4, 3, 41))
    metrics = de_metrics_for_maps(maps, signal, settings, n=100)
    for name in ('gen_error_normalized', 'r2_gen', 'slope_gen'):
        np.testing.assert_allclose(
            metrics[name], np.broadcast_to(metrics[name][0], metrics[name].shape),
            rtol=1e-10)
    assert np.max(np.ptp(metrics['r2_acc'], axis=0)) > 1e-3

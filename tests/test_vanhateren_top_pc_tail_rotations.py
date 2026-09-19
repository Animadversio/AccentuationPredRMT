import numpy as np

from scripts.validate_vanhateren_top_pc_tail_rotations import (
    build_top_pc_tail_family,
    feature_rows_pc,
    prediction_de_components,
    theory_metrics,
)


def _problem(d=1200):
    eigenvalues = np.geomspace(20.0, 1e-5, d)
    rng = np.random.default_rng(3)
    beta = rng.standard_normal(d)
    beta /= np.sqrt(np.sum(eigenvalues * beta ** 2))
    return eigenvalues, beta


def test_top_pc_endpoint_and_joint_invariance():
    eigenvalues, beta = _problem()
    fractions = np.asarray([0.0, 1e-4, 0.2, 1.0])
    family = build_top_pc_tail_family(
        eigenvalues, beta, 30, fractions, 0.6, seed=8)
    np.testing.assert_allclose(
        family['g_matrices'][0], np.eye(30), atol=2e-10)
    assert family['tail_b_residual'] < 1e-10
    assert family['tail_c_residual'] < 1e-8
    assert family['trace_g'][-1] > 100 * family['trace_g'][0]
    rows = feature_rows_pc(
        family, eigenvalues, np.asarray([0, 15, 29]),
        np.asarray([0.0, 1.0]))
    for output_index, row_index in enumerate((0, 15, 29)):
        expected = np.zeros(len(eigenvalues))
        expected[row_index] = 1.0
        np.testing.assert_allclose(rows[output_index, 0], expected, atol=1e-10)

    all_rows = feature_rows_pc(
        family, eigenvalues, np.arange(30), fractions)
    b = np.sqrt(eigenvalues) * beta
    c_constraint = beta / np.sqrt(eigenvalues)
    for fraction_index in range(len(fractions)):
        a_map = all_rows[:, fraction_index] * np.sqrt(eigenvalues)[None, :]
        np.testing.assert_allclose(
            a_map @ a_map.T, np.diag(eigenvalues[:30]), atol=2e-10)
        np.testing.assert_allclose(
            a_map @ b, eigenvalues[:30] * beta[:30], atol=2e-10)
        np.testing.assert_allclose(
            a_map @ c_constraint, beta[:30], atol=2e-10)


def test_theory_prediction_is_flat_but_accentuation_changes():
    eigenvalues, beta = _problem()
    family = build_top_pc_tail_family(
        eigenvalues, beta, 30, np.asarray([0.0, 0.1, 1.0]),
        0.6, seed=9)
    setting = {
        'sigma': 0.1,
        'kappa': 0.02,
        'noise_signal_ratio': 0.01,
        'alpha_cv': 1.0,
    }
    components = prediction_de_components(family, setting, n=200)
    assert 0 <= components['r2_gen'] <= 1
    metrics = theory_metrics(family, [setting], n=200)
    assert np.ptp(metrics['r2_gen'][:, 0]) == 0
    assert metrics['r2_acc'][0, 0] > metrics['r2_acc'][-1, 0]
    assert metrics['slope_acc'][0, 0] > metrics['slope_acc'][-1, 0]

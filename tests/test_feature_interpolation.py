import numpy as np

from scripts.validate_ffhq_feature_interpolation import make_feature_problem


def test_feature_interpolation_endpoints_and_hard_projection():
    eigenvalues = np.array([4.0, 2.0, 1.0, 0.5, 0.25])
    beta = np.array([1.0, -0.5, 0.25, 0.1, -0.2])

    whiten = make_feature_problem(
        'coupled', 0.0, eigenvalues, beta, cutoff=2, taper_power=2.0)
    assert np.allclose(whiten['feature_eigenvalues'], 1.0)
    assert np.allclose(whiten['backprop_weights'], 1.0 / eigenvalues)
    assert np.allclose(
        whiten['feature_teacher'], beta * np.sqrt(eigenvalues))

    pca = make_feature_problem(
        'power', 1.0, eigenvalues, beta, cutoff=2, taper_power=2.0)
    assert np.allclose(pca['feature_eigenvalues'], eigenvalues)
    assert np.allclose(pca['backprop_weights'], 1.0)
    assert np.allclose(pca['feature_teacher'], beta)

    soft = make_feature_problem(
        'coupled', 1.0, eigenvalues, beta, cutoff=2, taper_power=2.0)
    expected_gate = 1.0 / (1.0 + (eigenvalues[1] / eigenvalues) ** 2)
    assert np.allclose(soft['backprop_weights'], expected_gate)
    assert np.allclose(soft['feature_eigenvalues'], expected_gate * eigenvalues)

    hard = make_feature_problem(
        'hard', 1.0, eigenvalues, beta, cutoff=2, taper_power=2.0)
    total_signal = np.sum(eigenvalues * beta ** 2)
    retained_signal = np.sum(eigenvalues[:2] * beta[:2] ** 2)
    assert hard['feature_dim'] == 2
    assert np.allclose(hard['feature_eigenvalues'], eigenvalues[:2])
    assert np.isclose(
        hard['residual_signal'], total_signal - retained_signal)

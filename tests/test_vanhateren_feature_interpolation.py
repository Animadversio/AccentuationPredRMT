import numpy as np

from scripts.validate_vanhateren_disk_teacher import SPECTRUM_PATH
from scripts.validate_vanhateren_feature_interpolation import build_problems


def test_vanhateren_feature_family_endpoints_and_cutoff():
    with np.load(SPECTRUM_PATH) as data:
        eigenvalues = np.asarray(data['eigenvalues'], dtype=float)
        beta = np.asarray(data['beta_proj'], dtype=float)
    problems = build_problems(eigenvalues, beta, cutoff=100, taper_power=2.0)

    whitening_power = problems['power_t000']
    whitening_coupled = problems['coupled_t000']
    np.testing.assert_allclose(
        whitening_power['feature_eigenvalues'],
        whitening_coupled['feature_eigenvalues'])
    np.testing.assert_allclose(
        whitening_power['backprop_weights'],
        whitening_coupled['backprop_weights'])

    full_pca = problems['power_t100']
    np.testing.assert_allclose(full_pca['backprop_weights'], 1.0)
    np.testing.assert_allclose(full_pca['feature_eigenvalues'], eigenvalues)

    soft_top = problems['coupled_t100']
    assert soft_top['backprop_weights'][-1] < 1e-7
    assert soft_top['backprop_weights'][0] > 0.99

    hard_top = problems['hard_top100']
    assert hard_top['feature_dim'] == 100
    assert hard_top['retained_signal_fraction'] > 0.999

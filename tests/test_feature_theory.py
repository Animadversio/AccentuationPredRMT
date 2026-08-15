import numpy as np

from rmt_core import (
    SpectrumKappa,
    accentuation_alignment,
    generalization_r2_theory,
    spectral_feature_de_metrics,
)


def test_full_pca_reduces_to_existing_ridge_theory():
    eigenvalues = np.asarray([3.0, 1.2, 0.4, 0.08])
    teacher = np.asarray([0.5, -0.2, 0.7, 0.1])
    n = 20
    sigma = 0.3
    kappa = SpectrumKappa(eigenvalues, len(eigenvalues) / n)(0.05)
    signal = float(np.sum(eigenvalues * teacher ** 2))

    feature = spectral_feature_de_metrics(
        eigenvalues, teacher, np.ones_like(eigenvalues), 0.0, signal,
        kappa, sigma, n)
    r2_gen, gen_error, _ = generalization_r2_theory(
        eigenvalues, teacher, kappa, sigma, n)
    alignment, _, _ = accentuation_alignment(
        eigenvalues, teacher, kappa, sigma, n)

    assert np.isclose(feature['gen_error'], gen_error)
    assert np.isclose(feature['r2_gen'], r2_gen)
    assert np.isclose(feature['slope_acc'], alignment)


def test_truncation_adds_irreducible_signal_error():
    eigenvalues = np.asarray([4.0, 1.0])
    teacher = np.asarray([1.0, 2.0])
    residual = 0.5
    total = float(np.sum(eigenvalues * teacher ** 2) + residual)
    metrics = spectral_feature_de_metrics(
        eigenvalues, teacher, np.ones(2), residual, total,
        kappa=0.2, sigma_noise=0.1, n=100)
    assert metrics['gen_error'] >= residual
    assert np.isclose(
        metrics['effective_noise_variance'], 0.1 ** 2 + residual)


def test_whitening_changes_backprop_but_not_feature_covariance():
    input_eigenvalues = np.asarray([4.0, 0.25])
    beta = np.asarray([1.0, 1.0])
    gains = input_eigenvalues ** -0.5
    feature_eigenvalues = gains ** 2 * input_eigenvalues
    feature_teacher = beta / gains
    backprop_weights = gains ** 2
    signal = float(np.sum(input_eigenvalues * beta ** 2))
    metrics = spectral_feature_de_metrics(
        feature_eigenvalues, feature_teacher, backprop_weights, 0.0,
        signal, kappa=0.3, sigma_noise=0.5, n=20)
    assert np.allclose(feature_eigenvalues, 1.0)
    assert not np.isclose(metrics['slope_gen'], metrics['slope_acc'])

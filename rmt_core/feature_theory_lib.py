"""Deterministic equivalents for ridge on fixed linear spectral features.

Regression is performed on features ``z = A x`` while accentuation follows the
pixel gradient ``A.T @ theta_hat``.  In feature coordinates, prediction is
controlled by ``C = A Sigma_x A.T`` and pixel backpropagation by
``G = A A.T``.  This module covers the commuting/spectral case where both are
diagonal in the same feature basis.
"""
from __future__ import annotations

import numpy as np

from .kappa_lib import SpectrumKappa
from .ridge_theory_lib import compute_df2


def spectral_feature_de_metrics(
        feature_eigenvalues, feature_teacher, backprop_weights,
        residual_signal, total_signal, kappa, sigma_noise, n):
    """Return DE prediction and pixel-accentuation metrics for linear features.

    Parameters
    ----------
    feature_eigenvalues : array_like, shape (p,)
        Eigenvalues of ``C = A Sigma_x A.T``.
    feature_teacher : array_like, shape (p,)
        Population regression coefficient in feature coordinates.
    backprop_weights : array_like, shape (p,)
        Diagonal of ``G = A A.T`` in the feature covariance eigenbasis.
    residual_signal : float
        Teacher signal variance outside the retained feature span.
    total_signal : float
        Full noiseless teacher signal variance in input space.
    kappa : float
        Ridge deterministic-equivalent effective regularization.
    sigma_noise : float
        Original response-noise standard deviation.  For Gaussian spectral
        features, omitted teacher signal acts as additional independent noise.
    n : int
        Number of training samples.
    """
    c = np.asarray(feature_eigenvalues, dtype=float)
    theta = np.asarray(feature_teacher, dtype=float)
    g = np.asarray(backprop_weights, dtype=float)
    if not (c.ndim == theta.ndim == g.ndim == 1 and
            len(c) == len(theta) == len(g)):
        raise ValueError('feature arrays must be one-dimensional and equal length')
    if np.any(c <= 0) or np.any(g <= 0):
        raise ValueError('feature eigenvalues and backprop weights must be positive')
    if residual_signal < 0 or total_signal <= 0:
        raise ValueError('signal powers must be nonnegative with positive total')

    df2 = compute_df2(c, kappa)
    denominator = n - df2
    if denominator <= 0:
        raise ValueError('n - df2 must be positive')
    effective_noise_variance = sigma_noise ** 2 + residual_signal
    shrink = c / (c + kappa)
    c_sig = np.sum(c / (c + kappa) ** 2 * theta ** 2)
    variance_scale = (
        kappa ** 2 * c_sig + effective_noise_variance) / denominator
    variance_pc = variance_scale * c / (c + kappa) ** 2
    mean_pc = shrink * theta
    second_moment_pc = mean_pc ** 2 + variance_pc
    error_pc = (1.0 - shrink) ** 2 * theta ** 2 + variance_pc

    estimation_gen_error = float(np.sum(c * error_pc))
    gen_error = float(residual_signal + estimation_gen_error)
    r2_gen = 1.0 - gen_error / total_signal
    gen_numerator = float(np.sum(c * mean_pc * theta))
    gen_denominator = float(np.sum(c * second_moment_pc))
    slope_gen = gen_numerator / gen_denominator

    acc_numerator = float(np.sum(g * mean_pc * theta))
    acc_denominator = float(np.sum(g * second_moment_pc))
    slope_acc = acc_numerator / acc_denominator
    acc_error = float(total_signal * (1.0 - slope_acc) ** 2)
    r2_acc = (
        1.0 - (1.0 - acc_denominator / acc_numerator) ** 2
        if acc_numerator != 0 else -np.inf)

    return {
        'effective_noise_variance': float(effective_noise_variance),
        'df2': float(df2),
        'variance_scale': float(variance_scale),
        'gen_error': gen_error,
        'gen_error_normalized': gen_error / total_signal,
        'r2_gen': float(r2_gen),
        'slope_gen': float(slope_gen),
        'acc_error': acc_error,
        'acc_error_normalized': acc_error / total_signal,
        'r2_acc': float(r2_acc),
        'slope_acc': float(slope_acc),
        'acc_numerator': acc_numerator,
        'acc_denominator': acc_denominator,
        'mean_feature_norm_sq': float(np.sum(mean_pc ** 2)),
        'mean_pixel_gradient_norm_sq': float(np.sum(g * second_moment_pc)),
    }


def select_spectral_feature_de_alpha(
        feature_eigenvalues, feature_teacher, backprop_weights,
        residual_signal, total_signal, sigma_noise, n, alphas):
    """Select the RidgeCV alpha by the feature-space LOOCV DE risk."""
    c = np.asarray(feature_eigenvalues, dtype=float)
    alphas = np.asarray(alphas, dtype=float)
    n_fit = n - 1
    solver = SpectrumKappa(c, len(c) / n_fit)
    risks = np.empty(len(alphas), dtype=float)
    for index, alpha in enumerate(alphas):
        kappa = solver(alpha / n_fit)
        metrics = spectral_feature_de_metrics(
            c, feature_teacher, backprop_weights, residual_signal,
            total_signal, kappa, sigma_noise, n_fit)
        risks[index] = metrics['gen_error']
    selected = int(np.argmin(risks))
    return float(alphas[selected]), risks


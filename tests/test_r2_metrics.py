import numpy as np
import torch

from rmt_core.ridge_theory_lib import (
    accentuation_r2_theory,
    generalization_r2_theory,
    peer_review_error_theory,
    peer_review_r2_theory,
    peer_review_theory,
)
from rmt_core.simulation_lib import (
    accentuation_error_simulation,
    accentuation_r2_simulation,
    generalization_metrics_simulation,
    peer_review_metrics_simulation,
    ridge_cross_validated_estimator,
    ridge_estimator,
    ridge_path_estimators,
    run_paired_cv_monte_carlo,
)
from rmt_core.teacher_lib import make_spectral_teacher
from scripts.validate_ffhq_disk_teacher import (
    fit_metrics,
    interpolated_crossing,
    ridge_loocv_fit,
)


def test_generalization_metrics_use_natural_signal_variance():
    eigenvalues = np.array([3.0, 0.5])
    eigenvectors = np.eye(2)
    beta_star = np.array([1.0, 2.0])
    beta_hat = np.array([0.5, 2.5])

    error, r2, signal = generalization_metrics_simulation(
        beta_hat, beta_star, eigenvalues, eigenvectors)

    expected_signal = 3.0 * 1.0 ** 2 + 0.5 * 2.0 ** 2
    expected_error = 3.0 * (-0.5) ** 2 + 0.5 * 0.5 ** 2
    assert np.isclose(signal, expected_signal)
    assert np.isclose(error, expected_error)
    assert np.isclose(r2, 1.0 - expected_error / expected_signal)


def test_self_path_is_peer_path_when_vectors_are_identical():
    eigenvalues = np.array([2.0, 1.0, 0.25])
    eigenvectors = np.eye(3)
    beta_star = np.array([1.0, -0.5, 0.75])
    beta_hat = np.array([0.8, -0.2, 0.9])

    acc_error, _ = accentuation_error_simulation(
        beta_hat, beta_star, eigenvalues, eigenvectors)
    acc_r2 = accentuation_r2_simulation(beta_hat, beta_star)
    peer_error, peer_r2, _, _ = peer_review_metrics_simulation(
        beta_hat, beta_hat, beta_star, eigenvalues, eigenvectors)

    assert np.isclose(peer_error, acc_error)
    assert np.isclose(peer_r2, acc_r2)


def test_theory_r2_values_are_consistent_with_theory_errors():
    eigenvalues = np.array([2.0, 1.0, 0.4, 0.1])
    beta_proj = np.array([1.0, -0.7, 0.3, 0.2])
    kappa = 0.2
    sigma = 0.4
    n = 20

    r2_gen, gen_error, signal = generalization_r2_theory(
        eigenvalues, beta_proj, kappa, sigma, n)
    assert np.isclose(r2_gen, 1.0 - gen_error / signal)

    peer_error, peer_r2, peer_gain, details = peer_review_theory(
        eigenvalues, beta_proj, kappa, sigma, n)
    assert np.isclose(peer_r2, 1.0 - (1.0 - peer_gain) ** 2)
    assert np.isclose(
        peer_r2, 1.0 - peer_error / details['peer_true_variance'])


def test_accentuation_delta_correction_changes_only_r2_not_alignment():
    eigenvalues = np.array([2.0, 1.0, 0.4, 0.1])
    beta_proj = np.array([1.0, -0.7, 0.3, 0.2])
    args = (eigenvalues, beta_proj, 0.2, 0.4, 20)

    r2_leading, alignment_leading, correction0 = accentuation_r2_theory(*args)
    r2_corrected, alignment_corrected, correction = accentuation_r2_theory(
        *args, include_delta_correction=True)

    assert correction0 == 0.0
    assert np.isclose(alignment_leading, alignment_corrected)
    assert np.isclose(r2_corrected, r2_leading + correction)


def test_peer_delta_correction_is_internally_consistent():
    eigenvalues = np.array([2.0, 1.0, 0.4, 0.1])
    beta_proj = np.array([1.0, -0.7, 0.3, 0.2])
    args = (eigenvalues, beta_proj, 0.2, 0.4, 20)

    leading_r2, leading_gain, correction0, _ = peer_review_r2_theory(*args)
    corrected_r2, corrected_gain, correction, details = peer_review_r2_theory(
        *args, include_delta_correction=True)

    assert correction0 == 0.0
    assert details['peer_gain_variance'] >= 0.0
    assert np.isclose(corrected_r2, leading_r2 + correction)
    assert np.isclose(
        corrected_gain, leading_gain + details['peer_gain_bias'])

    corrected_error, leading_error, error_correction, error_details = (
        peer_review_error_theory(*args, include_delta_correction=True))
    assert error_details['peer_numerator_variance'] >= 0.0
    assert error_details['peer_norm_variance'] >= 0.0
    assert np.isclose(corrected_error, leading_error + error_correction)


def test_ridge_path_matches_individual_fits():
    rng = np.random.default_rng(12)
    X = rng.standard_normal((30, 7))
    y = rng.standard_normal(30)
    lambda_grid = np.array([1e-3, 0.1, 1.0])
    path = ridge_path_estimators(X, y, lambda_grid)
    for index, lam in enumerate(lambda_grid):
        assert np.allclose(path[:, index], ridge_estimator(X, y, lam))


def test_cross_validation_path_matches_brute_force_scores():
    rng_data = np.random.default_rng(7)
    X = rng_data.standard_normal((25, 6))
    y = X @ rng_data.standard_normal(6) + 0.5 * rng_data.standard_normal(25)
    lambda_grid = np.array([1e-3, 0.03, 0.3, 3.0])
    seed = 99

    beta_cv, selected, scores = ridge_cross_validated_estimator(
        X, y, lambda_grid, n_folds=5, rng=np.random.default_rng(seed))

    permutation = np.random.default_rng(seed).permutation(len(y))
    folds = np.array_split(permutation, 5)
    brute_sse = np.zeros(len(lambda_grid))
    for validation_indices in folds:
        training_mask = np.ones(len(y), dtype=bool)
        training_mask[validation_indices] = False
        for index, lam in enumerate(lambda_grid):
            beta = ridge_estimator(X[training_mask], y[training_mask], lam)
            brute_sse[index] += np.sum(
                (X[validation_indices] @ beta - y[validation_indices]) ** 2)
    brute_scores = brute_sse / len(y)

    assert np.allclose(scores, brute_scores)
    assert selected == lambda_grid[np.argmin(brute_scores)]
    assert np.allclose(beta_cv, ridge_estimator(X, y, selected))


def test_localized_teacher_controls_signal_allocation():
    eigenvalues = np.geomspace(5.0, 0.1, 20)
    beta, beta_proj, allocation = make_spectral_teacher(
        eigenvalues, profile='localized', center=0.75, bandwidth=0.08,
        seed=4)

    assert np.allclose(beta, beta_proj)
    assert np.isclose(np.sum(eigenvalues * beta_proj ** 2), 1.0)
    assert np.allclose(eigenvalues * beta_proj ** 2, allocation)
    rank_fraction = (np.arange(len(eigenvalues)) + 0.5) / len(eigenvalues)
    assert np.sum(rank_fraction * allocation) > 0.65


def test_cv_monte_carlo_can_return_fitted_coefficients():
    eigenvalues = np.array([2.0, 1.0, 0.5])
    beta = np.array([0.4, -0.2, 0.1])
    result = run_paired_cv_monte_carlo(
        12, eigenvalues, np.eye(3), beta, 0.2,
        np.array([0.01, 0.1]), n_folds=3, n_trials=2,
        rng=np.random.default_rng(5), return_coefficients=True)

    assert result['coefficient_trials'].shape == (2, 2, 3)
    assert np.all(np.isfinite(result['coefficient_trials']))


def test_stable_ridge_loocv_path_matches_brute_force():
    """Tiny ridge penalties must not lose the LOO residual to cancellation."""
    rng = np.random.default_rng(23)
    n, d = 18, 30
    left, _ = np.linalg.qr(rng.standard_normal((n, n)))
    right, _ = np.linalg.qr(rng.standard_normal((d, n)))
    singular_values = np.geomspace(20.0, 0.1, n)
    X = (left * singular_values) @ right.T
    beta = rng.standard_normal(d)
    y = X @ beta
    alphas = np.array([1e-6, 1e-4, 1e-2, 0.1, 10.0])

    _, selected, loo_mse = ridge_loocv_fit(
        torch.from_numpy(X.astype(np.float32)),
        torch.from_numpy(beta.astype(np.float32)), 0.0,
        torch.from_numpy(alphas.astype(np.float32)),
        torch.Generator().manual_seed(5))

    brute_mse = []
    for alpha in alphas:
        residuals = []
        for held_out in range(n):
            keep = np.arange(n) != held_out
            X_train = X[keep]
            y_train = y[keep]
            x_mean = X_train.mean(axis=0)
            y_mean = y_train.mean()
            X_centered = X_train - x_mean
            y_centered = y_train - y_mean
            weight = np.linalg.solve(
                X_centered.T @ X_centered + alpha * np.eye(d),
                X_centered.T @ y_centered)
            prediction = y_mean + (X[held_out] - x_mean) @ weight
            residuals.append(y[held_out] - prediction)
        brute_mse.append(np.mean(np.square(residuals)))
    brute_mse = np.asarray(brute_mse)

    assert np.allclose(loo_mse.numpy(), brute_mse, rtol=5e-3, atol=1e-8)
    assert np.isclose(selected, alphas[np.argmin(brute_mse)])


def test_ffhq_evaluation_slopes_use_true_on_fitted_orientation():
    eigenvalues = torch.tensor([4.0, 1.0])
    eigenvectors = torch.eye(2)
    beta = torch.tensor([1.0, 2.0])
    weight = torch.tensor([0.5, 3.0])
    signal_power = float((eigenvalues * beta.square()).sum())

    metrics, _ = fit_metrics(
        weight, beta, eigenvalues, eigenvectors, beta, signal_power)

    expected_gen = float(
        (weight * eigenvalues * beta).sum()
        / (weight * eigenvalues * weight).sum())
    expected_acc = float((weight * beta).sum() / weight.square().sum())
    assert np.isclose(metrics['slope_gen'], expected_gen)
    assert np.isclose(metrics['slope_acc'], expected_acc)


def test_crossing_interpolation_uses_log_noise_axis():
    crossing = interpolated_crossing(
        np.array([0.1, 1.0]), np.array([0.0, 1.0]),
        np.array([0.2, 0.8]))

    assert crossing is not None
    crossing_x, crossing_y = crossing
    assert np.isclose(crossing_x, np.sqrt(0.1))
    assert np.isclose(crossing_y, 0.5)

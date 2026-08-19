import numpy as np
import torch

from scripts.compare_ffhq_fixed_vs_cv import ridge_fixed_fit
from scripts.validate_ffhq_linear_features import ridge_loocv_from_data
from scripts.validate_ffhq_disk_teacher import ridge_loocv_fit
from scripts.validate_vanhateren_disk_teacher import ridge_cv_and_fixed_fit


def brute_force_loo_mse(X, y, alpha):
    residuals = []
    for held_out in range(len(X)):
        keep = np.arange(len(X)) != held_out
        x_train = X[keep]
        y_train = y[keep]
        x_mean = x_train.mean(axis=0)
        y_mean = y_train.mean()
        xc = x_train - x_mean
        coef = np.linalg.solve(
            xc.T @ xc + alpha * np.eye(X.shape[1]),
            xc.T @ (y_train - y_mean))
        prediction = y_mean + (X[held_out] - x_mean) @ coef
        residuals.append(prediction - y[held_out])
    return np.mean(np.square(residuals))


def test_feature_loocv_matches_brute_force_when_p_less_than_n():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(9, 3))
    y = X @ np.asarray([0.5, -0.3, 0.8]) + rng.normal(scale=0.2, size=9)
    alphas = np.asarray([0.1, 1.0, 10.0])
    _, selected, mse = ridge_loocv_from_data(
        torch.tensor(X, dtype=torch.float64),
        torch.tensor(y, dtype=torch.float64),
        torch.tensor(alphas, dtype=torch.float64))
    brute = np.asarray([brute_force_loo_mse(X, y, alpha) for alpha in alphas])
    assert np.allclose(mse.numpy(), brute, rtol=1e-8, atol=1e-10)
    assert selected == alphas[np.argmin(brute)]


def test_fixed_sample_space_ridge_matches_primal_solution():
    generator = torch.Generator().manual_seed(11)
    X = torch.randn((8, 5), generator=generator, dtype=torch.float64)
    beta = torch.randn(5, generator=generator, dtype=torch.float64)
    alpha = 2.3
    estimate = ridge_fixed_fit(
        X, beta, sigma=0.0, alpha=alpha, generator=generator)
    Xc = X - X.mean(dim=0, keepdim=True)
    yc = Xc @ beta
    expected = torch.linalg.solve(
        Xc.T @ Xc + alpha * torch.eye(5, dtype=torch.float64),
        Xc.T @ yc)
    assert torch.allclose(estimate, expected, rtol=1e-10, atol=1e-12)


def test_paired_vanhateren_solver_matches_separate_ridge_solvers():
    generator = torch.Generator().manual_seed(23)
    X = torch.randn((9, 6), generator=generator, dtype=torch.float64)
    beta = torch.randn(6, generator=generator, dtype=torch.float64)
    alphas = torch.logspace(-2, 2, 17, dtype=torch.float64)
    fixed_alpha = 3.7

    paired_generator = torch.Generator().manual_seed(91)
    cv_weight, fixed_weight, selected = ridge_cv_and_fixed_fit(
        X, beta, sigma=0.4, alphas=alphas, fixed_alpha=fixed_alpha,
        generator=paired_generator)

    cv_generator = torch.Generator().manual_seed(91)
    expected_cv, expected_alpha, _ = ridge_loocv_fit(
        X, beta, sigma=0.4, alphas=alphas, generator=cv_generator)
    assert selected == expected_alpha
    assert torch.allclose(cv_weight, expected_cv, rtol=1e-10, atol=1e-12)

    fixed_generator = torch.Generator().manual_seed(91)
    expected_fixed = ridge_fixed_fit(
        X, beta, sigma=0.4, alpha=fixed_alpha,
        generator=fixed_generator)
    assert torch.allclose(
        fixed_weight, expected_fixed, rtol=1e-10, atol=1e-12)

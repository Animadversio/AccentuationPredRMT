"""
Monte Carlo simulation functions for validating the RMT theory of ridge regression.

Provides:
- Data generation from Gaussian model
- Ridge estimator
- Per-PC error measurement
- Accentuation alignment and error measurement
- run_monte_carlo: aggregates over many trials
"""
import numpy as np


def sample_gaussian_data(n, eigenvalues, eigenvectors, beta_star, sigma_noise,
                          rng=None):
    """Sample n data points from y = β*ᵀx + σε, x ~ N(0, Σ).

    Parameters
    ----------
    n : int
    eigenvalues : array_like, shape (d,)
    eigenvectors : array_like, shape (d, d)  (columns are eigenvectors)
    beta_star : array_like, shape (d,)
    sigma_noise : float
    rng : np.random.Generator or None

    Returns
    -------
    X : np.ndarray, shape (n, d)
    y : np.ndarray, shape (n,)
    """
    if rng is None:
        rng = np.random.default_rng()
    d = len(eigenvalues)
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    eigenvectors = np.asarray(eigenvectors, dtype=float)
    beta_star = np.asarray(beta_star, dtype=float)

    # X = Z @ Sigma^{1/2} where Z ~ N(0, I)
    Z = rng.standard_normal((n, d))
    X = Z @ (eigenvectors * np.sqrt(eigenvalues)).T  # shape (n, d)
    noise = rng.standard_normal(n) * sigma_noise
    y = X @ beta_star + noise
    return X, y


def ridge_estimator(X, y, lam):
    """OLS/Ridge estimator β̂ = (XᵀX + nλI)⁻¹Xᵀy.

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
    y : np.ndarray, shape (n,)
    lam : float
        Ridge penalty (λ=0 → OLS).

    Returns
    -------
    beta_hat : np.ndarray, shape (d,)
    """
    n, d = X.shape
    A = X.T @ X + n * lam * np.eye(d)
    return np.linalg.solve(A, X.T @ y)


def ridge_error_per_pc_simulation(beta_hat, beta_star, eigenvectors):
    """Compute (uₖᵀ(β̂ - β*))² for each PC k.

    Parameters
    ----------
    beta_hat : np.ndarray, shape (d,)
    beta_star : np.ndarray, shape (d,)
    eigenvectors : np.ndarray, shape (d, d)  (columns = eigenvectors)

    Returns
    -------
    errors : np.ndarray, shape (d,)
    """
    delta = beta_hat - beta_star
    return (eigenvectors.T @ delta) ** 2  # shape (d,)


def accentuation_alignment_simulation(beta_hat, beta_star):
    """Compute β̂ᵀβ* / β̂ᵀβ̂ for a single trial."""
    num = beta_hat @ beta_star
    denom = beta_hat @ beta_hat
    return num / denom if denom > 0 else 0.0


def accentuation_error_simulation(beta_hat, beta_star, eigenvalues, eigenvectors):
    """Compute normalized accentuation error for a single trial.

    Accentuation direction is β̂. Input x = α·β̂ where Var[α] is matched to
    the natural stimulus variance β*ᵀΣβ*.

    Error = (β*ᵀΣβ*) · (1 - β̂ᵀβ* / β̂ᵀβ̂)²
    """
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    eigenvectors = np.asarray(eigenvectors, dtype=float)
    beta_proj = eigenvectors.T @ beta_star
    signal_power = np.sum(eigenvalues * beta_proj ** 2)

    R = accentuation_alignment_simulation(beta_hat, beta_star)
    return signal_power * (1 - R) ** 2, R


def run_monte_carlo(n, eigenvalues, eigenvectors, beta_star, sigma_noise, lam,
                    n_trials=500, rng=None):
    """Run many trials and return averages of key quantities.

    Returns
    -------
    results : dict with keys
        'error_per_pc'   : mean (uₖᵀ(β̂-β*))²  per PC, shape (d,)
        'acc_alignment'  : mean β̂ᵀβ*/β̂ᵀβ̂
        'acc_error'      : mean accentuation error
        'total_gen_error': mean β*ᵀΣ(β̂-β*) style error (= Σλₖ·err_per_pc)
    """
    if rng is None:
        rng = np.random.default_rng()
    d = len(eigenvalues)
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    eigenvectors = np.asarray(eigenvectors, dtype=float)
    beta_star = np.asarray(beta_star, dtype=float)

    error_per_pc_acc = np.zeros(d)
    acc_alignment_acc = 0.0
    acc_error_acc = 0.0

    for _ in range(n_trials):
        X, y = sample_gaussian_data(n, eigenvalues, eigenvectors, beta_star,
                                    sigma_noise, rng=rng)
        beta_hat = ridge_estimator(X, y, lam)
        error_per_pc_acc += ridge_error_per_pc_simulation(beta_hat, beta_star,
                                                          eigenvectors)
        acc_err, R = accentuation_error_simulation(beta_hat, beta_star,
                                                   eigenvalues, eigenvectors)
        acc_alignment_acc += R
        acc_error_acc += acc_err

    results = {
        'error_per_pc': error_per_pc_acc / n_trials,
        'acc_alignment': acc_alignment_acc / n_trials,
        'acc_error': acc_error_acc / n_trials,
        'total_gen_error': np.sum(eigenvalues * error_per_pc_acc / n_trials),
    }
    return results

"""
GPU-accelerated (PyTorch) simulation functions for large-d ridge regression validation.

Key design choices for large d (1000–10000):
- Kernel/dual form of ridge estimator: solves an n×n system instead of d×d.
  β̂ = Xᵀ(XXᵀ + nλI)⁻¹y   [equivalent to (XᵀX + nλI)⁻¹Xᵀy, much cheaper when d >> n]
- Randomized SVD for covariance eigendecomposition.
- All heavy ops run on CUDA when available.
"""
import numpy as np
import torch


def get_device(use_gpu=True):
    if use_gpu and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


# ──────────────────────────────────────────────
# Ridge estimator — dual/kernel form
# ──────────────────────────────────────────────

def ridge_estimator_dual(X, y, lam, device=None):
    """Ridge estimator via kernel form: β̂ = Xᵀ(XXᵀ + nλI)⁻¹y.

    Complexity O(n²d) vs O(nd²) for the primal. Preferred when d >> n.

    Parameters
    ----------
    X : torch.Tensor, shape (n, d)
    y : torch.Tensor, shape (n,)
    lam : float

    Returns
    -------
    beta_hat : torch.Tensor, shape (d,)
    """
    n = X.shape[0]
    K = X @ X.T  # (n, n)
    A = K + n * lam * torch.eye(n, dtype=X.dtype, device=X.device)
    alpha = torch.linalg.solve(A, y)  # (n,)
    return X.T @ alpha  # (d,)


def ridge_estimator_primal(X, y, lam, device=None):
    """Ridge estimator via primal form: β̂ = (XᵀX + nλI)⁻¹Xᵀy.

    Preferred when n >> d.
    """
    n, d = X.shape
    A = X.T @ X + n * lam * torch.eye(d, dtype=X.dtype, device=X.device)
    return torch.linalg.solve(A, X.T @ y)


def ridge_estimator_gpu(X_np, y_np, lam, device=None):
    """Auto-select primal vs dual based on d vs n."""
    if device is None:
        device = get_device()
    X = torch.tensor(X_np, dtype=torch.float32, device=device)
    y = torch.tensor(y_np, dtype=torch.float32, device=device)
    n, d = X.shape
    if d > n:
        beta = ridge_estimator_dual(X, y, lam)
    else:
        beta = ridge_estimator_primal(X, y, lam)
    return beta.cpu().numpy()


# ──────────────────────────────────────────────
# Data sampling on GPU
# ──────────────────────────────────────────────

def sample_gaussian_data_gpu(n, eigenvalues_np, eigenvectors_np, beta_star_np,
                              sigma_noise, residual_var=0.0, diagonal=False, device=None):
    """Sample n data points from y = β*ᵀx + σε on GPU.

    Modes
    -----
    diagonal=True:
        eigenvectors_np is ignored. Covariance is Σ = diag(eigenvalues_np).
        X[:, k] = Z[:, k] * sqrt(λ_k).  Exact for isotropic/power-law spectra.
        eigenvalues_np must be shape (d,) — the full spectrum.

    diagonal=False (default):
        Low-rank: Σ ≈ U_K diag(λ_K) U_Kᵀ + residual_var * I
        eigenvectors_np shape (d, K), eigenvalues_np shape (K,).

    Returns
    -------
    X_np, y_np : np.ndarray
    """
    if device is None:
        device = get_device()

    beta = torch.tensor(beta_star_np, dtype=torch.float32, device=device)

    if diagonal:
        d = len(eigenvalues_np)
        lam_sqrt = torch.tensor(np.sqrt(np.maximum(eigenvalues_np, 0)),
                                dtype=torch.float32, device=device)   # (d,)
        Z = torch.randn(n, d, device=device)
        X = Z * lam_sqrt  # (n, d)  — exact diagonal Gaussian
    else:
        d = eigenvectors_np.shape[0]
        K = len(eigenvalues_np)
        lam_sqrt = torch.tensor(np.sqrt(eigenvalues_np), dtype=torch.float32, device=device)
        U = torch.tensor(eigenvectors_np, dtype=torch.float32, device=device)  # (d, K)
        Z_K = torch.randn(n, K, device=device)
        X = Z_K @ (U * lam_sqrt).T  # (n, d)
        if residual_var > 0.0:
            X = X + torch.randn(n, d, device=device) * (residual_var ** 0.5)

    noise = torch.randn(n, device=device) * sigma_noise
    y = X @ beta + noise
    return X.cpu().numpy(), y.cpu().numpy()


# ──────────────────────────────────────────────
# Per-PC error projection (top-K only)
# ──────────────────────────────────────────────

def ridge_error_per_pc_gpu(beta_hat_np, beta_star_np, eigenvectors_np):
    """Compute (uₖᵀ(β̂ - β*))² for each PC. Returns np.ndarray shape (d,)."""
    delta = beta_hat_np - beta_star_np
    return (eigenvectors_np.T @ delta) ** 2


# ──────────────────────────────────────────────
# Monte Carlo — large d
# ──────────────────────────────────────────────

def run_monte_carlo_gpu(n, eigenvalues, eigenvectors, beta_star, sigma_noise, lam,
                        n_trials=200, residual_var=0.0, diagonal=False,
                        device=None, rng=None, verbose=True):
    """GPU-accelerated Monte Carlo over many trials.

    Parameters
    ----------
    n : int
    eigenvalues : np.ndarray, shape (K,)  top-K eigenvalues used for data generation
    eigenvectors : np.ndarray, shape (d, K)  top-K eigenvectors
    beta_star : np.ndarray, shape (d,)
    sigma_noise : float
    lam : float
    n_trials : int
    residual_var : float
        Isotropic residual variance for dimensions not covered by top-K eigenvectors.
        Set to mean of excluded eigenvalues for van Hateren. 0 for exact spectra.
    device : torch.device

    Returns
    -------
    dict with 'error_per_pc' (K,), 'acc_alignment', 'acc_error', 'total_gen_error'
    """
    if device is None:
        device = get_device()
    if rng is None:
        rng = np.random.default_rng()

    K = eigenvectors.shape[1]
    # eigenvalues may be full-spectrum (d,) for diagonal mode; use top-K slice for metrics
    eigenvalues_k = eigenvalues[:K]

    error_per_pc_acc = np.zeros(K)
    acc_align_acc = 0.0
    acc_error_acc = 0.0

    beta_proj_true = eigenvectors.T @ beta_star  # (K,)
    signal_power = float(np.sum(eigenvalues_k * beta_proj_true ** 2))

    for i in range(n_trials):
        if verbose and (i + 1) % 50 == 0:
            print(f"  trial {i+1}/{n_trials}", flush=True)

        X_np, y_np = sample_gaussian_data_gpu(
            n, eigenvalues, eigenvectors, beta_star, sigma_noise,
            residual_var=residual_var, diagonal=diagonal, device=device)
        beta_hat = ridge_estimator_gpu(X_np, y_np, lam, device=device)

        # Per-PC error (top-K PCs)
        delta = beta_hat - beta_star
        proj = eigenvectors.T @ delta  # (K,)
        error_per_pc_acc += proj ** 2

        # Accentuation alignment
        num = float(beta_hat @ beta_star)
        den = float(beta_hat @ beta_hat)
        R = num / den if den > 0 else 0.0
        acc_align_acc += R
        acc_error_acc += signal_power * (1 - R) ** 2

    return {
        'error_per_pc': error_per_pc_acc / n_trials,
        'acc_alignment': acc_align_acc / n_trials,
        'acc_error': acc_error_acc / n_trials,
        'total_gen_error': float(np.sum(eigenvalues_k * error_per_pc_acc / n_trials)),
    }

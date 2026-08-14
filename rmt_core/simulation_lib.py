"""
Monte Carlo simulation functions for validating the RMT theory of ridge regression.

Provides:
- Data generation from Gaussian model
- Ridge estimator
- Per-PC error measurement
- Generalization, accentuation, and independent-peer error/R² measurement
- run_monte_carlo: aggregates over many trials
"""
import time

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


def ridge_path_estimators(X, y, lambda_grid):
    """Fit an entire ridge path from one eigendecomposition of XᵀX/n.

    Returns an array of shape ``(d, n_lambda)``. This is substantially faster
    than solving a new d×d system for every candidate lambda during CV.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    lambda_grid = np.asarray(lambda_grid, dtype=float)
    if np.any(lambda_grid < 0):
        raise ValueError('Ridge penalties must be nonnegative.')
    n = X.shape[0]
    gram = X.T @ X / n
    rhs = X.T @ y / n
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    rhs_proj = eigenvectors.T @ rhs
    denominators = eigenvalues[:, None] + lambda_grid[None, :]
    # A zero eigenvalue with lambda=0 is outside the intended CV grid. Use the
    # Moore-Penrose convention if it is nevertheless requested.
    inverse_rhs = np.divide(
        rhs_proj[:, None], denominators,
        out=np.zeros_like(denominators), where=denominators > 0)
    return eigenvectors @ inverse_rhs


def ridge_cross_validated_estimator(X, y, lambda_grid, n_folds=5, rng=None):
    """Select lambda by K-fold validation MSE, then refit on all samples.

    The ridge path within each fold is evaluated using a single eigendecomposition.
    No intercept or feature standardization is applied, matching the zero-mean
    teacher/student model elsewhere in this repository.

    Returns
    -------
    beta_hat : ndarray, shape (d,)
    selected_lambda : float
    cv_mse : ndarray, shape (n_lambda,)
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    lambda_grid = np.asarray(lambda_grid, dtype=float)
    n = len(y)
    if not 2 <= n_folds <= n:
        raise ValueError('n_folds must be between 2 and the sample count.')
    if rng is None:
        rng = np.random.default_rng()

    permutation = rng.permutation(n)
    folds = np.array_split(permutation, n_folds)
    squared_error = np.zeros(len(lambda_grid), dtype=float)
    validation_count = 0
    all_indices = np.arange(n)

    for validation_indices in folds:
        training_mask = np.ones(n, dtype=bool)
        training_mask[validation_indices] = False
        training_indices = all_indices[training_mask]
        beta_path = ridge_path_estimators(
            X[training_indices], y[training_indices], lambda_grid)
        residual = X[validation_indices] @ beta_path - y[validation_indices, None]
        squared_error += np.sum(residual ** 2, axis=0)
        validation_count += len(validation_indices)

    cv_mse = squared_error / validation_count
    selected_index = int(np.argmin(cv_mse))
    selected_lambda = float(lambda_grid[selected_index])
    beta_hat = ridge_estimator(X, y, selected_lambda)
    return beta_hat, selected_lambda, cv_mse


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


def generalization_metrics_simulation(beta_hat, beta_star, eigenvalues,
                                      eigenvectors):
    """Return noiseless population generalization error and R² for one fit."""
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    eigenvectors = np.asarray(eigenvectors, dtype=float)
    beta_star = np.asarray(beta_star, dtype=float)
    delta_proj = eigenvectors.T @ (beta_hat - beta_star)
    beta_proj = eigenvectors.T @ beta_star
    gen_error = float(np.sum(eigenvalues * delta_proj ** 2))
    signal_power = float(np.sum(eigenvalues * beta_proj ** 2))
    r2_gen = 1.0 - gen_error / signal_power if signal_power > 0 else np.nan
    return gen_error, r2_gen, signal_power


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


def accentuation_r2_simulation(beta_hat, beta_star):
    """Return path-wise R² on a model's own zero-seed accentuation path.

    This is the fixed-calibration coefficient of determination from the note,
    not squared Pearson correlation.  It can be negative.
    """
    true_inner = float(beta_hat @ beta_star)
    pred_inner = float(beta_hat @ beta_hat)
    if true_inner == 0:
        return np.nan
    return 1.0 - (1.0 - pred_inner / true_inner) ** 2


def peer_review_metrics_simulation(beta_hat, beta_peer, beta_star,
                                   eigenvalues, eigenvectors):
    """Evaluate ``beta_hat`` on the accentuation path generated by ``beta_peer``.

    The peer path is x=αβ' with Var(α) chosen so that the peer model's
    predicted variance equals the natural teacher signal power.  Returns the
    peer error, path-wise R², predicted/true response gain, and true response
    variance on the normalized peer path.
    """
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    eigenvectors = np.asarray(eigenvectors, dtype=float)
    beta_proj = eigenvectors.T @ beta_star
    signal_power = float(np.sum(eigenvalues * beta_proj ** 2))

    peer_norm_sq = float(beta_peer @ beta_peer)
    teacher_peer = float(beta_peer @ beta_star)
    reviewer_peer = float(beta_peer @ beta_hat)
    if peer_norm_sq <= 0:
        return np.nan, np.nan, np.nan, np.nan

    teacher_scale = teacher_peer / peer_norm_sq
    reviewer_scale = reviewer_peer / peer_norm_sq
    peer_error = signal_power * (reviewer_scale - teacher_scale) ** 2
    peer_true_variance = signal_power * teacher_scale ** 2

    if teacher_peer == 0:
        peer_gain = np.nan
        peer_r2 = np.nan
    else:
        peer_gain = reviewer_peer / teacher_peer
        peer_r2 = 1.0 - (1.0 - peer_gain) ** 2
    return peer_error, peer_r2, peer_gain, peer_true_variance


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


def run_paired_monte_carlo(n, eigenvalues, eigenvectors, beta_star,
                           sigma_noise, lam, n_trials=500, rng=None,
                           progress=False, symmetrize_peer=True,
                           return_trials=False):
    """Monte Carlo for generalization, own-path, and independent-peer metrics.

    Each trial draws two independent training datasets and fits ``beta_hat``
    and ``beta_peer``.  When ``symmetrize_peer`` is true, the two directional
    peer scores are averaged within the trial; this uses both independent fits
    without pretending the directions are independent Monte Carlo trials.

    Parameters
    ----------
    progress : bool
        Show a tqdm progress bar with live ETA.
    return_trials : bool
        Include per-trial arrays under the ``trials`` key for diagnostics and
        durable caching.

    Returns
    -------
    results : dict
        Scalar means, standard errors, elapsed time, and optional trial arrays.
    """
    if rng is None:
        rng = np.random.default_rng()
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    eigenvectors = np.asarray(eigenvectors, dtype=float)
    beta_star = np.asarray(beta_star, dtype=float)

    metric_names = (
        'gen_error', 'r2_gen', 'acc_error', 'acc_alignment', 'r2_acc',
        'peer_error', 'peer_gain', 'r2_peer', 'peer_true_variance',
    )
    trials = {name: np.empty(n_trials, dtype=float) for name in metric_names}

    iterator = range(n_trials)
    if progress:
        try:
            from tqdm.auto import tqdm
            iterator = tqdm(iterator, desc='paired ridge MC', unit='trial')
        except ImportError:
            pass

    start = time.perf_counter()
    for i in iterator:
        X, y = sample_gaussian_data(
            n, eigenvalues, eigenvectors, beta_star, sigma_noise, rng=rng)
        beta_hat = ridge_estimator(X, y, lam)
        X_peer, y_peer = sample_gaussian_data(
            n, eigenvalues, eigenvectors, beta_star, sigma_noise, rng=rng)
        beta_peer = ridge_estimator(X_peer, y_peer, lam)

        gen_error, r2_gen, _ = generalization_metrics_simulation(
            beta_hat, beta_star, eigenvalues, eigenvectors)
        acc_error, alignment = accentuation_error_simulation(
            beta_hat, beta_star, eigenvalues, eigenvectors)
        r2_acc = accentuation_r2_simulation(beta_hat, beta_star)
        peer_values = peer_review_metrics_simulation(
            beta_hat, beta_peer, beta_star, eigenvalues, eigenvectors)

        if symmetrize_peer:
            reverse_values = peer_review_metrics_simulation(
                beta_peer, beta_hat, beta_star, eigenvalues, eigenvectors)
            peer_values = tuple(
                0.5 * (forward + reverse)
                for forward, reverse in zip(peer_values, reverse_values)
            )

        trials['gen_error'][i] = gen_error
        trials['r2_gen'][i] = r2_gen
        trials['acc_error'][i] = acc_error
        trials['acc_alignment'][i] = alignment
        trials['r2_acc'][i] = r2_acc
        trials['peer_error'][i] = peer_values[0]
        trials['r2_peer'][i] = peer_values[1]
        trials['peer_gain'][i] = peer_values[2]
        trials['peer_true_variance'][i] = peer_values[3]

    elapsed = time.perf_counter() - start
    results = {'n_trials': int(n_trials), 'elapsed_seconds': elapsed}
    for name, values in trials.items():
        finite = values[np.isfinite(values)]
        results[name] = float(np.mean(finite)) if len(finite) else np.nan
        results[f'{name}_std'] = (
            float(np.std(finite, ddof=1)) if len(finite) > 1 else np.nan)
        results[f'{name}_se'] = (
            results[f'{name}_std'] / np.sqrt(len(finite))
            if len(finite) > 1 else np.nan)
        results[f'{name}_n_finite'] = int(len(finite))
    if return_trials:
        results['trials'] = trials
    return results


def run_paired_cv_monte_carlo(n, eigenvalues, eigenvectors, beta_star,
                              sigma_noise, lambda_grid, n_folds=5,
                              n_trials=200, rng=None, progress=False,
                              return_trials=False,
                              return_coefficients=False):
    """Paired Monte Carlo with independently cross-validated ridge penalties.

    Two independent datasets are generated per trial. Each model selects its
    own lambda by K-fold CV and is refit on all ``n`` samples. Natural and
    own-path metrics are averaged across the pair; peer metrics are averaged
    across both reviewer/generator directions.
    """
    if rng is None:
        rng = np.random.default_rng()
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    eigenvectors = np.asarray(eigenvectors, dtype=float)
    beta_star = np.asarray(beta_star, dtype=float)
    lambda_grid = np.asarray(lambda_grid, dtype=float)

    metric_names = (
        'gen_error', 'r2_gen', 'acc_error', 'acc_alignment', 'r2_acc',
        'peer_error', 'peer_gain', 'r2_peer', 'peer_true_variance',
        'lambda_cv_a', 'lambda_cv_b', 'lambda_cv_mean',
    )
    trials = {name: np.empty(n_trials, dtype=float) for name in metric_names}
    coefficient_trials = (
        np.empty((n_trials, 2, len(beta_star)), dtype=float)
        if return_coefficients else None)

    iterator = range(n_trials)
    if progress:
        try:
            from tqdm.auto import tqdm
            iterator = tqdm(iterator, desc='paired K-fold ridge CV', unit='trial')
        except ImportError:
            pass

    start = time.perf_counter()
    for i in iterator:
        fitted = []
        own_metrics = []
        selected_lambdas = []
        for _ in range(2):
            X, y = sample_gaussian_data(
                n, eigenvalues, eigenvectors, beta_star, sigma_noise, rng=rng)
            beta_hat, selected_lambda, _ = ridge_cross_validated_estimator(
                X, y, lambda_grid, n_folds=n_folds, rng=rng)
            gen_error, r2_gen, _ = generalization_metrics_simulation(
                beta_hat, beta_star, eigenvalues, eigenvectors)
            acc_error, alignment = accentuation_error_simulation(
                beta_hat, beta_star, eigenvalues, eigenvectors)
            r2_acc = accentuation_r2_simulation(beta_hat, beta_star)
            fitted.append(beta_hat)
            selected_lambdas.append(selected_lambda)
            own_metrics.append((gen_error, r2_gen, acc_error, alignment, r2_acc))

        if coefficient_trials is not None:
            coefficient_trials[i, 0] = fitted[0]
            coefficient_trials[i, 1] = fitted[1]

        own_average = np.mean(np.asarray(own_metrics), axis=0)
        peer_forward = peer_review_metrics_simulation(
            fitted[0], fitted[1], beta_star, eigenvalues, eigenvectors)
        peer_reverse = peer_review_metrics_simulation(
            fitted[1], fitted[0], beta_star, eigenvalues, eigenvectors)
        peer_average = 0.5 * (np.asarray(peer_forward) + np.asarray(peer_reverse))

        trials['gen_error'][i], trials['r2_gen'][i] = own_average[:2]
        trials['acc_error'][i] = own_average[2]
        trials['acc_alignment'][i] = own_average[3]
        trials['r2_acc'][i] = own_average[4]
        trials['peer_error'][i] = peer_average[0]
        trials['r2_peer'][i] = peer_average[1]
        trials['peer_gain'][i] = peer_average[2]
        trials['peer_true_variance'][i] = peer_average[3]
        trials['lambda_cv_a'][i] = selected_lambdas[0]
        trials['lambda_cv_b'][i] = selected_lambdas[1]
        trials['lambda_cv_mean'][i] = np.mean(selected_lambdas)

    elapsed = time.perf_counter() - start
    results = {'n_trials': int(n_trials), 'elapsed_seconds': elapsed}
    for name, values in trials.items():
        finite = values[np.isfinite(values)]
        results[name] = float(np.mean(finite)) if len(finite) else np.nan
        results[f'{name}_std'] = (
            float(np.std(finite, ddof=1)) if len(finite) > 1 else np.nan)
        results[f'{name}_se'] = (
            results[f'{name}_std'] / np.sqrt(len(finite))
            if len(finite) > 1 else np.nan)
        results[f'{name}_median'] = (
            float(np.median(finite)) if len(finite) else np.nan)
        results[f'{name}_n_finite'] = int(len(finite))
    if return_trials:
        results['trials'] = trials
    if coefficient_trials is not None:
        results['coefficient_trials'] = coefficient_trials
    return results

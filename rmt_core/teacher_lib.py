"""Teacher constructions with controlled population-spectral alignment."""

import numpy as np


def make_spectral_teacher(eigenvalues, eigenvectors=None, *, profile='localized',
                          center=0.5, bandwidth=0.12, seed=0):
    """Construct a unit-signal teacher with controlled PC allocation.

    The controlled quantity is the fraction of natural response variance in
    each population eigenmode,

        s_k = lambda_k * (u_k.T @ beta_star) ** 2,

    with ``sum(s_k) = 1``.  A localized teacher places a Gaussian bump in
    eigenvalue-rank space (rank zero is the top PC); a random teacher draws
    i.i.d. Gaussian PC coefficients and then normalizes the signal power.

    Returns
    -------
    beta_star : ndarray, shape (d,)
        Teacher in the original coordinate basis.
    beta_proj : ndarray, shape (d,)
        Teacher coefficients in the population eigenbasis.
    signal_allocation : ndarray, shape (d,)
        Per-PC natural response variance, summing to one.
    """
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    if eigenvalues.ndim != 1 or len(eigenvalues) == 0:
        raise ValueError('eigenvalues must be a nonempty one-dimensional array.')
    if np.any(eigenvalues <= 0):
        raise ValueError('Controlled spectral teachers require positive eigenvalues.')
    d = len(eigenvalues)
    if eigenvectors is None:
        eigenvectors = np.eye(d)
    eigenvectors = np.asarray(eigenvectors, dtype=float)
    if eigenvectors.shape != (d, d):
        raise ValueError('eigenvectors must have shape (d, d).')

    rng = np.random.default_rng(seed)
    if profile == 'random':
        beta_proj = rng.standard_normal(d)
        signal_power = float(np.sum(eigenvalues * beta_proj ** 2))
        beta_proj = beta_proj / np.sqrt(signal_power)
        signal_allocation = eigenvalues * beta_proj ** 2
    elif profile == 'localized':
        if not 0.0 <= center <= 1.0:
            raise ValueError('center must lie between zero (top) and one (bottom).')
        if bandwidth <= 0:
            raise ValueError('bandwidth must be positive.')
        rank_fraction = (np.arange(d, dtype=float) + 0.5) / d
        signal_allocation = np.exp(
            -0.5 * ((rank_fraction - center) / bandwidth) ** 2)
        signal_allocation /= np.sum(signal_allocation)
        signs = rng.choice(np.array([-1.0, 1.0]), size=d)
        beta_proj = signs * np.sqrt(signal_allocation / eigenvalues)
    else:
        raise ValueError("profile must be 'localized' or 'random'.")

    beta_star = eigenvectors @ beta_proj
    return beta_star, beta_proj, signal_allocation


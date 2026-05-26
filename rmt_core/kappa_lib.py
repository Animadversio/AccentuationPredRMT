"""
Solvers for the deterministic equivalent κ(λ) appearing in ridge regression RMT.

The fixed-point equation (Marchenko-Pastur / resolvent):
    κ(z) - z = γ · (1/d) · Σ_k  κ(z)·λ_k / (κ(z) + λ_k)

where γ = d/n, λ_k are population eigenvalues, and z = λ (ridge penalty).

Adapted from DiffusionRMT_consistency/rmt_core/rmt_sc_lib.py and rmt_kappa_int_lib.py.
"""
import numpy as np
from scipy.optimize import fsolve


def solve_kappa(z, eigenvalues, gamma, weights=None, initial_guess=None):
    """Solve κ(z) for a single real z via Newton / fsolve.

    Parameters
    ----------
    z : float
        Ridge penalty (real, positive).
    eigenvalues : array_like, shape (d,)
        Population eigenvalues λ_k.
    gamma : float
        Aspect ratio d/n.
    weights : array_like or None
        Weights for each eigenvalue (default: uniform 1/d).
    initial_guess : float or None
        Starting point for the solver.

    Returns
    -------
    kappa : float
    """
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    d = len(eigenvalues)
    if weights is None:
        weights = np.ones(d) / d
    else:
        weights = np.asarray(weights, dtype=float)

    def F(kappa):
        val = kappa[0] - z - gamma * np.sum(weights * kappa[0] * eigenvalues / (kappa[0] + eigenvalues))
        return np.array([val])

    def Fprime(kappa):
        val = 1.0 - gamma * np.sum(weights * eigenvalues ** 2 / (kappa[0] + eigenvalues) ** 2)
        return np.array([[val]])

    x0 = np.array([z if initial_guess is None else initial_guess], dtype=float)
    result = fsolve(F, x0, fprime=Fprime, full_output=False)
    return float(result[0])


def solve_kappa_path(z_array, eigenvalues, gamma, weights=None, initial_guess=None):
    """Solve κ(z) for an array of z values using analytic continuation.

    Sweeps from large to small z for numerical stability, seeding each step
    from the previous solution.

    Parameters
    ----------
    z_array : array_like
        Array of ridge penalties (real, positive).
    eigenvalues : array_like
        Population eigenvalues.
    gamma : float
        Aspect ratio d/n.
    weights : array_like or None
    initial_guess : float or None

    Returns
    -------
    kappas : np.ndarray, same shape as z_array
    """
    z_array = np.asarray(z_array, dtype=float)
    original_order = np.argsort(z_array)[::-1]  # large → small
    z_sorted = z_array[original_order]

    kappas = np.empty_like(z_array)
    last_kappa = initial_guess
    for i, z in enumerate(z_sorted):
        kappa = solve_kappa(z, eigenvalues, gamma, weights, initial_guess=last_kappa)
        kappas[original_order[i]] = kappa
        last_kappa = kappa

    return kappas


class SpectrumKappa:
    """Callable κ(λ) backed by a fixed spectrum, with analytic-continuation caching."""

    def __init__(self, eigenvalues, gamma, weights=None):
        self.eigenvalues = np.asarray(eigenvalues, dtype=float)
        self.gamma = float(gamma)
        self.weights = (np.ones(len(self.eigenvalues)) / len(self.eigenvalues)
                        if weights is None else np.asarray(weights, dtype=float))
        self._cache = {}  # z -> kappa

    def __call__(self, z):
        if z in self._cache:
            return self._cache[z]
        nearest = None
        if self._cache:
            best_z = min(self._cache, key=lambda zc: abs(zc - z))
            nearest = self._cache[best_z]
        kappa = solve_kappa(z, self.eigenvalues, self.gamma, self.weights,
                            initial_guess=nearest)
        self._cache[z] = kappa
        return kappa

    def on_array(self, z_array):
        return solve_kappa_path(z_array, self.eigenvalues, self.gamma, self.weights)

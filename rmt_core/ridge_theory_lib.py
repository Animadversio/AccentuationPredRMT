"""
RMT-based theory formulas for ridge regression weight deviation and accentuation error.

Main results from linear_reg_weight_deviation.tex:

1. Per-PC squared error (three terms):

   E[(uₖᵀ(β̂ - β*))²] ≍
       κ²/(λₖ+κ)² · (β*ᵀuₖ)²                         [overshrinkage bias]
     + κ²λₖ/(λₖ+κ)² · C_sig / (n - df₂)               [finite-sample signal]
     + σ²λₖ/(λₖ+κ)² / (n - df₂)                       [finite-sample noise]

   where C_sig = β*ᵀ(Σ + κI)⁻²Σβ* = Σ_k λₖ(β*ᵀuₖ)²/(λₖ+κ)²

2. Accentuation alignment R_det:
   R_det ≍ β*ᵀ(Σ+κI)⁻¹Σβ* / [β*ᵀΣ²(Σ+κI)⁻²β* + (σ²/n)·df₂_deriv]

3. Accentuation error:
   E_acc ≍ (β*ᵀΣβ*) · (1 - R_det)²
"""
import numpy as np


# ──────────────────────────────────────────────
# Helper quantities
# ──────────────────────────────────────────────

def compute_kappa_from_lambda(lam, eigenvalues, gamma, weights=None):
    """Compute κ(λ) for a given ridge penalty λ.

    Thin wrapper around kappa_lib.solve_kappa for convenience.
    """
    from .kappa_lib import solve_kappa
    return solve_kappa(lam, eigenvalues, gamma, weights)


def compute_df2(eigenvalues, kappa, weights=None):
    """Compute df₂(κ) = Σ_k λ_k² / (λ_k + κ)².

    This is the (unnormalized) effective degrees of freedom that controls
    finite-sample variance.

    Parameters
    ----------
    eigenvalues : array_like, shape (d,)
    kappa : float
    weights : array_like or None
        If None, uniform 1/d weights (normalized so df₂ ∈ [0, d]).

    Returns
    -------
    df2 : float  (in [0, d])
    """
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    d = len(eigenvalues)
    if weights is None:
        weights = np.ones(d) / d
    else:
        weights = np.asarray(weights, dtype=float)
    # normalized: each term is w_k * λ_k²/(λ_k+κ)²; unnormalized = d * this
    return d * np.sum(weights * eigenvalues ** 2 / (eigenvalues + kappa) ** 2)


def _c_sig(eigenvalues, beta_proj, kappa, weights=None):
    """C_sig = β*ᵀ(Σ+κI)⁻²Σβ* = Σ_k λ_k/(λ_k+κ)² · (β*ᵀuₖ)²."""
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    beta_proj = np.asarray(beta_proj, dtype=float)
    return np.sum(eigenvalues / (eigenvalues + kappa) ** 2 * beta_proj ** 2)


# ──────────────────────────────────────────────
# Per-PC error formula
# ──────────────────────────────────────────────

def ridge_error_per_pc_theory(eigenvalues, beta_proj, kappa, sigma_noise, n,
                               weights=None,
                               eigenvalues_full=None, beta_proj_full=None):
    """RMT theory prediction for E[(uₖᵀ(β̂ - β*))²] for each PC k.

    Parameters
    ----------
    eigenvalues : array_like, shape (K,)
        Eigenvalues for the PCs we want to evaluate (top-K or all d).
    beta_proj : array_like, shape (K,)
        True coefficient projections β*ᵀuₖ for those PCs.
    kappa : float
        Deterministic equivalent κ(λ) at the ridge penalty λ.
    sigma_noise : float
        Noise std σ.
    n : int
        Number of samples.
    weights : array_like or None
    eigenvalues_full : array_like, shape (d,) or None
        Full spectrum for computing global quantities df₂ and C_sig.
        If None, uses eigenvalues (assumes evaluated PCs cover the full spectrum).
    beta_proj_full : array_like, shape (d,) or None
        Full β* projections onto all d eigenvectors (for C_sig).
        If None, uses beta_proj.

    Returns
    -------
    error : np.ndarray, shape (K,)
        Theory prediction per PC.
    term1, term2, term3 : np.ndarray, shape (K,)
        Individual contributions.
    """
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    beta_proj = np.asarray(beta_proj, dtype=float)

    # Use full spectrum for global quantities if provided
    eigs_full = np.asarray(eigenvalues_full, dtype=float) if eigenvalues_full is not None \
        else eigenvalues
    bproj_full = np.asarray(beta_proj_full, dtype=float) if beta_proj_full is not None \
        else beta_proj

    df2 = compute_df2(eigs_full, kappa)        # uses full spectrum
    denom = n - df2
    c_sig = _c_sig(eigs_full, bproj_full, kappa)  # uses full spectrum

    aniso = eigenvalues / (eigenvalues + kappa) ** 2
    shrink = kappa ** 2 / (eigenvalues + kappa) ** 2

    term1 = shrink * beta_proj ** 2
    term2 = kappa ** 2 * aniso * c_sig / denom
    term3 = sigma_noise ** 2 * aniso / denom

    return term1 + term2 + term3, term1, term2, term3


def ridge_error_total_theory(eigenvalues, beta_proj, kappa, sigma_noise, n,
                              weights=None):
    """Total in-distribution generalization error E[(β̂-β*)ᵀΣ(β̂-β*)].

    = Σ_k λ_k · E[(uₖᵀ(β̂-β*))²]
    """
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    err_per_pc, *_ = ridge_error_per_pc_theory(
        eigenvalues, beta_proj, kappa, sigma_noise, n, weights)
    return np.sum(eigenvalues * err_per_pc)


# ──────────────────────────────────────────────
# Accentuation error
# ──────────────────────────────────────────────

def accentuation_alignment(eigenvalues, beta_proj, kappa, sigma_noise, n,
                            weights=None):
    """Compute R_det ≍ E[β̂ᵀβ*] / E[β̂ᵀβ̂].

    Parameters
    ----------
    eigenvalues : array_like, shape (d,)
    beta_proj : array_like, shape (d,)
        β*ᵀuₖ projections.
    kappa : float
    sigma_noise : float
    n : int

    Returns
    -------
    R_det : float
        Cosine-like alignment between β̂ and β*.
    numerator, denominator : float
    """
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    beta_proj = np.asarray(beta_proj, dtype=float)

    df2 = compute_df2(eigenvalues, kappa, weights)
    denom_eff = n - df2

    # E[β̂ᵀβ*] ≍ β*ᵀ(Σ+κI)⁻¹Σβ* = Σ_k λ_k/(λ_k+κ) · (β*ᵀuₖ)²
    num = np.sum(eigenvalues / (eigenvalues + kappa) * beta_proj ** 2)

    # E[β̂ᵀβ̂] signal term: β*ᵀΣ²(Σ+κI)⁻²β* = Σ_k λ_k²/(λ_k+κ)² · (β*ᵀuₖ)²
    signal_sq = np.sum(eigenvalues ** 2 / (eigenvalues + kappa) ** 2 * beta_proj ** 2)

    # E[β̂ᵀβ̂] noise term: (σ²/n) · Tr[(Σ+κI)⁻²Σ]  = (σ²/n) · df₂_deriv
    # df₂_deriv here = Tr[(Σ+κI)⁻²Σ] = Σ_k λ_k/(λ_k+κ)²
    d = len(eigenvalues)
    weights_arr = (np.ones(d) / d if weights is None else np.asarray(weights, dtype=float))
    df2_deriv = d * np.sum(weights_arr * eigenvalues / (eigenvalues + kappa) ** 2)

    noise_sq = sigma_noise ** 2 / n * df2_deriv

    denom = signal_sq + noise_sq
    R_det = num / denom if denom > 0 else 0.0
    return R_det, num, denom


def accentuation_error_theory(eigenvalues, beta_proj, kappa, sigma_noise, n,
                               weights=None):
    """Theory prediction for accentuation evaluation error.

    E_acc ≍ (β*ᵀΣβ*) · (1 - R_det)²

    Returns
    -------
    error : float
    R_det : float
    signal_power : float  (β*ᵀΣβ*)
    """
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    beta_proj = np.asarray(beta_proj, dtype=float)

    signal_power = np.sum(eigenvalues * beta_proj ** 2)  # β*ᵀΣβ*
    R_det, _, _ = accentuation_alignment(
        eigenvalues, beta_proj, kappa, sigma_noise, n, weights)
    error = signal_power * (1 - R_det) ** 2
    return error, R_det, signal_power

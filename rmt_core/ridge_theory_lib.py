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
   R_det ≍ E[β̂ᵀβ*] / E[β̂ᵀβ̂]

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


def generalization_r2_theory(eigenvalues, beta_proj, kappa, sigma_noise, n,
                              weights=None):
    """Noiseless-signal population R² on the natural input distribution.

    R²_gen = 1 - E_gen / (β*ᵀΣβ*), where E_gen is the expected
    population prediction error of a fitted ridge model.  This definition
    scores predictions against the noiseless teacher response; it deliberately
    excludes fresh observation noise at evaluation time.

    Returns
    -------
    r2_gen : float
    gen_error : float
    signal_power : float
    """
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    beta_proj = np.asarray(beta_proj, dtype=float)
    signal_power = np.sum(eigenvalues * beta_proj ** 2)
    gen_error = ridge_error_total_theory(
        eigenvalues, beta_proj, kappa, sigma_noise, n, weights)
    r2_gen = 1.0 - gen_error / signal_power if signal_power > 0 else np.nan
    return r2_gen, gen_error, signal_power


# ──────────────────────────────────────────────
# Accentuation error
# ──────────────────────────────────────────────

def accentuation_alignment(eigenvalues, beta_proj, kappa, sigma_noise, n,
                            weights=None):
    """Compute R_det ≍ E[β̂ᵀβ*] / E[β̂ᵀβ̂].

    E[β̂ᵀβ*] = Σ_k λ_k/(λ_k+κ) · (β*ᵀuₖ)²

    E[β̂ᵀβ̂] = Σ_k (λ_k/(λ_k+κ))²·(β*ᵀuₖ)²  +  (κ²·C_sig + σ²)·df₂_deriv / (n − df₂)

    This follows from E[‖β̂‖²] = Σ_k E[(uₖᵀβ̂)²] and
    E[(uₖᵀβ̂)²] = (λ_k/(λ_k+κ))²·(β*ᵀuₖ)² + (κ²·C_sig + σ²)·λ_k/(λ_k+κ)² / (n−df₂)

    Returns
    -------
    R_det : float
    numerator, denominator : float
    """
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    beta_proj = np.asarray(beta_proj, dtype=float)

    df2 = compute_df2(eigenvalues, kappa)
    denom_eff = n - df2

    # E[β̂ᵀβ*] ≍ Σ_k λ_k/(λ_k+κ) · (β*ᵀuₖ)²
    num = np.sum(eigenvalues / (eigenvalues + kappa) * beta_proj ** 2)

    # E[β̂ᵀβ̂] — three parts:
    # (1) squared-mean signal: Σ_k (λ_k/(λ_k+κ))² · (β*ᵀuₖ)²
    signal_sq = np.sum(eigenvalues ** 2 / (eigenvalues + kappa) ** 2 * beta_proj ** 2)

    # (2+3) variance from per-PC error: (κ²·C_sig + σ²) · Tr(Σ(Σ+κI)⁻²) / (n−df₂)
    c_sig = _c_sig(eigenvalues, beta_proj, kappa)
    df2_deriv = np.sum(eigenvalues / (eigenvalues + kappa) ** 2)   # Tr(Σ(Σ+κI)⁻²)
    variance_term = (kappa ** 2 * c_sig + sigma_noise ** 2) * df2_deriv / denom_eff

    denom = signal_sq + variance_term
    R_det = num / denom if denom > 0 else 0.0
    return R_det, num, denom


def accentuation_var_R(eigenvalues, beta_proj, kappa, sigma_noise, n, R_det, D_det,
                       weights=None):
    """Deterministic equivalent for Var(R) = E[R²] - (E[R])².

    From accentuation_var_correction.tex, eq. (varR_det):

      Var(R) ≍  [σ²/n · (C_sig - 4·R_det·M₁ + 4·R_det²·M₂)
                 + 2·R_det²·σ⁴/n² · Σ_k λ_k²/(λ_k+κ)⁴]  /  D_det²

    where:
      C_sig = Σ_k λ_k·bk² / (λ_k+κ)²
      M₁    = Σ_k λ_k²·bk² / (λ_k+κ)³
      M₂    = Σ_k λ_k³·bk² / (λ_k+κ)⁴

    Parameters
    ----------
    R_det : float   — deterministic equivalent E[R]
    D_det : float   — denominator E[β̂ᵀβ̂]

    Returns
    -------
    var_R : float
    var_lin : float   — linear (noise) contribution
    var_quad : float  — quadratic correction
    """
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    beta_proj = np.asarray(beta_proj, dtype=float)
    lk = eigenvalues
    bk = beta_proj
    kp = kappa

    C_sig = np.sum(lk / (lk + kp) ** 2 * bk ** 2)
    M1    = np.sum(lk ** 2 / (lk + kp) ** 3 * bk ** 2)
    M2    = np.sum(lk ** 3 / (lk + kp) ** 4 * bk ** 2)
    T_quad_spec = np.sum(lk ** 2 / (lk + kp) ** 4)   # Tr[H²]

    var_lin  = (sigma_noise ** 2 / n) * (C_sig - 4 * R_det * M1 + 4 * R_det ** 2 * M2)
    var_quad = 2 * R_det ** 2 * (sigma_noise ** 4 / n ** 2) * T_quad_spec

    var_R = (var_lin + var_quad) / D_det ** 2
    return var_R, var_lin / D_det ** 2, var_quad / D_det ** 2


def accentuation_error_theory(eigenvalues, beta_proj, kappa, sigma_noise, n,
                               weights=None, include_var_R=False):
    """Theory prediction for accentuation evaluation error.

    Leading order:  E_acc ≍ (β*ᵀΣβ*) · (1 - R_det)²
    With Var(R):    E_acc ≍ (β*ᵀΣβ*) · [(1 - R_det)² + Var(R)]

    Parameters
    ----------
    include_var_R : bool
        If True, add the Var(R) second-order correction (eq. varR_det).

    Returns
    -------
    error : float
    R_det : float
    signal_power : float  (β*ᵀΣβ*)
    """
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    beta_proj = np.asarray(beta_proj, dtype=float)

    signal_power = np.sum(eigenvalues * beta_proj ** 2)  # β*ᵀΣβ*
    R_det, _, D_det = accentuation_alignment(
        eigenvalues, beta_proj, kappa, sigma_noise, n, weights)
    bias_sq = (1 - R_det) ** 2

    if include_var_R:
        var_R, _, _ = accentuation_var_R(
            eigenvalues, beta_proj, kappa, sigma_noise, n, R_det, D_det)
        error = signal_power * (bias_sq + var_R)
    else:
        error = signal_power * bias_sq

    return error, R_det, signal_power


def accentuation_r2_theory(eigenvalues, beta_proj, kappa, sigma_noise, n,
                            weights=None, include_delta_correction=False):
    """Deterministic-equivalent R² on a model's own accentuation path.

    For one fitted model, let R = β̂ᵀβ* / β̂ᵀβ̂.  The path-wise
    coefficient of determination in the note is

        R²_acc = 1 - (1 - 1/R)².

    The leading prediction substitutes the deterministic equivalent R_det.
    If ``include_delta_correction`` is true, a second-order delta-method term
    0.5 g''(R_det) Var(R) is added, where
    g(r) = 1 - (1 - 1/r)².  The available Var(R) formula captures the
    response-noise contribution but can miss random-design fluctuations.

    Returns
    -------
    r2_acc : float
    R_det : float
    delta_correction : float
    """
    R_det, _, D_det = accentuation_alignment(
        eigenvalues, beta_proj, kappa, sigma_noise, n, weights)
    if R_det == 0:
        return -np.inf, R_det, 0.0

    r2_acc = 1.0 - (1.0 - 1.0 / R_det) ** 2
    correction = 0.0
    if include_delta_correction:
        var_R, _, _ = accentuation_var_R(
            eigenvalues, beta_proj, kappa, sigma_noise, n, R_det, D_det,
            weights)
        # 0.5*g''(r) = (2r - 3)/r^4 for g(r)=2/r-1/r^2.
        correction = (2.0 * R_det - 3.0) / R_det ** 4 * var_R
        r2_acc += correction
    return r2_acc, R_det, correction


def peer_review_theory(eigenvalues, beta_proj, kappa, sigma_noise, n,
                       weights=None):
    """Leading deterministic equivalents for an independent peer model.

    The reviewer β̂ and path-generating peer β' are independent ridge fits
    with the same n, λ, σ, population covariance, and teacher.  With
    T = Σ(Σ+κI)⁻¹, independence gives

        β̂ᵀβ'       ≍ β*ᵀT²β*,
        β*ᵀβ'      ≍ β*ᵀTβ*,
        β'ᵀβ'       ≍ D_det.

    Therefore the peer path response gain is
    G_peer ≍ (β*ᵀT²β*)/(β*ᵀTβ*) and
    R²_peer ≍ 1-(1-G_peer)².  The peer error uses the generator's
    variance-matching normalization and is
    S[(β*ᵀT²β*-β*ᵀTβ*)/D_det]².

    These are leading, ratio-of-deterministic-equivalent predictions; they do
    not include second-order fluctuations of the two fitted vectors.

    Returns
    -------
    peer_error : float
    peer_r2 : float
    peer_gain : float
    details : dict
        Deterministic equivalents used in the ratios.
    """
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    beta_proj = np.asarray(beta_proj, dtype=float)
    shrink = eigenvalues / (eigenvalues + kappa)

    signal_power = np.sum(eigenvalues * beta_proj ** 2)
    teacher_peer = np.sum(shrink * beta_proj ** 2)       # β*ᵀTβ*
    reviewer_peer = np.sum(shrink ** 2 * beta_proj ** 2)  # β*ᵀT²β*
    _, _, peer_norm_sq = accentuation_alignment(
        eigenvalues, beta_proj, kappa, sigma_noise, n, weights)

    if teacher_peer == 0:
        peer_gain = np.nan
        peer_r2 = np.nan
    else:
        peer_gain = reviewer_peer / teacher_peer
        peer_r2 = 1.0 - (1.0 - peer_gain) ** 2

    if peer_norm_sq > 0:
        peer_error = signal_power * (
            (reviewer_peer - teacher_peer) / peer_norm_sq) ** 2
        peer_true_variance = signal_power * (teacher_peer / peer_norm_sq) ** 2
    else:
        peer_error = np.nan
        peer_true_variance = np.nan

    details = {
        'signal_power': signal_power,
        'teacher_peer_inner': teacher_peer,
        'reviewer_peer_inner': reviewer_peer,
        'peer_norm_sq': peer_norm_sq,
        'peer_true_variance': peer_true_variance,
    }
    return peer_error, peer_r2, peer_gain, details


def peer_review_r2_theory(eigenvalues, beta_proj, kappa, sigma_noise, n,
                          weights=None, include_delta_correction=False):
    """Predict peer-path R², optionally with a second-order ratio correction.

    Write independent reviewer and peer fits as h=m+ε and p=m+ζ, with
    m=Tβ* and independent fluctuations having diagonal per-PC variances
    supplied by Terms 2+3 of ``ridge_error_per_pc_theory``.  For

        C = hᵀp / (β*ᵀp),

    a delta expansion gives the O(1/n) bias and variance of C.  Applying a
    second delta expansion to R²_peer=1-(1-C)² yields

        E[R²_peer] ≈ g(C_det) + 2(1-C_det) Bias(C) - Var(C).

    This correction accounts for both finite-sample signal and label-noise
    fluctuations under the diagonal covariance deterministic equivalent.

    Returns
    -------
    r2_peer : float
        Leading value, or corrected value when requested.
    peer_gain : float
        Leading gain, or its bias-corrected value when requested.
    correction : float
        Additive R² correction (zero when not requested).
    details : dict
        Leading values plus delta-method gain bias and variance.
    """
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    beta_proj = np.asarray(beta_proj, dtype=float)
    _, leading_r2, leading_gain, base_details = peer_review_theory(
        eigenvalues, beta_proj, kappa, sigma_noise, n, weights)

    shrink = eigenvalues / (eigenvalues + kappa)
    mean_fit = shrink * beta_proj
    mean_cross = float(np.sum(mean_fit ** 2))
    teacher_cross = float(np.sum(beta_proj * mean_fit))

    _, _, finite_signal_var, noise_var = ridge_error_per_pc_theory(
        eigenvalues, beta_proj, kappa, sigma_noise, n, weights)
    fit_var = finite_signal_var + noise_var

    var_cross = (
        2.0 * np.sum(mean_fit ** 2 * fit_var) + np.sum(fit_var ** 2))
    var_teacher_cross = np.sum(beta_proj ** 2 * fit_var)
    cov_cross_teacher = np.sum(mean_fit * beta_proj * fit_var)

    if teacher_cross == 0:
        gain_bias = np.nan
        gain_var = np.nan
        correction = np.nan
        corrected_gain = np.nan
        corrected_r2 = np.nan
    else:
        gain_bias = (
            mean_cross * var_teacher_cross / teacher_cross ** 3
            - cov_cross_teacher / teacher_cross ** 2)
        gain_var = (
            var_cross / teacher_cross ** 2
            + mean_cross ** 2 * var_teacher_cross / teacher_cross ** 4
            - 2.0 * mean_cross * cov_cross_teacher / teacher_cross ** 3)
        correction = 2.0 * (1.0 - leading_gain) * gain_bias - gain_var
        corrected_gain = leading_gain + gain_bias
        corrected_r2 = leading_r2 + correction

    details = dict(base_details)
    details.update({
        'peer_gain_bias': gain_bias,
        'peer_gain_variance': gain_var,
        'peer_r2_delta_correction': correction,
        'peer_gain_delta': corrected_gain,
        'peer_r2_delta': corrected_r2,
    })
    if include_delta_correction:
        return corrected_r2, corrected_gain, correction, details
    return leading_r2, leading_gain, 0.0, details


def peer_review_error_theory(eigenvalues, beta_proj, kappa, sigma_noise, n,
                             weights=None, include_delta_correction=False):
    """Predict peer-path error with an optional second-order ratio correction.

    For L=pᵀ(h-β*) and W=pᵀp, peer error is S E[(L/W)²].  The
    leading deterministic equivalent uses E[L]²/E[W]².  The optional
    correction applies a bivariate delta expansion to f(L,W)=L²/W², using
    the same independent Gaussian per-PC fluctuation approximation as
    ``peer_review_r2_theory``.

    Returns
    -------
    peer_error : float
        Leading value, or corrected value when requested.
    leading_error : float
    correction : float
    details : dict
        Variance and covariance terms used by the expansion.
    """
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    beta_proj = np.asarray(beta_proj, dtype=float)
    leading_error, _, _, base_details = peer_review_theory(
        eigenvalues, beta_proj, kappa, sigma_noise, n, weights)

    shrink = eigenvalues / (eigenvalues + kappa)
    mean_fit = shrink * beta_proj
    _, _, finite_signal_var, noise_var = ridge_error_per_pc_theory(
        eigenvalues, beta_proj, kappa, sigma_noise, n, weights)
    fit_var = finite_signal_var + noise_var

    mean_cross = float(np.sum(mean_fit ** 2))
    teacher_cross = float(np.sum(beta_proj * mean_fit))
    mean_numerator = mean_cross - teacher_cross
    mean_denominator = mean_cross + float(np.sum(fit_var))
    signal_power = float(np.sum(eigenvalues * beta_proj ** 2))

    var_numerator = (
        np.sum(mean_fit ** 2 * fit_var)
        + np.sum((mean_fit - beta_proj) ** 2 * fit_var)
        + np.sum(fit_var ** 2))
    var_denominator = (
        4.0 * np.sum(mean_fit ** 2 * fit_var)
        + 2.0 * np.sum(fit_var ** 2))
    cov_num_den = 2.0 * np.sum(
        (mean_fit - beta_proj) * mean_fit * fit_var)

    if mean_denominator <= 0:
        correction = np.nan
        corrected_error = np.nan
    else:
        correction_normalized = (
            var_numerator / mean_denominator ** 2
            - 4.0 * mean_numerator * cov_num_den / mean_denominator ** 3
            + 3.0 * mean_numerator ** 2 * var_denominator
            / mean_denominator ** 4)
        correction = signal_power * correction_normalized
        corrected_error = leading_error + correction

    details = dict(base_details)
    details.update({
        'peer_error_delta_correction': correction,
        'peer_error_delta': corrected_error,
        'peer_numerator_variance': var_numerator,
        'peer_norm_variance': var_denominator,
        'peer_numerator_norm_covariance': cov_num_den,
    })
    if include_delta_correction:
        return corrected_error, leading_error, correction, details
    return leading_error, leading_error, 0.0, details

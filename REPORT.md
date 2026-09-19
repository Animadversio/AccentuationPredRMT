# RMT Theory Validation: Ridge Regression Weight Deviation & Accentuation Error

## Overview

This repo validates the RMT-based closed-form formulas derived in
`linear_reg_weight_deviation.tex` (Writings/AccentuationPredictionRMTDissociation).
We test across three data spectra (isotropic, power-law, van Hateren natural images)
and a range of dimensions (d = 40 → 10,000) using GPU-accelerated Monte Carlo.

---

## Theory

For ridge regression β̂ = (XᵀX + nλI)⁻¹Xᵀy with x ~ N(0, Σ), as n, d → ∞
with γ = d/n fixed, the **deterministic equivalent** κ = κ(λ) solves:

```
κ − λ = γ · κ · (1/d) Σ_k λ_k / (κ + λ_k)
```

### 1. Per-PC squared error (three terms)

```
E[(uₖᵀ(β̂ − β*))²]  ≍

    κ²/(λₖ+κ)² · (β*ᵀuₖ)²                       [Term 1: overshrinkage bias]
  + κ²λₖ/(λₖ+κ)² · C_sig / (n − df₂)            [Term 2: finite-sample signal]
  + σ²λₖ/(λₖ+κ)² / (n − df₂)                    [Term 3: finite-sample noise]
```

Global quantities:
- `df₂ = Σ_k λₖ² / (λₖ+κ)²`  (effective degrees of freedom)
- `C_sig = Σ_k λₖ(β*ᵀuₖ)² / (λₖ+κ)²`  (signal strength)

### 2. Accentuation alignment

```
R_det  =  E[β̂ᵀβ*] / E[β̂ᵀβ̂]

       ≍  Σ_k λₖ/(λₖ+κ)·(β*ᵀuₖ)²
          ─────────────────────────────────────────────────────────────
          Σ_k (λₖ/(λₖ+κ))²·(β*ᵀuₖ)²  +  (κ²·C_sig + σ²)·df₂' / (n−df₂)
```

where `df₂' = Tr(Σ(Σ+κI)⁻²) = Σ_k λₖ/(λₖ+κ)²`.

### 3. Accentuation error

```
E_acc ≍ (β*ᵀΣβ*) · (1 − R_det)²
```

---

## Validation results

### Small d: Per-PC error (isotropic, power-law, van Hateren)

| Spectrum | d | n | γ | λ | Gen error theory | Gen error MC |
|---|---|---|---|---|---|---|
| Isotropic | 40 | 100 | 0.40 | 0.10 | ✓ | ✓ |
| Power-law (α=1) | 64 | 200 | 0.32 | 0.05 | ✓ | ✓ |
| Van Hateren 8×8 | 64 | 500 | 0.13 | 0.05 | ✓ | ✓ |

Theory matches MC per-PC across all PCs, with correct decomposition
into the three terms. See `figures/ridge_error/nb_small_d_per_pc.png`.

### Large d: GPU validation (A100 40 GB)

Key implementation details for large d:
- **Kernel/dual form** for ridge estimator (n×n solve, O(n²d) not O(nd²))
- **Randomized SVD** (torch.svd_lowrank) for van Hateren covariance
- **Diagonal shortcut** for isotropic/power-law: X = Z · sqrt(λ) (no eigenvector matrix)
- For low-rank van Hateren: use γ_eff = K/n (rank of data / n)

| Spectrum | d | n | γ | Theory | MC | Match |
|---|---|---|---|---|---|---|
| Power-law α=1 | 1,000 | 1,000 | 1.0 | 5.641 | 5.621 | **<0.4%** |
| Power-law α=1 | 5,000 | 2,000 | 2.5 | 63.38 | 63.28 | **<0.2%** |
| Van Hateren 32×32 | 1,024 | 2,000 | 0.05* | 0.564 | 0.564 | **<0.1%** |
| Van Hateren 100×100 | 10,000 | 5,000 | 0.04* | 0.214 | 0.214 | **<0.1%** |

*γ_eff = K/n for van Hateren (K = number of retained PCs)

See `figures/ridge_error/nb_large_d.png` and
`figures/ridge_error/ridge_error_large_*.png`.

### σ sweep: varying noise level

Theory and MC tracked across σ ∈ [0.05, 3.0] at d=256, n=512, λ=0.05.

**Generalization error** (Σ_k λₖ E[(uₖᵀΔβ)²]):
- Theory matches MC to < 1% across the full noise range for both power-law and van Hateren.

**Accentuation alignment R**:
- After the formula bug fix (see below), theory matches MC to < 0.1%.

See `figures/accentuation/sigma_sweep_fixed_accentuation.png` and
`tables/sigma_sweep_*.npz`.

### λ sweep: varying ridge penalty

Theory predictions for total generalization error and accentuation error
vs λ ∈ [10⁻³, 10] (power-law d=256, n=512, σ=1.0):
- Both errors are monotonically increasing in λ (more regularization → more bias).
- R_det monotonically decreases toward 1 as λ → 0 (no regularization), and drops below 1 for large λ.
- Optimal λ (min gen error) can be read off from theory curve.

See `figures/accentuation/nb_lambda_sweep.png`.

---

## Bug fix: incorrect `E[β̂ᵀβ̂]` formula in accentuation alignment

### What was wrong

The original denominator formula for R_det was:

```
❌ WRONG:
E[β̂ᵀβ̂]  ≍  β*ᵀΣ²(Σ+κI)⁻²β*  +  (σ²/n) · Tr(Σ(Σ+κI)⁻²)
          =  Σ_k λₖ²/(λₖ+κ)² · (β*ᵀuₖ)²  +  (σ²/n) · Σ_k λₖ/(λₖ+κ)²
```

This caused a **~3% overestimate of R_det** (theory 1.256 vs MC 1.217 at d=256, n=512, σ=1).

### Diagnostic

Decomposing MC into signal-only and noise-only components revealed:

| Component | Theory (wrong) | MC | Ratio |
|---|---|---|---|
| E[β̂ᵀβ*] (numerator) | 176.14 | 176.13 | **0.9999** ✓ |
| E[β̂ᵀβ̂] signal part  | 139.39 | 143.40 | 1.029 ✗ |
| E[β̂ᵀβ̂] noise part   | 0.876  | 1.300  | 1.485 ✗ |
| E[β̂ᵀβ̂] total        | 140.27 | 144.73 | 1.032 ✗ |

The numerator was **exact**. Both parts of the denominator were underestimated.

### Root cause

`E[β̂ᵀβ̂]` must be computed as `Σ_k E[(uₖᵀβ̂)²]`, not `Σ_k (E[uₖᵀβ̂])²`.

The correct expansion via the second-moment identity:

```
E[(uₖᵀβ̂)²]  =  E[(uₖᵀβ*)² of (λₖ/(λₖ+κ))²]   ← mean-squared
              +  E[(uₖᵀΔβ)²]                     ← variance (per-PC error)
              −  (bias of uₖᵀβ̂)²                 ← subtract squared bias back

= (λₖ/(λₖ+κ))²·(β*ᵀuₖ)²  +  (κ²·C_sig + σ²) · λₖ/(λₖ+κ)² / (n−df₂)
```

where the `κ²·C_sig / (n−df₂)` and the `/(n−df₂)` (not `/n`) both come from
summing Terms 2 and 3 of the per-PC error formula and canceling with the bias² term.

### Correct formula

```
✓ CORRECT:
E[β̂ᵀβ̂]  =  Σ_k (λₖ/(λₖ+κ))²·(β*ᵀuₖ)²
           +  (κ²·C_sig + σ²) · Tr(Σ(Σ+κI)⁻²) / (n − df₂)

         =  Σ_k (λₖ/(λₖ+κ))²·(β*ᵀuₖ)²
           +  (κ²·C_sig + σ²) · df₂' / (n − df₂)
```

where `df₂' = Σ_k λₖ/(λₖ+κ)²` and `df₂ = Σ_k λₖ²/(λₖ+κ)²`.

### What was missing from the original formula

Two missing terms:

1. **The finite-sample signal variance** `κ²·C_sig·df₂'/(n−df₂)`:
   This accounts for the random variation of β̂ in the signal subspace
   (the same term that appears as Term 2 in the per-PC error). At d=256, n=512:
   `κ²·C_sig·df₂'/(n−df₂) ≈ 4.02` — the dominant missing piece.

2. **Wrong denominator in the noise term**: `σ²/n → σ²/(n−df₂)`.
   The effective sample size is `n−df₂` (degrees of freedom remaining after
   the ridge fit), not n. At this setting, `df₂ = 169` so `n−df₂ = 343` vs n=512 — a factor of 1.49.

### After fix

| Quantity | Theory (fixed) | MC | Gap |
|---|---|---|---|
| E[β̂ᵀβ̂] | 144.72 | 144.73 | **0.006%** |
| R_det | 1.2171 | 1.2174 | **0.02%** |

Theory/MC gap on R_det drops from **3% → 0.02%** across all σ values.
See `figures/accentuation/sigma_sweep_fixed_accentuation.png`.

---

## Other implementation notes

1. **κ solver initial guess for γ > 1 (over-parameterized regime):**
   Starting from κ_init = λ leads to the wrong (negative) branch. Fix: κ_init = λ + γ·mean(λ_k).

2. **df₂ and C_sig must use the full spectrum:**
   When evaluating only top-K PCs, these global quantities must still be summed over all d eigenvalues.

3. **Van Hateren γ_eff = K/n:**
   Data generated from rank-K Gaussian (low-rank SVD) has an effective aspect ratio K/n, not d_patch/n.

4. **Randomized SVD convention:**
   `torch.svd_lowrank` returns V (d, q) with eigenvectors as *columns* (not rows), i.e. use `V[:, :k]`.

---

## Repo structure

```
AccentuationPredRMT/
├── rmt_core/
│   ├── kappa_lib.py          # κ(λ) solver (scipy + analytic continuation)
│   ├── ridge_theory_lib.py   # Three-term formula, df₂, C_sig, accentuation
│   ├── simulation_lib.py     # CPU Monte Carlo
│   ├── gpu_simulation_lib.py # GPU Monte Carlo (dual form, diagonal, low-rank)
│   └── spectrum_lib.py       # Isotropic, power-law, van Hateren spectra
├── scripts/
│   ├── validate_ridge_error_formula.py  # Small-d per-PC validation (CLI)
│   ├── validate_large_d.py              # Large-d GPU validation (CLI)
│   └── validate_sigma_sweep.py          # σ sweep across noise levels (CLI)
├── notebooks/
│   └── validation_overview.ipynb        # Recreates all plots interactively
├── figures/                             # Outputs grouped by analysis/topic
│   └── README.md                        # Figure taxonomy and folder index
├── tables/                              # Saved .npz result tables
└── REPORT.md                            # This file
```

---

## Theory vs reality: accentuation error and the Var(R) gap

### What the theory computes

By default, `accentuation_error_theory` returns the leading term:

```
E_acc_theory  =  (β*ᵀΣβ*) · (1 − R_det)²
```

where `R_det` is the deterministic equivalent of the alignment ratio
`R = β̂ᵀβ* / β̂ᵀβ̂`.  This is a leading-order approximation that captures
only the squared-mean term.

### The full second-moment decomposition

The Monte Carlo estimator computes `E[(1−R)²]`.  By the bias-variance
decomposition:

```
E[(1−R)²]  =  (1 − E[R])²  +  Var(R)
            ┗━━━━━━━━━━━━━┛    ┗━━━━━┛
               theory term     missing term
```

The leading theory gives `E[R]` (to leading order in 1/d), but has **no
Var(R) term**.  When `R ≈ 1` (well-aligned β̂), `(1−E[R])²` becomes
tiny and `Var(R)` dominates.  The current implementation optionally adds an
approximate noise-driven correction with `include_var_R=True`.  It captures
the correct scale but can underestimate total variance when random-design
(`X`) fluctuations dominate.

### Numerical diagnosis — van Hateren 16×16, n=512, optimal λ*(σ)

At each σ, 500 MC trials were run at the theory-optimal λ*(σ).
`std(R)` is the trial-to-trial fluctuation of the alignment ratio.

| σ | lam* | E[R]_det | E[R]_MC | std(R) | (β*ᵀΣβ*)·(1−E[R])² | (β*ᵀΣβ*)·Var(R) | Var/bias² |
|---|---|---|---|---|---|---|---|
| 0.10 | 0.001 | 1.170 | 1.170 | 0.009 | 1.856 | 0.005 | 0.003 |
| 0.30 | 0.001 | 1.140 | 1.142 | 0.013 | 1.285 | 0.010 | 0.008 |
| 0.58 | 0.001 | 1.058 | 1.058 | 0.018 | 0.214 | 0.021 | 0.099 |
| 1.00 | 0.002 | 1.010 | 1.011 | 0.026 | 0.007 | 0.042 | 5.7 |
| 2.00 | 0.007 | 0.995 | 0.998 | 0.038 | 0.0004 | 0.093 | 252 |

Key observations:
- `E[R]` is predicted accurately at all σ (theory tracks MC).
- `std(R)` grows slowly with σ (O(1/√n) fluctuations).
- At σ ≲ 0.3, the bias² term dominates and theory matches MC to < 1%.
- At σ ≳ 1, `Var(R)` is 5–250× larger than bias² — theory is off by
  an order of magnitude for E_acc.
- This is a limitation of the leading deterministic equivalent, not a code
  bug. The optional correction is useful but is not a complete second-order
  random-matrix expansion.

---

## Generalization error vs accentuation error: crossing analysis

### Fixed λ

For a fixed ridge penalty, both errors grow with σ but at different
rates.  Generalization error grows as σ² (noise-dominated at large σ),
while accentuation error saturates (R → 0 as σ → ∞, but the Var term
keeps E_acc finite).

See `figures/accentuation/gen_vs_acc_error_crossing.png`.

### Optimal λ*(σ)

At each σ the ridge penalty is chosen to minimize theory generalization
error.  The key question: is there a noise level σ* where accentuation
error overtakes generalization error?

**Van Hateren 16×16 (d=256, n=512):**

| | Theory | MC |
|---|---|---|
| Crossing σ* | 0.572 | 0.582 |

Theory and MC agree to **~1.7%** at the crossing.  The agreement holds
because at the crossing point σ≈0.58, `Var(R)/bias² ≈ 0.10` — Var(R)
contributes only ~10% to E_acc, so the theory is still a good
approximation.

At σ > 0.58 theory underestimates E_acc badly (Var dominates), but that
regime is _above_ the crossing and does not affect the location of σ*.

See `figures/accentuation/gen_vs_acc_optimal_lambda_vh.png`.

**Interpretation:**
- At low noise (σ < σ*): accentuation error is the larger concern.
  The estimator is biased toward high-variance PCs (overshrinkage), so
  the accentuation of β̂ is inflated relative to β*.
- At high noise (σ > σ*): generalization error dominates.  Too much
  noise washes out the signal, and regularization cannot help.
- The crossing σ* moves with the spectrum: spiky spectra (van Hateren)
  have large accentuation bias even at moderate λ, shifting σ* higher.

---

## R² and independent-peer deterministic equivalents

### Definitions

Let `S = β*ᵀΣβ*` and let the evaluation target be the noiseless teacher
response. Then

```
R²_gen = 1 - E_gen/S.
```

For a model's own zero-seed accentuation path, define
`R = β̂ᵀβ*/(β̂ᵀβ̂)`. The note's fixed-calibration score is

```
R²_acc = 1 - (1 - 1/R)².
```

This is not squared Pearson correlation; on a noiseless one-dimensional path,
Pearson correlation has magnitude one. `R²_acc` can be negative when the
predicted response has sufficiently bad scale calibration.

For peer review, an independent ridge fit `β'` generates the path and `β̂`
evaluates it:

```
G_peer  = β̂ᵀβ' / (β*ᵀβ')
R²_peer = 1 - (1 - G_peer)².
```

### Leading peer deterministic equivalent

Let `t_k = λ_k/(λ_k+κ)`, `b_k = β*ᵀu_k`, and
`T = Σ(Σ+κI)⁻¹`. For two independent fits,

```
β̂ᵀβ'  ≍ β*ᵀT²β* = Σ_k t_k² b_k² = M
β*ᵀβ' ≍ β*ᵀT β* = Σ_k t_k  b_k² = N
```

so

```
G_peer,det  = M/N
R²_peer,det = 1 - (1-M/N)².
```

If `v_k` is the sum of the finite-sample signal and label-noise terms
(Terms 2+3 in the per-PC error formula), then the peer norm deterministic
equivalent is `D = M + Σ_k v_k`, giving

```
E_peer,det = S [(M-N)/D]².
```

### Second-order peer correction

Approximating the two independent fits as `m+ε` and `m+ζ`, with
`m_k=t_k b_k` and diagonal fluctuation variance `v_k`, gives delta-method
corrections for both peer metrics. For `U=β̂ᵀβ'` and `V=β*ᵀβ'`:

```
Var(U) = 2 Σ_k m_k²v_k + Σ_k v_k²
Var(V) = Σ_k b_k²v_k
Cov(U,V) = Σ_k m_k b_k v_k

Bias(G_peer) ≈ M Var(V)/N³ - Cov(U,V)/N²
Var(G_peer)  ≈ Var(U)/N² + M²Var(V)/N⁴ - 2M Cov(U,V)/N³

E[R²_peer] ≈ 1-(1-M/N)²
                   + 2(1-M/N) Bias(G_peer) - Var(G_peer).
```

`peer_review_error_theory` applies the analogous bivariate expansion to
`E_peer = S E[(L/W)²]`, where `L=β'ᵀ(β̂-β*)` and `W=β'ᵀβ'`.

### Validation

We tested `d=128`, `n=256`, `λ=0.05`, and `σ∈{0.1,0.6,1.0}` on
isotropic, power-law `α=1`, and steeper power-law `α=1.5` spectra. Each
of the nine conditions used 1,000 paired trials (two independent fits per
trial), with teacher signal power normalized to one.

| Quantity | Leading DE mean abs. gap | Corrected mean abs. gap | Corrected max gap |
|---|---:|---:|---:|
| R²_gen | 0.00148 | — | 0.00554 |
| R²_acc | 0.02233 | 0.00573 | 0.01790 |
| R²_peer | 0.00899 | 0.00113 | 0.00495 |
| E_gen | 0.00148 | — | 0.00554 |
| E_acc | 0.00108 | 0.00109 | 0.00456 |
| E_peer | 0.00303 | 0.00042 | 0.00141 |

The main conclusions are:

- The existing generalization deterministic equivalent directly and
  accurately predicts `R²_gen` after division by `S`.
- Leading plug-in formulas accurately predict own-path and peer-path R² at
  low/moderate noise. Nonlinear ratio fluctuations matter at high noise.
- The own-path delta correction materially improves `R²_acc`, although its
  incomplete `Var(R)` approximation leaves a visible residual in the hardest
  cases.
- The new independent-peer correction performs especially well: its maximum
  R² gap is below 0.005 across the nine tested conditions.

See `figures/peer_validation/r2_peer_validation.png`,
`figures/peer_validation/error_peer_validation.png`, and
the plot-ready cache `tables/r2_peer_validation_summary.csv`. Compressed
per-trial caches are in `tables/r2_peer_validation_cases/` and are intentionally
ignored by git.

---

## Cross-validated lambda from deterministic equivalents

### Fold-size correction

In K-fold cross-validation, each validation score is produced by a model fitted
on

```
n_cv = n (K-1)/K
```

samples. For a candidate lambda, expected validation error against noisy labels
is

```
E[CV_K(lambda)] = E_gen(lambda; n_cv) + sigma².
```

The fresh validation-noise term is constant in lambda. Therefore a leading
deterministic-equivalent prediction of the cross-validated penalty is

```
lambda_DE,CV = argmin_lambda E_gen,DE(lambda; n_cv).
```

After selecting this penalty, the final model is refitted on all `n` samples,
so its R² values are evaluated using `kappa(lambda_DE,CV; d/n)` and the full
sample-size formulas. Minimizing the full-`n` risk instead predicts the oracle
penalty for a model trained directly on all data; it is generally smaller than
the K-fold-selected penalty.

This approximation predicts the center of the CV-selected-lambda distribution.
It does not, at leading order, predict finite-sample fluctuations of the CV risk
curve or the resulting random argmin.

### Actual-CV experiment

We used the same three spectra and signal normalization as above, now with
`sigma in {0.1, 0.3, 0.6, 1.0}`. Each of the 12 conditions used 300 paired
trials. Every fitted model independently selected lambda by 5-fold CV from a
61-point logarithmic grid spanning `1e-4` to `10`, then refit on all `n=256`
samples. Ridge paths were evaluated with one eigendecomposition per fold.

The DE fold-size optimum matched the actual CV median exactly on the candidate
grid in 10 of 12 conditions and was within one grid step in the remaining two.
No condition selected the upper grid boundary; the largest lower-boundary
selection fraction was 1.7% at the lowest noise.

| Quantity | Mean absolute DE/MC gap | Maximum gap |
|---|---:|---:|
| R²_gen | 0.00348 | 0.00795 |
| R²_acc, leading | 0.00975 | 0.03801 |
| R²_acc, corrected | 0.00882 | 0.03358 |
| R²_peer, leading | 0.00480 | 0.01196 |
| R²_peer, corrected | 0.00449 | 0.01453 |

At the highest noise level:

| Spectrum | lambda DE-CV / CV median | R²_gen DE / MC | R²_acc corrected DE / MC | R²_peer corrected DE / MC |
|---|---:|---:|---:|---:|
| Isotropic | 0.681 / 0.681 | 0.577 / 0.570 | 0.987 / 0.977 | 0.768 / 0.782 |
| Power law alpha=1 | 0.464 / 0.464 | 0.661 / 0.654 | 0.985 / 0.964 | 0.739 / 0.748 |
| Power law alpha=1.5 | 0.316 / 0.316 | 0.808 / 0.800 | 0.995 / 0.962 | 0.798 / 0.795 |

The broad conclusion is that DE theory can predict both cross-validated lambda
and downstream R² well. The main residual is own-path R² at high noise, where
CV-selection variability compounds the incomplete random-design contribution
in the current `Var(R)` correction.

Scientifically, generalization-selected regularization keeps own-path
`R²_acc` close to one over the whole noise range, while independent-peer
`R²_peer` falls to roughly 0.74-0.80 at high noise. Thus self-accentuation can
remain apparently well calibrated even when independently trained models no
longer agree strongly on the accentuation direction.

See `figures/model_selection/cv_selected_lambda.png`,
`figures/model_selection/cv_selected_r2.png`, and the
plot-ready cache `tables/cv_selected_r2_summary.csv`. Per-trial selected
penalties, metrics, and DE risk paths are cached under
`tables/cv_selected_r2_cases/` and ignored by git.

---

## Power-law teacher spectral-alignment sweep

### Controlled teacher family

For a power-law covariance with `alpha=1`, we controlled the teacher through
its per-PC allocation of natural response variance,

```
s_k = lambda_k (u_k^T beta*)^2,       sum_k s_k = 1.
```

The localized family places a Gaussian bump of width 0.10 in normalized PC
rank, with center swept from 0.05 (top eigenspace) to 0.95 (bottom
eigenspace). This construction keeps teacher signal power exactly one while
moving where that signal lives. It necessarily gives bottom-aligned teachers
larger raw coefficients because low-variance input directions require larger
weights to produce the same response variance. An i.i.d.-Gaussian coefficient
teacher, separately normalized to unit signal power, provides a diffuse random
reference.

For each of 8 teachers and 4 noise levels, 200 paired trials independently
selected lambda by actual 5-fold CV on the same 61-point grid used by the DE
calculation. Raw fitted weights were retained and projected onto the population
eigenbasis. The coefficient plots show sign-aligned means and 10--90% bands;
the dashed curve is the deterministic mean
`lambda_k/(lambda_k+kappa) beta_k*` at the DE-CV penalty.

### Main result

At `sigma=1`, moving unit teacher signal toward lower-variance PCs produces a
large, monotone loss of natural and peer performance, while own-path R² remains
much more optimistic:

| Teacher | Signal-rank centroid | lambda DE-CV / CV median | R²_gen DE / MC | R²_acc corrected DE / MC | R²_peer corrected DE / MC |
|---|---:|---:|---:|---:|---:|
| Top localized | 0.101 | 0.562 / 0.562 | 0.791 / 0.785 | 0.976 / 0.959 | 0.888 / 0.883 |
| Middle localized | 0.500 | 0.215 / 0.215 | 0.518 / 0.508 | 0.988 / 0.971 | 0.789 / 0.786 |
| Bottom localized | 0.899 | 0.147 / 0.147 | 0.360 / 0.326 | 0.961 / 0.889 | 0.730 / 0.677 |
| i.i.d. random | 0.211 | 0.464 / 0.562 | 0.700 / 0.689 | 0.996 / 0.967 | 0.776 / 0.758 |

Across all 32 conditions, the corrected DE/MC mean absolute gaps were 0.0056
for `R²_gen`, 0.0100 for `R²_acc`, and 0.0060 for `R²_peer`. The DE-CV lambda
matched the actual CV median exactly on the grid in 28/32 cases. The largest
gaps occur for the bottom-aligned teacher at the highest noise, where the
selected-lambda distribution and nonlinear ratio fluctuations are widest.

The random teacher has nearly the same signal-rank centroid as the localized
0.20 teacher and nearly the same high-noise `R²_gen`, but substantially lower
`R²_peer` (0.758 versus 0.853). Thus a single spectral centroid is not a
sufficient description of peer agreement: the full distribution of teacher
power over PCs matters because ridge shrinkage varies mode by mode.

See `figures/teacher_alignment/powerlaw_teacher_alignment_r2.png`,
`figures/teacher_alignment/powerlaw_teacher_alignment_lambda.png`,
`figures/teacher_alignment/powerlaw_teacher_signal_profiles.png`,
`figures/teacher_alignment/powerlaw_teacher_weights_eigenbasis.png`, and
`figures/teacher_alignment/powerlaw_teacher_response_weights_eigenbasis.png`.
Plot-ready data are
stored in `tables/powerlaw_teacher_alignment_summary.csv` and
`tables/powerlaw_teacher_weight_summary.csv`; raw per-case metrics and fitted
coefficient ensembles are cached under
`tables/powerlaw_teacher_alignment_cases/`.

---

## Natural-image unit-disk teacher

### Mapping to the original experiment

The exact source experiment is
`Closed-loop-visual-insilico/notebooks/toy_model_linear_regression_sweep.py`
(converted from `20250528_toy_model_adv_generation.ipynb`). It uses 100x100
grayscale FFHQ images (`d=10,000`), the binary radius-0.3 central disk as
`beta*`, `n=1,000`, a fitted intercept, and scikit-learn `RidgeCV` with
`alpha in 10**[-4,...,5]`. Its saved coefficient maps and train/test scores
are under `AdvExampleLinearRegr/circ_mask_weights`. The downstream natural
image accentuation code is in
`Closed-loop-visual-insilico/scripts/accentuation_theory/exp2_accentuation.py`.
The reproduction below replaces the original ten decade-spaced candidates
with 181 log-spaced values over the same range (20 intervals per decade).
DE and empirical LOOCV always select from this identical dense grid.

Scikit-learn solves with `X.T X + alpha I`, whereas this repository uses
`X.T X + n lambda I`; hence the final-fit conversion is
`lambda=alpha/n`. `RidgeCV(cv=None)` is leave-one-out CV. We therefore select
`alpha` by minimizing the DE generalization error at sample size `n-1` with
`lambda=alpha/(n-1)`, then evaluate the final estimator at full `n`.

The evaluation slopes use the orientation in the response scatterplots: true
teacher response regressed on fitted-model response. On centered natural
images and on the fitted model's own accentuation path, respectively,

```
slope_gen = (beta_hat^T Sigma beta*) / (beta_hat^T Sigma beta_hat)
slope_acc = (beta_hat^T beta*)       / (beta_hat^T beta_hat).
```

The natural-image slope remains extremely close to one (MC 1.002 at
`sigma=1`, 1.008 at `sigma=10`) even as the accentuation slope falls to 0.945
and 0.820. This exposes the calibration dissociation more directly than error
alone: natural responses have almost unit slope while the teacher achieves
only 94.5% or 82.0% of the response change requested along the student weight.

### Exact-spectrum validation

We estimated the centered FFHQ population covariance from 20,000 images
independent of a 10,000-image training pool and diagonalized its full
10,000x10,000 spectrum on an H100. For each noise level, 100 natural-image
training sets were fit with exact analytic LOOCV. The DE used only this
population spectrum, the disk teacher's PC projections, `n`, and `sigma`.
The original saved Ridge alpha was not stored, so its value below is inferred
from the paired saved OLS/Ridge coefficient maps; it is least identifiable
when the low-noise CV risk curve is flat.

| sigma | alpha DE / MC median / saved inferred | squared weight error DE / MC / saved | accentuation error DE / MC / saved |
|---:|---:|---:|---:|
| 0.1 | 1e-4 / 1e-4 / 8.9e-4 | 69.9 / 69.0 / 67.5 | 0.00181 / 0.00295 / 0.00197 |
| 1 | 1e-4 / 1e-4 / 1 | 101.0 / 104.7 / 98.8 | 15.7 / 20.0 / 14.4 |
| 3 | 15.8 / 15.8 / 10 | 167.3 / 169.3 / 194.0 | 87.3 / 91.2 / 154.7 |
| 10 | 100 / 100 / 100 | 269.6 / 273.1 / 270.5 | 234.3 / 249.3 / 252.4 |

DE closely tracks the actual-CV median alpha on the dense candidate grid. At
`sigma=3`, for example, both select 15.8 rather than being quantized to the
original decade value 10. The fitted metrics remain close; residual gaps are
largest for nonlinear accentuation quantities, whose finite-sample alignment
ratio has appreciable variance. At very low noise relative accentuation gaps
look large only because both errors are nearly zero.

This gives a direct explanation of the visual result. Even at zero noise,
`n=1,000` observations cannot identify all 10,000 raw pixels: the squared
coefficient error is about 69 although the disk teacher has norm squared 692.
Most residual error lies in low-variance FFHQ PCs, so its natural-image cost is
small (`R2_gen` is about 0.9997), while those raw coefficient components can
strongly affect optimization along the fitted weight direction. Individual
fits consequently contain face/texture artifacts, even though their mean and
the DE mean recover a clean disk-like shape. This is precisely the
prediction/accentuation dissociation in the note, now observed beyond the
Gaussian-design assumption using real natural images.

See `figures/natural_image_disk_teacher/ffhq_disk_teacher_de_validation.png`,
`figures/natural_image_disk_teacher/ffhq_disk_teacher_gen_acc_gap.png`,
`figures/natural_image_disk_teacher/ffhq_disk_teacher_weights.png`, and
`figures/natural_image_disk_teacher/ffhq_disk_teacher_eigenbasis.png`.
Plot-ready results are in
`tables/ffhq_disk_teacher_de_summary.csv`; reusable per-condition arrays are
cached under `tables/ffhq_disk_teacher_cases/`. The default processing log is
`logs/ffhq_disk_teacher_de.log`; the dense-grid run reported here is preserved
in `logs/ffhq_disk_teacher_dense_alpha.log`.

### Computational audit

Loading separate PNG files was I/O-bound at about 16 images/s (roughly 12
minutes for 12,000 images). Sequential reads from the contiguous FFHQ zip
reached 251 images/s, staging 30,000 resized images to node-local scratch in
119 seconds. Once staged, the H100 covariance and eigendecomposition took 1.2
seconds. An exact-size 10-trial timing probe projected 2.5 minutes for the
dense grid; the complete 2,600 fitted trials actually took about 40 seconds,
and the full post-spectrum workload including figures took 50.4 seconds. Both
the staged array and eigenvectors are cached on node-local scratch; compact
spectrum, per-case numerical summaries, and plot-ready CSV are cached
separately so figures can be restyled without recomputation.

An implementation audit also found and fixed float32 cancellation in analytic
LOOCV at very small alpha. Forming `(I-H)y` directly in the sample eigensystem,
rather than subtracting two nearly equal vectors `y-Hy`, makes the selector
agree with scikit-learn on the original FFHQ design. A regression test now
compares this stable path with brute-force leave-one-out fits.

### High-noise extension

The cached dense-grid sweep contains 26 noise levels and reaches exact population
noise/signal variance ratios `sigma²/S` of 0.1, 1, and 10, where
`S=6778.3`. Every condition uses 100 actual natural-image RidgeCV fits. The
comparison figure has synchronized axes: response-noise SD `sigma` below and
noise/signal variance ratio above. Both DE and empirical LOOCV select from 181
log-spaced alphas between `1e-4` and `1e5`; adjacent candidates differ by only
1.122x.

| sigma²/S | sigma | alpha DE / MC median [IQR] | R²_gen DE / MC | slope_acc DE / MC | R²_acc DE / MC mean (median) |
|---:|---:|---:|---:|---:|---:|
| 0.1 | 26.04 | 355 / 336 [316,355] | 0.986 / 0.986 | 0.762 / 0.749 | 0.902 / 0.874 (0.896) |
| 1 | 82.33 | 1585 / 1585 [1259,1778] | 0.947 / 0.946 | 0.706 / 0.685 | 0.822 / 0.654 (0.817) |
| 10 | 260.35 | 7079 / 7079 [5623,7943] | 0.812 / 0.805 | 0.636 / 0.643 | 0.639 / 0.261 (0.612) |

The dense grid removes the artificial decade-scale dip-and-recovery. Between
`sigma=10` and 100, the largest adjacent change in selected alpha falls from
10x to 1.58x; total variation of the empirical accentuation-R² curve falls from
6.75 to 0.34, and total variation of squared weight error falls from 1591 to
270. Generalization remains especially well predicted. Own-path R² still
requires care at extreme noise: DE stays near the empirical median, while its
nonlinear alignment ratio leaves the empirical mean lower and increasingly
uncertain. The plot therefore retains both MC mean with standard error and MC
median with IQR.

### Fixed regularization versus RidgeCV

To isolate the effect of regularization selection, we compared the dense-grid
RidgeCV fits with a fixed alpha=100, or lambda=alpha/n=0.1. This fixed value is
anchored at sigma=10, where the DE and empirical RidgeCV selector also choose
alpha=100; the two policies are therefore identical there in theory and differ
only by Monte Carlo sampling error.

For each candidate alpha, the DE solves the kappa fixed point and predicts the
final-fit metrics. The CV branch uses

    alpha_CV_DE(sigma)
        = argmin_alpha E_gen_DE(n-1, sigma, lambda=alpha/(n-1)),
    lambda_final = alpha_CV_DE / n.

The fixed branch instead evaluates every noise level at lambda=0.1. Both use
the same per-PC DE moments. Natural error aggregates them with population
eigenvalues, while own-path accentuation uses Euclidean weight geometry:

    E_gen_DE = sum_j s_j E[(beta_hat_j-beta_j*)^2]
    slope_acc_DE = E[beta_hat^T beta*] / E[beta_hat^T beta_hat]
    E_acc_DE / S = (1-slope_acc_DE)^2
    R2_acc_DE = 1-(1-E[beta_hat^T beta_hat]/E[beta_hat^T beta*])^2.

The Monte Carlo comparison is paired: at each of the 26 noise levels, the 100
fixed-alpha fits use the same natural-image subsets and response-noise draws as
the cached RidgeCV experiment.

| sigma²/S | policy | E_gen/S DE / MC | E_acc/S DE / MC | R²_gen DE / MC | R²_acc DE / MC median |
|---:|:---|---:|---:|---:|---:|
| 0.01 | RidgeCV | 0.00358 / 0.00375 | 0.0275 / 0.0309 | 0.996 / 0.996 | 0.961 / 0.959 |
| 0.01 | fixed | 0.00365 / 0.00384 | 0.0129 / 0.0115 | 0.996 / 0.996 | 0.984 / 0.986 |
| 0.1 | RidgeCV | 0.0141 / 0.0141 | 0.0567 / 0.0657 | 0.986 / 0.986 | 0.902 / 0.896 |
| 0.1 | fixed | 0.0201 / 0.0197 | 0.449 / 0.437 | 0.980 / 0.980 | -3.13 / -2.82 |
| 1 | RidgeCV | 0.0533 / 0.0543 | 0.0870 / 0.111 | 0.947 / 0.946 | 0.822 / 0.817 |
| 1 | fixed | 0.185 / 0.182 | 0.911 / 0.909 | 0.815 / 0.818 | -453 / -414 |
| 10 | RidgeCV | 0.188 / 0.195 | 0.136 / 0.154 | 0.812 / 0.805 | 0.639 / 0.612 |
| 10 | fixed | 1.83 / 1.81 | 0.991 / 0.990 | -0.830 / -0.812 | -5.63e4 / -4.05e4 |

The fixed value has a narrow useful range and can even give slightly better
accentuation calibration near sigma²/S=0.01. Beyond that range it becomes
severely under-regularized. RidgeCV raises alpha from 1e-4 at very low noise to
about 7.1e3 at sigma²/S=10, suppressing noisy low-variance weight components.
This keeps both natural and accentuation performance usable.

The apparently different high-noise limits of E_acc/S and R²_acc are not
contradictory. Fixed-alpha E_acc/S saturates near one because it is normalized
by the natural teacher signal S. Pathwise R²_acc is normalized by the true
teacher variance along the generated direction; that variance approaches
zero, so R²_acc diverges negatively. See
`figures/model_selection/ffhq_fixed_vs_cv_comparison.png`; plot-ready results are in
tables/ffhq_fixed_vs_cv_summary.csv, and fixed-alpha trial caches are under
tables/ffhq_fixed_alpha_cases/.

### Van Hateren 100 x 100 disk-teacher replication

We repeated the paired fixed-alpha/RidgeCV comparison on log-luminance Van
Hateren patches. Raw 16-bit luminance was transformed as
`log1p(x)/log(65536)`, quantized to the same [0,1] scale as the FFHQ inputs,
and cropped to 100 x 100. The binary disk teacher has norm squared 692 and the
independent population estimate gives signal power S=1736.34.

To preserve the independent-sample assumption behind both ordinary LOOCV and
the DE, the training pool contains one crop from each of 2,000 source
photographs. Every n=1,000 training set therefore contains distinct source
images. The population covariance uses 20,000 patches from a disjoint set of
2,000 photographs. A preliminary 25-patches-per-image design was rejected:
patch-level LOOCV leaked through correlated sibling crops and spuriously chose
the interpolating alpha boundary in 26/100 trials already at sigma^2/S=0.03.

The final experiment uses the same 26 noise/signal ratios, 100 paired trials
per ratio, fixed alpha=100 (lambda=0.1), and 181-point alpha grid from 1e-4 to
1e5 as the FFHQ comparison. Exact LOOCV and the final sample-space solves are
performed in float64. This precision is necessary: on a preserved problematic
trial, float32 selected alpha=1e-4 whereas float64 selected alpha=199.5.

| sigma^2/S | policy | E_gen/S DE / MC | E_acc/S DE / MC | R^2_gen DE / MC | R^2_acc DE / MC median |
|---:|:---|---:|---:|---:|---:|
| 0.01 | RidgeCV | 0.00182 / 0.00181 | 0.0523 / 0.0447 | 0.998 / 0.998 | 0.912 / 0.934 |
| 0.01 | fixed | 0.00745 / 0.00738 | 0.0956 / 0.0847 | 0.993 / 0.993 | 0.944 / 0.949 |
| 0.1 | RidgeCV | 0.00702 / 0.00686 | 0.133 / 0.132 | 0.993 / 0.993 | 0.667 / 0.699 |
| 0.1 | fixed | 0.00925 / 0.00906 | 0.000472 / 0.000791 | 0.991 / 0.991 | 1.000 / 1.000 |
| 1 | RidgeCV | 0.0248 / 0.0241 | 0.261 / 0.264 | 0.975 / 0.976 | -0.119 / 0.0169 |
| 1 | fixed | 0.0273 / 0.0253 | 0.482 / 0.455 | 0.973 / 0.975 | -4.31 / -3.32 |
| 10 | RidgeCV | 0.0734 / 0.0795 | 0.291 / 0.274 | 0.927 / 0.921 | -0.636 / 0.497 |
| 10 | fixed | 0.208 / 0.191 | 0.925 / 0.919 | 0.792 / 0.809 | -830 / -494 |

DE closely predicts the stable errors and natural-image R^2 throughout the
curve. High-noise pathwise R^2_acc remains a broad ratio statistic: at
sigma^2/S=10 the empirical median is 0.497 but its IQR is [-2.23, 0.938] and
its mean is -19.7. The plot therefore continues to show median/IQR for this
metric. Fixed lambda=0.1 happens to nearly optimize accentuation around
sigma^2/S=0.1, but becomes catastrophically under-regularized at higher noise.

The staging probe projected 119.5 seconds and completed in 123.0 seconds. The
post-warmup GPU benchmark projected 33.7 seconds for 2,600 paired fits; the
simulation completed in 39.8 seconds. Plot-ready values are cached in
tables/vanhateren_fixed_vs_cv_summary.csv, per-noise trial data and mean weights
under tables/vanhateren_fixed_vs_cv_cases/, and the rendered comparison in
`figures/model_selection/vanhateren_fixed_vs_cv_gen_acc_gap.png`.

### Ratio-of-expectations versus response-noise second order

An audit prompted by the numerator/denominator expansion clarified what the
previous natural-image curves did.  They used

```
R0 = E[N]DE / E[D]DE,
N = beta_hat^T beta*,   D = beta_hat^T beta_hat,
```

and included `Var(R)` in `E_acc` plus `g''(R0) Var(R)/2` in
`R2_acc`, where `g(r)=1-(1-1/r)^2`.  They did **not** include the
second-order shift in the mean of the ratio,

```
bR = E[R] - R0
   ~= -Cov(N,D)/E[D]^2 + R0 Var(D)/E[D]^2.
```

The revised implementation exposes `Var(N)`, `Cov(N,D)`, and `Var(D)` and
compares two downstream predictions at exactly the same fitted-model
regularization:

```
ratio of expectations:
    E_acc/S = (1-R0)^2
    R2_acc  = g(R0)

response-noise second order:
    E_acc/S = (1-R0)^2 + 2(R0-1)bR + Var(R)
    R2_acc  = g(R0) + g'(R0)bR + g''(R0)Var(R)/2.
```

For RidgeCV, both curves reuse the DE-CV alpha selected by the generalization
risk model; the leading curve does not reselect alpha.  This makes the line
difference a clean audit of the nonlinear ratio approximation rather than a
regularization-policy difference.

| dataset | policy | target | ratio-of-expectations MAE | second-order MAE |
|:--|:--|:--|--:|--:|
| FFHQ | RidgeCV | mean normalized E_acc | 0.00615 | 0.00587 |
| FFHQ | RidgeCV | mean R2_acc | 0.0577 | 0.0547 |
| FFHQ | RidgeCV | median R2_acc | 0.0127 | 0.0121 |
| Van Hateren | RidgeCV | mean normalized E_acc | 0.00465 | 0.00549 |
| Van Hateren | RidgeCV | mean R2_acc | 1.016 | 0.983 |
| Van Hateren | RidgeCV | median R2_acc | 0.0657 | 0.0971 |

These MAEs use all 26 cached points through noise/signal variance ratio 10.
The second-order correction modestly improves the empirical **mean** of
`R2_acc`, which is the quantity the delta calculation targets.  It does not
uniformly improve the empirical median.  At Van Hateren noise/signal ratio 10,
for example, the two predictions are -0.288 and -0.662, while the empirical
mean is -19.68 and the median is +0.497.  This is evidence that a local
response-noise Taylor correction is insufficient once the reciprocal ratio is
heavy-tailed; random-design resolvent fluctuations and the mixture over
CV-selected alpha remain absent.

The solid and dashed theory curves are now included in both fixed-versus-CV
figures.  Per-noise moment and prediction columns are cached in the two
`fixed_vs_cv_summary.csv` files, and the aggregate error audit is
`tables/natural_image_ratio_theory_errors.csv`.  Recomputing all theory columns
and both figures from those compact caches takes about 6.3 seconds and performs
no image loading or regression refitting.

An aligned audit of the preserved coarse-grid cases separates two effects. The
first negative accentuation-R² trough spans the `alpha=100` to `alpha=1000`
transition: at `sigma=26.0`, 56/100 trials still select 100 and 44/100 select
1000, with conditional mean accentuation R² values -2.86 and 0.991. At
`sigma=30`, only 18/100 trials retain alpha 100, yet those trials keep the
unconditional mean negative while the median is 0.997. By `sigma=40`, all
trials select 1000 and the mean returns to 0.994. However, negative values at
`sigma=82.3` and 100 occur entirely within the fixed `alpha=1000` stage, so a
regularization jump is not necessary. Around the later 1000-to-10000
transition, the empirical median recovers near one while rare under-regularized
fits keep the nonlinear mean strongly negative. See
`figures/model_selection/ffhq_disk_teacher_coarse_alpha_aligned.png`; its
plot-ready values are
in `tables/ffhq_disk_teacher_coarse_alpha_aligned.csv`.

### Linear PCA features with pixel-space backpropagation

For a spectral linear feature map `z_j=a_j x_j`, regression sees feature
eigenvalues `c_j=a_j^2 s_j`, while pixel backpropagation uses the metric
`g_j=a_j^2`. If `theta_j*=beta_j*/a_j`, the leading feature-space DE is

```
mean_j = c_j/(c_j+kappa) theta_j*
Var_j  = v0 c_j/(c_j+kappa)^2
slope_acc = sum_j g_j mean_j theta_j*
            / sum_j g_j (mean_j^2 + Var_j).
```

Generalization additionally includes the teacher signal omitted by truncated
features. For Gaussian PC scores this omitted signal is independent of the
retained features and adds to both the irreducible evaluation error and the
effective training-noise variance. DE RidgeCV selection uses the feature
spectrum and this effective noise. We evaluated the measured FFHQ spectrum and
disk-teacher PC coefficients at `n=1000` with a 221-point alpha grid. Monte
Carlo validation used 50 Gaussian datasets matched to that exact spectrum at
each of four noise ratios.

Full PCA is an orthogonal rotation and therefore reproduces pixel ridge. Exact
whitening instead sets the regression covariance to identity but makes the
pixel-backpropagation metric `g_j=1/s_j`. With `p/n=10`, it both creates an
isotropic overparameterized regression problem and enormously amplifies the
lowest-variance PCs during accentuation. Top-100 PCA retains 99.52% of the disk
teacher's natural response variance while deleting those low-variance control
directions.

| sigma²/S | feature | E_gen/S DE / MC | E_acc/S leading DE / MC mean | slope_acc DE / MC | R²_acc DE / MC median |
|---:|:---|---:|---:|---:|---:|
| 0.1 | full PCA | 0.0141 / 0.0141 | 0.0566 / 0.0729 | 0.762 / 0.735 | 0.903 / 0.878 |
| 0.1 | full whitening | 0.910 / 0.911 | 1.000 / 1.000 | 1.68e-5 / 1.09e-5 | -3.54e9 / -2.08e9 |
| 0.1 | top 100 PCs | 0.0139 / 0.0141 | 1.44e-6 / 7.21e-4 | 0.999 / 0.997 | 1.000 / 1.000 |
| 1 | full PCA | 0.0533 / 0.0540 | 0.0863 / 0.104 | 0.706 / 0.695 | 0.827 / 0.830 |
| 1 | full whitening | 0.951 / 0.956 | 1.000 / 1.000 | 1.61e-5 / 2.22e-5 | -3.86e9 / -1.25e9 |
| 1 | top 100 PCs | 0.0514 / 0.0537 | 0.00268 / 0.00885 | 0.948 / 0.945 | 0.997 / 0.996 |
| 10 | full PCA | 0.188 / 0.200 | 0.133 / 0.186 | 0.636 / 0.616 | 0.672 / 0.652 |
| 10 | full whitening | 0.991 / 1.003 | 1.000 / 0.999 | 1.54e-5 / 3.50e-4 | -4.20e9 / -2.70e8 |
| 10 | top 100 PCs | 0.186 / 0.193 | 0.0424 / 0.0757 | 0.794 / 0.793 | 0.933 / 0.939 |

The generalization DE is accurate across all three representations. The
leading accentuation DE predicts typical calibration well, but when its error
is nearly zero, trial-to-trial fluctuations dominate the empirical mean
squared error; this explains the top-100 DE/MC gap at low noise. Exact
whitening is unequivocally harmful here: its true response barely changes
along the enormous low-variance pixel gradient, producing near-zero slope and
extremely negative path-wise R².

Sweeping the top-PC cutoff reveals an adaptive bias-variance tradeoff. The
generalization-optimal retained dimension decreases from `K=200` at
`sigma²/S=0.01`, to 150 at 0.1, 50 at 1, and 20 at 10. Thus higher response
noise favors a smaller, more signal-focused feature space. See
`figures/feature_space/ffhq_linear_feature_comparison.png` and
`figures/feature_space/ffhq_linear_feature_gen_acc_gap.png` for the direct experimental
generalization--accentuation gaps, and
`figures/feature_space/ffhq_top_pc_cutoff_sweep.png`; plot-ready data are in
`tables/ffhq_linear_feature_summary.csv` and
`tables/ffhq_top_pc_cutoff_summary.csv`.

---

## How to reproduce

```bash
# Small-d (CPU)
python scripts/validate_ridge_error_formula.py --spectrum powerlaw --d 64 --n 200

# Large-d (GPU)
python scripts/validate_large_d.py --spectrum vanhateren --patch_hw 32 --n 2000 --lam 0.05

# σ sweep
python scripts/validate_sigma_sweep.py --spectrum powerlaw --d 256 --n 512

# Natural/own/peer error and R² (benchmarks first, logs ETA, caches trials)
python scripts/validate_r2_peer_review.py

# Power-law teacher-alignment and eigenbasis-weight sweep
python scripts/validate_powerlaw_teacher_alignment.py

# Replot from the cached CSV without rerunning Monte Carlo
python scripts/validate_r2_peer_review.py --plot-only

# Actual K-fold CV versus DE-predicted lambda, followed by full-n refitting
python scripts/validate_cv_selected_r2.py

# Replot selected-lambda and downstream R² summaries without recomputation
python scripts/validate_cv_selected_r2.py --plot-only

# Exact 100x100 FFHQ disk-teacher validation (CUDA; image archive required)
python scripts/validate_ffhq_disk_teacher.py

# Replot all FFHQ figures from cached summary/case tables
python scripts/validate_ffhq_disk_teacher.py --plot-only

# PCA, whitening, and top-PC linear features with pixel backpropagation
python scripts/validate_ffhq_linear_features.py

# Replot feature-space summaries without Monte Carlo
python scripts/validate_ffhq_linear_features.py --plot-only

# Notebook (interactive)
jupyter notebook notebooks/validation_overview.ipynb
```

All scripts run in the `torch2` conda environment on the Kempner cluster.
For GPU runs, allocate an A100 or H100 node first via `salloc`.

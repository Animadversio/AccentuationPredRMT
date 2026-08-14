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
into the three terms. See `figures/nb_small_d_per_pc.png`.

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

See `figures/nb_large_d.png` and `figures/ridge_error_large_*.png`.

### σ sweep: varying noise level

Theory and MC tracked across σ ∈ [0.05, 3.0] at d=256, n=512, λ=0.05.

**Generalization error** (Σ_k λₖ E[(uₖᵀΔβ)²]):
- Theory matches MC to < 1% across the full noise range for both power-law and van Hateren.

**Accentuation alignment R**:
- After the formula bug fix (see below), theory matches MC to < 0.1%.

See `figures/sigma_sweep_fixed_accentuation.png` and `tables/sigma_sweep_*.npz`.

### λ sweep: varying ridge penalty

Theory predictions for total generalization error and accentuation error
vs λ ∈ [10⁻³, 10] (power-law d=256, n=512, σ=1.0):
- Both errors are monotonically increasing in λ (more regularization → more bias).
- R_det monotonically decreases toward 1 as λ → 0 (no regularization), and drops below 1 for large λ.
- Optimal λ (min gen error) can be read off from theory curve.

See `figures/nb_lambda_sweep.png`.

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
See `figures/sigma_sweep_fixed_accentuation.png`.

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
├── figures/                             # Saved PNG outputs
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

See `figures/gen_vs_acc_error_crossing.png`.

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

See `figures/gen_vs_acc_optimal_lambda_vh.png`.

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

See `figures/r2_peer_validation.png`, `figures/error_peer_validation.png`, and
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

See `figures/cv_selected_lambda.png`, `figures/cv_selected_r2.png`, and the
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

See `figures/powerlaw_teacher_alignment_r2.png`,
`figures/powerlaw_teacher_alignment_lambda.png`,
`figures/powerlaw_teacher_signal_profiles.png`,
`figures/powerlaw_teacher_weights_eigenbasis.png`, and
`figures/powerlaw_teacher_response_weights_eigenbasis.png`. Plot-ready data are
stored in `tables/powerlaw_teacher_alignment_summary.csv` and
`tables/powerlaw_teacher_weight_summary.csv`; raw per-case metrics and fitted
coefficient ensembles are cached under
`tables/powerlaw_teacher_alignment_cases/`.

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

# Notebook (interactive)
jupyter notebook notebooks/validation_overview.ipynb
```

All scripts run in the `torch2` conda environment on the Kempner cluster.
For GPU runs, allocate an A100 or H100 node first via `salloc`.

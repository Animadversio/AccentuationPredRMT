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

## How to reproduce

```bash
# Small-d (CPU)
python scripts/validate_ridge_error_formula.py --spectrum powerlaw --d 64 --n 200

# Large-d (GPU)
python scripts/validate_large_d.py --spectrum vanhateren --patch_hw 32 --n 2000 --lam 0.05

# σ sweep
python scripts/validate_sigma_sweep.py --spectrum powerlaw --d 256 --n 512

# Notebook (interactive)
jupyter notebook notebooks/validation_overview.ipynb
```

All scripts run in the `torch2` conda environment on the Kempner cluster.
For GPU runs, allocate an A100 or H100 node first via `salloc`.

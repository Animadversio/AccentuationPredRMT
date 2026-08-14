# AccentuationPredRMT

Numerical validation of the RMT theory for ridge regression weight deviation and accentuation prediction error.

Theory source: `~/Writings/AccentuationPredictionRMTDissociation/linear_reg_weight_deviation.tex`

## Theory Summary

For ridge regression `β̂ = (XᵀX + nλI)⁻¹Xᵀy` with Gaussian data `x ~ N(0,Σ)` and noise `y = β*ᵀx + σε`, the per-PC squared error decomposes (via RMT deterministic equivalents) into:

```
E[(uₖᵀ(β̂-β*))²] ≍
    κ²/(λₖ+κ)²  ·  (β*ᵀuₖ)²                   [overshrinkage bias]
  + κ²λₖ/(λₖ+κ)²  ·  C_sig / (n-df₂)           [finite-sample signal]
  + σ²λₖ/(λₖ+κ)²  /  (n-df₂)                   [finite-sample noise]
```

The accentuation error (error when testing along `β̂` direction) is:
```
E_acc ≍ (β*ᵀΣβ*) · (1 - R_det)²
```
where `R_det = E[β̂ᵀβ*] / E[β̂ᵀβ̂]`.

The library also predicts noiseless-signal `R²_gen`, own-path `R²_acc`,
and error/`R²_peer` when an independent ridge fit generates the accentuation
path. Optional second-order delta corrections account for finite-sample
fluctuations in the nonlinear own-path and peer ratios.

## Structure

```
rmt_core/          # Core library
  kappa_lib.py     # κ(λ) solver (Marchenko-Pastur fixed-point)
  ridge_theory_lib.py  # Three-term error formula, accentuation error
  simulation_lib.py    # Monte Carlo simulation functions
  teacher_lib.py       # Teachers with controlled population-PC alignment

scripts/
  validate_ridge_error_formula.py   # Per-PC error: theory vs MC
  validate_accentuation_error.py    # Accentuation error vs n
  validate_r2_peer_review.py        # Gen/own/peer error and R² validation
  validate_cv_selected_r2.py        # R² after actual vs DE-predicted K-fold CV
  validate_powerlaw_teacher_alignment.py  # Teacher spectral-alignment sweep
  validate_ffhq_disk_teacher.py     # Exact natural-image disk-teacher experiment

notebooks/
  theory_validation_demo.ipynb      # Interactive demo

figures/           # Output figures
tables/            # Output tables
bash/              # Slurm job scripts
```

## Usage

```bash
# Activate torch2 env
mamba activate torch2

# Validate per-PC error formula
cd scripts && python validate_ridge_error_formula.py --d 100 --n 200 --lam 0.1 --sigma 0.5

# Validate accentuation error vs n
python validate_accentuation_error.py --d 50 --lam 0.05 --sigma 0.5

# Validate R² and error on natural, own-accentuation, and peer paths.
# The script benchmarks first, shows tqdm ETA, and caches per-case trials.
python scripts/validate_r2_peer_review.py

# Re-render both figures from the cached summary without rerunning Monte Carlo
python scripts/validate_r2_peer_review.py --plot-only

# Compare actual 5-fold CV with the DE-predicted selected lambda
python scripts/validate_cv_selected_r2.py

# Re-render CV-selected lambda and R² figures from cached summaries
python scripts/validate_cv_selected_r2.py --plot-only

# Sweep a unit-signal teacher from top to bottom of a power-law spectrum,
# including actual/DE CV lambda selection and eigenbasis weight plots
python scripts/validate_powerlaw_teacher_alignment.py

# Restyle all five alignment/weight figures using plot-ready cached tables
python scripts/validate_powerlaw_teacher_alignment.py --plot-only

# Reproduce the 100x100 FFHQ unit-disk teacher experiment on a CUDA GPU.
# The default 26-point curve reaches noise/signal variance ratio 10; figures
# show response-noise SD on the lower x-axis and variance ratio on the upper.
# RidgeCV and DE share a 181-point log-alpha grid (20 intervals per decade).
# The script benchmarks first, stages the image archive locally, and caches cases.
python scripts/validate_ffhq_disk_teacher.py

# Restyle the four FFHQ figures from cached summaries (no images/GPU needed)
python scripts/validate_ffhq_disk_teacher.py --plot-only
```

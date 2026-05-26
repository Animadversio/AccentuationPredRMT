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

## Structure

```
rmt_core/          # Core library
  kappa_lib.py     # κ(λ) solver (Marchenko-Pastur fixed-point)
  ridge_theory_lib.py  # Three-term error formula, accentuation error
  simulation_lib.py    # Monte Carlo simulation functions

scripts/
  validate_ridge_error_formula.py   # Per-PC error: theory vs MC
  validate_accentuation_error.py    # Accentuation error vs n

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
```

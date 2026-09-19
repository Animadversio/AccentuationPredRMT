# Van Hateren full noise × regularization landscape

## Authoritative storage

`$STORE_DIR/Projects/AccentuationPredRMT/vanhateren_landscape_v1_20260918`

Resolved on the compute node to:
`/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Projects/AccentuationPredRMT/vanhateren_landscape_v1_20260918`

Computation was staged on holygpu8a11604 at
`/scratch/tmp/AccentuationPredRMT_landscape.vTGqSG/landscape_v1`.
The node-local copy is temporary; use the STORE_DIR archive for future work.
`full.log`, `pilot.log`, and validation reports accompany the archived results.

## Experiment

- Same Van Hateren circular teacher, full 10,000 pixels, n=1000.
- S=1736.34496504; empirical population centering and weight-only evaluation,
  consistent with the prior runs. Fitted intercept is included in LOOCV but
  excluded from the population response metrics.
- Natural-image MC: 100 paired trials, seeds 20260918+trial, 1000 samples from
  the same 2000-image training pool, separate 20000-patch population pool.
- 114 noise coordinates: zero plus a log grid of sigma²/S from 1e-8 to 10,
  including earlier noise cuts (rounded in log coordinates to avoid duplicate
  floating-point keys).
- 478 regularization coordinates: broad 401-point grid lambda=1e-8..1e4 plus
  earlier selected/cut values. Alpha=n*lambda; kappa is mapped on the physical
  branch. No actual lambda=0 column; the lower boundary is a finite grid bound.
- 54,492 points per surface, 5,449,200 trial-point evaluations.
- Errors E_gen and E_acc are normalized by S. Weight error is NOT normalized.
  R2_gen=1-E_gen; R2_acc=1-(D/N-1)^2; slope_acc=N/D;
  slope_gen is true-on-fitted covariance / fitted-response variance.

## Exact reuse across noise

At fixed design and alpha, w_hat(sigma)=w0+sigma*w1. Therefore N is linear,
D and prediction MSE are quadratic in sigma. Each trial stores their
coefficients, as well as those of LOOCV MSE. This is exact, not a noise-axis
interpolation or a separate Gaussian-design simulation. The same unit-noise
draw is deliberately paired across all sigma/alpha values.

## Files / array schema

Small version-controlled copies of `selected_cuts.csv`, `config.json`, and
`integrity_validation.json` are in `tables/vanhateren_landscape/`. Full surfaces,
per-trial arrays, and draw coefficients remain exclusively in the archive.

Every grid array uses **[noise_index, alpha_index]**. Raw natural-image MC
arrays use **[trial_index, noise_index, alpha_index]**.

- `coordinates.npz`: sigma, ratio, alpha, lam, kappa, S, n, eigenvalues,
  beta_proj. Spectrum/teacher coordinates make theoretical reconstruction
  independent of the repository's spectrum file.
- `config.json`: conventions, coordinate lists, seeds, source-code hash.
- `theory.npz`: E_gen, E_acc, R2_gen, R2_acc, slope_gen, slope_acc,
  weight_error; kappa is a one-dimensional coordinate.
- `theory_cv_nminus1.npz`: prediction risk used for the prior LOOCV-sized DE
  selection convention (still evaluated/refitted at n in theory.npz).
- `coefficients_0000.npz` ... `coefficients_0099.npz`: exact natural-image
  trial coefficients; reusable to add arbitrary noise values without fits.
- `mc_raw_<metric>.npz`: key `values`, all 100 trial evaluations.
- `mc_summary.npz`: `<metric>_mean`, `_std`, `_se`, `_median`, `_q10`, `_q90`;
  scalar `trials`. Quantiles are trial variability, not confidence intervals.
- `mc_loocv.npz`: full trial LOOCV risk surface and selected_index.
  LOOCV includes intercept leverage. Verified against actual sklearn RidgeCV
  at three noise levels; it is computed by the equivalent smoother formula,
  not by calling sklearn separately at every grid point.
- `gaussian_selection.npz`: independent 2048-draw acc objective, seed 123.
- `gaussian_evaluation.npz`: independent 8192-draw evaluation, seed 20260917;
  means and SE for all seven metrics; minimum abs(N), nonpositive-N fraction.
- `gaussian_coefficients_<selection/evaluation>.npz`: reduced paired draw
  coefficients, permitting arbitrary-noise surrogate reevaluation.
- `selected_indices.npz`: DE prediction, n-1 proxy, leading acc, Gaussian acc,
  MC empirical oracle, and individual-trial CV selections.
- `selected_cuts.csv`: selected regularization and cross-evaluation metrics.
- `cv_selected_evaluations.npz`: per-trial metrics evaluated at that trial's
  own CV choice, not at the median CV alpha.
- `validation.json`: direct ridge and sklearn CV checks.
- `integrity_validation.json`: finite arrays, trial counts, comparison with
  prior first-trial cuts and staged image SHA256.

The Gaussian method remains a diagonal coefficient-moment surrogate, NOT a
proved full distributional-DE law. Its R2_acc is a finite-draw inverse-N
estimate and may be heavy-tailed. Zero-noise projection identities are not
enforced by this surrogate. MC oracle minima are same-sample exploratory
minima; do not interpret their evaluated risk as independent test estimates
of the model-selection procedure. All selection paths here are grid minima.

## Replotting and cuts

```python
from pathlib import Path
import numpy as np
root = Path('/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Projects/AccentuationPredRMT/vanhateren_landscape_v1_20260918')
with np.load(root / 'coordinates.npz') as f:
    sigma, ratio, lam, kappa = (f[k] for k in ['sigma','ratio','lam','kappa'])
with np.load(root / 'mc_summary.npz') as f:
    Egen, Eacc = f['E_gen_mean'], f['E_acc_mean']
# Fixed noise: Egen[i, :]. Fixed regularization: Egen[:, j].
# Selected regularization path: Egen[np.arange(len(sigma)), selected_index].
```

Preview figures are in `figures/vanhateren_landscape/` (PNG + embedded
TrueType-font PDF). The heatmap itself is rasterized in PDF to keep file sizes
reasonable; labels and paths remain vector. Zero-noise row is cached but
excluded from the logarithmic preview. Shared color limits deliberately
saturate extremes; they do not clip the cached data. White is n-sample DE
prediction optimum; pink is leading-DE acc grid optimum, in all panels.
Gray cuts mark sigma²/S=0.001,0.01,0.1,1 and fixed lambda=0.1.

To render the same cached surfaces on a kappa y axis without rewriting data:

```bash
python -m scripts.plot_vanhateren_landscape \
  --data "$STORE_DIR/Projects/AccentuationPredRMT/vanhateren_landscape_v1_20260918" \
  --figures figures/vanhateren_landscape --plot-only --y-axis kappa
```

This produces `landscape_{error,r2,slope}_kappa.{png,pdf}`. Both the mesh
coordinates and the overlaid paths are transformed, not just tick labels.
The lambda=0.1 reference cut is mapped to its corresponding kappa. Color
limits and underlying metrics are unchanged. The nonzero physical lower
bound in kappa compresses the near-ridgeless region; no data below that bound
are extrapolated. Existing lambda-axis exports are retained.

Entry points: `scripts/vanhateren_landscape.py`,
`scripts/plot_vanhateren_landscape.py`,
`scripts/validate_vanhateren_landscape.py`.
Unit test: `tests/test_vanhateren_landscape.py` checks direct ridge metrics
and explicit leave-one-out refits on a small independent problem.

## One-dimensional cuts

`scripts/plot_vanhateren_landscape_slices.py` extracts three families directly
from the common grid: fixed lambda versus noise, fixed noise versus kappa, and
the prediction-selected DE / trialwise empirical-LOOCV paths. Each family is
exported for normalized error, R2, and true-on-fitted slope as PNG and
vector-font PDF under `figures/vanhateren_landscape/slices/`.

The R2 panels use a symmetric-log scale because accentuation R2 can be very
negative; the slope panels also use symmetric-log so rare negative MC values
are not silently dropped. Reference lines mark R2=0, R2=1, or slope=1. The
plot-ready cut coordinates and DE/MC summaries are cached in
`tables/vanhateren_landscape/slice_summary.csv`.

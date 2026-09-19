# Four-noise regularization paths (figure review, no final notebook yet)

Entry point: `scripts/vanhateren_regularization_paths.py`.

- Van Hateren 100×100 full pixels, circular teacher; n=1000, S=1736.344965.
- Noise/signal variance ratios: 0.001, 0.01, 0.1, 1.
- Theory: 401 log-spaced sklearn alpha values from 1e-5 to 1e7;
  lambda=alpha/n, kappa on the positive physical branch.
- Selection markers are **grid minima**, not continuous optimizers. Gen DE
  uses the n-1 training-size proxy, evaluated at n after selection.
- Gaussian surrogate: independent 2048-draw selection and 8192-draw evaluation;
  shared standard Gaussian draws along each path. This is the existing diagonal
  coefficient-moment surrogate, not a full distributional-DE theorem.
- MC: 100 paired trials with the existing CUDA seeds 20260918+trial, same image
  pool and centered weight-only population evaluation as the prior experiment.
  49 broad-grid alpha values plus the union of all theory-selected alpha values.
- Actual sklearn RidgeCV independently selects each of four noise targets on
  the same 401-alpha grid (LOOCV, intercept, per-target selection).
- MC points: arithmetic mean, error bars ±2 SE, band: trial 10–90%.
  R2_acc is computed per realization, not from mean slope. Inverse-N moments
  may be heavy-tailed; the Gaussian R2 curve is a finite-draw estimate.
- Plot error columns E_gen and E_acc are already normalized by S.
- For clarity, error display uses 1e-7..1e2; full numeric values are cached.
  Zoom figures display lambda approximately 1e-4..10 and explicitly restrict
  R2/slope display ranges. Full-range figures are also exported.

Small plot tables / PNG / vector-font PDF:
`notebooks/outputs/pixel_ridge/vanhateren_regularization_paths/`.

Per-trial cache on holygpu8a13203:
`/scratch/tmp/AccentuationPredRMT.bOcMkB/regularization_paths_v1/`.
Log: `/scratch/tmp/AccentuationPredRMT.bOcMkB/regularization_paths_full.log`.
Authorized non-Lustre backup:
`/n/netscratch/kempner_binxuwang_lab/Everyone/binxuwang/AccentuationPredRMT/vanhateren_selection_estimation_local_20260917/regularization_paths_v1/`.
Intended permanent project archive remains `$STORE_DIR/Projects/AccentuationPredRMT`;
node-local storage is temporary, NFS is the outage fallback.

Replot only: `python -m scripts.vanhateren_regularization_paths --plot-only`.
No simulation needed to switch between lambda and kappa figures.

Validation: batched DE risks and kappa matched the existing scalar `metrics`
implementation at eight broad-range noise/alpha combinations.

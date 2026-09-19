# Biological validation: basic variance contribution

Date: 2026-09-19. Completed CPU job **47276723** (2m50s, exit 0).

We computed `V = MSE_test / n × sum(q_k s_k / (s_k + kappa)^2)` for all 250 site/model pairs. This is an empirical variance-contribution proxy, not an identified prediction of absolute biological slope. Teacher signal/alignment and the stimulus optimization metric are deferred.

## Reproduction and coverage

- Five monkeys, 25 sites, 10 models, 10 cached geometry seeds; 774 training and 195 held-out natural stimuli per readout.
- All 250 biological slopes available; control coverage ranges from 36 to 110 presented stimuli per site/model.
- Original RidgeCV alpha recovered for all 250 readouts (50 source objects). Range 0.001–1,000,000. Maximum relative coefficient discrepancy versus exported readouts: 2.91e-8.
- `lambda = alpha/n`; solve `kappa*(1-sum(s/(s+kappa))/n) = lambda`. The ddof=1 training spectrum is used as a population-spectrum proxy. This is not a known population spectrum.
- Held-out MSE is against trial-averaged biological responses; no additional sigma² is added.
- Pinned upstream preprocessing: `9b6fd95228d1b77146f8b279c7c228914ca34bb8`; official peak windows, outlier rejection, anchorDay normalization, firing floor, post-hoc predictions, and control slope definition.
- Reconstructed Leap audit: 277 image pairs, 57 with flattened native RGB correlation ≥0.90. Upstream presentation-aware canonicalization applied. This is not a byte-identical audit of the unavailable frozen file.
- Three0 session 250508 has zero anchors and uses the upstream allday normalization fallback. Other sessions do not. Leap/three0 official noise ceilings are unavailable; no ceiling correction is used.
- Bulk caches and pinned upstream source: `$STORE_DIR/Projects/AccentuationPredRMT/nonlinear_control/biological_validation_v1`.

## Descriptive evidence

Exact V versus biological slope: pooled Spearman **−0.314**, site-demeaned Spearman **−0.421**. For within-seed control slope, corresponding values are **−0.313** and **−0.410**. These are descriptive correlations, not independent-observation significance tests.

**The separation is largely driven by CLIPAG and robust RN50.** Removing these two models reduces the pooled exact correlation to **−0.067**. Do not infer a universal monotone relation across models.

| Model | Exact V (mean of subject means) | Biological slope |
|---|---:|---:|
| AlexNet_training_seed_01 | 1.443 | 0.102 |
| clipag_vitb32 | 0.009733 | 0.571 |
| dinov2_vitb14_reg | 1.956 | 0.301 |
| radio_v2.5-b | 0.9384 | 0.345 |
| regnety_640 | 2.915 | 0.263 |
| resnet50 | 1.963 | 0.319 |
| resnet50_clip | 19.48 | 0.269 |
| resnet50_dino | 3.478 | 0.304 |
| resnet50_robust | 0.005238 | 0.518 |
| siglip2_vitb16 | 1.666 | 0.259 |

## Noise estimators

All four noise SDs (×255 = 0.5, 2, 8, 16) are reported; none is selected as an optimal noise level from these same data. Exact and local MC serve as zero-noise controls. Neighborhood averages squared local Jacobians; smooth squares the averaged Jacobian; variance uses centered finite response covariance / tau²; step also contains response drift.

| Estimator | Noise SD ×255 | V–slope Spearman | Site-demeaned |
|---|---:|---:|---:|
| exact | 0 | -0.314 | -0.421 |
| neighborhood | 0.5 | -0.318 | -0.427 |
| neighborhood | 2 | -0.342 | -0.462 |
| neighborhood | 8 | -0.398 | -0.531 |
| neighborhood | 16 | -0.434 | -0.556 |
| smooth | 0.5 | -0.327 | -0.459 |
| smooth | 2 | -0.351 | -0.508 |
| smooth | 8 | -0.359 | -0.514 |
| smooth | 16 | -0.309 | -0.482 |
| step | 0.5 | -0.252 | -0.383 |
| step | 2 | -0.139 | -0.317 |
| step | 8 | -0.125 | -0.313 |
| step | 16 | -0.104 | -0.342 |
| variance | 0.5 | -0.326 | -0.447 |
| variance | 2 | -0.342 | -0.487 |
| variance | 8 | -0.365 | -0.529 |
| variance | 16 | -0.379 | -0.539 |

Stein is presently unstable: negative site-level trace estimates at 60/250, 77/250, 210/250, and 217/250 site/model pairs as noise increases. All signed estimates are preserved. Log analyses exclude nonpositive values and must not be interpreted as unbiased evidence on the full cohort. CV candidates use the same positive subset within each estimator/noise condition.

CLIP RN50 directional finite-difference QC was marginal in the preceding production run. Smooth/neighborhood/local MC results involving it remain provisional; exclusion sensitivity is provided. Exact VJP and finite response covariance do not use that small finite-difference step.

## Held-out comparisons

These are empirical regressions of slope on log10 predictors, not a theoretical absolute-slope prediction. Fixed linear fits use leave-one-monkey-out and leave-one-model-out, with training-fold standardization and no hyperparameter search.

| Exact estimator predictor | Leave-model-out MSE | Leave-monkey-out MSE |
|---|---:|---:|
| intercept | 0.07128 | 0.07750 |
| mse_test | 0.07258 | 0.08440 |
| trace | 0.06043 | 0.06789 |
| V | 0.06009 | 0.06747 |
| mse_test+trace | 0.06204 | 0.07456 |

The product outperforms MSE-only in this basic comparison, but improves only slightly over trace alone. No uncertainty interval or claim of significant incremental improvement is made.

## Maintained 250-row synopsis

`tables/nonlinear_control/biological_validation/biological_validation_synopsis_v1.parquet`
is the maintained fast analysis table: one row per subject × unit × model, 250
rows and 221 columns. It combines provenance, RidgeCV/DE quantities, held-out
natural-image generalization, cross-phase anchor metrics, biological control
metrics, error normalizations and all geometry estimator/noise summaries.

At the individual 250-row level, exact V versus control slope gives Pearson
`r=-0.093` on the raw V scale, Pearson `r=-0.378` on log10 V, and Spearman
`rho=-0.314`. Removing CLIPAG and robust RN50 gives `-0.003`, `-0.063`, and
`-0.067`, respectively. The original site-demeaned log10-V Spearman is
`-0.421`; without those models it is `-0.057`.

Smooth V gives the following 250-row correlations with control slope:

| noise SD × 255 | Pearson(raw V) | Pearson(log10 V) | Spearman | Spearman without CLIPAG/robust RN50 |
|---:|---:|---:|---:|---:|
| 0.5 | -0.144 | -0.386 | -0.327 | -0.091 |
| 2 | -0.290 | -0.397 | -0.351 | -0.134 |
| 8 | -0.280 | -0.382 | -0.359 | -0.158 |
| 16 | -0.271 | -0.339 | -0.309 | -0.148 |

For error outcomes, exact log10 V correlates `+0.283` with raw control MSE
(Spearman `+0.344`) and `+0.269` with control MSE divided by the repeat-noise-
corrected natural signal variance (Spearman `+0.269`). After removing CLIPAG
and robust RN50, the latter becomes `+0.076` and `+0.072`. This is directionally
consistent with V as an error contribution, while again showing that the
cross-model separation drives much of the effect.

The theory denominator S is latent natural teacher signal power. The synopsis
therefore keeps observed, natural-repeat-noise-corrected, control-cloud and
control-noise-corrected normalizations under distinct names. Natural repeat
coverage is complete; control repeat coverage is incomplete, so raw control
MSE over corrected natural S is primary and numerator correction is a
sensitivity analysis. See `tables/nonlinear_control/biological_validation/SYNOPSIS.md`.

## Outputs and rerun

- Figures: `figures/nonlinear_control/biological_validation/` (22 bar variants and six scatter figures).
- `biological_outcomes.csv`: MSE, direct-prediction control R², control MSE, observed-variance normalized Eacc, pooled and within-seed slopes.
- `variance_predictors.csv`: 5,500 site/estimator rows; `variance_predictors_by_seed.csv.gz`: 55,000 seed rows. Full PC mass remains in the existing mass_v1 cache.
- `descriptive_associations.csv`, `heldout_slope_comparison.csv`, `heldout_slope_predictions.csv.gz`, `robust_pair_and_normalization_sensitivity.csv`: comparisons and sensitivity results.
- `control_clouds.csv.gz`, `preprocessing_audit.csv`, `ridge_metadata.csv`, `analysis_provenance.json`: plot-ready stimulus responses and QC/provenance.
- Main driver: `explorations/nonlinear_control/validate_biology.py`; reproducible CPU entry point: `run_biological_validation.sbatch`.
- Run `validate_biology.py validate` for cardinality, finite-value and product-identity checks. Kappa solver was independently checked against the analytic one-eigenvalue quadratic.
- HDF5 source files are opened read-only; the batch script disables file locking because shared-filesystem locks stalled the initial pilot. Numerical preprocessing is unchanged.

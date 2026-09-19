# Maintained biological-validation synopsis

`biological_validation_synopsis_v1.parquet` is the fast analysis table. It has
one row per subject × unit × encoding model: 25 biological sites × 10 models =
250 rows. Its composite key is `subject, monkey, unit, model`.

The compressed CSV is a portable mirror. The schema CSV gives the group, dtype
and definition of every column. The long-format `variance_predictors*.csv.gz`
tables remain the canonical source for adding geometry estimators or noise
levels; rebuild the synopsis rather than manually adding columns.

## Column groups

| Prefix | Contents |
|---|---|
| identity / provenance | subject, monkey, unit, model, region, site and geometry IDs, robust-model flag |
| `ridge_` | original RidgeCV alpha, alpha/n, solved kappa, df2 and readout verification |
| `gen_test_` | held-out natural-image MSE, RMSE, MAE, bias, Pearson r, slope/intercept, identity R², refit R², response and prediction variance |
| `crossphase_anchor_` | same metrics on natural-image anchors repeated in the control sessions |
| `control_` | biological accentuation MSE, slope, within-seed slope, identity R², refit R², normalization and repeat-noise quantities |
| `geom_<method>[_tau255_<level>]__` | ten-seed mean and sample SD of raw energy, actual-kappa control trace and V=MSE_test×trace/n |

`r2_identity = 1 - MSE(y, prediction) / Var(y)` measures absolute prediction
calibration. `r2_refit = Pearson r²` measures linear association after fitting
slope and intercept. They answer different questions and should not be merged.

## Error normalization

Theory uses the latent natural teacher signal power
`S = beta*' Sigma beta*`. It is not directly observed here.

- `gen_test_measured_variance` is `S_nat_observed`: variance across held-out
  natural-image trial means. It includes response measurement noise.
- `gen_test_S_nat_noise_corrected` subtracts the estimated trial-mean noise
  contribution using single-trial repeats. Repeat coverage is 100% for all 250
  rows, so this is the preferred empirical denominator.
- `control_error_over_S_nat_noise_corrected` is the primary normalized control
  error: raw biological control MSE divided by corrected natural signal power.
  It retains the observed held-out response error, matching the decision to use
  held-out prediction MSE in V.
- `control_error_noise_corrected_over_S_nat_noise_corrected` also subtracts
  estimated control trial-mean noise from the numerator. It is sensitivity-only:
  control repeat coverage ranges from 0.070 to 1.000, and only 102/250 rows have
  at least 80% coverage.
- `control_error_over_S_control_observed` is exactly `1-control_r2_identity`.
  It is descriptive and cannot independently validate the theory.

No negative noise-corrected MSE is clipped. Coverage and high-coverage flags
are stored beside every correction.

## Maintenance contract

Run:

```bash
MPLCONFIGDIR=/tmp/mpl-biological-synopsis \
XDG_CACHE_HOME=/tmp/xdg-biological-synopsis \
/n/home12/binxuwang/.conda/envs/torch2/bin/python \
  explorations/nonlinear_control/build_biological_synopsis.py
```

The builder asserts 250 unique keys, ten models per site, 25 rows per model,
the expected exact/smooth geometry blocks, full natural-image repeat coverage,
and the two normalized-error/R² identities. Add new raw measurements to the
long tables or biological cache first, then extend the builder and schema.

`synopsis_correlations.csv` contains raw-scale Pearson, log10-predictor
Pearson, Spearman, and site-demeaned Spearman correlations for the full 250
rows and after removing CLIPAG and robust RN50. Correlations are descriptive:
rows share animals, sites, feature geometries and image seeds.

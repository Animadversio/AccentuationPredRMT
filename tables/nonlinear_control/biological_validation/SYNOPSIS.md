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
| `control_session_gen_test_` | primary generalization metrics: encoding-held-out natural images evaluated with responses recorded during the control session |
| `encoding_session_gen_test_` | the same held-out split evaluated with original encoding-session responses; retained for session-drift sensitivity |
| `encoding_session_matched_gen_test_` | encoding-session responses restricted to the exact 22/24/50-image subset available in the control session; separates subset size/composition from session drift |
| `session_drift_` | direct matched-image response drift between encoding and control sessions, plus the two generalization-MSE difference/ratio |
| `crossphase_anchor_` | same metrics on natural-image anchors repeated in the control sessions |
| `control_` | biological accentuation MSE, slope, within-seed slope, identity R², refit R², normalization and repeat-noise quantities |
| `geom_<method>[_tau255_<level>]__` | ten-seed mean and sample SD of raw energy, actual-kappa trace, `V_control_session` and `V_encoding_session` |

`r2_identity = 1 - MSE(y, prediction) / Var(y)` measures absolute prediction
calibration. `r2_refit = Pearson r²` measures linear association after fitting
slope and intercept. They answer different questions and should not be merged.

## Error normalization

Theory uses the latent natural teacher signal power
`S = beta*' Sigma beta*`. It is not directly observed here.

- `control_session_gen_test_measured_variance` is the primary empirical
  `S_nat_observed`: variance across held-out natural images re-presented during
  the control session. This controls recording-session drift but includes
  response measurement noise.
- `control_session_gen_test_S_nat_noise_corrected` subtracts a repeat-estimated
  noise contribution. Repeat coverage is complete for red, paul and venus, but
  only 20.8% for Leap and 18.2% for Three0, so this is sensitivity-only.
- `encoding_session_gen_test_S_nat_noise_corrected` uses all 195 held-out images
  and has 100% repeat coverage. It is statistically cleaner but can reflect a
  different neuronal state from the subsequent control experiment.
- `control_error_over_S_nat_control_session_observed` is the primary normalized
  control error: biological control MSE divided by control-session natural
  response variance.
- `control_error_noise_corrected_over_S_nat_control_session_noise_corrected`
  also subtracts estimated control-stimulus trial-mean noise. It is
  sensitivity-only:
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
the expected exact/smooth geometry blocks, full encoding-session natural-image
repeat coverage, bounded control-session coverage, session-specific V identities,
and the normalized-error/R² identities. Add new raw measurements to the
long tables or biological cache first, then extend the builder and schema.

`synopsis_correlations.csv` contains raw-scale Pearson, log10-predictor
Pearson, Spearman, and site-demeaned Spearman correlations for the full 250
rows and after removing CLIPAG and robust RN50. Correlations are descriptive:
rows share animals, sites, feature geometries and image seeds.

The smoothing-level correlation figures are stored separately for
`V_control_session` and `V_encoding_session`. Neither endpoint overwrites the
other. The control-session endpoint matches the control recording state but
has only 22–50 held-out anchors; the encoding-session endpoint has 195 images
per row but may be affected by recording-session drift.

`session_endpoint_image_bootstrap_*.csv*` quantify finite-image uncertainty by
resampling held-out image identities jointly within each monkey. This preserves
the shared-stimulus dependence among sites/models. It does not quantify animal,
site, trial-response, Jacobian-probe or model-family uncertainty.

`site_centered_associations.csv` subtracts each site's mean from log10 V and
the outcome, then reports Pearson, Spearman, the fixed-intercept regression
coefficient and a site-clustered standard error. It covers control slope,
within-seed slope and direct control MSE for exact, smooth and neighborhood
geometry. The maintained figures show the smoothing paths and the direct
control-MSE computation/scatter separately.

`predictor_benchmark_site_centered.csv` applies that same transformation to a
fixed benchmark of 24 predictors: held-out generalization MSE, exact trace,
exact/local-MC V, and four noise levels for smooth, neighborhood, variance,
finite-step and Stein V. It stores both raw and direction-aligned Pearson and
Spearman correlations for all ten models and after removing CLIPAG and robust
RN50. Direction alignment multiplies slope correlations by -1 and leaves MSE
correlations unchanged, so positive values consistently mean that a larger
predicted control-error term accompanies worse biological control. The
clustered regression p value and within-panel Benjamini-Hochberg q value refer
to the linear site-residual coefficient; they are not p values for Spearman's
rho. Stein rows retain their actual positive-V sample counts because the
large-noise estimators can be nonpositive.

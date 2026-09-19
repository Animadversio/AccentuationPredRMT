# Site-residual predictor benchmark

This benchmark compares alternative empirical approximations to the variance
term in the feature-space control-error theory against two biological outcomes.
The default control-session benchmark now uses the one-per-site cross-session
affine fitted on encoding-training natural anchors and shared unchanged across
all ten models. Its slope, MSE, heldout generalization error and V therefore use
the same response mapping.
The slope panel is direction-aligned as `-Spearman rho`: larger predicted error
should correspond to a smaller control slope. The direct-MSE panel uses
`+Spearman rho`: larger predicted error should correspond to larger MSE.

For every predictor, the analysis first takes `log10`, subtracts the mean of
the 10 encoding models within each biological site, and subtracts the same site
mean from the biological outcome. This is a site fixed-intercept comparison of
models and removes all between-site and between-monkey level differences. The
filled circle uses all 10 models. The open circle removes CLIPAG and robust
RN50. The open diamond additionally removes the untrained AlexNet, leaving the
seven conventionally trained models. Their connecting segment exposes
sensitivity to these three model-family outliers.

Each endpoint benchmark contains 24 predictors:

- held-out natural-image generalization MSE and exact control trace as
  baselines;
- exact Jacobian V and local Monte-Carlo V;
- smooth, neighborhood, finite-difference variance, finite-step and Stein V at
  noise SD 0.5, 2, 8 and 16 on the 0–255 image scale.

The figures show Spearman correlation because model rankings are the primary
question and the predictors span very different scales. The CSV also stores
Pearson correlation and a linear standardized coefficient with site-clustered
standard error. A star directly above any filled circle, open circle or open
diamond marks Benjamini-Hochberg q < 0.05 for that marker's model subset. It
does not represent a clustered significance test of Spearman correlation.

The Stein estimates at large noise can be zero or negative and therefore
cannot enter the log analysis. Their actual effective sample sizes are printed
next to the markers. These small-n correlations are unstable and should not be
ranked directly against the full-coverage subset estimates.

`site_centered_predictor_benchmark_control_session.png` is the calibrated
default. `site_centered_predictor_benchmark_control_session_identity_reference.png`
preserves the previous direct identity-scale comparison. The encoding-session
figure remains an explicitly labeled identity reference because the
encoding-to-control map targets control-session response units.

The central descriptive result is outcome-specific. After removing the two
robust models, neighborhood V at 16/255 has a stronger direction-aligned slope
correlation than smooth V at 16/255 in both sessions. Smooth V is slightly
stronger for direct control MSE. Thus the neighborhood result supports a
less-local predictor for multiplicative control gain, while it does not
dominate smooth V for absolute prediction error.

Rebuild all tables and figures with:

```bash
MPLCONFIGDIR=/tmp/mpl-biological-synopsis \
XDG_CACHE_HOME=/tmp/xdg-biological-synopsis \
/n/home12/binxuwang/.conda/envs/torch2/bin/python \
  explorations/nonlinear_control/analyze_site_centered_effects.py
```

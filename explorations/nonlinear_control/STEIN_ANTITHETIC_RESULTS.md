# Nested-direction antithetic Stein validation

## Estimator

The original forward-only estimator used

\[
\delta_r^+=f(x+\tau z_r)-f(x).
\]

The improved estimator uses the antithetic odd response

\[
\delta_r^{\rm anti}={f(x+\tau z_r)-f(x-\tau z_r)\over2}.
\]

Both are inserted into the same off-diagonal U-statistic

\[
\widehat q_k={1\over R(R-1)\tau^2}
\sum_{r\ne s}\delta_{rk}\delta_{sk}z_r^\top z_s.
\]

By Gaussian symmetry they have the same population target. The antithetic
version exactly removes the even response component, whose expectation after
multiplication by the odd score is zero but whose finite-sample variance can be
large.

## Pilot design

The pilot used the official model loader and preprocessing, ten fixed seed
images, four RGB noise levels, and common nested Gaussian direction prefixes
R = 64, 128, 256, 512, 1024. One maximum-alias geometry per each of ten models
covered 72 biological site-model aliases. Raw one-sided and antithetic response
deltas and direction Gram matrices were saved, so every prefix uses exactly the
same random stream.

The R=64 one-sided reconstruction matches the existing production estimator
with relative L2 error below 1e-4. This checks model loading, direction streams,
PCA projection, spectral weighting, and U-statistic reconstruction jointly.

## Pilot result

At R=64, antithetic positive-trace coverage is 90--100% across the four noise
levels, compared with 9.7--73.6% for the one-sided estimator. At R=512 all 72
antithetic traces are positive at every noise level.

Relative to the existing independent-center smooth-Jacobian estimate:

| R | antithetic median MC SE / smooth | antithetic median absolute relative error |
|---:|---:|---:|
| 64 | 0.50--0.67 | 0.19--0.29 |
| 128 | 0.25--0.36 | 0.05--0.09 |
| 256 | 0.12--0.18 | 0.04--0.09 |
| 512 | 0.06--0.09 | 0.018--0.036 |
| 1024 | 0.03--0.04 | 0.008--0.023 |

At fixed R, the antithetic MC SE is approximately 5 times smaller than
one-sided at 0.5/255, 42 times smaller at 2/255, 150 times smaller at 8/255,
and 200 times smaller at 16/255. The scale dependence directly supports the
even-response variance explanation.

The observed MC SE decreases close to 1/R over this range, rather than the
generic asymptotic 1/sqrt(R) rate. The degenerate second-order U-statistic term
therefore dominates at the tested budgets. Earlier 1/sqrt(R) direction-count
projections were explicitly conservative and should be replaced by these
nested empirical curves.

## Full 250-row validation

The selected R=512 design was then run on all 84 geometries and all 250
site-model rows. All 250 traces are positive at every noise level. The original
R=64 one-sided trace is reproduced with relative L2 error 6.87e-6.

| noise SD x255 | median MC SE / smooth | median absolute relative error | median signed relative error |
|---:|---:|---:|---:|
| 0.5 | 0.061 | 0.034 | +0.001 |
| 2 | 0.057 | 0.022 | -0.003 |
| 8 | 0.053 | 0.022 | -0.018 |
| 16 | 0.062 | 0.027 | -0.024 |

Thus the full estimator has complete support, approximately 5--6% median MC
uncertainty relative to the smooth trace, and 2--3% median discrepancy from the
independent-center smooth estimate.

The stable estimator removes the original positive-support selection artifact.
For the primary calibrated-control endpoint, site-residual Spearman correlations
after excluding CLIPAG and robust RN50 are:

| noise SD x255 | control slope | calibrated control MSE |
|---:|---:|---:|
| 0.5 | -0.073 | 0.120 |
| 2 | -0.126 | 0.197 |
| 8 | -0.194 | 0.261 |
| 16 | -0.162 | 0.228 |

After additionally excluding untrained AlexNet, slope correlations range from
-0.103 to +0.064 and MSE correlations from 0.061 to 0.114. This is consistent
with the smooth/neighborhood benchmark: much of the all-model association is
the separation of robust models, while the seven conventional models do not
show a common monotonic effect.

## Decision

Use antithetic Stein at R=512 as the full biological benchmark estimator. It
has complete pilot support and is already within a few percent of the smooth
reference. Retain R=64, 128 and 256 as convergence diagnostics. R=1024 is useful
as a pilot validation but is not necessary for the full 84-geometry run.

Do not use the original one-sided Stein estimator for method ranking. Increasing
its direction count alone leaves severe large-noise instability, whereas the
antithetic control variate attacks the dominant variance source directly.

The maintained synopsis v1.4 contains every antithetic nested-R trace, its MC
SE, and session-specific V values. The primary method benchmark uses R=512.

Plot-ready outputs are `stein_nested_summary.csv`,
`stein_nested_site_estimates.csv.gz`,
`stein_nested_biological_correlations.csv`, and
`stein_nested_full_validation.png`. Bulk per-seed deltas and Gram matrices are
stored under `$STORE_DIR/Projects/AccentuationPredRMT/nonlinear_control/stein_antithetic_full_v1`.

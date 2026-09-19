# Stein estimator stability audit

## Question

The biological benchmark sometimes makes the forward-only Stein predictor look
strong, but many rows are absent. This audit asks whether those missing rows are
software failures, whether increasing the number of Monte Carlo directions
would fix them, and whether correlations computed on the remaining rows are
comparable with the other geometry estimators.

## Why a valid estimate can be negative

For PC response change

\[
\Delta_{rk}=f_k(x+\tau z_r)-f_k(x),\qquad z_r\sim N(0,I_d),
\]

the implemented estimator is the off-diagonal U-statistic

\[
\widehat q_k={1\over R(R-1)\tau^2}
\sum_{r\ne s}\Delta_{rk}\Delta_{sk}z_r^\top z_s.
\]

Its population expectation is the nonnegative smoothed-Jacobian power
\(\|J_\tau(x)^\top u_k\|^2\), but an individual unbiased U-statistic is not
constrained to be nonnegative. The apparent failures are therefore negative
finite-sample estimates that cannot be logged, rather than exceptions, NaNs,
OOMs, or corrupt archives.

This estimator is particularly noisy here because the input has
\(d=3\times224\times224=150{,}528\) coordinates, whereas each seed image uses
only \(R=64\) Gaussian directions. The irrelevant components of
\(z_r^\top z_s\) contribute large variance. Larger \(\tau\) further weakens the
linear score signal relative to nonlinear response variation.

## Empirical audit

All values below use the saved per-direction jackknife pseudovalues, preserve
covariance across 750 PCs, and apply the site-specific ridge spectral weight
before calculating Monte Carlo uncertainty.

| noise SD x255 | negative among 250 | median Stein / smooth | median abs Stein z | median R for MC SE = 50% of smooth |
|---:|---:|---:|---:|---:|
| 0.5 | 24.0% | 1.12 | 0.274 | 7,278 |
| 2 | 30.8% | 0.91 | 0.091 | 266,318 |
| 8 | 84.0% | -11.70 | 0.142 | 3,049,425 |
| 16 | 86.8% | -14.69 | 0.123 | 6,502,794 |

The last column is an asymptotic \(R^{-1/2}\) extrapolation, with the smooth
trace used only as a positive reference scale. It should not be read as a
precise compute prescription: a second-order U-statistic can improve faster
than this while its degenerate variance term dominates. It does show that a
small increase from 64 to 128 or 256 is not supported as a complete fix,
especially at 8/255 and 16/255. A nested-direction rerun is required to measure
the actual scaling curve.

The problem is strongly model dependent. Positive counts out of 25 at
\(\tau=0.5/255\) are 1 for RegNetY-640, 6 for DINO-RN50, 19 for standard RN50,
and 21--25 for most of the remaining models. Thus positivity filtering changes
model composition rather than merely reducing every model equally.

## Why the benchmark correlation can look good

The seven-model marker at \(\tau=0.5/255\) uses 115 positive rows out of the
175 possible rows. On those same 115 rows, calibrated-MSE Spearman correlations
are 0.192 for Stein, 0.169 for smooth, 0.170 for neighborhood, and 0.174 for
finite-noise variance. On all 175 rows, the three estimators that remain
positive have correlations 0.114, 0.103, and 0.110. The apparent Stein lead is
therefore small for MSE and is evaluated after estimator-dependent filtering.

The most dramatic case is \(\tau=8/255\) after excluding the two robust models.
There are 40 positive Stein rows, but only 29 rows from 12 sites have at least
two surviving models and can contribute a within-site contrast. Stein has
\(\rho=0.647\). On exactly those selected rows, smooth and finite-noise variance
also rise to 0.548 and 0.501; on their complete 200-row support they are only
0.259 and 0.257. This demonstrates substantial support-selection inflation.
The high correlation is not evidence that the noisy Stein estimate is the best
geometry estimator.

For seven-model control slope at 0.5/255, Stein is larger in magnitude on the
common 115-row support (rho=-0.249 versus -0.119 to -0.137 for the alternatives).
That result can remain an exploratory observation, but it is post-selected on
Stein positivity and should not be ranked against methods evaluated on all 175
rows.

## Recommended use and next computation

Keep Stein as a forward-only diagnostic and omit it from the primary method
ranking until its support is nearly complete. Do not clamp negative values to
zero or take their absolute value; either operation introduces a larger bias.
Use smooth or neighborhood estimates as the current finite-scale predictors.

The next focused GPU pilot should use common nested direction prefixes
\(R=64,128,256,512,1024\) on representative high- and low-stability models,
save the raw response changes, and report sign recovery and error relative to
the existing smooth estimator. It should also test the antithetic score

\[
{f(x+\tau z)-f(x-\tau z)\over2\tau}z,
\]

whose off-diagonal cross-product has the same population target but cancels the
baseline and even response terms. That variance reduction is likely more useful
than blindly multiplying the current one-sided sample count.

## Reproducible outputs

- `analyze_stein_stability.py`: reconstructs trace-level MC errors from all 840
  seed archives and performs common-support comparisons.
- `stein_mc_stability.csv.gz`: one row per site/model/noise level with signed
  trace, MC errors, and direction-count projections.
- `stein_stability_summary.csv`: noise-level stability summary.
- `stein_stability_model_support.csv`: positive support by model and noise level.
- `stein_common_support_correlations.csv`: full-support and Stein-positive
  within-site correlations for slope and calibrated MSE.
- `stein_stability_diagnostics.png`: model support, signed dispersion, projected
  sign failures, and common-support association plot.

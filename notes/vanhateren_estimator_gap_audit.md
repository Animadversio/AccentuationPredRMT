# Van Hateren estimator mismatch audit

This audit reuses the 100 paired natural-image trials in
`vanhateren_selection_estimation_local_20260917/mc_raw.csv` on the authorized
NFS scratch. It changes neither the primary experiment nor its estimators.

Reproduce from the repository root:

```bash
python scripts/dissect_vanhateren_estimator_gap.py --raw /n/netscratch/kempner_binxuwang_lab/Everyone/binxuwang/AccentuationPredRMT/vanhateren_selection_estimation_local_20260917/mc_raw.csv
```

Plot-ready audit tables and figures are cached under
`notebooks/outputs/pixel_ridge/vanhateren_selection_estimation/gap_audit`.

## What is actually compared

The generalization curve uses ordinary leading risk DE. There is no distinct
Gaussian integration correction to its quadratic expectation. The
"distributional DE" accentuation curve is a diagonal Gaussian weight surrogate,
not a proved full distributional deterministic equivalent. All MC trials use
the policy's one deterministic alpha at each noise level; this figure contains
no trial-to-trial empirical CV alpha variability.

## Exact paired-noise dissection

The first seven middle-column settings have the same lambda=1e-8. For a fixed
trial their weights have the exact form w(sigma)=w0+sigma*u, because both X and
the standard noise vector are shared. Consequently normalized generalization
error is a quadratic in sigma, and the accentuation slope is

    R(sigma) = (a+b*sigma)/(1+c*sigma+d*sigma^2).

Here d=||u||^2/||w0||^2; it is not the absolute noise-weight norm. Coefficients
are identifiable from the seven saved slopes. Reconstruction residuals across
100 trials are below 6.3e-14 (slope) and 3.4e-15 (generalization).

| Quantity | DE | MC mean | MC / DE |
| --- | ---: | ---: | ---: |
| Noiseless Egen/S | 5.67167e-5 | 6.01267e-5 | 1.0601 |
| Coefficient of sigma^2 in Egen/S | 5.03945e-4 | 1.88670e-3 | 3.7439 |
| Coefficient d in normalized D | 0.647787 | 3.830074 | 5.9125 |

At sigma=0.223349, Egen/S is 8.18559e-5 (DE) versus 1.53842e-4 (MC).
MC Eacc/S=0.0269217, split exactly into squared mean-slope bias 0.0250467
and slope variance 0.0018750. Thus only 7% is trial slope variance. The Gaussian
surrogate predicts 0.00098375, almost identical to leading DE 0.00097988.
Adding that surrogate's fluctuations cannot fix the underestimated base
noise amplification.

The mismatch grows while noise increases but regularization stays at its
lower boundary. At sigma=0.312477 lambda rises to 6.700884e-5; subsequently
the amplification discrepancy declines. This supports a near-ridgeless
noise-sensitivity explanation, not a high-noise-only phenomenon.

## Independent structural failure of the Gaussian surrogate

In the exactly noiseless min-norm limit w=P_X beta with P_X an orthogonal
projector. Thus N=beta^T P_X beta=D identically for every training design,
and Eacc=0 even though Egen may be positive. A Gaussian approximation with
independent PC-coordinate fluctuations need not preserve this exact identity.
Matching coordinate second moments does not match joint quadratic fluctuations.

At sigma=0, lambda=1e-8, MC Eacc/S=9.332e-13, whereas the Gaussian surrogate
gives 1.769e-6. The finite positive lambda explains why MC is not exactly zero;
it does not explain the surrogate's much larger floor.

For the Gaussian-oracle policy at sigma=0.611624, MC Eacc/S=4.4596e-5
versus surrogate 9.0156e-6. The empirical squared mean-slope bias is 3.6093e-5
(81% of total), with variance 8.5030e-6. Optimizing an almost-canceling predicted
mean accentuation error makes small errors in that predicted mean consequential.

## What remains unresolved

This audit identifies which risk terms fail, but does not isolate the ultimate
source of their mismatch. Natural-image non-Gaussian/dependent coordinates,
the finite training-pool/population mismatch, and finite-sample effects require
controlled tests. A Gaussian design with the identical spectrum/teacher and
matched n, followed by exact conditional-on-X noise risk, is the appropriate
next control. The covariance spectrum alone is not a validation of the assumed
design distribution. No evidence here justifies calling the discrepancy mere
Monte Carlo integration noise, nor proves that all of it is caused by
non-Gaussianity. Numerical conditioning should be checked in that control too.

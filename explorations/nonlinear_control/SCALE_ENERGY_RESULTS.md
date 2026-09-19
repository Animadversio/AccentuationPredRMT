# Initial scale-up energy analysis

## Definition and aggregation

For PCA direction (u_k), the exact local RGB-coordinate energy is

\[
q_k(x)=u_k^\top J(x)J(x)^\top u_k=\|J(x)^\top u_k\|_2^2.
\]

We report the raw retained-subspace energy (\sum_{k=1}^{750}q_k), but use
the Proposition 13 trace as the primary comparison:

\[
T(\kappa)=\sum_{k=1}^{750}q_k\frac{s_k}{(s_k+\kappa)^2}.
\]

For each site geometry, (\kappa) is chosen separately so that

\[
df_2(\kappa)=\sum_k\left(\frac{s_k}{s_k+\kappa}\right)^2=375.
\]

This matches the effective dimension across models and makes the trace
invariant to a uniform rescaling of the feature coordinates. The 84 unique
geometries are expanded through the site manifest to all 250 site-model pairs.
Each site's energy is averaged over the same ten seed images, then over five
sites per subject. Bars are means of the five subject means, dots are the
individual subjects and error bars are ±2 subject-level SE. These error bars
describe between-subject variation; they are not a formal population CI with
only five subjects.

## Exact local energy

At matched (df_2=375), the model means are:

| Model | Mean trace | ±2 subject SE |
|---|---:|---:|
| CLIP RN50 | 81,135 | 10,691 |
| AlexNet | 24,901 | 8,713 |
| DINO RN50 | 16,194 | 2,297 |
| RegNetY-640 | 12,419 | 2,277 |
| RN50 | 11,050 | 3,506 |
| DINOv2 | 8,182 | 637 |
| SigLIP2 | 7,234 | 1,976 |
| RADIO | 4,421 | 258 |
| CLIPAG | 61.9 | 29.0 |
| Robust RN50 | 42.4 | 2.69 |

The highest and lowest means differ by about 1,913-fold. Raw
(\sum_kq_k) gives a different ranking: DINOv2 and SigLIP2 are largest, while
AlexNet and CLIPAG are smallest. The disagreement confirms that raw Jacobian
norm is strongly affected by feature scale and spectrum; matched-(df_2)
(T(\kappa)) is the relevant quantity for comparing the control-error term.

## Finite-noise geometry

All finite-noise values below are divided by the exact local trace computed
with the same site's matched-(df_2) weights.

At (\tau\times255=0.5), CLIPAG and Robust RN50 are nearly local-linear:
their response-covariance ratios are 0.993 and 0.981, and their smoothed
Jacobian ratios are 0.993 and 0.973. Most other models already lose coherent
smoothed-gradient energy even though their neighborhood gradient energy stays
near the exact local value. For example, standard RN50 has smoothed and
neighborhood ratios 0.489 and 0.936. This gap is consistent with gradients
remaining individually large while rotating or cancelling across nearby
inputs.

At (\tau\times255=16), CLIPAG remains unusually stable: its covariance,
smoothed-Jacobian and neighborhood ratios are 1.00, 0.932 and 1.08. Robust
RN50 retains 0.710, 0.499 and 1.12. In most other models the covariance ratio
falls below 0.1 and the smoothed-Jacobian ratio below 0.05; AlexNet is the main
exception, with covariance 0.321 and neighborhood energy 1.05.

CLIPAG and Robust RN50 also have very large total response-change ratios at
the largest noise scale, 24.5 and 51.5, despite much smaller covariance ratios.
Because total response change is variance plus squared mean drift, this points
to a large coherent/even drift component rather than stochastic response
variance. The saved odd/even/drift spectra should be examined next.

All nine CLIP RN50 geometries have marginal autograd-versus-finite-difference
diagnostics. Its exact VJP, response covariance and total finite response
change remain valid, but its smoothed and neighborhood gradient estimates are
hatched in the figures and should be treated as provisional.

## Outputs

- `figures/nonlinear_control/scale_energy/model_exact_energy_bars.{png,pdf}`
- `figures/nonlinear_control/scale_energy/model_finite_response_bars.{png,pdf}`
- `figures/nonlinear_control/scale_energy/model_smoothed_gradient_bars.{png,pdf}`
- `tables/nonlinear_control/scale_energy/model_exact_energy_summary.csv`
- `tables/nonlinear_control/scale_energy/model_finite_energy_summary.csv`
- compressed site/seed tables and uncompressed subject-level plot-ready tables
  in the same table folder

Reproduce with:

```bash
python explorations/nonlinear_control/analyze_scale_energy.py \
  --input "$STORE_DIR/Projects/AccentuationPredRMT/nonlinear_control/mass_v1" \
  --site-manifest tables/nonlinear_control/scale_manifest/site_model_manifest.csv \
  --qc tables/nonlinear_control/scale_manifest/production_deep_qc.csv \
  --tables tables/nonlinear_control/scale_energy \
  --figures figures/nonlinear_control/scale_energy
```

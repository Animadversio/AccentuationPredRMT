# Why control- and encoding-session V correlations differ

Date: 2026-09-19. All correlations below exclude CLIPAG and robust RN50 unless
stated otherwise. They are descriptive and do not treat 200 site/model rows as
independent animals.

## Three endpoints

The maintained synopsis separates:

1. `V_encoding_session`: encoding-session responses on all 195 held-out images.
2. `V_encoding_session_matched`: encoding-session responses restricted to the
   exact 22/24/50-image subset re-presented during the control session.
3. `V_control_session`: control-session responses on that same matched subset.

Thus (1) versus (2) diagnoses held-out count/image composition, while (2)
versus (3) diagnoses recording-session response changes on identical images.

## Decomposition of the sign change

The trace term is shared by all three endpoints. Across the 200 non-robust
site/model rows, its Spearman correlation with control slope becomes more
negative as smoothing grows: -0.066, -0.135, -0.165, -0.151 at noise SD ×255
of 0.5, 2, 8, 16.

The MSE factor changes direction:

| Generalization MSE | Spearman with control slope |
|---|---:|
| Encoding session, all 195 | -0.092 |
| Encoding session, matched subset | -0.043 |
| Control session, matched subset | +0.245 |

Consequently, multiplying by the same trace yields weak negative encoding V
correlations but weak positive pooled control V correlations. The matched
encoding endpoint remains centered below zero, so reduced image count or image
composition alone does not explain the sign reversal.

There is also a between-site/within-site reversal. For smooth noise levels
0.5, 2, 8 and 16, site-demeaned control-session V correlations are -0.100,
-0.159, -0.189 and -0.179. Across the 25 individual sites, median within-site
Spearman correlations are -0.190, -0.214, -0.095 and -0.119; 14–16 of 25 sites
are negative. The weak positive pooled coefficient is therefore driven by
monkey/site baseline differences, a Simpson's-paradox pattern.

## Finite-image bootstrap

We resampled held-out image identities 2,000 times, using one shared resample
within each monkey for all its sites/models. This preserves shared-stimulus
dependence. It quantifies only finite-image uncertainty, not animal, site,
neural-trial, Jacobian-probe or model-family uncertainty.

For all five monkeys, control-session bootstrap means are +0.140, +0.108,
+0.079 and +0.072 across the four smooth levels. Their 95% intervals are
[-0.003, 0.288], [-0.037, 0.256], [-0.059, 0.210], and [-0.062, 0.200].

After restricting to the three monkeys with 50 control-session held-out images
(red, paul, venus), means shrink to +0.093, +0.090, +0.062 and +0.044, and all
95% intervals are wide and cross zero. The full-195 encoding endpoint remains
negative for all five monkeys, with intervals below zero; its matched-image
version remains negatively centered but has intervals crossing zero.

## Recommendation

Do not remove Leap and Three0 from the primary descriptive table. They are
entire animals and primarily STS sites; deleting them changes the biological
target and confounds sample count with monkey and region. Report them in the
primary endpoint with their actual coverage, and always accompany it with:

- the `n50_monkeys` sensitivity analysis;
- encoding-full and encoding-matched endpoints;
- site-demeaned or site-fixed-effect analysis;
- image-resampling uncertainty.

The current evidence supports neither a reliable positive control-session
association nor a universal negative association. The defensible conclusion
is that the pooled sign is endpoint- and grouping-sensitive. More control-
session natural anchors, especially for Leap and Three0, are needed to resolve
the marginal pooled association.

## Site-within effects and control error

The primary descriptive association now subtracts each site's model mean from
both log10 V and the biological outcome. Equivalently, it fits a site fixed
intercept. The regression table also reports a coefficient per log10 V and a
site-clustered standard error.

After excluding CLIPAG and robust RN50, control-session smooth V has the
following site-within associations:

| noise SD × 255 | slope Pearson / Spearman | direct control-MSE Pearson / Spearman |
|---:|---:|---:|
| 0.5 | -0.029 / -0.100 | +0.158 / +0.129 |
| 2 | -0.146 / -0.159 | +0.277 / +0.216 |
| 8 | -0.280 / -0.189 | +0.327 / +0.238 |
| 16 | -0.307 / -0.179 | +0.324 / +0.228 |

Thus larger V is associated within a site with lower control slope and higher
direct control error, the paired directional prediction of the theory. At
smooth 2/255 the pooled V--control-MSE correlation is almost zero (Pearson
0.015, Spearman 0.018), whereas the site-within association is positive. Site
baselines were masking rather than producing this error association.

Direct control MSE is computed without recalibration:

    MSE_control = mean_i (y_i_control - yhat_i_posthoc,floor)^2.

`y_i_control` is the trial-averaged standardized neural response to each
presented accentuated image. `yhat_i_posthoc,floor` is the generating model's
post-hoc prediction for its own target unit, clamped to the unit's firing floor
at comparison time. No slope or intercept is refitted; `control_refit_mse` is
stored separately and answers a different question.

Numerical site-within outputs:
`tables/nonlinear_control/biological_validation/site_centered_associations.csv`.

Numerical outputs:
`tables/nonlinear_control/biological_validation/session_endpoint_image_bootstrap_summary.csv`
and `session_endpoint_image_bootstrap_draws.csv.gz`.

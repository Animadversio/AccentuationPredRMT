# Encoding-to-control anchor affine calibration

## What was already applied

The canonical Jacob preprocessing first removes within-control-session daily
offset and gain. After peak-window averaging and outlier rejection, every raw
control trial on day `d` is transformed as

    y_ctrl,z = (y_ctrl,raw - mean(day-d natural anchors)) /
               sd(day-d natural anchors).

The source is `pnc/preproc/pipeline.py::standardize(method="anchorDay")` at the
pinned upstream commit `9b6fd95228d1b77146f8b279c7c228914ca34bb8`. A day with
fewer than five anchor trials uses all trials from that day. Only Three0 on
250508 uses this fallback; every other control day has at least six anchors.
The existing unqualified `control_*` synopsis fields already used these
anchorDay-standardized responses.

This operation does not align the encoding-session response scale with the
control-session response scale. The canonical `control_mse` compares the
floor-clamped encoding-model prediction directly with the anchorDay-normalized
control response.

## Added cross-session mappings

A single frozen site-level neural mapping estimates recording remapping directly:

    y_ctrl,anchor = a_s + b_s y_enc,anchor + error.

It then transforms every floor-clamped model prediction with the same
site-specific `a_s, b_s`. The mapping is fitted once per neuron and shared by
all ten models at that site. It therefore cannot absorb model-specific
natural-image prediction errors.

Each mapping is computed with three anchor subsets: all, encoding-training,
and encoding-heldout. The training-anchor version is the recommended pair for
the theory analysis because it fits the map on 24/25/50 training anchors and
then uses the disjoint 22/24/50 heldout anchors to estimate generalization MSE.

## Results and default

Across 250 site-model rows, the site-level all-anchor map reduces direct
control MSE by a median factor of 0.684 and improves 90.8% of rows. The
training-anchor site map reduces it by a median factor of 0.680 and improves
78.4% of rows. All fitted training-anchor site gains are positive; their
median is 0.760 across the 25 sites.

For theory-facing analyses, use:

- biological endpoint: `control_site_train_anchor_affine_mse`;
- matched heldout error: `control_session_site_train_anchor_affine_gen_test_mse`;
- matched geometry product: `geom_<method>__V_control_session_site_train_anchor_affine_mean`.

Keep `control_mse` as the identity-scale reference and
`control_refit_mse` as the explicitly outcome-fitted, leaky lower bound. The
all-anchor site map, `control_site_anchor_affine_mse`, is useful when only the
biological control endpoint is analyzed and maximum calibration precision is
preferred over a disjoint natural-image generalization estimate.

With robust RN50 and CLIPAG removed, matching the training-anchor site map in
both the outcome and V modestly strengthens site-residual Spearman association:
smooth 8/255 changes from 0.238 to 0.259, neighborhood 16/255 from 0.207 to
0.233, and variance 16/255 from 0.256 to 0.272.

## Reproduction

The row-level calibrated accentuation clouds are in
`control_anchor_affine_clouds.csv.gz`. Rebuild the synopsis, affine summaries
and figure with:

```bash
MPLCONFIGDIR=/tmp/mpl-biological-synopsis \
XDG_CACHE_HOME=/tmp/xdg-biological-synopsis \
/n/home12/binxuwang/.conda/envs/torch2/bin/python \
  explorations/nonlinear_control/build_biological_synopsis.py

MPLCONFIGDIR=/tmp/mpl-crossday \
XDG_CACHE_HOME=/tmp/xdg-crossday \
/n/home12/binxuwang/.conda/envs/torch2/bin/python \
  explorations/nonlinear_control/analyze_crosssession_affine.py
```

# Nonlinear feature geometry: exploratory initialization

Forward-only noise-scale extension: see `FORWARD_NOISE_RESULTS.md` for the
four-model results and `forward_noise_math.md` for the distinction between
smoothed Jacobian energy, neighborhood energy, noise covariance and mean drift.
The complete Chinese formalism and interpretation is in
`FORWARD_NOISE_FORMALISM_ZH.md`.

## Current analysis map

The scaled geometry and biological-validation line now has four maintained
documents:

- `SCALEUP_PLAN.md`: 25 sites × 10 models, 84 unique geometry caches, spectral
  storage and cluster execution design.
- `SCALE_ENERGY_RESULTS.md`: exact, smooth and neighborhood energy comparisons
  across encoding models.
- `BIOLOGICAL_VALIDATION_RESULTS.md`: cross-session calibration, maintained
  250-row synopsis and biological endpoint benchmarks.
- `STEIN_ANTITHETIC_RESULTS.md`: instability audit, nested-direction GPU
  validation and the final antithetic Stein estimator. The preceding failure
  analysis is retained in `STEIN_STABILITY_RESULTS.md`.

The original one-sided score estimator
`[f(x+tau*z)-f(x)]z/tau` is no longer used for method ranking. In 150,528 input
dimensions its finite-sample off-diagonal U-statistic was often negative and
created estimator-dependent benchmark support. The maintained forward-only
score estimator is

\[
{f(x+\tau z)-f(x-\tau z)\over2\tau}z,
\]

with common nested direction prefixes. The full 84-geometry validation uses
R=512 as the benchmark: all 250 traces are positive, median MC SE is 5.3--6.2%
of the independent-center smooth trace, and median absolute trace discrepancy
is 2.2--3.4%. Synopsis v1.4 stores R=64, 128, 256 and 512 estimates and MC SEs.

Reproduce the GPU estimator and CPU integration with:

```bash
sbatch --export=ALL,STEIN_PILOT_IMAGES=10,STEIN_PILOT_DIRECTIONS=512,\
STEIN_PILOT_GEOMETRIES=100,\
STEIN_PILOT_OUTPUT="$STORE_DIR/Projects/AccentuationPredRMT/nonlinear_control/stein_antithetic_full_v1" \
  explorations/nonlinear_control/run_stein_nested_pilot.sbatch

python -u explorations/nonlinear_control/analyze_stein_nested_pilot.py \
  --input "$STORE_DIR/Projects/AccentuationPredRMT/nonlinear_control/stein_antithetic_full_v1"
python -u explorations/nonlinear_control/build_biological_synopsis.py
python -u explorations/nonlinear_control/analyze_site_centered_effects.py
```

The full per-seed deltas and Gram matrices remain in STORE_DIR. Plot-ready
summaries, the 250-row synopsis and final figures are committed to this repo.

## What is exact, and what remains a conjecture

Let centered features be φ(x) ∈ Rᵖ, C = Cov(φ), J(x) = ∂φ/∂x ∈ Rᵖˣᵈ,
and C uₖ = sₖ uₖ. Choose an evaluation image distribution Q and define

\[
G_Q=\mathbb E_{x\sim Q}[J(x)J(x)^\top],\qquad
q_k= u_k^\top G_Q u_k
    =\mathbb E_Q\|\nabla_x(u_k^\top\phi(x))\|^2.
\]

For κ > 0, the following identity is exact for any differentiable feature map
(almost everywhere is sufficient for piecewise differentiable networks):

\[
T_Q(\kappa)=\operatorname{Tr}[G_Q(C+\kappa I)^{-2}C]
 =\sum_k q_k\frac{s_k}{(s_k+\kappa)^2}.
\]

No Jacobian needs to be materialized: backpropagate one PC score to the input,
square its gradient and sum. Average **squared gradients**, not gradients before
squaring. For φ(x)=Fx, J=F and this reduces exactly to the original expression.
G and C need not commute for this trace identity.

For a fixed fitted feature readout ŵ, prediction gradient is ĝ(x)=J(x)ᵀŵ.
If true response is f★(x), define

\[
D(x)=\|\hat g(x)\|^2,\qquad
N(x)=\nabla f^\star(x)^\top\hat g(x).
\]

A small predicted-target step δx=Δ ĝ/D gives predicted change Δ and true change
ΔN/D + O(‖δx‖²). Thus D/N−1 describes the reciprocal local response-gain error
when N≠0; it is not automatically a finite-trajectory control-error formula.
For a nonlinear teacher there is no single β★. For a linear teacher,
∇f★=β★ and the familiar alignment expression returns pointwise. If f★=w★ᵀφ,
then N=w★ᵀJJᵀŵ. Readout bias has no effect on gradients.

Conditional on a fixed test x (independent of training randomness), with
μ=E_train[ŵ] and V=Cov_train(ŵ), the following decomposition is exact:

\[
\mathbb E_{\rm train}D(x)
 =\mu^\top J(x)J(x)^\top\mu
  +\operatorname{Tr}[J(x)J(x)^\top V].
\]

If the **additional** DE ansatz

\[
\mu\simeq(C+\kappa I)^{-1}Cw^\star,\quad
V\simeq\frac{E_{\rm gen,DE}+\sigma^2}{n}
          (C+\kappa I)^{-1}C(C+\kappa I)^{-1}
\]

holds for these nonlinear features, its variance contribution is precisely
(E_gen,DE+σ²)T_Q/n. This ansatz is the statistical hypothesis to test, not a
consequence of substituting J for F. Correlated/non-Gaussian features,
misspecification, and feature-dependent residual variance can alter V.
Even ordinary heteroskedastic sandwich covariance involves
E[residual² φφᵀ], which need not be proportional to C.
The signal term requires off-diagonal entries of UᵀG_QU; qₖ alone is insufficient.
Also E[D/N] differs from E[D]/E[N], and evaluation on training images does not
satisfy the test-point independence used above. This initial training-image
pilot estimates geometry only.

## Why the first 750 PCs are useful, but not population truth

Use the *training-fitted*, unwhitened PCA coordinates and estimate sₖ from the
training score variance (ddof=1, matching sklearn PCA). Do not refit PCA on all
969 images and call that the training basis. The pilot checks that training
covariance in the exported basis is close to diagonal before using the formula.

For the actual PCA750 predictor, these 750 coordinates define its complete
feature space. For an untruncated backbone feature map they define only a
partial spectral trace. With the true eigenbasis, omitted contributions are
nonnegative, but no numerical tail bound follows from explained variance alone:
small sₖ can coexist with large qₖ. An empirical PCA plug-in is not a certified
lower bound on the population trace. Near κ=0, qₖ/sₖ makes tail uncertainty
especially consequential. For κ>0 the spectral weight peaks at sₖ=κ.

κ is the effective ridge parameter, not necessarily sklearn's alpha. Under
the same DE hypothesis, λ=alpha/n for an unnormalized sklearn squared-loss
objective, and κ solves κ−λ=(κ/n)Σₖsₖ/(sₖ+κ). The script sweeps κ directly,
without claiming this calibration has been validated for the neural features.

## Concrete experimental starting point

- Source project: `/n/home12/binxuwang/Github/Closed-loop-visual-insilico`.
- Subject: `red_20250428-20250430`, channel 2 selection, `resnet50_robust`,
  layer `.layer4.Bottleneck1`, exported unwhitened PCA750.
- Existing NSDencimg cache has 969 × 750 scores, image paths and `is_train`.
  Despite the cache filename, its rows include multiple stimulus families.
- The exported TorchScript PCA stores axes/mean, but no eigenvalues. Recover
  variances from the training rows of cached scores, avoiding the 16 GB
  all-layer PCA pickle. Check model outputs against cached scores before VJPs.
- Use the main project's model loader, feature hook and exported PCA. Its
  convenience PCA predictor has a hardcoded CUDA gradient check; this adapter
  composes the same pieces directly and also supports a CPU timing pilot.
- Geometry uses RGB [0,1] on the **post-resize/crop grid**: differentiate the
  normalized tensor, then divide each gradient channel by its normalization
  standard deviation. This is not the geometry of original 1024px images,
  random crops, Fourier parameterizations or the full accentuation optimizer.

## Running and caching

Small audit and timing pilot (one image, eight PCs spread across ranks):

```bash
python -u explorations/nonlinear_control/run_pilot.py \
  --output tables/nonlinear_control/robust_resnet50_pilot --gradients
```

The script saves training_spectrum.csv, audit.json, timing.json,
gradient_power.csv, pc_summary.csv and trace.csv. Eight-PC trace values are
explicitly **partial sums**, never an estimate obtained by scaling to 750.
With one image the standard error is undefined (NaN), not zero.

Full 750-PC geometry, on a GPU allocation, after checking the pilot ETA:

```bash
test -n "$STORE_DIR"
OUT="$STORE_DIR/Projects/AccentuationPredRMT/nonlinear_control/robust_resnet50_32img"
mkdir -p "$OUT"
python -u explorations/nonlinear_control/run_pilot.py --gradients --all-pcs \
  --images 32 --device cuda --output "$OUT" > "$OUT/run.log" 2>&1
# In a second terminal: tail -f "$OUT/run.log"
```

Each image's numerical results are checkpointed; this initial version does
not automatically resume interrupted runs. Never reuse one output directory
for different models, image sets or coordinate conventions.

## Most useful next experiments

1. Plot sₖ, qₖ and qₖ/sₖ against PC rank; plot cumulative T(κ) at several κ.
   Report rank cutoffs 50/100/250/500/750 and image-bootstrap intervals.
2. Evaluate held-out images and accentuation seeds separately from training
   images. Compare Q-dependence and the mixture of stimulus families.
3. Compare standard/robust ResNet and DINO with matched preprocessing geometry.
   Each model has its own training PCA and spectrum; channel-based layer
   selection is a confound, so also compare prespecified layers.
4. Validate the DE using synthetic teachers and repeated noisy ridge fits on
   these fixed features. Compare measured readout covariance against the
   predicted spectral covariance, and measured gradient energy against its
   signal + variance decomposition. Use held-out evaluation inputs.
5. Test the nonlinear *local* D/N directly with a teacher model, then increase
   target step size to measure where curvature invalidates the local formula.
6. If exact per-PC VJPs are too slow, use input Rademacher JVP probes:
   E_v[(UᵀJv)ₖ²]=qₖ. This estimates all PCs at once but requires probe-error
   estimates and a forward-mode-compatible model adapter. Cross-check against
   exact VJPs before adopting it. Weighted random-output VJPs can estimate
   T(κ) directly, but lose the per-PC profile.

## Fixed experiment seeds and GPU VJP blocks

The requested evaluation set is now available with `--evaluation seeds
--images 10`. It reads the ten seed paths from the model's cached experiment
configuration, preserves their order, and resolves the original PNGs under
`/n/holylabs/LABS/alvarez_lab/Lab/VVS_Accentuation/Stimuli`. Training images
continue to supply only the PCA spectrum. The saved evaluation_images.json
records each PNG path, corresponding cache row and training-membership flag.
Cache reproduction is checked on its original image (which can be a JPEG),
independently of computing the seed PNG gradient.

`--benchmark-chunks` benchmarks serial VJPs and batched exact VJPs with block
sizes 4, 16 and 32 on the same 32 PCs, checks numerical agreement, records
CUDA peak allocation and chooses the fastest measured block size. This uses
`torch.autograd.grad(..., is_grads_batched=True)`; it parallelizes the PC
directions without constructing the full pixel-by-feature Jacobian. Block
timings are a first-pass estimate, not a comprehensive kernel benchmark.

On 2026-09-19, existing Nanoclaw allocation 47072283 on holygpu8a11604 had an
idle H100 80 GB. The seed run reuses this allocation with `srun --overlap`:

```bash
srun --jobid=47072283 --overlap --ntasks=1 --cpus-per-task=2 \
  /n/home12/binxuwang/.conda/envs/torch2/bin/python -u \
  explorations/nonlinear_control/run_pilot.py --gradients --all-pcs \
  --evaluation seeds --images 10 --device cuda --benchmark-chunks \
  --output "$STORE_DIR/Projects/AccentuationPredRMT/nonlinear_control/robust_resnet50_seed10"
```

Allocation IDs are ephemeral; inspect `squeue -u binxuwang` and GPU utilization
before reusing this command. Log for this run: `run.log` in the output directory.

Cache validation now records relative L2 error and error measured in each
feature's training standard deviation (limits: 0.1% relative L2 and 0.5% of
feature SD at every coordinate). This avoids ill-conditioned relative errors
for near-zero image scores. GPU TF32 is disabled. A DINO run can additionally
use `--disable-xformers` to select native attention for batched derivatives;
its agreement with the original cache is still checked. Initial block timings
included cold backward kernels; subsequent code explicitly warms up serial
and batched paths. Do not quote the original serial-to-batched ratio as a
steady-state speedup.

Completed robust-ResNet seed run: all ten seeds are outside the training set.
The 750-PC image loop took about 14 seconds on H100, with 16-PC VJP blocks.
Block-vs-serial power agreement was better than 1e-7 relative on the pilot.
PC501–750 hold 8.53% of the retained training variance, but 43.39% of the
retained trace in the κ→0 limit; their fraction falls to 5.57% at κ/s₁=0.1.
These percentages describe this finite seed set and empirical PCA750 model,
not a population DE validation. Plot and small summaries are in
`figures/nonlinear_control/robust_resnet50_seed10/` and
`tables/nonlinear_control/robust_resnet50_seed10/`.

## Initial validation (2026-09-19)

The real-model eight-PC pilot passed cached-score reproduction. There are 774
training rows and 195 other rows. The relative Frobenius norm of off-diagonal
training covariance is 5.247e-5, so treating these exported coordinates as the
training eigenbasis is accurate to this diagnostic (not a bound on trace error).
The retained variances range from 1606.808 to 4.25954.
CPU pilot timing with two threads: about 0.066 s/PC, approximately 50 s/image
for 750 PCs, excluding setup. Two algebraic tests pass: the exact linear limit
with a noncommuting metric, and nonlinear VJPs with the RGB chain rule.

A two-image, all-750-PC run uses
`$STORE_DIR/Projects/AccentuationPredRMT/nonlinear_control/robust_resnet50_2img`;
its live log is `run.log` in that directory. Model setup and file access are
excluded from the projected VJP runtime.

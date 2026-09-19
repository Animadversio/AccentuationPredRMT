# Scale-up plan: 25 sites × 10 encoding models

## Inventory and unit of computation

The available data form exactly 250 site-model pairs: five subjects, five
selected sites per subject and ten models. Every pair has an exported PCA750,
969 cached PCA responses, the same 774-row training split and the same ten seed
images.

Geometry depends on model weights, preprocessing, selected layer, PCA axes and
evaluation images; it does not depend on the neural readout after layer
selection. Exact hashing of all 250 cached 969×750 PCA score matrices reduces
the computation to 84 unique model-layer geometries:

| Model | Pairs | Unique geometries |
|---|---:|---:|
| AlexNet training seed 01 | 25 | 11 |
| CLIPAG ViT-B/32 | 25 | 7 |
| DINOv2 ViT-B/14-reg | 25 | 9 |
| RADIO v2.5-B | 25 | 8 |
| RegNetY-640 | 25 | 9 |
| ResNet-50 | 25 | 9 |
| CLIP ResNet-50 | 25 | 9 |
| DINO ResNet-50 | 25 | 6 |
| Robust ResNet-50 | 25 | 6 |
| SigLIP2 ViT-B/16 | 25 | 10 |

The production preflight must hash/compare the TorchScript PCA `mean` and
`components` buffers for aliases, in addition to the exact score hash already
checked. If an alias fails, split that geometry ID rather than forcing dedup.
The 250-row manifest remains the authoritative mapping for site-level plots and
statistics; multiple sites may legitimately map to identical geometry values.

## Estimators to compute

For every unique geometry and each of the ten fixed seeds, retain the full
750-PC spectral vectors before any κ weighting:

1. Exact local Jacobian power, q_exact,k=||J(x)^T u_k||², from batched VJPs.
2. Forward-only local directional powers, for an independent exact-vs-forward
   check and Monte Carlo uncertainty.
3. Smoothed-Jacobian power, ||E_z J(x+τz)^T u_k||², using the debiased inner
   U-statistic.
4. Neighborhood gradient energy, E_z||J(x+τz)^T u_k||².
5. Centered noise covariance, Var[f_k(x+τz)]/τ².
6. Total finite response change, split into odd, even and mean-drift terms.
7. A direct Stein/score-function cross-estimator of smoothed-Jacobian power,
   used as a secondary forward-only diagnostic. For independent Gaussian
   perturbations z and z', its off-diagonal kernel is
   `[delta f_k(z) delta f_k(z') / tau^2] (z^T z')`. Its expectation is
   `||J_tau(x)^T u_k||^2`, but its variance grows badly with input dimension,
   so report its uncertainty and do not use it as the primary estimate.

These form three independent checks of the local limit: exact reverse-mode
VJP, forward-only randomized directional finite differences, and the Stein
cross-estimator as tau approaches zero. The finite-tau U-statistic remains the
primary estimate of smoothed-Jacobian power; neighborhood energy and response
covariance answer distinct nonlinear questions rather than serving as aliases
for it.

Use τ×255={0.5,2,8,16}, R=64 outer Gaussian directions and L=8 independent
smoothing centers. Do not clip RGB values in this Gaussian-smoothing analysis.
Use common random streams across geometries and τ, with independent streams
across seed images. Keep TF32 off. Choose finite-difference h per model from a
same-probe autograd pilot, then validate h and 2h at the largest τ on at least
one geometry per model.

## What to save

The primary artifact must be κ-free. For each geometry save:

- metadata: model/checkpoint ID, layer, PCA-buffer hash, preprocessing, input
  coordinate, split/seed hashes, RNG scheme, R/L/τ/h and software versions;
- spectrum s_k and PCA rank k;
- per seed × PC exact q_exact;
- per seed × τ × independent sampling unit × PC spectral values or jackknife
  pseudovalues for smooth, neighborhood, Stein, covariance, total, odd, even
  and drift estimators;
- per-PC means and MC standard errors as a lightweight summary;
- completeness/checksum records and timing/peak-memory logs.

For any later κ, construct mass

M_{r,k}(κ)=q_{r,k}s_k/(s_k+κ)²

and sum over k. Keeping outer-replicate vectors preserves PC covariance, so MC
errors for arbitrary κ can be recomputed correctly. Saving only per-PC standard
errors would not allow this. Also save cumulative sums ordered by PCA rank only
as derived plot-ready products, never as the sole data.

Raw inner derivatives d[r,l,k] are not required for changing κ. Keep them in a
separate optional archive for estimator-development audits. Based on the pilot,
ten-seed archives are about 100 MB/geometry with inner derivatives and 50–55 MB
without them before adding the small Stein pseudovalue array. Expected storage
for 84 geometries is about 8.4–9.1 GB full or 4.5–5.2 GB lean, plus small
exact-Jacobian and summary tables. Running all 250 independently would need
roughly 25 GB full and would duplicate scientific information.

Recommended layout under `$STORE_DIR/Projects/AccentuationPredRMT/nonlinear_control/mass_v1/`:

```text
manifest/{site_model_manifest.csv,geometry_manifest.csv,inventory_summary.json}
geometry/<geometry_id>/{config.json,spectrum.npz,seed_01.npz,...,seed_10.npz,DONE.json}
logs/<model>.{out,err}
summary/{pc_mass.parquet,trace_grid.parquet,qc.parquet}
```

Write a seed archive to a temporary filename, validate shapes/finiteness, then
atomically rename it. A geometry is complete only when DONE.json lists hashes
for all ten seeds. Reruns skip validated seeds, making preemption safe.

## Distributed execution

Use a ten-element Slurm array, one task per model, with an initial concurrency
cap of four (`--array=0-9%4`). Each task loads its backbone once and processes
the model's 6–11 unique layer/PCA geometries sequentially. This avoids 84
checkpoint loads and a Lustre metadata/I/O burst. Stage the model checkpoint
and the task's representative PCA files to node-local `/scratch/tmp` when the
loader supports path overrides; copy only finalized small/medium results back
to STORE_DIR. Never rely on node-local files after job exit.

Initial request per task: one H100/H200 GPU, 8 CPUs, 64 GB RAM, 2 hours, on a
requeue-capable partition. Pilot peak CUDA allocation was only 1.0–1.6 GB for
the measured models, but all five proxy-timed backbones must be benchmarked
before lowering the GPU class or memory request. Use `tqdm`/live logs at model,
geometry, seed and τ levels. A dependent CPU aggregation job validates all 84
DONE records, expands results to the 250-row alias manifest, and writes Parquet
summaries and figures.

Avoid one task per site-model: 250 concurrent loaders would repeatedly read
about 105 GiB of exported PCA files and large model checkpoints. Avoid one
multi-GPU monolith as well; independent model tasks are easier to resume and
load-balance.

## ETA and calibration plan

Measured on an H100, one ten-seed geometry at R=64, L=8 and four τ values
required approximately:

- CLIPAG: 13 s nested-MC projection; actual full loop was tens of seconds;
- robust RN50: 16 s nested-MC projection;
- standard RN50: 22 s nested-MC projection;
- DINOv2: 52 s nested-MC projection; actual image loop was 68 s.

The unified timing harness measured total projected compute (forward estimators
plus exact 750-PC VJPs) of 6.7 s for AlexNet, 16.5 s for CLIPAG and 73.2 s for
DINOv2 per geometry. Older complete pilots give about 34 s for standard RN50
and 29 s for robust RN50. Assigning the DINOv2 rate conservatively to RADIO,
RegNetY-640 and SigLIP2, and the RN50 rate to the remaining RN50 variants,
gives about **1.1 GPU-hour** for all 84 geometries. With 30% compute headroom,
the working estimate is **1.4 GPU-hours**, within a conservative 1–3 GPU-hour
range until all ten model pilots finish.

Greedy scheduling of those ten model-level estimates on four GPUs gives a
compute-only critical path of about **16 minutes**. Budget **20–40 minutes after
allocation** for cold checkpoint/PCA reads, result writes and model-dependent
variance. Queue time is not included. A two-hour task limit is therefore ample
and leaves room for selective MC increases or a slow unmeasured backbone.

Before launch, benchmark one representative geometry for each of the five
remaining proxy-timed models with one seed, 16 directions and four centers. Scale its
steady-state seed time by `(64/16)×(8/4)×10=80`, add the measured exact-VJP
time and 30% headroom, and record model-load/PCA-load time separately. If any
projection exceeds 45 minutes/model, profile batch size and I/O before launch.
After these five pilots, replace proxy rates with the sum of measured per-model
estimates in `eta.json`. The resumable benchmark harness and H100 submission
template are `benchmark_scale_models.py` and `benchmark_scale_h100.sbatch`.

## Quality gates before submitting the array

1. Manifest is exactly 250 pairs, 25 sites, ten models and 84 audited geometry
   IDs; one shared split hash and one seed-list hash.
2. Representative and alias PCA buffers match bitwise or within a recorded
   strict tolerance; otherwise split the key.
3. Each model reproduces its cached PCA scores within 0.5% relative L2 and
   10% of each PC training SD; also record the 99th percentile scaled error so
   a single low-variance tail PC cannot hide a broad mismatch.
4. Batched exact VJPs match serial VJPs on sampled PCs; forward local trace
   agrees with exact local trace within its MC interval.
5. Record autograd-vs-forward h sensitivity per geometry. Classify relative
   directional error as pass (≤20%), marginal (≤50%) or poor (>50%); keep exact
   VJP and h-independent finite-τ estimators even when an h-dependent estimate
   is marginal or poor.
6. `variance + drift = total` holds within float32 storage rounding; all arrays
   have expected shapes and finite values. Signed smooth U-statistics are not
   clipped.
7. A two-model end-to-end dry run produces κ sweeps from saved spectral mass
   without reopening a model.

The existing `build_scale_manifest.py` creates the two manifests and audit
summary. `run_scale_mass.py` is the model-grouped, resumable production runner;
`run_scale_array.sbatch` launches the ten model tasks and
`deep_validate_scale_mass.py` performs full archive and checksum validation.

## Production run

Production array 47119391 and targeted resumable repair jobs completed all 84
unique geometries on 2026-09-19. The final archive contains 840 seed files and
occupies 4.8 GB under the layout above. Deep-validation job 47124852 read every
archive and reported 84/84 passing geometries, no checksum, shape or finiteness
errors, and maximum relative residuals of 5.96e-8 for both `variance + drift =
total` and `odd + even = total`.

Finite-difference h diagnostics classified 75 geometries as pass and nine as
marginal. All nine marginal geometries are CLIP ResNet-50 sites, with relative
directional errors from 0.302 to 0.459; their cache reproduction errors remain
small (at most 0.194% relative L2 among those sites). Treat their h-dependent
local/smoothed estimates as diagnostics. Their exact VJP and h-independent
finite-tau response, covariance and Stein arrays remain valid.

The committed plot-ready QC snapshots are `production_status.{json,csv}` and
`production_deep_{validation.json,qc.csv}` in
`tables/nonlinear_control/scale_manifest/`. The complete κ-free spectral mass
remains in `$STORE_DIR/Projects/AccentuationPredRMT/nonlinear_control/mass_v1/`.

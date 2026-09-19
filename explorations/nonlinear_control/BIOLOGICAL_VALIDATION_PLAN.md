# Biological validation: prediction error × Jacobian geometry

Date: 2026-09-19. Status: basic variance-proxy validation completed for 250 site/model pairs.
See BIOLOGICAL_VALIDATION_RESULTS.md for results and implemented scope. Advanced corrections below remain future work.

## 1. Main question and identifiable quantities

For site a, model m, seed i and Jacobian estimator h, compute

    V_amih = MSE_test,am / n_am × sum_k q_amih,k s_am,k / (s_am,k + kappa_am)^2.

This is an empirical proxy for the variance contribution to D in
z_acc = D/N - 1. It is NOT an identified prediction of the complete ratio:
the shrinkage signal energy A = ||beta_tilde_kappa||² and teacher–student
input-gradient alignment N remain unknown. The slope predicted by the ideal
linear-path theory is b_acc = N / (A + V); thus z_acc = 1/b_acc - 1 only
under that model and for nonzero slope. Test b directly; do not invert noisy,
near-zero or negative biological slopes.

Stage 1: test whether V explains biological control beyond held-out MSE alone
and geometry T alone. Stage 2: audit the remaining terms and actual stimulus
optimization metric before attempting a calibrated absolute slope prediction.
An inverse association between V and slope is a hypothesis, not a mathematical
necessity when A and N vary across models.

No model retraining or new large Jacobian computation is needed for Stage 1.
Use actual per-site fitted alpha, not the illustrative matched-df2=375 values
used in the previous bar plots.

## 2. Sources verified on disk

Upstream biological implementation:
https://github.com/jacob-prince/parametric-neural-control/tree/jacob
Pinned commit: 9b6fd95228d1b77146f8b279c7c228914ca34bb8.
Read-only checkout for this audit: /tmp/parametric-neural-control-jacob.
Archive a pinned checkout/provenance under project bulk storage for execution;
/tmp is not a persistent dependency.

Let LAB = /n/holylabs/LABS/alvarez_lab/Lab/VVS_Accentuation.

| Input | Verified location / contents |
|---|---|
| Control recordings | LAB/Ephys_Data/vvs-accentuate_controlsessions_5monkeys_250504-250512.h5 |
| Encoding recordings | LAB/Ephys_Data/{subject}_vvs-encodingstimuli_z1_rw*.h5 |
| Session normalization and firing floor | LAB/NeuralData_raw/*_hp0_bs0_zs1_rw*_sessdata.pkl |
| Natural-image PCA + predictions + readouts | LAB/Encoding_models/{subject}/posthoc_model_predict_PCA_popul_unit/; exact paths in existing site_model_manifest.csv |
| Saved-stimulus post-hoc predictions | LAB/Encoding_models/{subject}/posthoc_model_predict/accentuated_stim_info_w_pred_resp_{subject}.pkl |
| Original RidgeCV objects with alpha_ | LAB/Encoding_models/{subject}/model_outputs_pca4all/{subject}_{model}_sweep_regressors_layers_fitmodels_RidgeCV.pth |
| Jacobian spectral mass | $STORE_DIR/Projects/AccentuationPredRMT/nonlinear_control/mass_v1/geometry/{geometry_id}/ |
| Stimuli for variant audit | LAB/Stimuli; resolve actual saved stimulus paths from prediction tables |

The full five-monkey control file was verified on the shared LAB filesystem.
The two inspected STORE roots Projects/VVS_Accentuation and
DL_Projects/VVS_Accentuation contain older/partial material; do not silently
substitute those for the verified 2025 recordings.

The site manifest exactly matches upstream MONKEY_UNITS × ten models: 250
pairs, 25 sites, five monkeys. Subject strings map to red, paul, venus, leap,
three0 via upstream MONKEY_DIRS. Sites represent different areas/animals;
region and monkey effects are largely confounded in this sample.

Inventory: 50 RidgeCV files, 17.94 GB total; 250 natural PCA caches, 1.01 GB;
control HDF5, 6.00 GB. Small file inventory is saved as
tables/nonlinear_control/biological_validation/ridge_source_inventory.csv.

## 3. Reproduce biological processing and outcomes

Use the pinned upstream functions, with a source-path adapter rather than
rewriting numerical preprocessing:

- scripts/preprocessing/run_preproc.py CONFIG and build_brain_main:
  peak-window averaging, robust outlier rejection (threshold 15), anchorDay
  per-session standardization, allday fallback below five anchor trials,
  then stimulus trial averages. Preserve exclusion/fallback counts.
- pnc/preproc/loader.py control_cloud and _measures: use the generating
  model's own saved-image prediction at its own target site; clamp predicted
  control scores to firing_floor immediately before comparison.
- Floor is min(mean_session(-mu/sigma), minimum observed calibration trial z).
- Use achieved post-hoc pred_resp, not target levels or synthesis filename
  scores. Match responses by actual stimulus identity.
- Reproduce Leap variant handling. loader.load_brain pools responses by
  trial count for sufficiently similar variants (pixel r >= .90);
  loader.load_predictions retains the canonical prediction row. Its docstring
  mentions averaging predictions, but the pinned executable code does not.
  Find/build leap_variant_similarity.json and use the code's actual behavior.
  Do not silently skip this because the file is absent.
- Report coverage per site/model/seed/level. Retain official no-exclusion
  control_cloud results, and separately reproduce permodel/drop2/intersect
  masks when testing synthesis failures or common stimulus coverage.

Primary reproducibility outcome: pooled official control_slope, OLS measured
on predicted with fitted intercept, all available accentuated stimuli.
Companion outcomes: official Pearson r, direct-prediction R², MSE_acc, intercept.

    b = Cov(p,y)/Var(p)
    MSE_acc = mean((y-p)^2)
    R²_acc = 1 - sum((y-p)^2)/sum((y-mean(y))^2)
    E_acc = MSE_acc / S_nat

Store S_nat (encoding response signal variance, noise-corrected where
estimable) and also an explicitly labeled observed-variance normalization.
Official R² is not Pearson r² and is not the R² of a refitted regression.
Generally R²_acc != 1-z_acc²: offsets, nonlinearity, seed baselines and
measurement noise invalidate that special ideal-path identity.

Mechanistic companion: within-seed slope, removing each seed's mean:

    b_within = sum_i,l (p_il-pbar_i)(y_il-ybar_i) /
               sum_i,l (p_il-pbar_i)^2.

Also save per-seed slopes and near-seed local slopes if measured seed anchors
are present. Otherwise explicitly call them within-seed sweep slopes, not
local derivatives. Analyze ascending/descending sweeps and floor-free points
separately. A pooled slope mixes natural seed-baseline prediction with control
of changes; the within-seed endpoint is closer to the pathwise theory.

Official clamped outcome is the primary replication. Mechanistic sensitivity
uses raw achieved predictions on points above the floor; don't interpret
clipped gradients as the original smooth ridge model's gradients.

## 4. Held-out error and noise alignment

Compute test MSE from encoding-phase natural stimuli, matching predictions and
recorded trial-mean responses in the SAME response units used in model fitting.
Use upstream encoding_cloud(..., split='test') for a reference, but require
an explicit valid split rather than accepting its all-stimulus fallback.

Observed first-cache flags: is_train is bool with 774 train / 195 nontrain;
is_test is object with strings 'False', 'True', '1'. Use upstream _flag_to_bool;
never cast these strings with astype(bool). Verify this for every source.
Distinguish model target_unit_resp (prediction) from measured neural y.

Held-out MSE estimates error against the observed trial mean. Its noise term
is the noise of that mean, not single-trial variance. If train/test repeat
counts and noise distributions match, MSE_test estimates E_gen + sigma_train².
Do not add a second noise term. If they differ, report the raw proxy and a
sensitivity MSE_test - v_test_mean + v_train_mean, with uncertainty. Account
for session effects separately; they are not ordinary independent trial noise.

Reproduce encoding z-score units, intercepts and day normalization. Use
shared control-session anchors to audit cross-phase gain changes; do not fit
an outcome-based rescaling on accentuated responses and then call it validation.
Audit layer selection: if the same 195 test images selected the winning layer,
this is post-selection test MSE, not an unbiased untouched test estimate.
Report that limitation or use genuinely independent available anchors/images.

## 5. Restore alpha and map to kappa

Read each subject-model RidgeCV dictionary once. Key:
((selected_layer, 'pca750'), 'RidgeCV'); choose alpha_[unit] only after verifying
channel indexing. Compare coefficients/intercept against the exported readout.
Exported torch Linear readouts alone do not retain alpha_.

Pilot: leap_250426-250501 / unit 282 / resnet50 / .layer3.Bottleneck1:
alpha_=1000, fit_intercept=True; original and exported coefficient relative
discrepancy 2.4768e-8. Loading the 476.9 MB model dictionary took 4.75 seconds.

For sklearn objective ||y-Xw||² + alpha ||w||² and theoretical
(1/n)||y-Xw||² + lambda ||w||²:

    lambda = alpha / n
    kappa - lambda = kappa/n × sum_k s_k/(s_k+kappa).

Use centered design; confirm actual n from fit metadata. The cached spectrum
uses ddof=1; record both ddof=1 and n-normalized eigenvalues, converting by
(n-1)/n for empirical Xc'Xc/n calculations. Bracket the positive stable root
and verify residual and 1-df2/n > 0. Pilot lambda=1000/774=1.2920 (before
resolving all normalization conventions), not kappa=1000.

Critical approximation: with 750 retained PCs and 774 training images, the
training spectrum is a noisy sample spectrum, not population truth, and the
PC subspace was itself learned on training data. Label the above computation
as a population-spectrum plug-in. Do not claim a second RMT transformation of
sample eigenvalues is exact. Sensitivities:

1. Primary exploratory calculation: existing 750-PC spectrum plug-in.
2. Compute empirical ridge df_hat=sum s_hat/(s_hat+lambda), and the alternative
   DE-motivated kappa=lambda/(1-df_hat/n). This is also an approximation.
3. Direct empirical resolvent sum q_k s_hat_k/(s_hat_k+lambda)^2, labeled as
   a different empirical statistic, not the same population DE trace.
4. Evaluate spectral regularization/truncation and independent natural-image
   feature covariance if available; a full held-out covariance in training-PC
   coordinates is generally not diagonal. A rotated covariance requires
   additional Jacobian cross-products or new projections.

## 6. Predictors and statistical validation

Compute T and V per site-model-seed with fixed actual kappa for exact,
smoothed tau*255={.5,2,8,16}, neighborhood and finite-noise covariance.
Do not conflate ||E J||², E||J||² and response covariance/tau².
Keep signed MC estimates and quality flags; do not log or clip negative values.
CLIP-RN50 h-dependent results remain marginal; exact and h-independent
response metrics are separate controls, not proofs of h-estimator accuracy.

Read replicate-by-PC seed archives for MC uncertainty of weighted sums;
do not sum independent per-PC SEs (PC covariance matters). Reused geometry
aliases and common RNG streams are correlated; resample seeds/probe units
jointly and retain alias mapping. Test-MSE uncertainty requires resampling
stimuli and neural trials/days separately from MC variability.

Join one row per (subject, unit, model); seed-detail table retained separately.
First plots: existing three-panel bar style with real kappa, then V bars;
paired scatter V versus official slope and within-seed slope, colored by model
and marked by monkey. Include model-mean and site-demeaned views.

Compare out-of-sample models using MSE only, T only, log(MSE)+log(T) with free
coefficients, and log(V) (equal coefficients constraint), plus kappa/df2/layer
and reliability diagnostics. For positive predictors, predeclare transformations.
Repeat exact and four smoothing settings; do not select best tau on the full
biological outcome and report that same fit as validation.

Use leave-one-monkey-out (five folds) for site generalization and
leave-one-model-out (ten folds) for model generalization; tune tau in inner
folds if selected at all. Summarize all ten, nine trained and seven conventional
models to check whether robust models or the AlexNet baseline drive results.
Use paired/hierarchical uncertainty with monkey/site and shared geometry
dependence recorded; 250 pairs or 2500 seeds are not independent samples.
Five monkeys provide weak population-level uncertainty, so emphasize effect
sizes, held-out predictions and sensitivity over nominal p-values.

## 7. What a full D/N test would additionally require

The stored q_k are diagonals of JJ' in PCA coordinates. They suffice for a
diagonal trace but not ||J' w||²: off-diagonal Gram terms matter. Do not replace
that norm by sum w_k² q_k without an additional assumption. One VJP per
site-model-seed can estimate the fitted readout-gradient energy D_hat; for
smoothed readout gradients, use a separate debiased cross-center estimate.
Subtracting V from D_hat is at most a DE diagnostic, not an unbiased A estimate.
Biological teacher-gradient alignment N remains unobserved. Inferring N from
the same control slope and then claiming to predict that slope is circular.

Audit actual optimization path before equating this with biological control:
the inspected configs use image_size=1024, model_input_size=224, random crops,
noise and Fourier decay. For an input preconditioner P the relevant gradient
metric becomes JPJ', not JJ'. Nonlinear trajectories and Gaussian smoothing
at four RGB noise scales do not automatically reproduce that optimization.
If needed, add algorithm-matched directional/augmentation estimates as a
follow-up with timing pilots, using held-out biological data for evaluation.

## 8. Execution and deliverables

All large data/caches/logs go under
$STORE_DIR/Projects/AccentuationPredRMT/nonlinear_control/biological_validation_v1/.
Keep source, provenance, compact joined tables and final figures in this repo.

1. Inventory + normalization audit; freeze upstream hash and input paths.
2. Port source paths and run official preprocessing; reproduce control_table;
   verify variant map, observed coverage, floors and same-key readouts.
3. Extract alpha/encoding-MSE; compute kappa and weighted geometry predictors.
4. Join the 250 pairs; compute within-seed/local outcomes and uncertainty.
5. Run prespecified comparison/CV; make biological validation plots and report.

CPU-first. Existing GPU spectral outputs are sufficient. Schedule at most four
CPU workers to avoid simultaneous 18 GB RidgeCV reads; cache small alpha and
readout summaries immediately. Five monkey preprocessing tasks can run as a
Slurm array with concurrency 2, 4–8 CPUs/task and 16–32 GB RAM, initially one
monkey as a timing pilot. Read only selected unit/window HDF5 slices.

Measured fit-object pilot projects ~4 minutes of serial loading for 50 files
if the representative latency holds; allow 5–15 minutes for extraction and
verification. Natural caches are ~1 GB and existing summaries are cheap.
Budget provisionally 30–60 minutes CPU wall time after allocation for source
adaptation/preprocessing/QC/tables, excluding queue and missing variant audits;
replace this budget after a one-monkey pilot. Full replicate uncertainty and
CV get separate small timing pilots. New readout-VJP / optimization-metric
GPU jobs are Stage 2 only, with no unmeasured GPU ETA promised.

Observable logs: logs/prepare_<job>.out, logs/brain_<job>_<monkey>.out,
logs/predictors_<job>.out, plus a stage_status.json completion record.
Outputs: source_inventory, normalization_audit, alpha_readout_audit,
biological_control_scores, heldout_prediction_error, kappa_audit,
geometry_predictors_by_seed, joined_site_model, CV_predictions, bootstrap
summaries and PNG/PDF figures. Every table includes source/version/unit/QC
metadata. No biological association claim before these checks pass.

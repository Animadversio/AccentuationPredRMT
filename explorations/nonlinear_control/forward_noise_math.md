# Forward-only noise estimators: targets and derivation

A complete Chinese exposition of the common formalism and plot interpretation
is available in `FORWARD_NOISE_FORMALISM_ZH.md`.

Work in the same post-resize RGB coordinates and fixed training PCA basis as
the exact-Jacobian analysis. Write f_k(x)=u_k^T φ(x), z,v~N(0,I_d) independently.
Noise scale τ is the **per RGB component** standard deviation, not total L2
radius; typical L2 size is τ sqrt(d), d=3×224×224. Do not clip perturbed pixels:
clipping changes the convolution, gradient and perturbation covariance.

## The smoothed Jacobian is identifiable by forward evaluations

Gaussian integration by parts gives, under the usual integrability conditions,

J_τ(x) = ∇ E_z φ(x+τz) = E_z J(x+τz)
       = E_z[(φ(x+τz)−φ(x))z^T]/τ.

This is also the population slope of local linear regression on isotropic
Gaussian perturbations: E[δφ δx^T] E[δx δx^T]^{-1}=J_τ. It recovers F exactly
for a linear feature map Fx. Gaussian smoothing makes it meaningful even for
piecewise differentiable networks.

This population identification does not mean a finite perturbation batch can
uniquely recover an arbitrary dense F with far fewer probes than input
dimensions. The experiment estimates **PC gradient energies and weighted
traces**, not a dense 750×150528 Jacobian. Those quadratic summaries are much
cheaper to estimate with random directions than the whole matrix.

However, squaring a finite-M Monte Carlo gradient estimate introduces a positive
variance term. In d=150528 dimensions the direct Stein outer-product estimator
can be very noisy. We instead estimate random directional derivatives using
only forward passes, and remove the inner-MC square bias with a U-statistic.

For independent directions v_r and iid centers z_{r,l}, define

d_{r,l,k} = [f_k(x+τz_{r,l}+hv_r)−f_k(x+τz_{r,l}−hv_r)]/(2h).

Conditional on v, the inner samples l must be independent (do not count z and
−z as independent centers). Then

U_{r,k} = [(Σ_l d_{r,l,k})²−Σ_l d_{r,l,k}²]/[L(L−1)]

has expectation (E_z d_{r,l,k})². Averaging U over independent r therefore
estimates, exactly at finite h,

E_v{[f_{k,τ}(x+hv)−f_{k,τ}(x−hv)]/(2h)}².

As h→0, this tends to q_smooth,k=||∇f_{k,τ}(x)||². This is MC-unbiased for the
finite-h target, not an exactly unbiased estimator of the derivative at h>0.
We pilot h against autograd directional derivatives; changing h is a numerical
approximation check, distinct from sweeping the smoothing scale τ.

Meanwhile mean_{r,l} d² tends to
q_neighborhood,k=E_z||∇f_k(x+τz)||².

q_neighborhood−q_smooth = E||∇f_k−E∇f_k||² ≥ 0, the variation in gradients
across the neighborhood. This separates smoothing by cancellation from low
gradient energy everywhere. The U-statistic can be negative in finite samples;
keep signed estimates and MC standard errors, never truncate at zero. An
unresolved signed estimate is not evidence of negative physical energy.

## Direct finite-step sensitivity is a different useful quantity

a_k(v)=[f_k(x+τv)−f_k(x−τv)]/(2τ)
b_k(v)=[f_k(x+τv)+f_k(x−τv)−2f_k(x)]/(2τ).

Q_odd,k=E a², Q_even,k=E b², and
Q_step,k=E[(f_k(x+τv)−f_k(x))/τ]²=Q_odd,k+Q_even,k.

Also distinguish Q_variance,k=Var[f_k(x+τz)]/τ² from Q_step. Their difference
is squared mean response drift, (E f_k(x+τz)−f_k(x))²/τ². Using the antithetic
pair mean b above, estimate drift by an off-diagonal pair U-statistic over
independent directions, and subtract it from Q_step. Treat each plus/minus
pair as one sampling unit; compute MC errors with jackknife pseudovalues.

An exact least-squares decomposition gives

Cov[φ(x+τz)]/τ² = J_τ J_τ^T + Cov[nonlinear regression residual]/τ².

Thus noise covariance includes both the smoothed linear sensitivity and
nonlinear residual variation. In particular Q_variance,k ≥ q_smooth,k.
This is often a more informative forward-only proxy for JJ^T than uncentered
squared response differences, which additionally include mean drift.

These describe finite-scale response change, not J_τ. The symmetric difference
alone misses even-order effects. For f(x)=x² in one dimension:

- q_smooth=4x²,
- q_neighborhood=4x²+4τ²,
- Q_odd=4x²,
- Q_step=4x²+3τ².

For this quadratic example, Q_variance=4x²+2τ² and drift power=τ².

For f(x)=x³, E f(x+τz)=x³+3xτ², so q_smooth=(3x²+3τ²)². These polynomial
examples and the linear limit are implementation checks before neural MC.

## Spectral aggregation and uncertainty

For any measured power q, T_q(κ)=Σ_k q_k s_k/(s_k+κ)², with the **original
training spectrum held fixed**. This isolates changes in input geometry.
It does not assert that smoothing leaves the feature population spectrum
unchanged; a new regression model on smoothed features would need its own C.

Use the previously matched df₂/750=0.5 to compare models. Compute trace per
outer direction first, then MC standard errors across directions, preserving
cross-PC dependence. Aggregate across the ten fixed seeds with independent
random streams; report MC error separately from seed-to-seed variability.
Common directions/noise across scales and models improve paired comparisons.

Initial design: four τ values 0.5,2,8,16 /255; 64 Gaussian directions; eight
independent centers per direction. Cache per-direction powers and per-PC means
and errors under STORE_DIR. Benchmark one seed and a few directions before
scaling. If smooth-trace signal is not resolved relative to MC error, increase
independent centers/directions selectively, or explicitly report unresolved
estimates rather than substituting neighborhood energy for smoothed energy.

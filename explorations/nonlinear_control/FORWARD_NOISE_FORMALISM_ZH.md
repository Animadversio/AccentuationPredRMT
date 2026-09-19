# Forward noise geometry：几个核心量的统一 formalism

本文统一定义实验图中的 **Local Jacobian**、**Smoothed Jacobian**、
**Mean local Jacobian energy**、**Noise covariance** 和
**Total response change**。这些量都描述输入扰动如何传入 feature space，
但它们回答的问题不同，不能互换使用。

## 1. 基本记号与 spectral trace

设 nonlinear feature map 为

\[
\phi:\mathbb R^d\rightarrow\mathbb R^p,
\qquad
J(x)=\frac{\partial\phi(x)}{\partial x}\in\mathbb R^{p\times d}.
\]

训练分布上的 centered feature covariance 记为

\[
C=\operatorname{Cov}[\phi(X)]
  =U\operatorname{diag}(s_1,\ldots,s_p)U^\top,
\qquad Cu_k=s_ku_k.
\]

第 \(k\) 个 feature-PC score 是

\[
f_k(x)=u_k^\top\phi(x),
\qquad
\nabla_x f_k(x)=J(x)^\top u_k.
\]

原始理论中的局部 input-to-feature metric 是

\[
M_{\rm local}(x)=J(x)J(x)^\top.
\]

在 PC \(k\) 上对应的 gradient power 为

\[
q_{k,{\rm local}}(x)
=u_k^\top M_{\rm local}(x)u_k
=\|J(x)^\top u_k\|^2
=\|\nabla_x f_k(x)\|^2.
\]

对于任意 feature-space PSD metric \(M\)，定义与 control-error 方差项相同的
spectral aggregation：

\[
\mathcal T_\kappa(M)
=\operatorname{Tr}\!\left[M(C+\kappa I)^{-2}C\right]
=\sum_{k=1}^p
  \frac{s_k}{(s_k+\kappa)^2}
  u_k^\top M u_k.
\]

令

\[
h_k=\frac{s_k}{s_k+\kappa},
\]

则也可以写成

\[
\mathcal T_\kappa(M)
=\sum_k \frac{u_k^\top M u_k}{s_k}h_k^2.
\]

实验在每个模型内固定原始 training spectrum \(C\)，并选择 \(\kappa\) 使

\[
\operatorname{df}_2(\kappa)=\sum_kh_k^2=375.
\]

所以图中的差别来自 input geometry \(M\)，而不是重新拟合一个 smoothed-feature
covariance。十张 seed 的结果是在每张图像上计算 \(\mathcal T_\kappa\) 后再平均。

## 2. Gaussian smoothing 与 Smoothed Jacobian

令

\[
z\sim\mathcal N(0,I_d),
\qquad
x_\tau=x+\tau z,
\]

其中 \(\tau\) 是每个 RGB component 的标准差。定义 Gaussian-smoothed feature map

\[
\phi_\tau(x)=\mathbb E_z[\phi(x+\tau z)].
\]

其 Jacobian 为

\[
J_\tau(x)
=\nabla_x\phi_\tau(x)
=\mathbb E_z[J(x+\tau z)].
\]

在适当的可积条件下，Gaussian integration by parts（Stein identity）给出

\[
\boxed{
J_\tau(x)
=\frac1\tau
 \mathbb E_z\!\left[
   \bigl(\phi(x+\tau z)-\phi(x)\bigr)z^\top
 \right].
}
\]

所以 \(J_\tau\) 原则上可以只用 forward evaluations 估计。它同时也是用
Gaussian input perturbations 对 feature response 做 population local linear
regression 时的 slope。

图中的 **Smoothed Jacobian** metric 是

\[
M_{\rm smooth}(x,\tau)=J_\tau(x)J_\tau(x)^\top.
\]

PC-wise power 为

\[
q_{k,{\rm smooth}}(x,\tau)
=\|J_\tau(x)^\top u_k\|^2
=\left\|\mathbb E_z\nabla f_k(x+\tau z)\right\|^2.
\]

它测量邻域内方向一致、经过平均后仍然存在的 gradient component。如果邻域内
gradients 很大但方向快速改变并彼此抵消，\(q_{k,{\rm smooth}}\) 会很小。
当 \(\tau\to0\) 时，在通常的连续性条件下

\[
J_\tau(x)\to J(x),
\qquad
M_{\rm smooth}(x,\tau)\to M_{\rm local}(x).
\]

对于线性 feature map \(\phi(x)=Fx\)，任意 \(\tau\) 下都有

\[
J_\tau(x)=F.
\]

## 3. Mean local Jacobian energy

Smoothed Jacobian 是“先平均 Jacobian，再平方”。另一个不同的量是“先平方，
再在邻域内平均”：

\[
M_{\rm neighborhood}(x,\tau)
=\mathbb E_z\!\left[
J(x+\tau z)J(x+\tau z)^\top
\right].
\]

其 PC-wise power 是

\[
q_{k,{\rm neighborhood}}
=\mathbb E_z
 \left\|\nabla f_k(x+\tau z)\right\|^2.
\]

两者的精确差为

\[
\begin{aligned}
M_{\rm neighborhood}-M_{\rm smooth}
&=\mathbb E_z\!\left[
  \bigl(J-\mathbb EJ\bigr)
  \bigl(J-\mathbb EJ\bigr)^\top
\right]\\
&\succeq0.
\end{aligned}
\]

因此

\[
\boxed{
M_{\rm smooth}\preceq M_{\rm neighborhood},
\qquad
\mathcal T_\kappa(M_{\rm smooth})
\leq
\mathcal T_\kappa(M_{\rm neighborhood}).
}
\]

两者的差刻画 Jacobian 在 Gaussian neighborhood 内的变化：

- 两者都小：整个邻域的 feature sensitivity 都小；
- neighborhood 大而 smooth 小：局部 gradients 很大，但方向不稳定、平均后抵消；
- 两者接近：gradient 在该尺度下较稳定。

## 4. Noise covariance

令 noisy feature response 为

\[
Y_\tau=\phi(x+\tau z),
\qquad
\mu_\tau=\mathbb E_z[Y_\tau].
\]

图中的 **Noise covariance** 定义为

\[
\boxed{
M_{\rm cov}(x,\tau)
=\frac1{\tau^2}\operatorname{Cov}_z[Y_\tau].
}
\]

PC-wise：

\[
q_{k,{\rm cov}}(x,\tau)
=\frac1{\tau^2}
  \operatorname{Var}_z[f_k(x+\tau z)].
\]

它测量尺度 \(\tau\) 下的随机 input noise 实际产生多少随机 feature variation。
它不是 smoothed Jacobian Gram matrix，但两者有一个精确的 population
least-squares decomposition。

把 centered noisy response 分解为

\[
Y_\tau-\mu_\tau
=\tau J_\tau z+r_\tau,
\qquad
\mathbb E_z[r_\tau z^\top]=0,
\]

其中 \(\tau J_\tau z\) 是从 \(z\) 线性预测 feature response 的最佳 population
linear predictor，\(r_\tau\) 是非线性 residual。于是

\[
\boxed{
M_{\rm cov}
=J_\tau J_\tau^\top
 +\frac1{\tau^2}\operatorname{Cov}[r_\tau].
}
\]

因此

\[
\boxed{
M_{\rm cov}\succeq M_{\rm smooth},
\qquad
\mathcal T_\kappa(M_{\rm cov})
\geq
\mathcal T_\kappa(M_{\rm smooth}).
}
\]

差值

\[
M_{\rm cov}-M_{\rm smooth}
=\frac1{\tau^2}\operatorname{Cov}[r_\tau]
\]

表示 finite-scale feature variation 中无法被 smoothed linear slope 解释的部分。
如果 \(\phi(x)=Fx\)，则 \(r_\tau=0\)，从而

\[
M_{\rm cov}=FF^\top=M_{\rm smooth}.
\]

## 5. Total response change 与 mean drift

图中的 **Total response change** 定义为 noisy feature 相对原点 feature 的总平方变化：

\[
\boxed{
M_{\rm total}(x,\tau)
=\frac1{\tau^2}
 \mathbb E_z\!\left[
  \bigl(Y_\tau-\phi(x)\bigr)
  \bigl(Y_\tau-\phi(x)\bigr)^\top
 \right].
}
\]

PC-wise：

\[
q_{k,{\rm total}}(x,\tau)
=\frac1{\tau^2}
 \mathbb E_z\!\left[
  \bigl(f_k(x+\tau z)-f_k(x)\bigr)^2
 \right].
\]

定义 normalized mean drift

\[
b_\tau(x)=\frac{\mu_\tau-\phi(x)}{\tau}.
\]

由 bias--variance decomposition，

\[
\boxed{
M_{\rm total}=M_{\rm cov}+b_\tau b_\tau^\top.
}
\]

因此

\[
\boxed{
M_{\rm total}\succeq M_{\rm cov}\succeq M_{\rm smooth}.
}
\]

对应 trace 满足

\[
\mathcal T_\kappa(M_{\rm total})
=\mathcal T_\kappa(M_{\rm cov})
 +\mathcal T_\kappa(b_\tau b_\tau^\top).
\]

所以 Total response change 同时包含：

1. noise 造成的随机 feature variation；
2. Gaussian smoothing 造成的系统性 mean-response drift。

这解释了为什么 total response change 可能随 \(\tau\) 快速增大，即使 smoothed
Jacobian 或 noise covariance 没有增大。未经中心化的 squared feature difference
不能直接解释为 Jacobian sensitivity。

## 6. 几个 PSD ordering，以及不存在的 ordering

对任意固定 \(x,\tau\)，有两条普适关系：

\[
\boxed{
M_{\rm smooth}\preceq M_{\rm neighborhood}
}
\]

和

\[
\boxed{
M_{\rm smooth}\preceq M_{\rm cov}\preceq M_{\rm total}.
}
\]

因为 spectral weight

\[
(C+\kappa I)^{-2}C\succeq0,
\]

这些 PSD ordering 也传递到对应的 \(\mathcal T_\kappa\)。但是
\(M_{\rm neighborhood}\) 与 \(M_{\rm cov}\) 之间一般没有固定 ordering：

- neighborhood average 看的是各个 noisy center 上的 infinitesimal Jacobian energy；
- noise covariance 看的是一个 finite Gaussian cloud 的 feature variation。

二者在 nonlinear map 上可以有不同的尺度依赖。

## 7. 图中曲线应如何阅读

| 曲线 | 矩阵 | 回答的问题 |
|---|---|---|
| Exact local Jacobian | \(J(x)J(x)^\top\) | 单点、无限小扰动的 sensitivity 是多少？ |
| Smoothed Jacobian | \(J_\tau J_\tau^\top\) | 邻域平均后仍保持一致的线性方向有多强？ |
| Mean local energy | \(\mathbb E[JJ^\top]\) | 邻域各处的局部 gradients 平均有多强？ |
| Noise covariance | \(\operatorname{Cov}[\phi(x+\tau z)]/\tau^2\) | finite noise 造成多少随机 feature variation？ |
| Total response change | \(\mathbb E[(\phi(x+\tau z)-\phi(x))(\cdot)^\top]/\tau^2\) | noisy feature cloud 相对原点总共移动多少？ |

如果研究原理论中 \(JJ^\top\) 的 finite-scale extension，最直接的对象是

\[
J_\tau J_\tau^\top.
\]

如果研究实际有限扰动下 feature response 的可靠变化，Noise covariance 可能更有
行为意义；如果 mean response drift 本身也是 control 的一部分，则还需要 Total
response change。三者对应不同的问题，不应统一称为“smoothed Jacobian”。

## 8. Forward-only Monte Carlo estimator

实验没有恢复完整的 \(750\times150528\) dense Jacobian，而是估计 PC-wise
quadratic summaries。令 \(v_r\sim\mathcal N(0,I_d)\) 为 independent outer
directions，并对每个 \(v_r\) 采样 independent smoothing centers
\(z_{r,1},\ldots,z_{r,L}\)。用 forward central difference

\[
d_{r,l,k}
=\frac{
f_k(x+\tau z_{r,l}+hv_r)
-f_k(x+\tau z_{r,l}-hv_r)
}{2h}.
\]

直接计算 \((L^{-1}\sum_ld_{r,l,k})^2\) 会把 inner Monte Carlo variance 当作
gradient power，产生正偏。实验使用 U-statistic

\[
U_{r,k}
=\frac{
 (\sum_ld_{r,l,k})^2-\sum_ld_{r,l,k}^2
}{L(L-1)}.
\]

在固定 \(v_r\) 条件下，

\[
\mathbb E[U_{r,k}\mid v_r]
=\left(\mathbb E_zd_{r,l,k}\right)^2.
\]

再对 isotropic \(v_r\) 平均，并令 \(h\to0\)，得到

\[
\mathbb E_v U_{r,k}
\longrightarrow
\|J_\tau(x)^\top u_k\|^2.
\]

有限 \(h\) 下，这是对应 central-difference target 的 MC-unbiased estimator；它并非
对无限小 derivative 完全无偏。因此实验另外用同方向 autograd 和 doubled-\(h\)
sensitivity check 检验 finite-difference error。

U-statistic 在有限样本中可以为负；这只表示 MC noise，不代表真实 energy 为负。
分析中保留 signed estimates 和 MC standard errors，不把负值裁剪为零。

## 9. 坐标和边界

当前实验中的 \(x\) 是各模型 spatial preprocessing 后的 \(224\times224\) RGB
坐标，noise 在 RGB \([0,1]\) 单位中定义，再通过各模型 normalization 输入网络。
Gaussian noise 没有 clipping。因此测量的是无界 Gaussian convolution geometry，
不是 bounded pixel ball、原始 \(425\times425\) 图像坐标或 natural-image manifold。

此外，本文只推广了原 control-error expression 中的 geometry/variance trace。
完整 \(D/N\) 仍然需要 signal term、teacher-gradient alignment、readout estimation
covariance，以及 finite-step trajectory curvature 的验证。

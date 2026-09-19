# 四模型的 forward-noise / smoothed-Jacobian 分析

这些量的统一中文定义、PSD decomposition、spectral trace 和图形解释见
`FORWARD_NOISE_FORMALISM_ZH.md`。

## 先区分估计对象

令 z~N(0,I)，φ_τ(x)=Eφ(x+τz)。Gaussian integration by parts 给出

\[
J_\tau(x)=\nabla\phi_\tau(x)=\mathbb E J(x+\tau z)
=\frac1\tau\mathbb E[(\phi(x+\tau z)-\phi(x))z^\top].
\]

所以只用 forward pass 可以估计 smoothed Jacobian；它也是对 Gaussian 输入扰动做
局部线性回归时的 population slope。对线性 φ(x)=Fx，它就是 F。

不过，以下量不能互换（fₖ=uₖᵀφ）：

| 名称 | 每个 PC 的量 | 含义 |
|---|---|---|
| Smoothed Jacobian | ‖E∇fₖ(x+τz)‖² | 邻域平均梯度的能量，方向会相互抵消 |
| Mean local energy | E‖∇fₖ(x+τz)‖² | 邻域中各点梯度能量的平均 |
| Noise covariance | Var[fₖ(x+τz)]/τ² | 有限尺度随机 response variation |
| Total response change | E[(fₖ(x+τz)−fₖ(x))²]/τ² | 随机变化加上平均响应漂移 |

局部线性回归的正交残差分解进一步给出

\[
\frac{\operatorname{Cov}[\phi(x+\tau z)]}{\tau^2}
=J_\tau J_\tau^\top+\frac{\operatorname{Cov}[r]}{\tau^2}.
\]

因此直接对 noisy feature cloud 算 covariance，会包含额外的非线性残差，通常不是
smoothed-Jacobian Gram matrix。未去均值的 response difference 还会混入 mean drift。

## 估计方法

对每个 seed、每个 τ，用 64 个 independent Gaussian directions v。每个 v 再采样
8 个 independent smoothing centers z，计算 forward central difference

\[
d_l=\frac{f_k(x+\tau z_l+hv)-f_k(x+\tau z_l-hv)}{2h}.
\]

用 \([(\sum_l d_l)^2-\sum_l d_l^2]/[L(L-1)]\) 而不是 (mean d)²，去掉 inner-MC
variance 造成的正偏。再平均 v，得到 smoothed gradient power 的有限 h 近似。
全矩阵 J 没有显式恢复；估计的是 750 个 PC 的 gradient powers 和加权 trace。

同一批正负方向扰动还得到 total change、odd/even response change、noise covariance
和 squared mean drift。Covariance/mean-drift 的误差用 independent antithetic pairs
上的 jackknife，不能把正负点当作独立样本。

先通过线性、quadratic/cubic 的解析验证和 1-seed pilot，再运行四模型全部 10 seeds。
网络 autograd 只用于有限差分校验和已有 exact baseline；noise estimates 本身只用 forward。

## 设置

- 沿用相同 10 张非训练 seed PNG，以及每个模型原来的 training PCA750。
- 固定原始训练 spectrum，固定各模型此前 df₂=375 对应的 κ。
- 对所有 geometry estimates 用同一权重 sₖ/(sₖ+κ)² 汇总。
- τ 为每个 RGB 分量的标准差：0.5、2、8、16 /255。
- 在原生 preprocessing 后的 224×224 RGB 坐标加噪声，除以各模型 normalization std
  后输入网络；Gaussian noise 不裁剪、不 clamp。
- 对应典型 L2 noise norm 约为 0.76、3.04、12.17、24.34。16/255 时约 3.8–4.1%
  的 noisy RGB 分量落在 [0,1] 外，这仍是无界 Gaussian smoothing，不是 bounded-image
  perturbation 或 natural-image-manifold smoothing。
- v/z 在模型和 noise level 之间共享随机种子；不同 seed image 使用独立随机流。

## 结果：平滑以后，原本的局部差距显著缩小

下表为 smoothed T / exact local T：

| 模型 | 0.5/255 | 2/255 | 8/255 | 16/255 |
|---|---:|---:|---:|---:|
| Standard RN50 | 44.7% | 17.8% | 3.59% | 1.21% |
| Robust RN50 | 97.2% | 91.8% | 70.8% | 50.9% |
| CLIPAG | 100.2% | 100.0% | 97.0% | 93.9% |
| DINOv2 | 68.0% | 17.6% | 2.40% | 0.79% |

CLIPAG 接近 100% 的小幅上下浮动处于 MC 误差范围内。

16/255 时的绝对 smoothed T（±1 MC SE）：

| 模型 | Exact local T | Smoothed T |
|---|---:|---:|
| Standard RN50 | 21364.04 | 258.66 ± 3.54 |
| Robust RN50 | 37.79 | 19.22 ± 0.19 |
| CLIPAG | 20.05 | 18.82 ± 0.28 |
| DINOv2 | 11928.31 | 94.17 ± 0.98 |

RN50 standard/robust 的 trace ratio 从约 565 缩至 **13.46**；DINO/CLIPAG 从约 595
缩至 **5.00**。与 robust RN50、CLIPAG 相比，standard RN50 和 DINO 的大量局部梯度
能量在 Gaussian neighborhood averaging 后消失。

这不全是平均梯度抵消：mean local energy 也会改变。16/255 时 standard RN50 的
mean local T 为 3422.21，而 smoothed T 为 258.66；DINO 对应 684.10 和 94.17。
Robust RN50 则是 43.58 和 19.22，CLIPAG 为 24.17 和 18.82。
两者差值刻画邻域内梯度的变化。

## 总 response change 讲述的是另一件事

16/255 时：

| 模型 | Noise covariance T | Total response-change T |
|---|---:|---:|
| Standard RN50 | 769.54 | 9203.86 |
| Robust RN50 | 27.54 | 2405.01 |
| CLIPAG | 21.35 | 1051.27 |
| DINOv2 | 246.99 | 1360.14 |

Robust RN50 和 CLIPAG 的 total-change 项增长很大，但大部分是 mean drift；随机
response covariance 没有对应的巨大增长。未经去均值的 squared differences 会给出
很不同的模型排序。因此，研究 finite-scale sensitivity 时至少应同时报告 noise
covariance 和 mean drift，不应都命名为 smoothed Jacobian。

## 数值与统计检查

- 6 项数学/实现测试通过，包括 Gaussian polynomial quadrature、U-statistic、
  antithetic covariance debiasing、线性极限和 RGB chain rule。
- 同方向 autograd 对照选择 h：standard RN50 3e-6，robust RN50 3e-5，CLIPAG 3e-4，
  DINO 3e-5。对应 pilot directional relative errors 约 1.88%、1.74%、0.66%、0.19%。
- τ=0 的 forward trace / exact trace：0.9996、0.9941、1.0020、0.9671；全部在约
  ±2 MC SE 内。DINO 的 baseline Monte Carlo 波动较大，不能将 3.3% 差别都归因于 bias。
- 各 smoothed trace 的信噪比至少约 54（mean/MC_SE）；未因负值而裁剪任何 U-statistic。
- 在第 1 张 seed、最大噪声 16/255 上，使用完全相同的 v/z 将 h 加倍：四模型的
  smoothed T 改变分别约 +0.207%、+0.010%、−0.002%、−0.072%；mean-local T 的改变
  也都小于 0.08%。这支持观察到的数十倍下降并非 finite-difference step artifact，
  但不是每个 seed/尺度上的严格误差界。详情见 `h_sensitivity.csv`。
- 误差条为固定 10 seeds 条件下的 ±2 MC SE，保留 PC 间协方差；不是 image-population CI，
  也不包含 training-spectrum estimation 或 finite-h approximation 的误差。
- Main run 在 existing H100 job 中完成；各模型循环约几十秒，DINO 约 68 秒。
  大型中间 arrays、每个 seed 的 inner derivatives、日志都在 STORE_DIR 中。

## 意义和边界

这个实验支持把 **noise scale τ** 作为 control geometry 的一个独立维度：原始
Jacobian 描述 infinitesimal geometry，而 smoothed Jacobian 描述较稳定的邻域线性
方向。Forward cloud covariance 还保留了这些平均方向无法解释的非线性变化。
两种 forward-only 指标都有用，但对应不同问题，不能相互替代。

这里没有重算 smoothed features 的训练 covariance，也没有验证完整 D/N 或 finite-step
targeted accentuation。所有 trace 都使用原始 spectrum，目的是单独比较 geometry。
τ 越大不必然越有意义：高维白噪声会离开自然图像分布。下一步可以让扰动 covariance
匹配实际 accentuation 更新或图像频率结构，再检验 teacher alignment 与实际响应改变。

## 文件与复现

- 推导：`explorations/nonlinear_control/forward_noise_math.md`。
- 运行：`run_forward_noise.py`；重画：`plot_forward_noise.py`。
- 小数表与 PC summaries：`tables/nonlinear_control/forward_noise/`。
- PNG/PDF：`figures/nonlinear_control/forward_noise/noise_geometry.*` 和 `model_comparison.*`。
- 原始缓存：`$STORE_DIR/Projects/AccentuationPredRMT/nonlinear_control/forward_noise_v1/`。

```bash
python explorations/nonlinear_control/plot_forward_noise.py \
  --input "$STORE_DIR/Projects/AccentuationPredRMT/nonlinear_control/forward_noise_v1" \
  --tables tables/nonlinear_control/forward_noise \
  --figures figures/nonlinear_control/forward_noise
```

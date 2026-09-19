# 第一轮：standard RN50 vs robust RN50

两者使用相同 `.layer4.Bottleneck1`，相同 774 张 training images，分别训练的
unwhitened PCA750，以及实验 YAML 中相同的 10 张 seed PNG。全部 969 行图像路径和
train/test 标记已逐行核对一致；这 10 张 seed 都是非训练图像。Jacobian 坐标统一为
resize 后 224×224 的 RGB [0,1]，包含 normalization 的导数。

复用已有 Nanoclaw job 47072283 的 H100 80GB；16-PC batched VJP 最快，完整
10×750 图像循环约 13–14 秒/模型（不含模型加载）。Standard RN50 在 warmup 后，
逐 PC 为 2.20 ms/PC，16-PC block 为 1.65 ms/PC；峰值分配显存约 0.74 GB。
分块结果与逐 PC 对照的最大相对差异为 2.4e-7 以下。

## 从 trace 到一个更好解释的量

令 hₖ=sₖ/(sₖ+κ)，qₖ=mean_seed ‖∇(uₖᵀφ)‖²，则

\[
T(\kappa)=\sum_k\frac{q_k s_k}{(s_k+\kappa)^2}
         =\sum_k\underbrace{\frac{q_k}{s_k}}_{\text{unit-variance PC 的梯度能量}}h_k^2.
\]

因此 qₖ/sₖ 是非常直接的探索对象：把一个 PC 的自然图像响应 variance 归一化以后，
它对像素扰动有多敏感？这也避免把简单的 feature amplitude scaling 当作 robustness。

两模型比较时匹配 df₂=Σhₖ²，允许各自 κ 不同。于是 T/df₂ 是以 hₖ² 加权的
qₖ/sₖ 平均，匹配 df₂ 的 T 比较对整体 feature rescaling 不变。它控制了 spectral
shrinkage 的总体强度，但尚未匹配真实 generalization error 或 teacher alignment。

## 数值结果

在 df₂/750=0.5 时：

| 指标 | Standard RN50 | Robust RN50 |
|---|---:|---:|
| κ | 5.6945 | 7.6157 |
| seed-mean T | 21364.0 | 37.7867 |
| T/df₂ | 56.9708 | 0.100764 |
| PC501–750 对 T 的贡献 | 26.58% | 26.29% |

Standard/robust 的 **mean trace 比值约 565.4**。每张 seed 单独比较也都更高，
paired ratio 范围约 130–3423；seed 7 对 standard 的均值影响尤其大。因此需要同时
看逐 seed 曲线，不能只报均值。这里只测得几何项，不能把这个倍数直接称为 control
error 的倍数。

两者归一化 covariance spectrum 的形状接近，但 qₖ/sₖ 在整个 PC 范围内差异很大。
这支持下一步检验：control 的差异是否主要由 covariance 未描述的 Jacobian geometry
解释，而不仅是 feature spectrum。另一个观察是，在匹配 df₂ 后，tail 的相对贡献相近，
而整体 trace 差异巨大；值得区分“整体梯度敏感度”与“敏感度在 PC 间如何分配”。

κ→0 的 retained trace 为 standard 45610.9，robust 84.2544。Robust 的最后 250 PCs
只占 retained variance 的 8.53%，却占这个 trace 的 43.39%；standard 对应的 trace
tail fraction 为 40.19%。κ=0 在这里仅表示有限正谱的代数极限，并不是已经校准好的
ridgeless DE operating point。

## 已验证与尚未验证

- 验证了线性极限、RGB chain rule、batched/serial VJP 一致性和 feature-scale invariance。
- 原缓存 score 重现按 feature SD 和整体范数校验；standard 的相对 L2 误差约 0.085%，
  最大坐标误差为该 PC training SD 的 0.208%。GPU TF32 已关闭。
- population spectrum 仍是 training empirical plug-in；750-PC 以外的 trace 未估计。
- 完整 D/N 还需要 signal gradient energy、teacher alignment，以及 readout estimation
  covariance 的 DE 检验；非线性 finite-step trajectory 还涉及 curvature。
- 十张 seed 是实验固定集合，这里的均值不是随机抽样得到的 population guarantee。
- DINO 的同集计算已缓存，按照用户要求留作下一步，不用于当前 RN50 对照结论。

## 文件

- 对比图：`figures/nonlinear_control/rn50_comparison/rn50_comparison.png` 和 `.pdf`。
- 匹配 df₂ 数表：`tables/nonlinear_control/rn50_matched_df2.csv`。
- 每模型 plot-ready 表：`tables/nonlinear_control/{standard,robust}_resnet50_seed10/`。
- 原始 per-image powers 与日志：
  `$STORE_DIR/Projects/AccentuationPredRMT/nonlinear_control/{standard,robust}_resnet50_seed10/`。
- `run_pilot.py` 负责加载、校验和 VJP；`compare_resnets.py` 从缓存重画比较图。

下一步优先在这对 RN50 上，用已知 synthetic teacher 和重复 ridge fits 验证
readout covariance → gradient energy 的统计联系，再研究整个 D/N；DINO 扩展在其后。

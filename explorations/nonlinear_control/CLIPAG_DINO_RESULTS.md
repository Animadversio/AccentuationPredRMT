# CLIPAG 与 DINOv2：相同 10 张 seed 的 PCA-Jacobian geometry

## 设置与可比范围

- CLIPAG ViT-B/32：`.transformer.resblocks.ResidualAttentionBlock8`。
- DINOv2 ViT-B/14-reg：`.blocks.NestedTensorBlock10`。
- 使用原实验 channel 2 所选的 feature layers 与各自 exported PCA750。
- 已逐行确认两份缓存的 969 张图像路径和 train/test 标记相同；谱由 774 张训练
  图像估计。10 张 seed 的原始 PNG、顺序与前一轮 RN50 相同，且全是非训练图像。
- 两者均计算 post-resize 224×224 RGB [0,1] 坐标中的梯度，分别除以其 normalization
  std。CLIPAG std=(0.26862954,0.26130258,0.27577711)，DINO std=(0.229,0.224,0.225)。
- 保留各模型原生 preprocessing；CLIPAG 使用 OpenCLIP 的 bicubic resize/center crop，
  DINO 使用 tensor Resize。原始 seed 均为 425×425，所以没有不同宽高比造成的
  crop-content mismatch，但插值路径仍不同。导数不包含 resize 本身。
- 这是跨架构、patch size、训练目标和 layer 的比较；不能当作仅改变 adversarial
  training 的消融实验。

## 测量与主要结果

qₖ=mean_seed ‖Jᵀuₖ‖²，sₖ=training variance，hₖ=sₖ/(sₖ+κ)。

\[
T(\kappa)=\sum_k(q_k/s_k)h_k^2,\qquad \mathrm{df}_2=\sum_kh_k^2.
\]

匹配 df₂=375（750 个 PC 的 mean shrinkage² 为 0.5）后：

| 指标 | CLIPAG | DINOv2 |
|---|---:|---:|
| κ | 0.954292 | 348.505 |
| κ/s₁ | 0.00874231 | 0.0105996 |
| seed-mean T | 20.0497 | 11928.3 |
| T/df₂ | 0.0534658 | 31.8088 |
| PC501–750 对 T 的贡献 | 30.98% | 30.21% |
| 逐 seed T 范围 | 11.44–39.60 | 1165.90–57866.19 |

DINO/CLIPAG 的 **均值之比为 594.94**。逐 seed 比值范围为 51.21–4625.95，
其中位数为 236.73；十张 seed 全部是 DINO 更高。第 7 张 seed 贡献 DINO 的
48.51% 总 trace。描述性地排除它后，均值之比仍为 326.71，因此差异不仅由单张
seed 决定；不应把这个事后排除结果当作预先指定的统计检验。

在 df₂/750 从 0.1 到 0.99 的已计算点，DINO 的 mean T 始终更高。
κ→0 的 retained-trace 极限分别是 CLIPAG 44.1463、DINO 24914.9。

## 初步解释

两者 spectrum 的形状有差异，但 qₖ/sₖ 在几乎整个 PC 范围内相隔多个数量级。
在 df₂=375 时 tail 的相对贡献却非常接近（约 30%）。这与 RN50 对照的现象类似：
“单位 feature variance 对应的像素梯度能量”比仅看 eigenvalue decay 更能区分
这两个模型的 local control geometry。

作为同一组 seed 上的描述性参照，匹配 df₂=375 的 mean T 为：
CLIPAG 20.05，robust RN50 37.79，DINOv2 11928.31，standard RN50 21364.04。
跨架构的这个排序不是 robustness 因果效应，也不是完整 control error 的排序。
完整 D/N 仍需要 signal term、teacher alignment、residual/noise multiplier，以及
非线性 features 上 readout covariance DE 的验证。

## 数值检查与运行

- CLIPAG training covariance 非对角相对 Frobenius norm：7.052e-6。
- CLIPAG 原缓存 score 重现：relative L2=2.629e-5，最大误差为 PC training SD 的
  6.019e-5。DINO 使用先前验证并缓存的结果（relative L2=4.697e-4）。
- H100 上 CLIPAG warm serial VJP 为 2.33 ms/PC，32-PC block 为 0.60 ms/PC。
  block/serial 最大相对差异 <4.7e-7，peak allocated memory 约 0.646 GB。
- 计时 pilot 预计 10×750 的主体计算约 4.6 秒；完整图像循环约 6 秒。
  初次模型加载另有约分钟级远程文件 I/O，不计入 VJP 时间。
- 复用 existing Nanoclaw job 47072283，不新申请 GPU job。
- 缓存、原始 per-image power 和日志：
  `$STORE_DIR/Projects/AccentuationPredRMT/nonlinear_control/clipag_seed10/`。
  DINO 缓存位于同级 `dinov2_seed10/`。

## 重画和结果文件

```bash
python explorations/nonlinear_control/compare_resnets.py \
  --first "$STORE_DIR/Projects/AccentuationPredRMT/nonlinear_control/clipag_seed10" \
  --second "$STORE_DIR/Projects/AccentuationPredRMT/nonlinear_control/dinov2_seed10" \
  --first-label 'CLIPAG ViT-B/32' --second-label 'DINOv2 ViT-B/14-reg' \
  --allow-different-layers --filename-base clipag_dinov2_comparison \
  --output figures/nonlinear_control/clipag_dinov2_comparison
```

- PNG/PDF：`figures/nonlinear_control/clipag_dinov2_comparison/`。
- 匹配 df₂ 数表：`tables/nonlinear_control/clipag_dinov2_matched_df2.csv`。
- 逐 seed 数表：`tables/nonlinear_control/clipag_dinov2_paired_seed_traces.csv`。
- 单模型可重画表：`tables/nonlinear_control/clipag_seed10/` 与 `dinov2_seed10/`。

下一步可检验第 7 张 seed 为何同时在 standard RN50 和 DINO 中产生大梯度，
以及这个梯度能量差异如何传递到已知 teacher 的局部 D/N。

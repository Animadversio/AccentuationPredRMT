# Pixel-space ridge：交互式风险与选参

从仓库根目录或 notebooks 目录启动 Jupyter，依次运行单元。

**Van Hateren 完整数据新版：`vanhateren_selection_vs_estimation.ipynb`**。已完成 d=10000、n=1000、26 个噪声水平、100 次真实自然图像 MC，并内嵌结果预览。图像通过公开镜像直接下载到本地 XFS，保持原始 .iml 文件名单与图像划分；结果及原图压缩包已备份到非 Lustre NFS scratch。详见 `notes/vanhateren_io_recovery.md`。

**推荐先看 `pixel_ridge_selection_vs_estimation.ipynb`**：新版三列图按选参方法排列，各列同轴比较 E_gen（DE/MC）与 E_acc（leading DE/Gaussian distributional DE/MC），下方显示 κ 与 λ。包含已生成的图片预览和可重画代码。复用 power-law 案例共有的四个噪声水平、每点十次 ridge MC；第三列为 distributional accentuation oracle，不是实际 CV。三种估计在每列同一组参数下评估。预览是静态附件，改参数后看绘图单元的新输出。

1. `pixel_ridge_noise_replot.ipynb`：完整 Van Hateren 圆盘 teacher 谱的 DE 曲线。比较固定 λ、按泛化风险选择的 DE-LOOCV、按 accentuation 风险选择的 oracle。可编辑噪声网格、正则化范围、图形样式。
2. `pixel_ridge_noise_experiment.ipynb`：默认 d=128、n=256 的 power-law pixel covariance；检查公式，并用实际 Gaussian-design ridge 与精确 LOOCV 验证。最后加入 Gaussian distributional-DE surrogate 选参及独立 ridge 验证。

需要 numpy、scipy、pandas、matplotlib、tqdm、threadpoolctl、Jupyter。

## 约定

- α=nλ；κ 通过 DE 自洽关系求解。选择 α 后在 n 个训练样本上重新拟合。
- DE_gen_CV 优化 n−1 样本的期望泛化风险，近似 LOOCV 的总体目标；empirical_RidgeCV 每次用真实 LOOCV 选 α，两者不可混称。
- 图中的 E 均除以 teacher signal variance S；横轴同时标 σ 和 σ²/S。
- `ACC_OBJECTIVE="E_acc"` 使用 leading ratio-of-means；`"E_acc_corrected"` 使用现有部分二阶修正，不是完整 Var(R)。负的修正风险是近似失效警告，不应解释成负的真实误差。
- Gaussian oracle 从匹配 DE 均值和逐坐标方差的权重分布采样，保留 N,D 的依赖和 D 的非负性。但非对角协方差及高阶矩未经完整推导，因此这是需验证的 surrogate，而非精确分布定理。
- oracle 都需要 teacher；不是未知 teacher 场景中直接可用的选参方法。
- 所有搜索只针对指定的有限 α 区间；boundary=True 时应扩展范围复查。提高积分样本数并更换种子检查 distributional 选参稳定性。
- 自然谱选项使用全部像素维度，不做 top-PC 截断或 whitening。MC 用 Gaussian 输入匹配该谱，并非重新抽样原始自然图像。

## 缓存与计算

全尺寸自然图像重跑的原始数据统一位于 `$STORE_DIR/Projects/AccentuationPredRMT/vanhateren_selection_estimation_v1/`，包含 progress.log、图像池、选参路径和逐 trial CSV。入口为 `python -m scripts.run_vanhateren_selection_estimation --trials 100`，需要 CUDA 和 STORE_DIR。完整像素 n=1000、d=10000，保持 Van Hateren 圆盘 teacher 及原 notebook 的 26 点噪声网格。

`notebooks/outputs/pixel_ridge/<hash>/` 保存配置、plot-ready CSV、图及 progress.log。hash 包含参数、谱、teacher 和计算代码；改画图样式不用重新拟合。Notebook 显示具体路径。默认先 pilot 报 ETA，再执行；超出 MAX_SECONDS 会停止。Gaussian 实验还有独立样本数、搜索网格和开关，建议先小规模运行。

主要可读函数位于 `scripts/pixel_ridge_notebook_utils.py`：`metrics`、`sweep`、`exact_loo_path`、`gaussian_acc_risk`、`gaussian_acc_selection`。

## 真实 sklearn RidgeCV 对照（可选显示）

`vanhateren_selection_vs_estimation.ipynb` 最后一个绘图单元设置
`SHOW_SKLEARN_CV=True/False`。新增序列直接调用 sklearn 1.6.0 的
`RidgeCV(cv=None, scoring=None, fit_intercept=True, gcv_mode='eigen', alpha_per_target=True)`，
即每个响应目标独立用 LOOCV MSE 选 alpha。26 个噪声水平共享设计矩阵分解，
但不共享所选 alpha；首个 trial 的三个目标与独立单目标调用核对通过。
沿用 181 点 alpha 网格 1e-5 至 1e7、100 个 paired trial 和原图像池。
评估仍使用总体中心化响应/权重，不添加截距误差，以与原图定义一致。

仅第二列添加点线三角形，对比 **DE 泛化选参与 empirical CV 选参**；
第一列 fixed 和第三列 accentuation oracle 不叠加此序列。第二列底行绿色为 lambda=alpha/1000 的 median 与 10–90% trial
区间；风险和 slope 的点为算术 mean，误差条 ±2 SE，斜线阴影为 10–90%。

小型绘图汇总：`outputs/pixel_ridge/vanhateren_selection_estimation/sklearn_cv_summary.csv`。
原始 trial、每个噪声的完整 LOOCV 风险路径和配置在
`/scratch/tmp/AccentuationPredRMT.bOcMkB/sklearn_ridgecv/`（holygpu8a13203），
并备份到非 Lustre NFS 的项目目录
`/n/netscratch/kempner_binxuwang_lab/Everyone/binxuwang/AccentuationPredRMT/vanhateren_selection_estimation_local_20260917/sklearn_ridgecv/`。
最终 STORE_DIR 归档仍待 Lustre 恢复；不把大缓存写入 HOME。

只重画带对照图：`python -m scripts.extend_vanhateren_evaluations --plot-only --show-sklearn-cv`。
去掉 `--show-sklearn-cv` 即恢复原图；两种版本用不同文件名，不互相覆盖。
导出 PNG 与 PDF；PDF 使用嵌入 TrueType 字体（pdf.fonttype=42），保留矢量线条和文本。

## Paper 内部 stationary point

`python -m scripts.pixel_cv_stationary` 独立求解
`(n-df2)*kappa*B23 = df23*(kappa²*B12+sigma²)`，在可行物理支上
扫描导数从负变正的根，并比较边界风险。不复用数值 alpha-grid 最优值。
此处使用 paper 的最终样本数 n=1000；原 DE-LOOCV 使用 n−1 选 alpha 后在 n 上评估。
当前 26 个噪声点中有 19 个内部极小值，另外 7 个点不画 stationary 公式。
边界转入内部的噪声阈值约 sigma=0.287107；网格首个内部解是 sigma=0.312477。

公式输出 z 后，Eacc/S=(z/(1+z))²，R²acc=1−z²，slope_acc=1/(1+z)。
三项与直接 leading-DE 在同一内部最优点的结果经数值测试一致；这不是高阶修正。
数值结果缓存为 `stationary_cv.csv`，notebook 用 `SHOW_STATIONARY` 切换。
命令行使用 `--show-stationary` 加到扩展绘图入口；仅中间列增加长短虚线和 x 标记，
并在底行给出 kappa*、lambda*；另存 `_stationary.png/pdf`，不覆盖原版本。

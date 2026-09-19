# Top-PC rotation notebooks

1. Open `top_pc_rotation_replot.ipynb` first. Run All reads the existing summary CSV, recreates the original figure, and exposes an editable Matplotlib plotting function. Set `P=100` or `P=500`; change colors, axes, or MC mean/median without fitting again.
2. Open `top_pc_rotation_experiment.ipynb` to inspect the construction and recompute. Defaults: p=500, n=1000, four noise levels, 30 exploratory trials. The original figure used 300 trials at p=500 and 1000 at p=100. The notebook exposes the tail band, rotation grid, alpha grid, and seeds.

Use a Python kernel with NumPy, SciPy, pandas, Matplotlib, PyTorch, IPython, threadpoolctl, and tqdm. PyTorch is an existing import dependency of the experiment scripts; CUDA is not required. Launch Jupyter from this repository or its notebooks directory so root discovery works.

The simulation uses Gaussian feature designs with the measured Van Hateren spectrum and disk-teacher alignment. Alpha is selected using DE LOOCV risk and then held fixed across MC trials. It is not selected by empirical RidgeCV separately in each trial. Invariance of the entire training distribution relies on the Gaussian model; preserving covariance alone does not establish that invariance for natural-image samples.

Generated figures and caches go to `notebooks/outputs/top_pc_rotation/`. The experiment notebook prints its progress-log path and saves configuration, plot-ready CSV, DE values, MC fits, and feature coefficients in a configuration-specific directory. Default MC includes a pilot ETA and a runtime budget. Set `RUN_MC=False` for a DE-only exploration. Set `FORCE=True` only when intentionally recomputing an existing configuration.

MC accentuation R² markers default to the median; other markers use means. Leading DE evaluates ratios of moments, so it is not an exact prediction of either nonlinear MC mean or median. Feature visualizations show coefficients in the population-PC basis, not pixel images.

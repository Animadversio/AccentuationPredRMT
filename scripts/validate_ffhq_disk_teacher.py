"""Validate ridge deterministic equivalents on the FFHQ disk-teacher task.

This reproduces the natural-image linear-regression experiment from
``Closed-loop-visual-insilico/notebooks/toy_model_linear_regression_sweep.py``:

* 100 x 100 grayscale FFHQ images;
* a binary radius-0.3 disk as the teacher weight;
* a fitted intercept;
* RidgeCV over scikit-learn alphas ``10**[-4, ..., 5]``.

Scikit's objective uses ``X.T X + alpha I``. The RMT library uses
``X.T X + n lambda I``, so the final-refit conversion is ``lambda=alpha/n``.
For leave-one-out CV, the DE risk path uses ``lambda=alpha/(n-1)`` at sample
size ``n-1``.

The expensive image array and population eigenvectors are cached on node-local
scratch. Plot-ready tables, small spectrum summaries, and per-condition case
caches are saved in this repository.
"""
from __future__ import annotations

import argparse
import csv
import io
import logging
import os
import pickle
import sys
import time
from pathlib import Path
from zipfile import ZipFile

os.environ.setdefault('MPLCONFIGDIR', '/tmp/accentuationpredrmt-matplotlib')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp/accentuationpredrmt-xdg-cache')

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from rmt_core import (  # noqa: E402
    SpectrumKappa,
    accentuation_error_theory,
    accentuation_r2_simulation,
    accentuation_r2_theory,
    generalization_r2_theory,
    ridge_error_per_pc_theory,
)
from scripts.validate_r2_peer_review import configure_logging, float_tag  # noqa: E402


FFHQ_ZIP = Path(
    '/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/'
    'Datasets/ffhq256-images-only.zip')
HISTORICAL_ROOT = Path(
    '/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/'
    'DL_Projects/AdvExampleLinearRegr/circ_mask_weights')

SUMMARY_PATH = REPO_ROOT / 'tables' / 'ffhq_disk_teacher_de_summary.csv'
SPECTRUM_PATH = REPO_ROOT / 'tables' / 'ffhq_disk_teacher_spectrum.npz'
CASE_DIR = REPO_ROOT / 'tables' / 'ffhq_disk_teacher_cases'
METRIC_FIGURE_PATH = REPO_ROOT / 'figures' / 'ffhq_disk_teacher_de_validation.png'
WEIGHT_FIGURE_PATH = REPO_ROOT / 'figures' / 'ffhq_disk_teacher_weights.png'
SPECTRAL_FIGURE_PATH = REPO_ROOT / 'figures' / 'ffhq_disk_teacher_eigenbasis.png'
DEFAULT_LOG_PATH = REPO_ROOT / 'logs' / 'ffhq_disk_teacher_de.log'


def create_disk_teacher(img_size: int = 100, radius: float = 0.3) -> np.ndarray:
    x = np.linspace(-1, 1, img_size)
    xx, yy = np.meshgrid(x, x)
    return ((xx ** 2 + yy ** 2) < radius ** 2).astype(np.float32).ravel()


def configure_logger(path: Path) -> logging.Logger:
    return configure_logging(path)


def sorted_png_names(archive: ZipFile) -> list[str]:
    names = [name for name in archive.namelist() if name.lower().endswith('.png')]
    return sorted(names, key=lambda name: int(Path(name).stem))


def stage_ffhq_uint8(zip_path: Path, cache_path: Path, total_images: int,
                     img_size: int, logger: logging.Logger) -> np.memmap:
    expected_shape = (total_images, img_size * img_size)
    meta_path = cache_path.with_suffix('.meta.npz')
    if cache_path.exists() and meta_path.exists():
        with np.load(meta_path) as meta:
            shape = tuple(int(value) for value in meta['shape'])
        if shape == expected_shape:
            logger.info('Loading staged FFHQ array %s, shape=%s', cache_path, shape)
            return np.load(cache_path, mmap_mode='r')

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info('Staging %d FFHQ images from contiguous zip into %s',
                total_images, cache_path)
    array = np.lib.format.open_memmap(
        cache_path, mode='w+', dtype=np.uint8, shape=expected_shape)
    start = time.perf_counter()
    try:
        from tqdm.auto import tqdm
        iterator = tqdm(range(total_images), desc='stage FFHQ zip', unit='image')
    except ImportError:
        iterator = range(total_images)
    with ZipFile(zip_path) as archive:
        names = sorted_png_names(archive)
        if total_images > len(names):
            raise ValueError(f'Requested {total_images} images but zip has {len(names)}.')
        for index in iterator:
            with archive.open(names[index]) as handle:
                raw = handle.read()
            with Image.open(io.BytesIO(raw)) as image:
                resized = image.resize((img_size, img_size)).convert('L')
                array[index] = np.asarray(resized, dtype=np.uint8).ravel()
    array.flush()
    elapsed = time.perf_counter() - start
    np.savez(meta_path, shape=np.asarray(expected_shape), elapsed_seconds=elapsed)
    logger.info('Staged data in %.1fs (%.1f images/s)',
                elapsed, total_images / elapsed)
    return np.load(cache_path, mmap_mode='r')


def compute_population_spectrum(
        staged: np.ndarray, train_pool_size: int, population_size: int,
        beta: np.ndarray, cache_dir: Path, device: torch.device,
        logger: logging.Logger) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                         np.ndarray, np.ndarray, np.ndarray]:
    eigvec_path = cache_dir / (
        f'eigvec_train{train_pool_size}_pop{population_size}_d{len(beta)}.npy')
    compact_path = cache_dir / (
        f'spectrum_train{train_pool_size}_pop{population_size}_d{len(beta)}.npz')
    if eigvec_path.exists() and compact_path.exists():
        logger.info('Loading cached population spectrum from %s', compact_path)
        with np.load(compact_path) as data:
            eigval_np = data['eigenvalues']
            beta_proj_np = data['beta_proj']
            mean_np = data['population_mean']
        eigvec_np = np.load(eigvec_path, mmap_mode='r')
        eigval = torch.from_numpy(eigval_np).to(device)
        # torch.tensor performs the required copy from the read-only memmap.
        eigvec = torch.tensor(np.asarray(eigvec_np), device=device)
        mean = torch.from_numpy(mean_np).to(device)
        return eigval, eigvec, mean, eigval_np, beta_proj_np, mean_np

    start_index = train_pool_size
    stop_index = start_index + population_size
    logger.info('Moving independent population images [%d:%d] to %s',
                start_index, stop_index, device)
    load_start = time.perf_counter()
    population = torch.from_numpy(
        np.asarray(staged[start_index:stop_index], dtype=np.float32)).to(device)
    population.div_(255.0)
    mean = population.mean(dim=0)
    population.sub_(mean)
    logger.info('Population tensor ready in %.1fs; computing covariance',
                time.perf_counter() - load_start)

    covariance_start = time.perf_counter()
    covariance = population.T @ population / population_size
    del population
    torch.cuda.empty_cache()
    logger.info('Covariance computed in %.1fs; running symmetric eigendecomposition',
                time.perf_counter() - covariance_start)

    eigh_start = time.perf_counter()
    eigval, eigvec = torch.linalg.eigh(covariance)
    del covariance
    eigval = torch.flip(eigval, dims=[0]).clamp_min_(0)
    eigvec = torch.flip(eigvec, dims=[1]).contiguous()
    beta_t = torch.from_numpy(beta).to(device)
    beta_proj = eigvec.T @ beta_t
    logger.info('Population eigendecomposition completed in %.1fs',
                time.perf_counter() - eigh_start)

    cache_dir.mkdir(parents=True, exist_ok=True)
    eigval_np = eigval.cpu().numpy()
    beta_proj_np = beta_proj.cpu().numpy()
    mean_np = mean.cpu().numpy()
    np.save(eigvec_path, eigvec.cpu().numpy())
    np.savez_compressed(
        compact_path, eigenvalues=eigval_np, beta_proj=beta_proj_np,
        population_mean=mean_np)
    SPECTRUM_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        SPECTRUM_PATH, eigenvalues=eigval_np, beta_proj=beta_proj_np,
        population_mean=mean_np, train_pool_size=train_pool_size,
        population_size=population_size, image_size=int(np.sqrt(len(beta))))
    logger.info('Cached eigenvectors at %s and compact spectrum at %s',
                eigvec_path, SPECTRUM_PATH)
    return eigval, eigvec, mean, eigval_np, beta_proj_np, mean_np


def ridge_loocv_fit(X: torch.Tensor, beta: torch.Tensor, sigma: float,
                    alphas: torch.Tensor, generator: torch.Generator
                    ) -> tuple[torch.Tensor, float, torch.Tensor]:
    """Fit intercept-aware RidgeCV using exact linear leave-one-out residuals."""
    n = X.shape[0]
    Xc = X - X.mean(dim=0, keepdim=True)
    noise = torch.randn(n, device=X.device, generator=generator) * sigma
    yc = Xc @ beta + noise
    yc = yc - yc.mean()

    gram = Xc @ Xc.T
    sample_eval, sample_evec = torch.linalg.eigh(gram)
    sample_eval = sample_eval.clamp_min_(0)
    response_proj = sample_evec.T @ yc
    # Centering creates one exact constant-vector null mode. Work only in the
    # n-1 centered modes and compute 1-h_ii directly as alpha/(s+alpha),
    # avoiding catastrophic cancellation when alpha is small.
    positive_eval = sample_eval[1:]
    positive_evec = sample_evec[:, 1:]
    positive_response = response_proj[1:]
    residual_shrink = alphas[None, :] / (
        positive_eval[:, None] + alphas[None, :])
    # Form (I-H)y directly.  Computing it as y-Hy catastrophically cancels in
    # float32 for the tiny alphas in the original experiment (the two terms
    # agree to several digits), which can change RidgeCV's selected alpha.
    residual_path = positive_evec @ (
        residual_shrink * positive_response[:, None])
    one_minus_leverage = positive_evec.square() @ residual_shrink
    loo_residual = residual_path / one_minus_leverage
    loo_mse = loo_residual.square().mean(dim=0)
    selected_index = int(torch.argmin(loo_mse))
    selected_alpha = alphas[selected_index]
    dual = positive_evec @ (
        positive_response / (positive_eval + selected_alpha))
    weight = Xc.T @ dual
    return weight, float(selected_alpha), loo_mse


def fit_metrics(weight: torch.Tensor, beta: torch.Tensor, eigval: torch.Tensor,
                eigvec: torch.Tensor, beta_proj: torch.Tensor,
                signal_power: float
                ) -> tuple[dict[str, float], torch.Tensor]:
    delta = weight - beta
    delta_proj = eigvec.T @ delta
    weight_proj = beta_proj + delta_proj
    weight_error = float(delta.square().sum())
    gen_error = float((eigval * delta_proj.square()).sum())
    gen_numerator = float((eigval * weight_proj * beta_proj).sum())
    gen_denominator = float((eigval * weight_proj.square()).sum())
    slope_gen = (gen_numerator / gen_denominator
                 if gen_denominator > 0 else np.nan)
    numerator = float(weight @ beta)
    denominator = float(weight @ weight)
    alignment = numerator / denominator if denominator > 0 else np.nan
    acc_error = signal_power * (1.0 - alignment) ** 2
    r2_acc = accentuation_r2_simulation(
        weight.detach().cpu().numpy(), beta.detach().cpu().numpy())
    return {
        'weight_error': weight_error,
        'gen_error': gen_error,
        'r2_gen': 1.0 - gen_error / signal_power,
        'slope_gen': slope_gen,
        'slope_acc': alignment,
        'acc_alignment': alignment,
        'acc_error': acc_error,
        'r2_acc': r2_acc,
    }, delta_proj


def aggregate(values: np.ndarray) -> tuple[float, float, float]:
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if len(values) > 1 else np.nan
    se = std / np.sqrt(len(values)) if len(values) > 1 else np.nan
    return mean, std, se


def select_de_alpha(eigenvalues: np.ndarray, beta_proj: np.ndarray,
                    sigma: float, n: int,
                    alphas: np.ndarray) -> tuple[float, np.ndarray]:
    n_fit = n - 1
    solver = SpectrumKappa(eigenvalues, len(eigenvalues) / n_fit)
    risks = np.empty(len(alphas), dtype=float)
    for index, alpha in enumerate(alphas):
        lam = alpha / n_fit
        kappa = solver(lam)
        _, gen_error, _ = generalization_r2_theory(
            eigenvalues, beta_proj, kappa, sigma, n_fit)
        risks[index] = gen_error
    return float(alphas[int(np.argmin(risks))]), risks


def theory_metrics(eigenvalues: np.ndarray, beta_proj: np.ndarray,
                   sigma: float, n: int, alpha: float) -> tuple[dict[str, float],
                                                                 np.ndarray]:
    lam = alpha / n
    kappa = SpectrumKappa(eigenvalues, len(eigenvalues) / n)(lam)
    error_pc, *_ = ridge_error_per_pc_theory(
        eigenvalues, beta_proj, kappa, sigma, n)
    shrink = eigenvalues / (eigenvalues + kappa)
    bias_pc = (1.0 - shrink) ** 2 * beta_proj ** 2
    variance_pc = np.maximum(error_pc - bias_pc, 0.0)
    second_moment_pc = shrink ** 2 * beta_proj ** 2 + variance_pc
    gen_slope_numerator = np.sum(
        eigenvalues * shrink * beta_proj ** 2)
    gen_slope_denominator = np.sum(eigenvalues * second_moment_pc)
    slope_gen = gen_slope_numerator / gen_slope_denominator
    r2_gen, gen_error, signal_power = generalization_r2_theory(
        eigenvalues, beta_proj, kappa, sigma, n)
    acc_error, alignment, _ = accentuation_error_theory(
        eigenvalues, beta_proj, kappa, sigma, n, include_var_R=True)
    r2_acc, _, _ = accentuation_r2_theory(
        eigenvalues, beta_proj, kappa, sigma, n,
        include_delta_correction=True)
    return {
        'theory_alpha_cv': alpha,
        'theory_lambda_final': lam,
        'theory_kappa_final': kappa,
        'theory_signal_power': signal_power,
        'theory_weight_error': float(np.sum(error_pc)),
        'theory_gen_error': gen_error,
        'theory_r2_gen': r2_gen,
        'theory_slope_gen': slope_gen,
        'theory_slope_acc': alignment,
        'theory_acc_alignment': alignment,
        'theory_acc_error': acc_error,
        'theory_r2_acc': r2_acc,
    }, error_pc


def historical_pickle(n: int, sigma: float) -> Path:
    return HISTORICAL_ROOT / (
        f'regression_results_n_samp{n}_noise{float(sigma)}.pkl')


def infer_historical_alpha(
        X: torch.Tensor, weight_ols: np.ndarray, weight_ridge: np.ndarray,
        alphas: torch.Tensor) -> tuple[float, float]:
    """Infer the omitted RidgeCV alpha using the paired saved OLS solution."""
    Xc = X - X.mean(dim=0, keepdim=True)
    gram = Xc @ Xc.T
    sample_eval, sample_evec = torch.linalg.eigh(gram)
    sample_eval = sample_eval.clamp_min_(0)
    ols = torch.from_numpy(weight_ols.astype(np.float32)).to(X.device)
    ridge = torch.from_numpy(weight_ridge.astype(np.float32)).to(X.device)
    response_proj = sample_evec.T @ (Xc @ ols)
    errors = []
    for alpha in alphas:
        dual = sample_evec @ (response_proj / (sample_eval + alpha))
        predicted = Xc.T @ dual
        errors.append(float((predicted - ridge).square().sum()))
    best = int(np.argmin(errors))
    relative = np.sqrt(errors[best]) / float(torch.linalg.vector_norm(ridge))
    return float(alphas[best]), relative


def summarize_historical(
        n: int, sigma: float, X_original: torch.Tensor, beta: torch.Tensor,
        eigval: torch.Tensor, eigvec: torch.Tensor, beta_proj: torch.Tensor,
        signal_power: float, alphas: torch.Tensor
        ) -> tuple[dict[str, float], np.ndarray]:
    path = historical_pickle(n, sigma)
    if not path.exists():
        return {}, np.full(len(beta), np.nan, dtype=np.float32)
    with path.open('rb') as handle:
        result = pickle.load(handle)
    ridge_np = np.asarray(result['coef_Ridge'], dtype=np.float32).ravel()
    ols_np = np.asarray(result['coef_OLS'], dtype=np.float32).ravel()
    if ridge_np.size != beta.numel():
        return {}, np.full(beta.numel(), np.nan, dtype=np.float32)
    inferred_alpha, residual = infer_historical_alpha(
        X_original, ols_np, ridge_np, alphas)
    ridge = torch.from_numpy(ridge_np).to(beta.device)
    metrics, _ = fit_metrics(
        ridge, beta, eigval, eigvec, beta_proj, signal_power)
    return {
        'historical_alpha_inferred': inferred_alpha,
        'historical_alpha_inference_relative_error': residual,
        'historical_weight_error': metrics['weight_error'],
        'historical_gen_error': metrics['gen_error'],
        'historical_r2_gen_population': metrics['r2_gen'],
        'historical_slope_gen': metrics['slope_gen'],
        'historical_slope_acc': metrics['slope_acc'],
        'historical_acc_error': metrics['acc_error'],
        'historical_r2_acc': metrics['r2_acc'],
        'historical_r2_test_saved': float(result['r2_test_Ridge']),
    }, ridge_np


def case_path(args: argparse.Namespace, sigma: float) -> Path:
    return CASE_DIR / (
        f'ffhq_disk_d{args.img_size ** 2}_n{args.n}_sigma{float_tag(sigma)}'
        f'_trials{args.n_trials}_pop{args.population_size}_seed{args.seed}.npz')


def save_case(path: Path, row: dict[str, object], **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {f'row_{key}': np.asarray(value) for key, value in row.items()}
    payload.update(arrays)
    np.savez_compressed(path, **payload)


def load_case(path: Path) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as data:
        row = {
            key.removeprefix('row_'): data[key].item()
            for key in data.files if key.startswith('row_')
        }
        arrays = {
            key: np.asarray(data[key]) for key in data.files
            if not key.startswith('row_')
        }
    return row, arrays


def write_summary(rows: list[dict[str, object]]) -> None:
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_PATH.open('w', newline='') as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    SUMMARY_PATH.chmod(0o644)


def read_summary() -> list[dict[str, object]]:
    with SUMMARY_PATH.open(newline='') as handle:
        rows: list[dict[str, object]] = list(csv.DictReader(handle))
    for row in rows:
        for key, value in list(row.items()):
            try:
                row[key] = float(value)
            except (TypeError, ValueError):
                pass
    return rows


def plot_metrics(rows: list[dict[str, object]]) -> None:
    rows = sorted(rows, key=lambda row: float(row['sigma']))
    sigma = np.asarray([float(row['sigma']) for row in rows])
    panels = [
        ('weight_error', 'theory_weight_error', 'historical_weight_error',
         r'$\|\hat\beta-\beta^*\|^2$', True),
        ('r2_gen', 'theory_r2_gen', 'historical_r2_gen_population',
         r'$R^2_{gen}$', False),
        ('r2_acc', 'theory_r2_acc', 'historical_r2_acc',
         r'$R^2_{acc}$', False),
        ('slope_gen', 'theory_slope_gen', 'historical_slope_gen',
         r'slope$_{gen}$: true on fitted', False),
        ('slope_acc', 'theory_slope_acc', 'historical_slope_acc',
         r'slope$_{acc}$: true on fitted', False),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14.2, 7.6), squeeze=False)
    legend_handles = legend_labels = None
    for ax, (mc_key, theory_key, history_key, ylabel, log_y) in zip(
            axes.ravel(), panels):
        mc = np.asarray([float(row[f'mc_{mc_key}']) for row in rows])
        se = np.asarray([float(row[f'mc_{mc_key}_se']) for row in rows])
        theory = np.asarray([float(row[theory_key]) for row in rows])
        historical = np.asarray([float(row.get(history_key, np.nan)) for row in rows])
        ax.errorbar(sigma, mc, yerr=2 * se, fmt='o-', color='black',
                    capsize=3, label='natural-image MC, mean +/- 2 SE')
        ax.plot(sigma, theory, 's--', color='C0', label='DE')
        ax.plot(sigma, historical, '*', color='C3', ms=9,
                label='original saved fit')
        ax.set_xscale('symlog', linthresh=0.01)
        if log_y and np.all(mc > 0) and np.all(theory > 0):
            ax.set_yscale('log')
        else:
            ax.axhline(1.0, color='0.5', lw=0.8, ls=':', zorder=0)
        ax.set_xlabel(r'response noise $\sigma$')
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.2)
        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()
    legend_ax = axes.ravel()[-1]
    legend_ax.axis('off')
    legend_ax.legend(legend_handles, legend_labels, loc='center', fontsize=10,
                     frameon=False)
    legend_ax.text(
        0.5, 0.28,
        'Slopes regress true teacher response\non fitted-model response.',
        ha='center', va='center', fontsize=10, transform=legend_ax.transAxes)
    fig.suptitle('FFHQ disk teacher: deterministic equivalent vs natural-image fits')
    fig.tight_layout()
    METRIC_FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(METRIC_FIGURE_PATH, dpi=180, bbox_inches='tight')
    METRIC_FIGURE_PATH.chmod(0o644)
    plt.close(fig)


def representative_sigmas(rows: list[dict[str, object]]) -> list[float]:
    available = sorted(float(row['sigma']) for row in rows)
    targets = [0.1, 1.0, 10.0]
    return [min(available, key=lambda value: abs(value - target))
            for target in targets]


def plot_weights(rows: list[dict[str, object]], cases: dict[float, dict[str, np.ndarray]],
                 img_size: int) -> None:
    sigmas = representative_sigmas(rows)
    beta = create_disk_teacher(img_size).reshape(img_size, img_size)
    fig, axes = plt.subplots(4, 4, figsize=(11.5, 10.8), squeeze=False)
    axes[0, 0].imshow(beta, cmap='coolwarm', vmin=-1, vmax=1)
    axes[0, 0].set_title('disk teacher')
    row_labels = ['Original saved Ridge', 'MC representative', 'MC mean', 'DE mean']
    axes[0, 0].set_ylabel(row_labels[0])
    for row_index in range(1, 4):
        axes[row_index, 0].imshow(beta, cmap='coolwarm', vmin=-1, vmax=1)
        axes[row_index, 0].set_ylabel(row_labels[row_index])
    for column, sigma in enumerate(sigmas, start=1):
        arrays = cases[sigma]
        maps = [
            arrays['historical_weight'], arrays['representative_weight'],
            arrays['mean_weight'], arrays['de_mean_weight'],
        ]
        for row_index, weight in enumerate(maps):
            if not np.any(np.isfinite(weight)):
                weight = np.zeros(img_size * img_size, dtype=float)
            vmax = max(float(np.nanmax(np.abs(weight))), 1e-6)
            axes[row_index, column].imshow(
                weight.reshape(img_size, img_size), cmap='coolwarm',
                vmin=-vmax, vmax=vmax)
            if row_index == 0:
                axes[row_index, column].set_title(rf'$\sigma={sigma:g}$')
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle('Recovered FFHQ disk-teacher weights')
    fig.tight_layout()
    WEIGHT_FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(WEIGHT_FIGURE_PATH, dpi=180, bbox_inches='tight')
    WEIGHT_FIGURE_PATH.chmod(0o644)
    plt.close(fig)


def binned_mean(values: np.ndarray, n_bins: int = 100) -> tuple[np.ndarray, np.ndarray]:
    edges = np.linspace(0, len(values), n_bins + 1, dtype=int)
    x = np.empty(n_bins)
    y = np.empty(n_bins)
    for index in range(n_bins):
        slc = slice(edges[index], edges[index + 1])
        x[index] = 0.5 * (edges[index] + edges[index + 1]) / len(values)
        y[index] = np.mean(values[slc])
    return x, y


def plot_spectrum(rows: list[dict[str, object]],
                  cases: dict[float, dict[str, np.ndarray]]) -> None:
    with np.load(SPECTRUM_PATH) as spectrum:
        eigenvalues = spectrum['eigenvalues']
        beta_proj = spectrum['beta_proj']
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.8))
    rank = (np.arange(len(eigenvalues)) + 0.5) / len(eigenvalues)
    axes[0].semilogy(rank, np.maximum(eigenvalues, 1e-12), lw=1)
    axes[0].set_ylabel(r'population eigenvalue $\lambda_k$')
    axes[0].set_title('Centered FFHQ spectrum')
    axes[1].semilogy(rank, beta_proj ** 2 + 1e-12, lw=1, color='C1')
    axes[1].set_ylabel(r'disk-teacher power $(u_k^T\beta^*)^2$')
    axes[1].set_title('Teacher spectral alignment')
    for sigma, color in zip(representative_sigmas(rows)[1:], ['C0', 'C3']):
        arrays = cases[sigma]
        x, mc = binned_mean(arrays['mc_error_per_pc'])
        _, theory = binned_mean(arrays['theory_error_per_pc'])
        axes[2].semilogy(x, mc + 1e-15, '-', color=color,
                         label=rf'MC $\sigma={sigma:g}$')
        axes[2].semilogy(x, theory + 1e-15, '--', color=color,
                         label=rf'DE $\sigma={sigma:g}$')
    axes[2].set_ylabel(r'mean PC weight error')
    axes[2].set_title('Weight-error allocation')
    axes[2].legend(fontsize=7)
    for ax in axes:
        ax.set_xlabel('PC rank (top 0 to bottom 1)')
        ax.set_xlim(0, 1)
        ax.grid(alpha=0.2)
    fig.tight_layout()
    SPECTRAL_FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(SPECTRAL_FIGURE_PATH, dpi=180, bbox_inches='tight')
    SPECTRAL_FIGURE_PATH.chmod(0o644)
    plt.close(fig)


def render_all(rows: list[dict[str, object]], img_size: int) -> None:
    cases = {}
    for row in rows:
        sigma = float(row['sigma'])
        _, arrays = load_case(Path(str(row['case_path'])))
        cases[sigma] = arrays
    plot_metrics(rows)
    plot_weights(rows, cases, img_size)
    plot_spectrum(rows, cases)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-size', type=int, default=100)
    parser.add_argument('--radius', type=float, default=0.3)
    parser.add_argument('--n', type=int, default=1000)
    parser.add_argument('--train-pool-size', type=int, default=10000)
    parser.add_argument('--population-size', type=int, default=20000)
    parser.add_argument('--sigmas', type=float, nargs='+',
                        default=[0.0, 0.01, 0.1, 1.0, 3.0, 10.0])
    parser.add_argument('--n-trials', type=int, default=100)
    parser.add_argument('--benchmark-trials', type=int, default=3)
    parser.add_argument('--seed', type=int, default=20260814)
    parser.add_argument('--zip-path', type=Path, default=FFHQ_ZIP)
    parser.add_argument('--scratch-dir', type=Path,
                        default=Path('/tmp/ffhq_disk_teacher_de'))
    parser.add_argument('--force-spectrum', action='store_true')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--plot-only', action='store_true')
    parser.add_argument('--no-progress', action='store_true')
    parser.add_argument('--log-file', type=Path, default=DEFAULT_LOG_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = configure_logger(args.log_file)
    if args.plot_only:
        rows = read_summary()
        render_all(rows, args.img_size)
        logger.info('Replotted three figures from cached summaries/cases')
        return
    if not torch.cuda.is_available():
        raise RuntimeError('The exact d=10,000 experiment requires a CUDA GPU.')

    device = torch.device('cuda')
    logger.info('GPU: %s', torch.cuda.get_device_name(device))
    args.scratch_dir.mkdir(parents=True, exist_ok=True)
    total_images = args.train_pool_size + args.population_size
    staged_path = args.scratch_dir / (
        f'ffhq_gray_{args.img_size}px_{total_images}_uint8.npy')
    staged = stage_ffhq_uint8(
        args.zip_path, staged_path, total_images, args.img_size, logger)

    beta_np = create_disk_teacher(args.img_size, args.radius)
    beta = torch.from_numpy(beta_np).to(device)
    if args.force_spectrum:
        for path in args.scratch_dir.glob(
                f'*train{args.train_pool_size}_pop{args.population_size}_d{len(beta_np)}*'):
            path.unlink()
    eigval, eigvec, _, eigenvalues, beta_proj, _ = compute_population_spectrum(
        staged, args.train_pool_size, args.population_size, beta_np,
        args.scratch_dir, device, logger)
    beta_proj_t = torch.from_numpy(beta_proj.astype(np.float32)).to(device)
    signal_power = float((eigval * beta_proj_t.square()).sum())
    logger.info('Disk teacher: norm²=%.1f, population signal power=%.6g',
                float(beta @ beta), signal_power)

    train_pool = torch.from_numpy(
        np.asarray(staged[:args.train_pool_size], dtype=np.float32)).to(device)
    train_pool.div_(255.0)
    X_original = train_pool[:args.n]
    alphas_np = np.logspace(-4, 5, 10)
    alphas = torch.from_numpy(alphas_np.astype(np.float32)).to(device)

    # Small-scale timing test using the exact n and d before the full loop.
    benchmark_generator = torch.Generator(device=device).manual_seed(args.seed)
    benchmark_start = time.perf_counter()
    for _ in range(args.benchmark_trials):
        indices = torch.randperm(
            args.train_pool_size, generator=benchmark_generator,
            device=device)[:args.n]
        ridge_loocv_fit(
            train_pool[indices], beta, args.sigmas[0], alphas,
            benchmark_generator)
    torch.cuda.synchronize()
    seconds_per_trial = (
        time.perf_counter() - benchmark_start) / args.benchmark_trials
    projected = seconds_per_trial * args.n_trials * len(args.sigmas)
    logger.info(
        'Benchmark: %.4fs/trial; projected simulation %.1fs (%.1fmin) for %d cases x %d trials',
        seconds_per_trial, projected, projected / 60,
        len(args.sigmas), args.n_trials)

    rows: list[dict[str, object]] = []
    run_start = time.perf_counter()
    for case_index, sigma in enumerate(args.sigmas, start=1):
        cache = case_path(args, sigma)
        if cache.exists() and not args.force:
            logger.info('[%d/%d] Loading %s', case_index, len(args.sigmas), cache)
            row, _ = load_case(cache)
            rows.append(row)
            continue

        logger.info('[%d/%d] sigma=%.3g: %d natural-image RidgeCV trials',
                    case_index, len(args.sigmas), sigma, args.n_trials)
        alpha_de, de_risk_path = select_de_alpha(
            eigenvalues, beta_proj, sigma, args.n, alphas_np)
        theoretical, theory_error_pc = theory_metrics(
            eigenvalues, beta_proj, sigma, args.n, alpha_de)

        metric_names = [
            'weight_error', 'gen_error', 'r2_gen', 'slope_gen', 'slope_acc',
            'acc_alignment', 'acc_error', 'r2_acc', 'alpha_cv']
        trials = {name: np.empty(args.n_trials, dtype=float)
                  for name in metric_names}
        coefficients = np.empty((args.n_trials, len(beta_np)), dtype=np.float32)
        error_pc_sum = torch.zeros_like(eigval)
        generator = torch.Generator(device=device).manual_seed(
            args.seed + 1000 * case_index)
        iterator = range(args.n_trials)
        if not args.no_progress:
            try:
                from tqdm.auto import tqdm
                iterator = tqdm(iterator, desc=f'FFHQ RidgeCV sigma={sigma:g}',
                                unit='trial')
            except ImportError:
                pass
        case_start = time.perf_counter()
        for trial_index in iterator:
            indices = torch.randperm(
                args.train_pool_size, generator=generator,
                device=device)[:args.n]
            weight, selected_alpha, _ = ridge_loocv_fit(
                train_pool[indices], beta, sigma, alphas, generator)
            metrics, delta_proj = fit_metrics(
                weight, beta, eigval, eigvec, beta_proj_t, signal_power)
            for name in metric_names[:-1]:
                trials[name][trial_index] = metrics[name]
            trials['alpha_cv'][trial_index] = selected_alpha
            coefficients[trial_index] = weight.cpu().numpy()
            error_pc_sum += delta_proj.square()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - case_start

        row: dict[str, object] = {
            'sigma': sigma,
            'n': args.n,
            'd': len(beta_np),
            'train_pool_size': args.train_pool_size,
            'population_size': args.population_size,
            'n_trials': args.n_trials,
            'seed': args.seed,
            'case_path': str(cache),
            'mc_elapsed_seconds': elapsed,
        }
        row.update(theoretical)
        for name, values in trials.items():
            mean_value, std_value, se_value = aggregate(values)
            row[f'mc_{name}'] = mean_value
            row[f'mc_{name}_std'] = std_value
            row[f'mc_{name}_se'] = se_value
        row['mc_alpha_cv_median'] = float(np.median(trials['alpha_cv']))
        row['mc_alpha_cv_q25'] = float(np.quantile(trials['alpha_cv'], 0.25))
        row['mc_alpha_cv_q75'] = float(np.quantile(trials['alpha_cv'], 0.75))

        historical, historical_weight = summarize_historical(
            args.n, sigma, X_original, beta, eigval, eigvec, beta_proj_t,
            signal_power, alphas)
        row.update(historical)

        mean_weight = coefficients.mean(axis=0)
        shrink = eigenvalues / (
            eigenvalues + float(theoretical['theory_kappa_final']))
        de_mean_weight = eigvec @ torch.from_numpy(
            (shrink * beta_proj).astype(np.float32)).to(device)
        arrays = {
            'trial_alpha_cv': trials['alpha_cv'],
            'trial_weight_error': trials['weight_error'],
            'trial_gen_error': trials['gen_error'],
            'trial_slope_gen': trials['slope_gen'],
            'trial_slope_acc': trials['slope_acc'],
            'trial_acc_error': trials['acc_error'],
            'trial_r2_acc': trials['r2_acc'],
            'mc_error_per_pc': (error_pc_sum / args.n_trials).cpu().numpy(),
            'theory_error_per_pc': theory_error_pc,
            'mean_weight': mean_weight,
            'representative_weight': coefficients[0],
            'historical_weight': historical_weight,
            'de_mean_weight': de_mean_weight.cpu().numpy(),
            'de_cv_risk_path': de_risk_path,
            'alpha_grid': alphas_np,
        }
        save_case(cache, row, **arrays)
        rows.append(row)
        logger.info(
            '[%d/%d] done %.1fs | alpha DE/MC median %.3g/%.3g | weight² DE/MC %.3g/%.3g | Eacc DE/MC %.3g/%.3g',
            case_index, len(args.sigmas), elapsed, alpha_de,
            row['mc_alpha_cv_median'], row['theory_weight_error'],
            row['mc_weight_error'], row['theory_acc_error'],
            row['mc_acc_error'])

    rows.sort(key=lambda row: float(row['sigma']))
    write_summary(rows)
    render_all(rows, args.img_size)
    logger.info('Finished post-spectrum workload in %.1fs',
                time.perf_counter() - run_start)
    logger.info('Summary: %s', SUMMARY_PATH)
    logger.info('Cases: %s', CASE_DIR)
    logger.info('Figures: %s, %s, %s',
                METRIC_FIGURE_PATH, WEIGHT_FIGURE_PATH, SPECTRAL_FIGURE_PATH)


if __name__ == '__main__':
    main()

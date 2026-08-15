"""Examine PCA, whitening, and top-PC feature spaces on the FFHQ disk teacher."""
from __future__ import annotations

import argparse
import csv
import logging
import os
import time
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/accentuationpredrmt-matplotlib')

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from rmt_core import (
    SpectrumKappa,
    select_spectral_feature_de_alpha,
    spectral_feature_de_metrics,
)
from scripts.validate_ffhq_disk_teacher import (
    SPECTRUM_PATH,
    add_noise_ratio_axis,
    configure_logger,
)
from scripts.validate_r2_peer_review import float_tag


REPO_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = REPO_ROOT / 'tables' / 'ffhq_linear_feature_summary.csv'
CUTOFF_PATH = REPO_ROOT / 'tables' / 'ffhq_top_pc_cutoff_summary.csv'
CASE_DIR = REPO_ROOT / 'tables' / 'ffhq_linear_feature_cases'
FIGURE_PATH = REPO_ROOT / 'figures' / 'ffhq_linear_feature_comparison.png'
CUTOFF_FIGURE_PATH = REPO_ROOT / 'figures' / 'ffhq_top_pc_cutoff_sweep.png'
GAP_FIGURE_PATH = (
    REPO_ROOT / 'figures' / 'ffhq_linear_feature_gen_acc_gap.png')
DEFAULT_LOG_PATH = REPO_ROOT / 'logs' / 'ffhq_linear_feature_validation.log'
FEATURE_NAMES = ('pca_full', 'whiten_full', 'top_pc_100')
FEATURE_LABELS = {
    'pca_full': 'full PCA',
    'whiten_full': 'full whitening',
    'top_pc_100': 'top 100 PCs',
}
FEATURE_COLORS = {
    'pca_full': 'C0',
    'whiten_full': 'C1',
    'top_pc_100': 'C2',
}
VALIDATION_RATIOS = np.asarray([0.01, 0.1, 1.0, 10.0])
CUTOFFS = np.asarray(
    [5, 10, 20, 30, 50, 70, 100, 150, 200, 300, 500, 700,
     1000, 1500, 2000, 3000, 5000, 7000, 10000])


def aggregate(values: np.ndarray) -> tuple[float, float, float]:
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if len(values) > 1 else np.nan
    se = std / np.sqrt(len(values)) if len(values) > 1 else np.nan
    return mean, std, se


def feature_problem(
        name: str, eigenvalues: np.ndarray, beta: np.ndarray
        ) -> dict[str, np.ndarray | float | int]:
    d = len(eigenvalues)
    if name == 'pca_full':
        k = d
        gains = np.ones(k)
    elif name == 'whiten_full':
        k = d
        gains = eigenvalues ** -0.5
    elif name.startswith('top_pc_'):
        k = int(name.removeprefix('top_pc_'))
        if not 1 <= k <= d:
            raise ValueError(f'invalid top-PC cutoff {k}')
        gains = np.ones(k)
    else:
        raise ValueError(f'unknown feature map {name!r}')
    retained_eigenvalues = eigenvalues[:k]
    retained_beta = beta[:k]
    total_signal = float(np.sum(eigenvalues * beta ** 2))
    retained_signal = float(np.sum(retained_eigenvalues * retained_beta ** 2))
    return {
        'feature_dim': k,
        'feature_eigenvalues': gains ** 2 * retained_eigenvalues,
        'feature_teacher': retained_beta / gains,
        'backprop_weights': gains ** 2,
        'residual_signal': max(total_signal - retained_signal, 0.0),
        'total_signal': total_signal,
        'retained_signal_fraction': retained_signal / total_signal,
    }


def de_row(
        feature_name: str, problem: dict[str, np.ndarray | float | int],
        sigma: float, n: int, alphas: np.ndarray) -> dict[str, float | str]:
    c = np.asarray(problem['feature_eigenvalues'])
    theta = np.asarray(problem['feature_teacher'])
    g = np.asarray(problem['backprop_weights'])
    residual = float(problem['residual_signal'])
    signal = float(problem['total_signal'])
    alpha, _ = select_spectral_feature_de_alpha(
        c, theta, g, residual, signal, sigma, n, alphas)
    kappa = SpectrumKappa(c, len(c) / n)(alpha / n)
    metrics = spectral_feature_de_metrics(
        c, theta, g, residual, signal, kappa, sigma, n)
    return {
        'feature': feature_name,
        'feature_dim': int(problem['feature_dim']),
        'n': n,
        'sigma': sigma,
        'noise_signal_ratio': sigma ** 2 / signal,
        'retained_signal_fraction': float(problem['retained_signal_fraction']),
        'residual_signal_fraction': residual / signal,
        'de_alpha_cv': alpha,
        'de_lambda': alpha / n,
        'de_kappa': kappa,
        **{f'de_{key}': value for key, value in metrics.items()},
    }


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    path.chmod(0o644)


def read_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(newline='') as handle:
        for row in csv.DictReader(handle):
            converted: dict[str, object] = {}
            for key, value in row.items():
                if value in ('', None):
                    converted[key] = np.nan
                    continue
                try:
                    converted[key] = float(value)
                except ValueError:
                    converted[key] = value
            rows.append(converted)
    return rows


def ridge_loocv_from_data(
        X: torch.Tensor, y: torch.Tensor, alphas: torch.Tensor
        ) -> tuple[torch.Tensor, float, torch.Tensor]:
    """Exact intercept-aware analytic LOOCV for a fixed feature dataset."""
    n = len(X)
    Xc = X - X.mean(dim=0, keepdim=True)
    yc = y - y.mean()
    gram = Xc @ Xc.T
    sample_eval, sample_evec = torch.linalg.eigh(gram)
    sample_eval.clamp_min_(0)
    response_proj = sample_evec.T @ yc
    residual_shrink = alphas[None, :] / (
        sample_eval[:, None] + alphas[None, :])
    residual_path = sample_evec @ (
        residual_shrink * response_proj[:, None])
    # The fitted intercept contributes leverage 1/n.  Keeping every Gram null
    # mode makes this expression valid for both p<n and p>n feature maps.
    one_minus_leverage = (
        sample_evec.square() @ residual_shrink - 1.0 / n)
    loo_residual = residual_path / one_minus_leverage.clamp_min(1e-8)
    selected_index = int(torch.argmin(loo_residual.square().mean(dim=0)))
    selected_alpha = alphas[selected_index]
    dual = sample_evec @ (
        response_proj / (sample_eval + selected_alpha))
    estimate = Xc.T @ dual
    return estimate, float(selected_alpha), loo_residual.square().mean(dim=0)


def ridge_loocv_feature_fit(
        c: torch.Tensor, theta: torch.Tensor, sigma_effective: float,
        alphas: torch.Tensor, generator: torch.Generator, n: int = 1000
        ) -> tuple[torch.Tensor, float]:
    """Generate a Gaussian feature sample and fit exact analytic RidgeCV."""
    p = len(c)
    X = torch.randn((n, p), device=c.device, generator=generator)
    X.mul_(torch.sqrt(c)[None, :])
    y = X @ theta
    y.add_(torch.randn(n, device=c.device, generator=generator) * sigma_effective)
    estimate, alpha, _ = ridge_loocv_from_data(X, y, alphas)
    return estimate, alpha


def simulation_metrics(
        estimate: torch.Tensor, theta: torch.Tensor, c: torch.Tensor,
        g: torch.Tensor, residual_signal: float, total_signal: float
        ) -> dict[str, float]:
    delta = estimate - theta
    gen_error = residual_signal + float(torch.sum(c * delta.square()))
    gen_num = float(torch.sum(c * estimate * theta))
    gen_den = float(torch.sum(c * estimate.square()))
    acc_num = float(torch.sum(g * estimate * theta))
    acc_den = float(torch.sum(g * estimate.square()))
    slope_acc = acc_num / acc_den
    return {
        'gen_error_normalized': gen_error / total_signal,
        'r2_gen': 1.0 - gen_error / total_signal,
        'slope_gen': gen_num / gen_den,
        'acc_error_normalized': (1.0 - slope_acc) ** 2,
        'r2_acc': (1.0 - (1.0 - acc_den / acc_num) ** 2
                   if acc_num != 0 else -np.inf),
        'slope_acc': slope_acc,
    }


def case_path(feature: str, ratio: float, n_trials: int, seed: int) -> Path:
    return CASE_DIR / (
        f'ffhq_feature_{feature}_nsr{float_tag(ratio)}_trials{n_trials}_'
        f'seed{seed}.npz')


def run_mc_case(
        feature: str, problem: dict[str, np.ndarray | float | int],
        sigma: float, ratio: float, alphas_np: np.ndarray, n_trials: int,
        seed: int, device: torch.device, force: bool, logger: logging.Logger,
        progress: bool = True) -> dict[str, float | str]:
    path = case_path(feature, ratio, n_trials, seed)
    if path.exists() and not force:
        with np.load(path) as data:
            return {key.removeprefix('row_'): data[key].item()
                    for key in data.files if key.startswith('row_')}
    c = torch.from_numpy(
        np.asarray(problem['feature_eigenvalues'], dtype=np.float32)).to(device)
    theta = torch.from_numpy(
        np.asarray(problem['feature_teacher'], dtype=np.float32)).to(device)
    g = torch.from_numpy(
        np.asarray(problem['backprop_weights'], dtype=np.float32)).to(device)
    residual = float(problem['residual_signal'])
    signal = float(problem['total_signal'])
    sigma_effective = float(np.sqrt(sigma ** 2 + residual))
    alphas = torch.from_numpy(alphas_np.astype(np.float32)).to(device)
    names = [
        'alpha_cv', 'gen_error_normalized', 'r2_gen', 'slope_gen',
        'acc_error_normalized', 'r2_acc', 'slope_acc']
    trials = {name: np.empty(n_trials) for name in names}
    generator = torch.Generator(device=device).manual_seed(
        seed + int(round(1e4 * np.log10(ratio + 1e-12))) % 100000)
    iterator = range(n_trials)
    if progress:
        try:
            from tqdm.auto import tqdm
            iterator = tqdm(iterator, desc=f'{feature} NSR={ratio:g}', unit='trial')
        except ImportError:
            pass
    start = time.perf_counter()
    for trial in iterator:
        estimate, alpha = ridge_loocv_feature_fit(
            c, theta, sigma_effective, alphas, generator)
        metrics = simulation_metrics(
            estimate, theta, c, g, residual, signal)
        trials['alpha_cv'][trial] = alpha
        for name, value in metrics.items():
            trials[name][trial] = value
    torch.cuda.synchronize()
    row: dict[str, float | str] = {
        'feature': feature,
        'feature_dim': int(problem['feature_dim']),
        'sigma': sigma,
        'noise_signal_ratio': ratio,
        'n_trials': n_trials,
        'mc_elapsed_seconds': time.perf_counter() - start,
        'case_path': str(path),
    }
    for name, values in trials.items():
        mean, std, se = aggregate(values)
        row[f'mc_{name}'] = mean
        row[f'mc_{name}_std'] = std
        row[f'mc_{name}_se'] = se
        row[f'mc_{name}_median'] = float(np.median(values))
        row[f'mc_{name}_q25'] = float(np.quantile(values, 0.25))
        row[f'mc_{name}_q75'] = float(np.quantile(values, 0.75))
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {f'row_{key}': np.asarray(value) for key, value in row.items()}
    payload.update({f'trial_{key}': values for key, values in trials.items()})
    np.savez_compressed(path, **payload)
    logger.info(
        '%s NSR=%g done %.1fs | alpha median %.3g | Egen/S %.3g | Eacc/S %.3g',
        feature, ratio, row['mc_elapsed_seconds'], row['mc_alpha_cv_median'],
        row['mc_gen_error_normalized'], row['mc_acc_error_normalized'])
    return row


def merge_mc(rows: list[dict[str, object]], mc_rows: list[dict[str, object]]) -> None:
    for mc in mc_rows:
        candidates = [
            row for row in rows
            if row['feature'] == mc['feature'] and np.isclose(
                float(row['noise_signal_ratio']), float(mc['noise_signal_ratio']))]
        if len(candidates) != 1:
            raise RuntimeError(f'could not uniquely merge MC row {mc}')
        candidates[0].update(mc)


def plot_noise_summary(rows: list[dict[str, object]]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14.4, 8.6), sharex=True)
    specs = [
        ('alpha_cv', r'RidgeCV $\alpha=n\lambda$', True),
        ('gen_error_normalized', r'$E_{gen}/S$', True),
        ('acc_error_normalized', r'$E_{acc}/S$', True),
        ('r2_gen', r'$R^2_{gen}$', False),
        ('r2_acc', r'$R^2_{acc}$', False),
        ('slope_acc', r'slope$_{acc}$', True),
    ]
    max_sigma = max(float(row['sigma']) for row in rows) * 1.15
    signal = max(float(row['sigma']) ** 2 / float(row['noise_signal_ratio'])
                 for row in rows if float(row['noise_signal_ratio']) > 0)
    for ax, (metric, ylabel, log_y) in zip(axes.ravel(), specs):
        for feature in FEATURE_NAMES:
            selected = sorted(
                [row for row in rows if row['feature'] == feature],
                key=lambda item: float(item['sigma']))
            sigma = np.asarray([float(row['sigma']) for row in selected])
            de = np.asarray([float(row[f'de_{metric}']) for row in selected])
            color = FEATURE_COLORS[feature]
            ax.plot(sigma, de, '-', color=color, lw=1.8,
                    label=FEATURE_LABELS[feature])
            mc_selected = [row for row in selected if f'mc_{metric}' in row]
            if mc_selected:
                mc_sigma = np.asarray([float(row['sigma']) for row in mc_selected])
                if metric in ('alpha_cv', 'r2_acc'):
                    mc = np.asarray([
                        float(row[f'mc_{metric}_median']) for row in mc_selected])
                    q25 = np.asarray([
                        float(row[f'mc_{metric}_q25']) for row in mc_selected])
                    q75 = np.asarray([
                        float(row[f'mc_{metric}_q75']) for row in mc_selected])
                    yerr = np.vstack((mc - q25, q75 - mc))
                else:
                    mc = np.asarray([
                        float(row[f'mc_{metric}']) for row in mc_selected])
                    se = np.asarray([
                        float(row[f'mc_{metric}_se']) for row in mc_selected])
                    yerr = 2 * se
                ax.errorbar(mc_sigma, mc, yerr=yerr, fmt='o', color=color,
                            mec='black', mew=0.45, capsize=2, ms=4, zorder=4)
        if log_y:
            ax.set_yscale('log')
        if metric == 'r2_acc':
            ax.set_yscale('symlog', linthresh=0.1)
            ax.axhline(0, color='0.45', lw=0.7)
        if metric.startswith('r2_') or metric.startswith('slope_'):
            ax.axhline(1, color='0.5', lw=0.7, ls=':')
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.18)
        ax.set_xscale('symlog', linthresh=0.01)
        ax.set_xlim(0, max_sigma)
    for ax in axes[0]:
        add_noise_ratio_axis(ax, signal, max_sigma)
    for ax in axes[1]:
        ax.set_xlabel(r'response noise $\sigma$')
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].text(
        0.02, 0.04,
        'lines: leading DE\npoints: MC mean +/- 2 SE\n'
        r'$\alpha$ and $R^2_{acc}$: median / IQR',
        transform=axes[0, 0].transAxes, fontsize=7)
    fig.suptitle(
        'FFHQ disk teacher: ridge in linear features, accentuation in pixels')
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=2.0)
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=185, bbox_inches='tight')
    FIGURE_PATH.chmod(0o644)
    plt.close(fig)


def plot_feature_gen_acc_gap(rows: list[dict[str, object]]) -> None:
    """Plot experimental gen--acc gaps, with the leading DE as reference."""
    fig, axes = plt.subplots(2, 3, figsize=(14.8, 8.8), sharex='col')
    max_sigma = max(float(row['sigma']) for row in rows) * 1.15
    signal = max(float(row['sigma']) ** 2 / float(row['noise_signal_ratio'])
                 for row in rows if float(row['noise_signal_ratio']) > 0)
    gen_color, acc_color = 'C0', 'C1'

    for column, feature in enumerate(FEATURE_NAMES):
        selected = sorted(
            [row for row in rows if row['feature'] == feature],
            key=lambda item: float(item['sigma']))
        sigma = np.asarray([float(row['sigma']) for row in selected])
        retained = float(selected[0]['retained_signal_fraction'])
        if feature == 'top_pc_100':
            title = f'top 100 PCs ({100 * retained:.2f}% signal)'
        else:
            title = FEATURE_LABELS[feature]
        axes[0, column].set_title(title, pad=34)

        error_ax = axes[0, column]
        de_gen_error = np.asarray([
            float(row['de_gen_error_normalized']) for row in selected])
        de_acc_error = np.asarray([
            float(row['de_acc_error_normalized']) for row in selected])
        error_ax.fill_between(
            sigma, de_gen_error, de_acc_error, color='0.45', alpha=0.12,
            linewidth=0, zorder=1)
        error_ax.plot(
            sigma, de_gen_error, color=gen_color, lw=2.0, zorder=2)
        error_ax.plot(
            sigma, de_acc_error, color=acc_color, lw=2.0, zorder=2)

        r2_ax = axes[1, column]
        de_r2_gen = np.asarray([float(row['de_r2_gen']) for row in selected])
        de_r2_acc = np.asarray([float(row['de_r2_acc']) for row in selected])
        r2_ax.fill_between(
            sigma, de_r2_gen, de_r2_acc, color='0.45', alpha=0.12,
            linewidth=0, zorder=1)
        r2_ax.plot(sigma, de_r2_gen, color=gen_color, lw=2.0, zorder=2)
        r2_ax.plot(sigma, de_r2_acc, color=acc_color, lw=2.0, zorder=2)

        mc_selected = [row for row in selected
                       if 'mc_gen_error_normalized' in row]
        if mc_selected:
            mc_sigma = np.asarray([float(row['sigma']) for row in mc_selected])
            for metric, color, marker in (
                    ('gen_error_normalized', gen_color, 'o'),
                    ('acc_error_normalized', acc_color, '^')):
                mean = np.asarray([
                    float(row[f'mc_{metric}']) for row in mc_selected])
                se = np.asarray([
                    float(row[f'mc_{metric}_se']) for row in mc_selected])
                error_ax.errorbar(
                    mc_sigma, mean, yerr=2 * se, fmt=marker + ':',
                    color=color, mec='black', mew=0.45, capsize=2.5,
                    ms=5.2, lw=1.2, zorder=4)

            mc_r2_gen = np.asarray([
                float(row['mc_r2_gen']) for row in mc_selected])
            mc_r2_gen_se = np.asarray([
                float(row['mc_r2_gen_se']) for row in mc_selected])
            r2_ax.errorbar(
                mc_sigma, mc_r2_gen, yerr=2 * mc_r2_gen_se, fmt='o:',
                color=gen_color, mec='black', mew=0.45, capsize=2.5,
                ms=5.2, lw=1.2, zorder=4)

            mc_r2_acc = np.asarray([
                float(row['mc_r2_acc_median']) for row in mc_selected])
            mc_r2_acc_q25 = np.asarray([
                float(row['mc_r2_acc_q25']) for row in mc_selected])
            mc_r2_acc_q75 = np.asarray([
                float(row['mc_r2_acc_q75']) for row in mc_selected])
            r2_ax.errorbar(
                mc_sigma, mc_r2_acc,
                yerr=np.vstack((mc_r2_acc - mc_r2_acc_q25,
                                mc_r2_acc_q75 - mc_r2_acc)),
                fmt='^:', color=acc_color, mec='black', mew=0.45,
                capsize=2.5, ms=5.2, lw=1.2, zorder=4)

        error_ax.set_yscale('log')
        error_ax.set_ylim(1e-13, 2.0)
        r2_ax.axhline(1, color='0.5', lw=0.8, ls=':')
        r2_ax.axhline(0, color='0.5', lw=0.8, ls='--')
        if feature == 'whiten_full':
            r2_ax.set_yscale('symlog', linthresh=0.1)
        else:
            r2_ax.set_ylim(0.55, 1.02)

        for ax in axes[:, column]:
            ax.grid(alpha=0.18)
            ax.set_xscale('symlog', linthresh=0.01)
            ax.set_xlim(0, max_sigma)
        add_noise_ratio_axis(error_ax, signal, max_sigma)
        r2_ax.set_xlabel(r'response noise $\sigma$')

    axes[0, 0].set_ylabel(r'normalized error $E/S$')
    axes[1, 0].set_ylabel(r'$R^2$')
    legend_handles = [
        Line2D([0], [0], color=gen_color, lw=2, label='generalization'),
        Line2D([0], [0], color=acc_color, lw=2, label='accentuation'),
        Line2D([0], [0], color='0.25', lw=2, label='leading DE'),
        Line2D([0], [0], color='0.25', marker='o', ls=':', lw=1.2,
               label=r'MC mean $\pm$ 2 SE'),
        Line2D([0], [0], color='0.25', marker='^', ls=':', lw=1.2,
               label=r'MC $R^2_{acc}$ median / IQR'),
        Patch(facecolor='0.45', alpha=0.12, edgecolor='none',
              label='DE gen--acc gap'),
    ]
    fig.legend(
        handles=legend_handles, loc='lower center', ncol=3, fontsize=8.5,
        frameon=False, bbox_to_anchor=(0.5, 0.005))
    fig.suptitle(
        'FFHQ disk teacher: experimental generalization--accentuation gaps',
        y=0.995)
    fig.tight_layout(rect=(0, 0.085, 1, 0.965), h_pad=2.5, w_pad=1.2)
    GAP_FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(GAP_FIGURE_PATH, dpi=185, bbox_inches='tight')
    GAP_FIGURE_PATH.chmod(0o644)
    plt.close(fig)


def plot_cutoff_summary(rows: list[dict[str, object]], n: int) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.5), sharex=True)
    ratios = sorted({float(row['noise_signal_ratio']) for row in rows})
    cmap = plt.get_cmap('viridis')
    colors = {ratio: cmap(index / max(len(ratios) - 1, 1))
              for index, ratio in enumerate(ratios)}
    first = sorted(
        [row for row in rows if np.isclose(
            float(row['noise_signal_ratio']), ratios[0])],
        key=lambda item: float(item['feature_dim']))
    k = np.asarray([float(row['feature_dim']) for row in first])
    retained = np.asarray([float(row['retained_signal_fraction']) for row in first])
    axes[0, 0].plot(k, retained, 'o-', color='C0')
    axes[0, 0].axhline(0.99, color='0.5', lw=0.7, ls=':')
    axes[0, 0].set_ylabel('retained teacher signal')
    axes[0, 0].set_ylim(0.45, 1.005)
    panels = [
        (axes[0, 1], 'de_gen_error_normalized', r'$E_{gen}/S$', True),
        (axes[1, 0], 'de_acc_error_normalized', r'$E_{acc}/S$', True),
        (axes[1, 1], 'de_slope_acc', r'slope$_{acc}$', False),
    ]
    for ratio in ratios:
        selected = sorted(
            [row for row in rows if np.isclose(
                float(row['noise_signal_ratio']), ratio)],
            key=lambda item: float(item['feature_dim']))
        k_ratio = np.asarray([float(row['feature_dim']) for row in selected])
        for ax, key, _, _ in panels:
            values = np.asarray([float(row[key]) for row in selected])
            ax.plot(k_ratio, values, 'o-', color=colors[ratio], ms=3,
                    label=rf'$\sigma^2/S={ratio:g}$')
    for ax, _, ylabel, log_y in panels:
        if log_y:
            ax.set_yscale('log')
        ax.set_ylabel(ylabel)
    axes[1, 1].axhline(1, color='0.5', lw=0.7, ls=':')
    axes[0, 1].legend(fontsize=8)
    for ax in axes.ravel():
        ax.set_xscale('log')
        ax.axvline(n, color='0.35', lw=0.8, ls='--')
        ax.grid(alpha=0.18)
    for ax in axes[1]:
        ax.set_xlabel('number of retained top PCs')
    axes[0, 0].text(
        n * 1.08, 0.48, r'$K=n$', fontsize=8, color='0.35')
    fig.suptitle('FFHQ disk teacher: top-PC cutoff tradeoff')
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    CUTOFF_FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(CUTOFF_FIGURE_PATH, dpi=185, bbox_inches='tight')
    CUTOFF_FIGURE_PATH.chmod(0o644)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=1000)
    parser.add_argument('--n-trials', type=int, default=50)
    parser.add_argument('--benchmark-trials', type=int, default=3)
    parser.add_argument('--seed', type=int, default=20260814)
    parser.add_argument('--de-only', action='store_true')
    parser.add_argument('--plot-only', action='store_true')
    parser.add_argument('--benchmark-only', action='store_true')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--no-progress', action='store_true')
    parser.add_argument('--log-file', type=Path, default=DEFAULT_LOG_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = configure_logger(args.log_file)
    if args.plot_only:
        rows = read_rows(SUMMARY_PATH)
        plot_noise_summary(rows)
        plot_feature_gen_acc_gap(rows)
        plot_cutoff_summary(read_rows(CUTOFF_PATH), args.n)
        return
    with np.load(SPECTRUM_PATH) as data:
        eigenvalues = np.asarray(data['eigenvalues'], dtype=float)
        beta = np.asarray(data['beta_proj'], dtype=float)
    signal = float(np.sum(eigenvalues * beta ** 2))
    alphas = np.logspace(-4, 7, 221)
    # Reuse the same noise points as the dense FFHQ pixel-space comparison.
    disk_rows_path = REPO_ROOT / 'tables' / 'ffhq_disk_teacher_de_summary.csv'
    with disk_rows_path.open(newline='') as handle:
        sigmas = sorted({float(row['sigma']) for row in csv.DictReader(handle)})
    problems = {name: feature_problem(name, eigenvalues, beta)
                for name in FEATURE_NAMES}
    rows = [
        de_row(name, problems[name], sigma, args.n, alphas)
        for name in FEATURE_NAMES for sigma in sigmas]

    cutoff_rows = []
    for ratio in (0.0, 0.01, 0.1, 1.0, 10.0):
        sigma = float(np.sqrt(ratio * signal))
        for cutoff in CUTOFFS:
            name = f'top_pc_{cutoff}'
            cutoff_rows.append(
                de_row(name, feature_problem(name, eigenvalues, beta),
                       sigma, args.n, alphas))

    if not args.de_only:
        if not torch.cuda.is_available():
            raise RuntimeError('Monte Carlo validation requires a CUDA GPU')
        device = torch.device('cuda')
        logger.info('GPU: %s', torch.cuda.get_device_name(device))
        benchmark_problem = problems['whiten_full']
        c = torch.from_numpy(np.asarray(
            benchmark_problem['feature_eigenvalues'], dtype=np.float32)).to(device)
        theta = torch.from_numpy(np.asarray(
            benchmark_problem['feature_teacher'], dtype=np.float32)).to(device)
        alpha_t = torch.from_numpy(alphas.astype(np.float32)).to(device)
        generator = torch.Generator(device=device).manual_seed(args.seed)
        benchmark_sigma = float(np.sqrt(
            VALIDATION_RATIOS[0] * signal +
            float(benchmark_problem['residual_signal'])))
        start = time.perf_counter()
        for _ in range(args.benchmark_trials):
            ridge_loocv_feature_fit(
                c, theta, benchmark_sigma, alpha_t, generator, args.n)
        torch.cuda.synchronize()
        seconds_per_trial = (
            time.perf_counter() - start) / args.benchmark_trials
        total_trials = len(FEATURE_NAMES) * len(VALIDATION_RATIOS) * args.n_trials
        logger.info(
            'Benchmark %.3fs/trial; projected %.1fs (%.1fmin) for %d trials',
            seconds_per_trial, seconds_per_trial * total_trials,
            seconds_per_trial * total_trials / 60, total_trials)
        if args.benchmark_only:
            logger.info('Benchmark-only run complete; no MC cases written')
            return
        mc_rows = []
        for name in FEATURE_NAMES:
            for ratio in VALIDATION_RATIOS:
                sigma = float(np.sqrt(ratio * signal))
                mc_rows.append(run_mc_case(
                    name, problems[name], sigma, float(ratio), alphas,
                    args.n_trials, args.seed, device, args.force, logger,
                    progress=not args.no_progress))
        merge_mc(rows, mc_rows)
    write_rows(SUMMARY_PATH, rows)
    write_rows(CUTOFF_PATH, cutoff_rows)
    plot_noise_summary(rows)
    plot_feature_gen_acc_gap(rows)
    plot_cutoff_summary(cutoff_rows, args.n)
    logger.info('Summary: %s', SUMMARY_PATH)
    logger.info('Cutoff sweep: %s', CUTOFF_PATH)
    logger.info(
        'Figures: %s, %s, %s', FIGURE_PATH, GAP_FIGURE_PATH,
        CUTOFF_FIGURE_PATH)


if __name__ == '__main__':
    main()

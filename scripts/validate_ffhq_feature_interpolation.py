"""Interpolate from full whitening to a soft top-PC feature projection.

For FFHQ population eigenvalues ``s_j`` and a target cutoff ``s_K``, the main
coupled family uses

    a_j(t)^2 = s_j^{-(1-t)} [1 + (s_K / s_j)^q]^{-t},  0 <= t <= 1.

Thus t=0 is exact whitening, while t=1 is a soft top-K projection.  A power
control family omits the bracketed taper and ends at full PCA, isolating the
effect of merely reducing the whitening exponent.  Hard top-K PCA is retained
as a reference endpoint.

The full 26-point curves are deterministic equivalents.  Monte Carlo uses the
measured FFHQ spectrum, exact analytic LOOCV, and reuses each feature-design
eigendecomposition across four response-noise levels.  Raw trials and the
plot-ready summary are cached separately.
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import time
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/accentuationpredrmt-matplotlib')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp/accentuationpredrmt-xdg-cache')

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from rmt_core import SpectrumKappa, spectral_feature_de_metrics
from rmt_core.ridge_theory_lib import compute_df2
from scripts.validate_ffhq_disk_teacher import (
    SPECTRUM_PATH,
    add_noise_ratio_axis,
    configure_logger,
)
from scripts.validate_ffhq_linear_features import (
    aggregate,
    read_rows,
    simulation_metrics,
    write_rows,
)
from scripts.validate_r2_peer_review import float_tag


REPO_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = REPO_ROOT / 'tables' / 'ffhq_feature_interpolation_summary.csv'
CASE_DIR = REPO_ROOT / 'tables' / 'ffhq_feature_interpolation_cases'
FIGURE_DIR = REPO_ROOT / 'figures' / 'feature_space'
FIGURE_PATH = (
    FIGURE_DIR / 'ffhq_feature_interpolation_noise_metrics.png')
ABLATION_FIGURE_PATH = (
    FIGURE_DIR / 'ffhq_feature_interpolation_ablation.png')
PROFILE_FIGURE_PATH = (
    FIGURE_DIR / 'ffhq_feature_interpolation_profiles.png')
FACET_COLUMN_FIGURE_PATH = (
    FIGURE_DIR /
    'ffhq_feature_interpolation_gen_acc_by_feature.png')
FACET_TRANSPOSED_FIGURE_PATH = (
    FIGURE_DIR /
    'ffhq_feature_interpolation_gen_acc_transposed.png')
POWER_FACET_COLUMN_FIGURE_PATH = (
    FIGURE_DIR /
    'ffhq_power_interpolation_gen_acc_by_feature.png')
POWER_FACET_TRANSPOSED_FIGURE_PATH = (
    FIGURE_DIR /
    'ffhq_power_interpolation_gen_acc_transposed.png')
DEFAULT_LOG_PATH = (
    REPO_ROOT / 'logs' / 'ffhq_feature_interpolation_validation.log')
REFERENCE_SUMMARY_PATH = (
    REPO_ROOT / 'tables' / 'ffhq_linear_feature_summary.csv')
DISK_SUMMARY_PATH = (
    REPO_ROOT / 'tables' / 'ffhq_disk_teacher_de_summary.csv')

PATH_VALUES = np.asarray([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
MC_PATH_VALUES = np.asarray([0.4, 0.6, 0.8, 1.0])
POWER_MC_PATH_VALUES = np.asarray([0.2, 0.4, 0.6, 0.8])
VALIDATION_RATIOS = np.asarray([0.01, 0.1, 1.0, 10.0])


def feature_name(family: str, t: float) -> str:
    return f'{family}_t{int(round(100 * t)):03d}'


def feature_label(family: str, t: float) -> str:
    if family == 'coupled' and np.isclose(t, 0):
        return 'whitening'
    if family == 'coupled' and np.isclose(t, 1):
        return 'soft top-100'
    if family == 'power' and np.isclose(t, 0):
        return 'whitening'
    if family == 'power' and np.isclose(t, 1):
        return 'full PCA'
    return rf'$t={t:g}$'


def make_feature_problem(
        family: str, t: float, eigenvalues: np.ndarray,
        beta: np.ndarray, cutoff: int = 100, taper_power: float = 2.0
        ) -> dict[str, np.ndarray | float | int | str]:
    """Construct one spectral feature map and its regression geometry."""
    s = np.asarray(eigenvalues, dtype=float)
    b = np.asarray(beta, dtype=float)
    if not 0.0 <= t <= 1.0:
        raise ValueError('t must lie in [0, 1]')
    if not 1 <= cutoff <= len(s):
        raise ValueError('cutoff must lie within the spectrum')
    if taper_power <= 0:
        raise ValueError('taper_power must be positive')
    total_signal = float(np.sum(s * b ** 2))

    if family == 'hard':
        retained_s = s[:cutoff]
        retained_b = b[:cutoff]
        retained_signal = float(np.sum(retained_s * retained_b ** 2))
        return {
            'family': family,
            't': 1.0,
            'feature_dim': cutoff,
            'feature_eigenvalues': retained_s,
            'feature_teacher': retained_b,
            'backprop_weights': np.ones(cutoff),
            'residual_signal': max(total_signal - retained_signal, 0.0),
            'total_signal': total_signal,
            'retained_signal_fraction': retained_signal / total_signal,
            'whitening_power': 0.0,
            'taper_strength': 1.0,
            'cutoff': cutoff,
            'taper_power': np.inf,
            'low_high_backprop_gain_ratio': 0.0,
        }
    if family not in ('power', 'coupled'):
        raise ValueError(f'unknown family {family!r}')

    whitening_power = 1.0 - t
    gain_sq = s ** (-whitening_power)
    if family == 'coupled':
        spectral_gate = (
            1.0 + (s[cutoff - 1] / s) ** taper_power) ** (-t)
        gain_sq = gain_sq * spectral_gate
    gains = np.sqrt(gain_sq)
    feature_spectrum = gain_sq * s
    return {
        'family': family,
        't': t,
        'feature_dim': len(s),
        'feature_eigenvalues': feature_spectrum,
        'feature_teacher': b / gains,
        'backprop_weights': gain_sq,
        'residual_signal': 0.0,
        'total_signal': total_signal,
        'retained_signal_fraction': 1.0,
        'whitening_power': whitening_power,
        'taper_strength': t if family == 'coupled' else 0.0,
        'cutoff': cutoff,
        'taper_power': taper_power if family == 'coupled' else 0.0,
        'low_high_backprop_gain_ratio': float(gain_sq[-1] / gain_sq[0]),
    }


def cv_risk_coefficients(
        problem: dict[str, np.ndarray | float | int | str], n: int,
        alphas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Precompute intercept and sigma^2 coefficient of the DE-CV risk."""
    c = np.asarray(problem['feature_eigenvalues'], dtype=float)
    theta = np.asarray(problem['feature_teacher'], dtype=float)
    residual = float(problem['residual_signal'])
    n_fit = n - 1
    solver = SpectrumKappa(c, len(c) / n_fit)
    intercept = np.empty(len(alphas), dtype=float)
    noise_coefficient = np.empty(len(alphas), dtype=float)
    for index, alpha in enumerate(alphas):
        kappa = solver(float(alpha) / n_fit)
        shrink = c / (c + kappa)
        df2 = compute_df2(c, kappa)
        denominator = n_fit - df2
        c_sig = np.sum(c / (c + kappa) ** 2 * theta ** 2)
        bias_gen = np.sum(c * (1.0 - shrink) ** 2 * theta ** 2)
        noise_coefficient[index] = df2 / denominator
        intercept[index] = (
            residual + bias_gen
            + (kappa ** 2 * c_sig + residual)
            * noise_coefficient[index])
    return intercept, noise_coefficient


def de_sweep(
        name: str, problem: dict[str, np.ndarray | float | int | str],
        sigmas: np.ndarray, n: int, alphas: np.ndarray
        ) -> list[dict[str, object]]:
    """Select DE-CV alpha and evaluate final-fit metrics for all noise levels."""
    c = np.asarray(problem['feature_eigenvalues'], dtype=float)
    theta = np.asarray(problem['feature_teacher'], dtype=float)
    g = np.asarray(problem['backprop_weights'], dtype=float)
    residual = float(problem['residual_signal'])
    signal = float(problem['total_signal'])
    risk_intercept, risk_noise = cv_risk_coefficients(problem, n, alphas)
    final_solver = SpectrumKappa(c, len(c) / n)
    rows: list[dict[str, object]] = []
    for sigma in sigmas:
        risk = risk_intercept + sigma ** 2 * risk_noise
        alpha = float(alphas[int(np.argmin(risk))])
        kappa = final_solver(alpha / n)
        metrics = spectral_feature_de_metrics(
            c, theta, g, residual, signal, kappa, sigma, n)
        rows.append({
            'feature': name,
            'family': str(problem['family']),
            't': float(problem['t']),
            'feature_dim': int(problem['feature_dim']),
            'cutoff': int(problem['cutoff']),
            'taper_power': float(problem['taper_power']),
            'whitening_power': float(problem['whitening_power']),
            'taper_strength': float(problem['taper_strength']),
            'low_high_backprop_gain_ratio': float(
                problem['low_high_backprop_gain_ratio']),
            'retained_signal_fraction': float(
                problem['retained_signal_fraction']),
            'n': n,
            'sigma': float(sigma),
            'noise_signal_ratio': float(sigma ** 2 / signal),
            'de_alpha_cv': alpha,
            'de_lambda': alpha / n,
            'de_kappa': kappa,
            **{f'de_{key}': value for key, value in metrics.items()},
        })
    return rows


def mc_case_path(
        name: str, n_trials: int, seed: int, n: int,
        case_dir: Path = CASE_DIR, dataset: str = 'ffhq') -> Path:
    return case_dir / (
        f'{dataset}_feature_path_{name}_n{n}_trials{n_trials}_seed{seed}.npz')


def _ridge_fit_from_sample_eigen(
        Xc: torch.Tensor, sample_eval: torch.Tensor,
        sample_evec: torch.Tensor, one_minus_leverage: torch.Tensor,
        y: torch.Tensor, alphas: torch.Tensor) -> tuple[torch.Tensor, float]:
    yc = y - y.mean()
    response_proj = sample_evec.T @ yc
    residual_shrink = alphas[None, :] / (
        sample_eval[:, None] + alphas[None, :])
    residual_path = sample_evec @ (
        residual_shrink * response_proj[:, None])
    loo_residual = residual_path / one_minus_leverage.clamp_min(1e-8)
    selected_index = int(torch.argmin(loo_residual.square().mean(dim=0)))
    selected_alpha = alphas[selected_index]
    dual = sample_evec @ (
        response_proj / (sample_eval + selected_alpha))
    estimate = Xc.T @ dual
    return estimate, float(selected_alpha)


def run_mc_variant(
        name: str, problem: dict[str, np.ndarray | float | int | str],
        ratios: np.ndarray, alphas_np: np.ndarray, n: int, n_trials: int,
        seed: int, device: torch.device, force: bool, progress: bool,
        logger: logging.Logger, case_dir: Path = CASE_DIR,
        dataset: str = 'ffhq') -> list[dict[str, object]]:
    """Run paired-noise MC, reusing each design eigendecomposition."""
    path = mc_case_path(
        name, n_trials, seed, n, case_dir=case_dir, dataset=dataset)
    metric_names = (
        'alpha_cv', 'gen_error_normalized', 'r2_gen', 'slope_gen',
        'acc_error_normalized', 'r2_acc', 'slope_acc')
    if path.exists() and not force:
        with np.load(path) as data:
            trial_arrays = {
                metric: np.asarray(data[f'trial_{metric}'])
                for metric in metric_names}
            elapsed = float(data['elapsed_seconds'])
    else:
        c = torch.from_numpy(np.asarray(
            problem['feature_eigenvalues'], dtype=np.float32)).to(device)
        theta = torch.from_numpy(np.asarray(
            problem['feature_teacher'], dtype=np.float32)).to(device)
        g = torch.from_numpy(np.asarray(
            problem['backprop_weights'], dtype=np.float32)).to(device)
        residual = float(problem['residual_signal'])
        signal = float(problem['total_signal'])
        alphas = torch.from_numpy(alphas_np.astype(np.float32)).to(device)
        sigmas_effective = np.sqrt(ratios * signal + residual)
        trial_arrays = {
            metric: np.empty((len(ratios), n_trials), dtype=float)
            for metric in metric_names}
        generator = torch.Generator(device=device).manual_seed(seed)
        iterator = range(n_trials)
        if progress:
            try:
                from tqdm.auto import tqdm
                iterator = tqdm(iterator, desc=name, unit='design')
            except ImportError:
                pass
        start = time.perf_counter()
        for trial in iterator:
            X = torch.randn(
                (n, len(c)), device=device, generator=generator)
            X.mul_(torch.sqrt(c)[None, :])
            Xc = X - X.mean(dim=0, keepdim=True)
            gram = Xc @ Xc.T
            sample_eval, sample_evec = torch.linalg.eigh(gram)
            sample_eval.clamp_min_(0)
            residual_shrink = alphas[None, :] / (
                sample_eval[:, None] + alphas[None, :])
            one_minus_leverage = (
                sample_evec.square() @ residual_shrink - 1.0 / n)
            noiseless_response = X @ theta
            for ratio_index, sigma_effective in enumerate(sigmas_effective):
                y = noiseless_response + torch.randn(
                    n, device=device, generator=generator) * sigma_effective
                estimate, alpha = _ridge_fit_from_sample_eigen(
                    Xc, sample_eval, sample_evec, one_minus_leverage,
                    y, alphas)
                metrics = simulation_metrics(
                    estimate, theta, c, g, residual, signal)
                trial_arrays['alpha_cv'][ratio_index, trial] = alpha
                for metric, value in metrics.items():
                    trial_arrays[metric][ratio_index, trial] = value
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, ratios=ratios, elapsed_seconds=elapsed,
            **{f'trial_{key}': value
               for key, value in trial_arrays.items()})
        path.chmod(0o644)
        logger.info('%s: %d paired designs in %.1fs', name, n_trials, elapsed)

    rows: list[dict[str, object]] = []
    for ratio_index, ratio in enumerate(ratios):
        row: dict[str, object] = {
            'feature': name,
            'noise_signal_ratio': float(ratio),
            'n_trials': n_trials,
            'mc_elapsed_seconds': elapsed,
            'case_path': str(path),
        }
        for metric, array in trial_arrays.items():
            values = array[ratio_index]
            mean, std, se = aggregate(values)
            row[f'mc_{metric}'] = mean
            row[f'mc_{metric}_std'] = std
            row[f'mc_{metric}_se'] = se
            row[f'mc_{metric}_median'] = float(np.median(values))
            row[f'mc_{metric}_q25'] = float(np.quantile(values, 0.25))
            row[f'mc_{metric}_q75'] = float(np.quantile(values, 0.75))
        rows.append(row)
    return rows


def merge_mc_rows(
        rows: list[dict[str, object]], mc_rows: list[dict[str, object]]) -> None:
    for mc in mc_rows:
        candidates = [
            row for row in rows
            if row['feature'] == mc['feature']
            and np.isclose(
                float(row['noise_signal_ratio']),
                float(mc['noise_signal_ratio']))]
        if len(candidates) != 1:
            raise RuntimeError(f'could not merge MC row {mc}')
        candidates[0].update(mc)


def merge_reference_mc(rows: list[dict[str, object]]) -> None:
    """Reuse the 50-trial endpoint validations from the original experiment."""
    references = read_rows(REFERENCE_SUMMARY_PATH)
    mapping = {
        feature_name('coupled', 0.0): 'whiten_full',
        feature_name('power', 0.0): 'whiten_full',
        feature_name('power', 1.0): 'pca_full',
        'hard_top100': 'top_pc_100',
    }
    for destination, source in mapping.items():
        for reference in references:
            if reference['feature'] != source or 'mc_r2_acc' not in reference:
                continue
            candidates = [
                row for row in rows
                if row['feature'] == destination
                and np.isclose(
                    float(row['noise_signal_ratio']),
                    float(reference['noise_signal_ratio']))]
            if len(candidates) != 1:
                raise RuntimeError(
                    f'could not merge reference {source} into {destination}')
            for key, value in reference.items():
                if key.startswith('mc_') or key in (
                        'n_trials', 'case_path', 'mc_elapsed_seconds'):
                    candidates[0][key] = value


def _mc_error(
        selected: list[dict[str, object]], metric: str
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    mc_rows = [row for row in selected if f'mc_{metric}' in row]
    if not mc_rows:
        return None
    sigma = np.asarray([float(row['sigma']) for row in mc_rows])
    if metric in ('r2_acc', 'alpha_cv'):
        center = np.asarray([
            float(row[f'mc_{metric}_median']) for row in mc_rows])
        q25 = np.asarray([
            float(row[f'mc_{metric}_q25']) for row in mc_rows])
        q75 = np.asarray([
            float(row[f'mc_{metric}_q75']) for row in mc_rows])
        error = np.vstack((center - q25, q75 - center))
    else:
        center = np.asarray([float(row[f'mc_{metric}']) for row in mc_rows])
        error = 2.0 * np.asarray([
            float(row[f'mc_{metric}_se']) for row in mc_rows])
    return sigma, center, error


def plot_noise_metrics(rows: list[dict[str, object]]) -> None:
    """Plot the coupled whitening-to-soft-top-PC family across noise."""
    fig, axes = plt.subplots(2, 3, figsize=(14.8, 8.8), sharex=True)
    specs = (
        ('gen_error_normalized', r'$E_{\rm gen}/S$', 'log'),
        ('acc_error_normalized', r'$E_{\rm acc}/S$', 'log'),
        ('r2_gen', r'$R^2_{\rm gen}$', 'linear'),
        ('r2_acc', r'$R^2_{\rm acc}$', 'symlog'),
        ('slope_gen', r'slope$_{\rm gen}$', 'linear'),
        ('slope_acc', r'slope$_{\rm acc}$', 'log'),
    )
    cmap = plt.get_cmap('viridis')
    colors = {
        float(t): cmap(0.08 + 0.84 * index / (len(PATH_VALUES) - 1))
        for index, t in enumerate(PATH_VALUES)}
    max_sigma = max(float(row['sigma']) for row in rows) * 1.15
    signal = max(
        float(row['sigma']) ** 2 / float(row['noise_signal_ratio'])
        for row in rows if float(row['noise_signal_ratio']) > 0)

    for ax, (metric, ylabel, scale) in zip(axes.ravel(), specs):
        for t in PATH_VALUES:
            name = feature_name('coupled', float(t))
            selected = sorted(
                [row for row in rows if row['feature'] == name],
                key=lambda row: float(row['sigma']))
            sigma = np.asarray([float(row['sigma']) for row in selected])
            values = np.asarray([float(row[f'de_{metric}']) for row in selected])
            color = colors[float(t)]
            ax.plot(
                sigma, values, color=color, lw=2.0,
                label=feature_label('coupled', float(t)))
            mc = _mc_error(selected, metric)
            if mc is not None:
                mc_sigma, center, error = mc
                ax.errorbar(
                    mc_sigma, center, yerr=error, fmt='o', ls='none',
                    color=color, mec='black', mew=0.4, ms=4.5,
                    capsize=2.0, alpha=0.8, zorder=4)
        hard = sorted(
            [row for row in rows if row['feature'] == 'hard_top100'],
            key=lambda row: float(row['sigma']))
        sigma_hard = np.asarray([float(row['sigma']) for row in hard])
        ax.plot(
            sigma_hard, [float(row[f'de_{metric}']) for row in hard],
            color='0.15', lw=1.8, ls='--', label='hard top-100')
        hard_mc = _mc_error(hard, metric)
        if hard_mc is not None:
            mc_sigma, center, error = hard_mc
            ax.errorbar(
                mc_sigma, center, yerr=error, fmt='s', ls='none',
                color='0.15', mec='black', mew=0.4, ms=4.2,
                capsize=2.0, alpha=0.75, zorder=4)

        if scale == 'log':
            ax.set_yscale('log')
        elif scale == 'symlog':
            ax.set_yscale('symlog', linthresh=0.1)
            ax.axhline(0, color='0.5', lw=0.7, ls='--')
        if metric.startswith('r2_') or metric.startswith('slope_'):
            ax.axhline(1, color='0.5', lw=0.7, ls=':')
        ax.set_ylabel(ylabel)
        ax.set_xscale('symlog', linthresh=0.01)
        ax.set_xlim(0, max_sigma)
        ax.grid(alpha=0.16)
    for ax in axes[0]:
        add_noise_ratio_axis(ax, signal, max_sigma)
    for ax in axes[1]:
        ax.set_xlabel(r'response noise $\sigma$')
    axes[0, 0].legend(fontsize=8, ncol=2, loc='best')
    fig.suptitle(
        'FFHQ disk teacher: whitening-to-soft-top-100 feature family',
        y=0.995, fontsize=14.5)
    fig.text(
        0.5, 0.012,
        r'lines: DE-selected RidgeCV; circles: Gaussian-spectrum MC '
        r'(mean $\pm2$ SE; $R^2_{acc}$ median/IQR); '
        r'$a_j(t)^2=s_j^{-(1-t)}[1+(s_{100}/s_j)^2]^{-t}$',
        ha='center', fontsize=8.5, color='0.3')
    fig.tight_layout(rect=(0, 0.055, 1, 0.96), h_pad=2.2, w_pad=1.5)
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=190, bbox_inches='tight')
    FIGURE_PATH.chmod(0o644)
    plt.close(fig)


def plot_ablation(rows: list[dict[str, object]]) -> None:
    """Compare reduced whitening alone with the coupled low-PC taper."""
    fig, axes = plt.subplots(3, 2, figsize=(12.8, 10.2), sharex='col')
    specs = (
        ('acc_error_normalized', r'$E_{\rm acc}/S$', 'log'),
        ('r2_acc', r'$R^2_{\rm acc}$', 'symlog'),
        ('slope_acc', r'slope$_{\rm acc}$', 'log'),
    )
    cmap = plt.get_cmap('viridis')
    colors = {
        float(t): cmap(0.08 + 0.84 * index / (len(PATH_VALUES) - 1))
        for index, t in enumerate(PATH_VALUES)}
    max_sigma = max(float(row['sigma']) for row in rows) * 1.15
    signal = max(
        float(row['sigma']) ** 2 / float(row['noise_signal_ratio'])
        for row in rows if float(row['noise_signal_ratio']) > 0)
    titles = {
        'power': r'de-whitening only: $a_j^2=s_j^{-(1-t)}$',
        'coupled': r'de-whitening + low-PC taper',
    }
    for column, family in enumerate(('power', 'coupled')):
        axes[0, column].set_title(titles[family], pad=32)
        for row_index, (metric, ylabel, scale) in enumerate(specs):
            ax = axes[row_index, column]
            for t in PATH_VALUES:
                name = feature_name(family, float(t))
                selected = sorted(
                    [row for row in rows if row['feature'] == name],
                    key=lambda row: float(row['sigma']))
                sigma = np.asarray([float(row['sigma']) for row in selected])
                values = np.asarray([
                    float(row[f'de_{metric}']) for row in selected])
                ax.plot(
                    sigma, values, color=colors[float(t)], lw=2.0,
                    label=feature_label(family, float(t)))
                mc = _mc_error(selected, metric)
                if mc is not None:
                    mc_sigma, center, error = mc
                    ax.errorbar(
                        mc_sigma, center, yerr=error, fmt='o', ls='none',
                        color=colors[float(t)], mec='black', mew=0.35,
                        ms=4.0, capsize=1.8, alpha=0.75, zorder=4)
            if family == 'coupled':
                hard = sorted(
                    [row for row in rows if row['feature'] == 'hard_top100'],
                    key=lambda row: float(row['sigma']))
                ax.plot(
                    [float(row['sigma']) for row in hard],
                    [float(row[f'de_{metric}']) for row in hard],
                    color='0.15', lw=1.7, ls='--', label='hard top-100')
            if scale == 'log':
                ax.set_yscale('log')
            else:
                ax.set_yscale('symlog', linthresh=0.1)
                ax.axhline(0, color='0.5', lw=0.7, ls='--')
            if metric.startswith('slope_'):
                ax.axhline(1, color='0.5', lw=0.7, ls=':')
            ax.set_ylabel(ylabel if column == 0 else '')
            ax.set_xscale('symlog', linthresh=0.01)
            ax.set_xlim(0, max_sigma)
            ax.grid(alpha=0.16)
        add_noise_ratio_axis(axes[0, column], signal, max_sigma)
        axes[2, column].set_xlabel(r'response noise $\sigma$')
    axes[0, 0].legend(fontsize=8, ncol=2)
    axes[0, 1].legend(fontsize=8, ncol=2)
    fig.suptitle(
        'FFHQ disk teacher: low-variance suppression, not de-whitening alone, '
        'restores accentuation', y=0.995, fontsize=14)
    fig.text(
        0.5, 0.012,
        'Same whitening exponent along both columns; the right column also '
        'attenuates modes below the 100th-PC eigenvalue.',
        ha='center', fontsize=8.5, color='0.3')
    fig.tight_layout(rect=(0, 0.045, 1, 0.96), h_pad=1.8, w_pad=1.4)
    ABLATION_FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ABLATION_FIGURE_PATH, dpi=190, bbox_inches='tight')
    ABLATION_FIGURE_PATH.chmod(0o644)
    plt.close(fig)


def plot_feature_profiles(
        eigenvalues: np.ndarray, cutoff: int = 100,
        taper_power: float = 2.0, family: str = 'coupled',
        dataset_title: str = 'FFHQ',
        output_path: Path = PROFILE_FIGURE_PATH) -> None:
    """Show how the interpolating feature maps reshape the PC spectrum."""
    if family not in ('coupled', 'power'):
        raise ValueError(f'unknown profile family {family!r}')
    s = np.asarray(eigenvalues, dtype=float)
    ranks = np.arange(1, len(s) + 1)
    cmap = plt.get_cmap('viridis')
    colors = {
        float(t): cmap(0.08 + 0.84 * index / (len(PATH_VALUES) - 1))
        for index, t in enumerate(PATH_VALUES)}
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.2), sharex=True)
    for t in PATH_VALUES:
        whitening_power = 1.0 - float(t)
        gain_sq = s ** (-whitening_power)
        if family == 'coupled':
            gate = (
                1.0 + (s[cutoff - 1] / s) ** taper_power) ** (-float(t))
            gain_sq *= gate
        feature_variance = gain_sq * s
        axes[0].plot(
            ranks, gain_sq / gain_sq[0], color=colors[float(t)], lw=2.0,
            label=feature_label(family, float(t)))
        axes[1].plot(
            ranks, feature_variance / feature_variance[0],
            color=colors[float(t)], lw=2.0)

    if family == 'coupled':
        hard_rank = ranks[:cutoff]
        axes[0].plot(
            hard_rank, np.ones(cutoff), color='0.15', lw=1.8, ls='--',
            label='hard top-100')
        axes[1].plot(
            hard_rank, s[:cutoff] / s[0], color='0.15', lw=1.8, ls='--')
    for ax in axes:
        if family == 'coupled':
            ax.axvline(cutoff, color='0.45', lw=0.8, ls=':')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('PC rank (low variance / high frequency to the right)')
        ax.grid(alpha=0.16)
    axes[0].set_ylabel(r'input backprop gain $a_j^2/a_1^2$')
    axes[1].set_ylabel(r'feature variance $a_j^2s_j/(a_1^2s_1)$')
    axes[0].set_title('Gain applied to input PCs')
    axes[1].set_title('Spectrum seen by ridge regression')
    axes[0].legend(fontsize=8, ncol=2, loc='best')
    family_title = (
        'whitening to soft top-100 projection' if family == 'coupled' else
        'whitening to full PCA (no PC taper)')
    fig.suptitle(
        f'{dataset_title} feature-space family: {family_title}',
        y=0.995, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94), w_pad=1.8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=190, bbox_inches='tight')
    output_path.chmod(0o644)
    plt.close(fig)


def _compact_noise_ratio_axis(
        ax: plt.Axes, signal: float, max_sigma: float,
        show_label: bool) -> None:
    ratio_ticks = np.asarray([0.0, 1e-4, 1e-2, 1.0, 10.0])
    sigma_ticks = np.sqrt(ratio_ticks * signal)
    visible = sigma_ticks <= max_sigma * 1.001
    ratio_ticks = ratio_ticks[visible]
    sigma_ticks = sigma_ticks[visible]
    labels = [
        '0' if value == 0 else
        rf'$10^{{{int(round(np.log10(value)))}}}$'
        for value in ratio_ticks]
    top = ax.twiny()
    top.set_xscale('symlog', linthresh=0.01)
    top.set_xlim(ax.get_xlim())
    top.set_xticks(sigma_ticks)
    top.set_xticklabels(labels)
    top.tick_params(axis='x', labelsize=6.5, pad=1)
    if show_label:
        top.set_xlabel(
            r'noise variance / signal variance  $\sigma^2/S$', fontsize=8)


def plot_gen_acc_facets(
        rows: list[dict[str, object]], transpose: bool = False,
        family: str = 'coupled', dataset_title: str = 'FFHQ disk teacher',
        output_path: Path | None = None) -> None:
    """Overlay generation and accentuation within every feature-map facet."""
    if family not in ('coupled', 'power'):
        raise ValueError(f'unknown facet family {family!r}')
    features = [
        (feature_name(family, float(t)),
         feature_label(family, float(t)))
        for t in PATH_VALUES]
    if family == 'coupled':
        features.append(('hard_top100', 'hard top-100'))
    metrics = (
        ('normalized error', 'gen_error_normalized',
         'acc_error_normalized', r'$E/S$', 'log'),
        (r'$R^2$', 'r2_gen', 'r2_acc', r'$R^2$', 'symlog'),
        ('fitted slope', 'slope_gen', 'slope_acc', 'slope', 'log'),
    )
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    gen_color, acc_color = colors[0], colors[1]
    if transpose:
        nrows, ncols = len(features), len(metrics)
        figsize = (13.8, 20.0 if family == 'coupled' else 17.2)
        if output_path is None:
            output_path = (
                FACET_TRANSPOSED_FIGURE_PATH if family == 'coupled' else
                POWER_FACET_TRANSPOSED_FIGURE_PATH)
    else:
        nrows, ncols = len(metrics), len(features)
        figsize = (22.0 if family == 'coupled' else 19.0, 10.2)
        if output_path is None:
            output_path = (
                FACET_COLUMN_FIGURE_PATH if family == 'coupled' else
                POWER_FACET_COLUMN_FIGURE_PATH)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=figsize, sharex=True, sharey=False,
        squeeze=False)
    signal = max(
        float(row['sigma']) ** 2 / float(row['noise_signal_ratio'])
        for row in rows if float(row['noise_signal_ratio']) > 0)
    max_sigma = max(float(row['sigma']) for row in rows) * 1.15

    for feature_index, (name, label) in enumerate(features):
        selected = sorted(
            [row for row in rows if row['feature'] == name],
            key=lambda row: float(row['sigma']))
        sigma = np.asarray([float(row['sigma']) for row in selected])
        for metric_index, (
                metric_title, gen_metric, acc_metric, ylabel,
                scale) in enumerate(metrics):
            row_index, column_index = (
                (feature_index, metric_index) if transpose else
                (metric_index, feature_index))
            ax = axes[row_index, column_index]
            gen = np.asarray([
                float(row[f'de_{gen_metric}']) for row in selected])
            acc = np.asarray([
                float(row[f'de_{acc_metric}']) for row in selected])
            ax.fill_between(
                sigma, gen, acc, where=gen >= acc, interpolate=True,
                color=gen_color, alpha=0.12, linewidth=0)
            ax.fill_between(
                sigma, gen, acc, where=acc > gen, interpolate=True,
                color=acc_color, alpha=0.12, linewidth=0)
            ax.plot(sigma, gen, color=gen_color, lw=1.8)
            ax.plot(sigma, acc, color=acc_color, lw=1.8)
            for metric, color, marker in (
                    (gen_metric, gen_color, 'o'),
                    (acc_metric, acc_color, '^')):
                mc = _mc_error(selected, metric)
                if mc is None:
                    continue
                mc_sigma, center, _ = mc
                ax.plot(
                    mc_sigma, center, marker=marker, ls='none',
                    color=color, mec='0.15', mew=0.35, ms=3.4,
                    alpha=0.78, zorder=4)
            if scale == 'log':
                ax.set_yscale('log')
            else:
                ax.set_yscale('symlog', linthresh=0.1)
                ax.axhline(0, color='0.5', lw=0.65, ls='--')
            if gen_metric.startswith('r2_') or gen_metric.startswith('slope_'):
                ax.axhline(1, color='0.5', lw=0.65, ls=':')
            ax.set_xscale('symlog', linthresh=0.01)
            ax.set_xlim(0, max_sigma)
            ax.grid(alpha=0.14)

            if not transpose:
                if metric_index == 0:
                    ax.set_title(label, fontsize=10, pad=27)
                if feature_index == 0:
                    ax.set_ylabel(ylabel)
            else:
                if feature_index == 0:
                    ax.set_title(metric_title, fontsize=11, pad=27)
                if metric_index == 0:
                    ax.set_ylabel(label, fontsize=9, labelpad=10)

    for column_index in range(ncols):
        _compact_noise_ratio_axis(
            axes[0, column_index], signal, max_sigma,
            show_label=(column_index == ncols // 2))
        axes[-1, column_index].set_xlabel(r'response noise $\sigma$')

    handles = (
        Line2D([], [], color=gen_color, lw=2, marker='o', ms=4,
               label='generalization (DE line; MC circle)'),
        Line2D([], [], color=acc_color, lw=2, marker='^', ms=4,
               label='accentuation (DE line; MC triangle)'),
        Patch(facecolor=gen_color, alpha=0.16,
              label='gap: generalization is larger'),
        Patch(facecolor=acc_color, alpha=0.16,
              label='gap: accentuation is larger'),
    )
    fig.legend(
        handles=handles, loc='lower center', ncol=4, frameon=False,
        fontsize=9, bbox_to_anchor=(0.5, 0.012))
    layout = 'transposed layout' if transpose else 'feature maps as columns'
    family_title = (
        'whitening to soft top-100' if family == 'coupled' else
        'whitening to full PCA (no PC taper)')
    fig.suptitle(
        f'{dataset_title}: {family_title} gen--acc gaps ({layout})',
        y=0.997, fontsize=15)
    fig.tight_layout(rect=(0, 0.045, 1, 0.965), h_pad=1.6, w_pad=0.9)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches='tight')
    output_path.chmod(0o644)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=1000)
    parser.add_argument('--n-trials', type=int, default=25)
    parser.add_argument('--seed', type=int, default=20260827)
    parser.add_argument('--cutoff', type=int, default=100)
    parser.add_argument('--taper-power', type=float, default=2.0)
    parser.add_argument('--alpha-min-exp', type=float, default=-6.0)
    parser.add_argument('--alpha-max-exp', type=float, default=9.0)
    parser.add_argument('--alpha-grid-size', type=int, default=301)
    parser.add_argument('--cpu-threads', type=int, default=8)
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
        with np.load(SPECTRUM_PATH) as data:
            eigenvalues = np.asarray(data['eigenvalues'], dtype=float)
        plot_noise_metrics(rows)
        plot_ablation(rows)
        plot_feature_profiles(eigenvalues, args.cutoff, args.taper_power)
        plot_gen_acc_facets(rows, transpose=False)
        plot_gen_acc_facets(rows, transpose=True)
        plot_gen_acc_facets(rows, transpose=False, family='power')
        plot_gen_acc_facets(rows, transpose=True, family='power')
        logger.info(
            'Replotted %s, %s, %s, %s, %s, %s, and %s', FIGURE_PATH,
            ABLATION_FIGURE_PATH, PROFILE_FIGURE_PATH,
            FACET_COLUMN_FIGURE_PATH, FACET_TRANSPOSED_FIGURE_PATH,
            POWER_FACET_COLUMN_FIGURE_PATH,
            POWER_FACET_TRANSPOSED_FIGURE_PATH)
        return

    with np.load(SPECTRUM_PATH) as data:
        eigenvalues = np.asarray(data['eigenvalues'], dtype=float)
        beta = np.asarray(data['beta_proj'], dtype=float)
    signal = float(np.sum(eigenvalues * beta ** 2))
    with DISK_SUMMARY_PATH.open(newline='') as handle:
        sigmas = np.asarray(sorted({
            float(row['sigma']) for row in csv.DictReader(handle)}))
    alphas = np.logspace(
        args.alpha_min_exp, args.alpha_max_exp, args.alpha_grid_size)
    problems: dict[str, dict[str, np.ndarray | float | int | str]] = {}
    for family in ('power', 'coupled'):
        for t in PATH_VALUES:
            name = feature_name(family, float(t))
            problems[name] = make_feature_problem(
                family, float(t), eigenvalues, beta,
                args.cutoff, args.taper_power)
    problems['hard_top100'] = make_feature_problem(
        'hard', 1.0, eigenvalues, beta, args.cutoff, args.taper_power)

    de_start = time.perf_counter()
    rows: list[dict[str, object]] = []
    for index, (name, problem) in enumerate(problems.items(), start=1):
        logger.info('[DE %d/%d] %s', index, len(problems), name)
        rows.extend(de_sweep(name, problem, sigmas, args.n, alphas))
    logger.info('DE family sweep completed in %.1fs', time.perf_counter() - de_start)
    merge_reference_mc(rows)

    if not args.de_only:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device.type == 'cpu':
            torch.set_num_threads(args.cpu_threads)
        logger.info('Monte Carlo device: %s%s', device,
                    (f' ({torch.cuda.get_device_name(device)})'
                     if device.type == 'cuda' else
                     f' with {torch.get_num_threads()} threads'))
        benchmark_name = feature_name('power', 0.6)
        benchmark_start = time.perf_counter()
        benchmark_rows = run_mc_variant(
            benchmark_name, problems[benchmark_name], VALIDATION_RATIOS,
            alphas, args.n, 1, args.seed + 991, device, True, False, logger)
        benchmark_seconds = time.perf_counter() - benchmark_start
        projected = (
            benchmark_seconds * len(POWER_MC_PATH_VALUES) * args.n_trials)
        logger.info(
            'Small-scale benchmark %.2fs/design; projected %.1fs (%.1fmin) '
            'for %d intermediate designs; each serves %d noise levels',
            benchmark_seconds, projected, projected / 60,
            len(POWER_MC_PATH_VALUES) * args.n_trials,
            len(VALIDATION_RATIOS))
        benchmark_path = Path(str(benchmark_rows[0]['case_path']))
        if benchmark_path.exists():
            benchmark_path.unlink()
        if args.benchmark_only:
            return
        mc_rows: list[dict[str, object]] = []
        for family, path_values in (
                ('coupled', MC_PATH_VALUES),
                ('power', POWER_MC_PATH_VALUES)):
            for t in path_values:
                name = feature_name(family, float(t))
                mc_rows.extend(run_mc_variant(
                    name, problems[name], VALIDATION_RATIOS, alphas,
                    args.n, args.n_trials, args.seed, device, args.force,
                    not args.no_progress, logger))
        merge_mc_rows(rows, mc_rows)

    rows.sort(key=lambda row: (
        str(row['family']), float(row['t']), float(row['sigma'])))
    write_rows(SUMMARY_PATH, rows)
    plot_noise_metrics(rows)
    plot_ablation(rows)
    plot_feature_profiles(eigenvalues, args.cutoff, args.taper_power)
    plot_gen_acc_facets(rows, transpose=False)
    plot_gen_acc_facets(rows, transpose=True)
    plot_gen_acc_facets(rows, transpose=False, family='power')
    plot_gen_acc_facets(rows, transpose=True, family='power')
    logger.info('Plot-ready summary: %s', SUMMARY_PATH)
    logger.info('Raw MC cases: %s', CASE_DIR)
    logger.info(
        'Figures: %s, %s, %s, %s, %s, %s, and %s', FIGURE_PATH,
        ABLATION_FIGURE_PATH, PROFILE_FIGURE_PATH,
        FACET_COLUMN_FIGURE_PATH, FACET_TRANSPOSED_FIGURE_PATH,
        POWER_FACET_COLUMN_FIGURE_PATH,
        POWER_FACET_TRANSPOSED_FIGURE_PATH)


if __name__ == '__main__':
    main()

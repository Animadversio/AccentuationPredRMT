"""Explore finite-sample ridge paths inside prediction/control geometry.

The figure keeps one Gaussian design matrix fixed and compares:

1. a noise-scale path at fixed ridge penalty, coupling all sigma values through
   one standardized response-noise realization; and
2. a regularization path at fixed noise scale, sweeping lambda and displaying
   the corresponding population-spectrum effective regularizer kappa.

For Gaussian response noise and fixed X, the ridge estimator is exactly
Gaussian. Analytic probability ellipses show how its mean and covariance
expand with sigma and translate, contract, and change aspect ratio with
lambda. Optional Monte Carlo points verify the analytic geometry, while one
coupled noise realization traces a highlighted sample path. All numerical
points are cached in plot-ready CSV files.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/accentuationpredrmt-matplotlib")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, Ellipse


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rmt_core import SpectrumKappa
from rmt_core.simulation_lib import ridge_estimator, ridge_path_estimators
from scripts.plot_iso_error_geometry import (
    add_control_contours,
    add_covariance_ellipse,
    add_prediction_contours,
    add_teacher,
    format_axis,
    geometry_limits,
)


DEFAULT_FIGURE = (
    REPO_ROOT / "figures" / "geometry" / "ridge_paths_iso_error_geometry.png")
DEFAULT_PDF = (
    REPO_ROOT / "figures" / "geometry" / "ridge_paths_iso_error_geometry.pdf")
DEFAULT_TABLE = REPO_ROOT / "tables" / "ridge_paths_iso_error_geometry.csv"
DEFAULT_GAUSSIAN_TABLE = (
    REPO_ROOT / "tables" / "ridge_gaussian_ellipses.csv")

PATH_COLOR = "#5B2A86"
MEAN_COLOR = "0.28"
MARKERS = ("o", "s", "^", "v", "P")
LINESTYLES = (
    (0, (1.5, 1.5)), (0, (5, 2)), (0, (7, 2, 1.5, 2)),
    (0, (2, 1, 6, 1)), "solid",
)


@dataclass(frozen=True)
class PlotElements:
    """Independent switches for every visual layer in the figure."""

    data_ellipse: bool = True
    prediction_contours: bool = True
    control_contours: bool = True
    teacher: bool = True
    realization_paths: bool = True
    cloud_points: bool = False
    gaussian_ellipses: bool = True
    covariance_crossover: bool = True
    stationary_highlights: bool = True
    mean_paths: bool = True
    point_labels: bool = True
    arrows: bool = True
    panel_notes: bool = True
    legend: bool = True


@dataclass(frozen=True)
class PlotStyle:
    """Layout and styling parameters intended for notebook experimentation."""

    figsize: tuple[float, float] = (13.1, 6.45)
    lambda_only_figsize: tuple[float, float] = (8.2, 7.0)
    path_color: str = PATH_COLOR
    mean_color: str = MEAN_COLOR
    geometry_alpha: float = 0.22
    geometry_linewidth_scale: float = 0.52
    cloud_point_alpha: float = 0.10
    cloud_point_size: float = 11.0
    gaussian_coverage: float = 0.68
    gaussian_fill_alpha: float = 0.085
    gaussian_linewidth: float = 2.1
    path_linewidth: float = 2.7
    legend_columns: int = 3


def ridge_map(X: np.ndarray, lam: float) -> np.ndarray:
    """Return the linear map y -> beta_hat for one ridge penalty."""
    n, d = X.shape
    return np.linalg.solve(X.T @ X + n * lam * np.eye(d), X.T)


def ridge_gaussian_moments(
        X: np.ndarray, beta_star: np.ndarray, lam: float,
        sigma2_noise: float) -> tuple[np.ndarray, np.ndarray]:
    r"""Return exact conditional moments of ridge under Gaussian noise.

    If ``y = X beta_star + xi`` with ``xi ~ N(0, sigma2_noise I)``, then
    conditional on X,

        beta_hat | X ~ N(A X beta_star, sigma2_noise A A^T),

    where ``A = (X^T X + n lambda I)^(-1) X^T``.
    """
    mapping = ridge_map(X, lam)
    mean = mapping @ (X @ beta_star)
    covariance = sigma2_noise * (mapping @ mapping.T)
    return mean, covariance


def estimator_metrics(
        beta_hat: np.ndarray, beta_star: np.ndarray,
        sigma_matrix: np.ndarray) -> dict[str, float]:
    delta = beta_hat - beta_star
    e_gen = float(delta @ sigma_matrix @ delta)
    signal_power = float(beta_star @ sigma_matrix @ beta_star)
    denominator = float(beta_hat @ beta_star)
    norm_sq = float(beta_hat @ beta_hat)
    if abs(denominator) < 1e-14:
        z = np.nan
        e_acc = np.nan
        r2_acc = np.nan
    else:
        z = norm_sq / denominator - 1.0
        e_acc = (z / (1.0 + z)) ** 2 if not np.isclose(z, -1.0) else np.inf
        r2_acc = 1.0 - z ** 2
    return {
        "E_gen": e_gen,
        "R2_gen": 1.0 - e_gen / signal_power,
        "z": z,
        "E_acc_over_S": e_acc,
        "R2_acc": r2_acc,
    }


def add_gaussian_ellipse(
        ax: plt.Axes, mean: np.ndarray, covariance: np.ndarray, color,
        linestyle, coverage: float, fill_alpha: float,
        linewidth: float) -> np.ndarray:
    """Draw an exact 2D Gaussian probability ellipse.

    For two dimensions, the squared Mahalanobis radius is chi-square with two
    degrees of freedom, whose quantile is ``-2 log(1 - coverage)``.
    """
    if not 0.0 < coverage < 1.0:
        raise ValueError("Gaussian ellipse coverage must lie between 0 and 1")
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    mahalanobis_radius = np.sqrt(-2.0 * np.log1p(-coverage))
    semiaxes = mahalanobis_radius * np.sqrt(eigenvalues)
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    ax.add_patch(Ellipse(
        tuple(mean), width=2.0 * semiaxes[0], height=2.0 * semiaxes[1],
        angle=angle, facecolor=color, edgecolor="none", linewidth=0,
        alpha=fill_alpha, zorder=4,
    ))
    ax.add_patch(Ellipse(
        tuple(mean), width=2.0 * semiaxes[0], height=2.0 * semiaxes[1],
        angle=angle, fill=False, edgecolor=color, linewidth=linewidth,
        linestyle=linestyle, alpha=0.92, zorder=6,
    ))
    return semiaxes


def add_arrow_on_path(
        ax: plt.Axes, path: np.ndarray, start_index: int,
        end_index: int, color: str = PATH_COLOR) -> None:
    ax.annotate(
        "", xy=path[end_index], xytext=path[start_index],
        arrowprops={"arrowstyle": "-|>", "color": color,
                    "linewidth": 2.0, "mutation_scale": 13},
        zorder=10,
    )


def golden_section_minimize_log(
        function, lower: float, upper: float,
        iterations: int = 80) -> float:
    """Minimize a scalar function over a positive interval in log space."""
    log_lower, log_upper = np.log(lower), np.log(upper)
    ratio = (np.sqrt(5.0) - 1.0) / 2.0
    left = log_upper - ratio * (log_upper - log_lower)
    right = log_lower + ratio * (log_upper - log_lower)
    left_value = function(float(np.exp(left)))
    right_value = function(float(np.exp(right)))
    for _ in range(iterations):
        if left_value <= right_value:
            log_upper = right
            right, right_value = left, left_value
            left = log_upper - ratio * (log_upper - log_lower)
            left_value = function(float(np.exp(left)))
        else:
            log_lower = left
            left, left_value = right, right_value
            right = log_lower + ratio * (log_upper - log_lower)
            right_value = function(float(np.exp(right)))
    return float(np.exp(0.5 * (log_lower + log_upper)))


def bisect_root_log(
        function, lower: float, upper: float,
        iterations: int = 80) -> float:
    """Bisect a bracketed scalar root over a positive interval in log space."""
    log_lower, log_upper = np.log(lower), np.log(upper)
    lower_value = function(lower)
    upper_value = function(upper)
    if lower_value == 0.0:
        return lower
    if upper_value == 0.0:
        return upper
    if np.signbit(lower_value) == np.signbit(upper_value):
        raise ValueError("Root is not bracketed")
    for _ in range(iterations):
        log_middle = 0.5 * (log_lower + log_upper)
        middle = float(np.exp(log_middle))
        middle_value = function(middle)
        if np.signbit(lower_value) == np.signbit(middle_value):
            log_lower, lower_value = log_middle, middle_value
        else:
            log_upper, upper_value = log_middle, middle_value
    return float(np.exp(0.5 * (log_lower + log_upper)))


def add_geometry_background(
        ax: plt.Axes, epsilon: float, beta_star: np.ndarray,
        prediction_errors: tuple[float, ...],
        z_values: tuple[float, ...], elements: PlotElements,
        style: PlotStyle) -> None:
    signal_power = float(beta_star @ np.diag([1.0, epsilon]) @ beta_star)
    patch_start = len(ax.patches)
    collection_start = len(ax.collections)
    if elements.data_ellipse:
        add_covariance_ellipse(ax, epsilon)
    if elements.prediction_contours:
        add_prediction_contours(
            ax, beta_star, epsilon, prediction_errors, signal_power,
            label_mode="caption", annotate=False)
    if elements.control_contours:
        add_control_contours(
            ax, beta_star, z_values, label_mode="caption", annotate=False)
    # Keep the landmarks readable but subordinate to the estimator paths.
    first_geometry_patch = patch_start + int(elements.data_ellipse)
    for patch in ax.patches[first_geometry_patch:]:
        base_alpha = 1.0 if patch.get_alpha() is None else patch.get_alpha()
        patch.set_alpha(style.geometry_alpha * base_alpha)
        patch.set_linewidth(
            style.geometry_linewidth_scale * patch.get_linewidth())
    for collection in ax.collections[collection_start:]:
        base_alpha = (
            1.0 if collection.get_alpha() is None else collection.get_alpha())
        collection.set_alpha(style.geometry_alpha * base_alpha)


def append_rows(
        rows: list[dict[str, float | int | str]], panel: str, series: str,
        sweep_name: str, sweep_values: np.ndarray, points: np.ndarray,
        beta_star: np.ndarray, sigma_matrix: np.ndarray,
        realization: int | str, sigma2_values: np.ndarray,
        lambda_values: np.ndarray, kappa_values: np.ndarray) -> None:
    for sweep_value, point, sigma2_noise, lam, kappa in zip(
            sweep_values, points, sigma2_values, lambda_values, kappa_values):
        rows.append({
            "panel": panel,
            "series": series,
            "sweep_name": sweep_name,
            "sweep_value": float(sweep_value),
            "realization": realization,
            "sigma2_noise": float(sigma2_noise),
            "sigma_noise": float(np.sqrt(sigma2_noise)),
            "lambda": float(lam),
            "kappa": float(kappa),
            "beta_hat_1": float(point[0]),
            "beta_hat_2": float(point[1]),
            **estimator_metrics(point, beta_star, sigma_matrix),
        })


def write_rows(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def gaussian_summary_row(
        panel: str, sweep_name: str, sweep_value: float,
        sigma2_noise: float, lam: float, kappa: float,
        mean: np.ndarray, covariance: np.ndarray,
        coverage: float) -> dict[str, float | str]:
    """Return one plot-ready row for an analytic Gaussian ellipse."""
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    radius = np.sqrt(-2.0 * np.log1p(-coverage))
    semiaxes = radius * np.sqrt(eigenvalues)
    major_angle = np.degrees(np.arctan2(
        eigenvectors[1, 0], eigenvectors[0, 0]))
    return {
        "panel": panel,
        "sweep_name": sweep_name,
        "sweep_value": float(sweep_value),
        "sigma2_noise": float(sigma2_noise),
        "lambda": float(lam),
        "kappa": float(kappa),
        "coverage": float(coverage),
        "mean_beta1": float(mean[0]),
        "mean_beta2": float(mean[1]),
        "cov_11": float(covariance[0, 0]),
        "cov_12": float(covariance[0, 1]),
        "cov_22": float(covariance[1, 1]),
        "ellipse_semimajor": float(semiaxes[0]),
        "ellipse_semiminor": float(semiaxes[1]),
        "ellipse_major_angle_deg": float(major_angle),
    }


def make_figure(
        epsilon: float, beta_star: np.ndarray, n: int,
        fixed_lambda: float, fixed_sigma2: float, n_cloud: int, seed: int,
        output: Path | None = DEFAULT_FIGURE,
        pdf_output: Path | None = DEFAULT_PDF,
        table_output: Path | None = DEFAULT_TABLE,
        gaussian_table_output: Path | None = DEFAULT_GAUSSIAN_TABLE,
        sigma2_range: tuple[float, float] = (1e-4, 4.0),
        lambda_range: tuple[float, float] = (1e-4, 10.0),
        n_sigma2: int = 81, n_lambda: int = 101,
        sigma2_cloud_levels: tuple[float, ...] = (0.01, 0.1, 1.0, 4.0),
        lambda_cloud_levels: tuple[float, ...] = (1e-4, 0.01, 1.0, 10.0),
        elements: PlotElements = PlotElements(),
        style: PlotStyle = PlotStyle(),
        close: bool = True) -> plt.Figure:
    if not 0.0 < epsilon < 1.0:
        raise ValueError("epsilon must be between zero and one")
    if n < 4 or n_cloud < 10:
        raise ValueError("n must be >=4 and n_cloud must be >=10")
    if fixed_sigma2 < 0.0:
        raise ValueError("fixed_sigma2 must be nonnegative")
    if sigma2_range[0] <= 0.0 or sigma2_range[1] <= sigma2_range[0]:
        raise ValueError("sigma2_range must contain increasing positive values")
    if lambda_range[0] <= 0.0 or lambda_range[1] <= lambda_range[0]:
        raise ValueError("lambda_range must contain increasing positive values")
    if not 0.0 < style.gaussian_coverage < 1.0:
        raise ValueError("style.gaussian_coverage must lie between zero and one")

    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "figure.dpi": 140,
        "savefig.dpi": 300,
    })

    rng = np.random.default_rng(seed)
    sigma_matrix = np.diag([1.0, epsilon])
    X = rng.standard_normal((n, 2)) @ np.diag([1.0, np.sqrt(epsilon)])
    noiseless_response = X @ beta_star
    gram_eigenvalues = np.linalg.eigvalsh(X.T @ X)
    covariance_crossover_lambda = float(
        np.sqrt(np.prod(gram_eigenvalues)) / n)

    # Use a fixed, reproducible realization with visible low-variance motion.
    representative_candidates = rng.standard_normal((4, n))
    representative_noise = representative_candidates[3]
    cloud_noise = rng.standard_normal((n_cloud, n))

    selected_sigma2 = np.asarray(sigma2_cloud_levels, dtype=float)
    selected_lambdas = np.asarray(lambda_cloud_levels, dtype=float)
    if (elements.covariance_crossover
            and lambda_range[0] <= covariance_crossover_lambda
            <= lambda_range[1]):
        selected_lambdas = np.unique(np.r_[
            selected_lambdas, covariance_crossover_lambda])
    if np.any(selected_sigma2 <= 0.0):
        raise ValueError("sigma2_cloud_levels must be positive")
    if np.any(selected_lambdas <= 0.0):
        raise ValueError("lambda_cloud_levels must be positive")
    sigma2_grid = np.unique(np.r_[
        0.0,
        np.geomspace(sigma2_range[0], sigma2_range[1], n_sigma2),
        selected_sigma2,
    ])
    lambda_grid = np.unique(np.r_[
        np.geomspace(lambda_range[0], lambda_range[1], n_lambda),
        fixed_lambda,
        selected_lambdas,
    ])
    kappa_solver = SpectrumKappa([1.0, epsilon], 2.0 / n)
    kappa_grid = kappa_solver.on_array(lambda_grid)
    fixed_kappa = float(kappa_solver(fixed_lambda))
    covariance_crossover_kappa = float(
        kappa_solver(covariance_crossover_lambda))

    fixed_map = ridge_map(X, fixed_lambda)
    representative_noise_path = np.array([
        fixed_map @ (noiseless_response + np.sqrt(sigma2) * representative_noise)
        for sigma2 in sigma2_grid
    ])
    fixed_lambda_mean = fixed_map @ noiseless_response

    left_clouds: dict[float, np.ndarray] = {}
    left_moments: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    for sigma2 in selected_sigma2:
        responses = noiseless_response[None, :] + np.sqrt(sigma2) * cloud_noise
        left_clouds[float(sigma2)] = responses @ fixed_map.T
        left_moments[float(sigma2)] = ridge_gaussian_moments(
            X, beta_star, fixed_lambda, float(sigma2))

    representative_response = (
        noiseless_response + np.sqrt(fixed_sigma2) * representative_noise)
    representative_lambda_path = ridge_path_estimators(
        X, representative_response, lambda_grid).T
    conditional_mean_path = ridge_path_estimators(
        X, noiseless_response, lambda_grid).T

    def representative_at_lambda(lam: float) -> np.ndarray:
        return ridge_map(X, lam) @ representative_response

    def representative_prediction_error(lam: float) -> float:
        beta_hat = representative_at_lambda(lam)
        delta = beta_hat - beta_star
        return float(delta @ sigma_matrix @ delta)

    prediction_stationary_lambda = golden_section_minimize_log(
        representative_prediction_error, lambda_range[0], lambda_range[1])
    boundary_candidates = np.array([
        lambda_range[0], prediction_stationary_lambda, lambda_range[1]])
    prediction_stationary_lambda = float(boundary_candidates[np.argmin([
        representative_prediction_error(value)
        for value in boundary_candidates])])
    prediction_stationary_beta = representative_at_lambda(
        prediction_stationary_lambda)
    prediction_stationary_error = representative_prediction_error(
        prediction_stationary_lambda)

    def control_circle_residual(lam: float) -> float:
        beta_hat = representative_at_lambda(lam)
        return float(beta_hat @ (beta_hat - beta_star))

    root_scan = np.geomspace(lambda_range[0], lambda_range[1], 2001)
    root_values = np.array([control_circle_residual(lam) for lam in root_scan])
    root_brackets = np.flatnonzero(
        np.signbit(root_values[:-1]) != np.signbit(root_values[1:]))
    control_crossing_lambda: float | None = None
    control_crossing_beta: np.ndarray | None = None
    if len(root_brackets):
        roots = np.array([
            bisect_root_log(
                control_circle_residual, root_scan[index], root_scan[index + 1])
            for index in root_brackets
        ])
        control_crossing_lambda = float(roots[np.argmin(np.abs(
            np.log(roots) - np.log(prediction_stationary_lambda)))])
        control_crossing_beta = representative_at_lambda(
            control_crossing_lambda)

    right_clouds: dict[float, np.ndarray] = {}
    right_moments: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    for lam in selected_lambdas:
        mapping = ridge_map(X, float(lam))
        responses = noiseless_response[None, :] + np.sqrt(fixed_sigma2) * cloud_noise
        right_clouds[float(lam)] = responses @ mapping.T
        right_moments[float(lam)] = ridge_gaussian_moments(
            X, beta_star, float(lam), fixed_sigma2)

    prediction_errors = (0.01, 0.04, 0.09)
    z_values = (-1.0 / 3.0, 0.0, 1.0)
    background_xlim, background_ylim = geometry_limits(
        epsilon, beta_star, prediction_errors, z_values)
    visible_point_groups = [
        representative_noise_path,
        representative_lambda_path,
        conditional_mean_path,
    ]
    if elements.cloud_points:
        visible_point_groups.extend(left_clouds.values())
        visible_point_groups.extend(right_clouds.values())
    if elements.gaussian_ellipses:
        ellipse_radius = np.sqrt(
            -2.0 * np.log1p(-style.gaussian_coverage))
        for mean, covariance in [
                *left_moments.values(), *right_moments.values()]:
            coordinate_extent = ellipse_radius * np.sqrt(
                np.maximum(np.diag(covariance), 0.0))
            visible_point_groups.extend([
                (mean - coordinate_extent)[None, :],
                (mean + coordinate_extent)[None, :],
            ])
    all_points = np.vstack(visible_point_groups)
    x_min = min(background_xlim[0], float(np.min(all_points[:, 0])))
    x_max = max(background_xlim[1], float(np.max(all_points[:, 0])))
    y_min = min(background_ylim[0], float(np.min(all_points[:, 1])))
    y_max = max(background_ylim[1], float(np.max(all_points[:, 1])))
    x_pad = 0.08 * (x_max - x_min)
    y_pad = 0.07 * (y_max - y_min)
    xlim = (x_min - x_pad, x_max + x_pad)
    ylim = (y_min - y_pad, y_max + y_pad)

    fig, axes = plt.subplots(
        1, 2, figsize=style.figsize, sharex=True, sharey=True)
    fig.suptitle(
        r"Exact conditional Gaussian geometry of ridge estimators "
        r"(fixed design $X$)",
        fontsize=14, y=0.985,
    )
    for ax in axes:
        add_geometry_background(
            ax, epsilon, beta_star, prediction_errors, z_values,
            elements, style)
        format_axis(ax, epsilon, xlim, ylim)

    left_cloud_colors = plt.cm.Purples(
        np.linspace(0.34, 0.68, len(selected_sigma2)))
    right_cloud_colors = plt.cm.Purples(
        np.linspace(0.38, 0.78, len(selected_lambdas)))

    # Panel a: fixed lambda, vary noise scale.
    ax = axes[0]
    ax.set_title(
        rf"a  Fixed $\lambda={fixed_lambda:g}$ "
        rf"($\kappa={fixed_kappa:.4f}$); vary $\sigma^2$ on log grid",
        loc="left",
    )
    for level_index, (color, sigma2) in enumerate(zip(
            left_cloud_colors, selected_sigma2)):
        marker = MARKERS[level_index % len(MARKERS)]
        linestyle = LINESTYLES[level_index % len(LINESTYLES)]
        points = left_clouds[float(sigma2)]
        if elements.cloud_points:
            ax.scatter(
                points[:, 0], points[:, 1], s=style.cloud_point_size,
                marker=marker, color=color, alpha=style.cloud_point_alpha,
                linewidths=0, zorder=5,
            )
        if elements.gaussian_ellipses:
            mean, covariance = left_moments[float(sigma2)]
            add_gaussian_ellipse(
                ax, mean, covariance, color, "solid",
                style.gaussian_coverage, style.gaussian_fill_alpha,
                style.gaussian_linewidth)
    if elements.realization_paths:
        ax.plot(
            representative_noise_path[:, 0], representative_noise_path[:, 1],
            color=style.path_color, linewidth=style.path_linewidth, zorder=8,
        )
        for sigma2 in selected_sigma2:
            index = int(np.argmin(np.abs(sigma2_grid - sigma2)))
            point = representative_noise_path[index]
            ax.scatter(
                point[0], point[1], s=39, facecolor="white",
                edgecolor=style.path_color, linewidth=1.4, zorder=9,
            )
    if elements.mean_paths:
        ax.scatter(
            fixed_lambda_mean[0], fixed_lambda_mean[1], marker="D", s=35,
            facecolor="white", edgecolor=style.mean_color,
            linewidth=1.3, zorder=9,
        )
    if elements.point_labels:
        ax.annotate(
            r"$\sigma^2=0$: $\mathbb{E}[\hat\beta\mid X]$",
            xy=fixed_lambda_mean,
            xytext=(14, -25), textcoords="offset points", ha="left",
            fontsize=8.5, color=style.mean_color,
        )
        ax.annotate(
            rf"$\sigma^2={sigma2_grid[-1]:g}$",
            xy=representative_noise_path[-1],
            xytext=(9, 8), textcoords="offset points", ha="left",
            fontsize=8.5, color=style.path_color,
            bbox={"facecolor": "white", "edgecolor": "none",
                  "alpha": 0.84, "pad": 0.5},
        )
    if elements.arrows and elements.realization_paths:
        add_arrow_on_path(
            ax, representative_noise_path, len(sigma2_grid) - 6,
            len(sigma2_grid) - 1, style.path_color)
    if elements.panel_notes:
        ax.text(
            0.02, 0.98,
            "Exact Gaussian ellipses are concentric; axes scale with $\\sigma$.\n"
            "One coupled realization is affine in $\\sigma$.",
            transform=ax.transAxes, ha="left", va="top", fontsize=9,
            color="0.28",
            bbox={"facecolor": "white", "edgecolor": "none",
                  "alpha": 0.86, "pad": 1.0},
        )
        sigma2_text = ",\,".join(f"{value:g}" for value in selected_sigma2)
        ax.text(
            0.02, 0.86,
            rf"{100 * style.gaussian_coverage:.0f}\% ellipses: "
            rf"$\sigma^2={sigma2_text}$",
            transform=ax.transAxes, ha="left", va="top", fontsize=8.5,
            color=style.path_color,
        )

    # Panel b: fixed noise scale, vary regularization.
    ax = axes[1]
    ax.set_title(
        rf"b  Fixed $\sigma^2={fixed_sigma2:g}$; vary $\lambda/\kappa$ on log grid",
        loc="left",
    )
    for level_index, (color, lam) in enumerate(zip(
            right_cloud_colors, selected_lambdas)):
        marker = MARKERS[level_index % len(MARKERS)]
        linestyle = LINESTYLES[level_index % len(LINESTYLES)]
        points = right_clouds[float(lam)]
        if elements.cloud_points:
            ax.scatter(
                points[:, 0], points[:, 1], s=style.cloud_point_size,
                marker=marker, color=color, alpha=style.cloud_point_alpha,
                linewidths=0, zorder=5,
            )
        if elements.gaussian_ellipses:
            mean, covariance = right_moments[float(lam)]
            add_gaussian_ellipse(
                ax, mean, covariance, color, "solid",
                style.gaussian_coverage, style.gaussian_fill_alpha,
                style.gaussian_linewidth)
    if elements.mean_paths:
        ax.plot(
            conditional_mean_path[:, 0], conditional_mean_path[:, 1],
            color=style.mean_color, linewidth=1.6,
            linestyle=(0, (4, 2)), zorder=7,
        )
        for color, lam in zip(right_cloud_colors, selected_lambdas):
            mean, _ = right_moments[float(lam)]
            ax.scatter(
                mean[0], mean[1], marker="D", s=28, color=color,
                edgecolor="white", linewidth=0.7, zorder=8,
            )
    if elements.realization_paths:
        ax.plot(
            representative_lambda_path[:, 0], representative_lambda_path[:, 1],
            color=style.path_color, linewidth=style.path_linewidth, zorder=8,
        )
        for lam in selected_lambdas:
            index = int(np.argmin(np.abs(lambda_grid - lam)))
            point = representative_lambda_path[index]
            ax.scatter(
                point[0], point[1], s=42, facecolor="white",
                edgecolor=style.path_color, linewidth=1.4, zorder=9,
            )
    if elements.point_labels:
        crossover_index = int(np.argmin(
            np.abs(selected_lambdas - covariance_crossover_lambda)))
        labeled_indices = {0, crossover_index, len(selected_lambdas) - 1}
        for label_index, lam in enumerate(selected_lambdas):
            if label_index not in labeled_indices:
                continue
            index = int(np.argmin(np.abs(lambda_grid - lam)))
            point = representative_lambda_path[index]
            if label_index == 0:
                offset, alignment, vertical_alignment = (12, 12), "left", "bottom"
            elif label_index == len(selected_lambdas) - 1:
                offset, alignment, vertical_alignment = (-8, -25), "right", "top"
            else:
                offset, alignment, vertical_alignment = (12, -20), "left", "top"
            if label_index == crossover_index:
                label = rf"$\lambda_\times={lam:.3g}$"
            else:
                label = rf"$\lambda={lam:g}$"
            ax.annotate(
                label,
                xy=point, xytext=offset, textcoords="offset points",
                ha=alignment, va=vertical_alignment, fontsize=8.3,
                color=style.path_color,
                bbox={"facecolor": "white", "edgecolor": "none",
                      "alpha": 0.86, "pad": 0.6},
            )
    if elements.stationary_highlights:
        prediction_color = "#2E75B6"
        control_color = "#B35C1E"
        ax.add_patch(Ellipse(
            tuple(beta_star),
            width=2.0 * np.sqrt(prediction_stationary_error),
            height=2.0 * np.sqrt(prediction_stationary_error / epsilon),
            fill=False, edgecolor=prediction_color, linewidth=2.5,
            alpha=0.82, zorder=7,
        ))
        gram = X.T @ X
        tangent = -n * np.linalg.solve(
            gram + n * prediction_stationary_lambda * np.eye(2),
            prediction_stationary_beta)
        tangent = tangent / np.linalg.norm(tangent)
        tangent_half_length = 0.105
        tangent_endpoints = np.vstack([
            prediction_stationary_beta - tangent_half_length * tangent,
            prediction_stationary_beta + tangent_half_length * tangent,
        ])
        ax.plot(
            tangent_endpoints[:, 0], tangent_endpoints[:, 1],
            color=prediction_color, linewidth=3.0,
            solid_capstyle="round", alpha=0.88, zorder=9,
        )
        ax.scatter(
            prediction_stationary_beta[0], prediction_stationary_beta[1],
            marker="D", s=72, facecolor="white",
            edgecolor=prediction_color, linewidth=2.2, zorder=10,
        )
        ax.annotate(
            "prediction stationary\n"
            + rf"$\lambda_{{\rm pred}}={prediction_stationary_lambda:.3g}$",
            xy=prediction_stationary_beta, xytext=(52, 34),
            textcoords="offset points", ha="left", va="bottom",
            fontsize=8.5, color=prediction_color,
            arrowprops={"arrowstyle": "-", "color": prediction_color,
                        "linewidth": 1.1},
            bbox={"facecolor": "white", "edgecolor": "none",
                  "alpha": 0.88, "pad": 0.7},
            zorder=11,
        )
        if (control_crossing_lambda is not None
                and control_crossing_beta is not None):
            control_center = 0.5 * beta_star
            control_radius = 0.5 * np.linalg.norm(beta_star)
            control_angle = np.degrees(np.arctan2(
                control_crossing_beta[1] - control_center[1],
                control_crossing_beta[0] - control_center[0]))
            ax.add_patch(Arc(
                tuple(control_center), 2.0 * control_radius,
                2.0 * control_radius, angle=0.0,
                theta1=control_angle - 12.0,
                theta2=control_angle + 12.0,
                color=control_color, linewidth=4.0,
                alpha=0.95, zorder=8.5,
            ))
            ax.scatter(
                control_crossing_beta[0], control_crossing_beta[1],
                marker="o", s=76, facecolor="white",
                edgecolor=control_color, linewidth=2.3, zorder=10,
            )
            ax.annotate(
                r"control calibration: $z=0$" + "\n"
                + rf"$\lambda_{{\rm ctrl}}={control_crossing_lambda:.3g}$",
                xy=control_crossing_beta, xytext=(52, -8),
                textcoords="offset points", ha="left", va="top",
                fontsize=8.5, color=control_color,
                arrowprops={"arrowstyle": "-", "color": control_color,
                            "linewidth": 1.1},
                bbox={"facecolor": "white", "edgecolor": "none",
                      "alpha": 0.88, "pad": 0.7},
                zorder=11,
            )
    if elements.arrows and elements.realization_paths:
        add_arrow_on_path(
            ax, representative_lambda_path, len(lambda_grid) - 9,
            len(lambda_grid) - 3, style.path_color)
    if elements.panel_notes:
        ax.text(
            0.02, 0.98,
            "Exact Gaussian mean translates toward zero; covariance\n"
            "contracts and swaps its major axis near $\\lambda_\\times$.",
            transform=ax.transAxes, ha="left", va="top", fontsize=9,
            color="0.28",
            bbox={"facecolor": "white", "edgecolor": "none",
                  "alpha": 0.86, "pad": 1.0},
        )
        ax.text(
            0.02, 0.88,
            rf"{100 * style.gaussian_coverage:.0f}\% ellipses; "
            rf"$(\lambda_\times,\kappa_\times)="
            rf"({covariance_crossover_lambda:.3g},"
            rf"{covariance_crossover_kappa:.3g})$",
            transform=ax.transAxes, ha="left", va="top", fontsize=8.5,
            color=style.path_color,
        )

    if elements.teacher:
        for ax in axes:
            add_teacher(ax, beta_star)

    legend_handles: list[Line2D] = []
    if elements.prediction_contours:
        legend_handles.append(Line2D(
            [0], [0], color="#2E75B6", linewidth=2.1,
            label=r"prediction iso-error $E_{\rm gen}$"))
    if elements.control_contours:
        legend_handles.extend([
            Line2D([0], [0], color="#B35C1E", linewidth=2.1,
                   linestyle=(0, (4, 2)), label=r"control iso-error $z\ne0$"),
            Line2D([0], [0], color="#B35C1E", linewidth=3.2,
                   marker="*", markerfacecolor="white", markersize=7,
                   label=r"control optimum $z=0$"),
        ])
    if elements.realization_paths:
        legend_handles.append(Line2D(
            [0], [0], color=style.path_color,
            linewidth=style.path_linewidth, marker="o",
            markerfacecolor="white", label="one coupled realization"))
    if elements.gaussian_ellipses:
        middle_cloud_color = plt.cm.Purples(0.58)
        legend_handles.append(Line2D(
            [0], [0], color=middle_cloud_color, linewidth=1.7,
            linestyle=(0, (5, 2)),
            label=rf"exact {100 * style.gaussian_coverage:.0f}\% Gaussian ellipse"))
    if elements.cloud_points:
        legend_handles.append(Line2D(
            [0], [0], color=plt.cm.Purples(0.48), linewidth=0,
            marker=".", markersize=7, label="optional Monte Carlo draws"))
    if elements.mean_paths:
        legend_handles.append(Line2D(
            [0], [0], color=style.mean_color, linewidth=1.6,
            linestyle=(0, (4, 2)), label="conditional mean path"))
    if elements.legend and legend_handles:
        fig.legend(
            handles=legend_handles, loc="lower center",
            ncol=style.legend_columns,
            bbox_to_anchor=(0.5, 0.015), frameon=False, fontsize=9,
            handlelength=3.0, columnspacing=1.6,
        )
    fig.subplots_adjust(
        left=0.07, right=0.99, bottom=0.18, top=0.90, wspace=0.14)

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, bbox_inches="tight")
    if pdf_output is not None:
        pdf_output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(pdf_output, bbox_inches="tight")

    rows: list[dict[str, float | int | str]] = []
    append_rows(
        rows, "fixed_lambda", "representative_path", "sigma2",
        sigma2_grid, representative_noise_path, beta_star, sigma_matrix,
        3, sigma2_grid, np.full_like(sigma2_grid, fixed_lambda),
        np.full_like(sigma2_grid, fixed_kappa))
    append_rows(
        rows, "fixed_lambda", "conditional_mean", "sigma2",
        np.array([0.0]), fixed_lambda_mean[None, :], beta_star, sigma_matrix,
        "mean", np.array([0.0]), np.array([fixed_lambda]),
        np.array([fixed_kappa]))
    for sigma2, points in left_clouds.items():
        append_rows(
            rows, "fixed_lambda", "cloud", "sigma2",
            np.full(len(points), sigma2), points, beta_star, sigma_matrix,
            "cloud", np.full(len(points), sigma2),
            np.full(len(points), fixed_lambda),
            np.full(len(points), fixed_kappa))
    append_rows(
        rows, "fixed_sigma", "representative_path", "lambda",
        lambda_grid, representative_lambda_path, beta_star, sigma_matrix,
        3, np.full_like(lambda_grid, fixed_sigma2), lambda_grid, kappa_grid)
    append_rows(
        rows, "fixed_sigma", "prediction_stationary", "lambda",
        np.array([prediction_stationary_lambda]),
        prediction_stationary_beta[None, :], beta_star, sigma_matrix,
        3, np.array([fixed_sigma2]),
        np.array([prediction_stationary_lambda]),
        np.array([float(kappa_solver(prediction_stationary_lambda))]))
    if (control_crossing_lambda is not None
            and control_crossing_beta is not None):
        append_rows(
            rows, "fixed_sigma", "control_zero_crossing", "lambda",
            np.array([control_crossing_lambda]),
            control_crossing_beta[None, :], beta_star, sigma_matrix,
            3, np.array([fixed_sigma2]),
            np.array([control_crossing_lambda]),
            np.array([float(kappa_solver(control_crossing_lambda))]))
    append_rows(
        rows, "fixed_sigma", "conditional_mean", "lambda",
        lambda_grid, conditional_mean_path, beta_star, sigma_matrix,
        "mean", np.zeros_like(lambda_grid), lambda_grid, kappa_grid)
    for lam, points in right_clouds.items():
        kappa = float(kappa_solver(lam))
        append_rows(
            rows, "fixed_sigma", "cloud", "lambda",
            np.full(len(points), lam), points, beta_star, sigma_matrix,
            "cloud", np.full(len(points), fixed_sigma2),
            np.full(len(points), lam), np.full(len(points), kappa))
    if table_output is not None:
        write_rows(table_output, rows)

    gaussian_rows: list[dict[str, float | str]] = []
    for sigma2, (mean, covariance) in left_moments.items():
        gaussian_rows.append(gaussian_summary_row(
            "fixed_lambda", "sigma2", sigma2, sigma2, fixed_lambda,
            fixed_kappa, mean, covariance, style.gaussian_coverage))
    for lam, (mean, covariance) in right_moments.items():
        gaussian_rows.append(gaussian_summary_row(
            "fixed_sigma2", "lambda", lam, fixed_sigma2, lam,
            float(kappa_solver(lam)), mean, covariance,
            style.gaussian_coverage))
    if gaussian_table_output is not None:
        write_rows(gaussian_table_output, gaussian_rows)

    # Invariants: fixed-lambda path is affine in sigma; kappa is monotone.
    expected_noise_path = (
        fixed_lambda_mean[None, :]
        + np.sqrt(sigma2_grid)[:, None]
        * (fixed_map @ representative_noise)[None, :])
    assert np.max(np.abs(representative_noise_path - expected_noise_path)) < 1e-10
    assert np.all(np.diff(kappa_grid) > 0.0)
    for lam, (mean, covariance) in right_moments.items():
        mapping = ridge_map(X, lam)
        assert np.allclose(mean, mapping @ noiseless_response)
        assert np.allclose(
            covariance, fixed_sigma2 * mapping @ mapping.T)
    if (lambda_range[0] < prediction_stationary_lambda
            < lambda_range[1]):
        prediction_derivative = -n * np.linalg.solve(
            X.T @ X + n * prediction_stationary_lambda * np.eye(2),
            prediction_stationary_beta)
        stationarity_residual = float(
            (prediction_stationary_beta - beta_star)
            @ sigma_matrix @ prediction_derivative)
        assert abs(stationarity_residual) < 1e-8
    if control_crossing_beta is not None:
        assert abs(float(
            control_crossing_beta @ (control_crossing_beta - beta_star)
        )) < 1e-10
    if close:
        plt.close(fig)
    return fig


def make_lambda_only_figure(
        *, output: Path | None, pdf_output: Path | None,
        style: PlotStyle = PlotStyle(), close: bool = True,
        **figure_kwargs) -> plt.Figure:
    """Render the lambda-path panel alone while reusing the full computation."""
    single_style = replace(
        style, figsize=style.lambda_only_figsize, legend_columns=2)
    figure_kwargs.pop("close", None)
    figure_kwargs.pop("output", None)
    figure_kwargs.pop("pdf_output", None)
    fig = make_figure(
        output=None, pdf_output=None, style=single_style,
        close=False, **figure_kwargs)
    sigma_axis, lambda_axis = fig.axes[:2]
    fig.delaxes(sigma_axis)
    lambda_axis.set_position([0.11, 0.24, 0.85, 0.65])
    fixed_sigma2 = float(figure_kwargs["fixed_sigma2"])
    lambda_axis.set_title(
        rf"Fixed $\sigma^2={fixed_sigma2:g}$; vary $\lambda/\kappa$ on log grid",
        loc="left")
    fig.suptitle(
        r"Conditional Gaussian geometry along the ridge path "
        r"(fixed design $X$)", fontsize=14, y=0.985)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, bbox_inches="tight")
    if pdf_output is not None:
        pdf_output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(pdf_output, bbox_inches="tight")
    if close:
        plt.close(fig)
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epsilon", type=float, default=0.04)
    parser.add_argument("--beta1", type=float, default=1.1)
    parser.add_argument("--beta2", type=float, default=0.65)
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--fixed-lambda", type=float, default=0.05)
    parser.add_argument("--fixed-sigma2", type=float, default=0.5)
    parser.add_argument("--sigma2-min", type=float, default=1e-4)
    parser.add_argument("--sigma2-max", type=float, default=4.0)
    parser.add_argument("--lambda-min", type=float, default=1e-4)
    parser.add_argument("--lambda-max", type=float, default=10.0)
    parser.add_argument("--n-cloud", type=int, default=250)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument(
        "--panel", choices=("both", "lambda"), default="both",
        help="Render both panels or the lambda-path panel alone")
    parser.add_argument("--output", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--pdf-output", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--table-output", type=Path, default=DEFAULT_TABLE)
    parser.add_argument(
        "--gaussian-table-output", type=Path, default=DEFAULT_GAUSSIAN_TABLE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    figure_arguments = dict(
        epsilon=args.epsilon,
        beta_star=np.array([args.beta1, args.beta2], dtype=float),
        n=args.n,
        fixed_lambda=args.fixed_lambda,
        fixed_sigma2=args.fixed_sigma2,
        n_cloud=args.n_cloud,
        seed=args.seed,
        table_output=args.table_output,
        gaussian_table_output=args.gaussian_table_output,
        sigma2_range=(args.sigma2_min, args.sigma2_max),
        lambda_range=(args.lambda_min, args.lambda_max),
    )
    if args.panel == "both":
        make_figure(
            output=args.output, pdf_output=args.pdf_output,
            **figure_arguments)
    else:
        make_lambda_only_figure(
            output=args.output, pdf_output=args.pdf_output,
            **figure_arguments)
    print(f"Saved {args.output}")
    print(f"Saved {args.pdf_output}")
    print(f"Saved {args.table_output}")
    print(f"Saved {args.gaussian_table_output}")


if __name__ == "__main__":
    main()

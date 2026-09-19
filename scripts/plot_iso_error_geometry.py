"""Illustrate prediction and self-control iso-error geometry in 2D.

For ``Sigma = diag(1, epsilon)``, natural prediction error has elliptical
level sets centered on the teacher,

    E_gen = (beta_hat - beta_star)^T Sigma (beta_hat - beta_star),

whereas a fixed own-path ratio

    z = ||beta_hat||^2 / (beta_hat^T beta_star) - 1

defines a circle with diameter from zero to ``(1 + z) beta_star``.  The
default control contours use the normalized accentuation MSE and show the two
distinct z branches that share the same nonzero error value.
"""
from __future__ import annotations

import argparse
import csv
import os
from fractions import Fraction
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/accentuationpredrmt-matplotlib")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIGURE = REPO_ROOT / "figures" / "iso_error_geometry.png"
DEFAULT_PDF = REPO_ROOT / "figures" / "iso_error_geometry.pdf"
DEFAULT_TABLE = REPO_ROOT / "tables" / "iso_error_geometry_contours.csv"


def control_mse(z: float) -> float:
    """Return E_acc / S for a finite own-path ratio z."""
    if np.isclose(z, -1.0):
        raise ValueError("z=-1 is singular for the own-path ratio")
    return (z / (1.0 + z)) ** 2


def control_circle(beta_star: np.ndarray, z: float) -> tuple[np.ndarray, float]:
    """Center and radius of the z-level circle in beta_hat space."""
    scale = 1.0 + z
    center = 0.5 * scale * beta_star
    radius = 0.5 * abs(scale) * float(np.linalg.norm(beta_star))
    return center, radius


def add_covariance_ellipse(ax: plt.Axes, epsilon: float) -> None:
    """Add the one-standard-deviation input covariance ellipse as context."""
    ellipse = Ellipse(
        (0.0, 0.0), width=2.0, height=2.0 * np.sqrt(epsilon),
        facecolor="0.75", edgecolor="0.38", linewidth=1.2,
        linestyle=(0, (3, 2)), alpha=0.24, zorder=0,
    )
    ax.add_patch(ellipse)
    ax.text(
        -0.10, -1.18 * np.sqrt(epsilon), "data manifold",
        color="0.32", fontsize=9, ha="right", va="top",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82,
              "pad": 0.8},
    )


def add_teacher(ax: plt.Axes, beta_star: np.ndarray) -> None:
    """Draw the teacher as a vector and mark its endpoint."""
    arrow = FancyArrowPatch(
        (0.0, 0.0), tuple(beta_star), arrowstyle="-|>", mutation_scale=15,
        linewidth=2.0, color="0.12", zorder=8,
    )
    ax.add_patch(arrow)
    ax.scatter(
        [beta_star[0]], [beta_star[1]], marker="*", s=90,
        facecolor="white", edgecolor="0.08", linewidth=1.3, zorder=9,
    )
    ax.annotate(
        r"teacher $\beta^\star$", xy=beta_star, xytext=(8, -18),
        textcoords="offset points", fontsize=10, color="0.12",
    )
    ax.scatter([0.0], [0.0], s=18, color="0.15", zorder=9)
    ax.annotate(
        r"$0$", xy=(0.0, 0.0), xytext=(-11, -13),
        textcoords="offset points", fontsize=9, color="0.2",
    )


def prediction_label_position(
        beta_star: np.ndarray, error: float, epsilon: float,
        angle: float) -> np.ndarray:
    return beta_star + np.array([
        np.sqrt(error) * np.cos(angle),
        np.sqrt(error / epsilon) * np.sin(angle),
    ])


def prediction_colors(n_levels: int) -> np.ndarray:
    return plt.cm.Blues(np.linspace(0.48, 0.9, n_levels))


def control_styles(
        z_values: tuple[float, ...]
        ) -> list[tuple[str, str | tuple, float, str]]:
    control_color = "#B35C1E"
    nonzero_styles = [(0, (4, 2)), (0, (1.4, 1.4)), (0, (6, 2, 1, 2))]
    styles: list[tuple[str, str | tuple, float, str]] = []
    nonzero_index = 0
    for z in z_values:
        if np.isclose(z, 0.0):
            styles.append((control_color, "solid", 3.2, "*"))
        else:
            styles.append((
                control_color,
                nonzero_styles[nonzero_index % len(nonzero_styles)],
                2.25,
                "o",
            ))
            nonzero_index += 1
    return styles


def add_prediction_contours(
        ax: plt.Axes, beta_star: np.ndarray, epsilon: float,
        errors: tuple[float, ...], signal_power: float,
        label_mode: str = "caption",
        annotate: bool = True) -> list[dict[str, float | str]]:
    colors = prediction_colors(len(errors))
    label_angles = np.linspace(0.78, 1.02, len(errors))
    rows: list[dict[str, float | str]] = []
    for error, color, angle in zip(errors, colors, label_angles):
        semi_x = np.sqrt(error)
        semi_y = np.sqrt(error / epsilon)
        contour = Ellipse(
            tuple(beta_star), width=2.0 * semi_x, height=2.0 * semi_y,
            fill=False, edgecolor=color, linewidth=2.2, zorder=3,
        )
        ax.add_patch(contour)
        label_xy = prediction_label_position(
            beta_star, error, epsilon, float(angle))
        if annotate:
            if label_mode == "caption":
                label = rf"$E_{{\rm gen}}={error:g}$"
            else:
                r2_gen = 1.0 - error / signal_power
                label = (
                    rf"$E_{{\rm gen}}={error:g}$"
                    + rf"  ($R^2_{{\rm gen}}={r2_gen:.3f}$)"
                )
            ax.annotate(
                label, xy=label_xy,
                xytext=(7, 2), textcoords="offset points", fontsize=9,
                color=color, ha="left", va="center",
                bbox={"facecolor": "white", "edgecolor": "none",
                      "alpha": 0.86, "pad": 0.8},
            )
        rows.append({
            "geometry": "prediction",
            "level": error,
            "r2": 1.0 - error / signal_power,
            "z": "",
            "center_beta1": float(beta_star[0]),
            "center_beta2": float(beta_star[1]),
            "semi_axis_beta1": float(semi_x),
            "semi_axis_beta2": float(semi_y),
            "radius": "",
        })
    return rows


def point_on_circle(center: np.ndarray, radius: float, angle: float) -> np.ndarray:
    return center + radius * np.array([np.cos(angle), np.sin(angle)])


def format_z(z: float) -> str:
    """Compact exact-looking label for simple z values."""
    fraction = Fraction(float(z)).limit_denominator(12)
    if np.isclose(float(fraction), z, rtol=0.0, atol=1e-9):
        if fraction.denominator == 1:
            return str(fraction.numerator)
        return f"{fraction.numerator}/{fraction.denominator}"
    return f"{z:.2g}"


def add_control_contours(
        ax: plt.Axes, beta_star: np.ndarray,
        z_values: tuple[float, ...], label_mode: str = "caption",
        annotate: bool = True) -> list[dict[str, float | str]]:
    label_angles = [np.pi, 2.35, 0.30, -0.35, -2.25]
    rows: list[dict[str, float | str]] = []
    for index, (z, (color, linestyle, linewidth, marker)) in enumerate(
            zip(z_values, control_styles(z_values))):
        center, radius = control_circle(beta_star, z)
        if np.isclose(z, 0.0):
            ax.add_patch(Circle(
                tuple(center), radius=radius, fill=False, edgecolor=color,
                linewidth=6.0, linestyle="solid", alpha=0.13, zorder=3.5,
            ))
        circle = Circle(
            tuple(center), radius=radius, fill=False, edgecolor=color,
            linewidth=linewidth, linestyle=linestyle, zorder=4,
        )
        ax.add_patch(circle)
        error = control_mse(z)
        if annotate:
            angle = label_angles[index % len(label_angles)]
            label_xy = point_on_circle(center, radius, angle)
            offset = 10.0 * np.array([np.cos(angle), np.sin(angle)])
            horizontal_alignment = "left" if np.cos(angle) >= 0 else "right"
            if label_mode == "caption":
                if np.isclose(z, 0.0):
                    label = r"$z=0$  (optimal)"
                else:
                    label = rf"$z={format_z(z)}$"
            else:
                r2_acc = 1.0 - z ** 2
                optimal = r"  (optimal)" if np.isclose(z, 0.0) else ""
                label = (
                    rf"$z={format_z(z)}$"
                    + rf"  ($E_{{\rm acc}}/S={error:g},\;"
                    + rf"R^2_{{\rm acc}}={r2_acc:.3f}$)"
                    + optimal
                )
            ax.annotate(
                label, xy=label_xy, xytext=tuple(offset),
                textcoords="offset points", fontsize=9, color=color,
                ha=horizontal_alignment, va="center",
                bbox={"facecolor": "white", "edgecolor": "none",
                      "alpha": 0.88, "pad": 0.8},
            )
        endpoint = (1.0 + z) * beta_star
        ax.scatter(
            [endpoint[0]], [endpoint[1]],
            marker=marker, s=46 if marker == "*" else 24, facecolor="white",
            edgecolor=color, linewidth=1.2, zorder=6,
        )
        rows.append({
            "geometry": "control",
            "level": error,
            "r2": 1.0 - z ** 2,
            "z": z,
            "center_beta1": float(center[0]),
            "center_beta2": float(center[1]),
            "semi_axis_beta1": "",
            "semi_axis_beta2": "",
            "radius": radius,
        })
    return rows


def add_metric_caption(
        ax: plt.Axes, epsilon: float, beta_star: np.ndarray,
        prediction_errors: tuple[float, ...],
        z_values: tuple[float, ...]) -> None:
    """Place full contour metrics in a color- and line-matched side caption."""
    signal_power = float(beta_star @ np.diag([1.0, epsilon]) @ beta_star)
    handles: list[Line2D] = []
    labels: list[str] = []
    for error, color in zip(
            prediction_errors, prediction_colors(len(prediction_errors))):
        r2_gen = 1.0 - error / signal_power
        handles.append(Line2D([0], [0], color=color, linewidth=2.4))
        labels.append(
            rf"Prediction: $E_{{\rm gen}}={error:g}$, "
            + rf"$R^2_{{\rm gen}}={r2_gen:.3f}$"
        )
    for z, (color, linestyle, linewidth, marker) in zip(
            z_values, control_styles(z_values)):
        error = control_mse(z)
        r2_acc = 1.0 - z ** 2
        handles.append(Line2D(
            [0], [0], color=color, linewidth=linewidth, linestyle=linestyle,
            marker=marker, markersize=7 if marker == "*" else 4,
            markerfacecolor="white",
            markeredgewidth=1.0,
        ))
        role = "Control (optimal)" if np.isclose(z, 0.0) else "Control"
        labels.append(
            role + rf": $z={format_z(z)}$, $E_{{\rm acc}}/S={error:g}$, "
            + rf"$R^2_{{\rm acc}}={r2_acc:.3f}$"
        )
    caption_title = (
        "Contour metrics\n"
        + rf"Prediction uses $S={signal_power:.3f}$; control uses signed $z$."
        + "\n"
        + r"Semi-axes: $\sqrt{E},\sqrt{E/\epsilon}$; "
        + r"circle diameter: $0\leftrightarrow(1+z)\beta^\star$."
    )
    legend = ax.legend(
        handles, labels, title=caption_title, loc="upper left",
        bbox_to_anchor=(1.03, 1.0), frameon=False, borderaxespad=0.0,
        handlelength=3.0, handletextpad=0.9, labelspacing=0.85,
        fontsize=9,
    )
    legend.get_title().set_fontsize(9)
    legend._legend_box.align = "left"


def geometry_limits(
        epsilon: float, beta_star: np.ndarray,
        prediction_errors: tuple[float, ...],
        z_values: tuple[float, ...]) -> tuple[tuple[float, float], tuple[float, float]]:
    """Compute padded limits containing every requested geometric element."""
    x_values = [-1.0, 1.0, 0.0, float(beta_star[0])]
    y_values = [-np.sqrt(epsilon), np.sqrt(epsilon), 0.0, float(beta_star[1])]
    for error in prediction_errors:
        x_values.extend(beta_star[0] + np.array([-1.0, 1.0]) * np.sqrt(error))
        y_values.extend(
            beta_star[1] + np.array([-1.0, 1.0]) * np.sqrt(error / epsilon))
    for z in z_values:
        center, radius = control_circle(beta_star, z)
        x_values.extend([center[0] - radius, center[0] + radius])
        y_values.extend([center[1] - radius, center[1] + radius])
    x_min, x_max = float(np.min(x_values)), float(np.max(x_values))
    y_min, y_max = float(np.min(y_values)), float(np.max(y_values))
    x_pad = max(0.18, 0.08 * (x_max - x_min))
    y_pad = max(0.18, 0.06 * (y_max - y_min))
    return (x_min - x_pad, x_max + x_pad), (y_min - y_pad, y_max + y_pad)


def format_axis(
        ax: plt.Axes, epsilon: float, xlim: tuple[float, float],
        ylim: tuple[float, float]) -> None:
    ax.axhline(0.0, color="0.82", linewidth=0.8, zorder=-2)
    ax.axvline(0.0, color="0.82", linewidth=0.8, zorder=-2)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$\hat\beta_1$  (high-variance PC, $s_1=1$)")
    ax.set_ylabel(
        rf"$\hat\beta_2$  (low-variance PC, $s_2=\epsilon={epsilon:g}$)"
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(direction="out", length=3)


def write_metadata(
        path: Path, rows: list[dict[str, float | str]], epsilon: float,
        beta_star: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "geometry", "level", "r2", "z", "center_beta1", "center_beta2",
        "semi_axis_beta1", "semi_axis_beta2", "radius", "epsilon",
        "beta_star_1", "beta_star_2",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                **row,
                "epsilon": epsilon,
                "beta_star_1": float(beta_star[0]),
                "beta_star_2": float(beta_star[1]),
            })


def make_figure(
        epsilon: float, beta_star: np.ndarray,
        prediction_errors: tuple[float, ...],
        z_values: tuple[float, ...], layout: str, label_mode: str,
        output: Path, pdf_output: Path | None, table_output: Path | None,
        requested_xlim: tuple[float, float] | None = None,
        requested_ylim: tuple[float, float] | None = None) -> None:
    if not 0.0 < epsilon < 1.0:
        raise ValueError("epsilon must lie strictly between zero and one")
    if beta_star.shape != (2,) or np.allclose(beta_star, 0.0):
        raise ValueError("beta_star must be a nonzero two-vector")
    if not prediction_errors or any(error <= 0.0 for error in prediction_errors):
        raise ValueError("prediction errors must all be positive")
    if not z_values:
        raise ValueError("at least one control z value is required")
    if any(np.isclose(z, -1.0) for z in z_values):
        raise ValueError("z=-1 is singular and cannot be plotted")
    if layout not in {"overlay", "side-by-side"}:
        raise ValueError(f"unknown layout: {layout}")
    if label_mode not in {"caption", "inline"}:
        raise ValueError(f"unknown label mode: {label_mode}")

    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        # Keep text as editable, embedded TrueType glyphs in vector exports.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "figure.dpi": 140,
        "savefig.dpi": 300,
    })
    auto_xlim, auto_ylim = geometry_limits(
        epsilon, beta_star, prediction_errors, z_values)
    xlim = requested_xlim or auto_xlim
    ylim = requested_ylim or auto_ylim

    title = (
        r"Prediction ellipses and control circles for "
        rf"$\Sigma=\mathrm{{diag}}(1,{epsilon:g})$"
    )
    signal_power = float(beta_star @ np.diag([1.0, epsilon]) @ beta_star)
    if layout == "overlay":
        data_ratio = (ylim[1] - ylim[0]) / (xlim[1] - xlim[0])
        figure_height = min(10.5, max(6.0, 6.8 * data_ratio * 0.82))
        figure_width = 10.8 if label_mode == "caption" else 7.2
        fig, ax = plt.subplots(figsize=(figure_width, figure_height))
        fig.suptitle(title, fontsize=14, y=0.985)
        add_covariance_ellipse(ax, epsilon)
        prediction_rows = add_prediction_contours(
            ax, beta_star, epsilon, prediction_errors, signal_power,
            label_mode=label_mode)
        control_rows = add_control_contours(
            ax, beta_star, z_values, label_mode=label_mode)
        add_teacher(ax, beta_star)
        format_axis(ax, epsilon, xlim, ylim)
        if label_mode == "caption":
            add_metric_caption(
                ax, epsilon, beta_star, prediction_errors, z_values)
            fig.subplots_adjust(
                left=0.09, right=0.64, bottom=0.10, top=0.93)
        else:
            ax.text(
                0.02, 0.985,
                r"prediction semi-axes: $\sqrt{E}$, $\sqrt{E/\epsilon}$"
                "\n" +
                r"control diameter: $0\;\longleftrightarrow\;(1+z)\beta^\star$",
                transform=ax.transAxes, ha="left", va="top", fontsize=9,
                color="0.28",
                bbox={"facecolor": "white", "edgecolor": "none",
                      "alpha": 0.88, "pad": 1.2},
            )
            fig.subplots_adjust(
                left=0.14, right=0.96, bottom=0.10, top=0.93)
    else:
        figure_width = 14.0 if label_mode == "caption" else 11.0
        fig, axes = plt.subplots(
            1, 2, figsize=(figure_width, 5.3), sharex=True, sharey=True)
        fig.suptitle(title, fontsize=14, y=0.985)
        for ax in axes:
            add_covariance_ellipse(ax, epsilon)
            add_teacher(ax, beta_star)
            format_axis(ax, epsilon, xlim, ylim)

        axes[0].set_title(
            "a  Prediction: covariance-weighted ellipses", loc="left")
        prediction_rows = add_prediction_contours(
            axes[0], beta_star, epsilon, prediction_errors, signal_power,
            label_mode=label_mode)

        axes[1].set_title("b  Control: Euclidean spheres", loc="left")
        control_rows = add_control_contours(
            axes[1], beta_star, z_values, label_mode=label_mode)
        if label_mode == "caption":
            add_metric_caption(
                axes[1], epsilon, beta_star, prediction_errors, z_values)
            fig.subplots_adjust(
                left=0.06, right=0.72, bottom=0.12, top=0.88, wspace=0.13)
        else:
            axes[0].text(
                0.02, 0.98,
                r"semi-axes $\sqrt{E}$ and $\sqrt{E/\epsilon}$",
                transform=axes[0].transAxes, ha="left", va="top", fontsize=9,
                color="0.28",
            )
            axes[1].text(
                0.02, 0.98,
                r"diameter: $0\;\longleftrightarrow\;(1+z)\beta^\star$",
                transform=axes[1].transAxes, ha="left", va="top", fontsize=9,
                color="0.28",
            )
            fig.subplots_adjust(
                left=0.075, right=0.985, bottom=0.12, top=0.88,
                wspace=0.13)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    if pdf_output is not None:
        pdf_output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(pdf_output, bbox_inches="tight")
    plt.close(fig)

    rows = prediction_rows + control_rows
    if table_output is not None:
        write_metadata(table_output, rows, epsilon, beta_star)

    # Cheap geometric checks guard against accidental formula drift.
    sigma = np.diag([1.0, epsilon])
    for error in prediction_errors:
        point = beta_star + np.array([np.sqrt(error), 0.0])
        delta = point - beta_star
        assert np.isclose(delta @ sigma @ delta, error)
    for z in z_values:
        center, radius = control_circle(beta_star, z)
        point = center + radius * np.array([0.0, 1.0])
        numerator = point @ point
        denominator = point @ beta_star
        assert np.isclose(numerator / denominator - 1.0, z)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--layout", choices=("overlay", "side-by-side"), default="overlay",
        help="Draw both geometries together or in separate panels.")
    parser.add_argument(
        "--label-mode", choices=("caption", "inline"), default="caption",
        help=("Use minimal labels plus a side metric caption, or put full "
              "metrics directly on the contours."))
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=1.1)
    parser.add_argument("--beta2", type=float, default=0.65)
    parser.add_argument(
        "--prediction-errors", type=float, nargs="+",
        default=(0.04, 0.14, 0.36),
        help="Natural prediction-error contour levels.")
    parser.add_argument(
        "--control-z", type=float, nargs="+",
        default=(-1.0 / 3.0, 0.0, 1.0),
        help="Signed z levels for the control circles.")
    parser.add_argument(
        "--xlim", type=float, nargs=2, metavar=("MIN", "MAX"),
        help="Optional manual x-axis limits.")
    parser.add_argument(
        "--ylim", type=float, nargs=2, metavar=("MIN", "MAX"),
        help="Optional manual y-axis limits.")
    parser.add_argument("--output", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--pdf-output", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--table-output", type=Path, default=DEFAULT_TABLE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    make_figure(
        epsilon=args.epsilon,
        beta_star=np.array([args.beta1, args.beta2], dtype=float),
        prediction_errors=tuple(args.prediction_errors),
        z_values=tuple(args.control_z),
        layout=args.layout,
        label_mode=args.label_mode,
        output=args.output,
        pdf_output=args.pdf_output,
        table_output=args.table_output,
        requested_xlim=tuple(args.xlim) if args.xlim else None,
        requested_ylim=tuple(args.ylim) if args.ylim else None,
    )
    print(f"Saved {args.output}")
    print(f"Saved {args.pdf_output}")
    print(f"Saved {args.table_output}")


if __name__ == "__main__":
    main()

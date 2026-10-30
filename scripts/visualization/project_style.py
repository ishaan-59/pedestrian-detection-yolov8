"""Shared Matplotlib styling for research and deployment figures."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


SKY_BLUE = "#87CEEB"
SALMON = "#FA8072"
MEDIUM_SEA_GREEN = "#3CB371"
CHARCOAL = "#222222"
MID_GRAY = "#777777"

EDGE_COLOR = "black"
GRID_STYLE = "--"
GRID_ALPHA = 0.7
GRID_LINEWIDTH = 0.8

FIGURE_DPI = 300
FIGSIZE_SINGLE = (10, 6)
FIGSIZE_WIDE = (15, 6)
FIGSIZE_LARGE = (15, 9)


def apply_project_style() -> None:
    """Apply the repository's existing clean Matplotlib visual language."""

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.edgecolor": EDGE_COLOR,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 16,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": FIGURE_DPI,
            "svg.fonttype": "none",
            "svg.hashsalt": "pedestrian-detection-yolov8",
        }
    )


def style_axis(axis, grid_axis: str = "x") -> None:
    """Apply the shared grid, spine, and layer conventions to an axis."""

    axis.grid(
        axis=grid_axis,
        linestyle=GRID_STYLE,
        alpha=GRID_ALPHA,
        linewidth=GRID_LINEWIDTH,
    )
    axis.set_axisbelow(True)

    for spine in axis.spines.values():
        spine.set_color(EDGE_COLOR)
        spine.set_linewidth(0.8)


def save_figure(
    figure: Figure,
    output_stem: Path,
    formats: tuple[str, ...] = ("png",),
) -> list[Path]:
    """Save a figure in one or more publication-friendly formats."""

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    output_paths = []

    for file_format in formats:
        output_path = output_stem.with_suffix(f".{file_format}")
        save_options = {"bbox_inches": "tight", "format": file_format}

        if file_format == "png":
            save_options["dpi"] = FIGURE_DPI
            save_options["metadata"] = {
                "Software": "Matplotlib; pedestrian-detection-yolov8",
            }
        elif file_format == "svg":
            save_options["metadata"] = {
                "Creator": "Matplotlib; pedestrian-detection-yolov8",
                "Date": None,
            }

        figure.savefig(output_path, **save_options)

        if file_format == "svg":
            svg_text = output_path.read_text(encoding="utf-8")
            normalized_svg = "\n".join(
                line.rstrip() for line in svg_text.splitlines()
            ) + "\n"
            output_path.write_text(normalized_svg, encoding="utf-8")

        output_paths.append(output_path)

    plt.close(figure)
    return output_paths

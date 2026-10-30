"""Generate reproducible Jetson benchmark figures from canonical CSV files."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from project_style import (
    CHARCOAL,
    EDGE_COLOR,
    FIGSIZE_SINGLE,
    FIGSIZE_WIDE,
    MEDIUM_SEA_GREEN,
    MID_GRAY,
    SALMON,
    SKY_BLUE,
    apply_project_style,
    save_figure,
    style_axis,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUMMARY_DIR = PROJECT_ROOT / "deployment" / "jetson" / "benchmarks" / "summary"
OUTPUT_DIR = PROJECT_ROOT / "deployment" / "jetson" / "artifacts" / "plots"

BENCHMARK_MATRIX_FILE = SUMMARY_DIR / "benchmark_matrix.csv"
POWER_MODES_FILE = SUMMARY_DIR / "power_modes.csv"
VIDEO_PIPELINE_FILE = SUMMARY_DIR / "video_pipeline_resolution.csv"

INPUT_SIZES = [512, 640, 768]
RUNTIME_SPECS = [
    ("pytorch", "fp32", "PyTorch FP32", SKY_BLUE),
    ("tensorrt", "fp32", "TensorRT FP32", SALMON),
    ("tensorrt", "fp16", "TensorRT FP16", MEDIUM_SEA_GREEN),
]
POWER_MODE_COLORS = {
    "15W": SKY_BLUE,
    "25W": MEDIUM_SEA_GREEN,
    "MAXN_SUPER": SALMON,
}


def require_columns(dataframe: pd.DataFrame, columns: set[str], source: Path) -> None:
    """Fail clearly when a canonical source does not match the expected schema."""

    missing_columns = columns - set(dataframe.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"{source} is missing required columns: {missing_text}")


def load_benchmark_matrix() -> pd.DataFrame:
    dataframe = pd.read_csv(BENCHMARK_MATRIX_FILE)
    require_columns(
        dataframe,
        {
            "runtime",
            "precision",
            "resolution",
            "map50_95",
            "inference_ms",
            "throughput_img_s",
        },
        BENCHMARK_MATRIX_FILE,
    )

    expected_pairs = {(runtime, precision) for runtime, precision, _, _ in RUNTIME_SPECS}
    actual_pairs = set(zip(dataframe["runtime"], dataframe["precision"]))
    if actual_pairs != expected_pairs:
        raise ValueError(
            f"Unexpected runtime/precision pairs in {BENCHMARK_MATRIX_FILE}: "
            f"{sorted(actual_pairs)}"
        )

    if set(dataframe["resolution"]) != set(INPUT_SIZES):
        raise ValueError(
            f"Expected input sizes {INPUT_SIZES} in {BENCHMARK_MATRIX_FILE}"
        )

    duplicate_rows = dataframe.duplicated(["runtime", "precision", "resolution"])
    if duplicate_rows.any() or len(dataframe) != len(RUNTIME_SPECS) * len(INPUT_SIZES):
        raise ValueError("Canonical benchmark matrix must contain exactly one 3×3 grid")

    return dataframe


def ordered_runtime_rows(
    dataframe: pd.DataFrame,
    runtime: str,
    precision: str,
) -> pd.DataFrame:
    rows = dataframe[
        (dataframe["runtime"] == runtime)
        & (dataframe["precision"] == precision)
    ].copy()
    rows = rows.set_index("resolution").reindex(INPUT_SIZES).reset_index()

    if rows.isna().any().any():
        raise ValueError(f"Incomplete rows for {runtime} {precision}")

    return rows


def plot_runtime_performance(dataframe: pd.DataFrame) -> list[Path]:
    """Plot canonical inference latency and measured validation throughput."""

    figure, axes = plt.subplots(nrows=1, ncols=2, figsize=FIGSIZE_WIDE)
    x_positions = np.arange(len(INPUT_SIZES))
    bar_width = 0.24

    metric_specs = [
        (axes[0], "inference_ms", "Inference latency (ms/image)", "Neural-network inference"),
        (axes[1], "throughput_img_s", "Measured throughput (images/s)", "Full-validation throughput"),
    ]

    for axis, metric_column, y_label, panel_title in metric_specs:
        for runtime_index, (runtime, precision, label, color) in enumerate(RUNTIME_SPECS):
            rows = ordered_runtime_rows(dataframe, runtime, precision)
            offsets = x_positions + (runtime_index - 1) * bar_width
            bars = axis.bar(
                offsets,
                rows[metric_column],
                width=bar_width,
                color=color,
                edgecolor=EDGE_COLOR,
                linewidth=0.8,
                label=label,
            )
            axis.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)

        axis.set_title(panel_title)
        axis.set_xlabel("Model input size (pixels)")
        axis.set_ylabel(y_label)
        axis.set_xticks(x_positions, [str(size) for size in INPUT_SIZES])
        axis.set_ylim(bottom=0, top=axis.get_ylim()[1] * 1.12)
        style_axis(axis, grid_axis="y")

    handles, labels = axes[0].get_legend_handles_labels()
    figure.suptitle("Jetson Runtime Performance Across Model Input Sizes")
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=3,
        frameon=False,
    )
    figure.tight_layout(rect=[0, 0, 1, 0.86], w_pad=3.0)

    return save_figure(
        figure,
        OUTPUT_DIR / "jetson_runtime_performance_by_input",
        formats=("png", "svg"),
    )


def plot_accuracy_latency_tradeoff(dataframe: pd.DataFrame) -> list[Path]:
    """Plot BDD100K accuracy against neural-network inference latency."""

    figure, axis = plt.subplots(figsize=FIGSIZE_SINGLE)

    for runtime, precision, label, color in RUNTIME_SPECS:
        rows = ordered_runtime_rows(dataframe, runtime, precision)
        axis.plot(
            rows["inference_ms"],
            rows["map50_95"],
            color=color,
            marker="o",
            markersize=7,
            markeredgecolor=EDGE_COLOR,
            linewidth=2,
            label=label,
        )

        for row in rows.itertuples(index=False):
            axis.annotate(
                f"{int(row.resolution)} px\n{row.map50_95:.3f}",
                (row.inference_ms, row.map50_95),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                va="bottom",
                fontsize=8,
                color=CHARCOAL,
            )

    latency_span = dataframe["inference_ms"].max() - dataframe["inference_ms"].min()
    accuracy_span = dataframe["map50_95"].max() - dataframe["map50_95"].min()
    axis.set_xlim(
        dataframe["inference_ms"].min() - latency_span * 0.08,
        dataframe["inference_ms"].max() + latency_span * 0.08,
    )
    axis.set_ylim(
        dataframe["map50_95"].min() - accuracy_span * 0.20,
        dataframe["map50_95"].max() + accuracy_span * 0.32,
    )
    axis.set_xlabel("Inference latency (ms/image)")
    axis.set_ylabel("BDD100K mAP50–95")
    axis.set_title("Input labels show the measured accuracy–latency operating points")
    axis.legend(loc="lower right", frameon=False)
    style_axis(axis, grid_axis="both")
    figure.suptitle("Jetson Accuracy–Latency Tradeoff")
    figure.tight_layout(rect=[0, 0, 1, 0.94])

    return save_figure(
        figure,
        OUTPUT_DIR / "jetson_accuracy_latency_tradeoff",
        formats=("png", "svg"),
    )


def plot_power_mode_tradeoff() -> list[Path]:
    """Plot sampled board power against inference latency by power mode."""

    dataframe = pd.read_csv(POWER_MODES_FILE)
    require_columns(
        dataframe,
        {"power_mode", "inference_ms", "avg_power_w"},
        POWER_MODES_FILE,
    )

    mode_order = ["15W", "25W", "MAXN_SUPER"]
    if set(dataframe["power_mode"]) != set(mode_order):
        raise ValueError(f"Unexpected power modes in {POWER_MODES_FILE}")

    rows = dataframe.set_index("power_mode").reindex(mode_order).reset_index()
    figure, axis = plt.subplots(figsize=FIGSIZE_SINGLE)
    axis.plot(
        rows["avg_power_w"],
        rows["inference_ms"],
        color=MID_GRAY,
        linestyle="--",
        linewidth=1.5,
        zorder=1,
    )

    for row in rows.itertuples(index=False):
        color = POWER_MODE_COLORS[row.power_mode]
        axis.scatter(
            row.avg_power_w,
            row.inference_ms,
            s=110,
            color=color,
            edgecolor=EDGE_COLOR,
            linewidth=0.9,
            zorder=2,
        )
        axis.annotate(
            f"{row.power_mode}\n{row.avg_power_w:.2f} W, {row.inference_ms:.1f} ms",
            (row.avg_power_w, row.inference_ms),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    x_span = rows["avg_power_w"].max() - rows["avg_power_w"].min()
    y_span = rows["inference_ms"].max() - rows["inference_ms"].min()
    axis.set_xlim(
        rows["avg_power_w"].min() - x_span * 0.15,
        rows["avg_power_w"].max() + x_span * 0.15,
    )
    axis.set_ylim(
        rows["inference_ms"].min() - y_span * 0.18,
        rows["inference_ms"].max() + y_span * 0.32,
    )
    axis.set_xlabel("Average sampled board power (W)")
    axis.set_ylabel("Inference latency (ms/image)")
    axis.set_title("TensorRT FP16 at 768-pixel model input")
    style_axis(axis, grid_axis="both")
    figure.suptitle("Jetson Power–Latency Tradeoff")
    figure.tight_layout(rect=[0, 0, 1, 0.94])

    return save_figure(
        figure,
        OUTPUT_DIR / "jetson_power_mode_tradeoff",
        formats=("png", "svg"),
    )


def plot_video_pipeline_throughput() -> list[Path]:
    """Plot measured end-to-end video throughput by source configuration."""

    dataframe = pd.read_csv(VIDEO_PIPELINE_FILE)
    require_columns(
        dataframe,
        {
            "source_resolution",
            "save_annotated_video",
            "frames",
            "wall_seconds",
            "end_to_end_fps",
        },
        VIDEO_PIPELINE_FILE,
    )

    boolean_text = dataframe["save_annotated_video"].astype(str).str.lower()
    if not set(boolean_text).issubset({"true", "false"}):
        raise ValueError(f"Invalid save_annotated_video values in {VIDEO_PIPELINE_FILE}")

    dataframe = dataframe.copy()
    dataframe["save_output"] = boolean_text == "true"
    dataframe["configuration"] = dataframe.apply(
        lambda row: (
            f"{str(row['source_resolution']).replace('x', '×')} — "
            f"{'save annotated video' if row['save_output'] else 'no output save'}"
        ),
        axis=1,
    )

    colors = [SALMON if save_output else SKY_BLUE for save_output in dataframe["save_output"]]
    figure, axis = plt.subplots(figsize=FIGSIZE_SINGLE)
    y_positions = np.arange(len(dataframe))
    bars = axis.barh(
        y_positions,
        dataframe["end_to_end_fps"],
        color=colors,
        edgecolor=EDGE_COLOR,
        linewidth=0.8,
        height=0.62,
    )
    axis.bar_label(bars, fmt="%.3f FPS", padding=4, fontsize=9)
    axis.set_yticks(y_positions, dataframe["configuration"])
    axis.invert_yaxis()
    axis.set_xlim(0, dataframe["end_to_end_fps"].max() * 1.22)
    axis.set_xlabel("Measured end-to-end throughput (frames/s)")
    axis.set_ylabel("Source-video configuration")
    axis.set_title("FP16 integrated-NMS engine at 768-pixel model input; 884 frames")
    axis.legend(
        handles=[
            Patch(facecolor=SKY_BLUE, edgecolor=EDGE_COLOR, label="No output saved"),
            Patch(facecolor=SALMON, edgecolor=EDGE_COLOR, label="Annotated video saved"),
        ],
        loc="upper right",
        frameon=False,
    )
    style_axis(axis, grid_axis="x")
    figure.suptitle("Jetson Video-Pipeline Throughput")
    figure.tight_layout(rect=[0, 0, 1, 0.94])

    return save_figure(
        figure,
        OUTPUT_DIR / "jetson_video_pipeline_throughput",
        formats=("png", "svg"),
    )


def main() -> None:
    apply_project_style()
    benchmark_matrix = load_benchmark_matrix()

    generated_paths = []
    generated_paths.extend(plot_runtime_performance(benchmark_matrix))
    generated_paths.extend(plot_accuracy_latency_tradeoff(benchmark_matrix))
    generated_paths.extend(plot_power_mode_tradeoff())
    generated_paths.extend(plot_video_pipeline_throughput())

    print("Generated Jetson benchmark figures:")
    for path in generated_paths:
        print(f"- {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()

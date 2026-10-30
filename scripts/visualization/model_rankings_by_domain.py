"""Generate model-ranking panels from the canonical performance matrix."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from project_style import (
    EDGE_COLOR,
    FIGSIZE_LARGE,
    SKY_BLUE,
    apply_project_style,
    save_figure,
    style_axis,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = PROJECT_ROOT / "analysis" / "summary_tables" / "model_performance_matrix.csv"
OUTPUT_STEM = PROJECT_ROOT / "analysis" / "figures" / "model_rankings_by_domain"


def main() -> None:
    apply_project_style()
    dataframe = pd.read_csv(INPUT_FILE)

    figure, axes_grid = plt.subplots(nrows=2, ncols=2, figsize=FIGSIZE_LARGE)
    axes = axes_grid.flatten()

    for axis, dataset_id in zip(axes, dataframe["dataset_id"].unique()):
        rows = dataframe[dataframe["dataset_id"] == dataset_id].copy()
        rows = rows.sort_values("map50_95", ascending=False)

        bars = axis.barh(
            rows["model_id"],
            rows["map50_95"],
            color=SKY_BLUE,
            edgecolor=EDGE_COLOR,
            height=0.6,
        )
        axis.bar_label(bars, fmt="%.3f", padding=-8, fontsize=9)
        axis.set_title(f"{dataset_id} model comparison")
        axis.set_xlabel("mAP50–95")
        axis.set_ylabel("Models")
        axis.set_xlim(0, 0.6)
        axis.invert_yaxis()
        style_axis(axis, grid_axis="x")

    figure.suptitle("Model Rankings Across Evaluation Domains")
    figure.tight_layout(rect=[0, 0, 1, 0.96], h_pad=2.0, w_pad=2.5)

    output_path = save_figure(figure, OUTPUT_STEM, formats=("png",))[0]
    print(f"Saved plot to: {output_path}")


if __name__ == "__main__":
    main()

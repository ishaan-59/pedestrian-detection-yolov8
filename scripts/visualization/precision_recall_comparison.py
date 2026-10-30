"""Generate cross-domain precision/recall panels from canonical metrics."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from project_style import (
    EDGE_COLOR,
    FIGSIZE_LARGE,
    SALMON,
    SKY_BLUE,
    apply_project_style,
    save_figure,
    style_axis,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = PROJECT_ROOT / "analysis" / "summary_tables" / "model_performance_matrix.csv"
OUTPUT_STEM = PROJECT_ROOT / "analysis" / "figures" / "precision_recall_by_domain"

DATASETS = [
    "caltech_test",
    "citypersons_val",
    "crowdhuman_val",
    "bdd100k_val",
]
DATASET_TITLES = {
    "caltech_test": "Caltech Test",
    "citypersons_val": "CityPersons Validation",
    "crowdhuman_val": "CrowdHuman Validation",
    "bdd100k_val": "BDD100K Validation",
}


def annotate_score_bars(axis, bars) -> None:
    for bar in bars:
        value = bar.get_width()
        label_x = value - 0.02 if value >= 0.12 else value + 0.01
        alignment = "right" if value >= 0.12 else "left"
        axis.text(
            label_x,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            ha=alignment,
            fontsize=8,
        )


def main() -> None:
    apply_project_style()
    dataframe = pd.read_csv(INPUT_FILE)

    figure, axes_grid = plt.subplots(nrows=2, ncols=2, figsize=FIGSIZE_LARGE)
    axes = axes_grid.flatten()
    bar_height = 0.35
    legend_handles = None

    for axis, dataset_id in zip(axes, DATASETS):
        rows = dataframe[dataframe["dataset_id"] == dataset_id].copy()
        rows = rows.sort_values("map50_95", ascending=False)
        y_positions = np.arange(len(rows))

        precision_bars = axis.barh(
            y_positions - bar_height / 2,
            rows["precision"],
            height=bar_height,
            color=SKY_BLUE,
            edgecolor=EDGE_COLOR,
            label="Precision",
        )
        recall_bars = axis.barh(
            y_positions + bar_height / 2,
            rows["recall"],
            height=bar_height,
            color=SALMON,
            edgecolor=EDGE_COLOR,
            label="Recall",
        )

        axis.set_yticks(y_positions, rows["model_id"])
        axis.set_title(DATASET_TITLES[dataset_id])
        axis.set_xlabel("Score")
        axis.set_ylabel("Models")
        axis.set_xlim(0, 1)
        axis.invert_yaxis()
        style_axis(axis, grid_axis="x")

        annotate_score_bars(axis, precision_bars)
        annotate_score_bars(axis, recall_bars)

        if legend_handles is None:
            legend_handles = [precision_bars[0], recall_bars[0]]

    figure.suptitle("Precision–Recall Trade-offs Across Evaluation Domains")
    figure.legend(
        legend_handles,
        ["Precision", "Recall"],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=2,
        frameon=False,
    )
    figure.tight_layout(rect=[0, 0, 1, 0.90], h_pad=2.0, w_pad=2.5)

    output_path = save_figure(figure, OUTPUT_STEM, formats=("png",))[0]
    print(f"Saved plot to: {output_path}")


if __name__ == "__main__":
    main()

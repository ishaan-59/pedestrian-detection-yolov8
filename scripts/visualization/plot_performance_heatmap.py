"""Generate the cross-dataset mAP50–95 heatmap from the canonical matrix."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from project_style import FIGSIZE_SINGLE, apply_project_style, save_figure


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = PROJECT_ROOT / "analysis" / "summary_tables" / "model_performance_matrix.csv"
OUTPUT_STEM = PROJECT_ROOT / "analysis" / "figures" / "map50_95_heatmap"

DATASET_ORDER = [
    "caltech_test",
    "citypersons_val",
    "crowdhuman_val",
    "bdd100k_val",
]
MODEL_ORDER = [
    "caltech_only",
    "caltech_to_citypersons",
    "citypersons_only",
    "citypersons_to_crowdhuman",
    "crowdhuman_only",
    "crowdhuman_to_citypersons",
]


def main() -> None:
    apply_project_style()

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Performance matrix path invalid: {INPUT_FILE}")

    dataframe = pd.read_csv(INPUT_FILE)
    required_columns = {"model_id", "dataset_id", "map50_95"}
    missing_columns = required_columns - set(dataframe.columns)
    if missing_columns:
        raise ValueError(f"Missing columns: {sorted(missing_columns)}")

    heatmap_data = dataframe.pivot(
        index="model_id",
        columns="dataset_id",
        values="map50_95",
    ).loc[MODEL_ORDER, DATASET_ORDER]

    figure, axis = plt.subplots(figsize=FIGSIZE_SINGLE)
    image = axis.imshow(heatmap_data.values, aspect="auto", cmap="viridis")

    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("mAP50–95")

    axis.set_xticks(range(len(heatmap_data.columns)))
    axis.set_yticks(range(len(heatmap_data.index)))
    axis.set_xticklabels(heatmap_data.columns)
    axis.set_yticklabels(heatmap_data.index)
    plt.setp(axis.get_xticklabels(), rotation=30, ha="right")

    text_threshold = heatmap_data.to_numpy().max() / 2
    for row_index in range(len(heatmap_data.index)):
        for column_index in range(len(heatmap_data.columns)):
            value = heatmap_data.iloc[row_index, column_index]
            axis.text(
                column_index,
                row_index,
                f"{value:.3f}",
                ha="center",
                va="center",
                color="black" if value > text_threshold else "white",
            )

    axis.set_title("Cross-Dataset Pedestrian Detection Performance")
    axis.set_xlabel("Evaluation Dataset")
    axis.set_ylabel("Trained Model")
    figure.tight_layout()

    output_path = save_figure(figure, OUTPUT_STEM, formats=("png",))[0]
    print(f"Saved heatmap to: {output_path}")


if __name__ == "__main__":
    main()

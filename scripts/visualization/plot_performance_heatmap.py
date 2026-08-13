from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (PROJECT_ROOT / "experiments" / "summary_tables" / "model_performance_matrix.csv")
OUTPUT_FILE = (PROJECT_ROOT / "docs" / "figures" / "map50_95_heatmap.png")

def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Performance matrix path invalid: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)
    columns_needed = {"model_id", "dataset_id", "map50_95"}
    missing_columns = columns_needed - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")
    
    heatmap_data = df.pivot(index="model_id", columns = "dataset_id", values = "map50_95")
    
    heatmap_data = df.pivot(
        index="model_id",
        columns="dataset_id",
        values="map50_95",
    )

    dataset_order = [
        "caltech_test",
        "citypersons_val",
        "crowdhuman_val",
        "bdd100k_val",
    ]

    model_order = [
        "caltech_only",
        "caltech_to_citypersons",
        "citypersons_only",
        "citypersons_to_crowdhuman",
        "crowdhuman_only",
        "crowdhuman_to_citypersons",
    ]

    heatmap_data = heatmap_data.loc[
        model_order,
        dataset_order,
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    image = ax.imshow(
        heatmap_data.values,
        aspect="auto",
    )

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("mAP50–95")

    ax.set_xticks(range(len(heatmap_data.columns)))
    ax.set_yticks(range(len(heatmap_data.index)))

    ax.set_xticklabels(heatmap_data.columns)
    ax.set_yticklabels(heatmap_data.index)

    plt.setp(
        ax.get_xticklabels(),
        rotation=30,
        ha="right",
    )

    threshold = heatmap_data.to_numpy().max() / 2

    for row_index in range(len(heatmap_data.index)):
        for column_index in range(len(heatmap_data.columns)):
            value = heatmap_data.iloc[row_index, column_index]

            text_color = "black" if value > threshold else "white"

            ax.text(
                column_index,
                row_index,
                f"{value:.3f}",
                ha="center",
                va="center",
                color=text_color,
            )
            
    ax.set_title("Cross-Dataset Pedestrian Detection Performance")
    ax.set_xlabel("Evaluation Dataset")
    ax.set_ylabel("Trained Model")

    fig.tight_layout()
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        OUTPUT_FILE,
        dpi=300,
        bbox_inches="tight",
    )

    print(f"Saved heatmap to: {OUTPUT_FILE}")


    plt.show()

if __name__ == "__main__":
    main()

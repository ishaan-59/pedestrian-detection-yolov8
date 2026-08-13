import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (PROJECT_ROOT / "experiments" / "summary_tables" / "model_performance_matrix.csv")
OUTPUT_FILE = (PROJECT_ROOT / "docs" / "figures" / "model_rankings_by_domain.png")


def main():
    df = pd.read_csv(INPUT_FILE)
    
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(15, 9))
    axes = ax.flatten()

    datasets_list = df['dataset_id'].unique().tolist()
    
    for i, dataset_id in enumerate(datasets_list):
        current_ax = axes[i]

        row = df[df["dataset_id"] == dataset_id].copy()
        row = row.sort_values('map50_95', ascending=False)

        bars = current_ax.barh(row['model_id'], row['map50_95'],color='skyblue', edgecolor='black', height=0.6) 

        for bar in bars:
            value = bar.get_width()

            current_ax.text(
                value - 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.3f}",
                va="center",
                ha="right",
                fontsize=9
            )

        current_ax.set_title(f"{dataset_id} model comparison")
        current_ax.set_xlabel("mAP50-95")
        current_ax.set_ylabel("Models")
        current_ax.set_xlim(0, 0.6)
        current_ax.grid(axis='x', linestyle='--', alpha=0.7)
        current_ax.invert_yaxis()
        current_ax.set_axisbelow(True) 
    fig.subplots_adjust(hspace=0.38, wspace=0.40)
    fig.suptitle("Model Rankings Across Evaluation Domains", fontsize=16)
    fig.tight_layout()

    fig.savefig(
                OUTPUT_FILE,
                dpi=300,
                bbox_inches="tight",
            )

    print(f"Saved plot to: {OUTPUT_FILE}")
    plt.show()

if __name__ == "__main__":
    main()


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "experiments" / "summary_tables" / "model_performance_matrix.csv"
OUTPUT_FILE = PROJECT_ROOT / "docs" / "figures" / "precision_recall_by_domain.png"


def main():
    df = pd.read_csv(INPUT_FILE)

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(15, 9))
    axes = ax.flatten()

    datasets_list = ["caltech_test", "citypersons_val", "crowdhuman_val", "bdd100k_val"]

    dataset_titles = {"caltech_test": "Caltech Test", "citypersons_val": "CityPersons Validation", "crowdhuman_val": "CrowdHuman Validation", "bdd100k_val": "BDD100K Validation"}

    bar_height = 0.35
    legend_handles = None

    for i, dataset_id in enumerate(datasets_list):
        current_ax = axes[i]

        row = df[df['dataset_id'] == dataset_id].copy()
        row = row.sort_values('map50_95', ascending=False)

        y_positions = np.arange(len(row))

        precision_bars = current_ax.barh(y_positions - bar_height / 2, row['precision'], height=bar_height, color='skyblue', edgecolor='black', label='Precision')
        recall_bars = current_ax.barh(y_positions + bar_height / 2, row['recall'], height=bar_height, color='salmon', edgecolor='black', label='Recall')

        current_ax.set_yticks(y_positions)
        current_ax.set_yticklabels(row['model_id'])

        current_ax.set_title(dataset_titles[dataset_id])
        current_ax.set_xlabel("Score")
        current_ax.set_ylabel("Models")
        current_ax.set_xlim(0, 1)
        current_ax.grid(axis='x', linestyle='--', alpha=0.7)
        current_ax.invert_yaxis()
        current_ax.set_axisbelow(True)

        for bar in precision_bars:
            value = bar.get_width()

            if value >= 0.12:
                label_x = value - 0.02
                horizontal_alignment = 'right'
            else:
                label_x = value + 0.01
                horizontal_alignment = 'left'

            current_ax.text(label_x, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va='center', ha=horizontal_alignment, fontsize=8)

        for bar in recall_bars:
            value = bar.get_width()

            if value >= 0.12:
                label_x = value - 0.02
                horizontal_alignment = 'right'
            else:
                label_x = value + 0.01
                horizontal_alignment = 'left'

            current_ax.text(label_x, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va='center', ha=horizontal_alignment, fontsize=8)

        if legend_handles is None:
            legend_handles = [precision_bars[0], recall_bars[0]]

    fig.suptitle("Precision-Recall Trade-offs Across Evaluation Domains", fontsize=16)

    fig.legend(legend_handles, ['Precision', 'Recall'], loc='upper center', bbox_to_anchor=(0.5, 0.94), ncol=2, frameon=False)

    fig.tight_layout(rect=[0, 0, 1, 0.90], h_pad=2.0, w_pad=2.5)

    fig.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")

    print(f"Saved plot to: {OUTPUT_FILE}")
    plt.show()


if __name__ == "__main__":
    main()

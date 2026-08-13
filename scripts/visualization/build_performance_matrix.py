from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

VALIDATION_ROOT = PROJECT_ROOT / "cross_dataset_validation"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "summary_tables"
OUTPUT_FILE = OUTPUT_DIR / "model_performance_matrix.csv"


def main() -> None:
    metric_files = sorted(VALIDATION_ROOT.glob("*/*/metrics.csv"))

    if not metric_files:
        raise FileNotFoundError(
            f"No metrics.csv files found under {VALIDATION_ROOT}"
        )

    rows = []

    for metrics_file in metric_files:
        df = pd.read_csv(metrics_file)

        if df.empty:
            print(f"Skipping empty file: {metrics_file}")
            continue

        row = df.iloc[0]

        evaluation_group = metrics_file.parents[1].name
        experiment_folder = metrics_file.parent.name

        required_columns = [
            "model_id",
            "dataset_id",
            "split",
            "precision",
            "recall",
            "f1",
            "map50",
            "map75",
            "map50_95",
        ]

        missing_columns = [
            column for column in required_columns
            if column not in df.columns
        ]

        if missing_columns:
            print(
                f"Skipping {metrics_file.relative_to(PROJECT_ROOT)} "
                f"because it is missing: {missing_columns}"
            )
            continue

        rows.append(
            {
                "model_id": row["model_id"],
                "dataset_id": row["dataset_id"],
                "split": row["split"],
                "evaluation_group": evaluation_group,
                "experiment_folder": experiment_folder,
                "precision": row["precision"],
                "recall": row["recall"],
                "f1": row["f1"],
                "map50": row["map50"],
                "map75": row["map75"],
                "map50_95": row["map50_95"],
                "runtime_seconds": row.get("runtime_seconds"),
                "preprocess_ms": row.get("preprocess_ms"),
                "inference_ms": row.get("inference_ms"),
                "postprocess_ms": row.get("postprocess_ms"),
                "source_metrics_file": str(
                    metrics_file.relative_to(PROJECT_ROOT)
                ),
            }
        )

    if not rows:
        raise RuntimeError(
            "No valid metrics rows were found."
        )

    result = pd.DataFrame(rows)

    model_order = [
        "caltech_only",
        "caltech_to_citypersons",
        "citypersons_only",
        "citypersons_to_crowdhuman",
        "crowdhuman_only",
        "crowdhuman_to_citypersons",
    ]

    dataset_order = [
        "caltech_test",
        "citypersons_val",
        "crowdhuman_val",
        "bdd100k_val",
    ]

    result["model_id"] = pd.Categorical(
        result["model_id"],
        categories=model_order,
        ordered=True,
    )

    result["dataset_id"] = pd.Categorical(
        result["dataset_id"],
        categories=dataset_order,
        ordered=True,
    )

    result = result.sort_values(
        ["dataset_id", "model_id"]
    ).reset_index(drop=True)

    result["model_id"] = result["model_id"].astype(str)
    result["dataset_id"] = result["dataset_id"].astype(str)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_FILE, index=False)

    print(f"Found {len(metric_files)} metrics files.")
    print(f"Saved {len(result)} rows to:")
    print(OUTPUT_FILE)

    expected_runs = 4 * 6

    if len(result) != expected_runs:
        print(
            f"Warning: expected {expected_runs} runs "
            f"but saved {len(result)}."
        )


if __name__ == "__main__":
    main()

    
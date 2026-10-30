# Reproducible Visualizations

All quantitative project figures are generated with Python, Matplotlib,
NumPy, and Pandas. Run commands from the repository root after installing
[`requirements.txt`](../../requirements.txt).

## Shared Visual Style

[`project_style.py`](project_style.py) centralizes the visual conventions
already established by the research figures:

- DejaVu Sans typography on white backgrounds;
- 16-point figure titles and 12-point panel titles;
- black spines and bar/marker edges;
- dashed grids with 0.7 alpha;
- `skyblue` and `salmon` as the primary comparison colors;
- a related medium sea green for the third Jetson runtime;
- three-decimal accuracy labels and one-decimal latency/throughput labels;
- 300-DPI PNG output and text-preserving SVG output for Jetson charts.

Runtime colors remain fixed across all deployment figures:

| Runtime | Color |
|---|---|
| PyTorch FP32 | Sky blue (`#87CEEB`) |
| TensorRT FP32 | Salmon (`#FA8072`) |
| TensorRT FP16 | Medium sea green (`#3CB371`) |

## Research Figures

The existing research figures are retained in [`analysis/figures/`](../../analysis/figures/).
Their scripts now read the canonical matrix from
[`analysis/summary_tables/model_performance_matrix.csv`](../../analysis/summary_tables/model_performance_matrix.csv)
and write directly to the committed figure paths.

```bash
python scripts/visualization/plot_performance_heatmap.py
python scripts/visualization/model_rankings_by_domain.py
python scripts/visualization/precision_recall_comparison.py
```

The canonical research matrix itself can be rebuilt from the 24 individual
`metrics.csv` files with:

```bash
python scripts/visualization/build_performance_matrix.py
```

## Jetson Figures

One script generates the complete deployment figure set from canonical summary
CSVs without embedding benchmark measurements in Python source:

```bash
python scripts/visualization/plot_jetson_benchmarks.py
```

| Figure | Canonical source |
|---|---|
| Runtime latency and validation throughput | `deployment/jetson/benchmarks/summary/benchmark_matrix.csv` |
| Accuracy–latency tradeoff | `deployment/jetson/benchmarks/summary/benchmark_matrix.csv` |
| Power-mode tradeoff | `deployment/jetson/benchmarks/summary/power_modes.csv` |
| Video-pipeline throughput | `deployment/jetson/benchmarks/summary/video_pipeline_resolution.csv` |

Each deployment figure is exported as a high-resolution PNG and scalable SVG
under [`deployment/jetson/artifacts/plots/`](../../deployment/jetson/artifacts/plots/).
The script validates required columns and the expected canonical benchmark grid
before rendering.

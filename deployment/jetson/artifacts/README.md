# Deployment Artifact Availability

The Git repository retains documentation, aggregate CSVs, raw text logs,
configuration examples, scripts, and lightweight system metadata. Large or
device-specific artifacts remain outside Git to keep the project inspectable
and portable.

## Reproducible Plots

[`plots/`](plots/) contains the committed high-resolution PNG and SVG benchmark
figures. They are generated from the canonical summary CSVs by
[`../../../scripts/visualization/plot_jetson_benchmarks.py`](../../../scripts/visualization/plot_jetson_benchmarks.py)
using the same shared Matplotlib style as the research figures. No benchmark
measurement is manually embedded in the plotting code.

## Intentionally Excluded

| Artifact class | Local experiment role | Why it is excluded |
|---|---|---|
| BDD100K validation images and labels | 10,000-image accuracy benchmark | Dataset redistribution and repository size |
| YOLO `.pt` checkpoint | Selected CrowdHuman-only model | Large binary; model weights are ignored project-wide |
| ONNX export | TensorRT intermediate | Regenerable 99 MB binary |
| TensorRT engines | FP32, FP16, integrated-NMS, and INT8 experiments | Device/runtime-specific binaries; rebuild on the target Jetson |
| INT8 calibration cache | Prior calibration attempt | Machine-generated cache; retained only in the local experiment backup |
| Source and annotated videos | Qualitative and pipeline benchmarks | Large media files; source footage is credited below |
| Generated predictions and auxiliary Mac/MPS validation plots | Demonstration and noncanonical runs | Redundant with retained research figures or unsupported as canonical Jetson evidence |

Checksums and byte sizes for the primary excluded artifacts are recorded in
[`artifact_manifest.csv`](artifact_manifest.csv). Their original paths also
remain visible in raw logs and summary metadata.

## Reproducing the Deployment Assets

1. Supply the selected CrowdHuman-only YOLOv8m checkpoint and verify SHA-256:

   ```text
   741d5cecb3326549bba2aef8248febfed02200576fb246a042e9bf190b3eabd6
   ```

2. Prepare BDD100K validation in YOLO single-class format and update
   [`../configs/bdd100k_val.example.yaml`](../configs/bdd100k_val.example.yaml).
   The audited dataset contained 10,000 images, 13,262 `person` boxes, and
   6,780 images with empty person-label files.
3. Build TensorRT engines on the target Jetson/runtime using the commands in
   the [deployment README](../README.md#13-docker--deployment-workflow).
4. Re-encode the benchmark video only if reproducing the source-resolution
   sweep. All three encodes must represent the same 884-frame sequence.

TensorRT plan files are not portable build products. A checked-in engine could
silently couple results to one Jetson, TensorRT release, and builder setup, so
the repository preserves its export and validation evidence instead.

## Video Attribution

The benchmark sequence is derived from the same Pexels footage used for the
project's qualitative comparison:

[People and Vehicles on the Streets during Daytime](https://www.pexels.com/video/people-and-vehicles-on-the-streets-during-daytime-2954065/)
by **George Morina**, via Pexels under the [Pexels
License](https://www.pexels.com/license/).

The footage was used for qualitative inspection and Jetson pipeline testing,
not for model training or BDD100K accuracy evaluation.

# Benchmark Evidence and Provenance

This directory separates publishable aggregate results from raw experiment
evidence. Raw logs are the primary evidence; summary CSVs are convenient views,
not substitutes for retained measurements.

## Canonical Sources

| Summary | Role | Evidence status |
|---|---|---|
| [`summary/benchmark_matrix.csv`](summary/benchmark_matrix.csv) | Canonical 3×3 backend / precision / input-size matrix | **Raw-verified.** Every row matches its validation log for dataset counts, metrics, stage timings, progress duration, and reported iteration rate. |
| [`summary/power_modes.csv`](summary/power_modes.csv) | FP16 768 comparison at 15W, 25W, and MAXN_SUPER | **Raw-verified.** Inference values match validation logs; telemetry aggregates reproduce from tegrastats via [`../scripts/parse_power_modes.py`](../scripts/parse_power_modes.py). |
| [`summary/repeated_latency_fp16_768.csv`](summary/repeated_latency_fp16_768.csv) | Regular FP16 repeated latency | **Summary-only timing.** The generating script is retained, but raw console output is not. |
| [`summary/repeated_latency_fp16_768_nms.csv`](summary/repeated_latency_fp16_768_nms.csv) | Integrated-NMS repeated latency | **Summary-only timing.** The generating script is retained, but raw console output is not. |
| [`summary/engine_roles.csv`](summary/engine_roles.csv) | Separates canonical accuracy from integrated-NMS deployment testing | **Mixed.** Regular-engine accuracy is raw-verified; repeated timings are summary-only; no textual raw log remains for the integrated-NMS accuracy row. |
| [`summary/video_pipeline_resolution.csv`](summary/video_pipeline_resolution.csv) | Source-resolution / output-saving pipeline comparison | **Summary-only wall timing.** Media metadata verifies dimensions and 884 frames, and FPS arithmetic is consistent, but the four raw timer outputs are not retained. |
| [`summary/sustained_video_1080p.csv`](summary/sustained_video_1080p.csv) | 1080p integrated-NMS sustained run | **Summary-only wall timing.** No raw timer output or synchronized telemetry was retained. |

No canonical matrix row was changed by this audit. Secondary summary-only rows
remain available, but their evidence limitations must travel with any claim.

## Raw Evidence Layout

```text
deployment/jetson/
├── benchmarks/
│   ├── summary/                       # publishable aggregate CSVs
│   ├── pytorch/{512,640,768}/         # canonical PyTorch validation logs
│   ├── tensorrt/
│   │   ├── fp32/{512,640,768}/        # canonical TensorRT FP32 logs
│   │   └── fp16/{512,640,768}/        # canonical TensorRT FP16 logs
│   ├── power_modes/{15w,25w,maxn_super}/
│   │                                     # validation + tegrastats pairs
│   └── video/
│       ├── sustained/4k/               # valid 4K prediction + telemetry logs
│       └── diagnostics/                # incomplete or idle captures
├── logs/
│   ├── exports/                        # engine export and INT8 diagnostic logs
│   └── system/                         # recorded canonical power states
├── scripts/                            # latency and telemetry utilities
└── artifacts/                          # excluded-binary manifest and guidance
```

The logs contain terminal progress control characters. This is expected and
preserves the original capture.

## Canonical Matrix Verification

The nine validation logs directly support:

- 10,000 images, 13,262 person instances, and 6,780 backgrounds;
- precision, recall, mAP50, and mAP50–95;
- preprocess, inference, and postprocess time;
- elapsed progress duration and reported processing rate;
- Ultralytics 8.4.108, Python 3.10.12, PyTorch 2.10.0, batch 1;
- fixed 25W state and locked clocks from the paired system snapshots.

`stage_total_ms` is the sum of the three reported stages. `throughput_img_s` is
the full validation rate, using wall-clock progress duration. Never replace it
with `1000 / inference_ms`; that reciprocal is only a single-stage theoretical
rate.

## Power-Mode Evidence

Each mode has a full BDD100K validation log and a tegrastats trace. The parser
retains samples where `GR3D_FREQ > 20%` to exclude setup and idle periods:

| Mode | Active samples |
|---|---:|
| 15W | 379 |
| 25W | 304 |
| MAXN_SUPER | 281 |

Power, temperature, RAM, and GPU-utilization aggregates in the summary use
those active samples. Tegrastats was sampled at roughly one-second intervals;
short GPU bursts can therefore alias badly. The utilization means describe the
retained samples, not exact GPU occupancy or efficiency.

## Repeated Latency and Engine Roles

The scripts [`../scripts/benchmark_fp16_latency.py`](../scripts/benchmark_fp16_latency.py)
and [`../scripts/benchmark_fp16_nms_latency.py`](../scripts/benchmark_fp16_nms_latency.py)
use:

- 20 fixed images loaded into RAM before timing;
- 20 warmup iterations;
- 200 measured predictions;
- `imgsz=768`, confidence 0.001, and IoU 0.7.

Their measurements are not directly comparable to the canonical full-dataset
`val` stage values because the protocols differ. The summary output is retained,
but the original console logs are not.

The regular FP16 engine remains the canonical accuracy engine. The integrated-
NMS engine reduced repeated postprocessing from 3.55 ms to 1.19 ms and wall time
from 32.19 ms to 30.42 ms, while its recorded BDD100K mAP50–95 was lower (0.233
versus 0.251). Because the textual NMS validation log is absent, that accuracy
row is a summary record rather than raw-verifiable evidence.

Two NMS export logs are retained:

| Engine | Output | Status |
|---|---:|---|
| `crowdhuman_768_fp16_nms.engine` | `(1, 300, 6)` | Validated summary exists; not canonical accuracy |
| `crowdhuman_768_fp16_nms_lowconf.engine` | `(1, 1000, 6)` | Exported; full BDD100K revalidation not retained |

Both exports record FP16, TensorRT 10.7.0, and Ultralytics 8.4.117. No claim is
made about the exact cause of the integrated-NMS accuracy difference.

## Video Source Provenance

The local media files are alternate encodes of the same sequence. They are not
tracked in Git; byte sizes and SHA-256 hashes remain available in the
[`artifact_manifest.csv`](../artifacts/artifact_manifest.csv):

| File | Source resolution | Frames | Nominal rate |
|---|---:|---:|---:|
| `media/video/Pedestrian_trial_video.mp4` | 3840×2160 | 884 | 30000/1001 FPS |
| `media/video/Pedestrian_trial_video_1080p.mp4` | 1920×1080 | 884 | 29.97 FPS |
| `media/video/Pedestrian_trial_video_720p.mp4` | 1280×720 | 884 | 29.97 FPS |

Source resolution is not model input resolution. `768x768` in Ultralytics
prediction output is the preprocessed network input.

The valid 4K sustained log supports 884 samples, mean inference 20.29 ms, and
P99 20.5 ms for the integrated-NMS FP16 engine at `imgsz=768`. Its paired
tegrastats trace belongs only to this 4K run. The diagnostic file
`video/diagnostics/tegrastats_1080p_fp16_nms_768_idle.log` contains three idle
samples and is excluded from all reported workload telemetry.

The 4K prediction log does not contain a wall timer, so it is not used to derive
end-to-end FPS. Likewise, the raw wall-timing stdout for the four resolution
summary rows and the separate 1080p sustained row was not retained.

The sequence is derived from [People and Vehicles on the Streets during
Daytime](https://www.pexels.com/video/people-and-vehicles-on-the-streets-during-daytime-2954065/)
by **George Morina**, via Pexels under the [Pexels
License](https://www.pexels.com/license/).
It was used for qualitative inspection and pipeline testing, not model training
or BDD100K accuracy evaluation.

## INT8 Classification

INT8 is a completed experiment with a negative deployment result, not a
canonical optimization:

| 768 engine | Inference | P | R | mAP50 | mAP50–95 |
|---|---:|---:|---:|---:|---:|
| TensorRT FP16 | 19.2 ms | 0.700 | 0.467 | 0.499 | 0.251 |
| TensorRT INT8 | 22.0 ms | 0.524 | 0.410 | 0.384 | 0.167 |

The retained INT8 engine was successfully built using an existing TensorRT
MinMax calibration cache and completed full BDD100K validation. It was slower
than FP16 and lost substantial accuracy, so it is excluded from the canonical
matrix. The logs do not establish the exact cause of the regression.

Three subsequent attempts to regenerate calibration scales failed during the
engine build with CUDA out-of-memory errors. Those logs are preserved under
[`../logs/exports/diagnostics/int8/`](../logs/exports/diagnostics/int8/). The
engine and prior calibration cache remain in the separate local experiment
backup; their checksums and roles are preserved in the artifact manifest, but
the generated binaries are intentionally excluded from Git.

## Environment Boundaries

[`../jetson_system_info.txt`](../jetson_system_info.txt) records TensorRT 10.3.0
host packages. All retained
engine export logs record TensorRT 10.7.0 inside their export environment, while
the validation logs omit the runtime version. Treat the export/runtime pairing
as an unresolved provenance limitation rather than assuming one global version.

The audited deployment copy and selected training output matched byte-for-byte:

```text
741d5cecb3326549bba2aef8248febfed02200576fb246a042e9bf190b3eabd6
```

The checkpoint is excluded from Git. Mac/MPS results belong to the research
evaluation context and are not used as the baseline for Jetson TensorRT speedups.

## Auxiliary Artifacts

The local backup's `runs/detect/` directories contain auxiliary Mac/MPS plots
and batch previews, but not enough invocation, device, and machine-readable
metadata to support canonical Jetson claims. They are intentionally excluded
from this integrated repository.

Generated `outputs/` are also excluded. Their annotated videos and qualitative
images are demonstrations rather than quantitative benchmark evidence; the
retained raw logs, summary CSVs, and main research figures are the auditable
sources used here.

## Interpretation Rules

1. Compare backend and precision only at the same `imgsz`.
2. Treat ±0.001 metric changes across backend/precision as numerical noise.
3. Do not describe FP16 as improving accuracy.
4. Do not calculate measured throughput as `1000 / inference_ms`.
5. Keep source resolution, model input, inference latency, and pipeline FPS
   explicit and separate.
6. Do not use integrated-NMS metrics as official model accuracy.
7. Do not use diagnostic or idle logs to support benchmark claims.
8. Describe Jetson memory as unified/shared memory, not discrete VRAM.

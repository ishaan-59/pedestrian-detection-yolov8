# Model Inventory 
This directory contains the YOLOv8m single-class pedestrian detection models for the project.

The models are divided into two groups:
- Single-dataset models - YOLOv8m model trained independently on a single dataset.
- Sequential transfer models - YOLOv8m model trained on one pedestrian dataset and subsequently fine-tuned on another.

These models were later compared through cross-dataset evaluation to study how training domain and sequential fine-tuning affected pedestrian-detection generalization. 

## Directory Structure

```text
models/ 
├── README.md 
│ 
├── single_dataset/ 
│   ├── caltech_only/ 
│   ├── citypersons_only/ 
│   └── crowdhuman_only/ 
│ 
└── transfer/ 
    ├── caltech_to_citypersons/ 
    ├── citypersons_to_crowdhuman/ 
    └── crowdhuman_to_citypersons/
```

## Model Inventory

| Model | Training Type | Training Sequence | Purpose |
| ----------- | ----------- | ----------- | ----------- | 
| Caltech | Single Dataset | Caltech | Evaluate a detector trained on a relatively narrow pedestrian domain | 
| CityPersons | Single Dataset | CityPersons | Establish an urban detection baseline | 
| CrowdHuman | Single Dataset | CrowdHuman | Evaluate training on a larger and more diverse pedestrian dataset |
| Caltech → CityPersons | Sequential Transfer | Caltech → CityPersons | Test whether prior Caltech training benefits CityPersons |
| CityPersons → CrowdHuman | Sequential Transfer | CityPersons → CrowdHuman | Test whether prior CityPersons training benefits CrowdHuman|
| CrowdHuman → CityPersons | Sequential Transfer | CrowdHuman → CityPersons | Test the reverse transfer direction |

## Single-Dataset Models 
### Caltech
This model was trained only on the Caltech pedestrian dataset.

It serves as a source-domain-specific baseline and was included to examine how a detector trained on a comparatively constrained pedestrian domain performs when evaluated on other datasets.

`single_dataset/caltech_only/`

### CityPersons

The CityPersons model was trained only on the CityPersons dataset.

It provides an independent urban pedestrian-detection baseline against which the sequential-transfer experiments involving CityPersons can be compared.

`single_dataset/citypersons_only/`

### CrowdHuman

The CrowdHuman model was trained only on CrowdHuman.

Among the evaluated training strategies, this model showed the strongest overall off-domain generalization and was therefore selected as the primary model for downstream qualitative analysis and edge deployment.

`single_dataset/crowdhuman_only/`

## Sequential-Transfer Models

Sequential-transfer experiments were used to investigate whether training on one pedestrian dataset before fine-tuning on another produced different generalization behavior from training on the final dataset alone.

### Caltech → CityPersons

YOLOv8m was first trained on Caltech and subsequently fine-tuned on CityPersons.

`transfer/caltech_to_citypersons/`

This experiment provides a direct comparison against the CityPersons-only model.

### CityPersons → CrowdHuman

YOLOv8m was first trained on CityPersons and subsequently fine-tuned on CrowdHuman.

`transfer/citypersons_to_crowdhuman/`

Under the evaluation setup used in this project, its performance was very similar to the CrowdHuman-only model. This suggests that the later CrowdHuman training stage dominated the earlier CityPersons stage in this particular training sequence.

This observation should not be interpreted as evidence that sequential training is ineffective in general.

### CrowdHuman → CityPersons

YOLOv8m was first trained on CrowdHuman and subsequently fine-tuned on CityPersons.

`transfer/crowdhuman_to_citypersons/`

This experiment evaluates the opposite transfer direction and helps examine how the final training domain influences the resulting detector.

## Experiment Folder Contents

Each model directory preserves the artifacts required to understand and inspect its training run. Depending on the run, this may include:

```text
model_name/
├── args.yaml
├── results.csv
├── plots/
├── samples/
└── best.pt
```

```args.yaml```

Stores the Ultralytics training configuration used for the experiment, including model, dataset, image size, optimization settings, and other training parameters.

```results.csv```

Contains epoch-level training and validation metrics recorded during training.

```plots/```

Contains generated training and evaluation figures such as loss curves, precision-recall curves, F1 curves, and confusion matrices.

```samples/```

Contains selected training or validation visualizations produced during the run.

```best.pt```

The checkpoint corresponding to the best-performing epoch according to the training run.

Model weight files are excluded from the public Git repository because of their size. The remaining training metadata and results are retained so that the provenance of each experiment remains documented.

## Selected Model

The CrowdHuman-only YOLOv8m model was selected as the final detector for downstream work.

The decision was based primarily on its stronger overall cross-dataset generalization rather than performance on its source dataset alone.

The selected model was subsequently used for:

- qualitative pedestrian-detection comparisons;
- NVIDIA Jetson Orin Nano deployment;
- TensorRT optimization and edge-inference benchmarking.

Detailed comparisons between all models are documented separately in the project's cross-dataset evaluation results.

## Interpretation

The model experiments highlight two important observations that are explored in greater detail elsewhere in the repository:

1. **Training domain strongly affects generalization**: The Caltech-only model exhibited strong source-domain specialization but generalized substantially less effectively to other pedestrian datasets.

2. **The final stage of sequential training can dominate earlier training**: CityPersons → CrowdHuman produced results very close to CrowdHuman-only under this setup, suggesting that the later CrowdHuman stage largely determined the final model behavior.

These findings are specific to the datasets, training procedure, architecture, and evaluation protocol used in this project and should not be interpreted as universal conclusions about transfer learning.
___

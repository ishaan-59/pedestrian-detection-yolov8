# Pedestrian Detection using YOLOv8m

A deep learning–based pedestrian detection system built using the YOLOv8m architecture and trained on the Caltech Pedestrian Dataset.

---

## Project Overview

Pedestrian detection is a critical computer vision task with applications in autonomous driving, traffic monitoring, and urban safety systems.

This project focuses on training a single-class object detection model to identify pedestrians in urban scenes using YOLOv8m. The goal was to:

- Train a robust pedestrian detector using a modern YOLO architecture  
- Evaluate performance using standard object detection metrics  
- Analyze generalization performance on external images and real-world video  
- Present results in a clean, reproducible engineering format  

---

## Model Architecture

- **Model:** YOLOv8m  
- **Framework:** Ultralytics YOLO  
- **Task:** Single-class object detection (`person`)  

YOLOv8m was selected as a balance between representational capacity and computational efficiency. The medium-sized architecture offers improved accuracy over lightweight variants while maintaining practical inference speed.

---

## Training Setup

- **Dataset:** Caltech Pedestrian Dataset (curated subset)  
- **Epochs:** 50  
- **Input Resolution:** 512 × 512  
- **Task Type:** Single-class detection  

The model was trained for 50 epochs to ensure stable convergence while avoiding unnecessary computational overhead.

### Training Behavior

- Bounding box loss decreased consistently for both training and validation sets.
- Validation loss closely tracked training loss, suggesting minimal overfitting within the dataset distribution.
- mAP@0.5 increased rapidly during early epochs and plateaued after ~40 epochs.
- mAP@0.5:0.95 improved more gradually, reflecting refinement in localization precision.

By approximately 40–50 epochs, performance gains diminished, making 50 epochs an efficient stopping point.

---

## Validation Performance (Curated Dataset)

Evaluation was performed on a held-out validation split.

| Metric        | Score |
|---------------|-------|
| Precision     | 0.97  |
| Recall        | 0.96  |
| mAP@0.5       | 0.98  |
| mAP@0.5:0.95  | 0.90  |

These results reflect strong detection and localization performance within the curated dataset distribution.

**Note:**  
These metrics were computed on a held-out validation split from the curated Caltech dataset. While performance within this distribution is strong, qualitative testing on external datasets reveals sensitivity to small-scale pedestrians and high-density crowd scenes. 
---

## Qualitative Results

The model was evaluated on:

- Held-out validation images  
- External COCO pedestrian images  
- Real-world urban street video  

### Observations

- Strong detection of medium-to-large pedestrians  
- High precision with minimal false positives  
- Reduced performance on small-scale or distant pedestrians  
- Decreased confidence in dense crowd scenes  

**While validation metrics are high, qualitative testing demonstrates that real-world generalization depends strongly on dataset diversity, pedestrian scale, and scene complexity.**

---

## Quick Start

Clone the repository:

```bash
git clone https://github.com/ishaan-59/pedestrian-detection-yolov8.git
cd pedestrian-detection-yolov8
```

Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Run image inference:

```bash
python inference.py --source inputs --weights weights/best.pt
```

Run video inference:

```bash
python video_inference.py --source inputs/urban_pedestrian_demo.mp4 --weights weights/best.pt
```

Outputs will be saved in the `outputs/` directory.

---

## Project Structure

```
pedestrian-detection-yolo/
│
├── weights/
│   └── best.pt
├── inputs/
│   ├── demo_images/
│   └── urban_pedestrian_demo.mp4
├── assets/
│   ├── training_curves.png
│   ├── pr_curve.png
│   └── confusion_matrix.png
├── inference.py
├── video_inference.py
├── requirements.txt
└── README.md
```

---

## Future Improvements

- Incorporate additional datasets to improve small-object detection  
- Train at higher resolutions (e.g., 640 or 768)  
- Include high-density crowd scenes during training  
- Explore multi-scale augmentation strategies  

---

## Conclusion

This project demonstrates:

- End-to-end model training using YOLOv8m  
- Quantitative evaluation using detection metrics  
- Qualitative analysis under distribution shift  
- Structured deployment via image and video inference pipelines  

While strong within the curated dataset distribution, further improvements are required for fully robust real-world pedestrian detection.

# References

### Datasets

- **Caltech Pedestrian Dataset**  
  http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/

- **COCO Dataset (Common Objects in Context)**  
  https://cocodataset.org/

### Example Images

- COCO Image ID 576187  
  http://cocodataset.org/#explore?id=576187  
  Original source: http://farm9.staticflickr.com/8371/8473797139_938f2bf9fb_z.jpg  

- COCO Image ID 474556  
  http://cocodataset.org/#explore?id=474556  
  Original source: http://farm1.staticflickr.com/17/21888132_87f6189a4a_z.jpg  

### Demo Video

- Pexels – Urban Pedestrian Footage  
  https://www.pexels.com/video/people-walking-on-the-street-5330925/
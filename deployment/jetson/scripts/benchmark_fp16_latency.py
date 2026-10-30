from pathlib import Path
from statistics import mean, median, stdev
import time

import cv2
from ultralytics import YOLO


ENGINE = "models/tensorrt/fp16/crowdhuman_768_fp16.engine"
IMAGE_DIR = Path("/data/pedestrian_detection/bdd100k_val/images/val")

NUM_IMAGES = 20
WARMUP_RUNS = 20
MEASURED_RUNS = 200
IMGSZ = 768


# ---------------------------------------------------------
# 1. Load a fixed image set into RAM BEFORE timing anything
# ---------------------------------------------------------

image_paths = sorted(IMAGE_DIR.glob("*"))[:NUM_IMAGES]

if len(image_paths) < NUM_IMAGES:
    raise RuntimeError(
        f"Expected {NUM_IMAGES} images, found only {len(image_paths)}"
    )

images = []

for path in image_paths:
    image = cv2.imread(str(path))

    if image is None:
        raise RuntimeError(f"Could not read {path}")

    images.append(image)

print(f"Loaded {len(images)} images into RAM.")


# ---------------------------------------------------------
# 2. Load the TensorRT engine
# ---------------------------------------------------------

model = YOLO(ENGINE, task="detect")


# ---------------------------------------------------------
# 3. Warm up the engine
# ---------------------------------------------------------

print(f"Warming up for {WARMUP_RUNS} iterations...")

for i in range(WARMUP_RUNS):
    image = images[i % len(images)]

    model.predict(
        source=image,
        imgsz=IMGSZ,
        device=0,
        conf=0.001,
        iou=0.7,
        verbose=False,
    )


# ---------------------------------------------------------
# 4. Repeated timed inference
# ---------------------------------------------------------

preprocess_times = []
inference_times = []
postprocess_times = []
wall_times = []

print(f"Measuring {MEASURED_RUNS} iterations...")

for i in range(MEASURED_RUNS):
    image = images[i % len(images)]

    start = time.perf_counter()

    results = model.predict(
        source=image,
        imgsz=IMGSZ,
        device=0,
        conf=0.001,
        iou=0.7,
        verbose=False,
    )

    end = time.perf_counter()

    speed = results[0].speed

    preprocess_times.append(speed["preprocess"])
    inference_times.append(speed["inference"])
    postprocess_times.append(speed["postprocess"])
    wall_times.append((end - start) * 1000)


# ---------------------------------------------------------
# 5. Print summary statistics
# ---------------------------------------------------------

def summarize(name, values):
    print(
        f"{name:<18}"
        f"mean={mean(values):6.2f} ms  "
        f"median={median(values):6.2f} ms  "
        f"std={stdev(values):5.2f} ms  "
        f"min={min(values):6.2f} ms  "
        f"max={max(values):6.2f} ms"
    )


print("\nFP16 @ 768 repeated latency benchmark")
print("----------------------------------------")

summarize("Preprocess", preprocess_times)
summarize("Inference", inference_times)
summarize("Postprocess", postprocess_times)
summarize("Wall time", wall_times)


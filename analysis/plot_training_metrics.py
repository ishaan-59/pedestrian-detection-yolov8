import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('/Users/ishaansingh/Desktop/pedestrian-detection-yolo/results/results.csv')

epochs = df["epoch"]
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].plot(epochs, df["train/box_loss"], label="Train Box Loss")
axes[0].plot(epochs, df["val/box_loss"], label="Val Box Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].set_title("Bounding Box Loss")
axes[0].legend()

# ---- mAP plot ----
axes[1].plot(epochs, df["metrics/mAP50(B)"], label="mAP@0.5")
axes[1].plot(epochs, df["metrics/mAP50-95(B)"], label="mAP@0.5:0.95")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("mAP")
axes[1].set_title("Detection Performance")
axes[1].legend()

plt.tight_layout()
plt.savefig("results/training_curves.png", dpi=200)
plt.show()
plt.close()
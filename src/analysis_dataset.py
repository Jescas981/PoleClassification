import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from Lima3D import load_dataset

# --- Load file paths + labels ---
files, labels = load_dataset("data/pole_dataset/output")

labels = np.array(labels)
num_points_list = []

# --- Load each PLY and count points ---
for f in files:
    pcd = o3d.io.read_point_cloud(f)
    points = np.asarray(pcd.points)
    num_points_list.append(points.shape[0])

num_points_list = np.array(num_points_list)

# -------------------------
# 📊 Class distribution
# -------------------------
unique, counts = np.unique(labels, return_counts=True)

plt.figure()
plt.bar(unique, counts)
plt.xticks([0, 1], ["monoposte", "biposte"])
plt.title("Class Distribution")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()

# -------------------------
# 📊 Points per sample (REAL)
# -------------------------
plt.figure()
plt.hist(num_points_list, bins=30)
plt.title("Raw Points per Sample")
plt.xlabel("Number of Points")
plt.ylabel("Frequency")
plt.show()

# -------------------------
# 📊 Log-scale histogram (better insight)
# -------------------------
plt.figure()
plt.hist(num_points_list, bins=30, log=True)
plt.title("Raw Points per Sample (Log Scale)")
plt.xlabel("Number of Points")
plt.ylabel("Frequency (log)")
plt.show()

# -------------------------
# 📊 Stats
# -------------------------
print("Min points:", np.min(num_points_list))
print("Max points:", np.max(num_points_list))
print("Mean points:", np.mean(num_points_list))
print("Std points:", np.std(num_points_list))

from Lima3D import PoleDataset

# --- Load dataset with oversampling ---
dataset = PoleDataset(
    "data/pole_dataset/output",
    num_points=256,
    model_set="train"  # IMPORTANT: triggers oversampling
)

labels_ds = []

# --- Iterate dataset ---
for i in range(len(dataset)):
    _, label = dataset[i]
    labels_ds.append(label.item())

labels_ds = np.array(labels_ds)

# -------------------------
# 📊 Class distribution AFTER oversampling
# -------------------------
unique_ds, counts_ds = np.unique(labels_ds, return_counts=True)

plt.figure()
plt.bar(unique_ds, counts_ds)
plt.xticks([0, 1], ["monoposte", "biposte"])
plt.title("Class Distribution (After Oversampling)")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()

# -------------------------
# 📊 Print stats
# -------------------------
print("\nAfter Oversampling:")
print("Monoposte:", counts_ds[0] if 0 in unique_ds else 0)
print("Biposte:", counts_ds[1] if 1 in unique_ds else 0)
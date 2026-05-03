from cProfile import label
import os
import torch
import numpy as np
import open3d as o3d
from torch.utils.data import Dataset
from utils import farthest_point_sampling, augment_pointcloud


def load_dataset(root_dir):

    files = []
    labels = []

    label_map = {
        "monoposte": 0,
        "biposte": 1
    }

    # --- Collect all files ---
    for scene in os.listdir(root_dir):
        scene_path = os.path.join(root_dir, scene)
        if not os.path.isdir(scene_path):
            continue

        poles_path = os.path.join(scene_path, "poles")
        if not os.path.exists(poles_path):
            continue

        for file in os.listdir(poles_path):
            if not file.endswith(".ply"):
                continue

            file_path = os.path.join(poles_path, file)

            if "monoposte" in file:
                label = label_map["monoposte"]
            elif "biposte" in file:
                label = label_map["biposte"]
            else:
                continue

            files.append(file_path)
            labels.append(label)

    return files, labels


class PoleDataset(Dataset):
    def __init__(
        self,
        root_dir,
        num_points=1024,
        use_cache=True,
        model_set: str = "train",
        use_augmentation=True
    ):
        self.files, self.labels = load_dataset(root_dir)
        self.model_set = model_set
        self.num_points = num_points
        self.use_cache = use_cache
        self.use_augmentation = use_augmentation
        self.cache = {}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        label = self.labels[idx]
        file_path = self.files[idx]

        # -------------------------
        # 📥 Load point cloud
        # -------------------------
        if self.use_cache and file_path in self.cache:
            points = self.cache[file_path]
        else:
            pcd = o3d.io.read_point_cloud(file_path)
            points = np.asarray(pcd.points)

            if self.use_cache:
                self.cache[file_path] = points

        # -------------------------
        # 🔀 Shuffle
        # -------------------------
        np.random.shuffle(points)

        # -------------------------
        # 🎯 FPS Sampling
        # -------------------------
        if len(points) >= self.num_points:
            idxs = farthest_point_sampling(points, self.num_points)
        else:
            fps_idxs = farthest_point_sampling(points, len(points))
            extra = np.random.choice(
                len(points),
                self.num_points - len(points),
                replace=True
            )
            idxs = np.concatenate([fps_idxs, extra])

        points = points[idxs]

        # -------------------------
        # 📏 Normalize
        # -------------------------
        centroid = np.mean(points, axis=0)
        points = points - centroid
        scale = np.max(np.linalg.norm(points, axis=1))
        points = points / (scale + 1e-8)

        # -------------------------
        # 🔥 Augmentation (train only)
        # -------------------------
        if self.model_set == "train" and self.use_augmentation:
            points = self.augment(points)

        return torch.from_numpy(points).float(), torch.tensor(label).long()

    # -------------------------
    # 🧪 Augmentation
    # -------------------------
    def augment(self, points):
        theta = np.random.uniform(0, 2 * np.pi)

        R = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0, 0, 1]
        ])

        points = points @ R.T
        points *= np.random.uniform(0.8, 1.2)
        points += np.random.normal(0, 0.01, points.shape)
        points += np.random.uniform(-0.1, 0.1, (1, 3))

        return points

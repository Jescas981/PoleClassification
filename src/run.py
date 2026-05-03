import os
import torch
import numpy as np
import open3d as o3d
import argparse

from model import DGCNN, PointNet
from utils import farthest_point_sampling

# -------------------------
# 📦 Preprocess
# -------------------------
def preprocess_pcd(file_path, num_points):
    pcd = o3d.io.read_point_cloud(file_path)
    points = np.asarray(pcd.points)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Invalid shape {points.shape} in {file_path}")

    if len(points) == 0:
        raise ValueError(f"Empty point cloud: {file_path}")

    np.random.shuffle(points)

    if len(points) >= num_points:
        idxs = farthest_point_sampling(points, num_points)
    else:
        fps_idxs = farthest_point_sampling(points, len(points))
        extra = np.random.choice(len(points), num_points - len(points), replace=True)
        idxs = np.concatenate([fps_idxs, extra])

    points = points[idxs]

    centroid = np.mean(points, axis=0)
    points = points - centroid
    scale = np.max(np.linalg.norm(points, axis=1))
    points = points / (scale + 1e-8)

    points = torch.from_numpy(points).float()
    points = points.unsqueeze(0).permute(0, 2, 1)

    return points


# -------------------------
# 🧠 Load model
# -------------------------
def load_model(model_path, device):
    model = DGCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


# -------------------------
# 🔮 Predict
# -------------------------
def predict(model, file_path, num_points, device):
    points = preprocess_pcd(file_path, num_points).to(device)

    with torch.no_grad():
        outputs = model(points)
        pred = torch.argmax(outputs, dim=1).item()

    return pred


# -------------------------
# 🚀 Main inference
# -------------------------
def run_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    LABEL_MAP = {
        0: "monoposte",
        1: "biposte"
    }

    model = load_model(args.model_path, device)

    results = []

    for fname in sorted(os.listdir(args.input_folder)):
        if not fname.endswith(".ply"):
            continue

        file_path = os.path.join(args.input_folder, fname)

        try:
            pred = predict(model, file_path, args.num_points, device)
            label_name = LABEL_MAP[pred]

            print(f"{fname} -> {label_name}")
            results.append(f"{fname} {label_name}")

        except Exception as e:
            print(f"Error processing {fname}: {e}")

    with open(args.output_file, "w") as f:
        for line in results:
            f.write(line + "\n")

    print(f"\nSaved results to {args.output_file}")


# -------------------------
# 🧾 Args parser
# -------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Point Cloud Inference")

    parser.add_argument("--model_path", type=str, default="pointnet_fold_1.pth",
                        help="Path to trained model")

    parser.add_argument("--input_folder", type=str, required=True,
                        help="Folder containing .ply files")

    parser.add_argument("--output_file", type=str, default="labels.txt",
                        help="Output txt file")

    parser.add_argument("--num_points", type=int, default=312,
                        help="Number of points per cloud")

    return parser.parse_args()


# -------------------------
# Entry point
# -------------------------
if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
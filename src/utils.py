import numpy as np

def farthest_point_sampling(points, num_samples):
    N, _ = points.shape
    sampled_pts = np.zeros((num_samples,), dtype=int)

    # Start from a random point
    sampled_pts[0] = np.random.randint(0, N)

    distances = np.full(N, np.inf)

    for i in range(1, num_samples):
        last_point = points[sampled_pts[i - 1]]
        dist = np.linalg.norm(points - last_point, axis=1)

        distances = np.minimum(distances, dist)
        sampled_pts[i] = np.argmax(distances)

    return sampled_pts


def augment_pointcloud(points, label):
    # --- Rotation around Z (safe for poles) ---
    theta = np.random.uniform(0, 2 * np.pi)
    rot = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0, 0, 1]
    ])
    points = points @ rot

    # --- Small jitter (noise) ---
    noise = np.random.normal(0, 0.01, points.shape)
    points = points + noise

    # --- Slight scaling ---
    scale = np.random.uniform(0.9, 1.1)
    points = points * scale

    # --- Optional: stronger aug for biposte ---
    if label == 1:
        # extra jitter
        points += np.random.normal(0, 0.02, points.shape)

    return points
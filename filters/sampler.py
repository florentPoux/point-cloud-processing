import numpy as np
from scipy.spatial import cKDTree

def random_sampling(points, target_size, colors=None):
    """Random downsampling"""
    if len(points) <= target_size:
        return points, colors, np.arange(len(points))

    indices = np.random.choice(len(points), target_size, replace=False)
    sampled_points = points[indices]

    sampled_colors = None
    if colors is not None:
        sampled_colors = colors[indices]

    return sampled_points, sampled_colors, indices

def fps_sampling(points, target_size):
    """Farthest Point Sampling for diverse coverage"""
    n_points = len(points)

    if n_points <= target_size:
        return points, np.arange(n_points)

    sampled_indices = np.zeros(target_size, dtype=np.int32)
    distances = np.full(n_points, np.inf)

    sampled_indices[0] = np.random.randint(0, n_points)

    for i in range(1, target_size):
        last_point = points[sampled_indices[i-1]]

        dists = np.linalg.norm(points - last_point, axis=1)
        distances = np.minimum(distances, dists)

        sampled_indices[i] = np.argmax(distances)

    return points[sampled_indices], sampled_indices

def stratified_sampling(points, labels, samples_per_class=None):
    """Sample while maintaining class distribution"""
    unique_labels = np.unique(labels)

    if samples_per_class is None:
        label_counts = np.bincount(labels)
        min_count = label_counts[label_counts > 0].min()
        samples_per_class = min_count

    sampled_indices = []

    for label in unique_labels:
        label_mask = labels == label
        label_indices = np.where(label_mask)[0]

        if len(label_indices) <= samples_per_class:
            sampled_indices.extend(label_indices)
        else:
            selected = np.random.choice(label_indices, samples_per_class, replace=False)
            sampled_indices.extend(selected)

    sampled_indices = np.array(sampled_indices)

    return points[sampled_indices], labels[sampled_indices], sampled_indices

def poisson_disk_sampling(points, radius):
    """Poisson disk sampling for uniform spacing"""
    n_points = len(points)
    active_list = []
    sampled_mask = np.zeros(n_points, dtype=bool)

    tree = cKDTree(points)

    first_idx = np.random.randint(0, n_points)
    active_list.append(first_idx)
    sampled_mask[first_idx] = True

    max_attempts = 30

    while active_list:
        active_idx = active_list.pop(np.random.randint(0, len(active_list)))
        active_point = points[active_idx]

        for _ in range(max_attempts):
            angle = np.random.uniform(0, 2 * np.pi)
            distance = np.random.uniform(radius, 2 * radius)

            candidate = active_point + distance * np.array([
                np.cos(angle),
                np.sin(angle),
                0
            ])

            neighbors = tree.query_ball_point(candidate, radius)

            valid = True
            for neighbor_idx in neighbors:
                if sampled_mask[neighbor_idx]:
                    valid = False
                    break

            if valid:
                dists = np.linalg.norm(points - candidate, axis=1)
                closest_idx = np.argmin(dists)

                if not sampled_mask[closest_idx]:
                    sampled_mask[closest_idx] = True
                    active_list.append(closest_idx)

    return points[sampled_mask], np.where(sampled_mask)[0]

def grid_sampling(points, grid_size, method='centroid', colors=None):
    """Grid-based downsampling"""
    from data.spatial_hash import voxel_downsample

    return voxel_downsample(points, grid_size, colors, method)

def adaptive_sampling(points, target_size, feature_weights=None):
    """Adaptive sampling based on local features"""
    if feature_weights is None:
        tree = cKDTree(points)
        distances, _ = tree.query(points, k=20)
        mean_distances = distances[:, 1:].mean(axis=1)
        feature_weights = 1.0 / (mean_distances + 1e-6)

    feature_weights = feature_weights / feature_weights.sum()

    indices = np.random.choice(
        len(points),
        size=min(target_size, len(points)),
        replace=False,
        p=feature_weights
    )

    return points[indices], indices

def hierarchical_sampling(points, levels=3, ratio=0.5):
    """Multi-resolution hierarchical sampling"""
    hierarchy = []

    current_points = points
    current_indices = np.arange(len(points))

    hierarchy.append({
        'points': current_points,
        'indices': current_indices,
        'level': 0
    })

    for level in range(1, levels):
        target_size = int(len(current_points) * ratio)
        sampled_points, sampled_indices = fps_sampling(current_points, target_size)

        current_indices = current_indices[sampled_indices]
        current_points = sampled_points

        hierarchy.append({
            'points': current_points,
            'indices': current_indices,
            'level': level
        })

    return hierarchy

def project_predictions_to_full_resolution(sampled_points, sampled_predictions, full_points, k=5):
    """Project predictions from sampled to full resolution using KDTree"""
    tree = cKDTree(sampled_points)

    distances, indices = tree.query(full_points, k=k)

    weights = 1.0 / (distances + 1e-6)
    weights = weights / weights.sum(axis=1, keepdims=True)

    if len(sampled_predictions.shape) == 1:
        full_predictions = np.zeros(len(full_points))
        for i in range(len(full_points)):
            full_predictions[i] = np.average(sampled_predictions[indices[i]], weights=weights[i])
    else:
        full_predictions = np.zeros((len(full_points), sampled_predictions.shape[1]))
        for i in range(len(full_points)):
            full_predictions[i] = np.average(sampled_predictions[indices[i]], axis=0, weights=weights[i])

    return full_predictions

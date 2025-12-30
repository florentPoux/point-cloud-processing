import numpy as np
from scipy.spatial import cKDTree

def region_growing_normals(points, normals, k=20, angle_threshold=15.0, curvature_threshold=0.05):
    """Region growing based on normal similarity"""
    angle_threshold_rad = np.radians(angle_threshold)

    tree = cKDTree(points)

    n_points = len(points)
    labels = np.full(n_points, -1, dtype=np.int32)
    processed = np.zeros(n_points, dtype=bool)

    from features.geometric import compute_curvature
    curvatures = compute_curvature(points, k)

    sorted_indices = np.argsort(curvatures)

    region_id = 0

    for seed_idx in sorted_indices:
        if processed[seed_idx]:
            continue

        if curvatures[seed_idx] > curvature_threshold:
            continue

        seeds = [seed_idx]
        region = []

        while seeds:
            current_idx = seeds.pop(0)

            if processed[current_idx]:
                continue

            processed[current_idx] = True
            labels[current_idx] = region_id
            region.append(current_idx)

            _, neighbors = tree.query(points[current_idx], k=k)

            for neighbor_idx in neighbors:
                if processed[neighbor_idx]:
                    continue

                angle = np.arccos(np.clip(np.dot(normals[current_idx], normals[neighbor_idx]), -1, 1))

                if angle < angle_threshold_rad:
                    if curvatures[neighbor_idx] < curvature_threshold:
                        seeds.append(neighbor_idx)
                    else:
                        processed[neighbor_idx] = True
                        labels[neighbor_idx] = region_id
                        region.append(neighbor_idx)

        if len(region) > 0:
            region_id += 1

    return labels

def region_growing_color(points, colors, k=20, color_threshold=0.1):
    """Region growing based on color similarity"""
    tree = cKDTree(points)

    n_points = len(points)
    labels = np.full(n_points, -1, dtype=np.int32)
    processed = np.zeros(n_points, dtype=bool)

    region_id = 0

    for seed_idx in range(n_points):
        if processed[seed_idx]:
            continue

        seeds = [seed_idx]
        region = []

        while seeds:
            current_idx = seeds.pop(0)

            if processed[current_idx]:
                continue

            processed[current_idx] = True
            labels[current_idx] = region_id
            region.append(current_idx)

            _, neighbors = tree.query(points[current_idx], k=k)

            for neighbor_idx in neighbors:
                if processed[neighbor_idx]:
                    continue

                color_diff = np.linalg.norm(colors[current_idx] - colors[neighbor_idx])

                if color_diff < color_threshold:
                    seeds.append(neighbor_idx)

        if len(region) > 0:
            region_id += 1

    return labels

def region_growing_hybrid(points, normals, colors, k=20, angle_threshold=15.0, color_threshold=0.1, curvature_threshold=0.05):
    """Region growing using both geometric and color features"""
    angle_threshold_rad = np.radians(angle_threshold)

    tree = cKDTree(points)

    n_points = len(points)
    labels = np.full(n_points, -1, dtype=np.int32)
    processed = np.zeros(n_points, dtype=bool)

    from features.geometric import compute_curvature
    curvatures = compute_curvature(points, k)

    sorted_indices = np.argsort(curvatures)

    region_id = 0

    for seed_idx in sorted_indices:
        if processed[seed_idx]:
            continue

        if curvatures[seed_idx] > curvature_threshold:
            continue

        seeds = [seed_idx]
        region = []

        while seeds:
            current_idx = seeds.pop(0)

            if processed[current_idx]:
                continue

            processed[current_idx] = True
            labels[current_idx] = region_id
            region.append(current_idx)

            _, neighbors = tree.query(points[current_idx], k=k)

            for neighbor_idx in neighbors:
                if processed[neighbor_idx]:
                    continue

                angle = np.arccos(np.clip(np.dot(normals[current_idx], normals[neighbor_idx]), -1, 1))

                color_diff = np.linalg.norm(colors[current_idx] - colors[neighbor_idx])

                if angle < angle_threshold_rad and color_diff < color_threshold:
                    if curvatures[neighbor_idx] < curvature_threshold:
                        seeds.append(neighbor_idx)
                    else:
                        processed[neighbor_idx] = True
                        labels[neighbor_idx] = region_id
                        region.append(neighbor_idx)

        if len(region) > 0:
            region_id += 1

    return labels

def smooth_region_boundaries(points, labels, k=10):
    """Smooth region boundaries by reassigning boundary points"""
    tree = cKDTree(points)

    smoothed_labels = labels.copy()

    for i in range(len(points)):
        if labels[i] < 0:
            continue

        _, neighbors = tree.query(points[i], k=k)

        neighbor_labels = labels[neighbors]
        neighbor_labels = neighbor_labels[neighbor_labels >= 0]

        if len(neighbor_labels) == 0:
            continue

        unique_labels, counts = np.unique(neighbor_labels, return_counts=True)

        if len(unique_labels) > 1:
            majority_label = unique_labels[np.argmax(counts)]
            smoothed_labels[i] = majority_label

    return smoothed_labels

def merge_small_regions(points, labels, min_region_size=50):
    """Merge small regions into neighboring larger regions"""
    unique_labels = np.unique(labels[labels >= 0])

    region_sizes = {label: (labels == label).sum() for label in unique_labels}

    small_regions = [label for label, size in region_sizes.items() if size < min_region_size]

    tree = cKDTree(points)

    merged_labels = labels.copy()

    for small_label in small_regions:
        small_region_mask = labels == small_label
        small_region_points = points[small_region_mask]

        centroid = small_region_points.mean(axis=0)

        _, neighbor_idx = tree.query(centroid, k=100)

        neighbor_labels = labels[neighbor_idx]
        neighbor_labels = neighbor_labels[neighbor_labels >= 0]
        neighbor_labels = neighbor_labels[neighbor_labels != small_label]

        if len(neighbor_labels) == 0:
            continue

        unique_neighbors, counts = np.unique(neighbor_labels, return_counts=True)

        merge_target = unique_neighbors[np.argmax(counts)]

        merged_labels[small_region_mask] = merge_target

    label_mapping = {old: new for new, old in enumerate(np.unique(merged_labels[merged_labels >= 0]))}
    for i in range(len(merged_labels)):
        if merged_labels[i] >= 0:
            merged_labels[i] = label_mapping[merged_labels[i]]

    return merged_labels

def extract_region_boundaries(points, labels, k=10):
    """Extract points on region boundaries"""
    tree = cKDTree(points)

    boundary_mask = np.zeros(len(points), dtype=bool)

    for i in range(len(points)):
        if labels[i] < 0:
            continue

        _, neighbors = tree.query(points[i], k=k)

        neighbor_labels = labels[neighbors]

        if len(np.unique(neighbor_labels[neighbor_labels >= 0])) > 1:
            boundary_mask[i] = True

    return boundary_mask

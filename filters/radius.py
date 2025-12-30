import numpy as np
from scipy.spatial import cKDTree

def radius_outlier_removal(points, radius=0.5, min_neighbors=10):
    """Remove points with few neighbors in radius"""
    tree = cKDTree(points)

    neighbor_counts = tree.query_ball_point(points, radius, return_length=True)

    inlier_mask = neighbor_counts >= min_neighbors

    return points[inlier_mask], inlier_mask

def radius_outlier_removal_adaptive(points, k_for_radius=20, min_neighbors_ratio=0.5):
    """Adaptive radius based on local point spacing"""
    tree = cKDTree(points)

    distances, _ = tree.query(points, k=k_for_radius+1)
    mean_distances = distances[:, 1:].mean(axis=1)

    adaptive_radii = mean_distances * 3.0

    inlier_mask = np.zeros(len(points), dtype=bool)

    for i in range(len(points)):
        neighbors = tree.query_ball_point(points[i], adaptive_radii[i])
        min_neighbors = int(k_for_radius * min_neighbors_ratio)

        if len(neighbors) >= min_neighbors:
            inlier_mask[i] = True

    return points[inlier_mask], inlier_mask

def radius_density_filter(points, radius=0.5, min_density=0.0, max_density=np.inf):
    """Filter based on local density"""
    tree = cKDTree(points)

    neighbor_counts = tree.query_ball_point(points, radius, return_length=True)

    volume = (4.0 / 3.0) * np.pi * (radius ** 3)
    densities = neighbor_counts / volume

    inlier_mask = (densities >= min_density) & (densities <= max_density)

    return points[inlier_mask], inlier_mask, densities

def remove_isolated_clusters(points, radius=1.0, min_cluster_size=50):
    """Remove small isolated point clusters"""
    tree = cKDTree(points)

    visited = np.zeros(len(points), dtype=bool)
    cluster_labels = np.full(len(points), -1, dtype=np.int32)

    cluster_id = 0

    for i in range(len(points)):
        if visited[i]:
            continue

        cluster = []
        stack = [i]

        while stack:
            idx = stack.pop()

            if visited[idx]:
                continue

            visited[idx] = True
            cluster.append(idx)

            neighbors = tree.query_ball_point(points[idx], radius)
            for neighbor_idx in neighbors:
                if not visited[neighbor_idx]:
                    stack.append(neighbor_idx)

        if len(cluster) >= min_cluster_size:
            cluster_labels[cluster] = cluster_id
            cluster_id += 1

    inlier_mask = cluster_labels >= 0

    return points[inlier_mask], inlier_mask, cluster_labels[inlier_mask]

def compute_local_density(points, radius=0.5):
    """Compute local point density"""
    tree = cKDTree(points)

    neighbor_counts = tree.query_ball_point(points, radius, return_length=True)

    volume = (4.0 / 3.0) * np.pi * (radius ** 3)
    densities = neighbor_counts / volume

    return densities

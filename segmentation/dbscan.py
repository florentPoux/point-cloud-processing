import numpy as np
from scipy.spatial import cKDTree

def dbscan_clustering(points, eps=0.5, min_samples=10):
    """DBSCAN clustering for point cloud segmentation"""
    n_points = len(points)

    labels = np.full(n_points, -1, dtype=np.int32)
    visited = np.zeros(n_points, dtype=bool)

    tree = cKDTree(points)

    cluster_id = 0

    for i in range(n_points):
        if visited[i]:
            continue

        visited[i] = True

        neighbors = tree.query_ball_point(points[i], eps)

        if len(neighbors) < min_samples:
            labels[i] = -1
        else:
            cluster_id = expand_cluster(
                i, neighbors, cluster_id, eps, min_samples,
                points, tree, labels, visited
            )
            cluster_id += 1

    return labels

def expand_cluster(point_idx, neighbors, cluster_id, eps, min_samples, points, tree, labels, visited):
    """Expand cluster in DBSCAN"""
    labels[point_idx] = cluster_id

    i = 0
    while i < len(neighbors):
        neighbor_idx = neighbors[i]

        if not visited[neighbor_idx]:
            visited[neighbor_idx] = True

            new_neighbors = tree.query_ball_point(points[neighbor_idx], eps)

            if len(new_neighbors) >= min_samples:
                neighbors = neighbors + [n for n in new_neighbors if n not in neighbors]

        if labels[neighbor_idx] == -1:
            labels[neighbor_idx] = cluster_id

        i += 1

    return cluster_id

def dbscan_adaptive(points, k=20, eps_multiplier=2.0, min_samples=10):
    """DBSCAN with adaptive eps based on local density"""
    tree = cKDTree(points)

    distances, _ = tree.query(points, k=k)
    mean_distances = distances[:, 1:].mean(axis=1)

    global_eps = mean_distances.mean() * eps_multiplier

    labels = dbscan_clustering(points, global_eps, min_samples)

    return labels

def cluster_points_euclidean(points, eps=0.5, min_cluster_size=50):
    """Simple Euclidean clustering"""
    labels = dbscan_clustering(points, eps, min_samples=min_cluster_size)

    unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)

    small_clusters_mask = labels >= 0
    for label, count in zip(unique_labels, counts):
        if count < min_cluster_size:
            small_clusters_mask[labels == label] = False

    labels[~small_clusters_mask] = -1

    label_mapping = {old: new for new, old in enumerate(np.unique(labels[labels >= 0]))}
    for i in range(len(labels)):
        if labels[i] >= 0:
            labels[i] = label_mapping[labels[i]]

    return labels

def filter_clusters_by_size(labels, min_size=None, max_size=None):
    """Filter clusters based on size constraints"""
    unique_labels = np.unique(labels[labels >= 0])

    filtered_labels = labels.copy()

    for label in unique_labels:
        cluster_size = (labels == label).sum()

        if min_size is not None and cluster_size < min_size:
            filtered_labels[labels == label] = -1

        if max_size is not None and cluster_size > max_size:
            filtered_labels[labels == label] = -1

    label_mapping = {old: new for new, old in enumerate(np.unique(filtered_labels[filtered_labels >= 0]))}
    for i in range(len(filtered_labels)):
        if filtered_labels[i] >= 0:
            filtered_labels[i] = label_mapping[filtered_labels[i]]

    return filtered_labels

def extract_cluster_features(points, labels):
    """Extract features for each cluster"""
    unique_labels = np.unique(labels[labels >= 0])

    features = []

    for label in unique_labels:
        cluster_mask = labels == label
        cluster_points = points[cluster_mask]

        centroid = cluster_points.mean(axis=0)
        bbox_min = cluster_points.min(axis=0)
        bbox_max = cluster_points.max(axis=0)
        bbox_size = bbox_max - bbox_min

        n_points = len(cluster_points)

        centered = cluster_points - centroid
        cov = np.dot(centered.T, centered) / n_points
        eigenvalues = np.linalg.eigvalsh(cov)

        features.append({
            'label': label,
            'n_points': n_points,
            'centroid': centroid,
            'bbox_min': bbox_min,
            'bbox_max': bbox_max,
            'bbox_size': bbox_size,
            'eigenvalues': eigenvalues
        })

    return features

def merge_nearby_clusters(points, labels, distance_threshold=1.0):
    """Merge clusters that are close to each other"""
    cluster_features = extract_cluster_features(points, labels)

    centroids = np.array([f['centroid'] for f in cluster_features])
    cluster_ids = np.array([f['label'] for f in cluster_features])

    if len(centroids) == 0:
        return labels

    tree = cKDTree(centroids)

    merged_labels = labels.copy()

    merge_map = {cid: cid for cid in cluster_ids}

    for i, centroid in enumerate(centroids):
        neighbors = tree.query_ball_point(centroid, distance_threshold)

        if len(neighbors) > 1:
            min_label = min([cluster_ids[n] for n in neighbors])

            for n in neighbors:
                merge_map[cluster_ids[n]] = min_label

    for i in range(len(merged_labels)):
        if merged_labels[i] >= 0:
            merged_labels[i] = merge_map[merged_labels[i]]

    label_mapping = {old: new for new, old in enumerate(np.unique(merged_labels[merged_labels >= 0]))}
    for i in range(len(merged_labels)):
        if merged_labels[i] >= 0:
            merged_labels[i] = label_mapping[merged_labels[i]]

    return merged_labels

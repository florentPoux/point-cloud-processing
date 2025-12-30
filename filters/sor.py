import numpy as np
from scipy.spatial import cKDTree

def statistical_outlier_removal(points, k=20, std_multiplier=2.0):
    """Remove statistical outliers using k-nearest neighbors"""
    tree = cKDTree(points)

    distances, _ = tree.query(points, k=k+1)
    mean_distances = distances[:, 1:].mean(axis=1)

    global_mean = mean_distances.mean()
    global_std = mean_distances.std()

    threshold = global_mean + std_multiplier * global_std

    inlier_mask = mean_distances < threshold

    return points[inlier_mask], inlier_mask

def statistical_outlier_removal_adaptive(points, k=20, std_multiplier=2.0):
    """SOR with adaptive threshold based on local density"""
    tree = cKDTree(points)

    distances, _ = tree.query(points, k=k+1)
    mean_distances = distances[:, 1:].mean(axis=1)

    local_densities = 1.0 / (mean_distances + 1e-6)
    density_quantiles = np.percentile(local_densities, [25, 75])

    inlier_mask = np.ones(len(points), dtype=bool)

    for low, high in [(0, density_quantiles[0]), (density_quantiles[0], density_quantiles[1]),
                       (density_quantiles[1], local_densities.max())]:
        region_mask = (local_densities >= low) & (local_densities < high)
        if region_mask.sum() == 0:
            continue

        region_distances = mean_distances[region_mask]
        region_mean = region_distances.mean()
        region_std = region_distances.std()

        threshold = region_mean + std_multiplier * region_std

        outliers = region_mask & (mean_distances >= threshold)
        inlier_mask[outliers] = False

    return points[inlier_mask], inlier_mask

def sor_streaming(points_iterator, k=20, std_multiplier=2.0, overlap=0.1):
    """SOR for massive datasets using streaming"""
    all_inliers = []

    for chunk_points, chunk_meta in points_iterator:
        cleaned, mask = statistical_outlier_removal(chunk_points, k, std_multiplier)
        all_inliers.append(cleaned)

    return np.vstack(all_inliers) if all_inliers else np.array([])

def compute_outlier_scores(points, k=20):
    """Compute outlier scores without removing"""
    tree = cKDTree(points)

    distances, _ = tree.query(points, k=k+1)
    mean_distances = distances[:, 1:].mean(axis=1)

    global_mean = mean_distances.mean()
    global_std = mean_distances.std()

    scores = (mean_distances - global_mean) / (global_std + 1e-6)

    return scores

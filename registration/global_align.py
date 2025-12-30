import numpy as np
from scipy.spatial import cKDTree

def compute_fpfh(points, normals, k=30, radius=0.5):
    """Compute Fast Point Feature Histograms"""
    tree = cKDTree(points)

    n_points = len(points)
    fpfh_features = np.zeros((n_points, 33))

    for i in range(n_points):
        neighbors = tree.query_ball_point(points[i], radius)

        if len(neighbors) < 2:
            continue

        point = points[i]
        normal = normals[i]

        spfh = np.zeros(33)

        for j in neighbors:
            if i == j:
                continue

            neighbor_point = points[j]
            neighbor_normal = normals[j]

            diff = neighbor_point - point
            diff_norm = np.linalg.norm(diff)

            if diff_norm < 1e-6:
                continue

            u = normal
            v = np.cross(u, diff / diff_norm)
            w = np.cross(u, v)

            alpha = np.dot(v, neighbor_normal)
            phi = np.dot(u, diff) / diff_norm
            theta = np.arctan2(np.dot(w, neighbor_normal), np.dot(u, neighbor_normal))

            alpha_bin = int(np.clip((alpha + 1) / 2 * 11, 0, 10))
            phi_bin = int(np.clip((phi + 1) / 2 * 11, 0, 10))
            theta_bin = int(np.clip((theta + np.pi) / (2 * np.pi) * 11, 0, 10))

            spfh[alpha_bin] += 1
            spfh[11 + phi_bin] += 1
            spfh[22 + theta_bin] += 1

        if len(neighbors) > 1:
            spfh = spfh / len(neighbors)

        weighted_sum = np.zeros(33)

        for j in neighbors:
            if i == j:
                continue

            weight = 1.0 / (np.linalg.norm(points[j] - point) + 1e-6)

            neighbor_spfh = np.zeros(33)

            weighted_sum += weight * neighbor_spfh

        if len(neighbors) > 1:
            fpfh_features[i] = spfh + weighted_sum / len(neighbors)

    fpfh_norm = np.linalg.norm(fpfh_features, axis=1, keepdims=True)
    fpfh_norm[fpfh_norm == 0] = 1
    fpfh_features = fpfh_features / fpfh_norm

    return fpfh_features

def ransac_global_registration(source_points, target_points, source_features, target_features,
                                distance_threshold=0.05, max_iterations=100000, confidence=0.999):
    """RANSAC-based global registration using feature correspondences"""
    feature_tree = cKDTree(target_features)

    n_source = len(source_points)

    best_transformation = np.eye(4)
    best_fitness = 0

    n_correspondences = 3

    iteration = 0
    max_iter = max_iterations

    while iteration < max_iter:
        sample_indices = np.random.choice(n_source, n_correspondences, replace=False)

        source_sample = source_points[sample_indices]
        source_features_sample = source_features[sample_indices]

        target_indices = []
        for feat in source_features_sample:
            _, idx = feature_tree.query(feat, k=1)
            target_indices.append(idx)

        target_sample = target_points[target_indices]

        transformation = compute_transformation_from_correspondences(source_sample, target_sample)

        transformed_source = transform_points(source_points, transformation)

        distances = compute_nearest_distances(transformed_source, target_points)

        inliers = distances < distance_threshold
        fitness = inliers.sum() / len(source_points)

        if fitness > best_fitness:
            best_fitness = fitness
            best_transformation = transformation

            inlier_ratio = fitness
            if inlier_ratio > 0:
                new_max_iter = np.log(1 - confidence) / np.log(1 - inlier_ratio ** n_correspondences)
                max_iter = min(max_iter, int(new_max_iter))

        iteration += 1

    return best_transformation, best_fitness

def compute_transformation_from_correspondences(source, target):
    """Compute 4x4 transformation matrix from point correspondences"""
    source_centroid = source.mean(axis=0)
    target_centroid = target.mean(axis=0)

    source_centered = source - source_centroid
    target_centered = target - target_centroid

    H = np.dot(source_centered.T, target_centered)

    U, S, Vt = np.linalg.svd(H)

    R = np.dot(Vt.T, U.T)

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(Vt.T, U.T)

    t = target_centroid - np.dot(R, source_centroid)

    transformation = np.eye(4)
    transformation[:3, :3] = R
    transformation[:3, 3] = t

    return transformation

def transform_points(points, transformation):
    """Apply 4x4 transformation to points"""
    homogeneous = np.hstack([points, np.ones((len(points), 1))])
    transformed = np.dot(homogeneous, transformation.T)

    return transformed[:, :3]

def compute_nearest_distances(source, target):
    """Compute distances to nearest neighbors"""
    tree = cKDTree(target)
    distances, _ = tree.query(source, k=1)

    return distances

def fast_global_registration(source_points, target_points, source_features, target_features,
                             distance_threshold=0.05):
    """Fast Global Registration"""
    feature_tree = cKDTree(target_features)

    correspondences = []

    for i, feat in enumerate(source_features):
        dist, idx = feature_tree.query(feat, k=1)

        if dist < 0.5:
            correspondences.append((i, idx))

    if len(correspondences) < 3:
        return np.eye(4), 0

    source_corr = source_points[[c[0] for c in correspondences]]
    target_corr = target_points[[c[1] for c in correspondences]]

    transformation = compute_transformation_from_correspondences(source_corr, target_corr)

    transformed = transform_points(source_points, transformation)
    distances = compute_nearest_distances(transformed, target_points)

    fitness = (distances < distance_threshold).sum() / len(source_points)

    return transformation, fitness

def evaluate_registration(source, target, transformation, threshold=0.05):
    """Evaluate registration quality"""
    transformed = transform_points(source, transformation)

    distances = compute_nearest_distances(transformed, target)

    inliers = distances < threshold

    fitness = inliers.sum() / len(source)
    inlier_rmse = np.sqrt((distances[inliers] ** 2).mean()) if inliers.sum() > 0 else 0

    return {
        'fitness': fitness,
        'inlier_rmse': inlier_rmse,
        'n_inliers': inliers.sum()
    }

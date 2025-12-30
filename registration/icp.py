import numpy as np
from scipy.spatial import cKDTree

def icp_point_to_point(source, target, max_iterations=50, tolerance=1e-6, initial_transform=None):
    """Point-to-point ICP registration"""
    if initial_transform is None:
        transformation = np.eye(4)
    else:
        transformation = initial_transform.copy()

    from registration.global_align import transform_points

    prev_error = float('inf')

    for iteration in range(max_iterations):
        transformed_source = transform_points(source, transformation)

        tree = cKDTree(target)
        distances, indices = tree.query(transformed_source, k=1)

        correspondences = target[indices]

        T = compute_transformation_svd(transformed_source, correspondences)

        transformation = np.dot(T, transformation)

        mean_error = distances.mean()

        if abs(prev_error - mean_error) < tolerance:
            break

        prev_error = mean_error

    return transformation, mean_error

def icp_point_to_plane(source, target, target_normals, max_iterations=50, tolerance=1e-6, initial_transform=None):
    """Point-to-plane ICP registration"""
    if initial_transform is None:
        transformation = np.eye(4)
    else:
        transformation = initial_transform.copy()

    from registration.global_align import transform_points

    prev_error = float('inf')

    for iteration in range(max_iterations):
        transformed_source = transform_points(source, transformation)

        tree = cKDTree(target)
        distances, indices = tree.query(transformed_source, k=1)

        correspondences = target[indices]
        normals = target_normals[indices]

        T = compute_transformation_point_to_plane(transformed_source, correspondences, normals)

        transformation = np.dot(T, transformation)

        point_to_plane_distances = compute_point_to_plane_distances(
            transformed_source, correspondences, normals
        )

        mean_error = np.abs(point_to_plane_distances).mean()

        if abs(prev_error - mean_error) < tolerance:
            break

        prev_error = mean_error

    return transformation, mean_error

def compute_transformation_svd(source, target):
    """Compute transformation using SVD"""
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

def compute_transformation_point_to_plane(source, target, normals):
    """Compute transformation for point-to-plane ICP"""
    A = np.zeros((len(source), 6))
    b = np.zeros(len(source))

    for i in range(len(source)):
        s = source[i]
        d = target[i]
        n = normals[i]

        A[i] = [
            n[2] * s[1] - n[1] * s[2],
            n[0] * s[2] - n[2] * s[0],
            n[1] * s[0] - n[0] * s[1],
            n[0],
            n[1],
            n[2]
        ]

        b[i] = np.dot(n, d - s)

    x = np.linalg.lstsq(A, b, rcond=None)[0]

    alpha, beta, gamma = x[0], x[1], x[2]
    tx, ty, tz = x[3], x[4], x[5]

    R = compute_rotation_from_angles(alpha, beta, gamma)

    transformation = np.eye(4)
    transformation[:3, :3] = R
    transformation[:3, 3] = [tx, ty, tz]

    return transformation

def compute_rotation_from_angles(alpha, beta, gamma):
    """Compute rotation matrix from small angles"""
    R = np.array([
        [1, -gamma, beta],
        [gamma, 1, -alpha],
        [-beta, alpha, 1]
    ])

    return R

def compute_point_to_plane_distances(source, target, normals):
    """Compute point-to-plane distances"""
    diff = source - target
    distances = np.sum(diff * normals, axis=1)

    return distances

def icp_with_rejection(source, target, distance_threshold=0.1, max_iterations=50, tolerance=1e-6):
    """ICP with outlier rejection"""
    from registration.global_align import transform_points

    transformation = np.eye(4)

    prev_error = float('inf')

    for iteration in range(max_iterations):
        transformed_source = transform_points(source, transformation)

        tree = cKDTree(target)
        distances, indices = tree.query(transformed_source, k=1)

        inliers = distances < distance_threshold

        if inliers.sum() < 3:
            break

        inlier_source = transformed_source[inliers]
        inlier_target = target[indices[inliers]]

        T = compute_transformation_svd(inlier_source, inlier_target)

        transformation = np.dot(T, transformation)

        mean_error = distances[inliers].mean()

        if abs(prev_error - mean_error) < tolerance:
            break

        prev_error = mean_error

    return transformation, mean_error

def icp_symmetric(source, target, max_iterations=50, tolerance=1e-6):
    """Symmetric ICP (bidirectional)"""
    from registration.global_align import transform_points

    transformation = np.eye(4)

    prev_error = float('inf')

    for iteration in range(max_iterations):
        transformed_source = transform_points(source, transformation)

        tree_target = cKDTree(target)
        tree_source = cKDTree(transformed_source)

        distances_s2t, indices_s2t = tree_target.query(transformed_source, k=1)
        distances_t2s, indices_t2s = tree_source.query(target, k=1)

        correspondences_s2t = target[indices_s2t]
        correspondences_t2s = transformed_source[indices_t2s]

        all_source = np.vstack([transformed_source, correspondences_t2s])
        all_target = np.vstack([correspondences_s2t, target])

        T = compute_transformation_svd(all_source, all_target)

        transformation = np.dot(T, transformation)

        mean_error = (distances_s2t.mean() + distances_t2s.mean()) / 2

        if abs(prev_error - mean_error) < tolerance:
            break

        prev_error = mean_error

    return transformation, mean_error

def trimmed_icp(source, target, overlap_ratio=0.8, max_iterations=50, tolerance=1e-6):
    """Trimmed ICP for partial overlap"""
    from registration.global_align import transform_points

    transformation = np.eye(4)

    prev_error = float('inf')

    n_overlapping = int(len(source) * overlap_ratio)

    for iteration in range(max_iterations):
        transformed_source = transform_points(source, transformation)

        tree = cKDTree(target)
        distances, indices = tree.query(transformed_source, k=1)

        sorted_indices = np.argsort(distances)
        overlapping_indices = sorted_indices[:n_overlapping]

        overlapping_source = transformed_source[overlapping_indices]
        overlapping_target = target[indices[overlapping_indices]]

        T = compute_transformation_svd(overlapping_source, overlapping_target)

        transformation = np.dot(T, transformation)

        mean_error = distances[overlapping_indices].mean()

        if abs(prev_error - mean_error) < tolerance:
            break

        prev_error = mean_error

    return transformation, mean_error

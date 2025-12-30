import numpy as np
from scipy.spatial import cKDTree

def estimate_normals(points, k=20, viewpoint=None):
    """Estimate normals using PCA on local neighborhoods"""
    tree = cKDTree(points)
    _, indices = tree.query(points, k=k)

    normals = np.zeros_like(points)

    for i in range(len(points)):
        neighborhood = points[indices[i]]

        centroid = neighborhood.mean(axis=0)
        centered = neighborhood - centroid

        cov = np.dot(centered.T, centered) / k

        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        normal = eigenvectors[:, 0]

        normals[i] = normal

    if viewpoint is not None:
        normals = orient_normals_towards_viewpoint(points, normals, viewpoint)
    else:
        normals = orient_normals_consistently(points, normals, k)

    return normals

def orient_normals_towards_viewpoint(points, normals, viewpoint):
    """Orient normals towards viewpoint"""
    viewpoint = np.array(viewpoint)

    for i in range(len(points)):
        view_vector = viewpoint - points[i]
        view_vector = view_vector / (np.linalg.norm(view_vector) + 1e-10)

        dot_product = np.dot(normals[i], view_vector)

        if dot_product < 0:
            normals[i] = -normals[i]

    return normals

def orient_normals_consistently(points, normals, k=20):
    """Orient normals consistently using propagation"""
    tree = cKDTree(points)

    oriented = np.zeros(len(points), dtype=bool)
    oriented[0] = True

    queue = [0]

    while queue:
        current_idx = queue.pop(0)
        current_normal = normals[current_idx]

        neighbors = tree.query_ball_point(points[current_idx], r=tree.query(points[current_idx], k=k)[0][-1] * 2)

        for neighbor_idx in neighbors:
            if oriented[neighbor_idx]:
                continue

            dot_product = np.dot(current_normal, normals[neighbor_idx])

            if dot_product < 0:
                normals[neighbor_idx] = -normals[neighbor_idx]

            oriented[neighbor_idx] = True
            queue.append(neighbor_idx)

    return normals

def estimate_normals_weighted(points, k=20, sigma=None):
    """Estimate normals with distance-weighted covariance"""
    tree = cKDTree(points)
    distances, indices = tree.query(points, k=k)

    if sigma is None:
        sigma = distances[:, -1].mean()

    normals = np.zeros_like(points)

    for i in range(len(points)):
        neighborhood = points[indices[i]]
        dists = distances[i]

        weights = np.exp(-(dists ** 2) / (2 * sigma ** 2))
        weights = weights / weights.sum()

        centroid = np.average(neighborhood, axis=0, weights=weights)
        centered = neighborhood - centroid

        weighted_centered = centered * weights[:, np.newaxis]

        cov = np.dot(weighted_centered.T, centered) / k

        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        normal = eigenvectors[:, 0]
        normals[i] = normal

    return normals

def estimate_normals_robust(points, k=20, max_iterations=3):
    """Robust normal estimation with outlier rejection"""
    tree = cKDTree(points)
    _, indices = tree.query(points, k=k)

    normals = np.zeros_like(points)

    for i in range(len(points)):
        neighborhood = points[indices[i]]

        for iteration in range(max_iterations):
            centroid = neighborhood.mean(axis=0)
            centered = neighborhood - centroid

            cov = np.dot(centered.T, centered) / len(neighborhood)

            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            normal = eigenvectors[:, 0]

            distances_to_plane = np.abs(np.dot(centered, normal))

            threshold = np.median(distances_to_plane) + 1.5 * np.std(distances_to_plane)

            inliers = distances_to_plane < threshold
            if inliers.sum() < 3:
                break

            neighborhood = neighborhood[inliers]

        normals[i] = normal

    return normals

def compute_normal_variation(points, normals, k=20):
    """Compute variation of normals in local neighborhood"""
    tree = cKDTree(points)
    _, indices = tree.query(points, k=k)

    variation = np.zeros(len(points))

    for i in range(len(points)):
        neighborhood_normals = normals[indices[i]]

        dot_products = np.dot(neighborhood_normals, normals[i])
        dot_products = np.clip(dot_products, -1, 1)

        angles = np.arccos(dot_products)

        variation[i] = angles.std()

    return variation

def smooth_normals(points, normals, k=20, iterations=3):
    """Smooth normals by averaging with neighbors"""
    tree = cKDTree(points)
    _, indices = tree.query(points, k=k)

    smoothed_normals = normals.copy()

    for _ in range(iterations):
        new_normals = np.zeros_like(smoothed_normals)

        for i in range(len(points)):
            neighborhood_normals = smoothed_normals[indices[i]]

            average_normal = neighborhood_normals.mean(axis=0)
            average_normal = average_normal / (np.linalg.norm(average_normal) + 1e-10)

            new_normals[i] = average_normal

        smoothed_normals = new_normals

    return smoothed_normals

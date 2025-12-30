import numpy as np
from scipy.spatial import cKDTree

def compute_height_above_ground(points, ground_mask):
    """Compute height above ground for all points"""
    ground_points = points[ground_mask]

    if len(ground_points) == 0:
        return points[:, 2] - points[:, 2].min()

    ground_tree = cKDTree(ground_points[:, :2])

    heights = np.zeros(len(points))

    for i in range(len(points)):
        point_2d = points[i, :2]

        _, idx = ground_tree.query(point_2d, k=1)
        ground_height = ground_points[idx, 2]

        heights[i] = points[i, 2] - ground_height

    return heights

def compute_height_above_ground_interpolated(points, ground_mask, k=3):
    """Compute HAG using interpolated ground surface"""
    ground_points = points[ground_mask]

    if len(ground_points) == 0:
        return points[:, 2] - points[:, 2].min()

    ground_tree = cKDTree(ground_points[:, :2])

    heights = np.zeros(len(points))

    for i in range(len(points)):
        point_2d = points[i, :2]

        distances, indices = ground_tree.query(point_2d, k=k)

        if distances[0] == 0:
            ground_height = ground_points[indices[0], 2]
        else:
            weights = 1.0 / (distances + 1e-6)
            weights = weights / weights.sum()

            ground_height = np.average(ground_points[indices, 2], weights=weights)

        heights[i] = points[i, 2] - ground_height

    return heights

def compute_relative_height(points, k=20):
    """Compute height relative to local neighborhood"""
    tree = cKDTree(points)
    _, indices = tree.query(points, k=k)

    relative_heights = np.zeros(len(points))

    for i in range(len(points)):
        neighborhood = points[indices[i]]
        min_height = neighborhood[:, 2].min()

        relative_heights[i] = points[i, 2] - min_height

    return relative_heights

def compute_height_range(points, k=20):
    """Compute range of heights in local neighborhood"""
    tree = cKDTree(points)
    _, indices = tree.query(points, k=k)

    height_ranges = np.zeros(len(points))

    for i in range(len(points)):
        neighborhood = points[indices[i]]

        height_ranges[i] = neighborhood[:, 2].max() - neighborhood[:, 2].min()

    return height_ranges

def compute_verticality_from_z(normals):
    """Compute verticality as alignment with Z axis"""
    vertical_vector = np.array([0, 0, 1])

    dot_products = np.abs(np.dot(normals, vertical_vector))

    verticality = dot_products

    return verticality

def compute_height_percentiles(points, k=50):
    """Compute height percentiles in local neighborhood"""
    tree = cKDTree(points)
    _, indices = tree.query(points, k=k)

    percentiles = np.zeros((len(points), 5))

    for i in range(len(points)):
        neighborhood = points[indices[i]]
        heights = neighborhood[:, 2]

        percentiles[i] = np.percentile(heights, [10, 25, 50, 75, 90])

    return percentiles

def classify_height_layers(points, layer_height=3.0):
    """Classify points into height layers"""
    min_z = points[:, 2].min()
    max_z = points[:, 2].max()

    n_layers = int(np.ceil((max_z - min_z) / layer_height))

    layers = np.floor((points[:, 2] - min_z) / layer_height).astype(np.int32)
    layers = np.clip(layers, 0, n_layers - 1)

    return layers

def compute_normalized_height(points):
    """Normalize heights to [0, 1] range"""
    min_z = points[:, 2].min()
    max_z = points[:, 2].max()

    if max_z == min_z:
        return np.zeros(len(points))

    normalized = (points[:, 2] - min_z) / (max_z - min_z)

    return normalized

def detect_height_jumps(points, threshold=0.5, k=20):
    """Detect significant height discontinuities"""
    tree = cKDTree(points[:, :2])
    _, indices = tree.query(points[:, :2], k=k)

    height_jumps = np.zeros(len(points))

    for i in range(len(points)):
        neighborhood = points[indices[i]]
        height_diffs = np.abs(neighborhood[:, 2] - points[i, 2])

        max_diff = height_diffs.max()
        height_jumps[i] = max_diff

    is_jump = height_jumps > threshold

    return height_jumps, is_jump

def compute_height_gradient(points, k=20):
    """Compute gradient of height in local neighborhood"""
    tree = cKDTree(points)
    _, indices = tree.query(points, k=k)

    gradients = np.zeros(len(points))

    for i in range(len(points)):
        neighborhood = points[indices[i]]

        if len(neighborhood) < 3:
            gradients[i] = 0
            continue

        A = np.column_stack([
            neighborhood[:, 0] - points[i, 0],
            neighborhood[:, 1] - points[i, 1],
            np.ones(len(neighborhood))
        ])

        b = neighborhood[:, 2] - points[i, 2]

        try:
            coeffs = np.linalg.lstsq(A, b, rcond=None)[0]
            gradients[i] = np.sqrt(coeffs[0]**2 + coeffs[1]**2)
        except:
            gradients[i] = 0

    return gradients

import numpy as np

def ransac_plane(points, distance_threshold=0.01, max_iterations=1000, min_inliers=100):
    """RANSAC plane detection"""
    best_inliers = []
    best_plane = None

    n_points = len(points)

    for _ in range(max_iterations):
        sample_indices = np.random.choice(n_points, 3, replace=False)
        sample_points = points[sample_indices]

        v1 = sample_points[1] - sample_points[0]
        v2 = sample_points[2] - sample_points[0]

        normal = np.cross(v1, v2)
        normal_length = np.linalg.norm(normal)

        if normal_length < 1e-6:
            continue

        normal = normal / normal_length

        d = -np.dot(normal, sample_points[0])

        distances = np.abs(np.dot(points, normal) + d)

        inliers = np.where(distances < distance_threshold)[0]

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_plane = (normal, d)

    if len(best_inliers) < min_inliers:
        return None, np.array([])

    return best_plane, best_inliers

def ransac_sphere(points, distance_threshold=0.01, max_iterations=1000, min_inliers=100):
    """RANSAC sphere detection"""
    best_inliers = []
    best_sphere = None

    n_points = len(points)

    for _ in range(max_iterations):
        sample_indices = np.random.choice(n_points, 4, replace=False)
        sample_points = points[sample_indices]

        A = 2 * (sample_points[1:] - sample_points[0])
        b = np.sum(sample_points[1:]**2, axis=1) - np.sum(sample_points[0]**2)

        try:
            center = np.linalg.solve(A, b)
        except:
            continue

        radius = np.linalg.norm(sample_points[0] - center)

        distances = np.abs(np.linalg.norm(points - center, axis=1) - radius)

        inliers = np.where(distances < distance_threshold)[0]

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_sphere = (center, radius)

    if len(best_inliers) < min_inliers:
        return None, np.array([])

    return best_sphere, best_inliers

def ransac_cylinder(points, distance_threshold=0.01, max_iterations=1000, min_inliers=100):
    """RANSAC cylinder detection"""
    best_inliers = []
    best_cylinder = None

    n_points = len(points)

    for _ in range(max_iterations):
        sample_indices = np.random.choice(n_points, 2, replace=False)
        sample_points = points[sample_indices]

        axis = sample_points[1] - sample_points[0]
        axis_length = np.linalg.norm(axis)

        if axis_length < 1e-6:
            continue

        axis = axis / axis_length

        point_on_axis = sample_points[0]

        third_idx = np.random.choice(n_points)
        third_point = points[third_idx]

        v = third_point - point_on_axis
        v_parallel = np.dot(v, axis) * axis
        v_perp = v - v_parallel
        radius = np.linalg.norm(v_perp)

        if radius < 1e-6:
            continue

        vectors = points - point_on_axis
        parallel_components = np.dot(vectors, axis)[:, np.newaxis] * axis
        perpendicular_components = vectors - parallel_components

        distances_to_axis = np.linalg.norm(perpendicular_components, axis=1)

        distances = np.abs(distances_to_axis - radius)

        inliers = np.where(distances < distance_threshold)[0]

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_cylinder = (point_on_axis, axis, radius)

    if len(best_inliers) < min_inliers:
        return None, np.array([])

    return best_cylinder, best_inliers

def detect_all_planes(points, distance_threshold=0.01, max_planes=10, min_inliers=100):
    """Detect multiple planes sequentially"""
    planes = []
    remaining_points = points.copy()
    remaining_indices = np.arange(len(points))

    for _ in range(max_planes):
        if len(remaining_points) < min_inliers:
            break

        plane, inliers = ransac_plane(remaining_points, distance_threshold, max_iterations=1000, min_inliers=min_inliers)

        if plane is None:
            break

        original_inliers = remaining_indices[inliers]

        planes.append({
            'plane': plane,
            'inliers': original_inliers,
            'n_inliers': len(original_inliers)
        })

        mask = np.ones(len(remaining_points), dtype=bool)
        mask[inliers] = False
        remaining_points = remaining_points[mask]
        remaining_indices = remaining_indices[mask]

    return planes

def refine_plane(points, plane, max_iterations=10):
    """Refine plane estimate using all inlier points"""
    normal, d = plane

    for _ in range(max_iterations):
        distances = np.dot(points, normal) + d

        centroid = points.mean(axis=0)
        centered = points - centroid

        cov = np.dot(centered.T, centered) / len(points)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        normal = eigenvectors[:, 0]
        d = -np.dot(normal, centroid)

    return (normal, d)

def compute_plane_features(points, plane):
    """Compute features for a detected plane"""
    normal, d = plane

    centroid = points.mean(axis=0)

    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)

    u = np.array([normal[1], -normal[0], 0])
    u_length = np.linalg.norm(u)
    if u_length > 1e-6:
        u = u / u_length
    else:
        u = np.array([1, 0, 0])

    v = np.cross(normal, u)

    points_2d = np.column_stack([
        np.dot(points - centroid, u),
        np.dot(points - centroid, v)
    ])

    area_2d = (points_2d.max(axis=0) - points_2d.min(axis=0)).prod()

    return {
        'normal': normal,
        'd': d,
        'centroid': centroid,
        'bbox_min': bbox_min,
        'bbox_max': bbox_max,
        'area': area_2d,
        'n_points': len(points)
    }

def segment_by_primitives(points, primitive_type='plane', **kwargs):
    """Segment point cloud by fitting geometric primitives"""
    if primitive_type == 'plane':
        planes = detect_all_planes(points, **kwargs)
        labels = np.full(len(points), -1, dtype=np.int32)

        for i, plane_data in enumerate(planes):
            labels[plane_data['inliers']] = i

        return labels, planes

    elif primitive_type == 'sphere':
        sphere, inliers = ransac_sphere(points, **kwargs)
        labels = np.full(len(points), -1, dtype=np.int32)
        if sphere is not None:
            labels[inliers] = 0
        return labels, [{'sphere': sphere, 'inliers': inliers}]

    elif primitive_type == 'cylinder':
        cylinder, inliers = ransac_cylinder(points, **kwargs)
        labels = np.full(len(points), -1, dtype=np.int32)
        if cylinder is not None:
            labels[inliers] = 0
        return labels, [{'cylinder': cylinder, 'inliers': inliers}]

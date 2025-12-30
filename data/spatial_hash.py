import numpy as np

def compute_voxel_grid(points, voxel_size):
    """Convert points to voxel coordinates"""
    voxel_coords = np.floor(points / voxel_size).astype(np.int32)
    return voxel_coords

def voxel_coords_to_hash(voxel_coords):
    """Convert 3D voxel coordinates to 1D hash"""
    p1 = 73856093
    p2 = 19349663
    p3 = 83492791

    hashes = (
        voxel_coords[:, 0] * p1 ^
        voxel_coords[:, 1] * p2 ^
        voxel_coords[:, 2] * p3
    )

    return hashes

def build_spatial_hash(points, voxel_size):
    """Build spatial hash table"""
    voxel_coords = compute_voxel_grid(points, voxel_size)
    hashes = voxel_coords_to_hash(voxel_coords)

    hash_table = {}
    for idx, h in enumerate(hashes):
        if h not in hash_table:
            hash_table[h] = []
        hash_table[h].append(idx)

    return hash_table, voxel_coords

def query_radius(points, query_point, radius, voxel_size, hash_table, voxel_coords):
    """Fast radius query using spatial hash"""
    query_voxel = np.floor(query_point / voxel_size).astype(np.int32)

    search_range = int(np.ceil(radius / voxel_size)) + 1

    candidates = []

    for dx in range(-search_range, search_range + 1):
        for dy in range(-search_range, search_range + 1):
            for dz in range(-search_range, search_range + 1):
                neighbor_voxel = query_voxel + np.array([dx, dy, dz])
                h = voxel_coords_to_hash(neighbor_voxel.reshape(1, -1))[0]

                if h in hash_table:
                    candidates.extend(hash_table[h])

    if not candidates:
        return np.array([], dtype=np.int32), np.array([])

    candidates = np.unique(candidates)
    candidate_points = points[candidates]

    distances = np.linalg.norm(candidate_points - query_point, axis=1)
    mask = distances <= radius

    neighbor_indices = candidates[mask]
    neighbor_distances = distances[mask]

    return neighbor_indices, neighbor_distances

def query_knn(points, query_point, k, voxel_size, hash_table, voxel_coords):
    """Approximate k-NN using expanding search"""
    search_radius = voxel_size * 2

    max_iterations = 10
    for iteration in range(max_iterations):
        indices, distances = query_radius(
            points, query_point, search_radius,
            voxel_size, hash_table, voxel_coords
        )

        if len(indices) >= k:
            sorted_idx = np.argsort(distances)
            return indices[sorted_idx[:k]], distances[sorted_idx[:k]]

        search_radius *= 2

    if len(indices) == 0:
        return np.array([], dtype=np.int32), np.array([])

    sorted_idx = np.argsort(distances)
    return indices[sorted_idx], distances[sorted_idx]

def voxel_downsample(points, voxel_size, colors=None, method='centroid'):
    """Downsample using voxel grid"""
    voxel_coords = compute_voxel_grid(points, voxel_size)
    hashes = voxel_coords_to_hash(voxel_coords)

    unique_hashes, inverse_indices = np.unique(hashes, return_inverse=True)

    downsampled_points = np.zeros((len(unique_hashes), 3))
    downsampled_colors = None
    if colors is not None:
        downsampled_colors = np.zeros((len(unique_hashes), colors.shape[1]))

    for i, h in enumerate(unique_hashes):
        mask = hashes == h
        voxel_points = points[mask]

        if method == 'centroid':
            downsampled_points[i] = voxel_points.mean(axis=0)
        elif method == 'random':
            downsampled_points[i] = voxel_points[np.random.randint(len(voxel_points))]
        elif method == 'closest_to_center':
            center = voxel_points.mean(axis=0)
            distances = np.linalg.norm(voxel_points - center, axis=1)
            downsampled_points[i] = voxel_points[np.argmin(distances)]

        if colors is not None:
            downsampled_colors[i] = colors[mask].mean(axis=0)

    return downsampled_points, downsampled_colors, inverse_indices

def count_points_per_voxel(points, voxel_size):
    """Count number of points in each voxel"""
    voxel_coords = compute_voxel_grid(points, voxel_size)
    hashes = voxel_coords_to_hash(voxel_coords)

    unique_hashes, counts = np.unique(hashes, return_counts=True)

    return unique_hashes, counts

def get_occupied_voxels(points, voxel_size):
    """Get list of occupied voxel coordinates"""
    voxel_coords = compute_voxel_grid(points, voxel_size)
    unique_voxels = np.unique(voxel_coords, axis=0)

    return unique_voxels

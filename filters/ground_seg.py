import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import morphology

def cloth_simulation_filter(points, cloth_resolution=0.5, rigidness=3, iterations=500, classification_threshold=0.5):
    """Ground segmentation using Cloth Simulation Filter"""
    points_2d = points[:, :2]
    heights = points[:, 2]

    min_xy = points_2d.min(axis=0)
    max_xy = points_2d.max(axis=0)

    grid_size_x = int((max_xy[0] - min_xy[0]) / cloth_resolution) + 1
    grid_size_y = int((max_xy[1] - min_xy[1]) / cloth_resolution) + 1

    cloth_nodes = np.zeros((grid_size_x, grid_size_y))

    max_height = heights.max()
    cloth_nodes[:, :] = max_height

    tree = cKDTree(points_2d)

    time_step = 0.65
    gravity = 0.2

    for iteration in range(iterations):
        cloth_velocity = np.zeros_like(cloth_nodes)

        for i in range(grid_size_x):
            for j in range(grid_size_y):
                node_xy = min_xy + np.array([i * cloth_resolution, j * cloth_resolution])

                _, idx = tree.query(node_xy, k=1)
                terrain_height = heights[idx]

                if cloth_nodes[i, j] > terrain_height:
                    cloth_velocity[i, j] -= gravity * time_step

                internal_force = 0.0
                neighbor_count = 0

                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue

                        ni, nj = i + di, j + dj
                        if 0 <= ni < grid_size_x and 0 <= nj < grid_size_y:
                            dist = cloth_resolution * np.sqrt(di**2 + dj**2)
                            displacement = cloth_nodes[ni, nj] - cloth_nodes[i, j]

                            if abs(displacement) > dist * 0.1:
                                internal_force += displacement * rigidness
                                neighbor_count += 1

                if neighbor_count > 0:
                    cloth_velocity[i, j] += internal_force / neighbor_count * time_step

        cloth_nodes += cloth_velocity

        for i in range(grid_size_x):
            for j in range(grid_size_y):
                node_xy = min_xy + np.array([i * cloth_resolution, j * cloth_resolution])
                _, idx = tree.query(node_xy, k=1)
                terrain_height = heights[idx]

                if cloth_nodes[i, j] < terrain_height:
                    cloth_nodes[i, j] = terrain_height

    ground_mask = np.zeros(len(points), dtype=bool)

    for idx in range(len(points)):
        point_2d = points_2d[idx]
        grid_x = int((point_2d[0] - min_xy[0]) / cloth_resolution)
        grid_y = int((point_2d[1] - min_xy[1]) / cloth_resolution)

        grid_x = np.clip(grid_x, 0, grid_size_x - 1)
        grid_y = np.clip(grid_y, 0, grid_size_y - 1)

        cloth_height = cloth_nodes[grid_x, grid_y]

        if abs(heights[idx] - cloth_height) < classification_threshold:
            ground_mask[idx] = True

    return ground_mask

def simple_ground_segmentation(points, grid_size=1.0, height_threshold=0.2):
    """Simple grid-based ground segmentation"""
    points_2d = points[:, :2]
    heights = points[:, 2]

    min_xy = points_2d.min(axis=0)
    max_xy = points_2d.max(axis=0)

    grid_size_x = int((max_xy[0] - min_xy[0]) / grid_size) + 1
    grid_size_y = int((max_xy[1] - min_xy[1]) / grid_size) + 1

    grid_min_heights = np.full((grid_size_x, grid_size_y), np.inf)

    for idx in range(len(points)):
        point_2d = points_2d[idx]
        grid_x = int((point_2d[0] - min_xy[0]) / grid_size)
        grid_y = int((point_2d[1] - min_xy[1]) / grid_size)

        grid_x = np.clip(grid_x, 0, grid_size_x - 1)
        grid_y = np.clip(grid_y, 0, grid_size_y - 1)

        if heights[idx] < grid_min_heights[grid_x, grid_y]:
            grid_min_heights[grid_x, grid_y] = heights[idx]

    ground_mask = np.zeros(len(points), dtype=bool)

    for idx in range(len(points)):
        point_2d = points_2d[idx]
        grid_x = int((point_2d[0] - min_xy[0]) / grid_size)
        grid_y = int((point_2d[1] - min_xy[1]) / grid_size)

        grid_x = np.clip(grid_x, 0, grid_size_x - 1)
        grid_y = np.clip(grid_y, 0, grid_size_y - 1)

        min_height = grid_min_heights[grid_x, grid_y]

        if heights[idx] - min_height < height_threshold:
            ground_mask[idx] = True

    return ground_mask

def progressive_morphological_filter(points, cell_size=1.0, max_window_size=20, slope=0.3, max_distance=3.0):
    """PMF ground segmentation"""
    points_2d = points[:, :2]
    heights = points[:, 2]

    min_xy = points_2d.min(axis=0)
    max_xy = points_2d.max(axis=0)

    grid_size_x = int((max_xy[0] - min_xy[0]) / cell_size) + 1
    grid_size_y = int((max_xy[1] - min_xy[1]) / cell_size) + 1

    grid_min = np.full((grid_size_x, grid_size_y), np.inf)

    for idx in range(len(points)):
        grid_x = int((points_2d[idx, 0] - min_xy[0]) / cell_size)
        grid_y = int((points_2d[idx, 1] - min_xy[1]) / cell_size)

        grid_x = np.clip(grid_x, 0, grid_size_x - 1)
        grid_y = np.clip(grid_y, 0, grid_size_y - 1)

        if heights[idx] < grid_min[grid_x, grid_y]:
            grid_min[grid_x, grid_y] = heights[idx]

    grid_min[np.isinf(grid_min)] = 0

    current_surface = grid_min.copy()

    window_sizes = [3, 5, 7, 9]
    window_sizes = [w for w in window_sizes if w <= max_window_size]

    for window_size in window_sizes:
        opened = morphology.grey_opening(current_surface, size=(window_size, window_size))

        threshold = slope * window_size * cell_size + max_distance

        current_surface = np.where(current_surface - opened < threshold, opened, current_surface)

    ground_mask = np.zeros(len(points), dtype=bool)

    for idx in range(len(points)):
        grid_x = int((points_2d[idx, 0] - min_xy[0]) / cell_size)
        grid_y = int((points_2d[idx, 1] - min_xy[1]) / cell_size)

        grid_x = np.clip(grid_x, 0, grid_size_x - 1)
        grid_y = np.clip(grid_y, 0, grid_size_y - 1)

        surface_height = current_surface[grid_x, grid_y]

        if abs(heights[idx] - surface_height) < max_distance:
            ground_mask[idx] = True

    return ground_mask

def extract_ground_points(points, method='csf', **kwargs):
    """Extract ground points using specified method"""
    if method == 'csf':
        ground_mask = cloth_simulation_filter(points, **kwargs)
    elif method == 'simple':
        ground_mask = simple_ground_segmentation(points, **kwargs)
    elif method == 'pmf':
        ground_mask = progressive_morphological_filter(points, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")

    ground_points = points[ground_mask]
    non_ground_points = points[~ground_mask]

    return ground_points, non_ground_points, ground_mask

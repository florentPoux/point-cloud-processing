import numpy as np
from scipy.spatial import cKDTree

def detect_planes(points, normals=None, distance_threshold=0.05, min_points=100):
    """Detect planar surfaces using RANSAC"""
    from segmentation.ransac_shapes import fit_plane_ransac

    detected_planes = []

    remaining_points = points.copy()
    remaining_indices = np.arange(len(points))

    while len(remaining_points) > min_points:
        plane_model, inliers = fit_plane_ransac(
            remaining_points,
            distance_threshold=distance_threshold,
            min_samples=3,
            max_iterations=1000
        )

        if plane_model is None or np.sum(inliers) < min_points:
            break

        global_inliers = remaining_indices[inliers]

        detected_planes.append({
            'points': points[global_inliers],
            'model': plane_model,
            'indices': global_inliers,
            'num_points': np.sum(inliers)
        })

        remaining_points = remaining_points[~inliers]
        remaining_indices = remaining_indices[~inliers]

    return detected_planes

def classify_planes_as_elements(planes, vertical_threshold=0.1):
    """Classify planes as walls, floors, ceilings"""
    classified = []

    for plane in planes:
        normal = plane['model'][:3]
        normal = normal / np.linalg.norm(normal)

        vertical_component = abs(normal[2])

        if vertical_component > (1 - vertical_threshold):
            if np.mean(plane['points'][:, 2]) < np.percentile(plane['points'][:, 2], 10):
                element_type = 'floor'
            else:
                element_type = 'ceiling'
        else:
            element_type = 'wall'

        classified.append({
            **plane,
            'element_type': element_type,
            'normal': normal
        })

    return classified

def extract_wall_boundaries(wall_points, resolution=0.1):
    """Extract wall boundaries as polylines"""
    from skimage import measure

    min_coords = np.min(wall_points[:, :2], axis=0)
    max_coords = np.max(wall_points[:, :2], axis=0)

    grid_width = int(np.ceil((max_coords[0] - min_coords[0]) / resolution))
    grid_height = int(np.ceil((max_coords[1] - min_coords[1]) / resolution))

    occupancy_grid = np.zeros((grid_height, grid_width))

    pixel_coords = ((wall_points[:, :2] - min_coords) / resolution).astype(int)
    pixel_coords = np.clip(pixel_coords, [0, 0], [grid_width - 1, grid_height - 1])

    for px, py in pixel_coords:
        occupancy_grid[py, px] = 1

    contours = measure.find_contours(occupancy_grid, 0.5)

    boundaries = []

    for contour in contours:
        if len(contour) < 4:
            continue

        world_contour = contour * resolution + min_coords

        simplified = contour[::max(1, len(contour) // 20)]
        world_simplified = simplified * resolution + min_coords

        boundaries.append(world_simplified)

    return boundaries

def detect_doors_windows(wall_points, wall_normal, opening_min_height=0.5):
    """Detect door and window openings in walls"""
    from scipy.ndimage import binary_erosion, label

    min_coords = np.min(wall_points[:, :2], axis=0)
    max_coords = np.max(wall_points[:, :2], axis=0)

    resolution = 0.05

    grid_width = int(np.ceil((max_coords[0] - min_coords[0]) / resolution))
    grid_height = int(np.ceil((wall_points[:, 2].max() - wall_points[:, 2].min()) / resolution))

    occupancy_grid = np.zeros((grid_height, grid_width))

    pixel_x = ((wall_points[:, 0] - min_coords[0]) / resolution).astype(int)
    pixel_y = ((wall_points[:, 2] - wall_points[:, 2].min()) / resolution).astype(int)

    pixel_x = np.clip(pixel_x, 0, grid_width - 1)
    pixel_y = np.clip(pixel_y, 0, grid_height - 1)

    for px, py in zip(pixel_x, pixel_y):
        occupancy_grid[py, px] = 1

    openings_grid = 1 - binary_erosion(occupancy_grid, iterations=2)

    labeled, num_features = label(openings_grid)

    openings = []

    for region_id in range(1, num_features + 1):
        region_mask = labeled == region_id
        region_points = np.argwhere(region_mask)

        if len(region_points) < 10:
            continue

        height = (region_points[:, 0].max() - region_points[:, 0].min()) * resolution
        width = (region_points[:, 1].max() - region_points[:, 1].min()) * resolution

        if height < opening_min_height:
            continue

        center_pixel = region_points.mean(axis=0)
        center_3d = np.array([
            center_pixel[1] * resolution + min_coords[0],
            min_coords[1],
            center_pixel[0] * resolution + wall_points[:, 2].min()
        ])

        if height > 1.5 and center_3d[2] - wall_points[:, 2].min() < 0.5:
            opening_type = 'door'
        else:
            opening_type = 'window'

        openings.append({
            'type': opening_type,
            'center': center_3d,
            'width': width,
            'height': height
        })

    return openings

def detect_columns(points, min_height=2.0, radius_range=(0.1, 0.5)):
    """Detect vertical columns"""
    from segmentation.dbscan import dbscan_clustering

    labels = dbscan_clustering(points[:, :2], eps=0.3, min_samples=20)

    columns = []

    for cluster_id in np.unique(labels):
        if cluster_id == -1:
            continue

        cluster_points = points[labels == cluster_id]

        height = cluster_points[:, 2].max() - cluster_points[:, 2].min()

        if height < min_height:
            continue

        centroid_xy = np.mean(cluster_points[:, :2], axis=0)

        distances = np.linalg.norm(cluster_points[:, :2] - centroid_xy, axis=1)
        radius = np.mean(distances)

        if radius < radius_range[0] or radius > radius_range[1]:
            continue

        columns.append({
            'center': np.append(centroid_xy, cluster_points[:, 2].min()),
            'radius': radius,
            'height': height,
            'points': cluster_points
        })

    return columns

def detect_stairs(points, step_height_range=(0.15, 0.25)):
    """Detect staircases"""
    z_sorted = np.sort(points[:, 2])

    step_heights = np.diff(z_sorted)

    step_candidates = step_heights[
        (step_heights > step_height_range[0]) &
        (step_heights < step_height_range[1])
    ]

    if len(step_candidates) < 3:
        return []

    stairs = []

    from segmentation.dbscan import dbscan_clustering

    labels = dbscan_clustering(points, eps=0.5, min_samples=50)

    for cluster_id in np.unique(labels):
        if cluster_id == -1:
            continue

        cluster_points = points[labels == cluster_id]

        z_range = cluster_points[:, 2].max() - cluster_points[:, 2].min()

        if z_range < 0.5:
            continue

        num_steps = int(z_range / np.mean(step_height_range))

        if num_steps >= 3:
            stairs.append({
                'points': cluster_points,
                'num_steps': num_steps,
                'height': z_range,
                'bounds': {
                    'min': np.min(cluster_points, axis=0),
                    'max': np.max(cluster_points, axis=0)
                }
            })

    return stairs

def extract_building_elements(points, colors=None):
    """Complete building element extraction pipeline"""
    print("Extracting building elements...")

    from features.normals import estimate_normals

    normals = estimate_normals(points, k=20)

    print("  Detecting planes...")
    planes = detect_planes(points, normals)

    print("  Classifying planes...")
    classified_planes = classify_planes_as_elements(planes)

    elements = {
        'walls': [],
        'floors': [],
        'ceilings': [],
        'columns': [],
        'doors': [],
        'windows': [],
        'stairs': []
    }

    for plane in classified_planes:
        if plane['element_type'] in elements:
            elements[plane['element_type']].append(plane)

    print("  Detecting columns...")
    vertical_points = points[abs(normals[:, 2]) < 0.5]
    columns = detect_columns(vertical_points)
    elements['columns'] = columns

    print("  Detecting doors and windows...")
    for wall in elements['walls']:
        openings = detect_doors_windows(wall['points'], wall['normal'])

        for opening in openings:
            if opening['type'] == 'door':
                elements['doors'].append(opening)
            elif opening['type'] == 'window':
                elements['windows'].append(opening)

    print("  Detecting stairs...")
    stairs = detect_stairs(points)
    elements['stairs'] = stairs

    print(f"  Found: {len(elements['walls'])} walls, {len(elements['floors'])} floors, "
          f"{len(elements['columns'])} columns, {len(elements['doors'])} doors, "
          f"{len(elements['windows'])} windows")

    return elements

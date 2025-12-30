import numpy as np

def build_octree(positions, max_depth=8, max_points_per_node=1000):
    """Build octree spatial index for Gaussians"""
    bounds_min = positions.min(axis=0)
    bounds_max = positions.max(axis=0)

    root = {
        'bounds_min': bounds_min,
        'bounds_max': bounds_max,
        'indices': np.arange(len(positions)),
        'children': None,
        'depth': 0
    }

    def subdivide(node):
        if node['depth'] >= max_depth or len(node['indices']) <= max_points_per_node:
            return

        center = (node['bounds_min'] + node['bounds_max']) / 2

        children = []
        for i in range(8):
            child_min = node['bounds_min'].copy()
            child_max = node['bounds_max'].copy()

            if i & 1:
                child_min[0] = center[0]
            else:
                child_max[0] = center[0]

            if i & 2:
                child_min[1] = center[1]
            else:
                child_max[1] = center[1]

            if i & 4:
                child_min[2] = center[2]
            else:
                child_max[2] = center[2]

            node_positions = positions[node['indices']]
            child_mask = (
                (node_positions[:, 0] >= child_min[0]) & (node_positions[:, 0] < child_max[0]) &
                (node_positions[:, 1] >= child_min[1]) & (node_positions[:, 1] < child_max[1]) &
                (node_positions[:, 2] >= child_min[2]) & (node_positions[:, 2] < child_max[2])
            )

            child_indices = node['indices'][child_mask]

            if len(child_indices) > 0:
                child = {
                    'bounds_min': child_min,
                    'bounds_max': child_max,
                    'indices': child_indices,
                    'children': None,
                    'depth': node['depth'] + 1
                }

                subdivide(child)
                children.append(child)

        node['children'] = children if children else None

    subdivide(root)

    return root

def query_octree_frustum(octree, view_matrix, proj_matrix):
    """Query octree for visible Gaussians in view frustum"""
    visible_indices = []

    def traverse(node):
        center = (node['bounds_min'] + node['bounds_max']) / 2
        extent = (node['bounds_max'] - node['bounds_min']) / 2

        corners = []
        for i in range(8):
            corner = center + extent * (2 * np.array([i & 1, (i & 2) >> 1, (i & 4) >> 2]) - 1)
            corners.append(corner)

        corners = np.array(corners)

        in_frustum = check_box_in_frustum(corners, view_matrix, proj_matrix)

        if not in_frustum:
            return

        if node['children'] is None:
            visible_indices.extend(node['indices'])
        else:
            for child in node['children']:
                traverse(child)

    traverse(octree)

    return np.array(visible_indices)

def check_box_in_frustum(corners, view_matrix, proj_matrix):
    """Check if bounding box intersects view frustum"""
    corners_homo = np.hstack([corners, np.ones((len(corners), 1))])

    corners_clip = corners_homo @ view_matrix.T @ proj_matrix.T

    w = corners_clip[:, 3]

    inside_any = np.any(
        (corners_clip[:, 0] >= -w) & (corners_clip[:, 0] <= w) &
        (corners_clip[:, 1] >= -w) & (corners_clip[:, 1] <= w) &
        (corners_clip[:, 2] >= -w) & (corners_clip[:, 2] <= w) &
        (w > 0)
    )

    return inside_any

def assign_to_tiles(positions, tile_size):
    """Assign Gaussians to spatial tiles"""
    bounds_min = positions.min(axis=0)

    tile_coords = np.floor((positions - bounds_min) / tile_size).astype(np.int32)

    unique_tiles = np.unique(tile_coords, axis=0)

    tile_map = {}
    for tile in unique_tiles:
        tile_key = tuple(tile)
        mask = np.all(tile_coords == tile, axis=1)
        tile_map[tile_key] = np.where(mask)[0]

    return tile_map

def save_octree_index(octree, filepath):
    """Save octree index to file"""
    np.savez_compressed(filepath, octree=octree)

def load_octree_index(filepath):
    """Load octree index from file"""
    data = np.load(filepath, allow_pickle=True)
    return data['octree'].item()

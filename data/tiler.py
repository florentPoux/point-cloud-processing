import numpy as np
import os

def create_grid_tiles(bounds_min, bounds_max, tile_size, overlap=0.0):
    """Generate tile boundaries with optional overlap"""
    tiles = []

    x_min, y_min = bounds_min[0], bounds_min[1]
    x_max, y_max = bounds_max[0], bounds_max[1]

    x_start = x_min
    tile_id = 0

    while x_start < x_max:
        x_end = min(x_start + tile_size, x_max)

        y_start = y_min
        while y_start < y_max:
            y_end = min(y_start + tile_size, y_max)

            tile_bounds_min = np.array([
                x_start - overlap,
                y_start - overlap,
                bounds_min[2]
            ])
            tile_bounds_max = np.array([
                x_end + overlap,
                y_end + overlap,
                bounds_max[2]
            ])

            tiles.append({
                'id': tile_id,
                'bounds_min': tile_bounds_min,
                'bounds_max': tile_bounds_max,
                'center': (tile_bounds_min + tile_bounds_max) / 2
            })

            tile_id += 1
            y_start += tile_size

        x_start += tile_size

    return tiles

def filter_points_to_tile(points, tile_bounds_min, tile_bounds_max):
    """Extract points within tile boundaries"""
    mask = (
        (points[:, 0] >= tile_bounds_min[0]) & (points[:, 0] <= tile_bounds_max[0]) &
        (points[:, 1] >= tile_bounds_min[1]) & (points[:, 1] <= tile_bounds_max[1]) &
        (points[:, 2] >= tile_bounds_min[2]) & (points[:, 2] <= tile_bounds_max[2])
    )
    return points[mask], mask

def tile_point_cloud(points, tile_size, overlap=0.0, colors=None, extra_fields=None):
    """Tile point cloud into spatial chunks"""
    bounds_min = points.min(axis=0)
    bounds_max = points.max(axis=0)

    tiles = create_grid_tiles(bounds_min, bounds_max, tile_size, overlap)

    tiled_data = []
    for tile in tiles:
        tile_points, mask = filter_points_to_tile(
            points,
            tile['bounds_min'],
            tile['bounds_max']
        )

        if len(tile_points) == 0:
            continue

        tile_data = {
            'id': tile['id'],
            'points': tile_points,
            'bounds_min': tile['bounds_min'],
            'bounds_max': tile['bounds_max']
        }

        if colors is not None:
            tile_data['colors'] = colors[mask]

        if extra_fields is not None:
            for field_name, field_data in extra_fields.items():
                tile_data[field_name] = field_data[mask]

        tiled_data.append(tile_data)

    return tiled_data

def save_tiles(tiled_data, output_dir, format='npy'):
    """Save tiles to disk"""
    os.makedirs(output_dir, exist_ok=True)

    for tile in tiled_data:
        tile_id = tile['id']

        if format == 'npy':
            filepath = os.path.join(output_dir, f'tile_{tile_id:06d}.npz')
            np.savez_compressed(filepath, **tile)
        elif format == 'las':
            from data.io_manager import write_las_with_predictions
            filepath = os.path.join(output_dir, f'tile_{tile_id:06d}.las')
            write_las_with_predictions(
                filepath,
                tile['points'],
                np.zeros(len(tile['points'])),
                colors=tile.get('colors')
            )

def load_tile(filepath):
    """Load tile from disk"""
    data = np.load(filepath, allow_pickle=True)
    return dict(data)

def get_neighboring_tiles(tile_id, tiles, distance=1):
    """Get tiles within distance of current tile"""
    current_tile = tiles[tile_id]
    current_center = current_tile['center']

    neighbors = []
    for tile in tiles:
        if tile['id'] == tile_id:
            continue

        tile_center = tile['center']
        dist = np.linalg.norm(current_center[:2] - tile_center[:2])

        tile_size = tile['bounds_max'][0] - tile['bounds_min'][0]
        if dist <= distance * tile_size * 1.5:
            neighbors.append(tile)

    return neighbors

def merge_tiles(tiles_data):
    """Merge multiple tiles into single point cloud"""
    all_points = []
    all_colors = []

    for tile in tiles_data:
        all_points.append(tile['points'])
        if 'colors' in tile:
            all_colors.append(tile['colors'])

    merged_points = np.vstack(all_points)

    merged_colors = None
    if all_colors:
        merged_colors = np.vstack(all_colors)

    return merged_points, merged_colors

def adaptive_tile_size(points, target_points_per_tile=1000000):
    """Compute adaptive tile size based on point density"""
    bounds_min = points.min(axis=0)
    bounds_max = points.max(axis=0)

    area = (bounds_max[0] - bounds_min[0]) * (bounds_max[1] - bounds_min[1])
    density = len(points) / area

    tile_size = np.sqrt(target_points_per_tile / density)

    return tile_size

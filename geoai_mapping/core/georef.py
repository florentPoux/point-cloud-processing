import numpy as np
from pathlib import Path

def read_geotiff_metadata(filepath):
    """Read GeoTIFF metadata including georeferencing"""
    try:
        from osgeo import gdal
        gdal.UseExceptions()

        dataset = gdal.Open(str(filepath))
        if dataset is None:
            return None

        geotransform = dataset.GetGeoTransform()
        projection = dataset.GetProjection()

        metadata = {
            'width': dataset.RasterXSize,
            'height': dataset.RasterYSize,
            'bands': dataset.RasterCount,
            'geotransform': geotransform,
            'projection': projection,
            'origin': (geotransform[0], geotransform[3]),
            'pixel_size': (geotransform[1], abs(geotransform[5])),
            'extent': {
                'min_x': geotransform[0],
                'max_y': geotransform[3],
                'max_x': geotransform[0] + geotransform[1] * dataset.RasterXSize,
                'min_y': geotransform[3] + geotransform[5] * dataset.RasterYSize
            }
        }

        dataset = None

        return metadata

    except ImportError:
        return read_worldfile(filepath)

def read_worldfile(image_path):
    """Read worldfile for georeferencing (.tfw, .jgw, etc.)"""
    image_path = Path(image_path)

    worldfile_extensions = [
        image_path.suffix + 'w',
        '.' + image_path.suffix[1] + image_path.suffix[-1] + 'w',
        '.wld'
    ]

    for ext in worldfile_extensions:
        worldfile = image_path.with_suffix(ext)
        if worldfile.exists():
            with open(worldfile, 'r') as f:
                lines = f.readlines()

            if len(lines) >= 6:
                pixel_size_x = float(lines[0].strip())
                rotation_y = float(lines[1].strip())
                rotation_x = float(lines[2].strip())
                pixel_size_y = float(lines[3].strip())
                origin_x = float(lines[4].strip())
                origin_y = float(lines[5].strip())

                return {
                    'geotransform': [origin_x, pixel_size_x, rotation_x, origin_y, rotation_y, pixel_size_y],
                    'origin': (origin_x, origin_y),
                    'pixel_size': (pixel_size_x, abs(pixel_size_y))
                }

    return None

def write_worldfile(image_path, origin, pixel_size):
    """Write worldfile for image georeferencing"""
    image_path = Path(image_path)

    worldfile_ext = '.' + image_path.suffix[1] + image_path.suffix[-1] + 'w'
    worldfile = image_path.with_suffix(worldfile_ext)

    with open(worldfile, 'w') as f:
        f.write(f"{pixel_size[0]}\n")
        f.write("0.0\n")
        f.write("0.0\n")
        f.write(f"{-pixel_size[1]}\n")
        f.write(f"{origin[0]}\n")
        f.write(f"{origin[1]}\n")

def georeference_point_cloud(points, origin, rotation=0, scale=1.0):
    """Apply georeferencing transform to point cloud"""
    cos_r = np.cos(rotation)
    sin_r = np.sin(rotation)

    rotation_matrix = np.array([
        [cos_r, -sin_r, 0],
        [sin_r, cos_r, 0],
        [0, 0, 1]
    ])

    scaled_points = points * scale
    rotated_points = (rotation_matrix @ scaled_points.T).T
    georeferenced = rotated_points + origin

    return georeferenced

def pixel_to_world(pixel_coords, geotransform):
    """Convert pixel coordinates to world coordinates"""
    x_geo = geotransform[0] + pixel_coords[:, 0] * geotransform[1] + pixel_coords[:, 1] * geotransform[2]
    y_geo = geotransform[3] + pixel_coords[:, 0] * geotransform[4] + pixel_coords[:, 1] * geotransform[5]

    return np.column_stack([x_geo, y_geo])

def world_to_pixel(world_coords, geotransform):
    """Convert world coordinates to pixel coordinates"""
    det = geotransform[1] * geotransform[5] - geotransform[2] * geotransform[4]

    if abs(det) < 1e-10:
        return np.zeros((len(world_coords), 2), dtype=int)

    dx = world_coords[:, 0] - geotransform[0]
    dy = world_coords[:, 1] - geotransform[3]

    pixel_x = (geotransform[5] * dx - geotransform[2] * dy) / det
    pixel_y = (-geotransform[4] * dx + geotransform[1] * dy) / det

    return np.column_stack([pixel_x, pixel_y]).astype(int)

def create_spatial_index_grid(points, bounds, cell_size):
    """Create spatial grid index for georeferenced points"""
    min_x, min_y = bounds[:2]
    max_x, max_y = bounds[2:4]

    grid_width = int(np.ceil((max_x - min_x) / cell_size))
    grid_height = int(np.ceil((max_y - min_y) / cell_size))

    grid_indices_x = ((points[:, 0] - min_x) / cell_size).astype(int)
    grid_indices_y = ((points[:, 1] - min_y) / cell_size).astype(int)

    grid_indices_x = np.clip(grid_indices_x, 0, grid_width - 1)
    grid_indices_y = np.clip(grid_indices_y, 0, grid_height - 1)

    linear_indices = grid_indices_y * grid_width + grid_indices_x

    spatial_index = {}
    for i, idx in enumerate(linear_indices):
        if idx not in spatial_index:
            spatial_index[idx] = []
        spatial_index[idx].append(i)

    return spatial_index, (grid_width, grid_height)

def query_spatial_grid(spatial_index, query_point, bounds, cell_size, grid_shape, radius=1):
    """Query points within radius cells of query point"""
    min_x, min_y = bounds[:2]
    grid_width, grid_height = grid_shape

    center_x = int((query_point[0] - min_x) / cell_size)
    center_y = int((query_point[1] - min_y) / cell_size)

    indices = []

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            gx = center_x + dx
            gy = center_y + dy

            if 0 <= gx < grid_width and 0 <= gy < grid_height:
                linear_idx = gy * grid_width + gx
                if linear_idx in spatial_index:
                    indices.extend(spatial_index[linear_idx])

    return indices

def compute_coverage_area(points, pixel_size):
    """Compute total coverage area of georeferenced points"""
    min_coords = np.min(points[:, :2], axis=0)
    max_coords = np.max(points[:, :2], axis=0)

    extent = max_coords - min_coords
    area_m2 = extent[0] * extent[1]

    grid_cells = (extent / pixel_size).astype(int)

    return {
        'area_m2': area_m2,
        'area_km2': area_m2 / 1e6,
        'extent_m': extent,
        'grid_cells': grid_cells
    }

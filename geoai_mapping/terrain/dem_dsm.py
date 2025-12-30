import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter, median_filter

def generate_dem(points, resolution, method='idw', power=2, smooth=False):
    """Generate Digital Elevation Model from point cloud"""
    min_coords = np.min(points[:, :2], axis=0)
    max_coords = np.max(points[:, :2], axis=0)

    grid_width = int(np.ceil((max_coords[0] - min_coords[0]) / resolution))
    grid_height = int(np.ceil((max_coords[1] - min_coords[1]) / resolution))

    dem = np.zeros((grid_height, grid_width))

    grid_x, grid_y = np.meshgrid(
        np.linspace(min_coords[0], max_coords[0], grid_width),
        np.linspace(min_coords[1], max_coords[1], grid_height)
    )

    grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    if method == 'nearest':
        tree = cKDTree(points[:, :2])
        distances, indices = tree.query(grid_points, k=1)
        dem = points[indices, 2].reshape(grid_height, grid_width)

    elif method == 'idw':
        tree = cKDTree(points[:, :2])
        k = min(8, len(points))
        distances, indices = tree.query(grid_points, k=k)

        weights = 1.0 / (distances ** power + 1e-10)
        weights /= weights.sum(axis=1, keepdims=True)

        elevations = points[indices, 2]
        interpolated = (elevations * weights).sum(axis=1)

        dem = interpolated.reshape(grid_height, grid_width)

    elif method == 'max':
        for i, point in enumerate(points):
            px = int((point[0] - min_coords[0]) / resolution)
            py = int((point[1] - min_coords[1]) / resolution)

            if 0 <= px < grid_width and 0 <= py < grid_height:
                dem[py, px] = max(dem[py, px], point[2])

    if smooth:
        dem = gaussian_filter(dem, sigma=1.0)

    origin = min_coords
    geotransform = [origin[0], resolution, 0, origin[1] + grid_height * resolution, 0, -resolution]

    return dem, geotransform

def generate_dsm(points, resolution, method='max'):
    """Generate Digital Surface Model (max elevation in each cell)"""
    return generate_dem(points, resolution, method='max', smooth=False)

def generate_dtm(ground_points, resolution, smooth=True):
    """Generate Digital Terrain Model from classified ground points"""
    dtm, geotransform = generate_dem(ground_points, resolution, method='idw', smooth=smooth)

    dtm = median_filter(dtm, size=3)

    return dtm, geotransform

def compute_chm(dsm, dtm):
    """Compute Canopy Height Model (DSM - DTM)"""
    chm = dsm - dtm
    chm[chm < 0] = 0

    return chm

def compute_slope(dem, resolution):
    """Compute slope from DEM in degrees"""
    dy, dx = np.gradient(dem, resolution)

    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    slope_deg = np.rad2deg(slope_rad)

    return slope_deg

def compute_aspect(dem, resolution):
    """Compute aspect (direction of slope) from DEM"""
    dy, dx = np.gradient(dem, resolution)

    aspect_rad = np.arctan2(-dy, dx)
    aspect_deg = np.rad2deg(aspect_rad)
    aspect_deg = (90 - aspect_deg) % 360

    return aspect_deg

def compute_hillshade(dem, resolution, azimuth=315, altitude=45):
    """Compute hillshade for visualization"""
    azimuth_rad = np.deg2rad(azimuth)
    altitude_rad = np.deg2rad(altitude)

    dy, dx = np.gradient(dem, resolution)

    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    aspect_rad = np.arctan2(-dy, dx)

    hillshade = (
        np.cos(altitude_rad) * np.cos(slope_rad) +
        np.sin(altitude_rad) * np.sin(slope_rad) * np.cos(azimuth_rad - aspect_rad)
    )

    hillshade = np.clip(hillshade * 255, 0, 255).astype(np.uint8)

    return hillshade

def compute_roughness(dem, window_size=3):
    """Compute terrain roughness (standard deviation of elevation)"""
    from scipy.ndimage import generic_filter

    roughness = generic_filter(dem, np.std, size=window_size)

    return roughness

def compute_tpi(dem, window_size=3):
    """Compute Topographic Position Index"""
    from scipy.ndimage import uniform_filter

    mean_elevation = uniform_filter(dem, size=window_size)
    tpi = dem - mean_elevation

    return tpi

def fill_depressions(dem, max_iterations=100):
    """Fill depressions in DEM for hydrological analysis"""
    filled = dem.copy()

    for iteration in range(max_iterations):
        changes = 0

        for i in range(1, filled.shape[0] - 1):
            for j in range(1, filled.shape[1] - 1):
                neighbors = [
                    filled[i-1, j-1], filled[i-1, j], filled[i-1, j+1],
                    filled[i, j-1], filled[i, j+1],
                    filled[i+1, j-1], filled[i+1, j], filled[i+1, j+1]
                ]

                min_neighbor = min(neighbors)

                if filled[i, j] < min_neighbor:
                    filled[i, j] = min_neighbor
                    changes += 1

        if changes == 0:
            break

    return filled

def compute_flow_direction(dem):
    """Compute D8 flow direction"""
    flow_dir = np.zeros_like(dem, dtype=int)

    directions = [
        (-1, -1, 32), (-1, 0, 64), (-1, 1, 128),
        (0, -1, 16), (0, 1, 1),
        (1, -1, 8), (1, 0, 4), (1, 1, 2)
    ]

    for i in range(1, dem.shape[0] - 1):
        for j in range(1, dem.shape[1] - 1):
            max_slope = -np.inf
            max_dir = 0

            for di, dj, direction_code in directions:
                neighbor_elev = dem[i + di, j + dj]
                slope = dem[i, j] - neighbor_elev

                if slope > max_slope:
                    max_slope = slope
                    max_dir = direction_code

            flow_dir[i, j] = max_dir

    return flow_dir

def compute_flow_accumulation(flow_dir):
    """Compute flow accumulation from flow direction"""
    accumulation = np.ones_like(flow_dir, dtype=float)

    direction_mapping = {
        1: (0, 1), 2: (1, 1), 4: (1, 0), 8: (1, -1),
        16: (0, -1), 32: (-1, -1), 64: (-1, 0), 128: (-1, 1)
    }

    for iteration in range(flow_dir.shape[0] * flow_dir.shape[1]):
        changed = False

        for i in range(flow_dir.shape[0]):
            for j in range(flow_dir.shape[1]):
                direction = flow_dir[i, j]

                if direction in direction_mapping:
                    di, dj = direction_mapping[direction]
                    ni, nj = i + di, j + dj

                    if 0 <= ni < flow_dir.shape[0] and 0 <= nj < flow_dir.shape[1]:
                        new_accum = accumulation[ni, nj] + accumulation[i, j]
                        if new_accum > accumulation[ni, nj]:
                            accumulation[ni, nj] = new_accum
                            changed = True

        if not changed:
            break

    return accumulation

def extract_contours(dem, geotransform, interval=10):
    """Extract contour lines from DEM"""
    from skimage import measure

    min_elev = np.floor(np.min(dem) / interval) * interval
    max_elev = np.ceil(np.max(dem) / interval) * interval

    contour_levels = np.arange(min_elev, max_elev + interval, interval)

    contours = []

    for level in contour_levels:
        contour_lines = measure.find_contours(dem, level)

        for contour in contour_lines:
            geo_contour = []
            for point in contour:
                x_geo = geotransform[0] + point[1] * geotransform[1]
                y_geo = geotransform[3] + point[0] * geotransform[5]
                geo_contour.append([x_geo, y_geo, level])

            contours.append({
                'elevation': level,
                'coordinates': np.array(geo_contour)
            })

    return contours

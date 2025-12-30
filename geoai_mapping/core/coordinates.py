import numpy as np

def transform_coordinates(points, from_epsg, to_epsg):
    """Transform points between coordinate systems using EPSG codes"""
    try:
        from pyproj import Transformer
        transformer = Transformer.from_crs(f"EPSG:{from_epsg}", f"EPSG:{to_epsg}", always_xy=True)

        x_transformed, y_transformed, z_transformed = transformer.transform(
            points[:, 0], points[:, 1], points[:, 2] if points.shape[1] > 2 else np.zeros(len(points))
        )

        if points.shape[1] == 2:
            return np.column_stack([x_transformed, y_transformed])
        else:
            return np.column_stack([x_transformed, y_transformed, z_transformed])

    except ImportError:
        return simple_utm_to_wgs84(points) if from_epsg >= 32601 else points

def simple_utm_to_wgs84(utm_points, zone=32, hemisphere='north'):
    """Simplified UTM to WGS84 conversion"""
    x = utm_points[:, 0]
    y = utm_points[:, 1]

    false_easting = 500000.0
    false_northing = 0.0 if hemisphere == 'north' else 10000000.0

    x = (x - false_easting) / 0.9996
    y = (y - false_northing) / 0.9996

    central_meridian = (zone - 1) * 6 - 180 + 3

    e = 0.0818191908426
    e_prime_sq = e**2 / (1 - e**2)

    M = y
    mu = M / (6378137.0 * (1 - e**2/4 - 3*e**4/64 - 5*e**6/256))

    lat_rad = mu
    lon_rad = central_meridian * np.pi / 180 + x / 6378137.0

    lat = lat_rad * 180 / np.pi
    lon = lon_rad * 180 / np.pi

    if utm_points.shape[1] == 3:
        return np.column_stack([lon, lat, utm_points[:, 2]])
    else:
        return np.column_stack([lon, lat])

def wgs84_to_utm(wgs84_points, zone=None):
    """Convert WGS84 lat/lon to UTM"""
    lon = wgs84_points[:, 0]
    lat = wgs84_points[:, 1]

    if zone is None:
        zone = int((lon[0] + 180) / 6) + 1

    central_meridian = (zone - 1) * 6 - 180 + 3
    lon_rad = lon * np.pi / 180
    lat_rad = lat * np.pi / 180
    cm_rad = central_meridian * np.pi / 180

    a = 6378137.0
    e = 0.0818191908426
    e_prime_sq = e**2 / (1 - e**2)

    N = a / np.sqrt(1 - e**2 * np.sin(lat_rad)**2)
    T = np.tan(lat_rad)**2
    C = e_prime_sq * np.cos(lat_rad)**2
    A = (lon_rad - cm_rad) * np.cos(lat_rad)

    M = a * ((1 - e**2/4 - 3*e**4/64 - 5*e**6/256) * lat_rad)

    x = 0.9996 * N * (A + (1-T+C)*A**3/6 + (5-18*T+T**2+72*C-58*e_prime_sq)*A**5/120) + 500000.0
    y = 0.9996 * (M + N*np.tan(lat_rad) * (A**2/2 + (5-T+9*C+4*C**2)*A**4/24 + (61-58*T+T**2+600*C-330*e_prime_sq)*A**6/720))

    if wgs84_points.shape[1] == 3:
        return np.column_stack([x, y, wgs84_points[:, 2]])
    else:
        return np.column_stack([x, y])

def compute_local_tangent_plane(points_wgs84, origin_wgs84):
    """Convert WGS84 points to local tangent plane (ENU) around origin"""
    lat0, lon0, alt0 = origin_wgs84

    lat0_rad = np.deg2rad(lat0)
    lon0_rad = np.deg2rad(lon0)

    lat_rad = np.deg2rad(points_wgs84[:, 0])
    lon_rad = np.deg2rad(points_wgs84[:, 1])
    alt = points_wgs84[:, 2] if points_wgs84.shape[1] > 2 else np.zeros(len(points_wgs84))

    a = 6378137.0
    b = 6356752.314245
    e_sq = 1 - (b**2 / a**2)

    N = a / np.sqrt(1 - e_sq * np.sin(lat_rad)**2)
    N0 = a / np.sqrt(1 - e_sq * np.sin(lat0_rad)**2)

    x_ecef = (N + alt) * np.cos(lat_rad) * np.cos(lon_rad)
    y_ecef = (N + alt) * np.cos(lat_rad) * np.sin(lon_rad)
    z_ecef = ((1 - e_sq) * N + alt) * np.sin(lat_rad)

    x0_ecef = (N0 + alt0) * np.cos(lat0_rad) * np.cos(lon0_rad)
    y0_ecef = (N0 + alt0) * np.cos(lat0_rad) * np.sin(lon0_rad)
    z0_ecef = ((1 - e_sq) * N0 + alt0) * np.sin(lat0_rad)

    dx = x_ecef - x0_ecef
    dy = y_ecef - y0_ecef
    dz = z_ecef - z0_ecef

    east = -np.sin(lon0_rad) * dx + np.cos(lon0_rad) * dy
    north = -np.sin(lat0_rad) * np.cos(lon0_rad) * dx - np.sin(lat0_rad) * np.sin(lon0_rad) * dy + np.cos(lat0_rad) * dz
    up = np.cos(lat0_rad) * np.cos(lon0_rad) * dx + np.cos(lat0_rad) * np.sin(lon0_rad) * dy + np.sin(lat0_rad) * dz

    return np.column_stack([east, north, up])

def get_utm_zone_from_point(lon, lat):
    """Determine UTM zone from longitude/latitude"""
    zone_number = int((lon + 180) / 6) + 1

    if 56 <= lat < 64 and 3 <= lon < 12:
        zone_number = 32
    elif 72 <= lat < 84:
        if 0 <= lon < 9:
            zone_number = 31
        elif 9 <= lon < 21:
            zone_number = 33
        elif 21 <= lon < 33:
            zone_number = 35
        elif 33 <= lon < 42:
            zone_number = 37

    hemisphere = 'north' if lat >= 0 else 'south'

    return zone_number, hemisphere

def compute_bounding_box_wgs84(points):
    """Compute bounding box in WGS84 coordinates"""
    min_coords = np.min(points[:, :2], axis=0)
    max_coords = np.max(points[:, :2], axis=0)

    return {
        'min_lon': min_coords[0],
        'min_lat': min_coords[1],
        'max_lon': max_coords[0],
        'max_lat': max_coords[1],
        'center_lon': (min_coords[0] + max_coords[0]) / 2,
        'center_lat': (min_coords[1] + max_coords[1]) / 2
    }

def create_affine_transform(origin, resolution):
    """Create affine transform for raster georeferencing"""
    return np.array([
        [resolution, 0, origin[0]],
        [0, -resolution, origin[1]],
        [0, 0, 1]
    ])

def apply_affine_transform(affine, pixel_coords):
    """Apply affine transform to pixel coordinates"""
    ones = np.ones((len(pixel_coords), 1))
    pixel_homogeneous = np.column_stack([pixel_coords, ones])

    geo_coords = (affine @ pixel_homogeneous.T).T

    return geo_coords[:, :2]

def inverse_affine_transform(affine, geo_coords):
    """Convert geographic coordinates to pixel coordinates"""
    affine_inv = np.linalg.inv(affine)

    ones = np.ones((len(geo_coords), 1))
    geo_homogeneous = np.column_stack([geo_coords, ones])

    pixel_coords = (affine_inv @ geo_homogeneous.T).T

    return pixel_coords[:, :2].astype(int)

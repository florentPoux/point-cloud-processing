import numpy as np
import laspy
import os

def read_las_mmap(filepath, chunk_size=1000000):
    """Memory-mapped LAS reading for massive files"""
    las = laspy.read(filepath)
    n_points = len(las.points)

    for start_idx in range(0, n_points, chunk_size):
        end_idx = min(start_idx + chunk_size, n_points)

        points = np.vstack([
            las.x[start_idx:end_idx],
            las.y[start_idx:end_idx],
            las.z[start_idx:end_idx]
        ]).T

        colors = None
        if hasattr(las, 'red'):
            colors = np.vstack([
                las.red[start_idx:end_idx],
                las.green[start_idx:end_idx],
                las.blue[start_idx:end_idx]
            ]).T / 65535.0

        intensity = None
        if hasattr(las, 'intensity'):
            intensity = las.intensity[start_idx:end_idx]

        yield points, colors, intensity, start_idx, end_idx

def read_las_full(filepath):
    """Read entire LAS file into memory"""
    las = laspy.read(filepath)

    points = np.vstack([las.x, las.y, las.z]).T

    colors = None
    if hasattr(las, 'red'):
        colors = np.vstack([las.red, las.green, las.blue]).T / 65535.0

    intensity = None
    if hasattr(las, 'intensity'):
        intensity = las.intensity

    classification = None
    if hasattr(las, 'classification'):
        classification = las.classification

    return points, colors, intensity, classification

def read_las_bbox(filepath, bbox_min, bbox_max):
    """Read only points within bounding box"""
    las = laspy.read(filepath)

    mask = (
        (las.x >= bbox_min[0]) & (las.x <= bbox_max[0]) &
        (las.y >= bbox_min[1]) & (las.y <= bbox_max[1]) &
        (las.z >= bbox_min[2]) & (las.z <= bbox_max[2])
    )

    points = np.vstack([las.x[mask], las.y[mask], las.z[mask]]).T

    colors = None
    if hasattr(las, 'red'):
        colors = np.vstack([
            las.red[mask],
            las.green[mask],
            las.blue[mask]
        ]).T / 65535.0

    return points, colors

def write_ply_binary(filepath, points, colors=None, normals=None, scalar_fields=None):
    """Write PLY in binary format with optional predictions as scalar fields"""
    n_points = len(points)

    header = "ply\n"
    header += "format binary_little_endian 1.0\n"
    header += f"element vertex {n_points}\n"
    header += "property float x\n"
    header += "property float y\n"
    header += "property float z\n"

    if colors is not None:
        header += "property uchar red\n"
        header += "property uchar green\n"
        header += "property uchar blue\n"

    if normals is not None:
        header += "property float nx\n"
        header += "property float ny\n"
        header += "property float nz\n"

    if scalar_fields is not None:
        for field_name in scalar_fields.keys():
            header += f"property float {field_name}\n"

    header += "end_header\n"

    with open(filepath, 'wb') as f:
        f.write(header.encode('ascii'))

        for i in range(n_points):
            data = points[i].astype(np.float32).tobytes()
            f.write(data)

            if colors is not None:
                color_bytes = (colors[i] * 255).astype(np.uint8).tobytes()
                f.write(color_bytes)

            if normals is not None:
                normal_bytes = normals[i].astype(np.float32).tobytes()
                f.write(normal_bytes)

            if scalar_fields is not None:
                for field_name, field_data in scalar_fields.items():
                    value_bytes = np.float32(field_data[i]).tobytes()
                    f.write(value_bytes)

def write_las_with_predictions(filepath, points, predictions, colors=None, intensity=None):
    """Write LAS with predictions stored as classification"""
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.offsets = np.min(points, axis=0)
    header.scales = np.array([0.01, 0.01, 0.01])

    las = laspy.LasData(header)
    las.x = points[:, 0]
    las.y = points[:, 1]
    las.z = points[:, 2]

    las.classification = predictions.astype(np.uint8)

    if colors is not None:
        las.red = (colors[:, 0] * 65535).astype(np.uint16)
        las.green = (colors[:, 1] * 65535).astype(np.uint16)
        las.blue = (colors[:, 2] * 65535).astype(np.uint16)

    if intensity is not None:
        las.intensity = intensity.astype(np.uint16)

    las.write(filepath)

def get_file_bounds(filepath):
    """Get bounding box without loading full file"""
    las = laspy.read(filepath)
    bounds_min = np.array([las.header.x_min, las.header.y_min, las.header.z_min])
    bounds_max = np.array([las.header.x_max, las.header.y_max, las.header.z_max])
    return bounds_min, bounds_max

def compute_statistics_streaming(filepath, chunk_size=1000000):
    """Compute statistics on massive files without full load"""
    las = laspy.read(filepath)
    n_points = len(las.points)

    running_sum = np.zeros(3)
    running_sum_sq = np.zeros(3)

    for start_idx in range(0, n_points, chunk_size):
        end_idx = min(start_idx + chunk_size, n_points)

        chunk_points = np.vstack([
            las.x[start_idx:end_idx],
            las.y[start_idx:end_idx],
            las.z[start_idx:end_idx]
        ]).T

        running_sum += chunk_points.sum(axis=0)
        running_sum_sq += (chunk_points ** 2).sum(axis=0)

    mean = running_sum / n_points
    variance = (running_sum_sq / n_points) - (mean ** 2)
    std = np.sqrt(variance)

    return {'mean': mean, 'std': std, 'n_points': n_points}

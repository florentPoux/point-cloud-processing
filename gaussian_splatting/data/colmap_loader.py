import numpy as np
from pathlib import Path

def read_colmap_cameras(cameras_file):
    """Read COLMAP cameras.txt file"""
    cameras = {}

    with open(cameras_file, 'r') as f:
        for line in f:
            if line.startswith('#') or len(line.strip()) == 0:
                continue

            parts = line.strip().split()

            camera_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])

            if model == 'PINHOLE':
                fx = float(parts[4])
                fy = float(parts[5])
                cx = float(parts[6])
                cy = float(parts[7])

                cameras[camera_id] = {
                    'model': model,
                    'width': width,
                    'height': height,
                    'fx': fx,
                    'fy': fy,
                    'cx': cx,
                    'cy': cy
                }

    return cameras

def read_colmap_images(images_file):
    """Read COLMAP images.txt file"""
    images = []

    with open(images_file, 'r') as f:
        lines = [line for line in f if not line.startswith('#') and len(line.strip()) > 0]

        for i in range(0, len(lines), 2):
            parts = lines[i].strip().split()

            image_id = int(parts[0])
            qw, qx, qy, qz = map(float, parts[1:5])
            tx, ty, tz = map(float, parts[5:8])
            camera_id = int(parts[8])
            name = parts[9]

            images.append({
                'id': image_id,
                'quat': np.array([qw, qx, qy, qz]),
                'translation': np.array([tx, ty, tz]),
                'camera_id': camera_id,
                'name': name
            })

    return images

def read_colmap_points3d(points3d_file):
    """Read COLMAP points3D.txt file"""
    points = []

    with open(points3d_file, 'r') as f:
        for line in f:
            if line.startswith('#') or len(line.strip()) == 0:
                continue

            parts = line.strip().split()

            point_id = int(parts[0])
            x, y, z = map(float, parts[1:4])
            r, g, b = map(int, parts[4:7])

            points.append({
                'id': point_id,
                'position': np.array([x, y, z]),
                'color': np.array([r, g, b]) / 255.0
            })

    return points

def load_colmap_dataset(colmap_dir):
    """Load complete COLMAP dataset"""
    colmap_path = Path(colmap_dir)

    cameras = read_colmap_cameras(colmap_path / 'cameras.txt')
    images = read_colmap_images(colmap_path / 'images.txt')
    points3d = read_colmap_points3d(colmap_path / 'points3D.txt')

    positions = np.array([p['position'] for p in points3d])
    colors = np.array([p['color'] for p in points3d])

    views = []
    for img in images:
        camera = cameras[img['camera_id']]

        quat = img['quat']
        trans = img['translation']

        R = quat_to_rotation_matrix(quat)
        t = trans

        view_matrix = np.eye(4)
        view_matrix[:3, :3] = R.T
        view_matrix[:3, 3] = -R.T @ t

        fx, fy = camera['fx'], camera['fy']
        cx, cy = camera['cx'], camera['cy']
        width, height = camera['width'], camera['height']

        fov_x = 2 * np.arctan(width / (2 * fx))
        fov_y = 2 * np.arctan(height / (2 * fy))

        views.append({
            'name': img['name'],
            'view_matrix': view_matrix,
            'width': width,
            'height': height,
            'fov_x': fov_x,
            'fov_y': fov_y,
            'fx': fx,
            'fy': fy,
            'cx': cx,
            'cy': cy
        })

    return {
        'points': positions,
        'colors': colors,
        'views': views,
        'cameras': cameras
    }

def quat_to_rotation_matrix(quat):
    """Convert quaternion to rotation matrix"""
    w, x, y, z = quat

    R = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ])

    return R

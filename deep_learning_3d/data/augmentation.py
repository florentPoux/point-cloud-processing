import numpy as np
import torch

def random_rotation_matrix():
    """Generate random 3D rotation matrix"""
    angles = np.random.uniform(0, 2 * np.pi, 3)

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(angles[0]), -np.sin(angles[0])],
        [0, np.sin(angles[0]), np.cos(angles[0])]
    ])

    Ry = np.array([
        [np.cos(angles[1]), 0, np.sin(angles[1])],
        [0, 1, 0],
        [-np.sin(angles[1]), 0, np.cos(angles[1])]
    ])

    Rz = np.array([
        [np.cos(angles[2]), -np.sin(angles[2]), 0],
        [np.sin(angles[2]), np.cos(angles[2]), 0],
        [0, 0, 1]
    ])

    return Rz @ Ry @ Rx

def rotate_points(points, rotation_matrix=None):
    """Apply rotation to point cloud"""
    if rotation_matrix is None:
        rotation_matrix = random_rotation_matrix()

    if isinstance(points, torch.Tensor):
        rotation_matrix = torch.from_numpy(rotation_matrix).float().to(points.device)
        return torch.matmul(points, rotation_matrix.T)
    else:
        return np.dot(points, rotation_matrix.T)

def rotate_points_z_axis(points, angle=None):
    """Rotate around Z axis only"""
    if angle is None:
        angle = np.random.uniform(0, 2 * np.pi)

    cos_a = np.cos(angle)
    sin_a = np.sin(angle)

    rotation_matrix = np.array([
        [cos_a, -sin_a, 0],
        [sin_a, cos_a, 0],
        [0, 0, 1]
    ])

    if isinstance(points, torch.Tensor):
        rotation_matrix = torch.from_numpy(rotation_matrix).float().to(points.device)
        return torch.matmul(points, rotation_matrix.T)
    else:
        return np.dot(points, rotation_matrix.T)

def random_scale(points, scale_low=0.8, scale_high=1.2):
    """Apply random scaling"""
    scale = np.random.uniform(scale_low, scale_high)

    if isinstance(points, torch.Tensor):
        return points * scale
    else:
        return points * scale

def random_jitter(points, sigma=0.01, clip=0.05):
    """Add random jitter to points"""
    if isinstance(points, torch.Tensor):
        noise = torch.randn_like(points) * sigma
        noise = torch.clamp(noise, -clip, clip)
        return points + noise
    else:
        noise = np.random.randn(*points.shape) * sigma
        noise = np.clip(noise, -clip, clip)
        return points + noise

def random_translate(points, translate_range=0.2):
    """Apply random translation"""
    if isinstance(points, torch.Tensor):
        translation = torch.rand(3, device=points.device) * translate_range - translate_range / 2
        return points + translation
    else:
        translation = np.random.uniform(-translate_range / 2, translate_range / 2, 3)
        return points + translation

def random_point_dropout(points, max_dropout_ratio=0.2):
    """Randomly drop points"""
    dropout_ratio = np.random.random() * max_dropout_ratio

    if isinstance(points, torch.Tensor):
        num_points = points.shape[0]
        keep_indices = torch.rand(num_points, device=points.device) > dropout_ratio
        return points[keep_indices]
    else:
        num_points = points.shape[0]
        keep_indices = np.random.random(num_points) > dropout_ratio
        return points[keep_indices]

def random_point_shuffle(points):
    """Randomly shuffle point order"""
    if isinstance(points, torch.Tensor):
        indices = torch.randperm(points.shape[0], device=points.device)
        return points[indices]
    else:
        indices = np.random.permutation(points.shape[0])
        return points[indices]

def normalize_points(points, method='center'):
    """Normalize point cloud"""
    if method == 'center':
        if isinstance(points, torch.Tensor):
            centroid = torch.mean(points, dim=0, keepdim=True)
            return points - centroid
        else:
            centroid = np.mean(points, axis=0, keepdims=True)
            return points - centroid

    elif method == 'unit_sphere':
        if isinstance(points, torch.Tensor):
            centroid = torch.mean(points, dim=0, keepdim=True)
            points = points - centroid
            max_dist = torch.max(torch.sqrt(torch.sum(points ** 2, dim=1)))
            return points / max_dist
        else:
            centroid = np.mean(points, axis=0, keepdims=True)
            points = points - centroid
            max_dist = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
            return points / max_dist

    return points

def augment_point_cloud(points, config=None):
    """Apply full augmentation pipeline"""
    if config is None:
        config = {
            'rotate': True,
            'scale': True,
            'jitter': True,
            'translate': False,
            'dropout': False
        }

    if config.get('rotate', True):
        points = rotate_points_z_axis(points)

    if config.get('scale', True):
        points = random_scale(points, scale_low=0.9, scale_high=1.1)

    if config.get('jitter', True):
        points = random_jitter(points, sigma=0.01, clip=0.05)

    if config.get('translate', False):
        points = random_translate(points, translate_range=0.1)

    if config.get('dropout', False):
        points = random_point_dropout(points, max_dropout_ratio=0.1)

    return points

def sample_fixed_points(points, num_samples):
    """Sample fixed number of points"""
    num_points = points.shape[0]

    if num_points >= num_samples:
        if isinstance(points, torch.Tensor):
            indices = torch.randperm(num_points, device=points.device)[:num_samples]
            return points[indices]
        else:
            indices = np.random.choice(num_points, num_samples, replace=False)
            return points[indices]
    else:
        if isinstance(points, torch.Tensor):
            repeat = num_samples // num_points
            remainder = num_samples % num_points

            repeated = points.repeat(repeat, 1)
            remainder_indices = torch.randperm(num_points, device=points.device)[:remainder]
            remainder_points = points[remainder_indices]

            return torch.cat([repeated, remainder_points], dim=0)
        else:
            repeat = num_samples // num_points
            remainder = num_samples % num_points

            repeated = np.tile(points, (repeat, 1))
            remainder_indices = np.random.choice(num_points, remainder, replace=False)
            remainder_points = points[remainder_indices]

            return np.vstack([repeated, remainder_points])

def create_augmentation_pipeline(rotate=True, scale=True, jitter=True, normalize=True):
    """Create augmentation function"""
    def augment(points):
        if normalize:
            points = normalize_points(points, method='center')

        if rotate:
            points = rotate_points_z_axis(points)

        if scale:
            points = random_scale(points)

        if jitter:
            points = random_jitter(points)

        return points

    return augment

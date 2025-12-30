import numpy as np
import torch

def create_gaussian_data(positions, colors, scales, rotations, opacities, sh_coeffs=None):
    """Create Gaussian data structure"""
    n_gaussians = len(positions)

    data = {
        'positions': positions,
        'colors': colors,
        'scales': scales,
        'rotations': rotations,
        'opacities': opacities,
        'n_gaussians': n_gaussians
    }

    if sh_coeffs is not None:
        data['sh_coeffs'] = sh_coeffs

    return data

def initialize_from_points(points, colors=None, initial_scale=0.01):
    """Initialize Gaussians from point cloud"""
    n_points = len(points)

    positions = torch.from_numpy(points).float()

    if colors is None:
        colors = torch.rand(n_points, 3)
    else:
        colors = torch.from_numpy(colors).float()

    scales = torch.ones(n_points, 3) * initial_scale

    rotations = torch.zeros(n_points, 4)
    rotations[:, 0] = 1.0

    opacities = torch.ones(n_points, 1) * 0.5

    return positions, colors, scales, rotations, opacities

def quaternion_to_rotation_matrix(quaternions):
    """Convert quaternions to rotation matrices"""
    w, x, y, z = quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], quaternions[:, 3]

    R = torch.zeros(len(quaternions), 3, 3, device=quaternions.device)

    R[:, 0, 0] = 1 - 2*y*y - 2*z*z
    R[:, 0, 1] = 2*x*y - 2*w*z
    R[:, 0, 2] = 2*x*z + 2*w*y

    R[:, 1, 0] = 2*x*y + 2*w*z
    R[:, 1, 1] = 1 - 2*x*x - 2*z*z
    R[:, 1, 2] = 2*y*z - 2*w*x

    R[:, 2, 0] = 2*x*z - 2*w*y
    R[:, 2, 1] = 2*y*z + 2*w*x
    R[:, 2, 2] = 1 - 2*x*x - 2*y*y

    return R

def build_scaling_rotation(scales, rotations):
    """Build scaling and rotation matrix"""
    R = quaternion_to_rotation_matrix(rotations)

    S = torch.zeros(len(scales), 3, 3, device=scales.device)
    S[:, 0, 0] = scales[:, 0]
    S[:, 1, 1] = scales[:, 1]
    S[:, 2, 2] = scales[:, 2]

    M = torch.bmm(R, S)

    return M

def compute_covariance_3d(scales, rotations):
    """Compute 3D covariance from scales and rotations"""
    M = build_scaling_rotation(scales, rotations)

    cov3d = torch.bmm(M, M.transpose(1, 2))

    return cov3d

def get_expon_lr_func(lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000):
    """Exponential learning rate schedule"""
    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            return 0.0
        if lr_delay_steps > 0:
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

def inverse_sigmoid(x):
    """Inverse sigmoid activation"""
    return torch.log(x / (1 - x + 1e-8) + 1e-8)

def activation_to_scale(activation, scale_activation='exp'):
    """Convert activation to scale"""
    if scale_activation == 'exp':
        return torch.exp(activation)
    elif scale_activation == 'softplus':
        return torch.nn.functional.softplus(activation)
    else:
        return activation

def activation_to_opacity(activation):
    """Convert activation to opacity"""
    return torch.sigmoid(activation)

def activation_to_rotation(activation):
    """Convert activation to normalized quaternion"""
    return torch.nn.functional.normalize(activation, dim=-1)

def RGB2SH(rgb):
    """Convert RGB to spherical harmonics C0"""
    return (rgb - 0.5) / 0.28209479177387814

def SH2RGB(sh):
    """Convert spherical harmonics C0 to RGB"""
    return sh * 0.28209479177387814 + 0.5

def eval_sh(deg, sh, dirs):
    """Evaluate spherical harmonics at given directions"""
    C0 = 0.28209479177387814

    result = C0 * sh[..., 0]

    if deg > 0:
        C1 = 0.4886025119029199
        result = result - C1 * sh[..., 1] * dirs[..., 1]
        result = result + C1 * sh[..., 2] * dirs[..., 2]
        result = result - C1 * sh[..., 3] * dirs[..., 0]

    if deg > 1:
        C2_0 = 1.0925484305920792
        C2_1 = -1.0925484305920792
        C2_2 = 0.31539156525252005
        C2_3 = -1.0925484305920792
        C2_4 = 0.5462742152960396

        xx = dirs[..., 0] * dirs[..., 0]
        yy = dirs[..., 1] * dirs[..., 1]
        zz = dirs[..., 2] * dirs[..., 2]
        xy = dirs[..., 0] * dirs[..., 1]
        xz = dirs[..., 0] * dirs[..., 2]
        yz = dirs[..., 1] * dirs[..., 2]

        result = result + C2_0 * sh[..., 4] * xy
        result = result + C2_1 * sh[..., 5] * yz
        result = result + C2_2 * sh[..., 6] * (2.0 * zz - xx - yy)
        result = result + C2_3 * sh[..., 7] * xz
        result = result + C2_4 * sh[..., 8] * (xx - yy)

    return result

def save_gaussian_model(filepath, positions, colors, scales, rotations, opacities, sh_coeffs=None):
    """Save Gaussian model to file"""
    data = {
        'positions': positions.cpu().numpy(),
        'colors': colors.cpu().numpy(),
        'scales': scales.cpu().numpy(),
        'rotations': rotations.cpu().numpy(),
        'opacities': opacities.cpu().numpy()
    }

    if sh_coeffs is not None:
        data['sh_coeffs'] = sh_coeffs.cpu().numpy()

    np.savez_compressed(filepath, **data)

def load_gaussian_model(filepath):
    """Load Gaussian model from file"""
    data = np.load(filepath)

    positions = torch.from_numpy(data['positions']).float()
    colors = torch.from_numpy(data['colors']).float()
    scales = torch.from_numpy(data['scales']).float()
    rotations = torch.from_numpy(data['rotations']).float()
    opacities = torch.from_numpy(data['opacities']).float()

    sh_coeffs = None
    if 'sh_coeffs' in data:
        sh_coeffs = torch.from_numpy(data['sh_coeffs']).float()

    return positions, colors, scales, rotations, opacities, sh_coeffs

import torch
import numpy as np

def compute_projection_matrix(fov_x, fov_y, znear, zfar):
    """Compute projection matrix from field of view"""
    tan_fov_x = np.tan(fov_x * 0.5)
    tan_fov_y = np.tan(fov_y * 0.5)

    top = tan_fov_y * znear
    bottom = -top
    right = tan_fov_x * znear
    left = -right

    P = torch.zeros(4, 4)

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[2, 2] = -(zfar + znear) / (zfar - znear)
    P[2, 3] = -2.0 * zfar * znear / (zfar - znear)
    P[3, 2] = -1.0

    return P

def compute_view_matrix(camera_position, target, up):
    """Compute view matrix from camera parameters"""
    camera_position = torch.tensor(camera_position, dtype=torch.float32)
    target = torch.tensor(target, dtype=torch.float32)
    up = torch.tensor(up, dtype=torch.float32)

    forward = torch.nn.functional.normalize(target - camera_position, dim=0)
    right = torch.nn.functional.normalize(torch.cross(forward, up), dim=0)
    up = torch.cross(right, forward)

    view_matrix = torch.eye(4)
    view_matrix[0, :3] = right
    view_matrix[1, :3] = up
    view_matrix[2, :3] = -forward
    view_matrix[:3, 3] = -torch.stack([
        torch.dot(right, camera_position),
        torch.dot(up, camera_position),
        -torch.dot(forward, camera_position)
    ])

    return view_matrix

def project_points_to_screen(points_3d, view_matrix, proj_matrix, width, height):
    """Project 3D points to 2D screen coordinates"""
    points_homogeneous = torch.cat([points_3d, torch.ones(len(points_3d), 1, device=points_3d.device)], dim=1)

    points_view = torch.matmul(points_homogeneous, view_matrix.T)
    points_clip = torch.matmul(points_view, proj_matrix.T)

    points_ndc = points_clip[:, :3] / (points_clip[:, 3:4] + 1e-7)

    points_screen_x = (points_ndc[:, 0] + 1.0) * 0.5 * width
    points_screen_y = (1.0 - points_ndc[:, 1]) * 0.5 * height
    points_depth = points_ndc[:, 2]

    return torch.stack([points_screen_x, points_screen_y, points_depth], dim=1)

def compute_cov2d_from_cov3d(mean3d, cov3d, view_matrix, proj_matrix, tan_fov_x, tan_fov_y):
    """Project 3D covariance to 2D screen space"""
    t = torch.matmul(view_matrix[:3, :3], mean3d) + view_matrix[:3, 3]

    limx = 1.3 * tan_fov_x
    limy = 1.3 * tan_fov_y
    txtz = t[0] / (t[2] + 1e-7)
    tytz = t[1] / (t[2] + 1e-7)

    t[0] = torch.clamp(txtz, -limx, limx) * t[2]
    t[1] = torch.clamp(tytz, -limy, limy) * t[2]

    J = torch.zeros(3, 3, device=mean3d.device)
    J[0, 0] = 1.0 / (t[2] + 1e-7)
    J[0, 2] = -t[0] / (t[2] * t[2] + 1e-7)
    J[1, 1] = 1.0 / (t[2] + 1e-7)
    J[1, 2] = -t[1] / (t[2] * t[2] + 1e-7)

    W = view_matrix[:3, :3]

    T = torch.matmul(J, W)

    cov2d = torch.matmul(torch.matmul(T, cov3d), T.T)

    cov2d[0, 0] += 0.3
    cov2d[1, 1] += 0.3

    return cov2d[:2, :2]

def compute_extent_from_cov2d(cov2d):
    """Compute screen-space extent from 2D covariance"""
    det = cov2d[0, 0] * cov2d[1, 1] - cov2d[0, 1] * cov2d[1, 0]
    det = torch.clamp(det, min=0.0)

    mid = 0.5 * (cov2d[0, 0] + cov2d[1, 1])

    lambda1 = mid + torch.sqrt(torch.clamp(mid * mid - det, min=0.0))
    lambda2 = mid - torch.sqrt(torch.clamp(mid * mid - det, min=0.0))

    radius = torch.ceil(3.0 * torch.sqrt(torch.max(lambda1, lambda2)))

    return radius

def compute_inverse_cov2d(cov2d):
    """Compute inverse of 2D covariance matrix"""
    det = cov2d[0, 0] * cov2d[1, 1] - cov2d[0, 1] * cov2d[1, 0]
    det = torch.clamp(det, min=1e-7)

    inv_cov2d = torch.zeros_like(cov2d)
    inv_cov2d[0, 0] = cov2d[1, 1] / det
    inv_cov2d[1, 1] = cov2d[0, 0] / det
    inv_cov2d[0, 1] = -cov2d[0, 1] / det
    inv_cov2d[1, 0] = -cov2d[1, 0] / det

    return inv_cov2d

def eval_gaussian_2d(point, mean, inv_cov2d):
    """Evaluate 2D Gaussian at given point"""
    diff = point - mean

    power = -0.5 * (
        diff[0] * diff[0] * inv_cov2d[0, 0] +
        diff[1] * diff[1] * inv_cov2d[1, 1] +
        2.0 * diff[0] * diff[1] * inv_cov2d[0, 1]
    )

    return torch.exp(power)

def compute_tiles(width, height, tile_size=16):
    """Compute tile grid for rendering"""
    tiles_x = (width + tile_size - 1) // tile_size
    tiles_y = (height + tile_size - 1) // tile_size

    return tiles_x, tiles_y

def get_tile_bounds(tile_x, tile_y, tile_size, width, height):
    """Get pixel bounds for a tile"""
    min_x = tile_x * tile_size
    min_y = tile_y * tile_size
    max_x = min(min_x + tile_size, width)
    max_y = min(min_y + tile_size, height)

    return min_x, min_y, max_x, max_y

def compute_color_from_sh(sh_coeffs, direction, degree=0):
    """Compute color from spherical harmonics"""
    from gaussian_splatting.core.gaussian import eval_sh

    color = eval_sh(degree, sh_coeffs, direction)
    color = torch.clamp(color, 0.0, 1.0)

    return color

def alpha_blend(colors, alphas):
    """Alpha composite colors"""
    n = len(colors)

    result = torch.zeros(3, device=colors.device)
    transmittance = 1.0

    for i in range(n):
        alpha = alphas[i]
        result += transmittance * alpha * colors[i]
        transmittance *= (1.0 - alpha)

    return result, transmittance

def compute_gaussian_depth_sort(depths):
    """Sort indices by depth (back to front)"""
    return torch.argsort(depths, descending=True)

def frustum_culling(points, view_matrix, proj_matrix):
    """Cull points outside view frustum"""
    points_homo = torch.cat([points, torch.ones(len(points), 1, device=points.device)], dim=1)

    points_clip = torch.matmul(torch.matmul(points_homo, view_matrix.T), proj_matrix.T)

    w = points_clip[:, 3]

    inside = (
        (points_clip[:, 0] >= -w) & (points_clip[:, 0] <= w) &
        (points_clip[:, 1] >= -w) & (points_clip[:, 1] <= w) &
        (points_clip[:, 2] >= -w) & (points_clip[:, 2] <= w) &
        (w > 0)
    )

    return inside

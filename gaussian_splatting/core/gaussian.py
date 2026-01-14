#%% 00. IMPORTS - All libraries loaded at beginning
import numpy as np
import torch

#%% 01. GAUSSIAN INITIALIZATION - Create 3D Gaussians from various sources

def initialize_from_points(points, colors=None, initial_scale=0.01):
    """Initialize Gaussians from point cloud with adaptive scaling"""
    n_points = len(points)

    positions = torch.from_numpy(points.astype(np.float32)) if isinstance(points, np.ndarray) else points.float()

    # Branch: Use k-NN distances for adaptive initial scales
    if colors is None:
        colors = torch.rand(n_points, 3)
    else:
        colors = torch.from_numpy(colors.astype(np.float32)) if isinstance(colors, np.ndarray) else colors.float()

    scales = torch.ones(n_points, 3) * initial_scale

    # Initialize as identity quaternions (w=1, x=y=z=0)
    rotations = torch.zeros(n_points, 4)
    rotations[:, 0] = 1.0

    opacities = torch.ones(n_points, 1) * 0.5

    return positions, colors, scales, rotations, opacities

# Time to test step 1: Create Gaussians from random points!
# pts = np.random.randn(1000, 3)
# pos, col, scl, rot, opa = initialize_from_points(pts)

def initialize_from_colmap(points, colors, normals=None, scale_factor=1.0):
    """Initialize Gaussians from COLMAP reconstruction with normals"""
    n_points = len(points)

    positions = torch.from_numpy(points.astype(np.float32))
    colors = torch.from_numpy(colors.astype(np.float32))

    # Compute adaptive scales from local point density
    from scipy.spatial import cKDTree
    tree = cKDTree(points)
    distances, _ = tree.query(points, k=4)
    local_scales = distances[:, 1:].mean(axis=1) * scale_factor

    scales = torch.from_numpy(local_scales[:, None].repeat(3, axis=1).astype(np.float32))

    # Initialize rotations from normals if available
    if normals is not None:
        rotations = rotation_from_normals(torch.from_numpy(normals.astype(np.float32)))
    else:
        rotations = torch.zeros(n_points, 4)
        rotations[:, 0] = 1.0

    opacities = torch.ones(n_points, 1) * 0.1

    return positions, colors, scales, rotations, opacities

# Time to test step 2: Initialize from COLMAP sparse reconstruction!
# Florent's Note: COLMAP normals give better initial orientation for planar surfaces

def initialize_random(n_gaussians, bounds=(-1, 1)):
    """Initialize random Gaussians within bounds"""
    positions = torch.rand(n_gaussians, 3) * (bounds[1] - bounds[0]) + bounds[0]
    colors = torch.rand(n_gaussians, 3)
    scales = torch.rand(n_gaussians, 3) * 0.1
    rotations = torch.randn(n_gaussians, 4)
    rotations = torch.nn.functional.normalize(rotations, dim=-1)
    opacities = torch.rand(n_gaussians, 1) * 0.5 + 0.25

    return positions, colors, scales, rotations, opacities

#%% 02. QUATERNION OPERATIONS - Complete quaternion mathematics

def quaternion_multiply(q1, q2):
    """Multiply two quaternions (Hamilton product)"""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2

    return torch.stack([w, x, y, z], dim=-1)

def quaternion_conjugate(q):
    """Compute quaternion conjugate (inverse for unit quaternions)"""
    result = q.clone()
    result[..., 1:] = -result[..., 1:]
    return result

def quaternion_to_rotation_matrix(quaternions):
    """Convert quaternions to 3x3 rotation matrices"""
    w, x, y, z = quaternions[..., 0], quaternions[..., 1], quaternions[..., 2], quaternions[..., 3]

    R = torch.zeros((*quaternions.shape[:-1], 3, 3), device=quaternions.device, dtype=quaternions.dtype)

    R[..., 0, 0] = 1 - 2*y*y - 2*z*z
    R[..., 0, 1] = 2*x*y - 2*w*z
    R[..., 0, 2] = 2*x*z + 2*w*y

    R[..., 1, 0] = 2*x*y + 2*w*z
    R[..., 1, 1] = 1 - 2*x*x - 2*z*z
    R[..., 1, 2] = 2*y*z - 2*w*x

    R[..., 2, 0] = 2*x*z - 2*w*y
    R[..., 2, 1] = 2*y*z + 2*w*x
    R[..., 2, 2] = 1 - 2*x*x - 2*y*y

    return R

# Time to test step 3: Convert quaternions to rotation matrices!
# q = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.707, 0.707, 0.0, 0.0]])
# R = quaternion_to_rotation_matrix(q)

def rotation_matrix_to_quaternion(R):
    """Convert rotation matrices to quaternions"""
    batch_shape = R.shape[:-2]
    R_flat = R.reshape(-1, 3, 3)
    n = R_flat.shape[0]

    q = torch.zeros(n, 4, device=R.device, dtype=R.dtype)

    trace = R_flat[:, 0, 0] + R_flat[:, 1, 1] + R_flat[:, 2, 2]

    # Case 1: trace > 0
    mask = trace > 0
    s = torch.sqrt(trace[mask] + 1.0) * 2
    q[mask, 0] = 0.25 * s
    q[mask, 1] = (R_flat[mask, 2, 1] - R_flat[mask, 1, 2]) / s
    q[mask, 2] = (R_flat[mask, 0, 2] - R_flat[mask, 2, 0]) / s
    q[mask, 3] = (R_flat[mask, 1, 0] - R_flat[mask, 0, 1]) / s

    # Case 2: R[0,0] is largest diagonal
    mask = (~mask) & (R_flat[:, 0, 0] > R_flat[:, 1, 1]) & (R_flat[:, 0, 0] > R_flat[:, 2, 2])
    s = torch.sqrt(1.0 + R_flat[mask, 0, 0] - R_flat[mask, 1, 1] - R_flat[mask, 2, 2]) * 2
    q[mask, 0] = (R_flat[mask, 2, 1] - R_flat[mask, 1, 2]) / s
    q[mask, 1] = 0.25 * s
    q[mask, 2] = (R_flat[mask, 0, 1] + R_flat[mask, 1, 0]) / s
    q[mask, 3] = (R_flat[mask, 0, 2] + R_flat[mask, 2, 0]) / s

    return q.reshape(*batch_shape, 4)

def rotation_from_normals(normals):
    """Compute quaternions from surface normals"""
    normals = torch.nn.functional.normalize(normals, dim=-1)

    # Use z-axis as reference, find rotation that aligns z to normal
    z_axis = torch.tensor([0.0, 0.0, 1.0], device=normals.device)

    # Compute rotation axis (cross product)
    axis = torch.cross(z_axis.expand_as(normals), normals)
    axis_norm = torch.norm(axis, dim=-1, keepdim=True)

    # Compute rotation angle (dot product)
    cos_angle = torch.sum(z_axis * normals, dim=-1, keepdim=True)
    angle = torch.acos(torch.clamp(cos_angle, -1.0, 1.0))

    # Build quaternion from axis-angle
    half_angle = angle / 2
    sin_half = torch.sin(half_angle)

    quaternions = torch.zeros(normals.shape[0], 4, device=normals.device)
    quaternions[:, 0:1] = torch.cos(half_angle)

    # Handle small axis norms (normals parallel to z)
    mask = axis_norm.squeeze() > 1e-6
    quaternions[mask, 1:] = axis[mask] / axis_norm[mask] * sin_half[mask]
    quaternions[~mask, 0] = 1.0

    return quaternions

# Time to test step 4: Create rotations from surface normals!
# Branch: Add exponential map and log map for quaternion optimization

#%% 03. COVARIANCE COMPUTATION - 3D Gaussian covariance matrices

def build_scaling_rotation(scales, rotations):
    """Build combined scaling and rotation matrix RS"""
    R = quaternion_to_rotation_matrix(rotations)

    S = torch.zeros((*scales.shape[:-1], 3, 3), device=scales.device, dtype=scales.dtype)
    S[..., 0, 0] = scales[..., 0]
    S[..., 1, 1] = scales[..., 1]
    S[..., 2, 2] = scales[..., 2]

    M = torch.matmul(R, S)

    return M

def compute_covariance_3d(scales, rotations):
    """Compute 3D covariance matrix from scales and rotations"""
    M = build_scaling_rotation(scales, rotations)

    # Covariance = M @ M^T
    cov3d = torch.matmul(M, M.transpose(-1, -2))

    return cov3d

def compute_covariance_3d_vec(scales, rotations):
    """Compute 3D covariance in vectorized 6D form (upper triangle)"""
    cov3d = compute_covariance_3d(scales, rotations)

    # Extract upper triangle [xx, xy, xz, yy, yz, zz]
    cov_vec = torch.stack([
        cov3d[..., 0, 0],
        cov3d[..., 0, 1],
        cov3d[..., 0, 2],
        cov3d[..., 1, 1],
        cov3d[..., 1, 2],
        cov3d[..., 2, 2]
    ], dim=-1)

    return cov_vec

# Time to test step 5: Compute 3D covariance matrices!
# Florent's Note: Vectorized covariance is more efficient for GPU rasterization

def project_covariance_2d(mean3d, cov3d, viewmat, projmat, img_width, img_height, tan_fovx, tan_fovy):
    """Project 3D Gaussian covariance to 2D screen space"""
    # Transform mean to camera space
    mean_cam = torch.matmul(viewmat[:3, :3], mean3d.T).T + viewmat[:3, 3]

    # Compute Jacobian of projection
    tx, ty, tz = mean_cam[:, 0], mean_cam[:, 1], mean_cam[:, 2]
    tz2 = tz * tz

    J = torch.zeros(mean3d.shape[0], 2, 3, device=mean3d.device)
    J[:, 0, 0] = 1.0 / tz
    J[:, 0, 2] = -tx / tz2
    J[:, 1, 1] = 1.0 / tz
    J[:, 1, 2] = -ty / tz2

    # Apply view matrix rotation to covariance
    W = viewmat[:3, :3]
    cov_cam = torch.matmul(torch.matmul(W, cov3d), W.T)

    # Project: cov2d = J @ cov_cam @ J^T
    cov2d = torch.matmul(torch.matmul(J, cov_cam), J.transpose(1, 2))

    # Add small diagonal for numerical stability
    cov2d[:, 0, 0] += 0.3
    cov2d[:, 1, 1] += 0.3

    return cov2d, mean_cam

# Time to test step 6: Project Gaussians to 2D screen space!
# Branch: Implement EWA splatting for proper resampling

#%% 04. SPHERICAL HARMONICS - View-dependent appearance

def RGB2SH(rgb):
    """Convert RGB colors to spherical harmonics C0 coefficient"""
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0

def SH2RGB(sh):
    """Convert spherical harmonics C0 to RGB colors"""
    C0 = 0.28209479177387814
    return sh * C0 + 0.5

def eval_sh_deg0(sh, dirs):
    """Evaluate degree 0 SH (constant color)"""
    C0 = 0.28209479177387814
    return C0 * sh[..., 0:1]

def eval_sh_deg1(sh, dirs):
    """Evaluate degree 1 SH (linear directional)"""
    result = eval_sh_deg0(sh, dirs)

    C1 = 0.4886025119029199
    x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]

    result = result - C1 * sh[..., 1:2] * y
    result = result + C1 * sh[..., 2:3] * z
    result = result - C1 * sh[..., 3:4] * x

    return result

def eval_sh_deg2(sh, dirs):
    """Evaluate degree 2 SH (quadratic directional)"""
    result = eval_sh_deg1(sh, dirs)

    C2_0 = 1.0925484305920792
    C2_1 = -1.0925484305920792
    C2_2 = 0.31539156525252005
    C2_3 = -1.0925484305920792
    C2_4 = 0.5462742152960396

    x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z

    result = result + C2_0 * sh[..., 4:5] * xy
    result = result + C2_1 * sh[..., 5:6] * yz
    result = result + C2_2 * sh[..., 6:7] * (2.0 * zz - xx - yy)
    result = result + C2_3 * sh[..., 7:8] * xz
    result = result + C2_4 * sh[..., 8:9] * (xx - yy)

    return result

def eval_sh_deg3(sh, dirs):
    """Evaluate degree 3 SH (cubic directional)"""
    result = eval_sh_deg2(sh, dirs)

    C3_0 = -0.5900435899266435
    C3_1 = 2.890611442640554
    C3_2 = -0.4570457994644658
    C3_3 = 0.3731763325901154
    C3_4 = -0.4570457994644658
    C3_5 = 1.445305721320277
    C3_6 = -0.5900435899266435

    x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z

    result = result + C3_0 * sh[..., 9:10] * y * (3*xx - yy)
    result = result + C3_1 * sh[..., 10:11] * xy * z
    result = result + C3_2 * sh[..., 11:12] * y * (4*zz - xx - yy)
    result = result + C3_3 * sh[..., 12:13] * z * (2*zz - 3*xx - 3*yy)
    result = result + C3_4 * sh[..., 13:14] * x * (4*zz - xx - yy)
    result = result + C3_5 * sh[..., 14:15] * z * (xx - yy)
    result = result + C3_6 * sh[..., 15:16] * x * (xx - 3*yy)

    return result

def eval_sh(deg, sh, dirs):
    """Evaluate spherical harmonics up to specified degree"""
    if deg == 0:
        return eval_sh_deg0(sh, dirs)
    elif deg == 1:
        return eval_sh_deg1(sh, dirs)
    elif deg == 2:
        return eval_sh_deg2(sh, dirs)
    elif deg == 3:
        return eval_sh_deg3(sh, dirs)
    else:
        return eval_sh_deg0(sh, dirs)

# Time to test step 7: Evaluate view-dependent colors with SH!
# Florent's Note: Higher SH degrees capture specular effects but increase memory

#%% 05. ACTIVATION FUNCTIONS - Parameter transformations

def inverse_sigmoid(x, eps=1e-8):
    """Inverse sigmoid for opacity initialization"""
    x = torch.clamp(x, eps, 1.0 - eps)
    return torch.log(x / (1.0 - x))

def activation_to_scale(activation, scale_activation='exp'):
    """Convert activation to positive scale values"""
    if scale_activation == 'exp':
        return torch.exp(activation)
    elif scale_activation == 'softplus':
        return torch.nn.functional.softplus(activation)
    elif scale_activation == 'relu':
        return torch.nn.functional.relu(activation) + 1e-6
    else:
        return torch.abs(activation) + 1e-6

def activation_to_opacity(activation):
    """Convert activation to opacity in [0, 1]"""
    return torch.sigmoid(activation)

def activation_to_rotation(activation):
    """Convert activation to normalized quaternion"""
    return torch.nn.functional.normalize(activation, dim=-1)

def scale_to_activation(scale, scale_activation='exp'):
    """Convert scale to activation (inverse transform)"""
    if scale_activation == 'exp':
        return torch.log(scale + 1e-8)
    elif scale_activation == 'softplus':
        return torch.log(torch.exp(scale) - 1.0 + 1e-8)
    else:
        return scale

# Time to test step 8: Transform parameters with activation functions!
# Branch: Add learnable activation functions for better optimization

#%% 06. ADAPTIVE PARAMETER UPDATES - Densification primitives

def clone_gaussians(positions, colors, scales, rotations, opacities, clone_mask):
    """Clone Gaussians at specified indices"""
    n_clones = clone_mask.sum().item()

    new_positions = positions[clone_mask].clone()
    new_colors = colors[clone_mask].clone()
    new_scales = scales[clone_mask].clone()
    new_rotations = rotations[clone_mask].clone()
    new_opacities = opacities[clone_mask].clone()

    positions_out = torch.cat([positions, new_positions], dim=0)
    colors_out = torch.cat([colors, new_colors], dim=0)
    scales_out = torch.cat([scales, new_scales], dim=0)
    rotations_out = torch.cat([rotations, new_rotations], dim=0)
    opacities_out = torch.cat([opacities, new_opacities], dim=0)

    return positions_out, colors_out, scales_out, rotations_out, opacities_out

def split_gaussians(positions, colors, scales, rotations, opacities, split_mask, split_factor=1.6):
    """Split Gaussians into two smaller ones"""
    n_splits = split_mask.sum().item()

    # Original gaussians to split
    split_positions = positions[split_mask]
    split_colors = colors[split_mask]
    split_scales = scales[split_mask] / split_factor
    split_rotations = rotations[split_mask]
    split_opacities = opacities[split_mask]

    # Create two samples per split Gaussian
    samples = torch.randn(n_splits, 2, 3, device=positions.device) * split_scales.unsqueeze(1)

    # Rotate samples by Gaussian orientation
    R = quaternion_to_rotation_matrix(split_rotations)
    samples = torch.matmul(samples, R.transpose(1, 2))

    # Create new positions as offsets from original
    new_pos1 = split_positions + samples[:, 0]
    new_pos2 = split_positions + samples[:, 1]

    # Replace split Gaussians with two new ones
    positions_out = torch.cat([positions[~split_mask], new_pos1, new_pos2], dim=0)
    colors_out = torch.cat([colors[~split_mask], split_colors, split_colors], dim=0)
    scales_out = torch.cat([scales[~split_mask], split_scales, split_scales], dim=0)
    rotations_out = torch.cat([rotations[~split_mask], split_rotations, split_rotations], dim=0)
    opacities_out = torch.cat([opacities[~split_mask], split_opacities, split_opacities], dim=0)

    return positions_out, colors_out, scales_out, rotations_out, opacities_out

def prune_gaussians(positions, colors, scales, rotations, opacities, keep_mask):
    """Remove Gaussians based on mask"""
    return (
        positions[keep_mask],
        colors[keep_mask],
        scales[keep_mask],
        rotations[keep_mask],
        opacities[keep_mask]
    )

# Time to test step 9: Clone and split Gaussians for densification!
# Florent's Note: Adaptive densification increases detail in complex regions

#%% 07. VISIBILITY AND CULLING - Efficient rendering

def compute_view_frustum_culling(positions, viewmat, projmat, img_width, img_height):
    """Cull Gaussians outside view frustum"""
    # Transform to clip space
    positions_h = torch.cat([positions, torch.ones(len(positions), 1, device=positions.device)], dim=1)
    positions_clip = torch.matmul(positions_h, torch.matmul(viewmat, projmat).T)

    # Perspective divide
    positions_ndc = positions_clip[:, :3] / (positions_clip[:, 3:4] + 1e-8)

    # Check if within [-1, 1] NDC bounds
    visible = (
        (positions_ndc[:, 0] >= -1.0) & (positions_ndc[:, 0] <= 1.0) &
        (positions_ndc[:, 1] >= -1.0) & (positions_ndc[:, 1] <= 1.0) &
        (positions_ndc[:, 2] >= -1.0) & (positions_ndc[:, 2] <= 1.0) &
        (positions_clip[:, 3] > 0.1)
    )

    return visible

def compute_tile_culling(positions_2d, radii, tile_size=16):
    """Compute which tiles each Gaussian overlaps"""
    min_x = (positions_2d[:, 0] - radii) // tile_size
    max_x = (positions_2d[:, 0] + radii) // tile_size
    min_y = (positions_2d[:, 1] - radii) // tile_size
    max_y = (positions_2d[:, 1] + radii) // tile_size

    return min_x.long(), max_x.long(), min_y.long(), max_y.long()

def compute_depth_sorting(positions, viewmat):
    """Compute depth for back-to-front sorting"""
    positions_h = torch.cat([positions, torch.ones(len(positions), 1, device=positions.device)], dim=1)
    positions_cam = torch.matmul(positions_h, viewmat.T)
    depths = positions_cam[:, 2]

    return depths

# Time to test step 10: Perform frustum culling and depth sorting!
# Branch: Implement hierarchical Z-buffer for early depth rejection

#%% 08. LEARNING RATE SCHEDULES - Adaptive optimization

def get_expon_lr_func(lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000):
    """Exponential learning rate schedule with optional delay"""
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

def get_cosine_lr_func(lr_init, lr_final, max_steps):
    """Cosine annealing learning rate schedule"""
    def helper(step):
        if step >= max_steps:
            return lr_final
        t = step / max_steps
        return lr_final + 0.5 * (lr_init - lr_final) * (1 + np.cos(np.pi * t))

    return helper

def get_warmup_cosine_lr_func(lr_init, lr_final, warmup_steps, max_steps):
    """Cosine schedule with linear warmup"""
    def helper(step):
        if step < warmup_steps:
            return lr_init * step / warmup_steps
        else:
            t = (step - warmup_steps) / (max_steps - warmup_steps)
            return lr_final + 0.5 * (lr_init - lr_final) * (1 + np.cos(np.pi * t))

    return helper

# Time to test step 11: Create learning rate schedules!
# Florent's Note: Different parameters need different LR schedules

#%% 09. SERIALIZATION AND BATCHING - Save/load and efficient processing

def create_gaussian_data(positions, colors, scales, rotations, opacities, sh_coeffs=None):
    """Create Gaussian data structure dictionary"""
    data = {
        'positions': positions,
        'colors': colors,
        'scales': scales,
        'rotations': rotations,
        'opacities': opacities,
        'n_gaussians': len(positions)
    }

    if sh_coeffs is not None:
        data['sh_coeffs'] = sh_coeffs

    return data

def save_gaussian_model(filepath, positions, colors, scales, rotations, opacities, sh_coeffs=None):
    """Save Gaussian model to compressed numpy format"""
    data = {
        'positions': positions.cpu().numpy() if torch.is_tensor(positions) else positions,
        'colors': colors.cpu().numpy() if torch.is_tensor(colors) else colors,
        'scales': scales.cpu().numpy() if torch.is_tensor(scales) else scales,
        'rotations': rotations.cpu().numpy() if torch.is_tensor(rotations) else rotations,
        'opacities': opacities.cpu().numpy() if torch.is_tensor(opacities) else opacities
    }

    if sh_coeffs is not None:
        data['sh_coeffs'] = sh_coeffs.cpu().numpy() if torch.is_tensor(sh_coeffs) else sh_coeffs

    np.savez_compressed(filepath, **data)

def load_gaussian_model(filepath, device='cpu'):
    """Load Gaussian model from file"""
    data = np.load(filepath)

    positions = torch.from_numpy(data['positions']).float().to(device)
    colors = torch.from_numpy(data['colors']).float().to(device)
    scales = torch.from_numpy(data['scales']).float().to(device)
    rotations = torch.from_numpy(data['rotations']).float().to(device)
    opacities = torch.from_numpy(data['opacities']).float().to(device)

    sh_coeffs = None
    if 'sh_coeffs' in data:
        sh_coeffs = torch.from_numpy(data['sh_coeffs']).float().to(device)

    return positions, colors, scales, rotations, opacities, sh_coeffs

def batch_gaussians(positions, colors, scales, rotations, opacities, batch_size):
    """Create batches of Gaussians for processing"""
    n_gaussians = len(positions)
    n_batches = (n_gaussians + batch_size - 1) // batch_size

    batches = []
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_gaussians)

        batch = {
            'positions': positions[start_idx:end_idx],
            'colors': colors[start_idx:end_idx],
            'scales': scales[start_idx:end_idx],
            'rotations': rotations[start_idx:end_idx],
            'opacities': opacities[start_idx:end_idx],
            'indices': (start_idx, end_idx)
        }
        batches.append(batch)

    return batches

# Time to test step 12: Save and load Gaussian models!
# Branch: Add PLY export for compatibility with other 3DGS viewers

# Florent's Note: This comprehensive Gaussian implementation covers all core 3DGS operations
# from initialization through optimization to rendering primitives.

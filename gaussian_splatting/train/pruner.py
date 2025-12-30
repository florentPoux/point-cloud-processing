import torch

def prune_low_opacity(positions, colors, scales, rotations, opacities, opacity_threshold=0.005, sh_coeffs=None):
    """Remove Gaussians with low opacity"""
    keep_mask = opacities.squeeze() > opacity_threshold

    n_pruned = (~keep_mask).sum().item()

    if n_pruned == 0:
        return positions, colors, scales, rotations, opacities, sh_coeffs

    pruned_positions = positions[keep_mask]
    pruned_colors = colors[keep_mask]
    pruned_scales = scales[keep_mask]
    pruned_rotations = rotations[keep_mask]
    pruned_opacities = opacities[keep_mask]

    if sh_coeffs is not None:
        pruned_sh = sh_coeffs[keep_mask]
    else:
        pruned_sh = None

    return pruned_positions, pruned_colors, pruned_scales, pruned_rotations, pruned_opacities, pruned_sh

def prune_large_scale(positions, colors, scales, rotations, opacities, scale_threshold=0.5, sh_coeffs=None):
    """Remove Gaussians that are too large"""
    max_scales = scales.max(dim=1)[0]
    keep_mask = max_scales < scale_threshold

    n_pruned = (~keep_mask).sum().item()

    if n_pruned == 0:
        return positions, colors, scales, rotations, opacities, sh_coeffs

    pruned_positions = positions[keep_mask]
    pruned_colors = colors[keep_mask]
    pruned_scales = scales[keep_mask]
    pruned_rotations = rotations[keep_mask]
    pruned_opacities = opacities[keep_mask]

    if sh_coeffs is not None:
        pruned_sh = sh_coeffs[keep_mask]
    else:
        pruned_sh = None

    return pruned_positions, pruned_colors, pruned_scales, pruned_rotations, pruned_opacities, pruned_sh

def prune_by_screen_size(positions, colors, scales, rotations, opacities, view_matrix, proj_matrix,
                        width, height, fov_x, fov_y, screen_size_threshold=2.0, sh_coeffs=None):
    """Remove Gaussians that are too small in screen space"""
    from gaussian_splatting.core.gaussian import compute_covariance_3d
    from gaussian_splatting.core.math_utils import compute_cov2d_from_cov3d, compute_extent_from_cov2d
    import numpy as np

    cov3d = compute_covariance_3d(scales, rotations)

    tan_fov_x = np.tan(fov_x * 0.5)
    tan_fov_y = np.tan(fov_y * 0.5)

    keep_mask = torch.zeros(len(positions), dtype=torch.bool, device=positions.device)

    for i in range(len(positions)):
        cov_2d = compute_cov2d_from_cov3d(
            positions[i], cov3d[i], view_matrix, proj_matrix, tan_fov_x, tan_fov_y
        )

        extent = compute_extent_from_cov2d(cov_2d)

        if extent > screen_size_threshold:
            keep_mask[i] = True

    n_pruned = (~keep_mask).sum().item()

    if n_pruned == 0:
        return positions, colors, scales, rotations, opacities, sh_coeffs

    pruned_positions = positions[keep_mask]
    pruned_colors = colors[keep_mask]
    pruned_scales = scales[keep_mask]
    pruned_rotations = rotations[keep_mask]
    pruned_opacities = opacities[keep_mask]

    if sh_coeffs is not None:
        pruned_sh = sh_coeffs[keep_mask]
    else:
        pruned_sh = None

    return pruned_positions, pruned_colors, pruned_scales, pruned_rotations, pruned_opacities, pruned_sh

def prune_by_contribution(positions, colors, scales, rotations, opacities, contribution_scores,
                         contribution_threshold=0.01, sh_coeffs=None):
    """Remove Gaussians that contribute little to rendering"""
    keep_mask = contribution_scores > contribution_threshold

    n_pruned = (~keep_mask).sum().item()

    if n_pruned == 0:
        return positions, colors, scales, rotations, opacities, sh_coeffs

    pruned_positions = positions[keep_mask]
    pruned_colors = colors[keep_mask]
    pruned_scales = scales[keep_mask]
    pruned_rotations = rotations[keep_mask]
    pruned_opacities = opacities[keep_mask]

    if sh_coeffs is not None:
        pruned_sh = sh_coeffs[keep_mask]
    else:
        pruned_sh = None

    return pruned_positions, pruned_colors, pruned_scales, pruned_rotations, pruned_opacities, pruned_sh

def compute_contribution_scores(positions, opacities, view_matrix):
    """Compute contribution score based on visibility and opacity"""
    positions_homo = torch.cat([positions, torch.ones(len(positions), 1, device=positions.device)], dim=1)
    positions_view = torch.matmul(positions_homo, view_matrix.T)

    depths = positions_view[:, 2]

    visibility_scores = torch.sigmoid(-depths / 10.0)

    contribution_scores = opacities.squeeze() * visibility_scores

    return contribution_scores

def comprehensive_pruning(positions, colors, scales, rotations, opacities,
                         opacity_threshold=0.005,
                         scale_threshold=0.5,
                         view_matrix=None,
                         contribution_threshold=0.01,
                         sh_coeffs=None):
    """Comprehensive pruning with multiple criteria"""
    positions, colors, scales, rotations, opacities, sh_coeffs = prune_low_opacity(
        positions, colors, scales, rotations, opacities, opacity_threshold, sh_coeffs
    )

    positions, colors, scales, rotations, opacities, sh_coeffs = prune_large_scale(
        positions, colors, scales, rotations, opacities, scale_threshold, sh_coeffs
    )

    if view_matrix is not None:
        contribution_scores = compute_contribution_scores(positions, opacities, view_matrix)

        positions, colors, scales, rotations, opacities, sh_coeffs = prune_by_contribution(
            positions, colors, scales, rotations, opacities, contribution_scores,
            contribution_threshold, sh_coeffs
        )

    return positions, colors, scales, rotations, opacities, sh_coeffs

def adaptive_pruning_schedule(iteration, total_iterations):
    """Compute adaptive pruning thresholds based on training progress"""
    progress = iteration / total_iterations

    opacity_threshold = 0.005 + 0.01 * progress

    scale_threshold = 0.5 - 0.3 * progress

    contribution_threshold = 0.01 + 0.02 * progress

    return {
        'opacity_threshold': opacity_threshold,
        'scale_threshold': scale_threshold,
        'contribution_threshold': contribution_threshold
    }

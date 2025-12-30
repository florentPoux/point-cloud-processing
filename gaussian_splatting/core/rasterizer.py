import torch
import numpy as np

def rasterize_gaussians(positions, colors, scales, rotations, opacities, view_matrix, proj_matrix,
                       width, height, fov_x, fov_y, sh_degree=0, sh_coeffs=None, background=None):
    """CPU rasterizer for Gaussian splatting"""
    from gaussian_splatting.core.gaussian import compute_covariance_3d
    from gaussian_splatting.core.math_utils import (
        project_points_to_screen, compute_cov2d_from_cov3d, compute_extent_from_cov2d,
        compute_inverse_cov2d, eval_gaussian_2d, frustum_culling, compute_gaussian_depth_sort,
        compute_tiles, get_tile_bounds, compute_color_from_sh
    )

    if background is None:
        background = torch.zeros(3, device=positions.device)

    n_gaussians = len(positions)

    inside_frustum = frustum_culling(positions, view_matrix, proj_matrix)
    visible_indices = torch.where(inside_frustum)[0]

    if len(visible_indices) == 0:
        return background.unsqueeze(0).unsqueeze(0).expand(height, width, 3)

    visible_positions = positions[visible_indices]
    visible_colors = colors[visible_indices]
    visible_scales = scales[visible_indices]
    visible_rotations = rotations[visible_indices]
    visible_opacities = opacities[visible_indices]

    cov3d = compute_covariance_3d(visible_scales, visible_rotations)

    tan_fov_x = np.tan(fov_x * 0.5)
    tan_fov_y = np.tan(fov_y * 0.5)

    points_2d = project_points_to_screen(visible_positions, view_matrix, proj_matrix, width, height)

    depths = points_2d[:, 2]
    sorted_indices = compute_gaussian_depth_sort(depths)

    output_image = torch.zeros(height, width, 3, device=positions.device)
    output_depth = torch.zeros(height, width, device=positions.device)

    tiles_x, tiles_y = compute_tiles(width, height, tile_size=16)

    for tile_y in range(tiles_y):
        for tile_x in range(tiles_x):
            min_x, min_y, max_x, max_y = get_tile_bounds(tile_x, tile_y, 16, width, height)

            for py in range(min_y, max_y):
                for px in range(min_x, max_x):
                    pixel_color = torch.zeros(3, device=positions.device)
                    transmittance = 1.0

                    for idx in sorted_indices:
                        if transmittance < 0.001:
                            break

                        pos_2d = points_2d[idx, :2]

                        cov_2d = compute_cov2d_from_cov3d(
                            visible_positions[idx],
                            cov3d[idx],
                            view_matrix,
                            proj_matrix,
                            tan_fov_x,
                            tan_fov_y
                        )

                        extent = compute_extent_from_cov2d(cov_2d)

                        pixel_pos = torch.tensor([px, py], dtype=torch.float32, device=positions.device)

                        if torch.abs(pixel_pos[0] - pos_2d[0]) > extent or torch.abs(pixel_pos[1] - pos_2d[1]) > extent:
                            continue

                        inv_cov_2d = compute_inverse_cov2d(cov_2d)

                        gaussian_val = eval_gaussian_2d(pixel_pos, pos_2d, inv_cov_2d)

                        alpha = torch.clamp(gaussian_val * visible_opacities[idx, 0], 0.0, 0.99)

                        if alpha < 1.0 / 255.0:
                            continue

                        color = visible_colors[idx]

                        if sh_coeffs is not None and sh_degree > 0:
                            view_dir = torch.nn.functional.normalize(
                                visible_positions[idx] - view_matrix[:3, 3], dim=0
                            )
                            color = compute_color_from_sh(sh_coeffs[visible_indices[idx]], view_dir, sh_degree)

                        pixel_color += transmittance * alpha * color
                        transmittance *= (1.0 - alpha)

                        if transmittance < 0.001:
                            break

                    output_image[py, px] = pixel_color + transmittance * background

    return output_image

def render_depth_map(positions, scales, rotations, opacities, view_matrix, proj_matrix,
                     width, height, fov_x, fov_y):
    """Render depth map from Gaussians"""
    from gaussian_splatting.core.gaussian import compute_covariance_3d
    from gaussian_splatting.core.math_utils import (
        project_points_to_screen, compute_cov2d_from_cov3d, compute_extent_from_cov2d,
        compute_inverse_cov2d, eval_gaussian_2d, frustum_culling, compute_gaussian_depth_sort
    )

    inside_frustum = frustum_culling(positions, view_matrix, proj_matrix)
    visible_indices = torch.where(inside_frustum)[0]

    if len(visible_indices) == 0:
        return torch.zeros(height, width, device=positions.device)

    visible_positions = positions[visible_indices]
    visible_scales = scales[visible_indices]
    visible_rotations = rotations[visible_indices]
    visible_opacities = opacities[visible_indices]

    cov3d = compute_covariance_3d(visible_scales, visible_rotations)

    tan_fov_x = np.tan(fov_x * 0.5)
    tan_fov_y = np.tan(fov_y * 0.5)

    points_2d = project_points_to_screen(visible_positions, view_matrix, proj_matrix, width, height)

    depths = points_2d[:, 2]
    sorted_indices = compute_gaussian_depth_sort(depths)

    output_depth = torch.zeros(height, width, device=positions.device)

    for py in range(height):
        for px in range(width):
            accumulated_depth = 0.0
            total_weight = 0.0

            for idx in sorted_indices:
                pos_2d = points_2d[idx, :2]

                cov_2d = compute_cov2d_from_cov3d(
                    visible_positions[idx],
                    cov3d[idx],
                    view_matrix,
                    proj_matrix,
                    tan_fov_x,
                    tan_fov_y
                )

                extent = compute_extent_from_cov2d(cov_2d)

                pixel_pos = torch.tensor([px, py], dtype=torch.float32, device=positions.device)

                if torch.abs(pixel_pos[0] - pos_2d[0]) > extent or torch.abs(pixel_pos[1] - pos_2d[1]) > extent:
                    continue

                inv_cov_2d = compute_inverse_cov2d(cov_2d)

                gaussian_val = eval_gaussian_2d(pixel_pos, pos_2d, inv_cov_2d)

                alpha = torch.clamp(gaussian_val * visible_opacities[idx, 0], 0.0, 0.99)

                if alpha < 1.0 / 255.0:
                    continue

                weight = alpha
                accumulated_depth += weight * depths[idx]
                total_weight += weight

            if total_weight > 0:
                output_depth[py, px] = accumulated_depth / total_weight

    return output_depth

def render_alpha_map(positions, scales, rotations, opacities, view_matrix, proj_matrix,
                     width, height, fov_x, fov_y):
    """Render alpha/opacity map from Gaussians"""
    from gaussian_splatting.core.gaussian import compute_covariance_3d
    from gaussian_splatting.core.math_utils import (
        project_points_to_screen, compute_cov2d_from_cov3d, compute_extent_from_cov2d,
        compute_inverse_cov2d, eval_gaussian_2d, frustum_culling, compute_gaussian_depth_sort
    )

    inside_frustum = frustum_culling(positions, view_matrix, proj_matrix)
    visible_indices = torch.where(inside_frustum)[0]

    if len(visible_indices) == 0:
        return torch.zeros(height, width, device=positions.device)

    visible_positions = positions[visible_indices]
    visible_scales = scales[visible_indices]
    visible_rotations = rotations[visible_indices]
    visible_opacities = opacities[visible_indices]

    cov3d = compute_covariance_3d(visible_scales, visible_rotations)

    tan_fov_x = np.tan(fov_x * 0.5)
    tan_fov_y = np.tan(fov_y * 0.5)

    points_2d = project_points_to_screen(visible_positions, view_matrix, proj_matrix, width, height)

    depths = points_2d[:, 2]
    sorted_indices = compute_gaussian_depth_sort(depths)

    output_alpha = torch.zeros(height, width, device=positions.device)

    for py in range(height):
        for px in range(width):
            transmittance = 1.0

            for idx in sorted_indices:
                if transmittance < 0.001:
                    break

                pos_2d = points_2d[idx, :2]

                cov_2d = compute_cov2d_from_cov3d(
                    visible_positions[idx],
                    cov3d[idx],
                    view_matrix,
                    proj_matrix,
                    tan_fov_x,
                    tan_fov_y
                )

                extent = compute_extent_from_cov2d(cov_2d)

                pixel_pos = torch.tensor([px, py], dtype=torch.float32, device=positions.device)

                if torch.abs(pixel_pos[0] - pos_2d[0]) > extent or torch.abs(pixel_pos[1] - pos_2d[1]) > extent:
                    continue

                inv_cov_2d = compute_inverse_cov2d(cov_2d)

                gaussian_val = eval_gaussian_2d(pixel_pos, pos_2d, inv_cov_2d)

                alpha = torch.clamp(gaussian_val * visible_opacities[idx, 0], 0.0, 0.99)

                if alpha < 1.0 / 255.0:
                    continue

                transmittance *= (1.0 - alpha)

            output_alpha[py, px] = 1.0 - transmittance

    return output_alpha

import torch
import numpy as np

def compute_gradient_magnitudes(positions_grad):
    """Compute magnitude of position gradients"""
    return torch.norm(positions_grad, dim=1)

def identify_split_candidates(positions, scales, grad_magnitudes, grad_threshold=0.0002, scale_threshold=None):
    """Identify Gaussians that need splitting"""
    high_gradient = grad_magnitudes > grad_threshold

    if scale_threshold is not None:
        max_scale = scales.max(dim=1)[0]
        large_scale = max_scale > scale_threshold
        candidates = high_gradient & large_scale
    else:
        candidates = high_gradient

    return candidates

def identify_clone_candidates(positions, scales, grad_magnitudes, grad_threshold=0.0002, scale_threshold=None):
    """Identify Gaussians that need cloning"""
    high_gradient = grad_magnitudes > grad_threshold

    if scale_threshold is not None:
        max_scale = scales.max(dim=1)[0]
        small_scale = max_scale <= scale_threshold
        candidates = high_gradient & small_scale
    else:
        candidates = high_gradient

    return candidates

def split_gaussians(positions, colors, scales, rotations, opacities, split_mask, sh_coeffs=None):
    """Split large Gaussians into smaller ones"""
    n_splits = split_mask.sum().item()

    if n_splits == 0:
        return positions, colors, scales, rotations, opacities, sh_coeffs

    split_positions = positions[split_mask]
    split_colors = colors[split_mask]
    split_scales = scales[split_mask]
    split_rotations = rotations[split_mask]
    split_opacities = opacities[split_mask]

    new_scales = split_scales / 1.6

    samples_per_gaussian = 2

    new_positions_list = []
    new_colors_list = []
    new_scales_list = []
    new_rotations_list = []
    new_opacities_list = []
    new_sh_list = [] if sh_coeffs is not None else None

    from gaussian_splatting.core.gaussian import build_scaling_rotation

    for i in range(n_splits):
        M = build_scaling_rotation(split_scales[i:i+1], split_rotations[i:i+1])[0]

        for _ in range(samples_per_gaussian):
            sample = torch.randn(3, device=positions.device) * split_scales[i]

            offset = torch.matmul(M, sample)

            new_pos = split_positions[i] + offset

            new_positions_list.append(new_pos)
            new_colors_list.append(split_colors[i])
            new_scales_list.append(new_scales[i])
            new_rotations_list.append(split_rotations[i])
            new_opacities_list.append(split_opacities[i])

            if sh_coeffs is not None:
                new_sh_list.append(sh_coeffs[split_mask][i])

    new_positions = torch.stack(new_positions_list)
    new_colors = torch.stack(new_colors_list)
    new_scales = torch.stack(new_scales_list)
    new_rotations = torch.stack(new_rotations_list)
    new_opacities = torch.stack(new_opacities_list)

    keep_mask = ~split_mask

    all_positions = torch.cat([positions[keep_mask], new_positions], dim=0)
    all_colors = torch.cat([colors[keep_mask], new_colors], dim=0)
    all_scales = torch.cat([scales[keep_mask], new_scales], dim=0)
    all_rotations = torch.cat([rotations[keep_mask], new_rotations], dim=0)
    all_opacities = torch.cat([opacities[keep_mask], new_opacities], dim=0)

    if sh_coeffs is not None:
        new_sh = torch.stack(new_sh_list)
        all_sh = torch.cat([sh_coeffs[keep_mask], new_sh], dim=0)
    else:
        all_sh = None

    return all_positions, all_colors, all_scales, all_rotations, all_opacities, all_sh

def clone_gaussians(positions, colors, scales, rotations, opacities, clone_mask, sh_coeffs=None):
    """Clone Gaussians for under-reconstructed areas"""
    n_clones = clone_mask.sum().item()

    if n_clones == 0:
        return positions, colors, scales, rotations, opacities, sh_coeffs

    clone_positions = positions[clone_mask]
    clone_colors = colors[clone_mask]
    clone_scales = scales[clone_mask]
    clone_rotations = rotations[clone_mask]
    clone_opacities = opacities[clone_mask]

    all_positions = torch.cat([positions, clone_positions], dim=0)
    all_colors = torch.cat([colors, clone_colors], dim=0)
    all_scales = torch.cat([scales, clone_scales], dim=0)
    all_rotations = torch.cat([rotations, clone_rotations], dim=0)
    all_opacities = torch.cat([opacities, clone_opacities], dim=0)

    if sh_coeffs is not None:
        clone_sh = sh_coeffs[clone_mask]
        all_sh = torch.cat([sh_coeffs, clone_sh], dim=0)
    else:
        all_sh = None

    return all_positions, all_colors, all_scales, all_rotations, all_opacities, all_sh

def densify_and_split(positions, colors, scales, rotations, opacities, positions_grad,
                      grad_threshold=0.0002, scale_threshold=0.01, sh_coeffs=None):
    """Perform densification by splitting large Gaussians"""
    grad_magnitudes = compute_gradient_magnitudes(positions_grad)

    split_mask = identify_split_candidates(
        positions, scales, grad_magnitudes,
        grad_threshold, scale_threshold
    )

    return split_gaussians(positions, colors, scales, rotations, opacities, split_mask, sh_coeffs)

def densify_and_clone(positions, colors, scales, rotations, opacities, positions_grad,
                      grad_threshold=0.0002, scale_threshold=0.01, sh_coeffs=None):
    """Perform densification by cloning small Gaussians"""
    grad_magnitudes = compute_gradient_magnitudes(positions_grad)

    clone_mask = identify_clone_candidates(
        positions, scales, grad_magnitudes,
        grad_threshold, scale_threshold
    )

    return clone_gaussians(positions, colors, scales, rotations, opacities, clone_mask, sh_coeffs)

def adaptive_density_control(positions, colors, scales, rotations, opacities, positions_grad,
                             grad_threshold=0.0002, scale_threshold=0.01, sh_coeffs=None):
    """Adaptive density control: split large + clone small"""
    grad_magnitudes = compute_gradient_magnitudes(positions_grad)

    split_mask = identify_split_candidates(
        positions, scales, grad_magnitudes,
        grad_threshold, scale_threshold
    )

    clone_mask = identify_clone_candidates(
        positions, scales, grad_magnitudes,
        grad_threshold, scale_threshold
    )

    clone_mask = clone_mask & ~split_mask

    positions, colors, scales, rotations, opacities, sh_coeffs = split_gaussians(
        positions, colors, scales, rotations, opacities, split_mask, sh_coeffs
    )

    positions, colors, scales, rotations, opacities, sh_coeffs = clone_gaussians(
        positions, colors, scales, rotations, opacities, clone_mask, sh_coeffs
    )

    return positions, colors, scales, rotations, opacities, sh_coeffs

def reset_gradient_accumulators(positions, scales, rotations, opacities, colors):
    """Reset gradient accumulators after densification"""
    if hasattr(positions, '_grad_accumulator'):
        delattr(positions, '_grad_accumulator')
    if hasattr(scales, '_grad_accumulator'):
        delattr(scales, '_grad_accumulator')
    if hasattr(rotations, '_grad_accumulator'):
        delattr(rotations, '_grad_accumulator')
    if hasattr(opacities, '_grad_accumulator'):
        delattr(opacities, '_grad_accumulator')
    if hasattr(colors, '_grad_accumulator'):
        delattr(colors, '_grad_accumulator')

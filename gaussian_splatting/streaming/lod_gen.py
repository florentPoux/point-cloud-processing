import numpy as np
import torch

def generate_lod_levels(positions, colors, scales, rotations, opacities, num_levels=4):
    """Generate multiple LOD levels for Gaussian splats"""
    lod_levels = []

    current_positions = positions
    current_colors = colors
    current_scales = scales
    current_rotations = rotations
    current_opacities = opacities

    for level in range(num_levels):
        lod_data = {
            'level': level,
            'positions': current_positions,
            'colors': current_colors,
            'scales': current_scales,
            'rotations': current_rotations,
            'opacities': current_opacities,
            'n_gaussians': len(current_positions)
        }

        lod_levels.append(lod_data)

        if level < num_levels - 1:
            current_positions, current_colors, current_scales, current_rotations, current_opacities = downsample_gaussians(
                current_positions, current_colors, current_scales, current_rotations, current_opacities, ratio=0.5
            )

    return lod_levels

def downsample_gaussians(positions, colors, scales, rotations, opacities, ratio=0.5):
    """Downsample Gaussians for lower LOD level"""
    n_gaussians = len(positions)
    n_keep = int(n_gaussians * ratio)

    importance_scores = compute_importance_scores(positions, scales, opacities)

    sorted_indices = torch.argsort(importance_scores, descending=True)
    keep_indices = sorted_indices[:n_keep]

    downsampled_positions = positions[keep_indices]
    downsampled_colors = colors[keep_indices]
    downsampled_scales = scales[keep_indices]
    downsampled_rotations = rotations[keep_indices]
    downsampled_opacities = opacities[keep_indices]

    return downsampled_positions, downsampled_colors, downsampled_scales, downsampled_rotations, downsampled_opacities

def compute_importance_scores(positions, scales, opacities):
    """Compute importance score for each Gaussian"""
    scale_volume = scales.prod(dim=1)

    importance = opacities.squeeze() * scale_volume

    return importance

def merge_nearby_gaussians(positions, colors, scales, rotations, opacities, merge_threshold=0.01):
    """Merge Gaussians that are close together"""
    from scipy.spatial import cKDTree

    if isinstance(positions, torch.Tensor):
        positions_np = positions.cpu().numpy()
    else:
        positions_np = positions

    tree = cKDTree(positions_np)

    pairs = tree.query_pairs(merge_threshold)

    merged_indices = set()
    keep_indices = []

    for i in range(len(positions)):
        if i in merged_indices:
            continue

        neighbors = [j for (a, b) in pairs if a == i for j in [b] if j not in merged_indices]
        neighbors = [j for (a, b) in pairs if b == i for j in [a] if j not in merged_indices] + neighbors

        if neighbors:
            group = [i] + neighbors

            merged_indices.update(group)

            merged_pos = positions[group].mean(dim=0)
            merged_color = colors[group].mean(dim=0)
            merged_scale = scales[group].mean(dim=0)
            merged_rotation = rotations[i]
            merged_opacity = opacities[group].mean()

            if isinstance(positions, torch.Tensor):
                keep_indices.append({
                    'position': merged_pos,
                    'color': merged_color,
                    'scale': merged_scale,
                    'rotation': merged_rotation,
                    'opacity': merged_opacity
                })
        else:
            keep_indices.append({
                'position': positions[i],
                'color': colors[i],
                'scale': scales[i],
                'rotation': rotations[i],
                'opacity': opacities[i]
            })

    if isinstance(positions, torch.Tensor):
        merged_positions = torch.stack([item['position'] for item in keep_indices])
        merged_colors = torch.stack([item['color'] for item in keep_indices])
        merged_scales = torch.stack([item['scale'] for item in keep_indices])
        merged_rotations = torch.stack([item['rotation'] for item in keep_indices])
        merged_opacities = torch.stack([item['opacity'] for item in keep_indices])
    else:
        merged_positions = np.stack([item['position'] for item in keep_indices])
        merged_colors = np.stack([item['color'] for item in keep_indices])
        merged_scales = np.stack([item['scale'] for item in keep_indices])
        merged_rotations = np.stack([item['rotation'] for item in keep_indices])
        merged_opacities = np.stack([item['opacity'] for item in keep_indices])

    return merged_positions, merged_colors, merged_scales, merged_rotations, merged_opacities

def adaptive_lod_selection(camera_position, gaussian_positions, lod_levels, distance_thresholds=None):
    """Select LOD level based on distance from camera"""
    if distance_thresholds is None:
        distance_thresholds = [5.0, 15.0, 30.0]

    distances = np.linalg.norm(gaussian_positions - camera_position, axis=1)

    lod_assignments = np.zeros(len(gaussian_positions), dtype=np.int32)

    for i, threshold in enumerate(distance_thresholds):
        lod_assignments[distances > threshold] = i + 1

    lod_assignments = np.clip(lod_assignments, 0, len(lod_levels) - 1)

    return lod_assignments

def save_lod_hierarchy(lod_levels, output_dir):
    """Save LOD hierarchy to disk"""
    import os
    os.makedirs(output_dir, exist_ok=True)

    for lod in lod_levels:
        filename = os.path.join(output_dir, f'lod_{lod["level"]}.npz')

        np.savez_compressed(
            filename,
            positions=lod['positions'].cpu().numpy() if isinstance(lod['positions'], torch.Tensor) else lod['positions'],
            colors=lod['colors'].cpu().numpy() if isinstance(lod['colors'], torch.Tensor) else lod['colors'],
            scales=lod['scales'].cpu().numpy() if isinstance(lod['scales'], torch.Tensor) else lod['scales'],
            rotations=lod['rotations'].cpu().numpy() if isinstance(lod['rotations'], torch.Tensor) else lod['rotations'],
            opacities=lod['opacities'].cpu().numpy() if isinstance(lod['opacities'], torch.Tensor) else lod['opacities']
        )

#!/usr/bin/env python3
"""
Complete example: Train Gaussian Splats from COLMAP reconstruction
"""

import torch
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gaussian_splatting.data.colmap_loader import load_colmap_dataset
from gaussian_splatting.core.gaussian import initialize_from_points, save_gaussian_model
from gaussian_splatting.train.trainer import create_optimizer, train_step
from gaussian_splatting.train.densification import adaptive_density_control
from gaussian_splatting.train.pruner import comprehensive_pruning
from gaussian_splatting.train.loss import psnr
from gaussian_splatting.streaming.lod_gen import generate_lod_levels, save_lod_hierarchy
from gaussian_splatting.streaming.serializer import save_streaming_format
from gaussian_splatting.deploy.asset_packager import package_for_marketplace

def main(colmap_dir, output_dir, num_iterations=10000):
    """Complete training pipeline from COLMAP to deployment"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    print("Loading COLMAP dataset...")
    dataset = load_colmap_dataset(colmap_dir)

    print(f"Loaded {len(dataset['points'])} initial points")
    print(f"Loaded {len(dataset['views'])} views")

    print("Initializing Gaussians...")
    positions, colors, scales, rotations, opacities = initialize_from_points(
        dataset['points'],
        dataset['colors'],
        initial_scale=0.01
    )

    positions = positions.to(device).requires_grad_(True)
    colors = colors.to(device).requires_grad_(True)
    scales = scales.to(device).requires_grad_(True)
    rotations = rotations.to(device).requires_grad_(True)
    opacities = opacities.to(device).requires_grad_(True)

    print(f"Initialized {len(positions)} Gaussians")

    print("Creating optimizer...")
    optimizer = create_optimizer(positions, colors, scales, rotations, opacities)

    split_views = int(len(dataset['views']) * 0.9)
    train_views = dataset['views'][:split_views]
    test_views = dataset['views'][split_views:]

    print(f"Train views: {len(train_views)}, Test views: {len(test_views)}")

    print(f"\nStarting training for {num_iterations} iterations...")

    for iteration in range(num_iterations):
        view_idx = np.random.randint(0, len(train_views))
        view_data = train_views[view_idx]

        view_matrix = torch.from_numpy(view_data['view_matrix']).float().to(device)
        width, height = view_data['width'], view_data['height']
        fov_x, fov_y = view_data['fov_x'], view_data['fov_y']

        proj_matrix = compute_proj_matrix(fov_x, fov_y, 0.01, 100.0).to(device)

        target_image = load_image(colmap_dir, view_data['name'], width, height).to(device)

        loss, loss_dict, rendered = train_step(
            positions, colors, scales, rotations, opacities,
            view_matrix, proj_matrix, target_image,
            width, height, fov_x, fov_y, optimizer
        )

        if iteration % 100 == 0:
            print(f"Iter {iteration}/{num_iterations}: Loss={loss:.4f}, "
                  f"L1={loss_dict['l1']:.4f}, SSIM={loss_dict['ssim']:.4f}, "
                  f"N={len(positions)}")

        if iteration >= 500 and iteration <= 15000 and iteration % 100 == 0:
            if positions.grad is not None:
                (positions, colors, scales, rotations, opacities, _) = adaptive_density_control(
                    positions.detach(), colors.detach(), scales.detach(),
                    rotations.detach(), opacities.detach(),
                    positions.grad, grad_threshold=0.0002, scale_threshold=0.01
                )

                positions = positions.to(device).requires_grad_(True)
                colors = colors.to(device).requires_grad_(True)
                scales = scales.to(device).requires_grad_(True)
                rotations = rotations.to(device).requires_grad_(True)
                opacities = opacities.to(device).requires_grad_(True)

                optimizer = create_optimizer(positions, colors, scales, rotations, opacities)

        if iteration % 500 == 0 and iteration > 0:
            (positions, colors, scales, rotations, opacities, _) = comprehensive_pruning(
                positions.detach(), colors.detach(), scales.detach(),
                rotations.detach(), opacities.detach(),
                opacity_threshold=0.005, scale_threshold=0.5,
                view_matrix=view_matrix
            )

            positions = positions.to(device).requires_grad_(True)
            colors = colors.to(device).requires_grad_(True)
            scales = scales.to(device).requires_grad_(True)
            rotations = rotations.to(device).requires_grad_(True)
            opacities = opacities.to(device).requires_grad_(True)

            optimizer = create_optimizer(positions, colors, scales, rotations, opacities)

        if iteration % 500 == 0 and len(test_views) > 0:
            with torch.no_grad():
                test_view = test_views[0]
                test_matrix = torch.from_numpy(test_view['view_matrix']).float().to(device)
                test_proj = compute_proj_matrix(test_view['fov_x'], test_view['fov_y'], 0.01, 100.0).to(device)
                test_target = load_image(colmap_dir, test_view['name'], test_view['width'], test_view['height']).to(device)

                from gaussian_splatting.core.rasterizer import rasterize_gaussians
                test_rendered = rasterize_gaussians(
                    positions, colors, scales, rotations, opacities,
                    test_matrix, test_proj,
                    test_view['width'], test_view['height'],
                    test_view['fov_x'], test_view['fov_y']
                )

                test_psnr = psnr(test_rendered, test_target)
                print(f"  Test PSNR: {test_psnr:.2f} dB")

    print("\nTraining complete!")
    print(f"Final model: {len(positions)} Gaussians")

    print("Saving model...")
    model_path = output_path / 'final_model.npz'
    save_gaussian_model(model_path, positions, colors, scales, rotations, opacities)

    print("Generating LOD hierarchy...")
    lod_levels = generate_lod_levels(positions, colors, scales, rotations, opacities, num_levels=4)
    lod_dir = output_path / 'lods'
    save_lod_hierarchy(lod_levels, lod_dir)

    print("Creating streaming format...")
    streaming_dir = output_path / 'streaming'
    save_streaming_format(positions, colors, scales, rotations, opacities, streaming_dir, chunk_size=10000)

    print("Packaging for marketplace...")
    metadata = {
        'title': 'COLMAP Reconstruction',
        'description': 'Gaussian Splatting model trained from COLMAP',
        'author': '3D Geodata Academy',
        'tags': ['gaussian-splatting', '3dgs', 'novel-view-synthesis']
    }

    package_for_marketplace(model_path, metadata, output_path / 'marketplace')

    print(f"\nAll outputs saved to: {output_path}")
    print("  - final_model.npz: Trained Gaussian splats")
    print("  - lods/: LOD hierarchy")
    print("  - streaming/: Streaming chunks")
    print("  - marketplace/: Distribution package")

def compute_proj_matrix(fov_x, fov_y, znear, zfar):
    """Compute projection matrix"""
    from gaussian_splatting.core.math_utils import compute_projection_matrix
    return compute_projection_matrix(fov_x, fov_y, znear, zfar)

def load_image(colmap_dir, image_name, width, height):
    """Load training image"""
    import os
    from PIL import Image

    image_path = os.path.join(colmap_dir, '..', 'images', image_name)

    if not os.path.exists(image_path):
        return torch.rand(height, width, 3)

    img = Image.open(image_path).convert('RGB')
    img = img.resize((width, height))

    img_array = np.array(img) / 255.0

    return torch.from_numpy(img_array).float()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train Gaussian Splats from COLMAP')
    parser.add_argument('colmap_dir', type=str, help='Path to COLMAP sparse directory')
    parser.add_argument('--output', type=str, default='./output', help='Output directory')
    parser.add_argument('--iterations', type=int, default=10000, help='Number of training iterations')

    args = parser.parse_args()

    main(args.colmap_dir, args.output, args.iterations)

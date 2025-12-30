import torch
import numpy as np
from pathlib import Path

def create_optimizer(positions, colors, scales, rotations, opacities, sh_coeffs=None, learning_rates=None):
    """Create optimizer for Gaussian parameters"""
    if learning_rates is None:
        learning_rates = {
            'positions': 1.6e-4,
            'colors': 2.5e-3,
            'scales': 5e-3,
            'rotations': 1e-3,
            'opacities': 5e-2,
            'sh': 2.5e-3
        }

    param_groups = [
        {'params': [positions], 'lr': learning_rates['positions'], 'name': 'positions'},
        {'params': [colors], 'lr': learning_rates['colors'], 'name': 'colors'},
        {'params': [scales], 'lr': learning_rates['scales'], 'name': 'scales'},
        {'params': [rotations], 'lr': learning_rates['rotations'], 'name': 'rotations'},
        {'params': [opacities], 'lr': learning_rates['opacities'], 'name': 'opacities'}
    ]

    if sh_coeffs is not None:
        param_groups.append({'params': [sh_coeffs], 'lr': learning_rates['sh'], 'name': 'sh'})

    optimizer = torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)

    return optimizer

def train_step(positions, colors, scales, rotations, opacities,
              view_matrix, proj_matrix, target_image,
              width, height, fov_x, fov_y,
              optimizer, sh_coeffs=None, background=None):
    """Single training step"""
    from gaussian_splatting.core.rasterizer import rasterize_gaussians
    from gaussian_splatting.train.loss import compute_full_loss

    optimizer.zero_grad()

    rendered_image = rasterize_gaussians(
        positions, colors, scales, rotations, opacities,
        view_matrix, proj_matrix, width, height, fov_x, fov_y,
        sh_degree=0, sh_coeffs=sh_coeffs, background=background
    )

    loss, loss_dict = compute_full_loss(
        rendered_image, target_image, opacities, scales
    )

    loss.backward()

    optimizer.step()

    return loss.item(), loss_dict, rendered_image

def train_epoch(positions, colors, scales, rotations, opacities,
               train_views, optimizer, epoch, sh_coeffs=None, background=None):
    """Train for one epoch over all views"""
    epoch_losses = []
    epoch_loss_dicts = []

    for view_idx, view_data in enumerate(train_views):
        view_matrix = view_data['view_matrix']
        proj_matrix = view_data['proj_matrix']
        target_image = view_data['image']
        width = view_data['width']
        height = view_data['height']
        fov_x = view_data['fov_x']
        fov_y = view_data['fov_y']

        loss, loss_dict, rendered = train_step(
            positions, colors, scales, rotations, opacities,
            view_matrix, proj_matrix, target_image,
            width, height, fov_x, fov_y,
            optimizer, sh_coeffs, background
        )

        epoch_losses.append(loss)
        epoch_loss_dicts.append(loss_dict)

    avg_loss = np.mean(epoch_losses)

    avg_loss_dict = {}
    for key in epoch_loss_dicts[0].keys():
        avg_loss_dict[key] = np.mean([d[key] for d in epoch_loss_dicts])

    return avg_loss, avg_loss_dict

def densification_step(positions, colors, scales, rotations, opacities,
                      positions_grad, iteration, densify_from_iter=500,
                      densify_until_iter=15000, densification_interval=100,
                      grad_threshold=0.0002, scale_threshold=0.01,
                      sh_coeffs=None):
    """Perform densification if conditions are met"""
    if iteration < densify_from_iter or iteration > densify_until_iter:
        return positions, colors, scales, rotations, opacities, sh_coeffs

    if iteration % densification_interval != 0:
        return positions, colors, scales, rotations, opacities, sh_coeffs

    from gaussian_splatting.train.densification import adaptive_density_control

    positions, colors, scales, rotations, opacities, sh_coeffs = adaptive_density_control(
        positions, colors, scales, rotations, opacities, positions_grad,
        grad_threshold, scale_threshold, sh_coeffs
    )

    return positions, colors, scales, rotations, opacities, sh_coeffs

def pruning_step(positions, colors, scales, rotations, opacities,
                iteration, prune_interval=500, opacity_threshold=0.005,
                scale_threshold=0.5, view_matrix=None, sh_coeffs=None):
    """Perform pruning if conditions are met"""
    if iteration % prune_interval != 0:
        return positions, colors, scales, rotations, opacities, sh_coeffs

    from gaussian_splatting.train.pruner import comprehensive_pruning

    positions, colors, scales, rotations, opacities, sh_coeffs = comprehensive_pruning(
        positions, colors, scales, rotations, opacities,
        opacity_threshold, scale_threshold, view_matrix,
        contribution_threshold=0.01, sh_coeffs=sh_coeffs
    )

    return positions, colors, scales, rotations, opacities, sh_coeffs

def save_checkpoint(filepath, positions, colors, scales, rotations, opacities,
                   optimizer, iteration, sh_coeffs=None):
    """Save training checkpoint"""
    checkpoint = {
        'iteration': iteration,
        'positions': positions.detach().cpu(),
        'colors': colors.detach().cpu(),
        'scales': scales.detach().cpu(),
        'rotations': rotations.detach().cpu(),
        'opacities': opacities.detach().cpu(),
        'optimizer_state': optimizer.state_dict()
    }

    if sh_coeffs is not None:
        checkpoint['sh_coeffs'] = sh_coeffs.detach().cpu()

    torch.save(checkpoint, filepath)

def load_checkpoint(filepath):
    """Load training checkpoint"""
    checkpoint = torch.load(filepath)

    positions = checkpoint['positions']
    colors = checkpoint['colors']
    scales = checkpoint['scales']
    rotations = checkpoint['rotations']
    opacities = checkpoint['opacities']
    iteration = checkpoint['iteration']
    optimizer_state = checkpoint['optimizer_state']

    sh_coeffs = checkpoint.get('sh_coeffs', None)

    return positions, colors, scales, rotations, opacities, iteration, optimizer_state, sh_coeffs

def full_training_loop(initial_positions, initial_colors, train_views, test_views=None,
                      num_iterations=30000, save_dir='./checkpoints', device='cuda'):
    """Complete training loop"""
    from gaussian_splatting.core.gaussian import initialize_from_points
    from gaussian_splatting.train.loss import psnr

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    positions, colors, scales, rotations, opacities = initialize_from_points(
        initial_positions, initial_colors, initial_scale=0.01
    )

    positions = positions.to(device).requires_grad_(True)
    colors = colors.to(device).requires_grad_(True)
    scales = scales.to(device).requires_grad_(True)
    rotations = rotations.to(device).requires_grad_(True)
    opacities = opacities.to(device).requires_grad_(True)

    optimizer = create_optimizer(positions, colors, scales, rotations, opacities)

    for iteration in range(num_iterations):
        view_idx = np.random.randint(0, len(train_views))
        view_data = train_views[view_idx]

        loss, loss_dict, rendered = train_step(
            positions, colors, scales, rotations, opacities,
            view_data['view_matrix'], view_data['proj_matrix'],
            view_data['image'], view_data['width'], view_data['height'],
            view_data['fov_x'], view_data['fov_y'], optimizer
        )

        if iteration % 100 == 0:
            print(f"Iteration {iteration}/{num_iterations}, Loss: {loss:.4f}, "
                  f"L1: {loss_dict['l1']:.4f}, SSIM: {loss_dict['ssim']:.4f}, "
                  f"N_Gaussians: {len(positions)}")

        if positions.grad is not None:
            positions, colors, scales, rotations, opacities, _ = densification_step(
                positions, colors, scales, rotations, opacities, positions.grad, iteration
            )

        positions, colors, scales, rotations, opacities, _ = pruning_step(
            positions, colors, scales, rotations, opacities, iteration,
            view_matrix=view_data['view_matrix']
        )

        if iteration % 1000 == 0 and iteration > 0:
            checkpoint_path = Path(save_dir) / f'checkpoint_{iteration:06d}.pth'
            save_checkpoint(checkpoint_path, positions, colors, scales, rotations, opacities, optimizer, iteration)

        if test_views is not None and iteration % 500 == 0:
            with torch.no_grad():
                test_view = test_views[0]
                from gaussian_splatting.core.rasterizer import rasterize_gaussians

                test_rendered = rasterize_gaussians(
                    positions, colors, scales, rotations, opacities,
                    test_view['view_matrix'], test_view['proj_matrix'],
                    test_view['width'], test_view['height'],
                    test_view['fov_x'], test_view['fov_y']
                )

                test_psnr = psnr(test_rendered, test_view['image'])
                print(f"  Test PSNR: {test_psnr:.2f} dB")

    final_path = Path(save_dir) / 'final_model.pth'
    save_checkpoint(final_path, positions, colors, scales, rotations, opacities, optimizer, num_iterations)

    print("Training complete!")

    return positions, colors, scales, rotations, opacities

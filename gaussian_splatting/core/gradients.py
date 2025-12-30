import torch

def compute_position_gradients(rendered_image, target_image, gaussians_contrib, positions):
    """Compute gradients with respect to Gaussian positions"""
    image_loss_grad = 2.0 * (rendered_image - target_image)

    position_grads = torch.zeros_like(positions)

    for i in range(len(positions)):
        contrib = gaussians_contrib[i]

        if len(contrib) == 0:
            continue

        pixel_grads = image_loss_grad[contrib[:, 0], contrib[:, 1]]

        position_grads[i] = pixel_grads.mean(dim=0)

    return position_grads

def compute_scale_gradients(rendered_image, target_image, gaussians_contrib, scales):
    """Compute gradients with respect to Gaussian scales"""
    image_loss_grad = 2.0 * (rendered_image - target_image)

    scale_grads = torch.zeros_like(scales)

    for i in range(len(scales)):
        contrib = gaussians_contrib[i]

        if len(contrib) == 0:
            continue

        pixel_grads = image_loss_grad[contrib[:, 0], contrib[:, 1]]

        scale_grads[i] = pixel_grads.mean(dim=0) * scales[i]

    return scale_grads

def compute_rotation_gradients(rendered_image, target_image, gaussians_contrib, rotations):
    """Compute gradients with respect to Gaussian rotations"""
    image_loss_grad = 2.0 * (rendered_image - target_image)

    rotation_grads = torch.zeros_like(rotations)

    for i in range(len(rotations)):
        contrib = gaussians_contrib[i]

        if len(contrib) == 0:
            continue

        pixel_grads = image_loss_grad[contrib[:, 0], contrib[:, 1]]

        rotation_grads[i] = pixel_grads.mean(dim=0)

    return rotation_grads

def compute_opacity_gradients(rendered_image, target_image, gaussians_contrib, opacities):
    """Compute gradients with respect to Gaussian opacities"""
    image_loss_grad = 2.0 * (rendered_image - target_image)

    opacity_grads = torch.zeros_like(opacities)

    for i in range(len(opacities)):
        contrib = gaussians_contrib[i]

        if len(contrib) == 0:
            continue

        pixel_grads = image_loss_grad[contrib[:, 0], contrib[:, 1]]

        opacity_grads[i] = pixel_grads.mean()

    return opacity_grads

def compute_color_gradients(rendered_image, target_image, gaussians_contrib, colors):
    """Compute gradients with respect to Gaussian colors"""
    image_loss_grad = 2.0 * (rendered_image - target_image)

    color_grads = torch.zeros_like(colors)

    for i in range(len(colors)):
        contrib = gaussians_contrib[i]

        if len(contrib) == 0:
            continue

        pixel_grads = image_loss_grad[contrib[:, 0], contrib[:, 1]]

        color_grads[i] = pixel_grads.mean(dim=0)

    return color_grads

def accumulate_gradients(gaussians, grad_dict):
    """Accumulate gradients for all Gaussian parameters"""
    if gaussians.positions.grad is None:
        gaussians.positions.grad = torch.zeros_like(gaussians.positions)
    gaussians.positions.grad += grad_dict['positions']

    if gaussians.scales.grad is None:
        gaussians.scales.grad = torch.zeros_like(gaussians.scales)
    gaussians.scales.grad += grad_dict['scales']

    if gaussians.rotations.grad is None:
        gaussians.rotations.grad = torch.zeros_like(gaussians.rotations)
    gaussians.rotations.grad += grad_dict['rotations']

    if gaussians.opacities.grad is None:
        gaussians.opacities.grad = torch.zeros_like(gaussians.opacities)
    gaussians.opacities.grad += grad_dict['opacities']

    if gaussians.colors.grad is None:
        gaussians.colors.grad = torch.zeros_like(gaussians.colors)
    gaussians.colors.grad += grad_dict['colors']

def finite_difference_gradient(func, params, param_idx, epsilon=1e-5):
    """Compute gradient using finite differences for verification"""
    original_value = params[param_idx].clone()

    params[param_idx] += epsilon
    loss_plus = func(params)

    params[param_idx] = original_value - epsilon
    loss_minus = func(params)

    params[param_idx] = original_value

    gradient = (loss_plus - loss_minus) / (2.0 * epsilon)

    return gradient

def verify_gradients(analytical_grads, numerical_grads, tolerance=1e-3):
    """Verify analytical gradients against numerical gradients"""
    diff = torch.abs(analytical_grads - numerical_grads)
    max_diff = diff.max()

    relative_error = diff / (torch.abs(numerical_grads) + 1e-8)
    max_relative_error = relative_error.max()

    is_valid = max_relative_error < tolerance

    return {
        'is_valid': is_valid,
        'max_diff': max_diff.item(),
        'max_relative_error': max_relative_error.item()
    }

def clip_gradients(parameters, max_norm=1.0):
    """Clip gradients to prevent exploding gradients"""
    total_norm = 0.0

    for param in parameters:
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2

    total_norm = total_norm ** 0.5

    clip_coef = max_norm / (total_norm + 1e-6)

    if clip_coef < 1:
        for param in parameters:
            if param.grad is not None:
                param.grad.data.mul_(clip_coef)

    return total_norm

def compute_gradient_statistics(parameters):
    """Compute statistics of gradients for monitoring"""
    stats = {}

    for name, param in parameters.items():
        if param.grad is not None:
            grad = param.grad

            stats[name] = {
                'mean': grad.mean().item(),
                'std': grad.std().item(),
                'min': grad.min().item(),
                'max': grad.max().item(),
                'norm': grad.norm().item()
            }

    return stats

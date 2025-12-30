import torch
import torch.nn.functional as F

def l1_loss(rendered, target):
    """L1 loss between rendered and target images"""
    return torch.abs(rendered - target).mean()

def l2_loss(rendered, target):
    """L2 loss between rendered and target images"""
    return F.mse_loss(rendered, target)

def ssim(img1, img2, window_size=11, size_average=True):
    """Structural Similarity Index"""
    def create_window(window_size, channel):
        _1D_window = torch.exp(
            -torch.arange(-(window_size // 2), window_size // 2 + 1, dtype=torch.float32) ** 2 / (2 * 1.5 ** 2)
        )
        _1D_window = _1D_window / _1D_window.sum()
        _2D_window = _1D_window.unsqueeze(1) @ _1D_window.unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    channel = img1.shape[2] if len(img1.shape) == 3 else 1

    if len(img1.shape) == 3:
        img1 = img1.permute(2, 0, 1).unsqueeze(0)
        img2 = img2.permute(2, 0, 1).unsqueeze(0)

    window = create_window(window_size, channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def ssim_loss(rendered, target, window_size=11):
    """SSIM loss (1 - SSIM)"""
    return 1.0 - ssim(rendered, target, window_size)

def combined_loss(rendered, target, lambda_l1=0.8, lambda_ssim=0.2):
    """Combined L1 and SSIM loss"""
    loss_l1 = l1_loss(rendered, target)
    loss_ssim = ssim_loss(rendered, target)

    total_loss = lambda_l1 * loss_l1 + lambda_ssim * loss_ssim

    return total_loss, {'l1': loss_l1.item(), 'ssim': loss_ssim.item()}

def perceptual_loss(rendered, target, vgg_model):
    """Perceptual loss using VGG features"""
    if len(rendered.shape) == 3:
        rendered = rendered.permute(2, 0, 1).unsqueeze(0)
        target = target.permute(2, 0, 1).unsqueeze(0)

    rendered_features = vgg_model(rendered)
    target_features = vgg_model(target)

    loss = 0.0
    for rf, tf in zip(rendered_features, target_features):
        loss += F.mse_loss(rf, tf)

    return loss / len(rendered_features)

def depth_loss(rendered_depth, target_depth, mask=None):
    """Loss for depth supervision"""
    if mask is not None:
        diff = torch.abs(rendered_depth - target_depth) * mask
        return diff.sum() / (mask.sum() + 1e-8)
    else:
        return torch.abs(rendered_depth - target_depth).mean()

def opacity_regularization(opacities, target_opacity=0.1):
    """Regularization to encourage sparsity"""
    return torch.abs(opacities - target_opacity).mean()

def scale_regularization(scales, max_scale=0.1):
    """Regularization to prevent overly large Gaussians"""
    return F.relu(scales - max_scale).mean()

def total_variation_loss(image):
    """Total variation loss for smoothness"""
    if len(image.shape) == 3:
        image = image.permute(2, 0, 1).unsqueeze(0)

    tv_h = torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :]).mean()
    tv_w = torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1]).mean()

    return tv_h + tv_w

def compute_full_loss(rendered, target, opacities, scales,
                     lambda_l1=0.8, lambda_ssim=0.2,
                     lambda_opacity=0.01, lambda_scale=0.01):
    """Compute full training loss with regularization"""
    loss_render, render_components = combined_loss(rendered, target, lambda_l1, lambda_ssim)

    loss_opacity = opacity_regularization(opacities)
    loss_scale = scale_regularization(scales)

    total_loss = (
        loss_render +
        lambda_opacity * loss_opacity +
        lambda_scale * loss_scale
    )

    loss_dict = {
        **render_components,
        'opacity_reg': loss_opacity.item(),
        'scale_reg': loss_scale.item(),
        'total': total_loss.item()
    }

    return total_loss, loss_dict

def psnr(rendered, target):
    """Peak Signal-to-Noise Ratio"""
    mse = F.mse_loss(rendered, target)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def lpips_loss(rendered, target, lpips_model):
    """Learned Perceptual Image Patch Similarity"""
    if len(rendered.shape) == 3:
        rendered = rendered.permute(2, 0, 1).unsqueeze(0)
        target = target.permute(2, 0, 1).unsqueeze(0)

    rendered = rendered * 2.0 - 1.0
    target = target * 2.0 - 1.0

    return lpips_model(rendered, target).mean()

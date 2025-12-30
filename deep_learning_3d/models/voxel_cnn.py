import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3d_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1, batch_norm=True):
    """3D convolutional block"""
    layers = [nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)]

    if batch_norm:
        layers.append(nn.BatchNorm3d(out_channels))

    layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)

def create_voxel_encoder(in_channels=1, base_channels=32):
    """Create 3D CNN encoder for voxel data"""
    encoder = nn.ModuleDict({
        'conv1': conv3d_block(in_channels, base_channels, kernel_size=3, stride=1, padding=1),
        'conv2': conv3d_block(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
        'conv3': conv3d_block(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1),
        'conv4': conv3d_block(base_channels * 4, base_channels * 8, kernel_size=3, stride=2, padding=1),
        'conv5': conv3d_block(base_channels * 8, base_channels * 16, kernel_size=3, stride=2, padding=1),
    })

    return encoder

def voxel_encoder_forward(x, encoder):
    """Forward pass through voxel encoder"""
    features = []

    x1 = encoder['conv1'](x)
    features.append(x1)

    x2 = encoder['conv2'](x1)
    features.append(x2)

    x3 = encoder['conv3'](x2)
    features.append(x3)

    x4 = encoder['conv4'](x3)
    features.append(x4)

    x5 = encoder['conv5'](x4)
    features.append(x5)

    return x5, features

def create_voxel_classifier(in_channels=1, num_classes=10, grid_size=32):
    """Create 3D CNN for voxel classification"""
    encoder = create_voxel_encoder(in_channels, base_channels=32)

    final_grid = grid_size // 16

    fc = nn.Sequential(
        nn.Flatten(),
        nn.Linear(512 * final_grid * final_grid * final_grid, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes)
    )

    return {'encoder': encoder, 'fc': fc}

def voxel_classify(x, model_components):
    """Run voxel classification"""
    features, _ = voxel_encoder_forward(x, model_components['encoder'])

    logits = model_components['fc'](features)

    return logits

def deconv3d_block(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1):
    """3D deconvolutional block"""
    return nn.Sequential(
        nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride, padding, output_padding),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True)
    )

def create_voxel_segmentation(in_channels=1, num_classes=10):
    """Create 3D U-Net for voxel segmentation"""
    encoder = create_voxel_encoder(in_channels, base_channels=32)

    decoder = nn.ModuleDict({
        'up1': deconv3d_block(512, 256),
        'conv1': conv3d_block(512, 256),

        'up2': deconv3d_block(256, 128),
        'conv2': conv3d_block(256, 128),

        'up3': deconv3d_block(128, 64),
        'conv3': conv3d_block(128, 64),

        'up4': deconv3d_block(64, 32),
        'conv4': conv3d_block(64, 32),

        'final': nn.Conv3d(32, num_classes, kernel_size=1)
    })

    return {'encoder': encoder, 'decoder': decoder}

def voxel_segment(x, model_components):
    """Run voxel segmentation with U-Net"""
    encoder = model_components['encoder']
    decoder = model_components['decoder']

    x5, features = voxel_encoder_forward(x, encoder)

    x4, x3, x2, x1, x0 = features[4], features[3], features[2], features[1], features[0]

    d1 = decoder['up1'](x5)
    d1 = torch.cat([d1, x4], dim=1)
    d1 = decoder['conv1'](d1)

    d2 = decoder['up2'](d1)
    d2 = torch.cat([d2, x3], dim=1)
    d2 = decoder['conv2'](d2)

    d3 = decoder['up3'](d2)
    d3 = torch.cat([d3, x2], dim=1)
    d3 = decoder['conv3'](d3)

    d4 = decoder['up4'](d3)
    d4 = torch.cat([d4, x1], dim=1)
    d4 = decoder['conv4'](d4)

    output = decoder['final'](d4)

    return output

def voxelize_points(points, grid_size=32, voxel_size=None):
    """Convert point cloud to voxel grid"""
    if voxel_size is None:
        min_coords = points.min(axis=0)
        max_coords = points.max(axis=0)
        voxel_size = (max_coords - min_coords) / grid_size
    else:
        min_coords = points.min(axis=0)

    voxel_indices = ((points - min_coords) / voxel_size).astype(int)

    voxel_indices = np.clip(voxel_indices, 0, grid_size - 1)

    voxel_grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)

    for idx in voxel_indices:
        voxel_grid[idx[0], idx[1], idx[2]] = 1.0

    return voxel_grid, voxel_size, min_coords

def devoxelize_predictions(voxel_predictions, points, voxel_size, min_coords, grid_size=32):
    """Map voxel predictions back to points"""
    voxel_indices = ((points - min_coords) / voxel_size).astype(int)
    voxel_indices = np.clip(voxel_indices, 0, grid_size - 1)

    point_predictions = voxel_predictions[
        voxel_indices[:, 0],
        voxel_indices[:, 1],
        voxel_indices[:, 2]
    ]

    return point_predictions

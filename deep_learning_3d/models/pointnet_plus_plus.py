import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import cKDTree
import numpy as np

def farthest_point_sampling(points, num_samples):
    """Farthest Point Sampling for PointNet++"""
    device = points.device
    batch_size, num_points, _ = points.shape

    centroids = torch.zeros(batch_size, num_samples, dtype=torch.long, device=device)
    distances = torch.ones(batch_size, num_points, device=device) * 1e10

    farthest = torch.randint(0, num_points, (batch_size,), dtype=torch.long, device=device)

    batch_indices = torch.arange(batch_size, dtype=torch.long, device=device)

    for i in range(num_samples):
        centroids[:, i] = farthest

        centroid_points = points[batch_indices, farthest, :].view(batch_size, 1, 3)

        dist = torch.sum((points - centroid_points) ** 2, dim=-1)

        mask = dist < distances
        distances[mask] = dist[mask]

        farthest = torch.max(distances, dim=-1)[1]

    return centroids

def ball_query(radius, num_samples, xyz, new_xyz):
    """Ball query for grouping points"""
    device = xyz.device
    batch_size, num_points, _ = xyz.shape
    _, num_centroids, _ = new_xyz.shape

    group_idx = torch.arange(num_points, dtype=torch.long, device=device).view(1, 1, num_points).repeat(batch_size, num_centroids, 1)

    sqrdists = square_distance(new_xyz, xyz)

    group_idx[sqrdists > radius ** 2] = num_points

    group_idx = group_idx.sort(dim=-1)[0][:, :, :num_samples]

    group_first = group_idx[:, :, 0].view(batch_size, num_centroids, 1).repeat(1, 1, num_samples)

    mask = group_idx == num_points
    group_idx[mask] = group_first[mask]

    return group_idx

def square_distance(src, dst):
    """Calculate squared Euclidean distance"""
    batch_size, N, _ = src.shape
    _, M, _ = dst.shape

    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, dim=-1).view(batch_size, N, 1)
    dist += torch.sum(dst ** 2, dim=-1).view(batch_size, 1, M)

    return dist

def group_points(points, idx):
    """Group points based on indices"""
    device = points.device
    batch_size = points.shape[0]
    num_points = points.shape[2]

    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)

    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1

    batch_indices = torch.arange(batch_size, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)

    grouped_points = points[batch_indices, :, idx.long(), :]

    return grouped_points

def sample_and_group(npoint, radius, nsample, xyz, features):
    """Sample and group operation for PointNet++"""
    batch_size, num_features, num_points = features.shape

    xyz_t = xyz.permute(0, 2, 1).contiguous()

    fps_idx = farthest_point_sampling(xyz_t, npoint)

    new_xyz = torch.gather(xyz_t, 1, fps_idx.unsqueeze(-1).repeat(1, 1, 3))

    idx = ball_query(radius, nsample, xyz_t, new_xyz)

    grouped_xyz = group_points(xyz, idx)
    grouped_xyz_norm = grouped_xyz - new_xyz.permute(0, 2, 1).unsqueeze(-1)

    grouped_features = group_points(features, idx)

    new_features = torch.cat([grouped_xyz_norm, grouped_features], dim=1)

    return new_xyz.permute(0, 2, 1).contiguous(), new_features

def pointnet_plus_plus_set_abstraction(in_channels, out_channels, npoint, radius, nsample, mlp_channels):
    """Set Abstraction module for PointNet++"""
    mlp = nn.ModuleList()

    last_channel = in_channels

    for out_channel in mlp_channels:
        mlp.append(nn.Conv2d(last_channel, out_channel, 1))
        mlp.append(nn.BatchNorm2d(out_channel))
        mlp.append(nn.ReLU())
        last_channel = out_channel

    return {
        'npoint': npoint,
        'radius': radius,
        'nsample': nsample,
        'mlp': nn.Sequential(*mlp),
        'out_channels': mlp_channels[-1]
    }

def pointnet_plus_plus_sa_forward(xyz, features, sa_module):
    """Forward pass through Set Abstraction module"""
    new_xyz, grouped_features = sample_and_group(
        sa_module['npoint'],
        sa_module['radius'],
        sa_module['nsample'],
        xyz,
        features
    )

    new_features = sa_module['mlp'](grouped_features)

    new_features = torch.max(new_features, dim=-1)[0]

    return new_xyz, new_features

def pointnet_plus_plus_encoder(in_channels=3, feature_dim=1024):
    """PointNet++ encoder"""
    sa1 = pointnet_plus_plus_set_abstraction(
        in_channels=in_channels + 3,
        out_channels=64,
        npoint=512,
        radius=0.2,
        nsample=32,
        mlp_channels=[64, 64, 128]
    )

    sa2 = pointnet_plus_plus_set_abstraction(
        in_channels=128 + 3,
        out_channels=128,
        npoint=128,
        radius=0.4,
        nsample=64,
        mlp_channels=[128, 128, 256]
    )

    sa3 = pointnet_plus_plus_set_abstraction(
        in_channels=256 + 3,
        out_channels=256,
        npoint=None,
        radius=None,
        nsample=None,
        mlp_channels=[256, 512, feature_dim]
    )

    return {'sa1': sa1, 'sa2': sa2, 'sa3': sa3}

def pointnet_plus_plus_forward(xyz, features, encoder):
    """Forward pass through PointNet++ encoder"""
    l1_xyz, l1_features = pointnet_plus_plus_sa_forward(xyz, features, encoder['sa1'])

    l2_xyz, l2_features = pointnet_plus_plus_sa_forward(l1_xyz, l1_features, encoder['sa2'])

    l3_features = encoder['sa3']['mlp'](l2_features.unsqueeze(-1))
    global_feature = torch.max(l3_features, dim=-1)[0].squeeze(-1)

    return global_feature, [l1_xyz, l2_xyz], [l1_features, l2_features]

def pointnet_plus_plus_classification_head(feature_dim=1024, num_classes=10, dropout=0.4):
    """Classification head for PointNet++"""
    return nn.Sequential(
        nn.Linear(feature_dim, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(256, num_classes)
    )

def create_pointnet_plus_plus_classifier(num_classes=10, in_channels=3, feature_dim=1024):
    """Create complete PointNet++ for classification"""
    encoder = pointnet_plus_plus_encoder(in_channels, feature_dim)
    classifier = pointnet_plus_plus_classification_head(feature_dim, num_classes)

    return {'encoder': encoder, 'classifier': classifier}

def pointnet_plus_plus_classify(xyz, features, model_components):
    """Run classification with PointNet++"""
    global_feature, _, _ = pointnet_plus_plus_forward(
        xyz, features, model_components['encoder']
    )

    logits = model_components['classifier'](global_feature)

    return logits

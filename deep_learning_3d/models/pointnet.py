import torch
import torch.nn as nn
import torch.nn.functional as F

def create_mlp(channels, batch_norm=True, activation='relu'):
    """Create multi-layer perceptron"""
    layers = []

    for i in range(len(channels) - 1):
        layers.append(nn.Conv1d(channels[i], channels[i + 1], 1))

        if batch_norm and i < len(channels) - 2:
            layers.append(nn.BatchNorm1d(channels[i + 1]))

        if i < len(channels) - 2:
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU(0.2))

    return nn.Sequential(*layers)

def pointnet_feature_transform(in_channels, out_channels=64):
    """T-Net for feature transformation"""
    mlp1 = create_mlp([in_channels, 64, 128, 1024])

    fc_layers = nn.Sequential(
        nn.Linear(1024, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Linear(256, in_channels * in_channels)
    )

    return mlp1, fc_layers, in_channels

def apply_feature_transform(x, transform_net):
    """Apply learned feature transformation"""
    mlp1, fc_layers, k = transform_net

    batch_size, num_features, num_points = x.size()

    features = mlp1(x)
    features = torch.max(features, 2, keepdim=True)[0]
    features = features.view(-1, 1024)

    matrix = fc_layers(features)
    matrix = matrix.view(-1, k, k)

    identity = torch.eye(k, device=x.device).unsqueeze(0).repeat(batch_size, 1, 1)
    matrix = matrix + identity

    x_transformed = torch.bmm(matrix, x)

    return x_transformed, matrix

def pointnet_encoder(in_channels=3, feature_dim=1024, use_transform=True):
    """PointNet encoder for global features"""
    components = {}

    if use_transform:
        components['input_transform'] = pointnet_feature_transform(in_channels)

    components['mlp1'] = create_mlp([in_channels, 64, 64])

    if use_transform:
        components['feature_transform'] = pointnet_feature_transform(64)

    components['mlp2'] = create_mlp([64, 128, feature_dim])

    return components

def pointnet_forward(x, encoder_components):
    """Forward pass through PointNet encoder"""
    batch_size = x.size(0)
    num_points = x.size(2)

    transform_matrices = []

    if 'input_transform' in encoder_components:
        x, trans_input = apply_feature_transform(x, encoder_components['input_transform'])
        transform_matrices.append(trans_input)

    point_features = encoder_components['mlp1'](x)

    if 'feature_transform' in encoder_components:
        point_features, trans_feat = apply_feature_transform(
            point_features, encoder_components['feature_transform']
        )
        transform_matrices.append(trans_feat)

    point_features = encoder_components['mlp2'](point_features)

    global_feature = torch.max(point_features, 2, keepdim=True)[0]
    global_feature = global_feature.view(-1, global_feature.size(1))

    return global_feature, point_features, transform_matrices

def pointnet_classification_head(feature_dim=1024, num_classes=10, dropout=0.3):
    """Classification head for PointNet"""
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

def pointnet_segmentation_head(global_dim=1024, point_dim=1024, num_classes=10):
    """Segmentation head for PointNet"""
    return create_mlp([global_dim + point_dim, 512, 256, 128, num_classes], batch_norm=True)

def pointnet_forward_segmentation(x, encoder_components, seg_head):
    """Forward pass for segmentation"""
    batch_size = x.size(0)
    num_points = x.size(2)

    global_feature, point_features, transforms = pointnet_forward(x, encoder_components)

    global_expanded = global_feature.unsqueeze(2).repeat(1, 1, num_points)

    concatenated = torch.cat([point_features, global_expanded], dim=1)

    output = seg_head(concatenated)
    output = output.transpose(2, 1).contiguous()

    return output, transforms

def feature_transform_regularizer(trans_matrices, reg_weight=0.001):
    """Regularization loss for feature transform matrices"""
    loss = 0

    for trans in trans_matrices:
        d = trans.size(1)

        trans_t = trans.transpose(2, 1)
        product = torch.bmm(trans, trans_t)

        identity = torch.eye(d, device=trans.device).unsqueeze(0).repeat(trans.size(0), 1, 1)

        diff = product - identity
        loss += torch.mean(torch.norm(diff, dim=(1, 2)))

    return loss * reg_weight

def create_pointnet_classifier(num_classes=10, in_channels=3, feature_dim=1024, dropout=0.3):
    """Create complete PointNet for classification"""
    encoder = pointnet_encoder(in_channels, feature_dim, use_transform=True)
    classifier = pointnet_classification_head(feature_dim, num_classes, dropout)

    return {'encoder': encoder, 'classifier': classifier}

def create_pointnet_segmentation(num_classes=10, in_channels=3, feature_dim=1024):
    """Create complete PointNet for segmentation"""
    encoder = pointnet_encoder(in_channels, feature_dim, use_transform=True)
    seg_head = pointnet_segmentation_head(feature_dim, feature_dim, num_classes)

    return {'encoder': encoder, 'seg_head': seg_head}

def pointnet_classify(x, model_components):
    """Run classification inference"""
    global_feature, _, transforms = pointnet_forward(x, model_components['encoder'])
    logits = model_components['classifier'](global_feature)

    return logits, transforms

def pointnet_segment(x, model_components):
    """Run segmentation inference"""
    return pointnet_forward_segmentation(x, model_components['encoder'], model_components['seg_head'])

def extract_pointnet_features(points, model_components):
    """Extract global features for downstream tasks"""
    if points.dim() == 2:
        points = points.unsqueeze(0)

    if points.size(1) > points.size(2):
        points = points.transpose(1, 2)

    global_feature, _, _ = pointnet_forward(points, model_components['encoder'])

    return global_feature

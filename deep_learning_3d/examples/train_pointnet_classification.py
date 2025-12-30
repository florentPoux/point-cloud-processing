#!/usr/bin/env python3
"""
Train PointNet for point cloud classification
"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from deep_learning_3d.models.pointnet import (
    create_pointnet_classifier,
    pointnet_classify
)
from deep_learning_3d.data.dataset import PointCloudDataset, create_dataloader
from deep_learning_3d.training.trainer import train_model

def main(data_root, num_classes=10, num_epochs=100, batch_size=32):
    """Train PointNet classifier"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    print("Creating model...")
    model_components = create_pointnet_classifier(
        num_classes=num_classes,
        in_channels=3,
        feature_dim=1024,
        dropout=0.3
    )

    for name, component in model_components.items():
        if isinstance(component, torch.nn.Module):
            component.to(device)
        elif isinstance(component, dict):
            for sub_name, sub_component in component.items():
                if isinstance(sub_component, torch.nn.Module):
                    sub_component.to(device)

    print("Loading datasets...")
    train_dataset = PointCloudDataset(
        data_root,
        split='train',
        num_points=2048,
        augment=True,
        task='classification'
    )

    val_dataset = PointCloudDataset(
        data_root,
        split='val',
        num_points=2048,
        augment=False,
        task='classification'
    )

    train_loader = create_dataloader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = create_dataloader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    print("Starting training...")
    best_acc = train_model(
        pointnet_classify,
        model_components,
        train_loader,
        val_loader,
        num_epochs=num_epochs,
        device=device,
        task='classification',
        save_dir='./checkpoints/pointnet_cls',
        lr=0.001
    )

    print(f"Training complete! Best accuracy: {best_acc:.2f}%")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train PointNet for classification')
    parser.add_argument('data_root', type=str, help='Path to dataset root')
    parser.add_argument('--num-classes', type=int, default=10, help='Number of classes')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')

    args = parser.parse_args()

    main(args.data_root, args.num_classes, args.epochs, args.batch_size)

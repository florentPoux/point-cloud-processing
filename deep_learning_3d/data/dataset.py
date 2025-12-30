import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path

class PointCloudDataset(Dataset):
    """Dataset for point cloud classification/segmentation"""

    def __init__(self, data_root, split='train', num_points=2048, augment=True, task='classification'):
        self.data_root = Path(data_root)
        self.split = split
        self.num_points = num_points
        self.augment = augment
        self.task = task

        self.data_files = list((self.data_root / split).glob('*.npy'))

        if len(self.data_files) == 0:
            self.data_files = list((self.data_root / split).glob('*.npz'))

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        filepath = self.data_files[idx]

        if filepath.suffix == '.npy':
            data = np.load(filepath)
            points = data[:, :3]
            if self.task == 'segmentation':
                labels = data[:, -1].astype(np.int64)
            else:
                labels = int(filepath.stem.split('_')[-1])

        elif filepath.suffix == '.npz':
            data = np.load(filepath)
            points = data['points']
            labels = data['labels'] if 'labels' in data else data['label']

        if points.shape[0] != self.num_points:
            from deep_learning_3d.data.augmentation import sample_fixed_points
            points = sample_fixed_points(points, self.num_points)

            if self.task == 'segmentation' and isinstance(labels, np.ndarray):
                labels = sample_fixed_points(labels.reshape(-1, 1), self.num_points).flatten()

        if self.augment and self.split == 'train':
            from deep_learning_3d.data.augmentation import augment_point_cloud
            points = augment_point_cloud(points)

        from deep_learning_3d.data.augmentation import normalize_points
        points = normalize_points(points, method='center')

        points = torch.from_numpy(points).float()

        if self.task == 'segmentation':
            labels = torch.from_numpy(labels).long()
        else:
            labels = torch.tensor(labels).long()

        points = points.transpose(0, 1)

        return points, labels

def create_dataloader(dataset, batch_size=32, shuffle=True, num_workers=4):
    """Create PyTorch DataLoader"""
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )

def prepare_batch_for_pointnet(batch):
    """Prepare batch for PointNet input format"""
    points, labels = batch

    return points, labels

def prepare_batch_for_pointnet_plus_plus(batch):
    """Prepare batch for PointNet++ input format"""
    points, labels = batch

    xyz = points[:, :3, :]
    features = points if points.size(1) > 3 else None

    return (xyz, features), labels

def collate_variable_points(batch):
    """Collate function for variable-sized point clouds"""
    points_list = []
    labels_list = []

    max_points = max([item[0].shape[1] for item in batch])

    for points, labels in batch:
        num_points = points.shape[1]

        if num_points < max_points:
            padding = torch.zeros(points.shape[0], max_points - num_points)
            points = torch.cat([points, padding], dim=1)

        points_list.append(points)
        labels_list.append(labels)

    points_batch = torch.stack(points_list)
    labels_batch = torch.stack(labels_list) if isinstance(labels_list[0], torch.Tensor) else torch.tensor(labels_list)

    return points_batch, labels_batch

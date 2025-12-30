# 🧠 3D Deep Learning
## System 05: Deep Learning for Point Clouds and 3D Data

Production-ready library for training and deploying deep learning models on point clouds and voxelized 3D data. Includes PointNet, PointNet++, and 3D CNNs with complete training pipelines and inference tools.

Part of the **Spatial AI Architect Program** by [3D Geodata Academy](https://learngeodata.eu).

## 🎯 Key Features

- **PointNet Architecture**: Classification and segmentation with T-Net transformations
- **PointNet++**: Hierarchical feature learning with set abstraction
- **3D CNNs**: Voxel-based processing with U-Net for segmentation
- **Training Utilities**: Complete training loops with checkpointing
- **Data Augmentation**: Rotation, scaling, jitter, dropout for robustness
- **Inference Tools**: Batch prediction, TTA, feature extraction
- **ONNX Export**: Deploy models to production environments

## 📦 Installation

```bash
# Install dependencies
pip install torch torchvision numpy scipy scikit-learn

# Clone and use
git clone https://github.com/florentPoux/point-cloud-processing.git
cd point-cloud-processing
```

## 🚀 Quick Start

### Train PointNet Classifier

```bash
python deep_learning_3d/examples/train_pointnet_classification.py ./data --num-classes 10 --epochs 100
```

### Python API

```python
from deep_learning_3d.models.pointnet import create_pointnet_classifier, pointnet_classify
from deep_learning_3d.inference.predictor import predict_classification

# Create model
model = create_pointnet_classifier(num_classes=10, in_channels=3)

# Predict
predicted_class, confidence, probabilities = predict_classification(
    points, pointnet_classify, model, device='cuda'
)

print(f"Class: {predicted_class}, Confidence: {confidence:.2f}")
```

## 📚 Modules

### Models (`models/`)
- `pointnet.py` - PointNet for classification and segmentation
- `pointnet_plus_plus.py` - PointNet++ with hierarchical learning
- `voxel_cnn.py` - 3D CNNs for voxel-based processing

### Training (`training/`)
- `trainer.py` - Complete training loops, optimizers, schedulers

### Data (`data/`)
- `augmentation.py` - Rotation, scaling, jitter, normalization
- `dataset.py` - PyTorch datasets and dataloaders

### Inference (`inference/`)
- `predictor.py` - Batch prediction, TTA, ONNX export, embeddings

## 🎓 Examples

### PointNet Classification

```python
from deep_learning_3d.models.pointnet import create_pointnet_classifier, pointnet_classify
import torch

# Create model
model = create_pointnet_classifier(num_classes=10, feature_dim=1024)

# Move to GPU
device = 'cuda'
for component in model.values():
    if isinstance(component, torch.nn.Module):
        component.to(device)

# Prepare data
points = torch.randn(1, 3, 2048).to(device)  # [batch, channels, points]

# Forward pass
logits, transforms = pointnet_classify(points, model)

# Get predictions
predictions = torch.argmax(logits, dim=1)
```

### PointNet Segmentation

```python
from deep_learning_3d.models.pointnet import create_pointnet_segmentation, pointnet_segment

# Create model
model = create_pointnet_segmentation(num_classes=5, feature_dim=1024)

# Forward pass
output, transforms = pointnet_segment(points, model)

# Get per-point labels
predictions = torch.argmax(output, dim=2)  # [batch, points]
```

### PointNet++ Classification

```python
from deep_learning_3d.models.pointnet_plus_plus import (
    create_pointnet_plus_plus_classifier,
    pointnet_plus_plus_classify
)

# Create model
model = create_pointnet_plus_plus_classifier(num_classes=10)

# Prepare input
xyz = torch.randn(1, 3, 2048).to(device)
features = xyz  # or additional features

# Classify
logits = pointnet_plus_plus_classify(xyz, features, model)
```

### 3D CNN for Voxels

```python
from deep_learning_3d.models.voxel_cnn import (
    create_voxel_classifier,
    voxel_classify,
    voxelize_points
)

# Convert point cloud to voxels
voxel_grid, voxel_size, min_coords = voxelize_points(points, grid_size=32)

# Create model
model = create_voxel_classifier(in_channels=1, num_classes=10, grid_size=32)

# Prepare input
voxels = torch.from_numpy(voxel_grid).unsqueeze(0).unsqueeze(0).float()

# Classify
logits = voxel_classify(voxels, model)
```

### Data Augmentation

```python
from deep_learning_3d.data.augmentation import (
    augment_point_cloud,
    rotate_points_z_axis,
    random_scale,
    random_jitter,
    normalize_points
)

# Full augmentation pipeline
augmented = augment_point_cloud(points, config={
    'rotate': True,
    'scale': True,
    'jitter': True,
    'dropout': False
})

# Individual augmentations
rotated = rotate_points_z_axis(points)
scaled = random_scale(points, scale_low=0.9, scale_high=1.1)
jittered = random_jitter(points, sigma=0.01, clip=0.05)
normalized = normalize_points(points, method='unit_sphere')
```

### Training

```python
from deep_learning_3d.training.trainer import train_model
from deep_learning_3d.data.dataset import PointCloudDataset, create_dataloader

# Create datasets
train_dataset = PointCloudDataset('./data', split='train', num_points=2048, augment=True)
val_dataset = PointCloudDataset('./data', split='val', num_points=2048, augment=False)

# Create dataloaders
train_loader = create_dataloader(train_dataset, batch_size=32, shuffle=True)
val_loader = create_dataloader(val_dataset, batch_size=32, shuffle=False)

# Train
best_acc = train_model(
    pointnet_classify,
    model_components,
    train_loader,
    val_loader,
    num_epochs=100,
    device='cuda',
    task='classification',
    save_dir='./checkpoints',
    lr=0.001
)
```

### Inference

```python
from deep_learning_3d.inference.predictor import (
    predict_classification,
    predict_segmentation,
    predict_with_tta,
    compute_feature_embeddings
)

# Single prediction
predicted_class, confidence, probs = predict_classification(
    points, pointnet_classify, model, device='cuda'
)

# Segmentation
labels = predict_segmentation(
    points, pointnet_segment, model, device='cuda'
)

# Test-time augmentation
predicted_class, confidence, probs = predict_with_tta(
    points, pointnet_classify, model, num_augmentations=8
)

# Feature extraction for retrieval
embeddings = compute_feature_embeddings(
    points_list, model, device='cuda'
)
```

## 🏆 Advanced Features

### Custom Training Loop

```python
from deep_learning_3d.training.trainer import (
    create_optimizer,
    create_scheduler,
    train_epoch_classification,
    validate_classification,
    save_checkpoint
)

# Create optimizer and scheduler
optimizer = create_optimizer(model.parameters(), optimizer_type='adam', lr=0.001)
scheduler = create_scheduler(optimizer, scheduler_type='cosine', T_max=100)

# Training loop
for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch_classification(
        pointnet_classify, model, train_loader, optimizer, device
    )

    val_loss, val_acc = validate_classification(
        pointnet_classify, model, val_loader, device
    )

    scheduler.step()

    if val_acc > best_acc:
        save_checkpoint('./best_model.pth', model, optimizer, epoch, val_acc)
```

### ONNX Export

```python
from deep_learning_3d.inference.predictor import export_onnx

# Export to ONNX
export_onnx(
    pointnet_classify,
    model_components,
    'pointnet_classifier.onnx',
    input_shape=(1, 3, 2048)
)
```

## 📖 Resources

- [PointNet Paper](https://arxiv.org/abs/1612.00593)
- [PointNet++ Paper](https://arxiv.org/abs/1706.02413)
- [3D Geodata Academy](https://learngeodata.eu)
- [Deep Learning for 3D](https://medium.com/@florentpoux)

## 🛠️ Built With

* [PyTorch](https://pytorch.org/) - Deep learning framework
* [NumPy](https://numpy.org/) - Array operations
* [SciPy](https://scipy.org/) - Scientific computing

## 👨‍💻 Author

**Florent Poux** - [3D Geodata Academy](https://learngeodata.eu)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

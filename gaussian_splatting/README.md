# ✨ Scalable Gaussian Splatting
## System 02: Production-Ready 3DGS Training and Rendering

A complete implementation of 3D Gaussian Splatting for novel view synthesis, from training to deployment.

## 🎯 Features

- **Full Training Pipeline**: Densification, pruning, and optimization
- **CPU Rasterizer**: Pure Python/PyTorch implementation for understanding
- **LOD Generation**: Multi-resolution hierarchy for massive scenes
- **Streaming Support**: Progressive loading for web deployment
- **Web Renderer**: WebGL2-based real-time viewer
- **Marketplace Tools**: Package and monetize your 3DGS models

## 📦 Installation

```bash
pip install torch numpy scipy matplotlib
```

## 🚀 Quick Start

### Train from COLMAP

```python
from gaussian_splatting.data.colmap_loader import load_colmap_dataset
from gaussian_splatting.train.trainer import full_training_loop

# Load COLMAP reconstruction
dataset = load_colmap_dataset('path/to/colmap')

# Train Gaussian splats
gaussians = full_training_loop(
    initial_positions=dataset['points'],
    initial_colors=dataset['colors'],
    train_views=dataset['views'],
    num_iterations=30000,
    save_dir='./checkpoints'
)
```

### Render Novel Views

```python
from gaussian_splatting.core.gaussian import load_gaussian_model
from gaussian_splatting.core.rasterizer import rasterize_gaussians

# Load trained model
positions, colors, scales, rotations, opacities, sh = load_gaussian_model('model.npz')

# Render from camera
rendered = rasterize_gaussians(
    positions, colors, scales, rotations, opacities,
    view_matrix, proj_matrix, width, height, fov_x, fov_y
)
```

### Generate LOD Hierarchy

```python
from gaussian_splatting.streaming.lod_gen import generate_lod_levels, save_lod_hierarchy

# Create multi-resolution pyramid
lod_levels = generate_lod_levels(
    positions, colors, scales, rotations, opacities,
    num_levels=4
)

# Save for streaming
save_lod_hierarchy(lod_levels, './output/lods')
```

### Package for Marketplace

```python
from gaussian_splatting.deploy.asset_packager import package_for_marketplace

# Create distribution package
metadata = {
    'title': 'My Awesome Scene',
    'description': 'High-quality 3DGS model',
    'author': 'Your Name',
    'tags': ['outdoor', 'architecture']
}

package = package_for_marketplace(
    model_path='final_model.npz',
    metadata=metadata,
    output_dir='./marketplace_package'
)
```

## 📚 Module Overview

### Core (`core/`)
- **`gaussian.py`**: Gaussian data structures, SH evaluation
- **`math_utils.py`**: Projection matrices, 2D covariance
- **`rasterizer.py`**: CPU rasterizer for rendering
- **`gradients.py`**: Gradient computation and verification

### Training (`train/`)
- **`loss.py`**: L1, SSIM, perceptual losses
- **`densification.py`**: Adaptive split/clone
- **`pruner.py`**: Opacity and scale-based pruning
- **`trainer.py`**: Complete training loop

### Streaming (`streaming/`)
- **`indexer.py`**: Octree spatial indexing
- **`lod_gen.py`**: LOD hierarchy generation
- **`serializer.py`**: Binary serialization for streaming

### Data (`data/`)
- **`colmap_loader.py`**: Load COLMAP reconstructions

### Deploy (`deploy/`)
- **`asset_packager.py`**: Marketplace packaging

### Web (`web/`)
- **`renderer/splat_renderer.js`**: WebGL2 renderer
- Additional web tools for viewing and editing

## 🎓 Training Pipeline

```python
# Initialize from point cloud
from gaussian_splatting.core.gaussian import initialize_from_points

positions, colors, scales, rotations, opacities = initialize_from_points(
    points, colors, initial_scale=0.01
)

# Create optimizer
from gaussian_splatting.train.trainer import create_optimizer

optimizer = create_optimizer(positions, colors, scales, rotations, opacities)

# Training loop with densification
for iteration in range(30000):
    # Forward pass
    rendered = rasterize_gaussians(...)

    # Compute loss
    loss = combined_loss(rendered, target)

    # Backward pass
    loss.backward()
    optimizer.step()

    # Densification (split/clone)
    if iteration % 100 == 0:
        gaussians = adaptive_density_control(gaussians, grad_threshold=0.0002)

    # Pruning
    if iteration % 500 == 0:
        gaussians = comprehensive_pruning(gaussians, opacity_threshold=0.005)
```

## 🌐 Web Deployment

```javascript
// Load and render Gaussian splats in browser
const renderer = new SplatRenderer(canvas);
renderer.initShaders();

// Load model
fetch('gaussians.json')
    .then(res => res.json())
    .then(data => {
        renderer.loadGaussians(data);
        renderer.render();
    });
```

## 🏆 Performance Features

- **Adaptive Densification**: Automatically adds detail where needed
- **Intelligent Pruning**: Removes redundant Gaussians
- **LOD Support**: Multi-resolution for large scenes
- **Binary Streaming**: Fast progressive loading
- **Octree Indexing**: Efficient spatial queries

## 📊 Training Tips

1. **Start Small**: Initialize with ~10K-100K points from COLMAP
2. **Densify Early**: Enable densification from iteration 500-15000
3. **Prune Regularly**: Every 500 iterations to prevent bloat
4. **Monitor PSNR**: Should reach >25dB for good quality
5. **Use SSIM**: Combined L1+SSIM loss works best

## 🎨 Use Cases

- Novel view synthesis
- VR/AR experiences
- Virtual tours
- Product visualization
- Archit

ectural walkthrough
- Game asset creation

## 📖 Resources

- [3D Gaussian Splatting Paper](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- [Original Implementation](https://github.com/graphdeco-inria/gaussian-splatting)

## 👨‍💻 Author

**Florent Poux** - [3D Geodata Academy](https://learngeodata.eu)

Part of the **Spatial AI Architect Program** - System 02

## 📄 License

MIT License - see LICENSE file for details

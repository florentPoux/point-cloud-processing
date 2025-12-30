# 🧠 The Smart Point Cloud
## System 03: High-Performance Point Cloud Processing Library

A production-ready Python library for cleaning, analyzing, segmenting, and processing massive point cloud data. Designed for real-time processing of large-scale datasets with efficient memory management and vectorized operations.

Part of the **Spatial AI Architect Program** by [3D Geodata Academy](https://learngeodata.eu).

## 🎯 Key Features

- **Massive Dataset Support**: Memory-mapped I/O and streaming processing for files >50GB
- **Real-Time Performance**: Vectorized KDTree operations with scipy
- **Smart Sampling**: Intelligent downsampling with full-resolution projection using KDTree
- **Binary PLY Export**: Fast binary writing with predictions as scalar fields
- **Function-Based Design**: No classes, pure functional approach for flexibility
- **Production Ready**: Optimized for real-world applications

## 📦 Installation

```bash
# Install dependencies
pip install numpy scipy matplotlib scikit-learn scikit-image laspy

# Clone and use
git clone https://github.com/florentPoux/point-cloud-processing.git
cd point-cloud-processing
```

## 🚀 Quick Start

### CLI Usage

```bash
# Basic processing
python cli/refine.py input.las output.ply --visualize

# Custom segmentation
python cli/refine.py input.las output.ply \
    --segment-method dbscan \
    --dbscan-eps 0.5 \
    --downsample 0.1

# Large file with tiling
python cli/refine.py huge.las output.ply --large-file --tile-size 100
```

### Python API

```python
from data.io_manager import read_las_full, write_ply_binary
from pipeline.refinery import process_point_cloud_pipeline

# Load and process
points, colors, _, _ = read_las_full('input.las')
results = process_point_cloud_pipeline(points, colors)

# Export with predictions as scalar fields
scalar_fields = {
    'labels': results['segment_labels'],
    'height': results['heights']
}
write_ply_binary('output.ply', points, colors=colors,
                 normals=results['normals'],
                 scalar_fields=scalar_fields)
```

## 📚 Modules

### Data (`data/`)
- `io_manager.py` - Memory-mapped I/O, binary PLY export
- `tiler.py` - Spatial tiling for large datasets
- `spatial_hash.py` - Fast spatial queries with voxel hashing
- `converter.py` - Format transcoding (LAS/PLY/NPZ)

### Filtering (`filters/`)
- `sor.py` - Statistical outlier removal
- `radius.py` - Radius-based filtering
- `ground_seg.py` - Ground segmentation (CSF, PMF)
- `sampler.py` - Intelligent downsampling + KDTree projection

### Features (`features/`)
- `geometric.py` - Eigenvalue-based features
- `normals.py` - Normal estimation
- `height.py` - Height above ground
- `color.py` - Color space transformations

### Segmentation (`segmentation/`)
- `dbscan.py` - DBSCAN clustering
- `ransac_shapes.py` - Geometric primitive detection
- `region_grow.py` - Region growing
- `ml_classifier.py` - Random Forest classification

### Registration (`registration/`)
- `global_align.py` - FPFH-based global registration
- `icp.py` - ICP variants (point-to-point, point-to-plane)
- `multi_way.py` - Multi-scan registration

## 🎓 Complete Example

```python
from data.io_manager import read_las_full, write_ply_binary
from filters.sor import statistical_outlier_removal
from filters.ground_seg import extract_ground_points
from filters.sampler import fps_sampling, project_predictions_to_full_resolution
from features.normals import estimate_normals
from segmentation.dbscan import dbscan_clustering

# Load
points, colors, _, _ = read_las_full('input.las')

# Filter outliers
points, mask = statistical_outlier_removal(points, k=20, std_multiplier=2.0)
colors = colors[mask]

# Segment ground
ground, non_ground, ground_mask = extract_ground_points(points, method='csf')

# Downsample for processing
sampled, idx = fps_sampling(points, target_size=50000)

# Segment on downsampled data
labels = dbscan_clustering(sampled, eps=0.5, min_samples=10)

# Project labels to full resolution using KDTree
full_labels = project_predictions_to_full_resolution(
    sampled, labels.astype(np.float32), points, k=5
)

# Export with predictions as scalar fields
write_ply_binary('output.ply', points, colors,
                 scalar_fields={'labels': full_labels})
```

## 🏆 Performance Features

- **Memory-Mapped I/O**: Process 50GB+ files without loading into RAM
- **Vectorized KDTree**: Fast scipy-based nearest neighbor queries
- **Spatial Hashing**: O(1) voxel lookups for massive datasets
- **Smart Sampling**: Process on downsampled data, project to full resolution
- **Binary PLY**: Real-time writing with predictions as scalar fields

## 📖 Resources

- [Point Cloud Basics](https://towardsdatascience.com/discover-3d-point-cloud-processing-with-python-6112d9ee38e7)
- [Medium Articles](https://medium.com/@florentpoux)
- [3D Geodata Academy](https://learngeodata.eu)

## 🛠️ Built With

* [Python](https://www.python.org/) - Programming language
* [NumPy](https://numpy.org/) - Array operations
* [SciPy](https://scipy.org/) - Vectorized KDTree operations
* [scikit-learn](https://scikit-learn.org/) - Machine learning
* [LasPy](https://laspy.readthedocs.io/) - LAS file I/O
* [Matplotlib](https://matplotlib.org/) - Visualization

## 👨‍💻 Author

**Florent Poux** - [3D Geodata Academy](https://learngeodata.eu)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

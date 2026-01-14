#!/usr/bin/env python3
"""
EXAMPLE: Complete Point Cloud Segmentation Pipeline
Following all coding guidelines for the Spatial AI Architect Program

This demonstrates proper structure with:
- Function-only design (no classes)
- Reusable visualization functions
- Branch notes for improvements
- Minimalist documentation
- Efficient vectorized operations
"""

#%% 00. IMPORTS - All at the beginning
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter
from pathlib import Path

# Florent's Note: Import discipline prevents circular dependencies

#%% 01. VISUALIZATION - Initialize once, reuse everywhere

_fig_3d = None
_ax_3d = None

def init_viz():
    """Create reusable 3D visualization"""
    global _fig_3d, _ax_3d
    _fig_3d = plt.figure(figsize=(12, 10))
    _ax_3d = _fig_3d.add_subplot(111, projection='3d')
    return _fig_3d, _ax_3d

def plot_cloud(points, colors=None, labels=None, s=1, title="Point Cloud"):
    """Plot point cloud with optional labels"""
    global _ax_3d
    if _ax_3d is None:
        init_viz()

    _ax_3d.clear()

    if labels is not None:
        unique_labels = np.unique(labels[labels >= 0])
        for label in unique_labels:
            mask = labels == label
            _ax_3d.scatter(points[mask, 0], points[mask, 1], points[mask, 2],
                          s=s, alpha=0.6)
    elif colors is not None:
        _ax_3d.scatter(points[:, 0], points[:, 1], points[:, 2],
                      c=colors, s=s, alpha=0.6)
    else:
        _ax_3d.scatter(points[:, 0], points[:, 1], points[:, 2],
                      s=s, alpha=0.6)

    _ax_3d.set_xlabel('X')
    _ax_3d.set_ylabel('Y')
    _ax_3d.set_zlabel('Z')
    _ax_3d.set_title(title)
    plt.draw()
    return _ax_3d

# Time to test step 1: Create our visualization tools!
# fig, ax = init_viz()
# test_pts = np.random.randn(1000, 3)
# plot_cloud(test_pts, title="Test Visualization")
# plt.show()

# Branch: Add real-time interactive 3D with Open3D for better UX

#%% 02. DATA LOADING

def load_ply_simple(filepath):
    """Load PLY file without external dependencies"""
    with open(filepath, 'rb') as f:
        header = []
        while True:
            line = f.readline().decode('ascii').strip()
            header.append(line)
            if line == 'end_header':
                break

        # Parse header - minimalist approach
        num_points = 0
        for line in header:
            if line.startswith('element vertex'):
                num_points = int(line.split()[-1])

        # Read binary data
        dtype = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                         ('r', 'u1'), ('g', 'u1'), ('b', 'u1')])
        data = np.frombuffer(f.read(), dtype=dtype, count=num_points)

        points = np.column_stack([data['x'], data['y'], data['z']])
        colors = np.column_stack([data['r'], data['g'], data['b']]) / 255.0

    return points, colors

# Time to test step 2: Load real point cloud data!
# points, colors = load_ply_simple('data/sample.ply')
# plot_cloud(points, colors, title="Loaded Point Cloud")

#%% 03. PREPROCESSING

def normalize_cloud(points):
    """Center and scale to unit cube"""
    centroid = np.mean(points, axis=0)
    points_centered = points - centroid
    scale = np.max(np.abs(points_centered))
    return points_centered / scale, centroid, scale

def voxel_downsample(points, voxel_size=0.05):
    """Fast voxel grid downsampling"""
    # Vectorized voxel assignment - no nested loops
    voxel_indices = np.floor(points / voxel_size).astype(int)

    # Unique voxel coordinates
    _, unique_indices = np.unique(voxel_indices, axis=0, return_index=True)

    return points[unique_indices], unique_indices

# Time to test step 3: Normalize and downsample efficiently!
# normalized, centroid, scale = normalize_cloud(points)
# downsampled, indices = voxel_downsample(normalized, voxel_size=0.02)
# plot_cloud(downsampled, title=f"Downsampled: {len(downsampled)} points")

# Branch: GPU-accelerated voxelization with CuPy for 100x speedup

#%% 04. NORMAL ESTIMATION

def estimate_normals_fast(points, k=20):
    """Estimate normals using PCA"""
    tree = cKDTree(points)
    _, indices = tree.query(points, k=k)

    # Vectorized covariance computation
    neighbors = points[indices]
    centered = neighbors - np.mean(neighbors, axis=1, keepdims=True)

    # Why: Compute all covariances at once
    cov = np.einsum('ijk,ijl->ikl', centered, centered) / k

    # Eigenvectors are normals
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    normals = eigenvectors[:, :, 0]

    # Why: Ensure consistent orientation toward camera
    camera_pos = np.array([0, 0, 10])
    view_direction = camera_pos - points
    dots = np.sum(normals * view_direction, axis=1)
    normals[dots < 0] *= -1

    return normals

# Time to test step 4: Compute surface normals!
# normals = estimate_normals_fast(downsampled, k=20)

# Florent's Note: Normal quality is crucial for all downstream tasks

#%% 05. FEATURE EXTRACTION

def compute_geometric_features(points, normals, k=20):
    """Extract geometric features for classification"""
    tree = cKDTree(points)
    _, indices = tree.query(points, k=k)

    neighbors = points[indices]

    # Why: Vectorized feature computation
    z_values = neighbors[:, :, 2]
    z_std = np.std(z_values, axis=1, keepdims=True)
    z_range = (np.max(z_values, axis=1) - np.min(z_values, axis=1)).reshape(-1, 1)

    verticality = np.abs(normals[:, 2:3])

    features = np.hstack([z_std, z_range, verticality])

    return features

# Time to test step 5: Extract geometric features!
# features = compute_geometric_features(downsampled, normals)

# Branch: Deep learned features with PointNet++ for richer representations

#%% 06. SEGMENTATION

def dbscan_simple(points, eps=0.1, min_samples=10):
    """Simplified DBSCAN clustering"""
    tree = cKDTree(points)

    labels = np.full(len(points), -1)
    current_label = 0

    for i in range(len(points)):
        if labels[i] != -1:
            continue

        neighbors = tree.query_ball_point(points[i], eps)

        if len(neighbors) < min_samples:
            continue

        # Start new cluster
        labels[i] = current_label
        seed_set = list(neighbors)

        # Why: Grow cluster without recursion
        j = 0
        while j < len(seed_set):
            neighbor_idx = seed_set[j]

            if labels[neighbor_idx] == -1:
                labels[neighbor_idx] = current_label

            if labels[neighbor_idx] >= 0:
                j += 1
                continue

            labels[neighbor_idx] = current_label
            new_neighbors = tree.query_ball_point(points[neighbor_idx], eps)

            if len(new_neighbors) >= min_samples:
                seed_set.extend(new_neighbors)

            j += 1

        current_label += 1

    return labels

# Time to test step 6: Segment the cloud into clusters!
# labels = dbscan_simple(downsampled, eps=0.05, min_samples=20)
# plot_cloud(downsampled, labels=labels, title=f"Segmented: {len(np.unique(labels))} clusters")

# Branch: Hierarchical clustering for multi-scale segmentation

#%% 07. CLASSIFICATION

def classify_segments(points, labels, features):
    """Simple rule-based classification"""
    unique_labels = np.unique(labels[labels >= 0])
    classifications = {}

    for label in unique_labels:
        mask = labels == label
        segment_features = features[mask]

        # Why: Simple thresholds for demo - replace with ML
        mean_verticality = np.mean(segment_features[:, 2])
        mean_z_range = np.mean(segment_features[:, 1])

        if mean_verticality > 0.8 and mean_z_range > 2.0:
            class_name = 'wall'
        elif mean_verticality < 0.2:
            class_name = 'ground'
        elif mean_z_range > 1.0:
            class_name = 'vegetation'
        else:
            class_name = 'other'

        classifications[int(label)] = class_name

    return classifications

# Time to test step 7: Classify segments!
# classifications = classify_segments(downsampled, labels, features)
# for label, class_name in classifications.items():
#     print(f"Cluster {label}: {class_name}")

#%% 08. POST-PROCESSING

def smooth_labels(points, labels, k=5):
    """Smooth labels using majority voting"""
    tree = cKDTree(points)
    _, indices = tree.query(points, k=k)

    # Why: Vectorized majority vote
    neighbor_labels = labels[indices]
    smoothed = np.array([
        np.bincount(nl[nl >= 0]).argmax() if np.any(nl >= 0) else -1
        for nl in neighbor_labels
    ])

    return smoothed

def remove_small_clusters(labels, min_size=50):
    """Remove clusters smaller than threshold"""
    unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)

    # Why: Vectorized filtering
    small_clusters = unique_labels[counts < min_size]
    mask = np.isin(labels, small_clusters)
    labels[mask] = -1

    return labels

# Time to test step 8: Clean up the segmentation!
# smoothed = smooth_labels(downsampled, labels)
# cleaned = remove_small_clusters(smoothed, min_size=30)
# plot_cloud(downsampled, labels=cleaned, title="Cleaned Segmentation")

# Florent's Note: Post-processing is where you get production-quality results

#%% 09. EXPORT

def export_results(points, labels, classifications, output_path):
    """Export segmented cloud with classifications"""
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True)

    # Export full cloud with labels
    with open(output_path / 'segmented.ply', 'wb') as f:
        header = f"""ply
format binary_little_endian 1.0
element vertex {len(points)}
property float x
property float y
property float z
property int label
end_header
"""
        f.write(header.encode('ascii'))

        # Why: Structured array for efficient binary writing
        dtype = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('label', '<i4')])
        data = np.empty(len(points), dtype=dtype)
        data['x'] = points[:, 0]
        data['y'] = points[:, 1]
        data['z'] = points[:, 2]
        data['label'] = labels

        f.write(data.tobytes())

    # Export classification metadata
    import json
    with open(output_path / 'classifications.json', 'w') as f:
        json.dump(classifications, f, indent=2)

    return output_path

# Time to test step 9: Export your results!
# output_dir = export_results(downsampled, cleaned, classifications, './output')

# Branch: Stream to cloud storage for distributed processing

#%% 10. PIPELINE - Put it all together

def process_point_cloud(input_file, output_dir, voxel_size=0.02, eps=0.05):
    """Complete processing pipeline"""
    # Load
    points, colors = load_ply_simple(input_file)
    plot_cloud(points, colors, title="1. Input")

    # Normalize
    points, _, _ = normalize_cloud(points)

    # Downsample
    points, _ = voxel_downsample(points, voxel_size)
    plot_cloud(points, title=f"2. Downsampled ({len(points)} pts)")

    # Normals
    normals = estimate_normals_fast(points)

    # Features
    features = compute_geometric_features(points, normals)

    # Segment
    labels = dbscan_simple(points, eps=eps)
    plot_cloud(points, labels=labels, title="3. Segmented")

    # Classify
    classifications = classify_segments(points, labels, features)

    # Clean
    labels = smooth_labels(points, labels)
    labels = remove_small_clusters(labels, min_size=30)
    plot_cloud(points, labels=labels, title="4. Final Result")

    # Export
    output_path = export_results(points, labels, classifications, output_dir)

    return points, labels, classifications, output_path

# Time to test step 10: Run the complete pipeline!
# results = process_point_cloud('data/scan.ply', './output')

# Florent's Note: This is your production-ready template - adapt it!

# Branch: Parallelize with Dask for multi-file batch processing

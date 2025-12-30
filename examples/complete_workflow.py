#!/usr/bin/env python3
"""
Complete point cloud processing workflow demonstrating all capabilities
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data.io_manager import read_las_full, write_ply_binary
from filters.sor import statistical_outlier_removal
from filters.ground_seg import extract_ground_points
from filters.sampler import fps_sampling, project_predictions_to_full_resolution
from features.normals import estimate_normals
from features.geometric import compute_covariance_features
from features.height import compute_height_above_ground
from features.color import compute_vegetation_index
from segmentation.dbscan import dbscan_clustering
from segmentation.ml_classifier import train_random_forest, prepare_feature_vector
from utils.viz import init_viz_3d, plot_points_3d, show_viz

def generate_sample_data(n_points=10000):
    """Generate sample point cloud data"""
    np.random.seed(42)

    ground = np.random.randn(n_points // 2, 3) * [10, 10, 0.1]
    ground[:, 2] = 0

    objects = []
    for i in range(3):
        center = np.random.randn(3) * [5, 5, 0] + [0, 0, 2]
        obj = np.random.randn(n_points // 6, 3) * 0.5 + center
        objects.append(obj)

    points = np.vstack([ground] + objects)

    colors = np.random.rand(len(points), 3)

    return points, colors

def workflow_1_filtering_and_cleaning():
    """Workflow 1: Filtering and cleaning"""
    print("\n" + "="*60)
    print("WORKFLOW 1: FILTERING AND CLEANING")
    print("="*60)

    points, colors = generate_sample_data(10000)

    noise_indices = np.random.choice(len(points), 500, replace=False)
    points[noise_indices] += np.random.randn(500, 3) * 5

    print(f"Input: {len(points)} points")

    print("Removing outliers...")
    clean_points, inlier_mask = statistical_outlier_removal(points, k=20, std_multiplier=2.0)

    print(f"After filtering: {len(clean_points)} points ({len(clean_points)/len(points)*100:.1f}%)")

    return clean_points, colors[inlier_mask]

def workflow_2_ground_segmentation(points):
    """Workflow 2: Ground segmentation"""
    print("\n" + "="*60)
    print("WORKFLOW 2: GROUND SEGMENTATION")
    print("="*60)

    ground_points, non_ground_points, ground_mask = extract_ground_points(
        points,
        method='simple',
        grid_size=1.0,
        height_threshold=0.5
    )

    print(f"Ground points: {len(ground_points)}")
    print(f"Non-ground points: {len(non_ground_points)}")

    return ground_mask

def workflow_3_feature_extraction(points, ground_mask):
    """Workflow 3: Feature extraction"""
    print("\n" + "="*60)
    print("WORKFLOW 3: FEATURE EXTRACTION")
    print("="*60)

    print("Estimating normals...")
    normals = estimate_normals(points, k=20)

    print("Computing geometric features...")
    geometric_features = compute_covariance_features(points, k=20)

    print("Computing height above ground...")
    heights = compute_height_above_ground(points, ground_mask)

    print(f"Features computed for {len(points)} points")
    print(f"  - Normals: {normals.shape}")
    print(f"  - Geometric features: {geometric_features.shape}")
    print(f"  - Heights: {heights.shape}")

    return normals, geometric_features, heights

def workflow_4_segmentation(points):
    """Workflow 4: Segmentation"""
    print("\n" + "="*60)
    print("WORKFLOW 4: SEGMENTATION")
    print("="*60)

    print("Clustering with DBSCAN...")
    labels = dbscan_clustering(points, eps=0.5, min_samples=10)

    unique_labels = np.unique(labels[labels >= 0])
    print(f"Found {len(unique_labels)} clusters")

    for label in unique_labels:
        count = (labels == label).sum()
        print(f"  Cluster {label}: {count} points")

    return labels

def workflow_5_sampling_and_projection(points, labels):
    """Workflow 5: Downsampling and prediction projection"""
    print("\n" + "="*60)
    print("WORKFLOW 5: SAMPLING AND PROJECTION")
    print("="*60)

    print("Downsampling with FPS...")
    target_size = len(points) // 10
    sampled_points, sampled_indices = fps_sampling(points, target_size)

    print(f"Sampled from {len(points)} to {len(sampled_points)} points")

    sampled_labels = labels[sampled_indices]

    print("Projecting labels back to full resolution...")
    full_labels = project_predictions_to_full_resolution(
        sampled_points,
        sampled_labels.astype(np.float32),
        points,
        k=5
    )

    full_labels = np.round(full_labels).astype(np.int32)

    accuracy = (full_labels == labels).sum() / len(labels)
    print(f"Projection accuracy: {accuracy*100:.1f}%")

    return sampled_points, full_labels

def workflow_6_ml_classification(points, normals, geometric_features, heights):
    """Workflow 6: Machine learning classification"""
    print("\n" + "="*60)
    print("WORKFLOW 6: MACHINE LEARNING CLASSIFICATION")
    print("="*60)

    manual_labels = np.zeros(len(points), dtype=np.int32)
    manual_labels[heights < 0.5] = 0
    manual_labels[(heights >= 0.5) & (heights < 2.0)] = 1
    manual_labels[heights >= 2.0] = 2

    print("Preparing feature vector...")
    features = prepare_feature_vector(
        points,
        normals=normals,
        height_features=heights.reshape(-1, 1),
        geometric_features=geometric_features
    )

    print(f"Feature vector shape: {features.shape}")

    print("Training Random Forest classifier...")
    results = train_random_forest(features, manual_labels, n_estimators=50, test_size=0.3)

    print(f"Accuracy: {results['accuracy']*100:.1f}%")
    print("\nFeature importance:")
    for i, importance in enumerate(results['feature_importance'][:5]):
        print(f"  Feature {i}: {importance:.4f}")

    return results['classifier']

def workflow_7_export(points, colors, normals, labels, heights, geometric_features, output_path):
    """Workflow 7: Export with all features"""
    print("\n" + "="*60)
    print("WORKFLOW 7: EXPORT WITH PREDICTIONS")
    print("="*60)

    scalar_fields = {
        'labels': labels.astype(np.float32),
        'height': heights,
        'linearity': geometric_features[:, 0],
        'planarity': geometric_features[:, 1],
        'sphericity': geometric_features[:, 2]
    }

    print(f"Exporting to {output_path}...")
    write_ply_binary(
        output_path,
        points,
        colors=colors,
        normals=normals,
        scalar_fields=scalar_fields
    )

    print("Export complete!")
    print(f"Output contains:")
    print(f"  - {len(points)} points")
    print(f"  - XYZ coordinates")
    print(f"  - RGB colors")
    print(f"  - Normal vectors")
    print(f"  - {len(scalar_fields)} scalar fields")

def main():
    print("\n" + "="*60)
    print("COMPLETE POINT CLOUD PROCESSING WORKFLOW")
    print("="*60)

    print("\nGenerating sample data...")
    points, colors = generate_sample_data(10000)

    points, colors = workflow_1_filtering_and_cleaning()

    ground_mask = workflow_2_ground_segmentation(points)

    normals, geometric_features, heights = workflow_3_feature_extraction(points, ground_mask)

    labels = workflow_4_segmentation(points)

    sampled_points, full_labels = workflow_5_sampling_and_projection(points, labels)

    clf = workflow_6_ml_classification(points, normals, geometric_features, heights)

    output_path = "/tmp/processed_point_cloud.ply"
    workflow_7_export(points, colors, normals, full_labels, heights, geometric_features, output_path)

    print("\n" + "="*60)
    print("ALL WORKFLOWS COMPLETE!")
    print("="*60)

    print("\nVisualizing final results...")
    init_viz_3d()
    plot_points_3d(points, labels=full_labels, s=1)
    show_viz()

if __name__ == '__main__':
    main()

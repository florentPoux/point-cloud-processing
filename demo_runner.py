#%% 00. IMPORTS - Demo functions for GUI integration
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

#%% 01. DEMO FUNCTIONS - Create tutorial demonstrations for each system

def demo_gaussian_splatting():
    """Demo for System 02 - Gaussian Splatting"""
    print("Running Gaussian Splatting demo...")

    # Generate synthetic point cloud
    n_points = 1000
    points = np.random.randn(n_points, 3)
    colors = np.random.rand(n_points, 3)

    results = {
        'points': points,
        'colors': colors,
        'step_visualizations': []
    }

    return results

# Time to test step 1: Run Gaussian Splatting demo!

def demo_smart_pointcloud():
    """Demo for System 03 - Smart Point Cloud"""
    print("Running Smart Point Cloud demo...")

    n_points = 5000
    points = np.random.randn(n_points, 3) * 10
    colors = np.random.rand(n_points, 3)

    results = {
        'points': points,
        'colors': colors,
        'filtered_points': points[::2],
        'normals': np.random.randn(n_points//2, 3)
    }

    return results

# Time to test step 2: Run Smart Point Cloud demo!

def demo_geoai_mapping():
    """Demo for System 04 - GeoAI & Mapping"""
    print("Running GeoAI & Mapping demo...")

    # Generate terrain-like data
    x = np.linspace(-100, 100, 100)
    y = np.linspace(-100, 100, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(np.sqrt(X**2 + Y**2) / 10) * 5

    points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    colors = plt.cm.terrain((Z.ravel() - Z.min()) / (Z.max() - Z.min()))[:, :3]

    results = {
        'points': points,
        'colors': colors,
        'dem': Z
    }

    return results

# Time to test step 3: Run GeoAI demo!

def demo_deep_learning():
    """Demo for System 05 - Deep Learning 3D"""
    print("Running Deep Learning demo...")

    n_points = 2048
    points = np.random.randn(n_points, 3)
    labels = np.random.randint(0, 10, n_points)
    colors = plt.cm.tab10(labels / 10.0)[:, :3]

    results = {
        'points': points,
        'colors': colors,
        'predictions': labels
    }

    return results

# Time to test step 4: Run Deep Learning demo!

def demo_zero_shot():
    """Demo for System 06 - Zero-Shot Foundation Models"""
    print("Running Zero-Shot demo...")

    n_points = 3000
    points = np.random.randn(n_points, 3)
    embeddings = np.random.randn(n_points, 512)  # CLIP embeddings
    colors = np.random.rand(n_points, 3)

    results = {
        'points': points,
        'colors': colors,
        'embeddings': embeddings
    }

    return results

# Time to test step 5: Run Zero-Shot demo!

def demo_agentic_ai():
    """Demo for System 07 - Spatial Agentic AI"""
    print("Running Agentic AI demo...")

    n_points = 4000
    points = np.random.randn(n_points, 3) * 5
    colors = np.random.rand(n_points, 3)
    agent_path = np.cumsum(np.random.randn(100, 3) * 0.5, axis=0)

    results = {
        'points': points,
        'colors': colors,
        'agent_trajectory': agent_path
    }

    return results

# Time to test step 6: Run Agentic AI demo!

def demo_scan_to_bim():
    """Demo for System 08 - Scan-to-BIM"""
    print("Running Scan-to-BIM demo...")

    # Generate building-like structure
    n_walls = 4
    points_list = []
    colors_list = []

    for i in range(n_walls):
        angle = i * np.pi / 2
        wall_points = np.random.randn(500, 3)
        wall_points[:, 0] = wall_points[:, 0] * 0.1 + np.cos(angle) * 5
        wall_points[:, 1] = wall_points[:, 1] * 0.1 + np.sin(angle) * 5
        points_list.append(wall_points)
        colors_list.append(np.tile([0.7, 0.7, 0.7], (500, 1)))

    points = np.vstack(points_list)
    colors = np.vstack(colors_list)

    results = {
        'points': points,
        'colors': colors,
        'detected_planes': n_walls
    }

    return results

# Time to test step 7: Run Scan-to-BIM demo!

def demo_digital_twin():
    """Demo for System 09 - Digital Twin"""
    print("Running Digital Twin demo...")

    n_points = 3500
    points = np.random.randn(n_points, 3) * 8
    colors = np.random.rand(n_points, 3)

    # Simulate sensor data
    sensor_data = {
        'temperature': np.random.rand(10) * 30 + 10,
        'humidity': np.random.rand(10) * 100,
        'timestamps': np.arange(10)
    }

    results = {
        'points': points,
        'colors': colors,
        'sensor_data': sensor_data
    }

    return results

# Time to test step 8: Run Digital Twin demo!

def create_all_demos():
    """Create all demo datasets"""
    demos = {
        'System 02 - Gaussian Splatting': demo_gaussian_splatting,
        'System 03 - Smart Point Cloud': demo_smart_pointcloud,
        'System 04 - GeoAI & Mapping': demo_geoai_mapping,
        'System 05 - Deep Learning 3D': demo_deep_learning,
        'System 06 - Zero-Shot Foundation': demo_zero_shot,
        'System 07 - Spatial Agentic AI': demo_agentic_ai,
        'System 08 - Scan-to-BIM': demo_scan_to_bim,
        'System 09 - Digital Twin': demo_digital_twin
    }

    return demos

# Time to test step 9: Create all tutorial demos!

# Florent's Note: These demo functions provide synthetic data for GUI testing
# and tutorial execution without requiring large external datasets.

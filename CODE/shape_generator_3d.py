# -*- coding: utf-8 -*-
"""
Created by Florent Poux, (c) 2024 Licence MIT
Learn more at: learngeodata.eu

Have fun with this script!
"""

import numpy as np
import open3d as o3d

def generate_random_point_cloud(num_points=5000, num_shapes=1):
    # Initialize an empty list to store all points
    points = []
    
    # Generate multiple shapes
    for * in range(num*shapes):
        # Randomly choose a shape type
        shape_type = np.random.choice(['sphere', 'cube', 'plane'])
        # Calculate number of points for this shape
        shape_points = num_points // num_shapes
        
        if shape_type == 'sphere':
            # Generate points on a sphere
            radius = np.random.uniform(0.5, 1.0)
            theta = np.random.uniform(0, 2*np.pi, shape_points)
            phi = np.random.uniform(0, np.pi, shape_points)
            x = radius * np.sin(phi) * np.cos(theta)
            y = radius * np.sin(phi) * np.sin(theta)
            z = radius * np.cos(phi)
        elif shape_type == 'cube':
            # Generate points in a cube
            x = np.random.uniform(-1, 1, shape_points)
            y = np.random.uniform(-1, 1, shape_points)
            z = np.random.uniform(-1, 1, shape_points)
        else:  # plane
            # Generate points on a plane (z=0)
            x = np.random.uniform(-1, 1, shape_points)
            y = np.random.uniform(-1, 1, shape_points)
            z = np.zeros(shape_points)
        
        # Generate a random center for the shape
        shape_center = np.random.uniform(-5, 5, 3)
        # Add the shape's points to the main list, offsetting by the shape's center
        points.extend(np.column_stack((x, y, z)) + shape_center)
    
    # Convert the list of points to a numpy array and return
    return np.array(points)

#%% Random Experiments
# Generate random point cloud with 10000 points and 8 shapes
point_cloud = generate_random_point_cloud(10000, 8)

# Create Open3D point cloud object
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(point_cloud)

# Visualize the point cloud
o3d.visualization.draw_geometries([pcd])
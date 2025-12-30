import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import lsqr

def pairwise_registration(point_clouds, max_distance=0.5):
    """Compute pairwise registrations between all scans"""
    from registration.global_align import fast_global_registration
    from registration.icp import icp_point_to_point
    from features.normals import estimate_normals
    from registration.global_align import compute_fpfh

    n_clouds = len(point_clouds)

    pose_graph = []

    for i in range(n_clouds):
        for j in range(i + 1, n_clouds):
            source = point_clouds[i]
            target = point_clouds[j]

            source_normals = estimate_normals(source, k=20)
            target_normals = estimate_normals(target, k=20)

            source_fpfh = compute_fpfh(source, source_normals, k=30, radius=0.5)
            target_fpfh = compute_fpfh(target, target_normals, k=30, radius=0.5)

            transformation, fitness = fast_global_registration(
                source, target, source_fpfh, target_fpfh, distance_threshold=max_distance
            )

            if fitness > 0.1:
                transformation, error = icp_point_to_point(
                    source, target, max_iterations=30, initial_transform=transformation
                )

                pose_graph.append({
                    'source_id': i,
                    'target_id': j,
                    'transformation': transformation,
                    'fitness': fitness,
                    'error': error
                })

    return pose_graph

def build_pose_graph(pose_graph_edges, n_clouds):
    """Build pose graph from pairwise registrations"""
    pose_graph = {
        'nodes': list(range(n_clouds)),
        'edges': pose_graph_edges
    }

    return pose_graph

def optimize_pose_graph(pose_graph, n_iterations=10):
    """Global optimization of pose graph"""
    n_nodes = len(pose_graph['nodes'])
    edges = pose_graph['edges']

    poses = [np.eye(4) for _ in range(n_nodes)]

    for iteration in range(n_iterations):
        for edge in edges:
            source_id = edge['source_id']
            target_id = edge['target_id']
            transformation = edge['transformation']

            predicted_target_pose = np.dot(poses[source_id], transformation)

            error = compute_pose_error(poses[target_id], predicted_target_pose)

            step_size = 0.1
            update = step_size * error

            poses[target_id] = apply_pose_update(poses[target_id], update)

    return poses

def compute_pose_error(pose1, pose2):
    """Compute error between two poses"""
    relative = np.dot(np.linalg.inv(pose1), pose2)

    rotation_error = rotation_matrix_to_axis_angle(relative[:3, :3])
    translation_error = relative[:3, 3]

    error = np.hstack([rotation_error, translation_error])

    return error

def rotation_matrix_to_axis_angle(R):
    """Convert rotation matrix to axis-angle representation"""
    trace = np.trace(R)
    angle = np.arccos((trace - 1) / 2)

    if angle < 1e-6:
        return np.zeros(3)

    axis = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1]
    ]) / (2 * np.sin(angle))

    return axis * angle

def apply_pose_update(pose, update):
    """Apply incremental update to pose"""
    rotation_update = update[:3]
    translation_update = update[3:]

    angle = np.linalg.norm(rotation_update)

    if angle < 1e-6:
        R_update = np.eye(3)
    else:
        axis = rotation_update / angle
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])

        R_update = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)

    updated_pose = pose.copy()
    updated_pose[:3, :3] = np.dot(R_update, pose[:3, :3])
    updated_pose[:3, 3] = pose[:3, 3] + translation_update

    return updated_pose

def merge_point_clouds(point_clouds, poses):
    """Merge multiple point clouds using optimized poses"""
    from registration.global_align import transform_points

    merged_points = []

    for i, (cloud, pose) in enumerate(zip(point_clouds, poses)):
        transformed = transform_points(cloud, pose)
        merged_points.append(transformed)

    all_points = np.vstack(merged_points)

    return all_points

def sequential_registration(point_clouds):
    """Register point clouds sequentially"""
    from registration.icp import icp_point_to_point

    poses = [np.eye(4)]

    accumulated_transform = np.eye(4)

    for i in range(1, len(point_clouds)):
        source = point_clouds[i]
        target = point_clouds[i - 1]

        transformation, error = icp_point_to_point(source, target, max_iterations=50)

        accumulated_transform = np.dot(accumulated_transform, transformation)

        poses.append(accumulated_transform.copy())

    return poses

def loop_closure_detection(point_clouds, poses, distance_threshold=5.0):
    """Detect loop closures between non-consecutive scans"""
    from registration.global_align import compute_nearest_distances

    n_clouds = len(point_clouds)

    loop_closures = []

    for i in range(n_clouds):
        for j in range(i + 5, n_clouds):
            centroid_i = np.mean(point_clouds[i], axis=0)
            centroid_j = np.mean(point_clouds[j], axis=0)

            from registration.global_align import transform_points

            transformed_centroid_i = transform_points(centroid_i.reshape(1, -1), poses[i])[0]
            transformed_centroid_j = transform_points(centroid_j.reshape(1, -1), poses[j])[0]

            distance = np.linalg.norm(transformed_centroid_i - transformed_centroid_j)

            if distance < distance_threshold:
                loop_closures.append({
                    'source_id': i,
                    'target_id': j,
                    'distance': distance
                })

    return loop_closures

def refine_with_loop_closures(point_clouds, poses, loop_closures):
    """Refine poses using detected loop closures"""
    from registration.icp import icp_point_to_point

    refined_poses = [p.copy() for p in poses]

    for closure in loop_closures:
        source_id = closure['source_id']
        target_id = closure['target_id']

        source = point_clouds[source_id]
        target = point_clouds[target_id]

        transformation, error = icp_point_to_point(source, target, max_iterations=30)

        if error < 0.1:
            from registration.global_align import transform_points

            refined_poses[source_id] = np.dot(refined_poses[target_id], np.linalg.inv(transformation))

    return refined_poses

def compute_registration_quality(point_clouds, poses, sample_size=1000):
    """Evaluate overall registration quality"""
    from registration.global_align import transform_points
    from filters.sampler import random_sampling

    merged = []

    for cloud, pose in zip(point_clouds, poses):
        sampled, _, _ = random_sampling(cloud, min(sample_size, len(cloud)))
        transformed = transform_points(sampled, pose)
        merged.append(transformed)

    all_points = np.vstack(merged)

    from scipy.spatial import cKDTree
    tree = cKDTree(all_points)

    distances, _ = tree.query(all_points, k=2)
    neighbor_distances = distances[:, 1]

    mean_distance = neighbor_distances.mean()
    std_distance = neighbor_distances.std()

    return {
        'mean_neighbor_distance': mean_distance,
        'std_neighbor_distance': std_distance,
        'total_points': len(all_points)
    }

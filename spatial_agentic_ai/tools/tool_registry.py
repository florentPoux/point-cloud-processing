import numpy as np

def create_tool_registry():
    """Create registry of available tools"""
    tools = []

    tools.append({
        'name': 'downsample',
        'description': 'Downsample point cloud',
        'parameters': ['voxel_size'],
        'function': tool_downsample
    })

    tools.append({
        'name': 'filter_outliers',
        'description': 'Remove outlier points',
        'parameters': ['k', 'std_multiplier'],
        'function': tool_filter_outliers
    })

    tools.append({
        'name': 'estimate_normals',
        'description': 'Estimate point normals',
        'parameters': ['k'],
        'function': tool_estimate_normals
    })

    tools.append({
        'name': 'dbscan',
        'description': 'DBSCAN clustering',
        'parameters': ['eps', 'min_samples'],
        'function': tool_dbscan
    })

    tools.append({
        'name': 'extract_features',
        'description': 'Extract geometric features',
        'parameters': ['k_neighbors'],
        'function': tool_extract_features
    })

    tools.append({
        'name': 'icp',
        'description': 'ICP registration',
        'parameters': ['max_iterations'],
        'function': tool_icp
    })

    tools.append({
        'name': 'triangulate',
        'description': 'Create triangle mesh',
        'parameters': ['alpha'],
        'function': tool_triangulate
    })

    return tools

def tool_downsample(data, voxel_size=0.1):
    """Downsample tool"""
    from filters.sampler import voxel_grid_sampling

    points = data['points']
    downsampled, indices = voxel_grid_sampling(points, voxel_size=voxel_size)

    result = {'points': downsampled}

    if 'colors' in data and data['colors'] is not None:
        result['colors'] = data['colors'][indices]

    return result

def tool_filter_outliers(data, k=20, std_multiplier=2.0):
    """Filter outliers tool"""
    from filters.sor import statistical_outlier_removal

    points = data['points']
    filtered_points, mask = statistical_outlier_removal(points, k=k, std_multiplier=std_multiplier)

    result = {'points': filtered_points}

    if 'colors' in data and data['colors'] is not None:
        result['colors'] = data['colors'][mask]

    return result

def tool_estimate_normals(data, k=20):
    """Estimate normals tool"""
    from features.normals import estimate_normals

    points = data['points']
    normals = estimate_normals(points, k=k)

    result = data.copy()
    result['normals'] = normals

    return result

def tool_dbscan(data, eps=0.5, min_samples=10):
    """DBSCAN clustering tool"""
    from segmentation.dbscan import dbscan_clustering

    points = data['points']
    labels = dbscan_clustering(points, eps=eps, min_samples=min_samples)

    result = data.copy()
    result['labels'] = labels
    result['num_clusters'] = len(np.unique(labels[labels != -1]))

    return result

def tool_extract_features(data, k_neighbors=20):
    """Extract features tool"""
    from features.geometric import compute_geometric_features

    points = data['points']
    features = compute_geometric_features(points, k=k_neighbors)

    result = data.copy()
    result['features'] = features

    return result

def tool_icp(data, max_iterations=50):
    """ICP registration tool"""
    if 'source_points' not in data or 'target_points' not in data:
        return None

    from registration.icp import icp_point_to_point

    transformation, _ = icp_point_to_point(
        data['source_points'],
        data['target_points'],
        max_iterations=max_iterations
    )

    result = data.copy()
    result['transformation'] = transformation

    aligned = (transformation[:3, :3] @ data['source_points'].T).T + transformation[:3, 3]
    result['aligned_points'] = aligned

    return result

def tool_triangulate(data, alpha=0.1):
    """Triangulation tool"""
    from geoai_mapping.terrain.mesh_generation import generate_mesh_from_points

    points = data['points']
    vertices, faces = generate_mesh_from_points(points, alpha=alpha)

    result = data.copy()
    result['vertices'] = vertices
    result['faces'] = faces
    result['mesh_created'] = True

    return result

def execute_tool_chain(tools_to_execute, initial_data):
    """Execute chain of tools"""
    data = initial_data.copy()

    results = []

    for tool_spec in tools_to_execute:
        tool_name = tool_spec['tool']
        parameters = tool_spec['parameters']

        tool_function = find_tool_function(tool_name)

        if tool_function is not None:
            result = tool_function(data, **parameters)

            if result is not None:
                data.update(result)
                results.append({
                    'tool': tool_name,
                    'success': True,
                    'result': result
                })
            else:
                results.append({
                    'tool': tool_name,
                    'success': False
                })
        else:
            results.append({
                'tool': tool_name,
                'success': False,
                'error': 'tool_not_found'
            })

    return data, results

def find_tool_function(tool_name):
    """Find tool function by name"""
    tool_map = {
        'downsample': tool_downsample,
        'filter_outliers': tool_filter_outliers,
        'estimate_normals': tool_estimate_normals,
        'dbscan': tool_dbscan,
        'extract_features': tool_extract_features,
        'icp': tool_icp,
        'triangulate': tool_triangulate
    }

    return tool_map.get(tool_name)

def validate_tool_parameters(tool_spec, tool_definition):
    """Validate tool parameters"""
    required_params = tool_definition['parameters']
    provided_params = tool_spec['parameters'].keys()

    missing = set(required_params) - set(provided_params)

    return len(missing) == 0

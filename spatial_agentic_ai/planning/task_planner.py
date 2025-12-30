import numpy as np

def create_execution_plan(task, perception, reasoning, available_tools):
    """Create execution plan for task"""
    plan = {
        'task': task,
        'strategy': reasoning['suggested_approach'],
        'steps': [],
        'estimated_duration': 0,
        'success_probability': 0.0
    }

    task_type = reasoning['task_type']

    if task_type == 'segmentation':
        plan['steps'] = plan_segmentation_task(perception, available_tools)
    elif task_type == 'classification':
        plan['steps'] = plan_classification_task(perception, available_tools)
    elif task_type == 'registration':
        plan['steps'] = plan_registration_task(perception, available_tools)
    elif task_type == 'reconstruction':
        plan['steps'] = plan_reconstruction_task(perception, available_tools)
    else:
        plan['steps'] = plan_general_task(task, available_tools)

    plan['estimated_duration'] = len(plan['steps'])
    plan['success_probability'] = estimate_success_probability(plan, perception)

    return plan

def plan_segmentation_task(perception, available_tools):
    """Plan segmentation task"""
    steps = []

    if perception['num_points'] > 10000:
        steps.append({
            'tool': 'downsample',
            'parameters': {'voxel_size': 0.1},
            'reason': 'reduce_computational_cost'
        })

    steps.append({
        'tool': 'estimate_normals',
        'parameters': {'k': 20},
        'reason': 'compute_geometric_features'
    })

    steps.append({
        'tool': 'dbscan',
        'parameters': {'eps': 0.5, 'min_samples': 10},
        'reason': 'cluster_points'
    })

    steps.append({
        'tool': 'extract_features',
        'parameters': {},
        'reason': 'characterize_segments'
    })

    return steps

def plan_classification_task(perception, available_tools):
    """Plan classification task"""
    steps = []

    steps.append({
        'tool': 'extract_features',
        'parameters': {'k_neighbors': 20},
        'reason': 'compute_descriptors'
    })

    if has_tool('pointnet', available_tools):
        steps.append({
            'tool': 'pointnet',
            'parameters': {'num_points': 2048},
            'reason': 'deep_learning_classification'
        })
    else:
        steps.append({
            'tool': 'random_forest',
            'parameters': {'n_estimators': 100},
            'reason': 'traditional_classification'
        })

    return steps

def plan_registration_task(perception, available_tools):
    """Plan registration task"""
    steps = []

    steps.append({
        'tool': 'downsample',
        'parameters': {'voxel_size': 0.05},
        'reason': 'reduce_points_for_registration'
    })

    steps.append({
        'tool': 'compute_fpfh',
        'parameters': {'radius': 0.25},
        'reason': 'compute_descriptors'
    })

    steps.append({
        'tool': 'global_registration',
        'parameters': {},
        'reason': 'coarse_alignment'
    })

    steps.append({
        'tool': 'icp',
        'parameters': {'max_iterations': 50},
        'reason': 'fine_alignment'
    })

    return steps

def plan_reconstruction_task(perception, available_tools):
    """Plan reconstruction task"""
    steps = []

    steps.append({
        'tool': 'estimate_normals',
        'parameters': {'k': 20},
        'reason': 'required_for_meshing'
    })

    steps.append({
        'tool': 'triangulate',
        'parameters': {'alpha': 0.1},
        'reason': 'create_mesh'
    })

    if perception.get('has_colors'):
        steps.append({
            'tool': 'texture_mesh',
            'parameters': {},
            'reason': 'add_appearance'
        })

    return steps

def plan_general_task(task, available_tools):
    """Plan general task"""
    steps = []

    steps.append({
        'tool': 'analyze',
        'parameters': {},
        'reason': 'understand_data'
    })

    return steps

def has_tool(tool_name, available_tools):
    """Check if tool is available"""
    return any(tool['name'] == tool_name for tool in available_tools)

def estimate_success_probability(plan, perception):
    """Estimate probability of plan success"""
    base_probability = 0.8

    if len(plan['steps']) > 10:
        base_probability *= 0.9

    if perception['num_points'] < 1000:
        base_probability *= 0.7

    return base_probability

def adapt_plan_from_feedback(plan, feedback):
    """Adapt plan based on execution feedback"""
    if feedback['success_rate'] < 0.5:
        plan['steps'] = add_robustness_steps(plan['steps'])

    if feedback['execution_time'] > feedback['expected_time'] * 2:
        plan['steps'] = optimize_for_speed(plan['steps'])

    return plan

def add_robustness_steps(steps):
    """Add steps to increase robustness"""
    robust_steps = []

    robust_steps.append({
        'tool': 'filter_outliers',
        'parameters': {'k': 20, 'std_multiplier': 2.0},
        'reason': 'improve_data_quality'
    })

    robust_steps.extend(steps)

    return robust_steps

def optimize_for_speed(steps):
    """Optimize plan for speed"""
    optimized = []

    for step in steps:
        if step['tool'] != 'downsample':
            optimized.insert(0, {
                'tool': 'downsample',
                'parameters': {'voxel_size': 0.2},
                'reason': 'speed_optimization'
            })
            break

    optimized.extend(steps)

    return optimized

def decompose_complex_task(task_description):
    """Decompose complex task into subtasks"""
    subtasks = []

    if 'and' in task_description.lower():
        parts = task_description.lower().split('and')

        for part in parts:
            subtasks.append(part.strip())
    else:
        subtasks.append(task_description)

    return subtasks

def prioritize_subtasks(subtasks, constraints):
    """Prioritize subtasks based on dependencies"""
    prioritized = []

    dependency_order = {
        'filter': 0,
        'segment': 1,
        'classify': 2,
        'reconstruct': 3
    }

    for subtask in subtasks:
        priority = 10

        for keyword, value in dependency_order.items():
            if keyword in subtask.lower():
                priority = value
                break

        prioritized.append({
            'task': subtask,
            'priority': priority
        })

    prioritized.sort(key=lambda x: x['priority'])

    return [item['task'] for item in prioritized]

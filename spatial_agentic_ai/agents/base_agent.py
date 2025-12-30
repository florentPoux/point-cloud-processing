import numpy as np
from typing import Dict, List, Any

def create_agent_state(task_description, available_tools, memory_capacity=100):
    """Create initial agent state"""
    return {
        'task': task_description,
        'available_tools': available_tools,
        'memory': [],
        'memory_capacity': memory_capacity,
        'observations': [],
        'actions_taken': [],
        'plan': None,
        'current_step': 0,
        'status': 'initialized'
    }

def perceive_environment(points, colors=None, features=None):
    """Agent perception of 3D environment"""
    perception = {
        'num_points': len(points),
        'bounds': {
            'min': np.min(points, axis=0),
            'max': np.max(points, axis=0),
            'center': np.mean(points, axis=0),
            'extent': np.max(points, axis=0) - np.min(points, axis=0)
        },
        'density': len(points) / np.prod(np.max(points, axis=0) - np.min(points, axis=0)),
        'has_colors': colors is not None,
        'has_features': features is not None
    }

    if colors is not None:
        perception['color_stats'] = {
            'mean': np.mean(colors, axis=0),
            'std': np.std(colors, axis=0)
        }

    return perception

def reason_about_task(task_description, perception, available_tools):
    """Reason about task requirements"""
    reasoning = {
        'task_type': classify_task_type(task_description),
        'required_capabilities': extract_required_capabilities(task_description),
        'environment_suitable': check_environment_suitability(perception),
        'available_tool_names': [tool['name'] for tool in available_tools],
        'suggested_approach': None
    }

    if reasoning['task_type'] == 'segmentation':
        reasoning['suggested_approach'] = 'cluster -> classify -> refine'
    elif reasoning['task_type'] == 'classification':
        reasoning['suggested_approach'] = 'extract_features -> classify'
    elif reasoning['task_type'] == 'registration':
        reasoning['suggested_approach'] = 'align -> verify -> refine'
    elif reasoning['task_type'] == 'reconstruction':
        reasoning['suggested_approach'] = 'mesh -> texture -> optimize'

    return reasoning

def classify_task_type(task_description):
    """Classify task based on description"""
    task_lower = task_description.lower()

    if any(word in task_lower for word in ['segment', 'cluster', 'group']):
        return 'segmentation'
    elif any(word in task_lower for word in ['classify', 'categorize', 'identify']):
        return 'classification'
    elif any(word in task_lower for word in ['align', 'register', 'match']):
        return 'registration'
    elif any(word in task_lower for word in ['mesh', 'reconstruct', 'model']):
        return 'reconstruction'
    elif any(word in task_lower for word in ['detect', 'find', 'locate']):
        return 'detection'
    else:
        return 'general'

def extract_required_capabilities(task_description):
    """Extract required capabilities from task"""
    capabilities = []

    if 'segment' in task_description.lower():
        capabilities.append('clustering')
    if 'classify' in task_description.lower():
        capabilities.append('classification')
    if 'filter' in task_description.lower():
        capabilities.append('filtering')
    if 'ground' in task_description.lower():
        capabilities.append('ground_detection')
    if 'normal' in task_description.lower():
        capabilities.append('normal_estimation')

    return capabilities

def check_environment_suitability(perception):
    """Check if environment is suitable for task"""
    if perception['num_points'] < 100:
        return False

    extent = perception['bounds']['extent']
    if np.any(extent <= 0):
        return False

    return True

def select_tools_for_task(task_type, available_tools):
    """Select appropriate tools based on task"""
    tool_map = {
        'segmentation': ['dbscan', 'region_growing', 'extract_features'],
        'classification': ['extract_features', 'classify', 'pointnet'],
        'registration': ['icp', 'fpfh', 'global_align'],
        'reconstruction': ['triangulate', 'mesh_generate', 'texture_map'],
        'detection': ['dbscan', 'ransac', 'extract_features']
    }

    required_tool_names = tool_map.get(task_type, [])

    selected_tools = []
    for tool in available_tools:
        if tool['name'] in required_tool_names:
            selected_tools.append(tool)

    return selected_tools

def execute_action(action, agent_state, data):
    """Execute agent action"""
    tool_name = action['tool']
    parameters = action['parameters']

    for tool in agent_state['available_tools']:
        if tool['name'] == tool_name:
            result = tool['function'](data, **parameters)

            observation = {
                'action': action,
                'result': result,
                'success': result is not None,
                'step': agent_state['current_step']
            }

            agent_state['observations'].append(observation)
            agent_state['actions_taken'].append(action)
            agent_state['current_step'] += 1

            update_memory(agent_state, observation)

            return result

    return None

def update_memory(agent_state, observation):
    """Update agent memory with observation"""
    agent_state['memory'].append(observation)

    if len(agent_state['memory']) > agent_state['memory_capacity']:
        agent_state['memory'].pop(0)

def reflect_on_progress(agent_state, current_result):
    """Agent reflection on progress"""
    reflection = {
        'steps_taken': agent_state['current_step'],
        'successful_actions': sum(1 for obs in agent_state['observations'] if obs['success']),
        'failed_actions': sum(1 for obs in agent_state['observations'] if not obs['success']),
        'task_complete': False,
        'quality_score': None,
        'next_action': None
    }

    if current_result is not None:
        reflection['quality_score'] = evaluate_result_quality(current_result, agent_state['task'])

        if reflection['quality_score'] > 0.8:
            reflection['task_complete'] = True
        else:
            reflection['next_action'] = suggest_next_action(
                agent_state, current_result, reflection['quality_score']
            )

    return reflection

def evaluate_result_quality(result, task_description):
    """Evaluate quality of result"""
    if isinstance(result, dict):
        if 'quality_metrics' in result:
            return np.mean(list(result['quality_metrics'].values()))

        if 'accuracy' in result:
            return result['accuracy']

    if isinstance(result, np.ndarray):
        if len(result) > 0:
            return 0.7

    return 0.5

def suggest_next_action(agent_state, current_result, quality_score):
    """Suggest next action based on current state"""
    if quality_score < 0.5:
        return {'type': 'retry', 'reason': 'low_quality'}

    if agent_state['current_step'] >= 10:
        return {'type': 'terminate', 'reason': 'max_steps'}

    recent_actions = agent_state['actions_taken'][-3:]

    if len(recent_actions) >= 2:
        if recent_actions[-1] == recent_actions[-2]:
            return {'type': 'change_approach', 'reason': 'stuck'}

    return {'type': 'continue', 'reason': 'making_progress'}

def autonomous_processing_loop(task, points, colors, available_tools,
                                max_iterations=10):
    """Autonomous processing loop"""
    agent_state = create_agent_state(task, available_tools)

    perception = perceive_environment(points, colors)
    reasoning = reason_about_task(task, perception, available_tools)

    from spatial_agentic_ai.planning.task_planner import create_execution_plan

    plan = create_execution_plan(task, perception, reasoning, available_tools)
    agent_state['plan'] = plan

    data = {'points': points, 'colors': colors}
    result = None

    for iteration in range(max_iterations):
        if iteration < len(plan['steps']):
            action = plan['steps'][iteration]

            result = execute_action(action, agent_state, data)

            if result is not None:
                data.update(result)

            reflection = reflect_on_progress(agent_state, result)

            if reflection['task_complete']:
                agent_state['status'] = 'completed'
                break

            if reflection['next_action']['type'] == 'terminate':
                agent_state['status'] = 'terminated'
                break

    return result, agent_state

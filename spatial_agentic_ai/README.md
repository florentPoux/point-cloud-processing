# 🤖 Spatial Agentic AI
## System 07: Autonomous Agents for 3D Spatial Tasks

Production-ready library for building autonomous agents that reason about and manipulate 3D spatial data through planning, tool use, and iterative refinement.

Part of the **Spatial AI Architect Program** by [3D Geodata Academy](https://learngeodata.eu).

## 🎯 Key Features

- **Autonomous Agents**: Self-directed processing of 3D tasks
- **Task Planning**: Automatic decomposition and execution planning
- **Tool Execution**: Registry of spatial processing tools
- **Memory Systems**: Agent memory for learning from experience
- **Reasoning**: Task understanding and approach selection
- **Reflection**: Progress evaluation and adaptation

## 📦 Installation

```bash
# Install dependencies
pip install numpy scipy

# Clone and use
git clone https://github.com/florentPoux/point-cloud-processing.git
cd point-cloud-processing
```

## 🚀 Quick Start

### Autonomous Processing

```python
from spatial_agentic_ai.agents.base_agent import autonomous_processing_loop
from spatial_agentic_ai.tools.tool_registry import create_tool_registry

# Define task
task = "Segment this point cloud into meaningful objects"

# Create tool registry
tools = create_tool_registry()

# Run autonomous agent
result, agent_state = autonomous_processing_loop(
    task, points, colors, tools, max_iterations=10
)

print(f"Task status: {agent_state['status']}")
print(f"Steps taken: {agent_state['current_step']}")
```

## 📚 Modules

### Agents (`agents/`)
- `base_agent.py` - Core agent architecture with perception and reasoning

### Planning (`planning/`)
- `task_planner.py` - Task decomposition and execution planning

### Tools (`tools/`)
- `tool_registry.py` - Tool definitions and execution

## 🎓 Examples

### Basic Agent

```python
from spatial_agentic_ai.agents.base_agent import (
    create_agent_state,
    perceive_environment,
    reason_about_task
)

# Initialize agent
tools = create_tool_registry()
agent_state = create_agent_state(
    "Classify objects in scene",
    tools,
    memory_capacity=100
)

# Perceive environment
perception = perceive_environment(points, colors)

# Reason about task
reasoning = reason_about_task(
    "Classify objects in scene",
    perception,
    tools
)

print(f"Task type: {reasoning['task_type']}")
print(f"Approach: {reasoning['suggested_approach']}")
```

### Task Planning

```python
from spatial_agentic_ai.planning.task_planner import create_execution_plan

# Create plan
plan = create_execution_plan(task, perception, reasoning, tools)

print(f"Strategy: {plan['strategy']}")
print(f"Steps: {len(plan['steps'])}")
print(f"Success probability: {plan['success_probability']:.2f}")

for i, step in enumerate(plan['steps']):
    print(f"Step {i+1}: {step['tool']} - {step['reason']}")
```

### Tool Execution

```python
from spatial_agentic_ai.tools.tool_registry import execute_tool_chain

# Define tool chain
tool_chain = [
    {'tool': 'filter_outliers', 'parameters': {'k': 20, 'std_multiplier': 2.0}},
    {'tool': 'estimate_normals', 'parameters': {'k': 20}},
    {'tool': 'dbscan', 'parameters': {'eps': 0.5, 'min_samples': 10}}
]

# Execute chain
initial_data = {'points': points, 'colors': colors}
result_data, execution_results = execute_tool_chain(tool_chain, initial_data)

# Check results
for result in execution_results:
    print(f"{result['tool']}: {'✓' if result['success'] else '✗'}")
```

### Agent Reflection

```python
from spatial_agentic_ai.agents.base_agent import reflect_on_progress

# Agent reflects on progress
reflection = reflect_on_progress(agent_state, current_result)

print(f"Steps taken: {reflection['steps_taken']}")
print(f"Success rate: {reflection['successful_actions'] / reflection['steps_taken']:.2f}")
print(f"Quality score: {reflection['quality_score']:.2f}")

if reflection['task_complete']:
    print("Task completed successfully!")
else:
    print(f"Next action: {reflection['next_action']}")
```

## 📖 Resources

- [3D Geodata Academy](https://learngeodata.eu)
- [Autonomous Agents Tutorial](https://medium.com/@florentpoux)

## 👨‍💻 Author

**Florent Poux** - [3D Geodata Academy](https://learngeodata.eu)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

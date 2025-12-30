# 🌐 Complete Digital Twin
## System 09: Real-Time Digital Twin Integration

Production-ready library for creating and maintaining digital twins of physical spaces with real-time synchronization, simulation, and IoT integration.

Part of the **Spatial AI Architect Program** by [3D Geodata Academy](https://learngeodata.eu).

## 🎯 Key Features

- **Twin Management**: Create and manage digital twin instances
- **Real-Time Sync**: Synchronize with physical reality through sensors and scans
- **State Tracking**: Monitor and track twin state over time
- **Simulation**: Predict future states and simulate scenarios
- **Health Monitoring**: Compute twin health and detect anomalies
- **IoT Integration**: Connect with sensor networks
- **Predictive Maintenance**: Predict maintenance needs

## 📦 Installation

```bash
# Install dependencies
pip install numpy scipy

# Clone and use
git clone https://github.com/florentPoux/point-cloud-processing.git
cd point-cloud-processing
```

## 🚀 Quick Start

### Create and Manage Digital Twin

```python
from digital_twin.core.twin_manager import (
    create_digital_twin,
    update_twin_geometry,
    register_sensor,
    compute_twin_health
)

# Create digital twin
twin = create_digital_twin(
    name="Building A",
    physical_asset_id="BLD_001",
    initial_geometry={'points': points, 'colors': colors}
)

# Register sensors
register_sensor(twin, "TEMP_01", "temperature", location=[10, 20, 2])
register_sensor(twin, "OCCUP_01", "occupancy", location=[15, 25, 1.5])

# Compute health
health = compute_twin_health(twin)
print(f"Twin health: {health:.2f}")
```

## 📚 Modules

### Core (`core/`)
- `twin_manager.py` - Twin creation, updates, sensor management, health monitoring

### Synchronization (`synchronization/`)
- `realtime_sync.py` - Real-time sync from sensors and scans

### Simulation (`simulation/`)
- `predictor.py` - Future state prediction and scenario simulation

## 🎓 Examples

### Real-Time Synchronization

```python
from digital_twin.synchronization.realtime_sync import (
    sync_from_sensor_data,
    sync_from_new_scan
)

# Sync from sensor readings
sensor_data = [
    {'sensor_id': 'TEMP_01', 'value': 22.5},
    {'sensor_id': 'OCCUP_01', 'value': 15}
]
sync_from_sensor_data(twin, sensor_data, sync_manager)

# Sync from new scan
result = sync_from_new_scan(twin, new_scan_points, new_scan_colors)
print(f"Updated: {result['updated']}, Deviation: {result['deviation']['mean_distance']:.3f}m")
```

### Predict Future State

```python
from digital_twin.simulation.predictor import (
    predict_future_state,
    predict_maintenance_needs
)

# Predict state in 24 hours
future_state = predict_future_state(twin, prediction_horizon_hours=24)
print(f"Predicted health: {future_state['health']:.2f}")

# Predict maintenance
maintenance = predict_maintenance_needs(twin)
print(f"Maintenance urgency: {maintenance['urgency']}")
for rec in maintenance['recommendations']:
    print(f"  - {rec['type']}: {rec['urgency']}")
```

### Scenario Simulation

```python
from digital_twin.simulation.predictor import simulate_scenario

# Simulate temperature increase
scenario = {
    'temperature_change': 2.0,
    'occupancy_change': -5,
    'health_change_rate': -0.01
}

results = simulate_scenario(twin, scenario, duration_hours=48)

for result in results[::6]:  # Every 6 hours
    print(f"Hour {result['time_offset_hours']}: "
          f"Temp={result['state']['properties'].get('temperature', 0):.1f}°C, "
          f"Health={result['state']['health']:.2f}")
```

### Anomaly Detection

```python
from digital_twin.core.twin_manager import detect_anomalies

# Detect anomalies
anomalies = detect_anomalies(twin, threshold=0.5)

for anomaly in anomalies:
    print(f"{anomaly['type']}: {anomaly['severity']}")
```

## 📖 Resources

- [Digital Twin Concepts](https://en.wikipedia.org/wiki/Digital_twin)
- [3D Geodata Academy](https://learngeodata.eu)
- [Digital Twin Tutorial](https://medium.com/@florentpoux)

## 👨‍💻 Author

**Florent Poux** - [3D Geodata Academy](https://learngeodata.eu)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

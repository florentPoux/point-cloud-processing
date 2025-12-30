import numpy as np
from datetime import datetime

def create_digital_twin(name, physical_asset_id, initial_geometry=None):
    """Create digital twin instance"""
    twin = {
        'id': generate_twin_id(),
        'name': name,
        'physical_asset_id': physical_asset_id,
        'created_at': datetime.now().isoformat(),
        'last_updated': datetime.now().isoformat(),
        'geometry': initial_geometry if initial_geometry is not None else {},
        'state': {
            'status': 'initialized',
            'health': 1.0,
            'properties': {}
        },
        'sensors': [],
        'history': [],
        'metadata': {}
    }

    return twin

def generate_twin_id():
    """Generate unique twin ID"""
    import uuid
    return str(uuid.uuid4())

def update_twin_geometry(twin, new_geometry, source='scan', timestamp=None):
    """Update digital twin geometry"""
    if timestamp is None:
        timestamp = datetime.now().isoformat()

    twin['geometry'] = new_geometry
    twin['last_updated'] = timestamp

    twin['history'].append({
        'timestamp': timestamp,
        'event': 'geometry_update',
        'source': source,
        'metadata': {
            'num_points': len(new_geometry.get('points', [])),
            'has_colors': 'colors' in new_geometry
        }
    })

    return twin

def update_twin_state(twin, state_updates):
    """Update digital twin state"""
    timestamp = datetime.now().isoformat()

    twin['state'].update(state_updates)
    twin['last_updated'] = timestamp

    twin['history'].append({
        'timestamp': timestamp,
        'event': 'state_update',
        'updates': state_updates
    })

    return twin

def register_sensor(twin, sensor_id, sensor_type, location, metadata=None):
    """Register sensor with digital twin"""
    sensor = {
        'id': sensor_id,
        'type': sensor_type,
        'location': location,
        'status': 'active',
        'last_reading': None,
        'metadata': metadata if metadata is not None else {}
    }

    twin['sensors'].append(sensor)

    return sensor

def update_sensor_reading(twin, sensor_id, value, timestamp=None):
    """Update sensor reading"""
    if timestamp is None:
        timestamp = datetime.now().isoformat()

    for sensor in twin['sensors']:
        if sensor['id'] == sensor_id:
            sensor['last_reading'] = {
                'value': value,
                'timestamp': timestamp
            }

            twin['history'].append({
                'timestamp': timestamp,
                'event': 'sensor_reading',
                'sensor_id': sensor_id,
                'value': value
            })

            break

    return twin

def compute_twin_health(twin):
    """Compute digital twin health score"""
    health_factors = []

    time_since_update = (datetime.now() - datetime.fromisoformat(twin['last_updated'])).total_seconds()
    freshness_score = max(0, 1.0 - time_since_update / 86400)
    health_factors.append(freshness_score)

    active_sensors = sum(1 for s in twin['sensors'] if s['status'] == 'active')
    total_sensors = len(twin['sensors'])
    sensor_score = active_sensors / total_sensors if total_sensors > 0 else 1.0
    health_factors.append(sensor_score)

    has_geometry = len(twin['geometry']) > 0
    geometry_score = 1.0 if has_geometry else 0.5
    health_factors.append(geometry_score)

    health = np.mean(health_factors)

    return health

def query_twin_history(twin, event_type=None, start_time=None, end_time=None):
    """Query digital twin history"""
    filtered_history = twin['history']

    if event_type is not None:
        filtered_history = [h for h in filtered_history if h['event'] == event_type]

    if start_time is not None:
        filtered_history = [h for h in filtered_history
                          if h['timestamp'] >= start_time]

    if end_time is not None:
        filtered_history = [h for h in filtered_history
                          if h['timestamp'] <= end_time]

    return filtered_history

def compute_deviation_from_physical(twin_geometry, current_scan):
    """Compute deviation between twin and physical reality"""
    from scipy.spatial import cKDTree

    twin_points = twin_geometry.get('points', np.array([]))
    scan_points = current_scan.get('points', np.array([]))

    if len(twin_points) == 0 or len(scan_points) == 0:
        return None

    tree_scan = cKDTree(scan_points)
    distances, _ = tree_scan.query(twin_points, k=1)

    deviation = {
        'mean_distance': np.mean(distances),
        'max_distance': np.max(distances),
        'std_distance': np.std(distances),
        'percentile_95': np.percentile(distances, 95)
    }

    return deviation

def detect_anomalies(twin, threshold=0.5):
    """Detect anomalies in digital twin"""
    anomalies = []

    for sensor in twin['sensors']:
        if sensor['last_reading'] is None:
            anomalies.append({
                'type': 'missing_sensor_data',
                'sensor_id': sensor['id'],
                'severity': 'medium'
            })
            continue

        reading_time = datetime.fromisoformat(sensor['last_reading']['timestamp'])
        age = (datetime.now() - reading_time).total_seconds()

        if age > 3600:
            anomalies.append({
                'type': 'stale_sensor_data',
                'sensor_id': sensor['id'],
                'age_hours': age / 3600,
                'severity': 'low'
            })

    health = compute_twin_health(twin)

    if health < threshold:
        anomalies.append({
            'type': 'low_health_score',
            'health': health,
            'severity': 'high'
        })

    return anomalies

def export_twin_snapshot(twin, filepath):
    """Export digital twin snapshot"""
    import json

    snapshot = {
        'id': twin['id'],
        'name': twin['name'],
        'timestamp': datetime.now().isoformat(),
        'state': twin['state'],
        'geometry_summary': {
            'has_points': 'points' in twin['geometry'],
            'num_points': len(twin['geometry'].get('points', [])),
            'has_colors': 'colors' in twin['geometry']
        },
        'num_sensors': len(twin['sensors']),
        'health': compute_twin_health(twin)
    }

    with open(filepath, 'w') as f:
        json.dump(snapshot, f, indent=2)

    return snapshot

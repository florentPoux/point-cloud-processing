import numpy as np
from datetime import datetime, timedelta

def predict_future_state(twin, prediction_horizon_hours=24):
    """Predict future state of digital twin"""
    current_state = twin['state'].copy()

    sensor_trends = analyze_sensor_trends(twin)

    predicted_state = current_state.copy()

    for sensor_id, trend in sensor_trends.items():
        if trend['direction'] != 'stable':
            future_value = trend['current_value'] + trend['rate'] * prediction_horizon_hours

            predicted_state['properties'][f'{sensor_id}_predicted'] = future_value

    health = twin['state'].get('health', 1.0)

    if any(t['direction'] == 'decreasing' for t in sensor_trends.values()):
        predicted_health = max(0, health - 0.1 * prediction_horizon_hours / 24)
    else:
        predicted_health = min(1.0, health + 0.05 * prediction_horizon_hours / 24)

    predicted_state['health'] = predicted_health

    return predicted_state

def analyze_sensor_trends(twin):
    """Analyze sensor reading trends"""
    trends = {}

    for sensor in twin['sensors']:
        if sensor['last_reading'] is None:
            continue

        sensor_history = [h for h in twin['history']
                         if h['event'] == 'sensor_reading' and h['sensor_id'] == sensor['id']]

        if len(sensor_history) < 2:
            trends[sensor['id']] = {
                'direction': 'unknown',
                'rate': 0,
                'current_value': sensor['last_reading']['value']
            }
            continue

        values = [h['value'] for h in sensor_history[-10:]]
        timestamps = [datetime.fromisoformat(h['timestamp']) for h in sensor_history[-10:]]

        time_diffs = [(timestamps[i+1] - timestamps[i]).total_seconds() / 3600
                     for i in range(len(timestamps) - 1)]

        value_diffs = np.diff(values)

        if len(time_diffs) > 0:
            rate = np.mean(value_diffs) / np.mean(time_diffs)

            if abs(rate) < 0.01:
                direction = 'stable'
            elif rate > 0:
                direction = 'increasing'
            else:
                direction = 'decreasing'

            trends[sensor['id']] = {
                'direction': direction,
                'rate': rate,
                'current_value': values[-1]
            }

    return trends

def simulate_scenario(twin, scenario_parameters, duration_hours=24):
    """Simulate scenario on digital twin"""
    simulation_results = []

    time_steps = int(duration_hours)

    current_state = twin['state'].copy()

    for hour in range(time_steps):
        if 'temperature_change' in scenario_parameters:
            current_temp = current_state['properties'].get('temperature', 20)
            current_temp += scenario_parameters['temperature_change']
            current_state['properties']['temperature'] = current_temp

        if 'occupancy_change' in scenario_parameters:
            current_occ = current_state['properties'].get('occupancy', 0)
            current_occ += scenario_parameters['occupancy_change']
            current_state['properties']['occupancy'] = max(0, current_occ)

        health = current_state.get('health', 1.0)
        health_change = scenario_parameters.get('health_change_rate', 0)
        current_state['health'] = np.clip(health + health_change, 0, 1)

        simulation_results.append({
            'time_offset_hours': hour,
            'state': current_state.copy()
        })

    return simulation_results

def predict_maintenance_needs(twin, failure_threshold=0.3):
    """Predict maintenance requirements"""
    health_trend = analyze_health_trend(twin)

    if health_trend['current'] < failure_threshold:
        urgency = 'immediate'
        estimated_time_to_failure = 0
    elif health_trend['rate'] < 0:
        estimated_time_to_failure = (health_trend['current'] - failure_threshold) / abs(health_trend['rate'])
        if estimated_time_to_failure < 24:
            urgency = 'urgent'
        elif estimated_time_to_failure < 168:
            urgency = 'scheduled'
        else:
            urgency = 'routine'
    else:
        urgency = 'none'
        estimated_time_to_failure = None

    recommendations = []

    if urgency != 'none':
        recommendations.append({
            'type': 'inspection',
            'urgency': urgency,
            'reason': 'declining_health'
        })

    sensor_anomalies = [s for s in twin['sensors']
                       if s['last_reading'] is None or s['status'] != 'active']

    for sensor in sensor_anomalies:
        recommendations.append({
            'type': 'sensor_maintenance',
            'sensor_id': sensor['id'],
            'urgency': 'scheduled',
            'reason': 'sensor_malfunction'
        })

    return {
        'urgency': urgency,
        'estimated_time_to_failure_hours': estimated_time_to_failure,
        'recommendations': recommendations
    }

def analyze_health_trend(twin):
    """Analyze digital twin health trend"""
    health_history = [h for h in twin['history']
                     if 'health' in h.get('updates', {})]

    if len(health_history) < 2:
        current_health = twin['state'].get('health', 1.0)
        return {
            'current': current_health,
            'rate': 0,
            'trend': 'stable'
        }

    health_values = [h['updates']['health'] for h in health_history]
    timestamps = [datetime.fromisoformat(h['timestamp']) for h in health_history]

    time_diffs = [(timestamps[i+1] - timestamps[i]).total_seconds() / 3600
                 for i in range(len(timestamps) - 1)]

    health_diffs = np.diff(health_values)

    rate = np.mean(health_diffs) / np.mean(time_diffs) if len(time_diffs) > 0 else 0

    if abs(rate) < 0.001:
        trend = 'stable'
    elif rate > 0:
        trend = 'improving'
    else:
        trend = 'declining'

    return {
        'current': health_values[-1],
        'rate': rate,
        'trend': trend
    }

import numpy as np
from datetime import datetime, timedelta

def create_sync_manager(update_interval=60, deviation_threshold=0.1):
    """Create synchronization manager"""
    return {
        'update_interval': update_interval,
        'deviation_threshold': deviation_threshold,
        'last_sync': None,
        'sync_history': [],
        'pending_updates': []
    }

def sync_from_sensor_data(twin, sensor_data, sync_manager):
    """Synchronize twin from sensor readings"""
    timestamp = datetime.now()

    updates_applied = 0

    for sensor_reading in sensor_data:
        sensor_id = sensor_reading['sensor_id']
        value = sensor_reading['value']

        from digital_twin.core.twin_manager import update_sensor_reading

        update_sensor_reading(twin, sensor_id, value, timestamp.isoformat())
        updates_applied += 1

    sync_manager['last_sync'] = timestamp.isoformat()
    sync_manager['sync_history'].append({
        'timestamp': timestamp.isoformat(),
        'updates_applied': updates_applied,
        'data_source': 'sensors'
    })

    return twin

def sync_from_new_scan(twin, scan_points, scan_colors=None):
    """Synchronize twin from new point cloud scan"""
    from digital_twin.core.twin_manager import (
        update_twin_geometry,
        compute_deviation_from_physical
    )

    deviation = compute_deviation_from_physical(
        twin['geometry'],
        {'points': scan_points, 'colors': scan_colors}
    )

    if deviation is not None and deviation['mean_distance'] > 0.1:
        new_geometry = {'points': scan_points}
        if scan_colors is not None:
            new_geometry['colors'] = scan_colors

        update_twin_geometry(twin, new_geometry, source='realtime_scan')

        return {
            'updated': True,
            'deviation': deviation,
            'reason': 'significant_change'
        }

    return {
        'updated': False,
        'deviation': deviation,
        'reason': 'no_significant_change'
    }

def incremental_update(twin, point_delta, update_type='add'):
    """Apply incremental update to twin geometry"""
    current_points = twin['geometry'].get('points', np.array([]))

    if update_type == 'add':
        updated_points = np.vstack([current_points, point_delta]) if len(current_points) > 0 else point_delta

    elif update_type == 'remove':
        from scipy.spatial import cKDTree

        if len(current_points) == 0:
            return twin

        tree = cKDTree(point_delta)
        distances, _ = tree.query(current_points, k=1)

        keep_mask = distances > 0.05
        updated_points = current_points[keep_mask]

    else:
        updated_points = current_points

    from digital_twin.core.twin_manager import update_twin_geometry

    update_twin_geometry(twin, {'points': updated_points}, source='incremental')

    return twin

def schedule_periodic_sync(twin, sync_function, interval_seconds=300):
    """Schedule periodic synchronization"""
    schedule = {
        'twin_id': twin['id'],
        'sync_function': sync_function,
        'interval': interval_seconds,
        'last_execution': None,
        'next_execution': (datetime.now() + timedelta(seconds=interval_seconds)).isoformat(),
        'enabled': True
    }

    return schedule

def execute_scheduled_sync(twin, schedule, data_source):
    """Execute scheduled synchronization"""
    if not schedule['enabled']:
        return None

    current_time = datetime.now()

    if schedule['last_execution'] is not None:
        last_time = datetime.fromisoformat(schedule['last_execution'])
        elapsed = (current_time - last_time).total_seconds()

        if elapsed < schedule['interval']:
            return None

    result = schedule['sync_function'](twin, data_source)

    schedule['last_execution'] = current_time.isoformat()
    schedule['next_execution'] = (current_time + timedelta(seconds=schedule['interval'])).isoformat()

    return result

def merge_twin_states(twin1, twin2, strategy='latest'):
    """Merge two digital twin states"""
    if strategy == 'latest':
        time1 = datetime.fromisoformat(twin1['last_updated'])
        time2 = datetime.fromisoformat(twin2['last_updated'])

        return twin1 if time1 > time2 else twin2

    elif strategy == 'merge_geometry':
        merged = twin1.copy()

        points1 = twin1['geometry'].get('points', np.array([]))
        points2 = twin2['geometry'].get('points', np.array([]))

        if len(points1) > 0 and len(points2) > 0:
            merged_points = np.vstack([points1, points2])

            from filters.sampler import voxel_grid_sampling

            merged_points, _ = voxel_grid_sampling(merged_points, voxel_size=0.05)

            merged['geometry']['points'] = merged_points

        return merged

    return twin1

def conflict_resolution(twin, conflicting_update, resolution_strategy='twin_priority'):
    """Resolve conflicts in updates"""
    if resolution_strategy == 'twin_priority':
        return twin

    elif resolution_strategy == 'update_priority':
        from digital_twin.core.twin_manager import update_twin_state

        return update_twin_state(twin, conflicting_update)

    elif resolution_strategy == 'timestamp':
        update_time = conflicting_update.get('timestamp', datetime.min.isoformat())
        twin_time = twin['last_updated']

        if update_time > twin_time:
            from digital_twin.core.twin_manager import update_twin_state

            return update_twin_state(twin, conflicting_update)

    return twin

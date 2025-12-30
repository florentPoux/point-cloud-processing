import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter

def detect_changes_c2c(points_t1, points_t2, threshold=0.5):
    """Cloud-to-cloud change detection"""
    tree_t2 = cKDTree(points_t2[:, :2])

    distances, indices = tree_t2.query(points_t1[:, :2], k=1)

    z_diff = points_t1[:, 2] - points_t2[indices, 2]

    change_magnitude = np.abs(z_diff)
    change_mask = change_magnitude > threshold

    change_type = np.zeros(len(points_t1), dtype=int)
    change_type[z_diff > threshold] = 1
    change_type[z_diff < -threshold] = -1

    return change_mask, change_magnitude, change_type

def detect_changes_dem(dem_t1, dem_t2, threshold=0.5):
    """DEM-to-DEM change detection"""
    diff = dem_t2 - dem_t1

    change_magnitude = np.abs(diff)
    change_mask = change_magnitude > threshold

    change_type = np.zeros_like(diff, dtype=int)
    change_type[diff > threshold] = 1
    change_type[diff < -threshold] = -1

    return change_mask, change_magnitude, change_type

def detect_building_changes(dsm_t1, dtm_t1, dsm_t2, dtm_t2, height_threshold=3.0, change_threshold=1.0):
    """Detect building construction/demolition"""
    ndsm_t1 = dsm_t1 - dtm_t1
    ndsm_t2 = dsm_t2 - dtm_t2

    buildings_t1 = ndsm_t1 > height_threshold
    buildings_t2 = ndsm_t2 > height_threshold

    new_buildings = buildings_t2 & ~buildings_t1
    demolished_buildings = buildings_t1 & ~buildings_t2

    height_change = ndsm_t2 - ndsm_t1
    modified_buildings = (np.abs(height_change) > change_threshold) & buildings_t1 & buildings_t2

    return {
        'new': new_buildings,
        'demolished': demolished_buildings,
        'modified': modified_buildings,
        'height_change': height_change
    }

def detect_vegetation_changes(chm_t1, chm_t2, threshold=2.0):
    """Detect vegetation growth/loss"""
    diff = chm_t2 - chm_t1

    growth_mask = diff > threshold
    loss_mask = diff < -threshold
    stable_mask = np.abs(diff) <= threshold

    return {
        'growth': growth_mask,
        'loss': loss_mask,
        'stable': stable_mask,
        'change_magnitude': diff
    }

def compute_volumetric_change(dem_t1, dem_t2, resolution, mask=None):
    """Compute volumetric change between DEMs"""
    diff = dem_t2 - dem_t1

    if mask is not None:
        diff = diff * mask

    cell_area = resolution ** 2

    volume_gain = np.sum(diff[diff > 0]) * cell_area
    volume_loss = np.sum(np.abs(diff[diff < 0])) * cell_area
    net_volume = np.sum(diff) * cell_area

    return {
        'volume_gain_m3': volume_gain,
        'volume_loss_m3': volume_loss,
        'net_volume_m3': net_volume,
        'area_m2': np.sum(mask) * cell_area if mask is not None else diff.size * cell_area
    }

def filter_noise_changes(change_mask, change_magnitude, min_area_pixels=10, gaussian_sigma=1.0):
    """Filter noise from change detection results"""
    from skimage import morphology, measure

    filtered = gaussian_filter(change_magnitude.astype(float), sigma=gaussian_sigma)

    change_mask_filtered = filtered > np.mean(filtered)

    change_mask_filtered = morphology.remove_small_objects(change_mask_filtered, min_size=min_area_pixels)

    return change_mask_filtered

def segment_change_regions(change_mask, change_magnitude):
    """Segment change detection into individual regions"""
    from skimage import measure

    labeled = measure.label(change_mask)

    regions = []

    for region in measure.regionprops(labeled):
        region_mask = labeled == region.label

        regions.append({
            'area_pixels': region.area,
            'centroid': region.centroid,
            'bbox': region.bbox,
            'mean_change': np.mean(change_magnitude[region_mask]),
            'max_change': np.max(change_magnitude[region_mask]),
            'mask': region_mask
        })

    return regions

def track_objects_temporal(regions_t1, regions_t2, max_distance=50):
    """Track objects across time periods"""
    if len(regions_t1) == 0 or len(regions_t2) == 0:
        return []

    centroids_t1 = np.array([r['centroid'] for r in regions_t1])
    centroids_t2 = np.array([r['centroid'] for r in regions_t2])

    tree = cKDTree(centroids_t2)
    distances, indices = tree.query(centroids_t1, k=1)

    matches = []

    for i, (dist, idx) in enumerate(zip(distances, indices)):
        if dist < max_distance:
            matches.append({
                'region_t1': regions_t1[i],
                'region_t2': regions_t2[idx],
                'distance': dist,
                'area_change': regions_t2[idx]['area_pixels'] - regions_t1[i]['area_pixels']
            })

    return matches

def create_change_map_rgb(change_type, change_magnitude):
    """Create RGB visualization of changes"""
    rgb = np.zeros((*change_type.shape, 3), dtype=np.uint8)

    gain_mask = change_type == 1
    loss_mask = change_type == -1

    rgb[gain_mask, 1] = np.clip(change_magnitude[gain_mask] * 50, 0, 255).astype(np.uint8)

    rgb[loss_mask, 0] = np.clip(change_magnitude[loss_mask] * 50, 0, 255).astype(np.uint8)

    rgb[~(gain_mask | loss_mask)] = [128, 128, 128]

    return rgb

def compute_change_statistics(change_mask, change_magnitude, change_type, resolution):
    """Compute statistics for detected changes"""
    cell_area = resolution ** 2

    total_changed_area = np.sum(change_mask) * cell_area
    gain_area = np.sum(change_type == 1) * cell_area
    loss_area = np.sum(change_type == -1) * cell_area

    mean_change = np.mean(change_magnitude[change_mask]) if np.any(change_mask) else 0
    max_change = np.max(change_magnitude[change_mask]) if np.any(change_mask) else 0

    return {
        'total_changed_area_m2': total_changed_area,
        'gain_area_m2': gain_area,
        'loss_area_m2': loss_area,
        'mean_change_m': mean_change,
        'max_change_m': max_change,
        'percentage_changed': (np.sum(change_mask) / change_mask.size) * 100
    }

def multi_temporal_analysis(dems, timestamps, threshold=0.5):
    """Analyze changes across multiple time periods"""
    n_epochs = len(dems)

    changes = []

    for i in range(n_epochs - 1):
        change_mask, change_mag, change_type = detect_changes_dem(
            dems[i], dems[i + 1], threshold
        )

        changes.append({
            'from_time': timestamps[i],
            'to_time': timestamps[i + 1],
            'change_mask': change_mask,
            'change_magnitude': change_mag,
            'change_type': change_type
        })

    cumulative_change = np.zeros_like(dems[0])

    for dem in dems:
        cumulative_change += dem

    cumulative_change = cumulative_change / n_epochs

    return changes, cumulative_change

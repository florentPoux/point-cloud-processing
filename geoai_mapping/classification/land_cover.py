import numpy as np
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

def extract_features_for_classification(points, colors, normals, k_neighbors=20):
    """Extract geometric and color features for land cover classification"""
    features_list = []

    if colors is not None:
        features_list.append(colors)

        hsv = rgb_to_hsv(colors)
        features_list.append(hsv)

    if normals is not None:
        features_list.append(normals)
        features_list.append(normals[:, 2:3])

    tree = cKDTree(points)
    distances, indices = tree.query(points, k=k_neighbors + 1)

    z_values = points[indices[:, 1:], 2]
    z_std = np.std(z_values, axis=1, keepdims=True)
    z_range = np.max(z_values, axis=1, keepdims=True) - np.min(z_values, axis=1, keepdims=True)

    features_list.append(z_std)
    features_list.append(z_range)

    local_heights = points[:, 2:3] - np.min(z_values, axis=1, keepdims=True)
    features_list.append(local_heights)

    features = np.hstack(features_list)

    return features

def rgb_to_hsv(rgb):
    """Convert RGB to HSV color space"""
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]

    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    diff = max_c - min_c

    h = np.zeros_like(r)

    mask = diff > 0

    r_mask = (max_c == r) & mask
    h[r_mask] = (60 * ((g[r_mask] - b[r_mask]) / diff[r_mask]) + 360) % 360

    g_mask = (max_c == g) & mask
    h[g_mask] = (60 * ((b[g_mask] - r[g_mask]) / diff[g_mask]) + 120) % 360

    b_mask = (max_c == b) & mask
    h[b_mask] = (60 * ((r[b_mask] - g[b_mask]) / diff[b_mask]) + 240) % 360

    s = np.zeros_like(r)
    s[max_c > 0] = diff[max_c > 0] / max_c[max_c > 0]

    v = max_c

    return np.column_stack([h / 360.0, s, v])

def train_land_cover_classifier(features, labels, n_estimators=100):
    """Train Random Forest classifier for land cover"""
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=20,
        min_samples_split=50,
        min_samples_leaf=20,
        n_jobs=-1,
        random_state=42
    )

    classifier.fit(features_scaled, labels)

    return classifier, scaler

def predict_land_cover(features, classifier, scaler):
    """Predict land cover classes for points"""
    features_scaled = scaler.transform(features)

    predictions = classifier.predict(features_scaled)
    probabilities = classifier.predict_proba(features_scaled)

    confidence = np.max(probabilities, axis=1)

    return predictions, confidence

def classify_vegetation_density(points, chm_values, threshold_low=2.0, threshold_high=5.0):
    """Classify vegetation density from Canopy Height Model"""
    labels = np.zeros(len(points), dtype=int)

    labels[chm_values < threshold_low] = 0
    labels[(chm_values >= threshold_low) & (chm_values < threshold_high)] = 1
    labels[chm_values >= threshold_high] = 2

    return labels

def classify_urban_features(points, features, building_height_min=3.0):
    """Classify urban features (ground, buildings, vegetation)"""
    z_range = features[:, -2]
    height = points[:, 2]

    labels = np.zeros(len(points), dtype=int)

    labels[z_range < 0.1] = 0

    labels[(z_range >= 0.1) & (height > building_height_min)] = 1

    labels[(z_range >= 0.5) & (height < building_height_min)] = 2

    return labels

def segment_road_surface(points, colors, normals, slope_threshold=10, intensity_threshold=0.7):
    """Segment road surfaces based on geometry and appearance"""
    from scipy.ndimage import gaussian_filter1d

    if normals is not None:
        flatness = normals[:, 2]
        flat_mask = flatness > np.cos(np.deg2rad(slope_threshold))
    else:
        flat_mask = np.ones(len(points), dtype=bool)

    if colors is not None:
        gray = np.mean(colors, axis=1)
        gray_mask = gray > intensity_threshold
    else:
        gray_mask = np.ones(len(points), dtype=bool)

    road_mask = flat_mask & gray_mask

    return road_mask

def classify_water_bodies(points, colors, z_variation_threshold=0.2, blue_threshold=0.4):
    """Detect water bodies based on low elevation variation and color"""
    tree = cKDTree(points[:, :2])
    distances, indices = tree.query(points[:, :2], k=20)

    z_std = np.std(points[indices, 2], axis=1)

    flat_mask = z_std < z_variation_threshold

    if colors is not None:
        blue_ratio = colors[:, 2] / (colors[:, 0] + colors[:, 1] + colors[:, 2] + 1e-6)
        blue_mask = blue_ratio > blue_threshold
    else:
        blue_mask = np.zeros(len(points), dtype=bool)

    water_mask = flat_mask & blue_mask

    return water_mask

def create_land_cover_map(points, labels, resolution, num_classes=5):
    """Create rasterized land cover map from classified points"""
    min_coords = np.min(points[:, :2], axis=0)
    max_coords = np.max(points[:, :2], axis=0)

    grid_width = int(np.ceil((max_coords[0] - min_coords[0]) / resolution))
    grid_height = int(np.ceil((max_coords[1] - min_coords[1]) / resolution))

    land_cover_map = np.zeros((grid_height, grid_width), dtype=int)
    counts = np.zeros((grid_height, grid_width, num_classes))

    for i, point in enumerate(points):
        px = int((point[0] - min_coords[0]) / resolution)
        py = int((point[1] - min_coords[1]) / resolution)

        if 0 <= px < grid_width and 0 <= py < grid_height:
            label = labels[i]
            if 0 <= label < num_classes:
                counts[py, px, label] += 1

    for i in range(grid_height):
        for j in range(grid_width):
            if np.sum(counts[i, j]) > 0:
                land_cover_map[i, j] = np.argmax(counts[i, j])

    origin = min_coords
    geotransform = [origin[0], resolution, 0, origin[1] + grid_height * resolution, 0, -resolution]

    return land_cover_map, geotransform

def compute_class_statistics(labels, class_names=None):
    """Compute statistics for classified points"""
    unique, counts = np.unique(labels, return_counts=True)

    total = len(labels)

    stats = []

    for label, count in zip(unique, counts):
        class_name = class_names[int(label)] if class_names and int(label) < len(class_names) else f"Class {int(label)}"

        stats.append({
            'label': int(label),
            'name': class_name,
            'count': int(count),
            'percentage': count / total * 100
        })

    return stats

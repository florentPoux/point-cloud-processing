import numpy as np
from scipy.spatial import cKDTree

def rgb_to_hsv(rgb):
    """Convert RGB to HSV color space"""
    rgb = np.clip(rgb, 0, 1)

    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]

    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    diff = max_c - min_c

    h = np.zeros(len(rgb))
    s = np.zeros(len(rgb))
    v = max_c

    mask = diff != 0

    s[mask] = diff[mask] / max_c[mask]

    r_mask = (max_c == r) & mask
    g_mask = (max_c == g) & mask
    b_mask = (max_c == b) & mask

    h[r_mask] = (60 * ((g[r_mask] - b[r_mask]) / diff[r_mask]) + 360) % 360
    h[g_mask] = (60 * ((b[g_mask] - r[g_mask]) / diff[g_mask]) + 120) % 360
    h[b_mask] = (60 * ((r[b_mask] - g[b_mask]) / diff[b_mask]) + 240) % 360

    return np.column_stack([h / 360.0, s, v])

def hsv_to_rgb(hsv):
    """Convert HSV to RGB color space"""
    h, s, v = hsv[:, 0] * 360, hsv[:, 1], hsv[:, 2]

    c = v * s
    x = c * (1 - np.abs(((h / 60) % 2) - 1))
    m = v - c

    rgb = np.zeros_like(hsv)

    mask0 = (h >= 0) & (h < 60)
    mask1 = (h >= 60) & (h < 120)
    mask2 = (h >= 120) & (h < 180)
    mask3 = (h >= 180) & (h < 240)
    mask4 = (h >= 240) & (h < 300)
    mask5 = (h >= 300) & (h < 360)

    rgb[mask0] = np.column_stack([c[mask0], x[mask0], np.zeros(mask0.sum())])
    rgb[mask1] = np.column_stack([x[mask1], c[mask1], np.zeros(mask1.sum())])
    rgb[mask2] = np.column_stack([np.zeros(mask2.sum()), c[mask2], x[mask2]])
    rgb[mask3] = np.column_stack([np.zeros(mask3.sum()), x[mask3], c[mask3]])
    rgb[mask4] = np.column_stack([x[mask4], np.zeros(mask4.sum()), c[mask4]])
    rgb[mask5] = np.column_stack([c[mask5], np.zeros(mask5.sum()), x[mask5]])

    rgb += m[:, np.newaxis]

    return rgb

def compute_vegetation_index(rgb):
    """Compute ExG (Excess Green) vegetation index"""
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]

    total = r + g + b + 1e-6

    r_norm = r / total
    g_norm = g / total
    b_norm = b / total

    exg = 2 * g_norm - r_norm - b_norm

    return exg

def compute_normalized_rgb(rgb):
    """Compute normalized RGB values"""
    total = rgb.sum(axis=1, keepdims=True) + 1e-6
    normalized = rgb / total

    return normalized

def compute_color_variance(points, colors, k=20):
    """Compute local color variance"""
    tree = cKDTree(points)
    _, indices = tree.query(points, k=k)

    variance = np.zeros(len(points))

    for i in range(len(points)):
        neighborhood_colors = colors[indices[i]]

        variance[i] = neighborhood_colors.var(axis=0).mean()

    return variance

def compute_dominant_color(points, colors, k=20):
    """Compute dominant color in local neighborhood"""
    tree = cKDTree(points)
    _, indices = tree.query(points, k=k)

    dominant_colors = np.zeros_like(colors)

    for i in range(len(points)):
        neighborhood_colors = colors[indices[i]]

        dominant_colors[i] = neighborhood_colors.mean(axis=0)

    return dominant_colors

def normalize_intensity(intensity, method='minmax'):
    """Normalize intensity values"""
    if method == 'minmax':
        min_val = intensity.min()
        max_val = intensity.max()

        if max_val == min_val:
            return np.zeros_like(intensity)

        normalized = (intensity - min_val) / (max_val - min_val)

    elif method == 'zscore':
        mean_val = intensity.mean()
        std_val = intensity.std()

        if std_val == 0:
            return np.zeros_like(intensity)

        normalized = (intensity - mean_val) / std_val

    elif method == 'percentile':
        p1 = np.percentile(intensity, 1)
        p99 = np.percentile(intensity, 99)

        normalized = np.clip((intensity - p1) / (p99 - p1 + 1e-6), 0, 1)

    return normalized

def detect_reflective_surfaces(intensity, threshold_percentile=95):
    """Detect highly reflective surfaces"""
    threshold = np.percentile(intensity, threshold_percentile)

    is_reflective = intensity > threshold

    return is_reflective

def compute_color_gradient(points, colors, k=20):
    """Compute color gradient magnitude"""
    tree = cKDTree(points)
    _, indices = tree.query(points, k=k)

    gradients = np.zeros(len(points))

    for i in range(len(points)):
        neighborhood_points = points[indices[i]]
        neighborhood_colors = colors[indices[i]]

        if len(neighborhood_points) < 3:
            gradients[i] = 0
            continue

        color_diff = neighborhood_colors - colors[i]
        color_magnitude = np.linalg.norm(color_diff, axis=1)

        spatial_dist = np.linalg.norm(neighborhood_points - points[i], axis=1) + 1e-6

        gradient = color_magnitude / spatial_dist

        gradients[i] = gradient.mean()

    return gradients

def classify_by_color(colors, color_classes):
    """Classify points based on color similarity to predefined classes"""
    labels = np.zeros(len(colors), dtype=np.int32)

    for i in range(len(colors)):
        distances = []

        for class_color in color_classes:
            dist = np.linalg.norm(colors[i] - class_color)
            distances.append(dist)

        labels[i] = np.argmin(distances)

    return labels

def enhance_color_contrast(colors, factor=1.5):
    """Enhance color contrast"""
    hsv = rgb_to_hsv(colors)

    hsv[:, 1] = np.clip(hsv[:, 1] * factor, 0, 1)

    enhanced_rgb = hsv_to_rgb(hsv)

    return enhanced_rgb

def compute_brightness(colors):
    """Compute brightness from RGB"""
    brightness = 0.299 * colors[:, 0] + 0.587 * colors[:, 1] + 0.114 * colors[:, 2]

    return brightness

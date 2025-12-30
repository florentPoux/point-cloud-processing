import numpy as np
import torch

def load_sam_model(model_type='vit_h', checkpoint_path=None, device='cuda'):
    """Load Segment Anything Model"""
    try:
        from segment_anything import sam_model_registry, SamPredictor

        if checkpoint_path is None:
            return None, None

        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam.to(device)

        predictor = SamPredictor(sam)

        return sam, predictor
    except ImportError:
        return None, None

def segment_point_cloud_with_sam(points, colors, sam_predictor,
                                  point_prompts=None, box_prompts=None,
                                  image_size=1024):
    """Segment point cloud using SAM on rendered views"""
    rendered_view = render_point_cloud_to_image(points, colors, image_size)

    sam_predictor.set_image(rendered_view)

    if point_prompts is not None:
        pixel_prompts = project_3d_to_pixel(point_prompts, points, image_size)

        masks, scores, logits = sam_predictor.predict(
            point_coords=pixel_prompts,
            point_labels=np.ones(len(pixel_prompts)),
            multimask_output=True
        )

    elif box_prompts is not None:
        pixel_boxes = project_boxes_to_pixel(box_prompts, points, image_size)

        masks, scores, logits = sam_predictor.predict(
            box=pixel_boxes[0],
            multimask_output=True
        )

    else:
        return None, None

    best_mask_idx = np.argmax(scores)
    best_mask = masks[best_mask_idx]

    point_labels = project_mask_to_points(best_mask, points, image_size)

    return point_labels, scores[best_mask_idx]

def render_point_cloud_to_image(points, colors, image_size=1024):
    """Render point cloud to RGB image for SAM"""
    min_coords = np.min(points[:, :2], axis=0)
    max_coords = np.max(points[:, :2], axis=0)
    extent = max_coords - min_coords
    scale = image_size / np.max(extent)

    image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    z_buffer = np.full((image_size, image_size), -np.inf)

    pixel_coords = ((points[:, :2] - min_coords) * scale).astype(int)
    pixel_coords = np.clip(pixel_coords, 0, image_size - 1)

    for i, (px, py) in enumerate(pixel_coords):
        if points[i, 2] > z_buffer[py, px]:
            z_buffer[py, px] = points[i, 2]
            if colors is not None:
                image[py, px] = (colors[i] * 255).astype(np.uint8)

    return image

def project_3d_to_pixel(points_3d, all_points, image_size):
    """Project 3D points to pixel coordinates"""
    min_coords = np.min(all_points[:, :2], axis=0)
    max_coords = np.max(all_points[:, :2], axis=0)
    extent = max_coords - min_coords
    scale = image_size / np.max(extent)

    pixel_coords = ((points_3d[:, :2] - min_coords) * scale).astype(int)
    pixel_coords = np.clip(pixel_coords, 0, image_size - 1)

    return pixel_coords

def project_boxes_to_pixel(boxes_3d, all_points, image_size):
    """Project 3D bounding boxes to pixel coordinates"""
    pixel_boxes = []

    for box in boxes_3d:
        min_3d, max_3d = box[:3], box[3:]

        min_pixel = project_3d_to_pixel(min_3d.reshape(1, 3), all_points, image_size)[0]
        max_pixel = project_3d_to_pixel(max_3d.reshape(1, 3), all_points, image_size)[0]

        pixel_boxes.append([min_pixel[0], min_pixel[1], max_pixel[0], max_pixel[1]])

    return np.array(pixel_boxes)

def project_mask_to_points(mask, points, image_size):
    """Project 2D mask back to 3D points"""
    min_coords = np.min(points[:, :2], axis=0)
    max_coords = np.max(points[:, :2], axis=0)
    extent = max_coords - min_coords
    scale = image_size / np.max(extent)

    pixel_coords = ((points[:, :2] - min_coords) * scale).astype(int)
    pixel_coords = np.clip(pixel_coords, 0, image_size - 1)

    point_labels = np.zeros(len(points), dtype=bool)

    for i, (px, py) in enumerate(pixel_coords):
        point_labels[i] = mask[py, px]

    return point_labels

def auto_segment_point_cloud(points, colors, sam_predictor, grid_size=16,
                             image_size=1024, min_mask_region_area=100):
    """Automatic segmentation using grid prompts"""
    from segment_anything import SamAutomaticMaskGenerator

    rendered_view = render_point_cloud_to_image(points, colors, image_size)

    mask_generator = SamAutomaticMaskGenerator(
        sam_predictor.model,
        points_per_side=grid_size,
        min_mask_region_area=min_mask_region_area
    )

    masks = mask_generator.generate(rendered_view)

    point_segments = []

    for mask_data in masks:
        mask = mask_data['segmentation']
        point_labels = project_mask_to_points(mask, points, image_size)

        point_segments.append({
            'points': points[point_labels],
            'colors': colors[point_labels] if colors is not None else None,
            'mask': point_labels,
            'area': mask_data['area'],
            'stability_score': mask_data['stability_score']
        })

    return point_segments

def interactive_segment_with_clicks(points, colors, sam_predictor,
                                     positive_clicks, negative_clicks=None,
                                     image_size=1024):
    """Interactive segmentation with positive/negative clicks"""
    rendered_view = render_point_cloud_to_image(points, colors, image_size)
    sam_predictor.set_image(rendered_view)

    positive_pixels = project_3d_to_pixel(positive_clicks, points, image_size)

    if negative_clicks is not None:
        negative_pixels = project_3d_to_pixel(negative_clicks, points, image_size)

        all_pixels = np.vstack([positive_pixels, negative_pixels])
        labels = np.array([1] * len(positive_pixels) + [0] * len(negative_pixels))
    else:
        all_pixels = positive_pixels
        labels = np.ones(len(positive_pixels))

    masks, scores, logits = sam_predictor.predict(
        point_coords=all_pixels,
        point_labels=labels,
        multimask_output=True
    )

    best_idx = np.argmax(scores)
    point_labels = project_mask_to_points(masks[best_idx], points, image_size)

    return point_labels, scores[best_idx], masks

def refine_segmentation_boundary(points, labels, sam_predictor, image_size=1024):
    """Refine segmentation boundaries using SAM"""
    boundary_points = find_boundary_points(points, labels)

    if len(boundary_points) == 0:
        return labels

    refined_labels = labels.copy()

    for boundary_point in boundary_points:
        local_region = get_local_region(points, boundary_point, radius=0.5)

        refined_mask, _ = interactive_segment_with_clicks(
            points[local_region],
            None,
            sam_predictor,
            boundary_point.reshape(1, 3),
            image_size=image_size
        )

        refined_labels[local_region] = refined_mask

    return refined_labels

def find_boundary_points(points, labels, k=10):
    """Find points on segmentation boundaries"""
    from scipy.spatial import cKDTree

    tree = cKDTree(points)

    boundary_indices = []

    for i, point in enumerate(points):
        _, indices = tree.query(point, k=k)
        neighbor_labels = labels[indices]

        if not np.all(neighbor_labels == labels[i]):
            boundary_indices.append(i)

    return points[boundary_indices]

def get_local_region(points, center_point, radius):
    """Get points within radius of center"""
    distances = np.linalg.norm(points - center_point, axis=1)
    return distances < radius

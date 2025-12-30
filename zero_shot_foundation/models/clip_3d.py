import numpy as np
import torch
import torch.nn.functional as F

def load_clip_model(model_name='ViT-B/32', device='cuda'):
    """Load CLIP model for text-image encoding"""
    try:
        import clip
        model, preprocess = clip.load(model_name, device=device)
        return model, preprocess
    except ImportError:
        return None, None

def encode_text_prompts(prompts, clip_model, device='cuda'):
    """Encode text prompts using CLIP"""
    import clip

    tokens = clip.tokenize(prompts).to(device)

    with torch.no_grad():
        text_features = clip_model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return text_features.cpu().numpy()

def render_point_cloud_views(points, colors=None, num_views=8, image_size=224):
    """Render point cloud from multiple viewpoints"""
    from PIL import Image

    views = []

    for i in range(num_views):
        angle = 2 * np.pi * i / num_views

        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])

        rotated_points = points @ rotation_matrix.T

        rendered_image = render_orthographic_view(
            rotated_points, colors, image_size
        )

        views.append(rendered_image)

    return views

def render_orthographic_view(points, colors=None, image_size=224):
    """Render orthographic view of point cloud"""
    min_coords = np.min(points[:, :2], axis=0)
    max_coords = np.max(points[:, :2], axis=0)
    extent = max_coords - min_coords
    scale = image_size / np.max(extent)

    image = np.ones((image_size, image_size, 3), dtype=np.uint8) * 255
    z_buffer = np.full((image_size, image_size), -np.inf)

    pixel_coords = ((points[:, :2] - min_coords) * scale).astype(int)
    pixel_coords = np.clip(pixel_coords, 0, image_size - 1)

    for i, (px, py) in enumerate(pixel_coords):
        if points[i, 2] > z_buffer[py, px]:
            z_buffer[py, px] = points[i, 2]
            if colors is not None:
                image[py, px] = (colors[i] * 255).astype(np.uint8)
            else:
                intensity = int((points[i, 2] - points[:, 2].min()) /
                               (points[:, 2].max() - points[:, 2].min() + 1e-6) * 255)
                image[py, px] = [intensity, intensity, intensity]

    return image

def encode_point_cloud_with_clip(points, colors, clip_model, preprocess,
                                  num_views=8, device='cuda'):
    """Encode point cloud as multi-view CLIP features"""
    views = render_point_cloud_views(points, colors, num_views=num_views)

    view_features = []

    for view in views:
        from PIL import Image
        pil_image = Image.fromarray(view)
        preprocessed = preprocess(pil_image).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = clip_model.encode_image(preprocessed)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        view_features.append(image_features.cpu().numpy())

    aggregated_features = np.mean(view_features, axis=0)
    aggregated_features = aggregated_features / np.linalg.norm(aggregated_features)

    return aggregated_features

def zero_shot_classify_point_cloud(points, colors, class_prompts, clip_model,
                                    preprocess, num_views=8, device='cuda'):
    """Zero-shot classification using CLIP"""
    text_features = encode_text_prompts(class_prompts, clip_model, device)

    point_cloud_features = encode_point_cloud_with_clip(
        points, colors, clip_model, preprocess, num_views, device
    )

    similarities = point_cloud_features @ text_features.T
    predicted_class = np.argmax(similarities)
    confidence = similarities[0, predicted_class]

    return predicted_class, confidence, similarities[0]

def create_text_templates(class_names, templates=None):
    """Create text prompts from class names using templates"""
    if templates is None:
        templates = [
            "a photo of a {}",
            "a 3D model of a {}",
            "a point cloud of a {}",
            "a rendering of a {}"
        ]

    prompts = []

    for class_name in class_names:
        class_prompts = [template.format(class_name) for template in templates]
        prompts.append(class_prompts)

    return prompts

def encode_text_with_ensembling(class_names, clip_model, templates=None, device='cuda'):
    """Encode text with prompt ensembling"""
    prompt_lists = create_text_templates(class_names, templates)

    class_features = []

    for class_prompts in prompt_lists:
        features = encode_text_prompts(class_prompts, clip_model, device)
        mean_feature = np.mean(features, axis=0, keepdims=True)
        mean_feature = mean_feature / np.linalg.norm(mean_feature)
        class_features.append(mean_feature)

    return np.vstack(class_features)

def compute_clip_similarity_matrix(points_list, colors_list, clip_model,
                                    preprocess, device='cuda'):
    """Compute pairwise CLIP similarity between point clouds"""
    features_list = []

    for points, colors in zip(points_list, colors_list):
        features = encode_point_cloud_with_clip(
            points, colors, clip_model, preprocess, num_views=8, device=device
        )
        features_list.append(features)

    features_matrix = np.vstack(features_list)

    similarity_matrix = features_matrix @ features_matrix.T

    return similarity_matrix

def retrieve_similar_point_clouds(query_points, query_colors, database_points,
                                   database_colors, clip_model, preprocess,
                                   top_k=5, device='cuda'):
    """Retrieve similar point clouds using CLIP features"""
    query_features = encode_point_cloud_with_clip(
        query_points, query_colors, clip_model, preprocess, device=device
    )

    database_features = []
    for points, colors in zip(database_points, database_colors):
        features = encode_point_cloud_with_clip(
            points, colors, clip_model, preprocess, device=device
        )
        database_features.append(features)

    database_features = np.vstack(database_features)

    similarities = query_features @ database_features.T
    top_indices = np.argsort(similarities[0])[::-1][:top_k]
    top_similarities = similarities[0, top_indices]

    return top_indices, top_similarities

def project_clip_features_to_points(points, colors, clip_features, projection_dim=128):
    """Project CLIP features to point-level features"""
    num_points = len(points)

    point_features = np.tile(clip_features, (num_points, 1))

    if colors is not None:
        point_features = np.concatenate([point_features, colors], axis=1)

    from sklearn.decomposition import PCA

    pca = PCA(n_components=projection_dim)
    projected_features = pca.fit_transform(point_features)

    return projected_features

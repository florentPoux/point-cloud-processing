import numpy as np
from scipy.spatial import cKDTree

def zero_shot_point_segmentation(points, colors, class_prompts, clip_model,
                                  preprocess, num_views=8, device='cuda'):
    """Zero-shot semantic segmentation using CLIP"""
    from zero_shot_foundation.models.clip_3d import (
        encode_text_with_ensembling,
        render_point_cloud_views
    )

    text_features = encode_text_with_ensembling(
        class_prompts, clip_model, device=device
    )

    views = render_point_cloud_views(points, colors, num_views=num_views)

    from PIL import Image
    import torch

    view_predictions = []

    for view_image in views:
        pil_image = Image.fromarray(view_image)
        preprocessed = preprocess(pil_image).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = clip_model.encode_image(preprocessed)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        similarities = image_features.cpu().numpy() @ text_features.T
        predictions = np.argmax(similarities, axis=1)

        view_predictions.append(predictions[0])

    consensus_prediction = np.bincount(view_predictions).argmax()

    point_labels = np.full(len(points), consensus_prediction)

    return point_labels

def zero_shot_instance_segmentation(points, colors, object_prompts,
                                     clip_model, preprocess, sam_predictor=None,
                                     device='cuda'):
    """Zero-shot instance segmentation combining CLIP and SAM"""
    if sam_predictor is None:
        return zero_shot_point_segmentation(
            points, colors, object_prompts, clip_model, preprocess, device=device
        )

    from zero_shot_foundation.models.sam_3d import auto_segment_point_cloud
    from zero_shot_foundation.models.clip_3d import encode_text_with_ensembling

    segments = auto_segment_point_cloud(points, colors, sam_predictor)

    text_features = encode_text_with_ensembling(
        object_prompts, clip_model, device=device
    )

    instance_labels = np.zeros(len(points), dtype=int)
    instance_id = 1

    for segment in segments:
        segment_points = segment['points']
        segment_colors = segment['colors']

        from zero_shot_foundation.models.clip_3d import encode_point_cloud_with_clip

        segment_features = encode_point_cloud_with_clip(
            segment_points, segment_colors, clip_model, preprocess, device=device
        )

        similarities = segment_features @ text_features.T
        predicted_class = np.argmax(similarities)

        if similarities[0, predicted_class] > 0.2:
            instance_labels[segment['mask']] = instance_id
            instance_id += 1

    return instance_labels

def open_vocabulary_search(query_text, points_database, colors_database,
                           clip_model, preprocess, top_k=10, device='cuda'):
    """Open-vocabulary search in point cloud database"""
    from zero_shot_foundation.models.clip_3d import (
        encode_text_prompts,
        encode_point_cloud_with_clip
    )

    query_features = encode_text_prompts([query_text], clip_model, device)

    database_features = []
    for points, colors in zip(points_database, colors_database):
        features = encode_point_cloud_with_clip(
            points, colors, clip_model, preprocess, device=device
        )
        database_features.append(features)

    database_features = np.vstack(database_features)

    similarities = query_features @ database_features.T
    top_indices = np.argsort(similarities[0])[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            'index': idx,
            'similarity': similarities[0, idx],
            'points': points_database[idx],
            'colors': colors_database[idx]
        })

    return results

def semantic_scene_graph(points, colors, entity_prompts, relation_prompts,
                         clip_model, preprocess, device='cuda'):
    """Generate semantic scene graph from point cloud"""
    from zero_shot_foundation.models.clip_3d import encode_text_with_ensembling

    entity_features = encode_text_with_ensembling(entity_prompts, clip_model, device=device)
    relation_features = encode_text_with_ensembling(relation_prompts, clip_model, device=device)

    from segmentation.dbscan import dbscan_clustering

    cluster_labels = dbscan_clustering(points, eps=0.5, min_samples=10)

    entities = []
    for cluster_id in np.unique(cluster_labels):
        if cluster_id == -1:
            continue

        cluster_mask = cluster_labels == cluster_id
        cluster_points = points[cluster_mask]
        cluster_colors = colors[cluster_mask] if colors is not None else None

        from zero_shot_foundation.models.clip_3d import encode_point_cloud_with_clip

        cluster_features = encode_point_cloud_with_clip(
            cluster_points, cluster_colors, clip_model, preprocess, device=device
        )

        similarities = cluster_features @ entity_features.T
        entity_type = np.argmax(similarities)

        entities.append({
            'id': cluster_id,
            'type': entity_prompts[entity_type],
            'points': cluster_points,
            'centroid': np.mean(cluster_points, axis=0)
        })

    scene_graph = {'entities': entities, 'relations': []}

    for i, entity1 in enumerate(entities):
        for entity2 in entities[i+1:]:
            spatial_relation = compute_spatial_relation(
                entity1['centroid'], entity2['centroid']
            )

            scene_graph['relations'].append({
                'subject': entity1['id'],
                'object': entity2['id'],
                'relation': spatial_relation
            })

    return scene_graph

def compute_spatial_relation(centroid1, centroid2):
    """Compute spatial relationship between two objects"""
    diff = centroid2 - centroid1

    if abs(diff[2]) > np.linalg.norm(diff[:2]):
        return 'above' if diff[2] > 0 else 'below'
    else:
        distance = np.linalg.norm(diff[:2])
        if distance < 1.0:
            return 'next_to'
        else:
            return 'far_from'

def visual_question_answering_3d(points, colors, question, clip_model,
                                  preprocess, answer_candidates, device='cuda'):
    """Answer questions about 3D scenes using CLIP"""
    from zero_shot_foundation.models.clip_3d import (
        encode_text_prompts,
        encode_point_cloud_with_clip
    )

    scene_features = encode_point_cloud_with_clip(
        points, colors, clip_model, preprocess, device=device
    )

    question_prompts = [f"{question} Answer: {answer}" for answer in answer_candidates]

    answer_features = encode_text_prompts(question_prompts, clip_model, device)

    similarities = scene_features @ answer_features.T
    best_answer_idx = np.argmax(similarities)

    return answer_candidates[best_answer_idx], similarities[0, best_answer_idx]

def progressive_refinement_prediction(points, colors, class_prompts,
                                      clip_model, preprocess, num_iterations=3,
                                      device='cuda'):
    """Progressively refine predictions using multi-scale features"""
    from zero_shot_foundation.models.clip_3d import encode_text_with_ensembling
    from filters.sampler import fps_sampling

    text_features = encode_text_with_ensembling(class_prompts, clip_model, device=device)

    predictions = np.zeros(len(points), dtype=int)
    confidences = np.zeros(len(points))

    for iteration in range(num_iterations):
        sample_size = min(len(points), 2048 * (iteration + 1))

        sampled_points, sampled_indices = fps_sampling(points, target_size=sample_size)
        sampled_colors = colors[sampled_indices] if colors is not None else None

        from zero_shot_foundation.models.clip_3d import encode_point_cloud_with_clip

        features = encode_point_cloud_with_clip(
            sampled_points, sampled_colors, clip_model, preprocess, device=device
        )

        similarities = features @ text_features.T
        prediction = np.argmax(similarities)
        confidence = similarities[0, prediction]

        predictions[sampled_indices] = prediction
        confidences[sampled_indices] = confidence

    tree = cKDTree(points)

    for i, point in enumerate(points):
        if confidences[i] == 0:
            _, indices = tree.query(point, k=5)
            neighbor_predictions = predictions[indices]
            neighbor_confidences = confidences[indices]

            valid_neighbors = neighbor_confidences > 0
            if np.any(valid_neighbors):
                predictions[i] = np.bincount(
                    neighbor_predictions[valid_neighbors]
                ).argmax()

    return predictions, confidences

import torch
import numpy as np

def predict_classification(points, model_forward_fn, model_components, device='cuda', num_points=2048):
    """Run classification inference on point cloud"""
    from deep_learning_3d.data.augmentation import sample_fixed_points, normalize_points

    if len(points) != num_points:
        points = sample_fixed_points(points, num_points)

    points = normalize_points(points, method='center')

    points_tensor = torch.from_numpy(points).float().unsqueeze(0).to(device)
    points_tensor = points_tensor.transpose(1, 2)

    for component in model_components.values():
        if isinstance(component, torch.nn.Module):
            component.eval()
        elif isinstance(component, dict):
            for sub_component in component.values():
                if isinstance(sub_component, torch.nn.Module):
                    sub_component.eval()

    with torch.no_grad():
        logits, _ = model_forward_fn(points_tensor, model_components)

        probabilities = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, predicted_class].item()

    return predicted_class, confidence, probabilities.cpu().numpy()[0]

def predict_segmentation(points, model_forward_fn, model_components, device='cuda', num_points=2048):
    """Run segmentation inference on point cloud"""
    from deep_learning_3d.data.augmentation import sample_fixed_points, normalize_points

    original_num_points = len(points)

    if len(points) != num_points:
        from scipy.spatial import cKDTree

        sampled_points = sample_fixed_points(points, num_points)
        sampled_points = normalize_points(sampled_points, method='center')

        points_tensor = torch.from_numpy(sampled_points).float().unsqueeze(0).to(device)
        points_tensor = points_tensor.transpose(1, 2)
    else:
        sampled_points = points
        points = normalize_points(points, method='center')
        points_tensor = torch.from_numpy(points).float().unsqueeze(0).to(device)
        points_tensor = points_tensor.transpose(1, 2)

    for component in model_components.values():
        if isinstance(component, torch.nn.Module):
            component.eval()
        elif isinstance(component, dict):
            for sub_component in component.values():
                if isinstance(sub_component, torch.nn.Module):
                    sub_component.eval()

    with torch.no_grad():
        output, _ = model_forward_fn(points_tensor, model_components)

        predictions = torch.argmax(output, dim=2).cpu().numpy()[0]

    if original_num_points != num_points:
        tree = cKDTree(sampled_points)
        _, indices = tree.query(points, k=1)
        predictions = predictions[indices]

    return predictions

def batch_predict_classification(points_list, model_forward_fn, model_components,
                                 device='cuda', batch_size=32, num_points=2048):
    """Batch inference for classification"""
    from deep_learning_3d.data.augmentation import sample_fixed_points, normalize_points

    predictions = []
    confidences = []

    for i in range(0, len(points_list), batch_size):
        batch_points = points_list[i:i + batch_size]

        batch_tensors = []
        for points in batch_points:
            if len(points) != num_points:
                points = sample_fixed_points(points, num_points)

            points = normalize_points(points, method='center')
            points_tensor = torch.from_numpy(points).float()
            batch_tensors.append(points_tensor)

        batch_tensor = torch.stack(batch_tensors).transpose(1, 2).to(device)

        with torch.no_grad():
            logits, _ = model_forward_fn(batch_tensor, model_components)

            probabilities = torch.softmax(logits, dim=1)
            predicted_classes = torch.argmax(probabilities, dim=1).cpu().numpy()
            batch_confidences = torch.max(probabilities, dim=1)[0].cpu().numpy()

        predictions.extend(predicted_classes)
        confidences.extend(batch_confidences)

    return np.array(predictions), np.array(confidences)

def export_onnx(model_forward_fn, model_components, output_path, input_shape=(1, 3, 2048)):
    """Export model to ONNX format"""
    dummy_input = torch.randn(input_shape)

    class ModelWrapper(torch.nn.Module):
        def __init__(self, forward_fn, components):
            super().__init__()
            self.forward_fn = forward_fn
            self.components = components

        def forward(self, x):
            logits, _ = self.forward_fn(x, self.components)
            return logits

    wrapped_model = ModelWrapper(model_forward_fn, model_components)
    wrapped_model.eval()

    torch.onnx.export(
        wrapped_model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )

def predict_with_tta(points, model_forward_fn, model_components, device='cuda',
                     num_augmentations=8, num_points=2048):
    """Test-time augmentation for more robust predictions"""
    from deep_learning_3d.data.augmentation import (
        sample_fixed_points, normalize_points, rotate_points_z_axis
    )

    if len(points) != num_points:
        points = sample_fixed_points(points, num_points)

    points = normalize_points(points, method='center')

    all_predictions = []

    for i in range(num_augmentations):
        if i == 0:
            augmented_points = points
        else:
            angle = 2 * np.pi * i / num_augmentations
            augmented_points = rotate_points_z_axis(points, angle=angle)

        points_tensor = torch.from_numpy(augmented_points).float().unsqueeze(0).to(device)
        points_tensor = points_tensor.transpose(1, 2)

        with torch.no_grad():
            logits, _ = model_forward_fn(points_tensor, model_components)
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
            all_predictions.append(probabilities)

    avg_probabilities = np.mean(all_predictions, axis=0)
    predicted_class = np.argmax(avg_probabilities)
    confidence = avg_probabilities[predicted_class]

    return predicted_class, confidence, avg_probabilities

def compute_feature_embeddings(points_list, model_components, device='cuda', num_points=2048):
    """Extract feature embeddings for clustering/retrieval"""
    from deep_learning_3d.models.pointnet import extract_pointnet_features
    from deep_learning_3d.data.augmentation import sample_fixed_points, normalize_points

    embeddings = []

    for points in points_list:
        if len(points) != num_points:
            points = sample_fixed_points(points, num_points)

        points = normalize_points(points, method='center')
        points_tensor = torch.from_numpy(points).float().unsqueeze(0).to(device)
        points_tensor = points_tensor.transpose(1, 2)

        features = extract_pointnet_features(points_tensor, model_components)
        embeddings.append(features.cpu().numpy())

    return np.vstack(embeddings)

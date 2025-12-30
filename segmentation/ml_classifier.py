import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def prepare_feature_vector(points, normals=None, colors=None, height_features=None, geometric_features=None):
    """Prepare feature vector for ML classification"""
    features_list = []

    features_list.append(points)

    if normals is not None:
        features_list.append(normals)

    if colors is not None:
        features_list.append(colors)

    if height_features is not None:
        if height_features.ndim == 1:
            height_features = height_features.reshape(-1, 1)
        features_list.append(height_features)

    if geometric_features is not None:
        features_list.append(geometric_features)

    feature_vector = np.hstack(features_list)

    return feature_vector

def train_random_forest(features, labels, n_estimators=100, max_depth=None, test_size=0.2, random_state=42):
    """Train Random Forest classifier"""
    valid_mask = labels >= 0
    features_clean = features[valid_mask]
    labels_clean = labels[valid_mask]

    X_train, X_test, y_train, y_test = train_test_split(
        features_clean, labels_clean,
        test_size=test_size,
        random_state=random_state,
        stratify=labels_clean
    )

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1
    )

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    results = {
        'classifier': clf,
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report,
        'feature_importance': clf.feature_importances_
    }

    return results

def predict_labels(clf, features):
    """Predict labels for new data"""
    predictions = clf.predict(features)

    return predictions

def predict_labels_with_confidence(clf, features):
    """Predict labels with confidence scores"""
    predictions = clf.predict(features)
    probabilities = clf.predict_proba(features)

    confidence = probabilities.max(axis=1)

    return predictions, confidence

def cross_validate_classifier(features, labels, n_folds=5, n_estimators=100):
    """Cross-validate classifier performance"""
    from sklearn.model_selection import cross_val_score

    valid_mask = labels >= 0
    features_clean = features[valid_mask]
    labels_clean = labels[valid_mask]

    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)

    scores = cross_val_score(clf, features_clean, labels_clean, cv=n_folds)

    return {
        'mean_accuracy': scores.mean(),
        'std_accuracy': scores.std(),
        'scores': scores
    }

def active_learning_selection(clf, unlabeled_features, n_samples=100, method='uncertainty'):
    """Select most informative samples for labeling"""
    if method == 'uncertainty':
        probabilities = clf.predict_proba(unlabeled_features)
        uncertainty = 1.0 - probabilities.max(axis=1)

        selected_indices = np.argsort(uncertainty)[-n_samples:]

    elif method == 'margin':
        probabilities = clf.predict_proba(unlabeled_features)
        sorted_probs = np.sort(probabilities, axis=1)
        margin = sorted_probs[:, -1] - sorted_probs[:, -2]

        selected_indices = np.argsort(margin)[:n_samples]

    elif method == 'entropy':
        probabilities = clf.predict_proba(unlabeled_features)
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-10), axis=1)

        selected_indices = np.argsort(entropy)[-n_samples:]

    return selected_indices

def stratified_sampling_for_training(points, labels, samples_per_class=1000):
    """Create balanced training set"""
    from filters.sampler import stratified_sampling

    sampled_points, sampled_labels, sampled_indices = stratified_sampling(
        points, labels, samples_per_class
    )

    return sampled_indices

def compute_feature_importance(clf, feature_names=None):
    """Analyze feature importance"""
    importances = clf.feature_importances_

    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(len(importances))]

    importance_dict = {
        name: importance
        for name, importance in zip(feature_names, importances)
    }

    sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)

    return sorted_importance

def iterative_self_training(features, labeled_mask, labels, confidence_threshold=0.9, max_iterations=10):
    """Semi-supervised learning using self-training"""
    current_labeled_mask = labeled_mask.copy()
    current_labels = labels.copy()

    for iteration in range(max_iterations):
        labeled_features = features[current_labeled_mask]
        labeled_labels = current_labels[current_labeled_mask]

        clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        clf.fit(labeled_features, labeled_labels)

        unlabeled_mask = ~current_labeled_mask
        if unlabeled_mask.sum() == 0:
            break

        unlabeled_features = features[unlabeled_mask]

        predictions, confidence = predict_labels_with_confidence(clf, unlabeled_features)

        high_confidence_mask = confidence >= confidence_threshold

        if high_confidence_mask.sum() == 0:
            break

        unlabeled_indices = np.where(unlabeled_mask)[0]
        pseudo_labeled_indices = unlabeled_indices[high_confidence_mask]

        current_labels[pseudo_labeled_indices] = predictions[high_confidence_mask]
        current_labeled_mask[pseudo_labeled_indices] = True

        print(f"Iteration {iteration + 1}: Added {high_confidence_mask.sum()} pseudo-labels")

    return current_labels, current_labeled_mask

def evaluate_per_class(y_true, y_pred, class_names=None):
    """Detailed per-class evaluation"""
    unique_classes = np.unique(y_true)

    results = {}

    for cls in unique_classes:
        cls_mask_true = y_true == cls
        cls_mask_pred = y_pred == cls

        tp = (cls_mask_true & cls_mask_pred).sum()
        fp = (~cls_mask_true & cls_mask_pred).sum()
        fn = (cls_mask_true & ~cls_mask_pred).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        class_name = class_names[cls] if class_names else f"Class_{cls}"

        results[class_name] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'support': cls_mask_true.sum()
        }

    return results

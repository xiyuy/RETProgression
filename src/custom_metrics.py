import numpy as np
import torch
from typing import Tuple, List, Union, Optional, Dict, Any


def roc_auc_score(y_true, y_score):
    """
    Custom implementation of ROC AUC calculation.
    
    Args:
        y_true: Ground truth binary labels (0, 1)
        y_score: Predicted scores or probabilities
    
    Returns:
        Area under the ROC curve
    """
    # Convert to numpy if they're torch tensors
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy()
    if torch.is_tensor(y_score):
        y_score = y_score.cpu().numpy()
    
    # Convert to numpy arrays
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    
    # Edge case handling
    if len(np.unique(y_true)) < 2:
        return 0.5  # Random performance if only one class
    
    # Sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score)[::-1]
    y_true = y_true[desc_score_indices]
    
    # Count positive and negative samples
    n_pos = np.sum(y_true)
    n_neg = len(y_true) - n_pos
    
    if n_pos == 0 or n_neg == 0:
        return 0.5  # Random performance if only one class
    
    # Calculate true positive rates and false positive rates
    tps = np.cumsum(y_true)
    fps = np.cumsum(1 - y_true)
    
    # Calculate rates
    tpr = tps / n_pos
    fpr = fps / n_neg
    
    # Calculate area using trapezoidal rule
    # Add (0,0) point
    tpr = np.concatenate([[0], tpr])
    fpr = np.concatenate([[0], fpr])
    
    # Calculate AUC using trapezoidal rule
    width = np.diff(fpr)
    height = (tpr[1:] + tpr[:-1]) / 2
    
    return np.sum(width * height)


def balanced_accuracy_score(y_true, y_pred):
    """
    Compute the balanced accuracy score.
    
    The balanced accuracy is defined as the average of recall for each class.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
    
    Returns:
        Balanced accuracy score
    """
    # Convert to numpy if they're torch tensors
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.cpu().numpy()
    
    # Convert to numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Get unique classes
    classes = np.unique(y_true)
    
    # Special case for binary classification
    if len(classes) == 2:
        # Calculate sensitivity (recall of positive class)
        pos_indices = (y_true == 1)
        if np.sum(pos_indices) > 0:
            sensitivity = np.sum((y_pred == 1) & pos_indices) / np.sum(pos_indices)
        else:
            sensitivity = 0.0
        
        # Calculate specificity (recall of negative class)
        neg_indices = (y_true == 0)
        if np.sum(neg_indices) > 0:
            specificity = np.sum((y_pred == 0) & neg_indices) / np.sum(neg_indices)
        else:
            specificity = 0.0
        
        # Return the mean of sensitivity and specificity
        return (sensitivity + specificity) / 2
    
    # For multiclass, calculate per-class recall and average
    recalls = []
    for c in classes:
        # Get indices for this class
        class_indices = (y_true == c)
        if np.sum(class_indices) > 0:
            # Calculate recall for this class
            recall = np.sum((y_pred == c) & class_indices) / np.sum(class_indices)
            recalls.append(recall)
    
    # Return the mean of all recalls (balanced accuracy)
    return np.mean(recalls) if recalls else 0.0


def roc_curve(y_true, y_score, drop_intermediate: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Custom implementation of ROC curve calculation.
    
    Args:
        y_true: Ground truth binary labels (0, 1)
        y_score: Predicted scores or probabilities
        drop_intermediate: Whether to drop some suboptimal thresholds to reduce size
                         of returned arrays (similar to sklearn implementation)
    
    Returns:
        fpr: False positive rates
        tpr: True positive rates
        thresholds: Thresholds used to compute the curve
    """
    # Convert to numpy if they're torch tensors
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy()
    if torch.is_tensor(y_score):
        y_score = y_score.cpu().numpy()
    
    # Convert to numpy arrays
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    
    # Ensure binary classification
    if len(np.unique(y_true)) > 2:
        raise ValueError("ROC curve is only defined for binary classification")
    
    # Count positives and negatives
    n_pos = np.sum(y_true)
    n_neg = len(y_true) - n_pos
    
    # Edge case handling
    if n_pos == 0 or n_neg == 0:
        # Return a minimal curve if only one class is present
        return np.array([0, 1]), np.array([0, 1]), np.array([np.inf, -np.inf])
    
    # Get the distinct score values for thresholds
    distinct_value_indices = np.where(np.diff(np.sort(y_score)))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_score.size - 1]
    
    # Use all thresholds to ensure proper curve
    thresholds = y_score[threshold_idxs]
    
    # Initialize arrays to store metrics
    tps = np.zeros(len(thresholds) + 1)
    fps = np.zeros(len(thresholds) + 1)
    
    # Compute confusion matrix for each threshold
    for i, threshold in enumerate(thresholds):
        y_pred = (y_score >= threshold).astype(int)
        tps[i+1] = np.sum((y_pred == 1) & (y_true == 1))
        fps[i+1] = np.sum((y_pred == 1) & (y_true == 0))
    
    # Calculate rates
    tpr = tps / n_pos
    fpr = fps / n_neg
    
    # Add (1, 1) point at the end
    if tps[-1] < n_pos or fps[-1] < n_neg:
        tpr = np.r_[tpr, 1.0]
        fpr = np.r_[fpr, 1.0]
        thresholds = np.r_[thresholds, -np.inf]
    
    # If drop_intermediate is True, drop some unnecessary points
    if drop_intermediate and len(thresholds) > 2:
        # Find the points that don't add much to the curve
        optimal_idxs = _support_vertices(fpr, tpr)
        
        # Keep only the optimal vertices
        fpr = fpr[optimal_idxs]
        tpr = tpr[optimal_idxs]
        thresholds = thresholds[optimal_idxs[:-1] - 1] if len(optimal_idxs) > 1 else np.array([])
    
    return fpr, tpr, thresholds


def precision_recall_curve(y_true, y_score) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Custom implementation of Precision-Recall curve calculation.
    
    Args:
        y_true: Ground truth binary labels (0, 1)
        y_score: Predicted scores or probabilities
    
    Returns:
        precision: Precision values
        recall: Recall values
        thresholds: Thresholds used to compute the curve
    """
    # Convert to numpy if they're torch tensors
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy()
    if torch.is_tensor(y_score):
        y_score = y_score.cpu().numpy()
    
    # Convert to numpy arrays
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    
    # Ensure binary classification
    if len(np.unique(y_true)) > 2:
        raise ValueError("Precision-Recall curve is only defined for binary classification")
    
    # Count positives
    n_pos = np.sum(y_true)
    
    # Edge case handling
    if n_pos == 0:
        # Return a minimal curve if no positive samples
        return np.array([0, 1]), np.array([0, 0]), np.array([np.inf, -np.inf])
    
    # Get the distinct score values for thresholds
    distinct_value_indices = np.where(np.diff(np.sort(y_score)))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_score.size - 1]
    
    # Get thresholds for the curve
    thresholds = y_score[threshold_idxs]
    
    # Initialize arrays to store metrics
    tps = np.zeros(len(thresholds) + 1)
    fps = np.zeros(len(thresholds) + 1)
    
    # Compute confusion matrix for each threshold
    for i, threshold in enumerate(thresholds):
        y_pred = (y_score >= threshold).astype(int)
        tps[i+1] = np.sum((y_pred == 1) & (y_true == 1))
        fps[i+1] = np.sum((y_pred == 1) & (y_true == 0))
    
    # Calculate precision and recall
    precision = tps / np.maximum(tps + fps, 1e-10)
    recall = tps / n_pos
    
    # Handle special cases for precision at zero recall
    zero_idx = np.where(tps == 0)[0]
    if len(zero_idx) > 0:
        precision[zero_idx] = 1.0
    
    # Ensure precision is decreasing (for proper PR curve)
    # by taking the maximum precision for each level of recall
    precision = _maximize_precision_for_recall_levels(precision, recall)
    
    # Add (0, 1) point for PR curve completeness
    if precision[0] < 1:
        precision = np.r_[1, precision]
        recall = np.r_[0, recall]
        thresholds = np.r_[np.inf, thresholds]
    
    return precision, recall, thresholds


def average_precision_score(y_true, y_score) -> float:
    """
    Custom implementation of Average Precision Score (area under PR curve).
    
    Args:
        y_true: Ground truth binary labels (0, 1)
        y_score: Predicted scores or probabilities
    
    Returns:
        Average precision score
    """
    # Convert to numpy if they're torch tensors
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy()
    if torch.is_tensor(y_score):
        y_score = y_score.cpu().numpy()
    
    # Convert to numpy arrays
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    
    # Edge case handling
    if len(np.unique(y_true)) < 2:
        return 1.0 if np.all(y_true == 1) else 0.0
    
    # Get precision and recall values
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    
    # Calculate area using step function
    # We compute the sum of (recall[i+1] - recall[i]) * precision[i+1]
    recall_diff = np.diff(recall)
    precision_next = precision[1:]
    
    # Calculate the area under the PR curve
    return np.sum(recall_diff * precision_next)


def auc(x, y) -> float:
    """
    Custom implementation of Area Under Curve calculation using trapezoidal rule.
    
    Args:
        x: X-coordinate values (e.g., false positive rate, recall)
        y: Y-coordinate values (e.g., true positive rate, precision)
    
    Returns:
        Area under the curve
    """
    # Convert to numpy if they're torch tensors
    if torch.is_tensor(x):
        x = x.cpu().numpy()
    if torch.is_tensor(y):
        y = y.cpu().numpy()
    
    # Convert to numpy arrays
    x = np.asarray(x)
    y = np.asarray(y)
    
    if len(x) < 2:
        raise ValueError("At least 2 points are required to compute AUC")
    
    # Sort by x values for proper calculation
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    
    # Compute trapezoidal rule
    width = np.diff(x)
    height = (y[1:] + y[:-1]) / 2
    
    return np.sum(width * height)


def _support_vertices(x, y):
    """
    Return indices of the points that are on the convex hull of the curve.
    This is used for dropping intermediate points in roc_curve.
    
    Args:
        x: X-coordinate values (e.g., false positive rate)
        y: Y-coordinate values (e.g., true positive rate)
    
    Returns:
        Indices of the vertices to keep
    """
    # Always include the first and last points
    vertices = [0]
    
    # Find points that improve the slope
    slopes = np.zeros(len(x) - 1)
    for i in range(len(x) - 1):
        dx = x[i+1] - x[i]
        dy = y[i+1] - y[i]
        slopes[i] = dy / dx if dx > 0 else np.inf
    
    # Keep only those points that improve the slope
    # This simplifies the curve without changing its shape
    prev_slope = -np.inf
    for i in range(len(slopes)):
        if slopes[i] > prev_slope:
            vertices.append(i + 1)
            prev_slope = slopes[i]
    
    # Convert to numpy array and return
    return np.array(vertices)


def _maximize_precision_for_recall_levels(precision, recall):
    """
    Ensure precision is monotonically decreasing for a PR curve.
    This is done by taking the maximum precision for each level of recall.
    
    Args:
        precision: Array of precision values
        recall: Array of recall values
    
    Returns:
        Modified precision array
    """
    # Work backwards from end to ensure monotonicity
    decreasing_max_precision = np.maximum.accumulate(precision[::-1])[::-1]
    return decreasing_max_precision


def confusion_matrix(y_true, y_pred, labels=None, normalize=None) -> np.ndarray:
    """
    Custom implementation of confusion matrix calculation.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        labels: List of labels to index the matrix. If None, the unique values
                from y_true and y_pred are used in sorted order.
        normalize: Normalization option ('true', 'pred', 'all', or None)
                   'true': normalize by row (true labels)
                   'pred': normalize by column (predicted labels)
                   'all': normalize by all values
                   None: no normalization
    
    Returns:
        Confusion matrix with shape (n_classes, n_classes)
    """
    # Convert to numpy if they're torch tensors
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.cpu().numpy()
    
    # Convert to numpy arrays and flatten
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    # Check inputs have the same shape
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true and y_pred must have the same shape. "
                         f"Got {y_true.shape} and {y_pred.shape}")
    
    # Get unique classes
    if labels is None:
        classes = np.unique(np.concatenate((y_true, y_pred)))
        classes = np.sort(classes)
    else:
        classes = np.asarray(labels)
    
    # Map original labels to indices
    n_classes = len(classes)
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    
    # Initialize confusion matrix
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    
    # Fill confusion matrix
    for i in range(len(y_true)):
        # Skip values not in the label set if using provided labels
        if y_true[i] not in class_to_idx or y_pred[i] not in class_to_idx:
            continue
        true_idx = class_to_idx[y_true[i]]
        pred_idx = class_to_idx[y_pred[i]]
        cm[true_idx, pred_idx] += 1
    
    # Apply normalization if requested
    if normalize is not None:
        cm = cm.astype(np.float64)
        if normalize == 'true':
            # Normalize by row (true labels)
            row_sums = cm.sum(axis=1, keepdims=True)
            # Avoid division by zero
            row_sums = np.maximum(row_sums, np.ones_like(row_sums) * 1e-15)
            cm = cm / row_sums
        elif normalize == 'pred':
            # Normalize by column (predicted labels)
            col_sums = cm.sum(axis=0, keepdims=True)
            # Avoid division by zero
            col_sums = np.maximum(col_sums, np.ones_like(col_sums) * 1e-15)
            cm = cm / col_sums
        elif normalize == 'all':
            # Normalize by all values
            total = cm.sum()
            # Avoid division by zero
            total = max(total, 1e-15)
            cm = cm / total
        else:
            raise ValueError(f"Invalid normalize option: {normalize}. "
                             f"Must be one of: 'true', 'pred', 'all', or None")
    
    return cm


def confusion_matrix_with_stats(y_true, y_pred, labels=None) -> Dict[str, Any]:
    """
    Calculate confusion matrix and derived statistics for binary classification.
    This is particularly useful for getting a complete set of metrics 
    like precision, recall, F1 score, etc. in one call.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        labels: List of labels to index the matrix. For binary classification,
                this should be [negative_class, positive_class]
    
    Returns:
        Dictionary containing:
        - 'matrix': Raw confusion matrix
        - 'TP', 'FP', 'FN', 'TN': Individual counts
        - 'accuracy': Overall accuracy
        - 'precision': Precision score (TP / (TP + FP))
        - 'recall': Recall/Sensitivity (TP / (TP + FN))
        - 'specificity': Specificity (TN / (TN + FP))
        - 'f1_score': F1 score (2 * precision * recall / (precision + recall))
        - 'balanced_accuracy': Balanced accuracy ((sensitivity + specificity) / 2)
    """
    # Convert to numpy if they're torch tensors
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.cpu().numpy()
    
    # For binary classification, ensure we have 0 and 1 as classes
    # If not specified, use [0, 1] as default labels
    if labels is None:
        all_classes = np.unique(np.concatenate((y_true, y_pred)))
        if len(all_classes) <= 2:
            # Binary classification
            if len(all_classes) == 1:
                # Only one class is present, add the other
                if all_classes[0] == 0:
                    labels = [0, 1]
                else:
                    labels = [0, all_classes[0]]
            else:
                labels = sorted(all_classes)
        else:
            raise ValueError("confusion_matrix_with_stats is designed for binary classification. "
                             f"Got {len(all_classes)} classes.")
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Check if this is a 2x2 matrix (binary classification)
    if cm.shape != (2, 2):
        raise ValueError(f"Expected 2x2 confusion matrix for binary classification. "
                         f"Got shape {cm.shape}.")
    
    # Extract counts
    tn, fp = cm[0, 0], cm[0, 1]
    fn, tp = cm[1, 0], cm[1, 1]
    
    # Calculate metrics with epsilon to avoid division by zero
    epsilon = 1e-15
    
    # Total samples
    total = tp + tn + fp + fn
    
    # Accuracy
    accuracy = (tp + tn) / max(total, epsilon)
    
    # Precision
    precision = tp / max(tp + fp, epsilon)
    
    # Recall (Sensitivity)
    recall = sensitivity = tp / max(tp + fn, epsilon)
    
    # Specificity
    specificity = tn / max(tn + fp, epsilon)
    
    # F1 Score
    f1_score = 2 * precision * recall / max(precision + recall, epsilon)
    
    # Balanced Accuracy
    balanced_acc = (sensitivity + specificity) / 2
    
    # Return all calculated metrics
    return {
        'matrix': cm,
        'TP': int(tp),
        'FP': int(fp),
        'FN': int(fn),
        'TN': int(tn),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'f1_score': float(f1_score),
        'balanced_accuracy': float(balanced_acc)
    }
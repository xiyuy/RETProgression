import numpy as np
import torch

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
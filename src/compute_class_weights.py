"""
Utility functions for computing class weights automatically
"""
import torch
import numpy as np
import logging
from typing import Union, List


def compute_class_weights(
    dataset,
    method: str = 'inverse_frequency',
    normalize: bool = False
) -> torch.Tensor:
    """
    Compute class weights from dataset to handle class imbalance.
    
    Args:
        dataset: PyTorch dataset or labels array
        method: Weighting method
            - 'inverse_frequency': weight = n_samples / (n_classes * n_samples_per_class)
            - 'sqrt': weight = sqrt(n_majority / n_minority) (more conservative)
            - 'effective': Effective number of samples (best for extreme imbalance)
        normalize: Whether to normalize weights so minority class = 1.0
    
    Returns:
        Tensor of class weights [weight_class_0, weight_class_1, ...]
    """
    # Extract labels from dataset
    if hasattr(dataset, 'img_labels'):
        # JoslinData format
        labels = dataset.img_labels.iloc[:, 1].values
    elif hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'img_labels'):
        # CachedDataset wrapping JoslinData
        labels = dataset.dataset.img_labels.iloc[:, 1].values
    elif hasattr(dataset, 'targets'):
        # Standard PyTorch dataset
        labels = np.array(dataset.targets)
    elif isinstance(dataset, (list, np.ndarray)):
        # Direct array of labels
        labels = np.array(dataset)
    else:
        raise ValueError(f"Cannot extract labels from dataset of type {type(dataset)}")
    
    # Get class counts
    unique_classes, class_counts = np.unique(labels, return_counts=True)
    n_classes = len(unique_classes)
    total_samples = len(labels)
    
    # Log class distribution
    logging.info(f"Class distribution:")
    for cls, count in zip(unique_classes, class_counts):
        percentage = 100.0 * count / total_samples
        logging.info(f"  Class {cls}: {count} samples ({percentage:.1f}%)")
    
    # Compute weights based on method
    if method == 'inverse_frequency':
        # Standard sklearn approach: n_samples / (n_classes * n_samples_per_class)
        weights = total_samples / (n_classes * class_counts)
        
    elif method == 'sqrt':
        # Square root of inverse frequency (more conservative)
        # For binary: weight_minority = sqrt(n_majority / n_minority)
        if n_classes == 2:
            max_count = np.max(class_counts)
            weights = np.sqrt(max_count / class_counts)
        else:
            # For multiclass, use sqrt of inverse frequency
            weights = np.sqrt(total_samples / (n_classes * class_counts))
            
    elif method == 'effective':
        # Effective number of samples (Cui et al., 2019)
        # Good for extreme imbalance
        beta = 0.999  # Recommended for very imbalanced datasets
        effective_nums = (1.0 - np.power(beta, class_counts)) / (1.0 - beta)
        weights = 1.0 / effective_nums
        
    else:
        raise ValueError(f"Unknown method: {method}. Choose from 'inverse_frequency', 'sqrt', 'effective'")
    
    # Normalize so that minority class (smallest count) has weight 1.0
    if normalize:
        # Find minority class
        minority_idx = np.argmin(class_counts)
        weights = weights / weights[minority_idx]
    
    # Convert to tensor
    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    
    # Log computed weights
    logging.info(f"Computed class weights (method={method}):")
    for cls, weight in zip(unique_classes, weights_tensor):
        logging.info(f"  Class {cls}: weight = {weight:.2f}")
    
    # Log imbalance ratio
    if n_classes == 2:
        ratio = class_counts[1] / class_counts[0] if class_counts[0] < class_counts[1] else class_counts[0] / class_counts[1]
        logging.info(f"Imbalance ratio: {ratio:.1f}:1")
    
    return weights_tensor


def should_use_class_weights(config) -> bool:
    """
    Determine whether to use class weights based on configuration.
    
    Logic:
    - If weighted_sampler is enabled: Don't use class weights (already balanced)
    - If use_class_weights is explicitly set: Use that setting
    - Otherwise: Auto-detect based on dataset balance
    
    Args:
        config: Hydra configuration
    
    Returns:
        Boolean indicating whether to use class weights
    """
    # Check if weighted sampler is being used
    weighted_sampler_enabled = False
    if hasattr(config, 'data') and hasattr(config.data, 'weighted_sampler'):
        weighted_sampler_enabled = getattr(config.data.weighted_sampler, 'enabled', False)
    
    # If weighted sampler is used, dataset is already balanced
    if weighted_sampler_enabled:
        logging.info("Weighted sampler enabled - using equal class weights [1.0, 1.0]")
        return False
    
    # Check if explicitly set in config
    if hasattr(config, 'criterion') and hasattr(config.criterion, 'use_class_weights'):
        use_weights = config.criterion.use_class_weights
        logging.info(f"Class weights explicitly set to: {use_weights}")
        return use_weights
    
    # Default: use class weights for imbalanced datasets
    logging.info("Auto-enabling class weights for imbalanced dataset")
    return True


def get_class_weights_from_config(config, dataset, device='cuda') -> torch.Tensor:
    """
    Get class weights based on configuration and dataset.
    
    Args:
        config: Hydra configuration
        dataset: Training dataset
        device: Device to place weights on
    
    Returns:
        Tensor of class weights
    """
    # Check if we should use class weights
    if not should_use_class_weights(config):
        # Return equal weights
        n_classes = getattr(config.model, 'num_classes', 2)
        weights = torch.ones(n_classes, dtype=torch.float32)
        logging.info(f"Using equal class weights: {weights.tolist()}")
        return weights.to(device)
    
    # Check if weights are manually specified in config
    if (hasattr(config, 'criterion') and 
        hasattr(config.criterion, 'class_weights') and
        config.criterion.class_weights is not None):
        
        from omegaconf import ListConfig
        class_weights_raw = config.criterion.class_weights
        
        if isinstance(class_weights_raw, (list, tuple, ListConfig)):
            weights = torch.tensor(class_weights_raw, dtype=torch.float32)
            logging.info(f"Using manually specified class weights: {weights.tolist()}")
            return weights.to(device)
    
    # Auto-compute weights
    method = 'sqrt'  # Default method (conservative)
    if hasattr(config, 'criterion') and hasattr(config.criterion, 'weight_method'):
        method = config.criterion.weight_method
    
    logging.info(f"Auto-computing class weights using method: {method}")
    weights = compute_class_weights(dataset, method=method, normalize=True)
    
    return weights.to(device)
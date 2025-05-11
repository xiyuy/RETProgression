import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import logging

class FocalLoss(nn.Module):
    """
    Focal Loss implementation as described in https://arxiv.org/abs/1708.02002
    Extremely useful for imbalanced classification tasks.
    
    Args:
        alpha: Weighting factor for the rare class (higher alpha gives more weight to rare class)
        gamma: Focusing parameter (higher gamma reduces the contribution of easy examples)
        reduction: 'mean', 'sum' or 'none'
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Model predictions (logits, before softmax)
            targets: Ground truth labels
            
        Returns:
            Focal loss value
        """
        # Convert targets to one-hot encoding for multi-class support
        if inputs.shape[1] > 1:  # Multi-class case
            num_classes = inputs.shape[1]
            target_one_hot = F.one_hot(targets, num_classes).float()
            
            # Compute softmax and cross-entropy
            log_softmax = F.log_softmax(inputs, dim=1)
            ce_loss = -target_one_hot * log_softmax
            
            # Get probabilities
            probs = torch.exp(log_softmax)
            pt = torch.sum(target_one_hot * probs, dim=1)
            
            # Apply alpha weighting for different classes
            alpha = self.alpha * target_one_hot + (1 - self.alpha) * (1 - target_one_hot)
            alpha_t = torch.sum(alpha * target_one_hot, dim=1)
            
            # Apply focal term and alpha weights
            focal_weight = alpha_t * (1 - pt).pow(self.gamma)
            loss = focal_weight.unsqueeze(1) * ce_loss
            
            if self.reduction == 'mean':
                return loss.sum() / loss.shape[0]
            elif self.reduction == 'sum':
                return loss.sum()
            else:
                return loss
        else:  # Binary class case
            # Convert to binary case [0, 1]
            targets = targets.float()
            
            # Binary cross entropy with logits for numerical stability
            BCE_loss = F.binary_cross_entropy_with_logits(inputs.squeeze(), targets, reduction='none')
            
            # Apply the focal term
            pt = torch.exp(-BCE_loss)  # Probability of being correct
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_loss = alpha_t * (1 - pt).pow(self.gamma) * BCE_loss
            
            if self.reduction == 'mean':
                return torch.mean(focal_loss)
            elif self.reduction == 'sum':
                return torch.sum(focal_loss)
            else:
                return focal_loss

def get_loss_function(config, device):
    """
    Factory function to create loss function based on configuration.
    
    Args:
        config: Configuration with loss parameters
        device: Device to place tensors on
        
    Returns:
        Loss function
    """
    # Default class weights based on empirical 93:7 distribution
    default_class_weights = torch.tensor([1.0, 13.0])
    
    # Check what type of loss function is requested
    loss_type = getattr(config.criterion, 'type', 'cross_entropy') if hasattr(config, 'criterion') else 'cross_entropy'
    
    if loss_type == 'focal_loss':
        # Get focal loss parameters if provided
        alpha = getattr(config.criterion, 'alpha', 0.25) if hasattr(config, 'criterion') else 0.25
        gamma = getattr(config.criterion, 'gamma', 2.0) if hasattr(config, 'criterion') else 2.0
        
        focal_loss = FocalLoss(alpha=alpha, gamma=gamma)
        
        if hasattr(config, 'criterion') and hasattr(config.criterion, 'use_class_weights') and config.criterion.use_class_weights:
            # Log that both focal loss and class weights are active
            logging.info(f"Using Focal Loss (alpha={alpha}, gamma={gamma}) with class-balancing")
        else:
            logging.info(f"Using Focal Loss (alpha={alpha}, gamma={gamma})")
            
        return focal_loss
    else:  # Default to cross entropy
        if hasattr(config, 'criterion') and hasattr(config.criterion, 'use_class_weights') and config.criterion.use_class_weights:
            # Get class weights if provided, otherwise use default
            if hasattr(config.criterion, 'class_weights'):
                if isinstance(config.criterion.class_weights, (list, tuple)):
                    class_weights = torch.tensor(config.criterion.class_weights)
                else:
                    class_weights = default_class_weights
            else:
                class_weights = default_class_weights
                
            class_weights = class_weights.to(device)
            logging.info(f"Using weighted Cross Entropy loss with class weights: {class_weights}")
            return nn.CrossEntropyLoss(weight=class_weights)
        else:
            logging.info("Using standard Cross Entropy loss")
            return nn.CrossEntropyLoss()
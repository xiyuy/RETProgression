import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from omegaconf import ListConfig


class FocalLoss(nn.Module):
    """
    Focal Loss implementation for imbalanced classification tasks.
    Reference: https://arxiv.org/abs/1708.02002
    
    Args:
        alpha: Weighting factor for the rare class (higher gives more weight to rare class)
        gamma: Focusing parameter (higher reduces contribution of easy examples)
        reduction: 'mean', 'sum' or 'none'
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Handle multi-class case
        if inputs.shape[1] > 1:
            # Convert targets to one-hot encoding
            num_classes = inputs.shape[1]
            target_one_hot = F.one_hot(targets, num_classes).float()
            
            # Compute log softmax and cross-entropy
            log_softmax = F.log_softmax(inputs, dim=1)
            ce_loss = -target_one_hot * log_softmax
            
            # Calculate focal weighting
            probs = torch.exp(log_softmax)
            pt = torch.sum(target_one_hot * probs, dim=1)
            
            # Apply alpha weighting for different classes
            alpha = self.alpha * target_one_hot + (1 - self.alpha) * (1 - target_one_hot)
            alpha_t = torch.sum(alpha * target_one_hot, dim=1)
            
            # Apply focal term and alpha weights
            focal_weight = alpha_t * (1 - pt).pow(self.gamma)
            loss = focal_weight.unsqueeze(1) * ce_loss
            
        # Handle binary case
        else:
            targets = targets.float()
            
            # Binary cross entropy with logits for numerical stability
            bce_loss = F.binary_cross_entropy_with_logits(
                inputs.squeeze(), targets, reduction='none'
            )
            
            # Apply focal weighting
            pt = torch.exp(-bce_loss)
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * (1 - pt).pow(self.gamma) * bce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def get_loss_function(config, device):
    """
    Factory function to create loss function based on configuration.
    
    Args:
        config: Configuration with loss parameters
        device: Device to place tensors on
    
    Returns:
        Loss function
    """
    # Default class weights for imbalanced datasets (93:7 distribution)
    default_class_weights = torch.tensor([1.0, 13.0])
    
    # Get loss type from config, default to cross_entropy
    has_criterion = hasattr(config, 'criterion')
    loss_type = getattr(config.criterion, 'type', 'cross_entropy') if has_criterion else 'cross_entropy'
    
    # Check if class weights should be used
    use_class_weights = (
        has_criterion and 
        hasattr(config.criterion, 'use_class_weights') and
        config.criterion.use_class_weights
    )
    
    # Create appropriate loss function
    if loss_type == 'focal_loss':
        # Get focal loss parameters
        alpha = getattr(config.criterion, 'alpha', 0.25) if has_criterion else 0.25
        gamma = getattr(config.criterion, 'gamma', 2.0) if has_criterion else 2.0
        
        # Create focal loss
        focal_loss = FocalLoss(alpha=alpha, gamma=gamma)
        
        # Log configuration
        log_msg = f"Using Focal Loss (alpha={alpha}, gamma={gamma})"
        if use_class_weights:
            log_msg += " with class-balancing"
        logging.info(log_msg)
        
        return focal_loss
    
    # Default to cross entropy
    else:
        if use_class_weights:
            if hasattr(config.criterion, 'class_weights'):
                class_weights_raw = config.criterion.class_weights

                if isinstance(class_weights_raw, (list, tuple, ListConfig)):
                    class_weights = torch.tensor(class_weights_raw)
                else:
                    logging.warning(f"Unsupported type for class_weights: {type(class_weights_raw)}. Using default.")
                    class_weights = default_class_weights
            else:
                class_weights = default_class_weights
            
            class_weights = class_weights.to(device)
            logging.info(f"Using weighted Cross Entropy loss with class weights: {class_weights}")
            return nn.CrossEntropyLoss(weight=class_weights)
        
        else:
            logging.info("Using standard Cross Entropy loss")
            return nn.CrossEntropyLoss()
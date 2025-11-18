import torch
import math
import logging
from typing import List, Optional, Union, Dict, Any
from torch.optim.lr_scheduler import _LRScheduler, StepLR


class WarmupStepLR(_LRScheduler):
    """
    Learning rate scheduler with linear warmup followed by step decay.
    
    Args:
        optimizer: Optimizer to schedule learning rate for
        warmup_steps: Number of warmup steps
        step_size: Period of step decay (epochs)
        gamma: Multiplicative factor of learning rate decay
        min_lr: Minimum learning rate during warmup
        last_epoch: Last epoch (-1 for initialization)
        verbose: Whether to log learning rate changes
        
    Example:
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        >>> scheduler = WarmupStepLR(optimizer, warmup_steps=100, step_size=30, gamma=0.1)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     scheduler.step()
    """
    def __init__(
        self, 
        optimizer: torch.optim.Optimizer, 
        warmup_steps: int, 
        step_size: int, 
        gamma: float = 0.1,
        min_lr: float = 1e-7, 
        last_epoch: int = -1, 
        verbose: bool = False
    ):
        self.warmup_steps = warmup_steps
        self.step_size = step_size
        self.gamma = gamma
        self.min_lr = min_lr
        super(WarmupStepLR, self).__init__(optimizer, last_epoch, verbose)
    
    def get_lr(self) -> List[float]:
        # During warmup phase
        if self.last_epoch < self.warmup_steps:
            # Calculate warmup scaling factor (0 to 1)
            alpha = float(self.last_epoch) / float(max(1, self.warmup_steps))
            # Scale from min_lr to base_lr
            return [self.min_lr + alpha * (base_lr - self.min_lr) 
                   for base_lr in self.base_lrs]
        
        # After warmup, apply step decay
        decay_factor = self.gamma ** ((self.last_epoch - self.warmup_steps) // self.step_size)
        return [base_lr * decay_factor for base_lr in self.base_lrs]
    
    def _get_closed_form_lr(self) -> List[float]:
        # Simple implementation using get_lr for correctness
        return self.get_lr()


class CosineWarmupScheduler(_LRScheduler):
    """
    Scheduler with linear warmup followed by cosine annealing.
    
    Args:
        optimizer: Optimizer to schedule learning rate for
        warmup_steps: Number of warmup steps
        total_steps: Total number of training steps
        min_lr: Minimum learning rate during warmup and after cosine decay
        last_epoch: Last epoch (-1 for initialization)
        verbose: Whether to log learning rate changes
        
    Example:
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        >>> scheduler = CosineWarmupScheduler(
        >>>     optimizer, 
        >>>     warmup_steps=100, 
        >>>     total_steps=10000
        >>> )
        >>> for step in range(10000):
        >>>     train_batch(...)
        >>>     scheduler.step()
    """
    def __init__(
        self, 
        optimizer: torch.optim.Optimizer, 
        warmup_steps: int, 
        total_steps: int, 
        min_lr: float = 1e-7,
        last_epoch: int = -1, 
        verbose: bool = False
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super(CosineWarmupScheduler, self).__init__(optimizer, last_epoch, verbose)
    
    def get_lr(self) -> List[float]:
        # Warmup phase
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            alpha = float(self.last_epoch) / float(max(1, self.warmup_steps))
            return [self.min_lr + alpha * (base_lr - self.min_lr) 
                   for base_lr in self.base_lrs]
        
        # Cosine annealing phase
        # Calculate position in cosine cycle
        progress = float(self.last_epoch - self.warmup_steps) / float(
            max(1, self.total_steps - self.warmup_steps))
        # Cosine decay from base_lr to min_lr
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return [self.min_lr + cosine_factor * (base_lr - self.min_lr) 
               for base_lr in self.base_lrs]


def create_scheduler(
    optimizer: torch.optim.Optimizer, 
    config: Any, 
    train_loader_len: int, 
    world_size: int = 1, 
    rank: int = 0
) -> _LRScheduler:
    """
    Create learning rate scheduler based on configuration.
    
    Args:
        optimizer: Optimizer to schedule learning rate for
        config: Configuration with scheduler parameters
        train_loader_len: Number of batches per epoch
        world_size: Number of distributed processes
        rank: Process rank (for logging)
    
    Returns:
        Learning rate scheduler
    """
    # Extract warmup configuration with defaults
    warmup_enabled = False
    warmup_proportion = 0.1  # Default: 10% of first epoch for warmup
    min_lr = 1.0e-7  # Default minimum learning rate
    
    # Check if warmup configuration exists
    if hasattr(config, 'lr_scheduler') and hasattr(config.lr_scheduler, 'warmup'):
        warmup_enabled = getattr(config.lr_scheduler.warmup, 'enabled', False)
        warmup_proportion = getattr(config.lr_scheduler.warmup, 'warmup_proportion', 0.1)
        min_lr = getattr(config.lr_scheduler.warmup, 'min_lr', 1.0e-7)
    
    # Get other scheduler parameters
    step_size = getattr(config.lr_scheduler, 'step_size', 7) if hasattr(config, 'lr_scheduler') else 7
    gamma = getattr(config.lr_scheduler, 'gamma', 0.1) if hasattr(config, 'lr_scheduler') else 0.1
    
    # Calculate training details
    steps_per_epoch = train_loader_len
    total_epochs = getattr(config.exp, 'num_epochs', 10) if hasattr(config, 'exp') else 10
    
    # If warmup is enabled, create appropriate warmup scheduler
    if warmup_enabled:
        # Calculate warmup steps based on proportion of first epoch
        warmup_steps = int(warmup_proportion * steps_per_epoch)
        
        # Log warmup configuration on rank 0
        if rank == 0:
            logging.info(f"Learning rate warmup enabled: {warmup_steps} steps "
                        f"from {min_lr} to {optimizer.param_groups[0]['lr']}")
        
        # Determine scheduler type
        scheduler_type = getattr(config.lr_scheduler, 'type', 'step') if hasattr(config, 'lr_scheduler') else 'step'
        
        if scheduler_type == 'cosine':
            total_steps = steps_per_epoch * total_epochs
            return CosineWarmupScheduler(
                optimizer,
                warmup_steps=warmup_steps,
                total_steps=total_steps,
                min_lr=min_lr,
                verbose=(rank == 0)
            )
        else:  # Default to step
            return WarmupStepLR(
                optimizer,
                warmup_steps=warmup_steps,
                step_size=step_size,
                gamma=gamma,
                min_lr=min_lr,
                verbose=(rank == 0)
            )
    else:
        # Use standard StepLR without warmup
        return StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma,
            verbose=(rank == 0)
        )


def log_lr(
    optimizer: torch.optim.Optimizer, 
    epoch: int, 
    batch_idx: int, 
    batch_total: int, 
    rank: int = 0
) -> None:
    """
    Log learning rate at current step.
    
    Args:
        optimizer: Optimizer to get learning rate from
        epoch: Current epoch
        batch_idx: Current batch index
        batch_total: Total batches in epoch
        rank: Process rank
    """
    if rank == 0:
        current_lr = optimizer.param_groups[0]['lr']
        progress = 100.0 * batch_idx / batch_total
        logging.info(f"[LR] Epoch {epoch}: {progress:.1f}% - LR: {current_lr:.8f}")
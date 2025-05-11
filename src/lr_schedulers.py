import torch
import math
import logging
import numpy as np
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
    """
    def __init__(self, optimizer, warmup_steps, step_size, gamma=0.1, 
                 min_lr=1e-7, last_epoch=-1, verbose=False):
        self.warmup_steps = warmup_steps
        self.step_size = step_size
        self.gamma = gamma
        self.min_lr = min_lr
        self.base_lrs_backup = None  # Store original base_lrs
        super(WarmupStepLR, self).__init__(optimizer, last_epoch, verbose)
        
    def get_lr(self):
        # During warmup phase
        if self.last_epoch < self.warmup_steps:
            # Calculate warmup scaling factor (0 to 1)
            alpha = float(self.last_epoch) / float(max(1, self.warmup_steps))
            # Scale from min_lr to base_lr
            return [self.min_lr + alpha * (base_lr - self.min_lr) 
                    for base_lr in self.base_lrs]
        # After warmup, apply step decay
        else:
            decay_factor = self.gamma ** ((self.last_epoch - self.warmup_steps) // self.step_size)
            return [base_lr * decay_factor for base_lr in self.base_lrs]
    
    def _get_closed_form_lr(self):
        # Override for more efficient state dict
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
    """
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-7, 
                 last_epoch=-1, verbose=False):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super(CosineWarmupScheduler, self).__init__(optimizer, last_epoch, verbose)
        
    def get_lr(self):
        # Warmup phase
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            alpha = float(self.last_epoch) / float(max(1, self.warmup_steps))
            return [self.min_lr + alpha * (base_lr - self.min_lr) 
                    for base_lr in self.base_lrs]
        # Cosine annealing phase
        else:
            # Calculate position in cosine cycle
            progress = float(self.last_epoch - self.warmup_steps) / float(
                max(1, self.total_steps - self.warmup_steps))
            # Cosine decay from base_lr to min_lr
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            return [self.min_lr + cosine_factor * (base_lr - self.min_lr) 
                    for base_lr in self.base_lrs]

def create_scheduler(optimizer, config, train_loader_len, world_size=1, rank=0):
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
    # Check if warmup is enabled
    warmup_enabled = False
    warmup_proportion = 0.1  # Default: 10% of first epoch for warmup
    min_lr = 1.0e-7  # Default minimum learning rate
    
    if hasattr(config, 'lr_scheduler') and hasattr(config.lr_scheduler, 'warmup'):
        warmup_enabled = getattr(config.lr_scheduler.warmup, 'enabled', False)
        warmup_proportion = getattr(config.lr_scheduler.warmup, 'warmup_proportion', 0.1)
        min_lr = getattr(config.lr_scheduler.warmup, 'min_lr', 1.0e-7)
    
    # Get other parameters
    step_size = getattr(config.lr_scheduler, 'step_size', 7)
    gamma = getattr(config.lr_scheduler, 'gamma', 0.1)
    
    # Calculate number of warmup steps
    steps_per_epoch = train_loader_len
    total_epochs = getattr(config.exp, 'num_epochs', 10)
    
    # If warmup is enabled, create appropriate warmup scheduler
    if warmup_enabled:
        # Calculate warmup steps based on proportion of first epoch
        warmup_steps = int(warmup_proportion * steps_per_epoch)
        
        # Log warmup configuration on rank 0
        if rank == 0:
            logging.info(f"Learning rate warmup enabled: {warmup_steps} steps "
                         f"from {min_lr} to {optimizer.param_groups[0]['lr']}")
        
        # Create scheduler with warmup
        scheduler_type = getattr(config.lr_scheduler, 'type', 'step') if hasattr(config.lr_scheduler, 'type') else 'step'
        
        if scheduler_type == 'cosine':
            total_steps = steps_per_epoch * total_epochs
            scheduler = CosineWarmupScheduler(
                optimizer,
                warmup_steps=warmup_steps,
                total_steps=total_steps,
                min_lr=min_lr,
                verbose=(rank == 0)
            )
        else:  # Default to step
            scheduler = WarmupStepLR(
                optimizer,
                warmup_steps=warmup_steps,
                step_size=step_size,
                gamma=gamma,
                min_lr=min_lr,
                verbose=(rank == 0)
            )
    else:
        # Use standard StepLR
        scheduler = StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma,
            verbose=(rank == 0)
        )
    
    return scheduler

def log_lr(optimizer, epoch, batch_idx, batch_total, rank=0):
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
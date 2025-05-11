import os 
import shutil 
import hydra 
import torch 
import logging 
import torch.distributed as dist 
import torch.multiprocessing as mp 
from torch.nn.parallel import DistributedDataParallel as DDP 
from torch.utils.data import DataLoader 
from torch.utils.data.distributed import DistributedSampler 
from torchvision import transforms 
from timm import create_model 
import time 
import sys 
import numpy as np 
from omegaconf import OmegaConf
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Import custom modules
from datasets import JoslinData, get_transforms, visualize_augmentations
from loss_functions import get_loss_function, FocalLoss
from lr_schedulers import create_scheduler, log_lr
from visualization_tools import TrainingMonitor
from utils_parallel import train_model_custom_progress, CachedDataset, compute_metrics_from_confusion_matrix
from custom_metrics import roc_auc_score, balanced_accuracy_score

def safe_clear_directory(directory_path, logger=None): 
    """
    Safely clear all contents of a directory without deleting the directory itself.
    
    Args:
        directory_path: Path to the directory to clear
        logger: Optional logger to use for logging messages
    """
    if not os.path.exists(directory_path): 
        return
    
    # Log action
    if logger: 
        logger.info(f"Clearing contents of directory: {directory_path}") 
    else: 
        logging.info(f"Clearing contents of directory: {directory_path}") 
    
    # Remove all files in the directory
    for item in os.listdir(directory_path): 
        item_path = os.path.join(directory_path, item) 
        try: 
            if os.path.isfile(item_path): 
                os.unlink(item_path) 
                if logger: 
                    logger.debug(f"Removed file: {item_path}") 
            elif os.path.isdir(item_path): 
                shutil.rmtree(item_path) 
                if logger: 
                    logger.debug(f"Removed directory: {item_path}") 
        except Exception as e: 
            error_msg = f"Error clearing item {item_path}: {str(e)}"
            if logger: 
                logger.error(error_msg) 
            else: 
                logging.error(error_msg) 

def setup(rank, world_size): 
    """
    Initialize the distributed environment with minimal overhead.
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    
    # Initialize the process group without excessive logging
    dist.init_process_group("nccl", rank=rank, world_size=world_size) 
    if rank == 0: 
        logging.info(f"Initialized process group with world_size={world_size}") 

def cleanup(): 
    """
    Clean up distributed environment.
    """
    dist.destroy_process_group() 

def configure_rank_logger(rank, log_dir): 
    """
    Configure process-specific logging
    """
    log_file = os.path.join(log_dir, f'rank_{rank}_training.log') 
    
    # Configure logging for this process
    logger = logging.getLogger() 
    logger.handlers = [] # Remove any existing handlers
    
    # Add file handler
    file_handler = logging.FileHandler(log_file) 
    file_formatter = logging.Formatter( 
        f'[%(asctime)s][%(levelname)s][Rank {rank}] - %(message)s', 
        datefmt='%Y-%m-%d %H:%M:%S'
    ) 
    file_handler.setFormatter(file_formatter) 
    logger.addHandler(file_handler) 
    
    # Add console handler only for rank 0 to reduce output clutter
    if rank == 0: 
        console_handler = logging.StreamHandler(sys.stdout) 
        console_handler.setFormatter(file_formatter) 
        logger.addHandler(console_handler) 
    
    logger.setLevel(logging.INFO) 
    
    return logger 

def collect_predictions(outputs, targets, all_outputs, all_targets):
    """
    Collect model predictions and targets for later metrics computation.
    
    Args:
        outputs: Current batch outputs
        targets: Current batch targets
        all_outputs: List to collect all outputs
        all_targets: List to collect all targets
    """
    # Ensure detached tensors
    outputs_cpu = outputs.detach().cpu()
    targets_cpu = targets.detach().cpu()
    
    # Append to lists
    all_outputs.append(outputs_cpu)
    all_targets.append(targets_cpu)

def train_on_device(rank, world_size, config): 
    """
    Training function to be run on each GPU with optimized synchronization.
    """
    # Setup distributed environment
    setup(rank, world_size) 
    
    # Set up process-specific logging
    logger = configure_rank_logger(rank, config.exp.checkpoint_dir) 
    
    # Set device for this process
    device = torch.device(f"cuda:{rank}") 
    torch.cuda.set_device(device) 
    
    # Initialize training monitor for visualizations (only on rank 0)
    training_monitor = None
    if rank == 0:
        # Create visualization directory if it doesn't exist
        vis_dir = getattr(config.exp, 'visualization_dir', os.path.join(config.exp.checkpoint_dir, 'visualizations'))
        os.makedirs(vis_dir, exist_ok=True)
        
        # Initialize monitor with experiment name
        experiment_name = getattr(config.exp, 'experiment_name', 'experiment')
        training_monitor = TrainingMonitor(vis_dir, experiment_name)
    
    # Only rank 0 logs detailed information to reduce overhead
    if rank == 0: 
        logger.info(f"Running training on rank {rank} with world size {world_size}") 
        logger.info(f"Training for {config.exp.num_epochs} epochs") 
        logger.info(f"Batch size: {config.data.batch_size} (per GPU)") 
        logger.info(f"Total effective batch size: {config.data.batch_size * world_size}") 
        logger.info(f"Learning rate: {config.optimizer.lr}") 
        logger.info(f"Weight decay: {config.optimizer.weight_decay}") 
        logger.info(f"Experiment name: {getattr(config.exp, 'experiment_name', 'default')}")
    
    # Get data augmentation settings
    augmentation_enabled = getattr(config.data, 'augmentation', {}).get('enabled', True)
    augmentation_strength = getattr(config.data, 'augmentation', {}).get('strength', 'moderate')
    
    if not augmentation_enabled:
        augmentation_strength = 'none'
    
    if rank == 0:
        logger.info(f"Data augmentation: {augmentation_strength}")
    
    # Get transforms based on augmentation settings
    transforms_dict = get_transforms(augmentation_strength)
    train_transform = transforms_dict['train']
    val_transform = transforms_dict['val']
    
    # Define GPU normalization transform (applied after moving to GPU)
    gpu_transform = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    # Dataset creation with timing
    if rank == 0: 
        logger.info("Creating datasets...") 
    
    dataset_start_time = time.time() 
    
    try: 
        # Get cache sizes from config or use defaults
        train_cache_size = getattr(config.caching, 'train_cache_size', 1000) if hasattr(config, 'caching') else 1000
        val_cache_size = getattr(config.caching, 'val_cache_size', 1000) if hasattr(config, 'caching') else 1000
        cache_enabled = getattr(config.caching, 'enabled', True) if hasattr(config, 'caching') else True
        
        # Create base datasets with appropriate transforms
        base_datasets = { 
            "train": JoslinData( 
                data_dir=config.data.data_dir, 
                annotations_file=config.data.annotations_file_name + "train.csv", 
                img_dir="Exports_02052025", 
                transform=train_transform
            ), 
            "val": JoslinData( 
                data_dir=config.data.data_dir, 
                annotations_file=config.data.annotations_file_name + "val.csv", 
                img_dir="Exports_02052025", 
                transform=val_transform
            ) 
        } 
        
        # Visualize augmentations if enabled (only on rank 0)
        if rank == 0 and augmentation_enabled and getattr(config.data.augmentation, 'visualize', False):
            vis_dir = getattr(config.exp, 'visualization_dir', os.path.join(config.exp.checkpoint_dir, 'visualizations'))
            aug_vis_path = os.path.join(vis_dir, f"{getattr(config.exp, 'experiment_name', 'default')}_augmentations.png")
            visualize_augmentations(base_datasets["train"], num_samples=3, num_augmentations=5, save_path=aug_vis_path)
            logger.info(f"Saved augmentation visualization to {aug_vis_path}")
        
        # Cache settings - proportional to dataset size to avoid memory issues
        train_cache_size = min(len(base_datasets["train"]), train_cache_size) 
        val_cache_size = min(len(base_datasets["val"]), val_cache_size) 
        
        # Wrap datasets with caching
        if cache_enabled: 
            joslin_data = { 
                "train": CachedDataset( 
                    base_datasets["train"], 
                    cache_size=train_cache_size, 
                    cache_probability=1.0, 
                    rank=rank,
                    world_size=world_size
                ), 
                "val": CachedDataset( 
                    base_datasets["val"], 
                    cache_size=val_cache_size, 
                    cache_probability=1.0, 
                    rank=rank,
                    world_size=world_size
                ) 
            } 
        else: 
            joslin_data = base_datasets 
        
        dataset_time = time.time() - dataset_start_time 
        if rank == 0: 
            logger.info(f"Datasets created in {dataset_time:.2f}s") 
            logger.info(f"Train dataset size: {len(joslin_data['train'])}") 
            logger.info(f"Validation dataset size: {len(joslin_data['val'])}") 
            if cache_enabled: 
                logger.info(f"Caching enabled - Train cache size: {train_cache_size}, Val cache size: {val_cache_size}") 
            else: 
                logger.info("Caching disabled") 
    except Exception as e: 
        if rank == 0: 
            logger.error(f"Error creating datasets: {str(e)}") 
            import traceback 
            logger.error(traceback.format_exc()) 
        cleanup() 
        return
    
    # Sampler creation with fixed seed for reproducibility
    samplers = { 
        "train": DistributedSampler( 
            joslin_data["train"], 
            num_replicas=world_size, 
            rank=rank, 
            shuffle=config.data.shuffle, 
            seed=42 # Fixed seed for reproducibility
        ), 
        "val": DistributedSampler( 
            joslin_data["val"], 
            num_replicas=world_size, 
            rank=rank, 
            shuffle=False, 
            seed=42
        ) 
    } 
    
    # Configure optimal number of workers per GPU
    workers_per_gpu = min(16, max(4, config.data.num_workers)) 
    
    # DataLoader creation with performance optimizations
    if rank == 0: 
        logger.info("Creating dataloaders...") 
    
    dataloader_start_time = time.time() 
    
    try: 
        # Get prefetch factor from config or use default
        prefetch_factor = getattr(config.caching, 'prefetch_factor', 4) if hasattr(config, 'caching') else 4
        
        # Get drop_last from config or default to True for better batch consistency
        drop_last = getattr(config.optimization, 'drop_last', True) if hasattr(config, 'optimization') else True
        
        dataloader_kwargs = { 
            'batch_size': config.data.batch_size, 
            'shuffle': False, # We're using DistributedSampler instead
            'num_workers': workers_per_gpu, 
            'pin_memory': True, # Important for faster data transfer to GPU
            'persistent_workers': True if workers_per_gpu > 0 else False, 
            'drop_last': drop_last # Drop incomplete batches for better performance
        } 
        
        # Add prefetch_factor if workers > 0
        if workers_per_gpu > 0: 
            dataloader_kwargs['prefetch_factor'] = prefetch_factor 
        
        joslin_dataloaders = { 
            "train": DataLoader( 
                joslin_data["train"], 
                sampler=samplers["train"], 
                **dataloader_kwargs 
            ), 
            "val": DataLoader( 
                joslin_data["val"], 
                sampler=samplers["val"], 
                **dataloader_kwargs 
            ) 
        } 
        
        dataloader_time = time.time() - dataloader_start_time 
        if rank == 0: 
            logger.info(f"Dataloaders created in {dataloader_time:.2f}s") 
            logger.info(f"Workers per GPU: {workers_per_gpu}") 
            if workers_per_gpu > 0: 
                logger.info(f"Prefetch factor: {dataloader_kwargs.get('prefetch_factor', 1)}") 
            logger.info(f"Drop last batch: {dataloader_kwargs.get('drop_last', False)}") 
    except Exception as e: 
        if rank == 0: 
            logger.error(f"Error creating dataloaders: {str(e)}") 
            import traceback 
            logger.error(traceback.format_exc()) 
        cleanup() 
        return
    
    dataset_sizes = {x: len(joslin_data[x]) // world_size for x in ["train", "val"]} 
    
    # Model creation with timing
    if rank == 0: 
        logger.info(f"Creating model: {config.model.name}...") 
    
    model_start_time = time.time() 
    
    try: 
        # Create model with efficient initialization
        model = create_model( 
            config.model.name, 
            pretrained=config.model.pretrained, 
            num_classes=config.model.num_classes 
        ) 
        
        # Log model parameters count (only on rank 0)
        if rank == 0: 
            total_params = sum(p.numel() for p in model.parameters()) 
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) 
            logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable") 
        
        # Efficient model placement on GPU
        model = model.to(device, non_blocking=True) 
        
        # Apply SyncBatchNorm if configured
        sync_bn = getattr(config.optimization, 'sync_bn', False) if hasattr(config, 'optimization') else False
        if sync_bn and world_size > 1: 
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model) 
            if rank == 0: 
                logger.info("SyncBatchNorm applied to model") 
        
        # Configure DDP with optimized settings
        find_unused_parameters = getattr(config.optimization, 'find_unused_parameters', False) if hasattr(config, 'optimization') else False
        
        ddp_model = DDP( 
            model, 
            device_ids=[rank], 
            output_device=rank, 
            find_unused_parameters=find_unused_parameters,
            broadcast_buffers=False, # Reduces communication overhead
            gradient_as_bucket_view=True # Memory optimization
        ) 
        
        model_time = time.time() - model_start_time 
        if rank == 0: 
            logger.info(f"Model created and initialized in {model_time:.2f}s") 
    except Exception as e: 
        if rank == 0: 
            logger.error(f"Error creating model: {str(e)}") 
            import traceback 
            logger.error(traceback.format_exc()) 
        cleanup() 
        return
    
    # Configure loss function based on configuration
    criterion = get_loss_function(config, device)
    
    # Create optimizer
    if config.optimizer.name == "adam": 
        optimizer = torch.optim.Adam( 
            ddp_model.parameters(), 
            lr=config.optimizer.lr, 
            weight_decay=config.optimizer.weight_decay 
        ) 
    else: 
        optimizer = torch.optim.SGD( 
            ddp_model.parameters(), 
            lr=config.optimizer.lr, 
            momentum=config.optimizer.momentum, 
            weight_decay=config.optimizer.weight_decay 
        ) 
    
    # Create learning rate scheduler with warmup if configured
    scheduler = create_scheduler(
        optimizer, 
        config, 
        len(joslin_dataloaders['train']),
        world_size,
        rank
    )
    
    # Checkpoint path setup
    checkpoint_dir = config.exp.checkpoint_dir 
    checkpoint_name = config.exp.checkpoint_name 
    
    if checkpoint_name is not None: 
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name) 
    else: 
        checkpoint_path = checkpoint_dir 
    
    if rank == 0: 
        logger.info(f"Checkpoint path: {checkpoint_path}") 
        logger.info("======= Starting Training =======") 
    
    # Get gradient accumulation steps from config
    gradient_accumulation_steps = getattr(config.optimization, 'gradient_accumulation_steps', 1) if hasattr(config, 'optimization') else 1
    
    # Get AMP (automatic mixed precision) setting from config
    amp_enabled = getattr(config.optimization, 'amp_enabled', True) if hasattr(config, 'optimization') else True
    
    # Get gradient clipping value from config
    gradient_clipping = getattr(config.optimization, 'gradient_clipping', 0.0) if hasattr(config, 'optimization') else 0.0
    
    # Early stopping configuration
    early_stopping_params = None
    if hasattr(config, 'early_stopping') and getattr(config.early_stopping, 'enabled', False): 
        early_stopping_params = { 
            'patience': config.early_stopping.patience, 
            'metric': config.early_stopping.metric, 
            'mode': config.early_stopping.mode, 
            'min_delta': config.early_stopping.min_delta, 
            'verbose': True
        } 
        if rank == 0: 
            logger.info(f"Early stopping enabled with patience={early_stopping_params['patience']}, "
                        f"metric={early_stopping_params['metric']}, mode={early_stopping_params['mode']}") 
    
    if rank == 0: 
        if gradient_accumulation_steps > 1: 
            logger.info(f"Using gradient accumulation with {gradient_accumulation_steps} steps") 
        if amp_enabled: 
            logger.info("Automatic mixed precision (AMP) enabled") 
        if gradient_clipping > 0: 
            logger.info(f"Gradient clipping enabled with max norm {gradient_clipping}") 
    
    # Total training start time
    total_start_time = time.time() 
    
    # Modified training loop to collect metrics for visualization
    # Lists to store predictions for ROC/PR curves
    all_outputs = []
    all_targets = []
    
    try:
        # Track batches processed by phase
        batches_processed = {"train": 0, "val": 0}
        epoch_predictions = {}
        
        # Store for tracking learning rate
        lr_track_steps = []
        lr_track_values = []
        
        # Epoch loop
        for epoch in range(config.exp.num_epochs):
            epoch_start_time = time.time()
            
            if rank == 0:
                logger.info(f"Epoch {epoch}/{config.exp.num_epochs - 1}")
                logger.info("-" * 10)
            
            # Set epoch for samplers
            if samplers["train"] is not None:
                samplers["train"].set_epoch(epoch)
            if samplers["val"] is not None:
                samplers["val"].set_epoch(epoch)
            
            # Dictionary to store results for this epoch
            epoch_results = {}
            
            # Reset per-epoch prediction tracking
            epoch_predictions = {
                "train": {"outputs": [], "targets": []},
                "val": {"outputs": [], "targets": []}
            }
            
            # Training and validation phases
            for phase in ['train', 'val']:
                # Set model mode
                ddp_model.train(phase == 'train')
                
                # Metrics tracking
                running_loss = 0.0
                running_corrects = 0
                processed_samples = 0
                
                # Initialize confusion matrix components
                running_true_positive = 0
                running_false_positive = 0
                running_false_negative = 0
                running_true_negative = 0
                
                # Timing metrics
                data_time = 0.0
                forward_time = 0.0
                backward_time = 0.0
                phase_start_time = time.time()
                
                # Batch processing
                batch_start = time.time()
                
                # Calculate logging frequency
                total_batches = len(joslin_dataloaders[phase])
                log_every_n_batches = max(1, min(20, total_batches // 20))
                
                # Step counter for gradient accumulation
                steps = 0
                
                # Create gradient scaler for mixed precision
                scaler = torch.cuda.amp.GradScaler() if amp_enabled else None
                
                # Process batches
                for batch_idx, (inputs, labels) in enumerate(joslin_dataloaders[phase]):
                    # Log progress at regular intervals
                    if rank == 0 and (batch_idx % log_every_n_batches == 0 or batch_idx == total_batches - 1):
                        progress = 100.0 * batch_idx / total_batches
                        logger.info(f"{phase} Epoch {epoch}/{config.exp.num_epochs-1}: "
                                    f"{progress:.1f}% ({batch_idx}/{total_batches})")
                    
                    # Log learning rate periodically
                    if phase == 'train' and rank == 0:
                        if batch_idx % (log_every_n_batches * 5) == 0:
                            current_lr = optimizer.param_groups[0]['lr']
                            logger.info(f"Current learning rate: {current_lr:.8f}")
                            
                            # Track for visualization
                            global_step = epoch * total_batches + batch_idx
                            lr_track_steps.append(global_step)
                            lr_track_values.append(current_lr)
                            
                            # Update training monitor
                            if training_monitor:
                                training_monitor.update_learning_rate(current_lr, global_step)
                    
                    # Measure data loading time
                    current_data_time = time.time() - batch_start
                    data_time += current_data_time
                    
                    # Non-blocking transfer to GPU
                    inputs = inputs.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    
                    # Apply GPU transform
                    if gpu_transform is not None:
                        inputs = gpu_transform(inputs)
                    
                    # Zero gradients at accumulation boundaries or in eval mode
                    if phase == 'train' and (steps % gradient_accumulation_steps == 0):
                        optimizer.zero_grad(set_to_none=True)
                    
                    # Forward pass timing
                    forward_start = time.time()
                    
                    # Forward pass with mixed precision if enabled
                    with torch.cuda.amp.autocast() if amp_enabled else torch.no_grad() if phase == 'val' else torch.enable_grad():
                        outputs = ddp_model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        
                        # Scale loss by gradient accumulation steps in training
                        if phase == 'train':
                            loss = loss / gradient_accumulation_steps
                    
                    # Record forward pass time
                    forward_time += time.time() - forward_start
                    
                    # Backward pass and optimization (training only)
                    if phase == 'train':
                        backward_start = time.time()
                        
                        # Backward pass with or without mixed precision
                        if amp_enabled:
                            scaler.scale(loss).backward()
                            
                            # Step optimizer at accumulation boundaries
                            if (steps + 1) % gradient_accumulation_steps == 0 or (batch_idx == len(joslin_dataloaders[phase]) - 1):
                                # Apply gradient clipping if configured
                                if gradient_clipping > 0:
                                    scaler.unscale_(optimizer)
                                    torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), max_norm=gradient_clipping)
                                
                                # Update weights
                                scaler.step(optimizer)
                                scaler.update()
                        else:
                            loss.backward()
                            
                            # Step optimizer at accumulation boundaries
                            if (steps + 1) % gradient_accumulation_steps == 0 or (batch_idx == len(joslin_dataloaders[phase]) - 1):
                                # Apply gradient clipping if configured
                                if gradient_clipping > 0:
                                    torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), max_norm=gradient_clipping)
                                
                                # Update weights
                                optimizer.step()
                        
                        backward_time += time.time() - backward_start
                    
                    # Store original loss value for logging
                    loss_value = loss.item() * gradient_accumulation_steps if phase == 'train' else loss.item()
                    
                    # Update statistics
                    batch_size = inputs.size(0)
                    running_loss += loss_value * batch_size
                    running_corrects += torch.sum(preds == labels).item()
                    processed_samples += batch_size
                    
                    # Update confusion matrix
                    predicted_positive = (preds == 1)
                    actual_positive = (labels == 1)
                    
                    true_positive = torch.sum((predicted_positive) & (actual_positive)).item()
                    false_positive = torch.sum((predicted_positive) & (~actual_positive)).item()
                    false_negative = torch.sum((~predicted_positive) & (actual_positive)).item()
                    true_negative = torch.sum((~predicted_positive) & (~actual_positive)).item()
                    
                    # Accumulate confusion matrix elements
                    running_true_positive += true_positive
                    running_false_positive += false_positive
                    running_false_negative += false_negative
                    running_true_negative += true_negative
                    
                    # Store predictions for ROC/PR curves
                    if phase == 'val':
                        # Get probabilities
                        probs = torch.nn.functional.softmax(outputs, dim=1)[:, 1]  # Class 1 probability
                        
                        # Store for epoch-level metrics
                        epoch_predictions[phase]["outputs"].append(probs.detach().cpu())
                        epoch_predictions[phase]["targets"].append(labels.detach().cpu())
                    
                    # Update step counter
                    steps += 1
                    batches_processed[phase] += 1
                    
                    # Prepare for next batch
                    batch_start = time.time()
                
                # Update learning rate scheduler at end of training phase
                if phase == 'train':
                    scheduler.step()
                
                # Calculate epoch metrics for this process
                epoch_loss = running_loss / processed_samples if processed_samples > 0 else 0
                epoch_acc = running_corrects / processed_samples if processed_samples > 0 else 0
                
                # Gather confusion matrix values from all processes
                confusion_values = torch.tensor(
                    [running_true_positive, running_false_positive, running_false_negative, running_true_negative],
                    dtype=torch.float64,
                    device=device
                )
                
                # Synchronize metrics across processes
                if world_size > 1:
                    dist.all_reduce(confusion_values, op=dist.ReduceOp.SUM)
                
                # Extract aggregated confusion matrix values
                tp = confusion_values[0].item()
                fp = confusion_values[1].item()
                fn = confusion_values[2].item()
                tn = confusion_values[3].item()
                
                # Compute additional metrics from confusion matrix
                additional_metrics = compute_metrics_from_confusion_matrix(tp, fp, fn, tn)
                
                # Collect all predictions for ROC/AUC calculation
                if phase == 'val':
                    # Concatenate all outputs and targets for this epoch
                    all_outputs = torch.cat(epoch_predictions[phase]["outputs"]) if epoch_predictions[phase]["outputs"] else torch.tensor([])
                    all_targets = torch.cat(epoch_predictions[phase]["targets"]) if epoch_predictions[phase]["targets"] else torch.tensor([])
                    
                    # Gather predictions from all processes if distributed
                    if world_size > 1:
                        # Create list to gather outputs and targets
                        gathered_outputs = [torch.zeros_like(all_outputs) for _ in range(world_size)]
                        gathered_targets = [torch.zeros_like(all_targets) for _ in range(world_size)]
                        
                        # Gather from all processes
                        dist.all_gather(gathered_outputs, all_outputs)
                        dist.all_gather(gathered_targets, all_targets)
                        
                        # Combine gathered tensors
                        all_outputs = torch.cat(gathered_outputs)
                        all_targets = torch.cat(gathered_targets)
                    
                    # Calculate ROC AUC if we have both classes
                    if torch.unique(all_targets).numel() > 1:
                        try:
                            roc_auc = roc_auc_score(all_targets.numpy(), all_outputs.numpy())
                        except Exception as e:
                            if rank == 0:
                                logger.warning(f"Error calculating ROC AUC: {e}")
                            roc_auc = 0.0
                    else:
                        roc_auc = 0.5  # Default for single class
                else:
                    # For training phase, use simpler AUC calculation
                    roc_auc = additional_metrics.get('auc', 0.5)
                
                # Get metrics for this phase
                phase_results = {
                    'loss': epoch_loss,
                    'acc': additional_metrics['accuracy'],
                    'sensitivity': additional_metrics['sensitivity'],
                    'specificity': additional_metrics['specificity'],
                    'f1_score': additional_metrics['f1_score'],
                    'balanced_accuracy': additional_metrics['balanced_accuracy'],
                    'auc': roc_auc,
                    'confusion_matrix': {
                        'TP': tp,
                        'FP': fp,
                        'FN': fn,
                        'TN': tn
                    }
                }
                
                # Store metrics for this phase
                epoch_results[phase] = phase_results
                
                # Update training monitor
                if rank == 0 and training_monitor:
                    cm_data = phase_results['confusion_matrix']
                    
                    # Add predictions to monitor for ROC/PR curves (validation only)
                    if phase == 'val' and all_outputs.numel() > 0 and all_targets.numel() > 0:
                        training_monitor.update_predictions(epoch, phase, all_outputs, all_targets)
                    
                    # Add confusion matrix data
                    training_monitor.update_confusion_matrix(epoch, phase, cm_data)
                
                # Log results (rank 0 only)
                if rank == 0:
                    phase_duration = time.time() - phase_start_time
                    
                    logger.info(f'{phase} Loss: {phase_results["loss"]:.4f} Acc: {phase_results["acc"]:.4f} '
                                f'F1: {phase_results["f1_score"]:.4f}')
                    logger.info(f'{phase} Balanced Acc: {phase_results["balanced_accuracy"]:.4f} '
                                f'AUC-ROC: {phase_results["auc"]:.4f}')
                    logger.info(f'{phase} Sensitivity: {phase_results["sensitivity"]:.4f} '
                                f'Specificity: {phase_results["specificity"]:.4f}')
                    logger.info(f'{phase} Confusion Matrix: TP={tp}, FP={fp}, FN={fn}, TN={tn}')
                    logger.info(f'{phase} completed in {phase_duration:.2f}s')
                    
                    # Log timing information for training phase
                    if phase == 'train':
                        logger.info(f'Data loading time: {data_time:.2f}s, Forward time: {forward_time:.2f}s, '
                                    f'Backward time: {backward_time:.2f}s')
                    
                    # Log cache stats if available
                    try:
                        if hasattr(joslin_dataloaders[phase].dataset, 'get_cache_stats'):
                            stats = joslin_dataloaders[phase].dataset.get_cache_stats()
                            logger.info(f'{phase} cache: {stats["cache_size"]}/{stats["cache_size"] + 1} items, '
                                        f'hit rate: {stats["hit_rate"]:.2f}%')
                    except Exception as e:
                        logger.debug(f"Error getting cache stats: {str(e)}")
            
            # Update training monitor with all metrics for this epoch
            if rank == 0 and training_monitor:
                training_monitor.update_metrics(epoch, epoch_results)
                
                # Generate plots every 2 epochs and at the end
                if epoch % 2 == 0 or epoch == config.exp.num_epochs - 1:
                    training_monitor.generate_all_plots()
            
            # Check early stopping
            if early_stopping_params and rank == 0:
                from utils_parallel import EarlyStopping
                
                # Initialize early stopping if not already done
                if 'early_stopping' not in locals():
                    early_stopping = EarlyStopping(
                        patience=early_stopping_params['patience'],
                        mode=early_stopping_params['mode'],
                        min_delta=early_stopping_params['min_delta'],
                        verbose=early_stopping_params['verbose']
                    )
                
                # Get monitoring metric
                metric_name = early_stopping_params.get('metric', 'balanced_accuracy')
                if metric_name in epoch_results.get('val', {}):
                    metric_value = epoch_results['val'][metric_name]
                    
                    # Check if training should stop
                    if early_stopping(epoch, metric_value):
                        logger.info(f"Early stopping triggered at epoch {epoch}")
                        break
            
            # Handle checkpointing (rank 0 only)
            if rank == 0 and checkpoint_dir:
                # Save epoch checkpoint
                epoch_path = os.path.join(checkpoint_dir, f"epoch_{epoch}.pth")
                try:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': ddp_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_loss': epoch_results['train']['loss'],
                        'val_loss': epoch_results['val']['loss'],
                        'train_acc': epoch_results['train']['acc'],
                        'val_acc': epoch_results['val']['acc'],
                        'train_f1': epoch_results['train']['f1_score'],
                        'val_f1': epoch_results['val']['f1_score'],
                        'train_balanced_acc': epoch_results['train']['balanced_accuracy'],
                        'val_balanced_acc': epoch_results['val']['balanced_accuracy'],
                        'train_auc': epoch_results['train']['auc'],
                        'val_auc': epoch_results['val']['auc']
                    }, epoch_path)
                    logger.info(f"Saved epoch checkpoint at {epoch_path}")
                except Exception as e:
                    logger.error(f"Error saving epoch checkpoint: {str(e)}")
            
            # Log epoch summary (rank 0 only)
            if rank == 0:
                epoch_time = time.time() - epoch_start_time
                logger.info(f"Epoch {epoch} summary:")
                logger.info(f" Train Loss: {epoch_results['train']['loss']:.4f}, "
                            f"Train Acc: {epoch_results['train']['acc']:.4f}, "
                            f"Train F1: {epoch_results['train']['f1_score']:.4f}")
                logger.info(f" Train Balanced Acc: {epoch_results['train']['balanced_accuracy']:.4f}, "
                            f"Train AUC: {epoch_results['train']['auc']:.4f}")
                logger.info(f" Train Sensitivity: {epoch_results['train']['sensitivity']:.4f}, "
                            f"Train Specificity: {epoch_results['train']['specificity']:.4f}")
                logger.info(f" Val Loss: {epoch_results['val']['loss']:.4f}, "
                            f"Val Acc: {epoch_results['val']['acc']:.4f}, "
                            f"Val F1: {epoch_results['val']['f1_score']:.4f}")
                logger.info(f" Val Balanced Acc: {epoch_results['val']['balanced_accuracy']:.4f}, "
                            f"Val AUC: {epoch_results['val']['auc']:.4f}")
                logger.info(f" Val Sensitivity: {epoch_results['val']['sensitivity']:.4f}, "
                            f"Val Specificity: {epoch_results['val']['specificity']:.4f}")
                logger.info(f" Total Epoch Time: {epoch_time:.2f}s")
            
            # Synchronize processes at end of epoch
            if world_size > 1:
                dist.barrier()
        
        # Final summary (rank 0 only)
        if rank == 0:
            total_time = time.time() - total_start_time
            hours, remainder = divmod(total_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            logger.info("========== Training Completed ==========")
            logger.info(f"Training complete in {int(hours):02d}:{int(minutes):02d}:{seconds:.2f}")
            
            # Generate final visualizations
            if training_monitor:
                training_monitor.generate_all_plots()
    
    except Exception as e:
        if rank == 0:
            logger.error(f"Error during training: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    
    # Clean up distributed environment
    cleanup()
    
    return ddp_model

def train_wrapper(rank, world_size, config): 
    """
    Wrapper function for train_on_device
    """
    return train_on_device(rank, world_size, config) 

def merge_experiment_config(base_config, experiment_name=None):
    """
    Merge base configuration with experiment-specific configuration.
    
    Args:
        base_config: Base configuration
        experiment_name: Name of experiment to use (e.g., "experiment_focal_loss")
        
    Returns:
        Merged configuration
    """
    if experiment_name is None:
        return base_config
    
    # Check if experiment configuration exists
    if not hasattr(base_config, experiment_name):
        print(f"Warning: Experiment configuration '{experiment_name}' not found.")
        return base_config
    
    # Get experiment configuration
    experiment_config = getattr(base_config, experiment_name)
    
    # Create a deep copy of the base config to avoid modifying the original
    merged_config = OmegaConf.create(OmegaConf.to_container(base_config, resolve=True))
    
    # Merge experiment configuration into base configuration
    experiment_dict = OmegaConf.to_container(experiment_config, resolve=True)
    
    # Update sections recursively
    for section, values in experiment_dict.items():
        if isinstance(values, dict):
            # If section exists in base config, update it
            if not hasattr(merged_config, section):
                setattr(merged_config, section, OmegaConf.create({}))
            
            section_config = getattr(merged_config, section)
            
            # Update each key in the section
            for key, value in values.items():
                # Handle nested dictionaries
                if isinstance(value, dict) and hasattr(section_config, key) and isinstance(getattr(section_config, key), dict):
                    nested_base = getattr(section_config, key)
                    nested_merged = {**nested_base, **value}
                    setattr(section_config, key, OmegaConf.create(nested_merged))
                else:
                    setattr(section_config, key, value)
        else:
            # Direct assignment for top-level values
            setattr(merged_config, section, values)
    
    return merged_config

@hydra.main(config_path='config', config_name='pretrain_parallel', version_base="1.3") 
def run(config): 
    """
    Main entry point with improved distributed training setup
    """
    # Get experiment name if specified
    experiment_name = os.environ.get('EXPERIMENT', None)
    
    if experiment_name:
        # Merge configurations
        config = merge_experiment_config(config, experiment_name)
        print(f"Running experiment: {experiment_name}")
    
    # Handle checkpoint directory
    if hasattr(config.exp, 'checkpoint_dir'): 
        checkpoint_dir = config.exp.checkpoint_dir 
        
        # Create parent directories if they don't exist
        os.makedirs(os.path.dirname(checkpoint_dir), exist_ok=True)
        
        # Check if directory exists and clear it
        if os.path.exists(checkpoint_dir): 
            logging.info(f"Checkpoint directory exists at {checkpoint_dir}, clearing contents...") 
            safe_clear_directory(checkpoint_dir) 
            logging.info(f"Checkpoint directory cleared successfully.") 
        
        # Create the directory (or recreate if just cleared)
        os.makedirs(checkpoint_dir, exist_ok=True) 
        logging.info(f"Checkpoint directory prepared at {checkpoint_dir}") 
    
    # Create visualization directory if specified
    if hasattr(config.exp, 'visualization_dir'):
        os.makedirs(config.exp.visualization_dir, exist_ok=True)
    
    # Configure main logging
    main_log_file = os.path.join( 
        config.exp.checkpoint_dir if hasattr(config.exp, 'checkpoint_dir') else '.', 
        'pretrain_parallel.log'
    ) 
    
    logging.basicConfig( 
        level=logging.INFO, 
        format='[%(asctime)s][%(levelname)s] - %(message)s', 
        datefmt='%Y-%m-%d %H:%M:%S', 
        handlers=[ 
            logging.FileHandler(main_log_file), 
            logging.StreamHandler() 
        ], 
        force=True
    ) 
    
    # Log configuration summary
    logging.info("Starting training with configuration summary:") 
    
    # Log configuration in a controlled, minimal way
    for section_name in ['exp', 'data', 'model', 'optimizer', 'lr_scheduler', 'distributed', 'early_stopping', 'criterion']: 
        if hasattr(config, section_name): 
            section = getattr(config, section_name) 
            logging.info(f" {section_name}:") 
            for key, value in vars(section).items(): 
                logging.info(f"   {key}: {value}") 
    
    # Check available GPUs
    num_gpus = torch.cuda.device_count() 
    logging.info(f"Number of available GPUs: {num_gpus}") 
    
    if num_gpus == 0: 
        logging.warning("No GPUs found! Training will be very slow on CPU.") 
    else: 
        logging.info("Available GPUs:") 
        for i in range(num_gpus): 
            logging.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}") 
    
    # Determine optimal world size (number of GPUs to use)
    world_size = min(num_gpus, config.distributed.world_size) if hasattr(config.distributed, 'world_size') else num_gpus 
    
    if world_size < 1: 
        world_size = 1
        logging.warning("Setting world_size to 1 (CPU-only training)") 
    
    # Set multiprocessing start method
    try: 
        if world_size > 1: 
            mp.set_start_method('spawn', force=True) 
    except RuntimeError: 
        logging.warning("Failed to set multiprocessing start method to 'spawn', it may have already been set") 
    
    # Run training
    try: 
        if world_size > 1: 
            logging.info(f"Training with {world_size} GPUs using DistributedDataParallel") 
            # Use multiprocessing to spawn multiple processes
            mp.spawn( 
                train_wrapper, 
                args=(world_size, config), 
                nprocs=world_size, 
                join=True
            ) 
            
            logging.info("Multi-GPU training completed successfully") 
        else: 
            logging.info("Training with single GPU or CPU") 
            # Call the training function directly
            train_on_device(0, 1, config) 
            logging.info("Single-device training completed successfully") 
        
        # Add a clear end marker to the log
        logging.info("==========================================") 
        logging.info("TRAINING SCRIPT EXECUTION COMPLETED") 
        logging.info("==========================================") 
        
    except Exception as e: 
        logging.error(f"Error during training: {str(e)}") 
        import traceback 
        logging.error(traceback.format_exc()) 

if __name__ == '__main__': 
    # Only execute run() once when the script is called
    run()
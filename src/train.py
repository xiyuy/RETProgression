import hydra
import os, logging, time, torch, sys, numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from timm import create_model
from omegaconf import OmegaConf
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Import custom modules
from datasets import JoslinData, get_transforms
from loss_functions import get_loss_function
from lr_schedulers import create_scheduler
from utils import (
    train_model_custom_progress, CachedDataset, 
    compute_metrics_from_confusion_matrix, EarlyStopping
)
from custom_metrics import roc_auc_score, balanced_accuracy_score

def safe_clear_directory(directory_path, logger=None):
    """Safely clear contents of a directory without deleting the directory itself."""
    if not os.path.exists(directory_path):
        return
    
    log_fn = logger.info if logger else logging.info
    log_fn(f"Clearing contents of directory: {directory_path}")
    
    for item in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item)
        try:
            if os.path.isfile(item_path):
                os.unlink(item_path)
            elif os.path.isdir(item_path):
                import shutil
                shutil.rmtree(item_path)
        except Exception as e:
            error_msg = f"Error clearing item {item_path}: {str(e)}"
            (logger.error if logger else logging.error)(error_msg)

def setup(rank, world_size, config):
    """Initialize the distributed environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(config.exp.master_port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    if rank == 0:
        logging.info(f"Initialized process group with world_size={world_size}")

def cleanup():
    """Clean up distributed environment."""
    dist.destroy_process_group()

def configure_rank_logger(rank, log_dir):
    """Configure process-specific logging."""
    log_file = os.path.join(log_dir, f'rank_{rank}_training.log')
    
    logger = logging.getLogger()
    logger.handlers = []  # Remove existing handlers
    
    file_handler = logging.FileHandler(log_file)
    file_formatter = logging.Formatter(
        f'[%(asctime)s][%(levelname)s][Rank {rank}] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    if rank == 0:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(file_formatter)
        logger.addHandler(console_handler)
    
    logger.setLevel(logging.INFO)
    return logger

def train_on_device(rank, world_size, config):
    """Training function to run on each GPU."""
    # Setup distributed environment
    setup(rank, world_size, config)
    
    # Set up process-specific logging
    logger = configure_rank_logger(rank, config.exp.checkpoint_dir)
    
    # Set device for this process
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    
    # Log basic configuration
    if rank == 0:
        logger.info(f"Running training on rank {rank} with world size {world_size}")
        logger.info(f"Training for {config.exp.num_epochs} epochs")
        logger.info(f"Batch size: {config.data.batch_size} (per GPU)")
        logger.info(f"Total effective batch size: {config.data.batch_size * world_size}")
        logger.info(f"Learning rate: {config.optimizer.lr}")
        logger.info(f"Experiment name: {getattr(config.exp, 'experiment_name', 'default')}")
    
    # Get data augmentation settings
    augmentation_enabled = getattr(config.data, 'augmentation', {}).get('enabled', True)
    augmentation_strength = getattr(config.data, 'augmentation', {}).get('strength', 'moderate')
    resolution = getattr(config.data, 'resolution', 224)
    
    if not augmentation_enabled:
        augmentation_strength = 'none'
    
    if rank == 0:
        logger.info(f"Data augmentation: {augmentation_strength}")
        logger.info(f"Image resolution: {resolution}x{resolution}")
    
    # Get transforms based on augmentation settings
    transforms_dict = get_transforms(augmentation_strength, resolution=resolution)
    train_transform = transforms_dict['train']
    val_transform = transforms_dict['val']
    
    # Define GPU normalization transform
    gpu_transform = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    # Dataset creation
    if rank == 0:
        logger.info("Creating datasets...")
    
    dataset_start_time = time.time()
    
    try:
        # Get cache settings
        train_cache_size = getattr(config.caching, 'train_cache_size', 1000) if hasattr(config, 'caching') else 1000
        val_cache_size = getattr(config.caching, 'val_cache_size', 1000) if hasattr(config, 'caching') else 1000
        cache_enabled = getattr(config.caching, 'enabled', True) if hasattr(config, 'caching') else True
        
        # Create base datasets
        base_datasets = {
            "train": JoslinData(
                data_dir=config.data.data_dir,
                annotations_file=config.data.annotations_file_name + "train.csv",
                img_dir="Exports_02052025", # default: Exports_02052025
                transform=train_transform
            ),
            "val": JoslinData(
                data_dir=config.data.data_dir,
                annotations_file=config.data.annotations_file_name + "val.csv",
                img_dir="Exports_02052025", # default: Exports_02052025
                transform=val_transform
            )
        }
        
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
    
    # Create samplers for distributed training
    samplers = {
        "train": DistributedSampler(
            joslin_data["train"],
            num_replicas=world_size,
            rank=rank,
            shuffle=config.data.shuffle,
            seed=42  # Fixed seed for reproducibility
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
    
    # DataLoader creation
    if rank == 0:
        logger.info("Creating dataloaders...")
    
    dataloader_start_time = time.time()
    
    try:
        # Get dataloader settings
        prefetch_factor = getattr(config.caching, 'prefetch_factor', 4) if hasattr(config, 'caching') else 4
        drop_last = getattr(config.optimization, 'drop_last', True) if hasattr(config, 'optimization') else True
        
        # Adjust batch size based on resolution if needed
        resolution = getattr(config.data, 'resolution', 224)
        original_batch_size = config.data.batch_size
        
        # Optional automatic batch size adjustment (commented out)
        # if resolution > 224 and not hasattr(config.data, 'adjusted_batch_size'):
        #     scale_factor = (resolution / 224) ** 2
        #     adjusted_batch_size = max(1, int(original_batch_size / scale_factor))
        #     if adjusted_batch_size < original_batch_size and rank == 0:
        #         logger.warning(f"Automatically reducing batch size from {original_batch_size} to {adjusted_batch_size} "
        #                        f"due to higher resolution ({resolution}x{resolution})")
        #         logger.warning("Set data.adjusted_batch_size=False in config to disable this behavior")
        #     config.data.batch_size = adjusted_batch_size
        
        dataloader_kwargs = {
            'batch_size': config.data.batch_size,
            'shuffle': False,  # We're using DistributedSampler instead
            'num_workers': workers_per_gpu,
            'pin_memory': True,
            'persistent_workers': True if workers_per_gpu > 0 else False,
            'drop_last': drop_last
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
    
    # Model creation
    if rank == 0:
        logger.info(f"Creating model: {config.model.name}...")
    
    model_start_time = time.time()
    
    try:
        # Get model-specific resolution or use data resolution
        img_size = getattr(config.model, 'img_size', getattr(config.data, 'resolution', 224))
        
        # Create model with configurable image size
        if 'vit' or 'swinv2' in config.model.name.lower():
            model = create_model(
                config.model.name,
                pretrained=config.model.pretrained,
                num_classes=config.model.num_classes,
                img_size=img_size  # Pass image size to ViT model
            )
            if rank == 0:
                logger.info(f"Created ViT model with image size: {img_size}x{img_size}")
        else:
            # For non-ViT models
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
            broadcast_buffers=False,
            gradient_as_bucket_view=True
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
    
    # Configure loss function
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
    
    # Create learning rate scheduler
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
    
    # Get training configuration
    gradient_accumulation_steps = getattr(config.optimization, 'gradient_accumulation_steps', 1) if hasattr(config, 'optimization') else 1
    amp_enabled = getattr(config.optimization, 'amp_enabled', True) if hasattr(config, 'optimization') else True
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
    
    # Use the optimized training function
    try:
        train_model_custom_progress(
            dataloaders=joslin_dataloaders,
            dataset_sizes=dataset_sizes,
            model=ddp_model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            gpu_transform=gpu_transform,
            train_sampler=samplers["train"],
            val_sampler=samplers["val"],
            rank=rank,
            world_size=world_size,
            num_epochs=config.exp.num_epochs,
            checkpoint_path=checkpoint_path,
            profiler=None,
            gradient_accumulation_steps=gradient_accumulation_steps,
            amp_enabled=amp_enabled,
            gradient_clipping=gradient_clipping,
            progress_log_freq=5,
            early_stopping_params=early_stopping_params
        )
        
        if rank == 0:
            logger.info("Training completed successfully")
                
    except Exception as e:
        if rank == 0:
            logger.error(f"Error during training: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    
    # Clean up distributed environment
    cleanup()
    
    return ddp_model

def train_wrapper(rank, world_size, config):
    """Wrapper function for train_on_device"""
    return train_on_device(rank, world_size, config)

def merge_experiment_config(base_config, experiment_name=None):
    """Merge base configuration with experiment-specific configuration."""
    if experiment_name is None:
        return base_config
    
    # Check if experiment configuration exists
    if not hasattr(base_config, experiment_name):
        print(f"Warning: Experiment configuration '{experiment_name}' not found.")
        return base_config
    
    # Create deep copy of base config and merge with experiment config
    merged_config = OmegaConf.create(OmegaConf.to_container(base_config, resolve=True))
    experiment_dict = OmegaConf.to_container(getattr(base_config, experiment_name), resolve=True)
    
    # Update sections recursively
    for section, values in experiment_dict.items():
        if isinstance(values, dict):
            # Ensure section exists in merged config
            if not hasattr(merged_config, section):
                setattr(merged_config, section, OmegaConf.create({}))
            
            section_config = getattr(merged_config, section)
            
            # Update each key in the section
            for key, value in values.items():
                if isinstance(value, dict) and hasattr(section_config, key) and isinstance(getattr(section_config, key), dict):
                    # Handle nested dictionaries
                    nested_base = getattr(section_config, key)
                    nested_merged = {**nested_base, **value}
                    setattr(section_config, key, OmegaConf.create(nested_merged))
                else:
                    setattr(section_config, key, value)
        else:
            # Direct assignment for top-level values
            setattr(merged_config, section, values)
    
    return merged_config

@hydra.main(config_path='config', config_name='train', version_base="1.3")
def run(config):
    """Main entry point with improved distributed training setup"""
    # Get experiment name if specified
    experiment_name = os.environ.get('EXPERIMENT', None)
    
    if experiment_name:
        # Merge configurations
        config = merge_experiment_config(config, experiment_name)
        print(f"Running experiment: {experiment_name}")
    
    # Log resolution information
    resolution = getattr(config.data, 'resolution', 224)
    print(f"Using image resolution: {resolution}x{resolution}")
    
    # Handle checkpoint directory
    if hasattr(config.exp, 'checkpoint_dir'):
        checkpoint_dir = config.exp.checkpoint_dir
        
        # Create parent directories if they don't exist
        os.makedirs(os.path.dirname(checkpoint_dir), exist_ok=True)
        
        # Check if directory exists and clear it
        if os.path.exists(checkpoint_dir) and config.exp.checkpoint_name is None:
            logging.info(f"Checkpoint directory exists at {checkpoint_dir}, clearing contents...")
            safe_clear_directory(checkpoint_dir)
            logging.info(f"Checkpoint directory cleared successfully.")
        
        # Create the directory (or recreate if just cleared)
        os.makedirs(checkpoint_dir, exist_ok=True)
        logging.info(f"Checkpoint directory prepared at {checkpoint_dir}")
    
    # Configure main logging
    main_log_file = os.path.join(
        config.exp.checkpoint_dir if hasattr(config.exp, 'checkpoint_dir') else '.',
        'train.log'
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
    for section_name in ['exp', 'data', 'model', 'optimizer', 'lr_scheduler', 'distributed', 'early_stopping', 'criterion']:
        if hasattr(config, section_name):
            section = getattr(config, section_name)
            logging.info(f" {section_name}:")
            for key, value in vars(section).items():
                logging.info(f" {key}: {value}")
    
    # Check available GPUs
    num_gpus = torch.cuda.device_count()
    logging.info(f"Number of available GPUs: {num_gpus}")
    
    if num_gpus == 0:
        logging.warning("No GPUs found! Training will be very slow on CPU.")
    else:
        logging.info("Available GPUs:")
        for i in range(num_gpus):
            logging.info(f" GPU {i}: {torch.cuda.get_device_name(i)}")
    
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
        logging.info("=" * 50)
        logging.info("TRAINING SCRIPT EXECUTION COMPLETED")
        logging.info("=" * 50)
    
    except Exception as e:
        logging.error(f"Error during training: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())

if __name__ == '__main__':
    # Only execute run() once when the script is called
    run()
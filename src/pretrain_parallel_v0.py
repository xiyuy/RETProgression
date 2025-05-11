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
from datasets import JoslinData 
from utils_parallel import train_model_custom_progress, CachedDataset 
import time 
import sys 
import numpy as np
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
    os.environ['MASTER_PORT'] = '12355'
    
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
    
    # Only rank 0 logs detailed information to reduce overhead
    if rank == 0: 
        logger.info(f"Running training on rank {rank} with world size {world_size}") 
        logger.info(f"Training for {config.exp.num_epochs} epochs") 
        logger.info(f"Batch size: {config.data.batch_size} (per GPU)") 
        logger.info(f"Total effective batch size: {config.data.batch_size * world_size}") 
        logger.info(f"Learning rate: {config.optimizer.lr}") 
        logger.info(f"Weight decay: {config.optimizer.weight_decay}") 
    
    # Enhanced data augmentation pipeline
    if rank == 0:
        logger.info("Setting up enhanced data augmentation pipeline...")
    
    # Define training transforms with enhanced augmentation
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
    ])

    # Define simpler validation transforms
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Keep the GPU normalization transform separate
    gpu_transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.2)),  # Move here to ensure tensor input
    ])
    
    if rank == 0:
        logger.info("Data augmentation settings:")
        logger.info(" - RandomResizedCrop: scale=(0.75, 1.0)")
        logger.info(" - RandomRotation: 30 degrees")
        logger.info(" - ColorJitter: brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1")
        logger.info(" - RandomAffine: translate=(0.1, 0.1), scale=(0.9, 1.1)")
        logger.info(" - RandomErasing: p=0.2, scale=(0.02, 0.2)")

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
                transform=train_transform  # Use enhanced augmentation for training
            ),
            "val": JoslinData(
                data_dir=config.data.data_dir,
                annotations_file=config.data.annotations_file_name + "val.csv",
                img_dir="Exports_02052025",
                transform=val_transform    # Use simpler transform for validation
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
                    rank=rank, # Pass rank for per-process caching
                    world_size=world_size # Pass world_size for coordination
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
    
    # Configure optimal number of workers per GPU - increased for better performance
    workers_per_gpu = min(16, max(4, config.data.num_workers)) 
    
    # DataLoader creation with performance optimizations
    if rank == 0: 
        logger.info("Creating dataloaders...") 
    
    dataloader_start_time = time.time() 
    
    try: 
        # Get prefetch factor from config or use default - increased for better performance
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
            find_unused_parameters=find_unused_parameters, # Improves performance if False
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
    
    # Configure loss with class weighting (for class imbalance 93:7)
    if hasattr(config, 'criterion') and hasattr(config.criterion, 'class_weights') and config.criterion.class_weights:
        # Calculate class weights based on 93:7 distribution
        class_weights = torch.tensor([1.0, 13.0]).to(device)  # Approximate weight for 93:7 distribution
        if rank == 0:
            logger.info(f"Using weighted loss with class weights: {class_weights}")
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    
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
    
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=config.lr_scheduler.step_size, 
        gamma=config.lr_scheduler.gamma 
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
    
    # Check for profiling configuration
    profiling_enabled = False
    if hasattr(config, 'profiling'): 
        profiling_enabled = getattr(config.profiling, 'enabled', False) 
    
    # Simple custom monitoring system to replace PyTorch profiler
    if profiling_enabled and rank == 0: 
        logger.info("Custom performance monitoring enabled for rank 0") 
    
    try: 
        # Set up a simple custom timer-based profiling
        training_stats = { 
            'data_loading_time': 0.0, 
            'forward_time': 0.0, 
            'backward_time': 0.0, 
            'total_time': 0.0, 
            'gpu_util': [], 
            'memory_used': [] 
        } 
        
        # Configure manual memory monitoring
        def monitor_gpu(): 
            if torch.cuda.is_available(): 
                mem_allocated = torch.cuda.memory_allocated(device) / (1024 ** 2) 
                mem_reserved = torch.cuda.memory_reserved(device) / (1024 ** 2) 
                return { 
                    'allocated': mem_allocated, 
                    'reserved': mem_reserved 
                } 
            return {'allocated': 0, 'reserved': 0} 
        
        # Log initial GPU state
        gpu_stats = monitor_gpu() 
        logger.info(f"Initial GPU memory: {gpu_stats['allocated']:.1f}MB allocated, {gpu_stats['reserved']:.1f}MB reserved") 
        
        # Define progress logging frequency
        progress_log_freq = 5 # Log progress every 5% or 10 batches, whichever is more frequent
        
        # Train with custom progress tracking
        ddp_model = train_model_custom_progress(
            joslin_dataloaders, 
            dataset_sizes, 
            ddp_model, 
            criterion, 
            optimizer, 
            scheduler, 
            device, 
            gpu_transform=gpu_transform, 
            train_sampler=samplers["train"], 
            val_sampler=samplers["val"], 
            rank=rank, 
            world_size=world_size, 
            num_epochs=config.exp.num_epochs, 
            checkpoint_path=checkpoint_path, 
            profiler=None, # No PyTorch profiler
            gradient_accumulation_steps=gradient_accumulation_steps, 
            amp_enabled=amp_enabled, 
            gradient_clipping=gradient_clipping, 
            progress_log_freq=progress_log_freq, # Custom progress logging frequency
            early_stopping_params=early_stopping_params  # Add early stopping params
        ) 
        
        # Log final training stats
        training_stats['total_time'] = time.time() - total_start_time 
        gpu_stats = monitor_gpu() 
        logger.info(f"Final GPU memory: {gpu_stats['allocated']:.1f}MB allocated, {gpu_stats['reserved']:.1f}MB reserved") 
        logger.info(f"Total training time: {training_stats['total_time']:.2f}s") 
        
    except Exception as e: 
        logger.error(f"Error during training with monitoring: {str(e)}") 
        import traceback 
        logger.error(traceback.format_exc()) 
    
    if rank == 0: 
        total_time = time.time() - total_start_time 
        hours, remainder = divmod(total_time, 3600) 
        minutes, seconds = divmod(remainder, 60) 
        logger.info(f"====== Training completed in {int(hours):02d}:{int(minutes):02d}:{seconds:.2f} ======") 
    
    # Clean up distributed environment
    cleanup() 
    
    return ddp_model 

def train_wrapper(rank, world_size, config): 
    """
    Wrapper function for train_on_device
    """
    return train_on_device(rank, world_size, config) 

@hydra.main(config_path='config', config_name='pretrain_parallel', version_base="1.3") 
def run(config): 
    """
    Main entry point with improved distributed training setup
    """
    # Handle checkpoint directory - clear if it exists
    if hasattr(config.exp, 'checkpoint_dir'): 
        checkpoint_dir = config.exp.checkpoint_dir 
        
        # Check if directory exists
        if os.path.exists(checkpoint_dir): 
            logging.info(f"Checkpoint directory exists at {checkpoint_dir}, clearing contents...") 
            safe_clear_directory(checkpoint_dir) 
            logging.info(f"Checkpoint directory cleared successfully.") 
        
        # Create the directory (or recreate if just cleared)
        os.makedirs(checkpoint_dir, exist_ok=True) 
        logging.info(f"Checkpoint directory prepared at {checkpoint_dir}") 
    
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
    
    # Log configuration summary (avoid excessive detail)
    logging.info("Starting training with configuration summary:") 
    
    # Log configuration in a controlled, minimal way
    for section_name in ['exp', 'data', 'model', 'optimizer', 'lr_scheduler', 'distributed', 'early_stopping', 'criterion']: 
        if hasattr(config, section_name): 
            section = getattr(config, section_name) 
            logging.info(f" {section_name}:") 
            for key, value in vars(section).items(): 
                logging.info(f" {key}: {value}") 
    
    # Log optimization settings if available
    if hasattr(config, 'optimization'): 
        logging.info(" optimization:") 
        for key, value in vars(config.optimization).items(): 
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
    
    # Run training only once
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
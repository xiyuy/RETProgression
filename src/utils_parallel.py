import logging 
import os 
import time 
import torch 
import random 
import copy 
import threading 
import torch.distributed as dist 
from torch.amp import GradScaler, autocast 
from collections import defaultdict 
import numpy as np
from custom_metrics import roc_auc_score, balanced_accuracy_score

# Improved caching mechanism with per-process storage
class ProcessLocalCache: 
    """Singleton cache manager that maintains separate caches for each process/rank"""
    _instance = None
    _lock = threading.Lock() 
    
    def __new__(cls): 
        with cls._lock: 
            if cls._instance is None: 
                cls._instance = super(ProcessLocalCache, cls).__new__(cls) 
                cls._instance.rank_caches = defaultdict(dict) 
                cls._instance.stats = defaultdict(lambda: {"hits": 0, "requests": 0}) 
        return cls._instance 
    
    def get(self, rank, key, default=None): 
        """Get an item from the cache for a specific rank"""
        self.stats[rank]["requests"] += 1
        if key in self.rank_caches[rank]: 
            self.stats[rank]["hits"] += 1
            return self.rank_caches[rank][key] 
        return default 
    
    def set(self, rank, key, value, max_size=1000): 
        """Store an item in the cache for a specific rank with size limit"""
        if len(self.rank_caches[rank]) >= max_size: 
            return False
        self.rank_caches[rank][key] = value 
        return True
    
    def clear(self, rank=None): 
        """Clear cache for a specific rank or all caches if rank is None"""
        if rank is not None: 
            self.rank_caches[rank].clear() 
            self.stats[rank] = {"hits": 0, "requests": 0} 
        else: 
            self.rank_caches.clear() 
            self.stats.clear() 
    
    def size(self, rank=None): 
        """Get cache size for a specific rank or total size if rank is None"""
        if rank is not None: 
            return len(self.rank_caches[rank]) 
        return sum(len(cache) for cache in self.rank_caches.values()) 
    
    def get_stats(self, rank): 
        """Get statistics for a specific rank"""
        stats = self.stats[rank] 
        hit_rate = 0
        if stats["requests"] > 0: 
            hit_rate = (stats["hits"] / stats["requests"]) * 100
        
        return { 
            'cache_size': len(self.rank_caches[rank]), 
            'total_cache_size': self.size(), 
            'cache_hits': stats["hits"], 
            'total_requests': stats["requests"], 
            'hit_rate': hit_rate 
        } 
    
    def reset_stats(self, rank=None): 
        """Reset statistics for a specific rank or all ranks"""
        if rank is not None: 
            self.stats[rank] = {"hits": 0, "requests": 0} 
        else: 
            for rank in self.stats: 
                self.stats[rank] = {"hits": 0, "requests": 0} 

# Create global cache instance
PROCESS_CACHE = ProcessLocalCache() 

class CachedDataset(torch.utils.data.Dataset): 
    """
    Enhanced dataset wrapper that caches data in memory with process-local storage
    to avoid unnecessary communication in distributed training.
    """
    def __init__(self, dataset, cache_size=1000, cache_probability=1.0, rank=0, world_size=1): 
        """
        Initialize the CachedDataset with process awareness.
        
        Args:
            dataset: The original dataset to wrap
            cache_size: Maximum number of items to cache in memory per process
            cache_probability: Probability of caching an item (0.0 to 1.0)
            rank: Process rank for distributed training
            world_size: Total number of processes
        """
        self.dataset = dataset 
        self.cache_size = cache_size 
        self.cache_probability = cache_probability 
        self.rank = rank 
        self.world_size = world_size 
        
        # Ensure cache is initialized for this rank
        PROCESS_CACHE.reset_stats(self.rank) 
        
        if self.rank == 0: 
            logging.info(f"CachedDataset initialized for rank {rank} with cache_size={cache_size}") 
        
    def __getitem__(self, index): 
        """
        Get an item from the dataset, using process-local cache if available.
        """
        # Generate a unique cache key using the dataset object ID and index
        cache_key = f"{id(self.dataset)}_{index}"
        
        # Check process-local cache
        cached_item = PROCESS_CACHE.get(self.rank, cache_key) 
        if cached_item is not None: 
            return cached_item 
        
        # Cache miss - get item from dataset
        item = self.dataset[index] 
        
        # Determine if this item should be cached based on probability
        if random.random() < self.cache_probability: 
            try: 
                # Clone tensors to avoid reference issues
                if isinstance(item, tuple): 
                    cached_item = [] 
                    for x in item: 
                        if torch.is_tensor(x): 
                            cached_item.append(x.clone().detach()) 
                        else: 
                            cached_item.append(copy.deepcopy(x)) 
                    
                    # Store in process-local cache
                    PROCESS_CACHE.set(self.rank, cache_key, tuple(cached_item), self.cache_size) 
                elif torch.is_tensor(item): 
                    PROCESS_CACHE.set(self.rank, cache_key, item.clone().detach(), self.cache_size) 
                else: 
                    PROCESS_CACHE.set(self.rank, cache_key, copy.deepcopy(item), self.cache_size) 
                
                # Log occasional cache stats (minimized to reduce overhead)
                if self.rank == 0 and PROCESS_CACHE.size(self.rank) % 100 == 0: 
                    stats = PROCESS_CACHE.get_stats(self.rank) 
                    logging.debug(f"Rank {self.rank} cache: {stats['cache_size']}/{self.cache_size} items") 
            except Exception as e: 
                logging.debug(f"Failed to cache item {index}: {str(e)}") 
        
        return item 
    
    def __len__(self): 
        """
        Return the length of the dataset.
        """
        return len(self.dataset) 
    
    def get_cache_stats(self): 
        """
        Return cache performance statistics for this process.
        """
        return PROCESS_CACHE.get_stats(self.rank) 
    
    def reset_cache_stats(self): 
        """
        Reset the cache statistics for this process.
        """
        PROCESS_CACHE.reset_stats(self.rank) 
    
    @staticmethod
    def clear_cache(rank=None): 
        """
        Clear the cache for a specific rank or all ranks.
        """
        PROCESS_CACHE.clear(rank) 
        logging.info(f"Cleared cache for {'all ranks' if rank is None else f'rank {rank}'}") 

# CudaMemoryMonitor for leak detection and benchmarking
class CudaMemoryMonitor: 
    """Utility class to monitor CUDA memory usage during training"""
    
    def __init__(self, enabled=True, log_interval=5): 
        self.enabled = enabled and torch.cuda.is_available() 
        self.log_interval = log_interval 
        self.last_log_time = time.time() 
        
        if self.enabled: 
            self.baseline = self._get_memory_stats() 
            logging.info(f"CUDA memory monitor initialized: {self.baseline}") 
        
    def _get_memory_stats(self): 
        """Get current memory statistics"""
        if not self.enabled: 
            return {} 
        
        try: 
            # Check all devices
            stats = {} 
            for device_idx in range(torch.cuda.device_count()): 
                device_stats = torch.cuda.memory_stats(device_idx) 
                stats[device_idx] = { 
                    'allocated': device_stats.get('allocated_bytes.all.current', 0) / (1024 ** 2), 
                    'reserved': device_stats.get('reserved_bytes.all.current', 0) / (1024 ** 2), 
                    'active': device_stats.get('active_bytes.all.current', 0) / (1024 ** 2), 
                } 
            return stats 
        except Exception as e: 
            logging.warning(f"Error getting CUDA memory stats: {str(e)}") 
            return {} 
        
    def log_memory_stats(self, phase="", step=0, force=False): 
        """Log memory usage details"""
        if not self.enabled: 
            return
        
        current_time = time.time() 
        if not force and (current_time - self.last_log_time) < self.log_interval: 
            return
        
        self.last_log_time = current_time 
        current_stats = self._get_memory_stats() 
        
        log_message = f"CUDA Memory [{phase}][Step {step}]:"
        for device_idx, stats in current_stats.items(): 
            log_message += (f" Device {device_idx}: "
                          f"Allocated: {stats['allocated']:.1f} MB, "
                          f"Reserved: {stats['reserved']:.1f} MB") 
            
            # Calculate difference from baseline
            if device_idx in self.baseline: 
                allocated_diff = stats['allocated'] - self.baseline[device_idx]['allocated'] 
                log_message += f" (Δ: {allocated_diff:+.1f} MB)"
        
        logging.info(log_message) 
        
    def reset_baseline(self): 
        """Reset the baseline memory statistics"""
        if self.enabled: 
            self.baseline = self._get_memory_stats() 
            logging.info(f"Reset CUDA memory baseline: {self.baseline}") 

# Additional utility function to compute metrics from confusion matrix
def compute_metrics_from_confusion_matrix(true_positive, false_positive, false_negative, true_negative): 
    """
    Compute common classification metrics from confusion matrix elements.
    
    Args:
        true_positive: Number of true positives
        false_positive: Number of false positives
        false_negative: Number of false negatives
        true_negative: Number of true negatives
    
    Returns:
        Dictionary containing accuracy, precision, recall (sensitivity), 
        specificity, F1 score, and balanced accuracy
    """
    # Avoid division by zero
    epsilon = 1e-7
    
    # Total samples
    total = true_positive + false_positive + false_negative + true_negative 
    
    # Accuracy
    accuracy = (true_positive + true_negative) / max(total, epsilon) 
    
    # Precision
    precision = true_positive / max(true_positive + false_positive, epsilon) 
    
    # Recall (Sensitivity)
    sensitivity = true_positive / max(true_positive + false_negative, epsilon) 
    
    # Specificity
    specificity = true_negative / max(true_negative + false_positive, epsilon) 
    
    # F1 Score
    f1_score = 2 * precision * sensitivity / max(precision + sensitivity, epsilon) 
    
    # Balanced Accuracy
    balanced_acc = (sensitivity + specificity) / 2.0
    
    return { 
        'accuracy': accuracy, 
        'precision': precision, 
        'sensitivity': sensitivity, 
        'specificity': specificity, 
        'f1_score': f1_score,
        'balanced_accuracy': balanced_acc
    } 

class EarlyStopping:
    """Early stops the training if validation metric doesn't improve after a given patience."""
    def __init__(self, patience=7, mode='max', min_delta=0, verbose=False):
        """
        Args:
            patience (int): How long to wait after last improvement.
            mode (str): 'min' or 'max' depending on whether we want to minimize or maximize the metric.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            verbose (bool): If True, prints a message for each validation metric improvement.
        """
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        
        # Set comparison function based on mode
        if self.mode == 'min':
            self.comparison = lambda current, best: current < best - self.min_delta
        else:  # mode == 'max'
            self.comparison = lambda current, best: current > best + self.min_delta
    
    def __call__(self, epoch, metric_value):
        """
        Call instance with current epoch and metric value.
        
        Args:
            epoch (int): Current epoch number
            metric_value (float): Current metric value to evaluate
            
        Returns:
            bool: True if training should stop, False otherwise
        """
        if self.best_score is None:
            # First call
            self.best_score = metric_value
            self.best_epoch = epoch
            return False
        
        if self.comparison(metric_value, self.best_score):
            # Improvement
            self.best_score = metric_value
            self.best_epoch = epoch
            self.counter = 0
            if self.verbose:
                logging.info(f'EarlyStopping: Metric improved to {metric_value:.6f}')
            return False
        else:
            # No improvement
            self.counter += 1
            if self.verbose:
                logging.info(f'EarlyStopping: Counter {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                logging.info(f'EarlyStopping: Triggered at epoch {epoch}. '
                             f'Best score was {self.best_score:.6f} at epoch {self.best_epoch}')
                return True
            return False

def train_model_custom_progress(dataloaders, dataset_sizes, model, criterion, optimizer, scheduler, device, 
                            gpu_transform=None, train_sampler=None, val_sampler=None, rank=0, world_size=1, 
                            num_epochs=25, checkpoint_path=None, profiler=None, gradient_accumulation_steps=1, 
                            amp_enabled=True, gradient_clipping=0.0, progress_log_freq=5,
                            early_stopping_params=None): 
    """
    Optimized train_model function with custom progress tracking (no tqdm).
    
    Args:
        dataloaders: Dictionary with 'train' and 'val' dataloaders
        dataset_sizes: Dictionary with dataset sizes
        model: Model to train
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to train on
        gpu_transform: Transforms to apply on GPU
        train_sampler: Sampler for training data
        val_sampler: Sampler for validation data
        rank: Process rank
        world_size: Total number of processes
        num_epochs: Number of epochs to train for
        checkpoint_path: Path to save checkpoints
        profiler: Profiler object (not used)
        gradient_accumulation_steps: Number of steps to accumulate gradients
        amp_enabled: Whether to use automatic mixed precision
        gradient_clipping: Gradient clipping value
        progress_log_freq: Frequency of progress logging (in percentage)
        early_stopping_params: Dictionary with early stopping parameters
    """
    since = time.time() 
    start_epoch = 0
    best_acc = 0.0
    best_f1 = 0.0 
    best_balanced_acc = 0.0
    best_auc = 0.0
    
    # FIX: Use checkpoint_path directly as checkpoint_dir if it's a directory,
    # otherwise use the path as is (which might be a file)
    if checkpoint_path: 
        if os.path.isdir(checkpoint_path): 
            checkpoint_dir = checkpoint_path 
        else: 
            checkpoint_dir = os.path.dirname(checkpoint_path) 
    else: 
        checkpoint_dir = None
    
    first_batch_printed = False
    
    # Initialize mixed precision scaler if enabled
    scaler = None
    if amp_enabled: 
        scaler = GradScaler() 
    
    # Initialize memory monitor for debugging (only on rank 0)
    memory_monitor = None
    if rank == 0: 
        memory_monitor = CudaMemoryMonitor(enabled=True, log_interval=60) # Log every minute
    
    # Initialize early stopping if parameters are provided
    early_stopping = None
    if early_stopping_params and rank == 0:
        early_stopping = EarlyStopping(
            patience=early_stopping_params.get('patience', 7),
            mode=early_stopping_params.get('mode', 'max'),
            min_delta=early_stopping_params.get('min_delta', 0),
            verbose=early_stopping_params.get('verbose', True)
        )
        logging.info(f"Early stopping initialized with patience={early_stopping.patience}, "
                    f"mode={early_stopping.mode}, min_delta={early_stopping.min_delta}")
    
    # Checkpoint loading optimization
    if rank == 0: 
        # Load checkpoint if provided
        if checkpoint_path and os.path.exists(checkpoint_path) and os.path.isfile(checkpoint_path): 
            checkpoint = torch.load(checkpoint_path, map_location=device) 
            
            # For DDP, handle loading state_dict efficiently
            model_state_dict = checkpoint['model_state_dict'] 
            
            # If model was saved with DDP wrapper, it will have 'module.' prefix in keys
            if all(k.startswith('module.') for k in model_state_dict.keys()): 
                model.load_state_dict(model_state_dict) 
            else: 
                # If the model wasn't saved with DDP wrapper
                model.module.load_state_dict(model_state_dict) 
            
            optimizer.load_state_dict(checkpoint['optimizer_state_dict']) 
            start_epoch = checkpoint['epoch'] + 1
            best_acc = checkpoint.get('best_acc', 0.0) 
            best_f1 = checkpoint.get('best_f1', 0.0)
            best_balanced_acc = checkpoint.get('best_balanced_acc', 0.0)
            best_auc = checkpoint.get('best_auc', 0.0)
            logging.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}, best_acc: {best_acc:.4f}, "
                        f"best_f1: {best_f1:.4f}, best_balanced_acc: {best_balanced_acc:.4f}, "
                        f"best_auc: {best_auc:.4f}")
        else: 
            if checkpoint_dir is not None and not os.path.exists(checkpoint_dir): 
                os.makedirs(checkpoint_dir) 
    
    # Minimal synchronization - broadcast start_epoch and best metrics once at beginning
    if world_size > 1: 
        start_epoch_tensor = torch.tensor([start_epoch], device=device) 
        best_acc_tensor = torch.tensor([best_acc], device=device) 
        best_f1_tensor = torch.tensor([best_f1], device=device)
        best_balanced_acc_tensor = torch.tensor([best_balanced_acc], device=device)
        best_auc_tensor = torch.tensor([best_auc], device=device)
        
        # Only broadcast once before training loop
        dist.broadcast(start_epoch_tensor, src=0) 
        dist.broadcast(best_acc_tensor, src=0) 
        dist.broadcast(best_f1_tensor, src=0)
        dist.broadcast(best_balanced_acc_tensor, src=0)
        dist.broadcast(best_auc_tensor, src=0)
        
        start_epoch = start_epoch_tensor.item() 
        best_acc = best_acc_tensor.item() 
        best_f1 = best_f1_tensor.item()
        best_balanced_acc = best_balanced_acc_tensor.item()
        best_auc = best_auc_tensor.item()
    
    # Only rank 0 prints training info
    if rank == 0: 
        logging.info("========== Starting Training ==========") 
        if torch.cuda.is_available(): 
            for gpu_id in range(min(world_size, torch.cuda.device_count())): 
                logging.info(f"GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}") 
    
    # Training loop
    for epoch in range(start_epoch, num_epochs): 
        if rank == 0: 
            logging.info(f'Epoch {epoch}/{num_epochs - 1}') 
            logging.info('-' * 10) 
        
        epoch_start_time = time.time() 
        
        # Set epoch for samplers (needed for distributed training)
        if train_sampler is not None: 
            train_sampler.set_epoch(epoch) 
        if val_sampler is not None: 
            val_sampler.set_epoch(epoch) 
        
        # Each epoch has a training and validation phase
        phase_results = {} 
        
        for phase in ['train', 'val']: 
            phase_start_time = time.time() 
            
            # Reset metrics for this phase
            running_loss = 0.0
            running_corrects = 0
            
            # Initialize confusion matrix components
            running_true_positive = 0
            running_false_positive = 0
            running_false_negative = 0
            running_true_negative = 0
            
            # For AUC-ROC calculation
            all_labels = []
            all_probs = []
            
            if phase == 'train': 
                model.train() 
            else: 
                model.eval() 
            
            # Performance metrics
            data_time = 0.0
            forward_time = 0.0
            backward_time = 0.0
            
            # Step counter for gradient accumulation
            steps = 0
            processed_samples = 0
            
            # Log initial memory state
            if memory_monitor: 
                memory_monitor.log_memory_stats(phase=phase, step=0, force=True) 
            
            # Calculate progress logging points
            total_batches = len(dataloaders[phase]) 
            # Log at either fixed percentage intervals or every N batches, whichever is more frequent
            log_every_n_batches = max(1, min(10, total_batches // 20)) 
            
            # Print total number of batches information
            if rank == 0: 
                logging.info(f"{phase} phase: total {total_batches} batches") 
            
            # Batch processing
            batch_start = time.time() 
            for batch_idx, (inputs, labels) in enumerate(dataloaders[phase]): 
                # Log progress at regular intervals
                if rank == 0 and (batch_idx % log_every_n_batches == 0 or batch_idx == total_batches - 1): 
                    progress = 100.0 * batch_idx / total_batches 
                    logging.info(f"{phase} Epoch {epoch}/{num_epochs-1}: {progress:.1f}% ({batch_idx}/{total_batches})") 
                
                # Measure data loading time
                current_data_time = time.time() - batch_start 
                data_time += current_data_time 
                
                # Non-blocking transfer to GPU
                inputs = inputs.to(device, non_blocking=True) 
                labels = labels.to(device, non_blocking=True) 
                
                # Apply GPU transform if provided (more efficient on GPU)
                if gpu_transform is not None: 
                    inputs = gpu_transform(inputs) 
                
                # Print input shape once (only on rank 0)
                if not first_batch_printed and rank == 0: 
                    logging.info(f"Input batch size: {inputs.shape}") 
                    first_batch_printed = True
                
                # Only zero gradients at accumulation boundaries or in eval mode
                if phase == 'train' and (steps % gradient_accumulation_steps == 0): 
                    optimizer.zero_grad(set_to_none=True) # More efficient
                
                # Forward pass timing with mixed precision
                forward_start = time.time() 
                
                # Mixed precision training
                if amp_enabled and phase == 'train': 
                    # Using AMP for mixed precision training
                    with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'): 
                        outputs = model(inputs) 
                        _, preds = torch.max(outputs, 1) 
                        loss = criterion(outputs, labels) 
                        
                        # Scale loss by gradient accumulation steps
                        loss = loss / gradient_accumulation_steps 
                    
                    # Backward pass timing
                    backward_start = time.time() 
                    forward_time += backward_start - forward_start 
                    
                    # Scale and accumulate gradients
                    scaler.scale(loss).backward() 
                    
                    # Only step optimizer at accumulation boundaries
                    if (steps + 1) % gradient_accumulation_steps == 0 or (batch_idx == len(dataloaders[phase]) - 1): 
                        # Apply gradient clipping if configured
                        if gradient_clipping > 0: 
                            # Unscale gradients before clipping
                            scaler.unscale_(optimizer) 
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clipping) 
                        
                        # Update weights
                        scaler.step(optimizer) 
                        scaler.update() 
                    
                    backward_time += time.time() - backward_start 
                else: 
                    # Standard precision training/evaluation
                    with torch.set_grad_enabled(phase == 'train'): 
                        outputs = model(inputs) 
                        _, preds = torch.max(outputs, 1) 
                        loss = criterion(outputs, labels) 
                        
                        if phase == 'train': 
                            # Scale loss by gradient accumulation steps
                            loss = loss / gradient_accumulation_steps 
                        
                        # Backward pass timing
                        backward_start = time.time() 
                        forward_time += backward_start - forward_start 
                        
                        if phase == 'train':
                            # Standard backward pass
                            loss.backward() 
                            
                            # Only step optimizer at accumulation boundaries
                            if (steps + 1) % gradient_accumulation_steps == 0 or (batch_idx == len(dataloaders[phase]) - 1): 
                                # Apply gradient clipping if configured
                                if gradient_clipping > 0: 
                                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clipping) 
                                
                                # Update weights
                                optimizer.step() 
                            
                            backward_time += time.time() - backward_start 
                        else: 
                            forward_time += time.time() - forward_start 
                
                # Collect data for AUC-ROC calculation
                softmax_outputs = torch.nn.functional.softmax(outputs, dim=1)
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(softmax_outputs[:, 1].detach().cpu().numpy())  # Class 1 probabilities
                
                # Statistics - track full batch loss (not accumulated loss)
                full_loss = loss.item() * gradient_accumulation_steps 
                running_loss += full_loss * inputs.size(0) 
                running_corrects += torch.sum(preds == labels.data) 
                
                # Update confusion matrix values for each batch
                # For binary classification:
                # TP: predicted positive (1) and actually positive (1)
                # FP: predicted positive (1) but actually negative (0)
                # FN: predicted negative (0) but actually positive (1)
                # TN: predicted negative (0) and actually negative (0)
                
                # Compute batch confusion matrix elements
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
                
                # Count processed samples
                processed_samples += inputs.size(0) 
                steps += 1
                
                # Log memory stats periodically
                if memory_monitor and batch_idx % 30 == 0: 
                    memory_monitor.log_memory_stats(phase=phase, step=batch_idx) 
                
                # Prepare for next batch
                batch_start = time.time() 
            
            # Update learning rate scheduler once per epoch
            if phase == 'train': 
                scheduler.step() 
            
            # Calculate metrics for this process
            epoch_loss = running_loss / processed_samples if processed_samples > 0 else 0
            epoch_acc = running_corrects.double() / processed_samples if processed_samples > 0 else 0
            
            # Aggregate confusion matrix values for distributed training
            confusion_values = torch.tensor(
                [running_true_positive, running_false_positive, running_false_negative, running_true_negative, processed_samples], 
                dtype=torch.float64, 
                device=device 
            ) 
            
            # Efficient metrics aggregation across all processes
            if world_size > 1: 
                # Single all-reduce operation for confusion matrix values
                dist.all_reduce(confusion_values, op=dist.ReduceOp.SUM) 
                
                # Create tensor for loss and accuracy
                metrics = torch.tensor([epoch_loss, epoch_acc, processed_samples], device=device) 
                dist.all_reduce(metrics, op=dist.ReduceOp.SUM) 
                
                # Correctly compute weighted averages
                total_samples = metrics[2].item() 
                if total_samples > 0: 
                    epoch_loss = metrics[0].item() / world_size 
                    epoch_acc = metrics[1].item() / world_size 
            
            # Extract aggregated confusion matrix values
            tp = confusion_values[0].item() 
            fp = confusion_values[1].item() 
            fn = confusion_values[2].item() 
            tn = confusion_values[3].item() 
            
            # Compute additional metrics from confusion matrix
            additional_metrics = compute_metrics_from_confusion_matrix(tp, fp, fn, tn) 
            
            # Calculate AUC-ROC if we have sufficient samples and both classes
            epoch_auc = 0.0
            if len(set(all_labels)) > 1:  # Make sure we have both classes
                # Collect all labels and predictions across processes
                if world_size > 1:
                    # Convert to tensors for all_gather
                    all_labels_tensor = torch.tensor(all_labels, device=device)
                    all_probs_tensor = torch.tensor(all_probs, device=device)
                    
                    # Get sizes for padding
                    local_size = torch.tensor([len(all_labels)], device=device)
                    sizes = [torch.zeros(1, device=device) for _ in range(world_size)]
                    dist.all_gather(sizes, local_size)
                    
                    # Pad tensors to maximum size
                    max_size = int(max(sizes).item())
                    padded_labels = torch.zeros(max_size, device=device)
                    padded_probs = torch.zeros(max_size, device=device)
                    
                    # Copy data to padded tensors
                    padded_labels[:len(all_labels)] = all_labels_tensor
                    padded_probs[:len(all_probs)] = all_probs_tensor
                    
                    # Gather padded tensors
                    gathered_labels = [torch.zeros(max_size, device=device) for _ in range(world_size)]
                    gathered_probs = [torch.zeros(max_size, device=device) for _ in range(world_size)]
                    
                    dist.all_gather(gathered_labels, padded_labels)
                    dist.all_gather(gathered_probs, padded_probs)
                    
                    # Combine and remove padding
                    combined_labels = []
                    combined_probs = []
                    for i in range(world_size):
                        size = int(sizes[i].item())
                        combined_labels.extend(gathered_labels[i][:size].cpu().numpy())
                        combined_probs.extend(gathered_probs[i][:size].cpu().numpy())
                    
                    # Calculate AUC-ROC
                    try:
                        epoch_auc = roc_auc_score(combined_labels, combined_probs)
                    except ValueError as e:
                        if rank == 0:
                            logging.warning(f"Error calculating AUC-ROC: {e}")
                        epoch_auc = 0.0
                else:
                    # Single process calculation
                    try:
                        epoch_auc = roc_auc_score(all_labels, all_probs)
                    except ValueError as e:
                        if rank == 0:
                            logging.warning(f"Error calculating AUC-ROC: {e}")
                        epoch_auc = 0.0
            
            # Use the more accurate computed accuracy instead of epoch_acc
            epoch_acc = additional_metrics['accuracy'] 
            epoch_sensitivity = additional_metrics['sensitivity'] 
            epoch_specificity = additional_metrics['specificity'] 
            epoch_f1 = additional_metrics['f1_score']
            epoch_balanced_acc = additional_metrics['balanced_accuracy']
            
            phase_time = time.time() - phase_start_time 
            
            # Store the results for this phase
            phase_results[phase] = { 
                'loss': epoch_loss, 
                'acc': epoch_acc, 
                'sensitivity': epoch_sensitivity, 
                'specificity': epoch_specificity, 
                'f1_score': epoch_f1,
                'balanced_accuracy': epoch_balanced_acc,
                'auc': epoch_auc,
                'time': phase_time, 
                # Store confusion matrix values for reference
                'confusion_matrix': { 
                    'TP': tp, 
                    'FP': fp, 
                    'FN': fn, 
                    'TN': tn 
                } 
            } 
            
            # Log results (rank 0 only)
            if rank == 0: 
                logging.info(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} F1: {epoch_f1:.4f}') 
                logging.info(f'{phase} Balanced Acc: {epoch_balanced_acc:.4f} AUC-ROC: {epoch_auc:.4f}')
                logging.info(f'{phase} Sensitivity: {epoch_sensitivity:.4f} Specificity: {epoch_specificity:.4f}') 
                logging.info(f'{phase} Confusion Matrix: TP={tp}, FP={fp}, FN={fn}, TN={tn}') 
                logging.info(f'{phase} completed in {phase_time:.2f}s') 
            
            # Log timing information
            if phase == 'train' and rank == 0: 
                logging.info(f'Data loading time: {data_time:.2f}s, Forward time: {forward_time:.2f}s, Backward time: {backward_time:.2f}s') 
            
            # Get and log cache stats if available
            try: 
                if hasattr(dataloaders[phase].dataset, 'get_cache_stats') and rank == 0: 
                    stats = dataloaders[phase].dataset.get_cache_stats() 
                    logging.info(f'{phase} cache: {stats["cache_size"]}/{stats["cache_size"] + 1} items, hit rate: {stats["hit_rate"]:.2f}%') 
            except Exception as e: 
                logging.debug(f"Error getting cache stats: {str(e)}") 
        
        # Check early stopping based on validation metrics
        if early_stopping and phase_results.get('val') and rank == 0:
            # Get the monitoring metric based on early_stopping_params
            metric_name = early_stopping_params.get('metric', 'balanced_accuracy')
            metric_value = phase_results['val'].get(metric_name, 0.0)
            
            # Call early stopping to check if we should stop
            if early_stopping(epoch, metric_value):
                logging.info(f"Early stopping triggered at epoch {epoch}")
                break  # Exit the epoch loop
        
        # Update best model based on all metrics
        if rank == 0 and phase_results.get('val') and checkpoint_dir is not None: 
            val_metrics = phase_results['val']
            
            # Update best model based on regular accuracy if it's the best so far
            if val_metrics['acc'] > best_acc: 
                best_acc = val_metrics['acc'] 
                best_acc_model_path = os.path.join(checkpoint_dir, f"best_acc_model.pth") 
                try: 
                    torch.save({ 
                        'epoch': epoch, 
                        'model_state_dict': model.state_dict(), 
                        'optimizer_state_dict': optimizer.state_dict(), 
                        'best_acc': best_acc, 
                        'best_f1': best_f1,
                        'best_balanced_acc': best_balanced_acc,
                        'best_auc': best_auc,
                        'metrics': { 
                            'accuracy': val_metrics['acc'], 
                            'sensitivity': val_metrics['sensitivity'], 
                            'specificity': val_metrics['specificity'], 
                            'f1_score': val_metrics['f1_score'],
                            'balanced_accuracy': val_metrics['balanced_accuracy'],
                            'auc': val_metrics['auc']
                        } 
                    }, best_acc_model_path) 
                    logging.info(f"Saved new best accuracy model with accuracy: {best_acc:.4f} at {best_acc_model_path}") 
                except Exception as e: 
                    logging.error(f"Error saving best accuracy model: {str(e)}") 
            
            # Update best model based on F1 score if it's the best so far
            if val_metrics['f1_score'] > best_f1: 
                best_f1 = val_metrics['f1_score'] 
                best_f1_model_path = os.path.join(checkpoint_dir, f"best_f1_model.pth") 
                try: 
                    model_save = { 
                        'epoch': epoch, 
                        'model_state_dict': model.state_dict(), 
                        'optimizer_state_dict': optimizer.state_dict(), 
                        'best_acc': best_acc, 
                        'best_f1': best_f1,
                        'best_balanced_acc': best_balanced_acc,
                        'best_auc': best_auc,
                        'metrics': { 
                            'accuracy': val_metrics['acc'], 
                            'sensitivity': val_metrics['sensitivity'], 
                            'specificity': val_metrics['specificity'], 
                            'f1_score': val_metrics['f1_score'],
                            'balanced_accuracy': val_metrics['balanced_accuracy'],
                            'auc': val_metrics['auc']
                        } 
                    } 
                    torch.save(model_save, best_f1_model_path) 
                    logging.info(f"Saved new best F1 model with F1 score: {best_f1:.4f} at {best_f1_model_path}") 
                except Exception as e: 
                    logging.error(f"Error saving best F1 model: {str(e)}") 
            
            # Update best model based on Balanced Accuracy if it's the best so far
            if val_metrics['balanced_accuracy'] > best_balanced_acc: 
                best_balanced_acc = val_metrics['balanced_accuracy'] 
                best_balanced_acc_model_path = os.path.join(checkpoint_dir, f"best_balanced_acc_model.pth") 
                try: 
                    model_save = { 
                        'epoch': epoch, 
                        'model_state_dict': model.state_dict(), 
                        'optimizer_state_dict': optimizer.state_dict(), 
                        'best_acc': best_acc, 
                        'best_f1': best_f1,
                        'best_balanced_acc': best_balanced_acc,
                        'best_auc': best_auc,
                        'metrics': { 
                            'accuracy': val_metrics['acc'], 
                            'sensitivity': val_metrics['sensitivity'], 
                            'specificity': val_metrics['specificity'], 
                            'f1_score': val_metrics['f1_score'],
                            'balanced_accuracy': val_metrics['balanced_accuracy'],
                            'auc': val_metrics['auc']
                        } 
                    } 
                    # Save as best balanced accuracy model
                    torch.save(model_save, best_balanced_acc_model_path) 
                    
                    # Also save as general best model for consistent reference
                    best_model_path = os.path.join(checkpoint_dir, f"best_model.pth")
                    torch.save(model_save, best_model_path)
                    
                    logging.info(f"Saved new best balanced accuracy model with balanced acc: {best_balanced_acc:.4f} at {best_balanced_acc_model_path}") 
                    logging.info(f"Also saved as general best model at {best_model_path}")
                except Exception as e: 
                    logging.error(f"Error saving best balanced accuracy model: {str(e)}") 
            
            # Update best model based on AUC-ROC if it's the best so far
            if val_metrics['auc'] > best_auc: 
                best_auc = val_metrics['auc'] 
                best_auc_model_path = os.path.join(checkpoint_dir, f"best_auc_model.pth") 
                try: 
                    model_save = { 
                        'epoch': epoch, 
                        'model_state_dict': model.state_dict(), 
                        'optimizer_state_dict': optimizer.state_dict(), 
                        'best_acc': best_acc, 
                        'best_f1': best_f1,
                        'best_balanced_acc': best_balanced_acc,
                        'best_auc': best_auc,
                        'metrics': { 
                            'accuracy': val_metrics['acc'], 
                            'sensitivity': val_metrics['sensitivity'], 
                            'specificity': val_metrics['specificity'], 
                            'f1_score': val_metrics['f1_score'],
                            'balanced_accuracy': val_metrics['balanced_accuracy'],
                            'auc': val_metrics['auc']
                        } 
                    } 
                    torch.save(model_save, best_auc_model_path) 
                    logging.info(f"Saved new best AUC-ROC model with AUC: {best_auc:.4f} at {best_auc_model_path}") 
                except Exception as e: 
                    logging.error(f"Error saving best AUC model: {str(e)}") 
        
        # Log epoch summary (rank 0 only)
        if rank == 0: 
            epoch_time = time.time() - epoch_start_time 
            logging.info(f"Epoch {epoch} summary:") 
            logging.info(f" Train Loss: {phase_results['train']['loss']:.4f}, Train Acc: {phase_results['train']['acc']:.4f}, Train F1: {phase_results['train']['f1_score']:.4f}") 
            logging.info(f" Train Balanced Acc: {phase_results['train']['balanced_accuracy']:.4f}, Train AUC: {phase_results['train']['auc']:.4f}")
            logging.info(f" Train Sensitivity: {phase_results['train']['sensitivity']:.4f}, Train Specificity: {phase_results['train']['specificity']:.4f}") 
            logging.info(f" Val Loss: {phase_results['val']['loss']:.4f}, Val Acc: {phase_results['val']['acc']:.4f}, Val F1: {phase_results['val']['f1_score']:.4f}") 
            logging.info(f" Val Balanced Acc: {phase_results['val']['balanced_accuracy']:.4f}, Val AUC: {phase_results['val']['auc']:.4f}")
            logging.info(f" Val Sensitivity: {phase_results['val']['sensitivity']:.4f}, Val Specificity: {phase_results['val']['specificity']:.4f}") 
            logging.info(f" Total Epoch Time: {epoch_time:.2f}s") 
            
            # Save epoch checkpoint 
            if checkpoint_dir is not None: 
                epoch_path = os.path.join(checkpoint_dir, f"epoch_{epoch}.pth") 
                try: 
                    torch.save({ 
                        'epoch': epoch, 
                        'model_state_dict': model.state_dict(), 
                        'optimizer_state_dict': optimizer.state_dict(), 
                        'best_acc': best_acc, 
                        'best_f1': best_f1,
                        'best_balanced_acc': best_balanced_acc,
                        'best_auc': best_auc,
                        'train_loss': phase_results['train']['loss'], 
                        'train_acc': phase_results['train']['acc'], 
                        'train_sensitivity': phase_results['train']['sensitivity'], 
                        'train_specificity': phase_results['train']['specificity'], 
                        'train_f1': phase_results['train']['f1_score'],
                        'train_balanced_acc': phase_results['train']['balanced_accuracy'],
                        'train_auc': phase_results['train']['auc'],
                        'val_loss': phase_results['val']['loss'], 
                        'val_acc': phase_results['val']['acc'], 
                        'val_sensitivity': phase_results['val']['sensitivity'], 
                        'val_specificity': phase_results['val']['specificity'], 
                        'val_f1': phase_results['val']['f1_score'],
                        'val_balanced_acc': phase_results['val']['balanced_accuracy'],
                        'val_auc': phase_results['val']['auc'],
                    }, epoch_path) 
                    logging.info(f"Saved epoch checkpoint at {epoch_path}") 
                except Exception as e: 
                    logging.error(f"Error saving epoch checkpoint: {str(e)}") 
        
        # Minimal synchronization - just one barrier per epoch
        if world_size > 1: 
            dist.barrier() 
    
    # Final summary (rank 0 only)
    if rank == 0: 
        time_elapsed = time.time() - since 
        minutes, seconds = divmod(time_elapsed, 60) 
        hours, minutes = divmod(minutes, 60) 
        logging.info("========== Training Completed ==========") 
        logging.info(f'Training complete in {int(hours)}h {int(minutes)}m {seconds:.2f}s') 
        logging.info(f'Best val Acc: {best_acc:.6f}, Best val F1: {best_f1:.6f}')
        logging.info(f'Best val Balanced Acc: {best_balanced_acc:.6f}, Best val AUC: {best_auc:.6f}')
        
        # Load best model weights based on balanced accuracy (new default)
        if checkpoint_dir is not None: 
            best_balanced_acc_model_path = os.path.join(checkpoint_dir, f"best_balanced_acc_model.pth") 
            if os.path.exists(best_balanced_acc_model_path): 
                best_checkpoint = torch.load(best_balanced_acc_model_path) 
                model.load_state_dict(best_checkpoint['model_state_dict']) 
                logging.info(f"Loaded best balanced accuracy model with balanced acc: {best_checkpoint.get('best_balanced_acc', 0.0):.6f}") 
            else: 
                # Fall back to best model
                best_model_path = os.path.join(checkpoint_dir, f"best_model.pth") 
                if os.path.exists(best_model_path): 
                    best_checkpoint = torch.load(best_model_path) 
                    model.load_state_dict(best_checkpoint['model_state_dict']) 
                    logging.info("Loaded best model weights") 
    
    # Minimal synchronization - broadcast best model once at the end
    if world_size > 1: 
        # Wait for rank 0 to load best model
        dist.barrier() 
        
        # Broadcast model parameters
        for param in model.parameters(): 
            dist.broadcast(param.data, src=0) 
    
    return model
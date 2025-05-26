import logging
import os
import time
import torch
import random
import copy
import threading
from collections import defaultdict
import torch.distributed as dist
from torch.amp import autocast
import numpy as np

# Optimized caching system for distributed training
class ProcessLocalCache:
    """Singleton cache manager for per-process storage in distributed environments"""
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
        """Clear cache for a specific rank or all caches"""
        if rank is not None:
            self.rank_caches[rank].clear()
            self.stats[rank] = {"hits": 0, "requests": 0}
        else:
            self.rank_caches.clear()
            self.stats.clear()
    
    def size(self, rank=None):
        """Get cache size for a specific rank or total size"""
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

# Global cache instance
PROCESS_CACHE = ProcessLocalCache()

class CachedDataset(torch.utils.data.Dataset):
    """Enhanced dataset wrapper that caches data in memory with process-local storage"""
    def __init__(self, dataset, cache_size=1000, cache_probability=1.0, rank=0, world_size=1):
        self.dataset = dataset
        self.cache_size = cache_size
        self.cache_probability = cache_probability
        self.rank = rank
        self.world_size = world_size
        
        # Initialize cache stats for this rank
        PROCESS_CACHE.reset_stats(self.rank)
        
        if self.rank == 0:
            logging.info(f"CachedDataset initialized for rank {rank} with cache_size={cache_size}")
    
    def __getitem__(self, index):
        # Generate a unique cache key
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
                # Store in process-local cache with appropriate cloning
                if isinstance(item, tuple):
                    cached_item = [x.clone().detach() if torch.is_tensor(x) else copy.deepcopy(x) for x in item]
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
        return len(self.dataset)
    
    def get_cache_stats(self):
        return PROCESS_CACHE.get_stats(self.rank)
    
    def reset_cache_stats(self):
        PROCESS_CACHE.reset_stats(self.rank)
    
    @staticmethod
    def clear_cache(rank=None):
        PROCESS_CACHE.clear(rank)
        logging.info(f"Cleared cache for {'all ranks' if rank is None else f'rank {rank}'}")

class CudaMemoryMonitor:
    """Utility for monitoring CUDA memory usage during training"""
    def __init__(self, enabled=True, log_interval=5):
        self.enabled = enabled and torch.cuda.is_available()
        self.log_interval = log_interval
        self.last_log_time = time.time()
        
        if self.enabled:
            self.baseline = self._get_memory_stats()
            logging.info(f"CUDA memory monitor initialized: {self.baseline}")
    
    def _get_memory_stats(self):
        if not self.enabled:
            return {}
        
        try:
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
        if not self.enabled:
            return
        
        current_time = time.time()
        if not force and (current_time - self.last_log_time) < self.log_interval:
            return
        
        self.last_log_time = current_time
        current_stats = self._get_memory_stats()
        
        log_message = f"CUDA Memory [{phase}][Step {step}]:"
        for device_idx, stats in current_stats.items():
            log_message += f" Device {device_idx}: Allocated: {stats['allocated']:.1f} MB, Reserved: {stats['reserved']:.1f} MB"
            
            # Calculate difference from baseline
            if device_idx in self.baseline:
                allocated_diff = stats['allocated'] - self.baseline[device_idx]['allocated']
                log_message += f" (Δ: {allocated_diff:+.1f} MB)"
        
        logging.info(log_message)
    
    def reset_baseline(self):
        if self.enabled:
            self.baseline = self._get_memory_stats()
            logging.info(f"Reset CUDA memory baseline: {self.baseline}")

def compute_metrics_from_confusion_matrix(true_positive, false_positive, false_negative, true_negative):
    """Compute common classification metrics from confusion matrix elements"""
    epsilon = 1e-7  # Avoid division by zero
    
    # Total samples
    total = true_positive + false_positive + false_negative + true_negative
    
    # Calculate metrics
    accuracy = (true_positive + true_negative) / max(total, epsilon)
    precision = true_positive / max(true_positive + false_positive, epsilon)
    sensitivity = true_positive / max(true_positive + false_negative, epsilon)
    specificity = true_negative / max(true_negative + false_positive, epsilon)
    f1_score = 2 * precision * sensitivity / max(precision + sensitivity, epsilon)
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
    """Early stops the training if validation metric doesn't improve after a given patience"""
    def __init__(self, patience=7, mode='max', min_delta=0, verbose=False):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        
        # Set comparison function based on mode
        self.comparison = (lambda current, best: current < best - self.min_delta) if mode == 'min' else \
                          (lambda current, best: current > best + self.min_delta)
    
    def __call__(self, epoch, metric_value):
        # First call
        if self.best_score is None:
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

def train_model_custom_progress(dataloaders, dataset_sizes, model, criterion, optimizer, scheduler, 
                              device, gpu_transform=None, train_sampler=None, val_sampler=None, 
                              rank=0, world_size=1, num_epochs=25, checkpoint_path=None, 
                              profiler=None, gradient_accumulation_steps=1, amp_enabled=True, 
                              gradient_clipping=0.0, progress_log_freq=5, early_stopping_params=None):
    """Optimized train_model function with custom progress tracking"""
    since = time.time()
    start_epoch = 0
    best_metrics = {'acc': 0.0, 'f1': 0.0, 'balanced_acc': 0.0, 'auc': 0.0}
    
    # Set up checkpoint directory
    if checkpoint_path:
        checkpoint_dir = checkpoint_path if os.path.isdir(checkpoint_path) else os.path.dirname(checkpoint_path)
    else:
        checkpoint_dir = None
    
    first_batch_printed = False
    
    # Initialize mixed precision scaler if enabled
    scaler = torch.cuda.amp.GradScaler() if amp_enabled else None
    
    # Initialize memory monitor for debugging (only on rank 0)
    memory_monitor = CudaMemoryMonitor(enabled=True, log_interval=60) if rank == 0 else None
    
    # Initialize early stopping if parameters are provided
    early_stopping = None
    if early_stopping_params and rank == 0:
        early_stopping = EarlyStopping(
            patience=early_stopping_params.get('patience', 7),
            mode=early_stopping_params.get('mode', 'max'),
            min_delta=early_stopping_params.get('min_delta', 0),
            verbose=early_stopping_params.get('verbose', True)
        )
        logging.info(f"Early stopping initialized: patience={early_stopping.patience}, "
                    f"mode={early_stopping.mode}, min_delta={early_stopping.min_delta}")
    
    # Load checkpoint if provided
    if rank == 0 and checkpoint_path and os.path.exists(checkpoint_path) and os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Handle loading state_dict with or without DDP wrapper
        model_state_dict = checkpoint['model_state_dict']
        if all(k.startswith('module.') for k in model_state_dict.keys()):
            model.load_state_dict(model_state_dict)
        else:
            model.module.load_state_dict(model_state_dict)
        
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        for metric in best_metrics:
            best_metrics[metric] = checkpoint.get(f'best_{metric}', 0.0)
        
        logging.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}, "
                    f"best metrics: {', '.join([f'{k}: {v:.4f}' for k, v in best_metrics.items()])}")
    elif checkpoint_dir is not None and not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    # Broadcast initial values to all processes
    if world_size > 1:
        tensors = {
            'start_epoch': torch.tensor([start_epoch], device=device),
            **{f'best_{k}': torch.tensor([v], device=device) for k, v in best_metrics.items()}
        }
        
        # Broadcast all tensors
        for name, tensor in tensors.items():
            dist.broadcast(tensor, src=0)
        
        # Update local values
        start_epoch = tensors['start_epoch'].item()
        for metric in best_metrics:
            best_metrics[metric] = tensors[f'best_{metric}'].item()
    
    # Print training info (rank 0 only)
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
        
        # Track results for this epoch
        phase_results = {}
        
        # Training and validation phases
        for phase in ['train', 'val']:
            phase_start_time = time.time()
            
            # Reset metrics
            running = {
                'loss': 0.0,
                'corrects': 0,
                'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0,  # Confusion matrix components
                'samples': 0
            }
            
            # For AUC-ROC calculation
            all_labels, all_probs = [], []
            
            # Set model mode
            model.train(phase == 'train')
            
            # Performance metrics
            timing = {'data': 0.0, 'forward': 0.0, 'backward': 0.0}
            steps = 0  # Step counter for gradient accumulation
            
            # Log initial memory state
            if memory_monitor:
                memory_monitor.log_memory_stats(phase=phase, step=0, force=True)
            
            # Calculate progress logging points
            total_batches = len(dataloaders[phase])
            log_every_n_batches = max(1, min(10, total_batches // 20))
            
            if rank == 0:
                logging.info(f"{phase} phase: total {total_batches} batches")
            
            # Process batches
            batch_start = time.time()
            for batch_idx, (inputs, labels) in enumerate(dataloaders[phase]):
                # Log progress periodically
                if rank == 0 and (batch_idx % log_every_n_batches == 0 or batch_idx == total_batches - 1):
                    progress = 100.0 * batch_idx / total_batches
                    logging.info(f"{phase} Epoch {epoch}/{num_epochs-1}: {progress:.1f}% ({batch_idx}/{total_batches})")
                
                # Measure data loading time
                timing['data'] += time.time() - batch_start
                
                # Move data to device
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                # Apply GPU transform if provided
                if gpu_transform is not None:
                    inputs = gpu_transform(inputs)
                
                # Print input shape once
                if not first_batch_printed and rank == 0:
                    logging.info(f"Input batch size: {inputs.shape}")
                    first_batch_printed = True
                
                # Zero gradients at accumulation boundaries
                if phase == 'train' and (steps % gradient_accumulation_steps == 0):
                    optimizer.zero_grad(set_to_none=True)
                
                # Forward pass timing
                forward_start = time.time()
                
                # Forward pass (with mixed precision in training)
                with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu') if amp_enabled and phase == 'train' else \
                     torch.set_grad_enabled(phase == 'train'):
                    # Forward pass
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # Scale loss by gradient accumulation steps in training
                    if phase == 'train':
                        loss = loss / gradient_accumulation_steps
                    
                    # Backward pass timing
                    timing['forward'] += time.time() - forward_start
                    backward_start = time.time()
                    
                    if phase == 'train':
                        # Backward pass (with or without mixed precision)
                        if amp_enabled:
                            # Scale and accumulate gradients
                            scaler.scale(loss).backward()
                            
                            # Only step optimizer at accumulation boundaries
                            if (steps + 1) % gradient_accumulation_steps == 0 or batch_idx == len(dataloaders[phase]) - 1:
                                # Apply gradient clipping if configured
                                if gradient_clipping > 0:
                                    scaler.unscale_(optimizer)
                                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clipping)
                                
                                # Update weights
                                scaler.step(optimizer)
                                scaler.update()
                        else:
                            # Standard backward pass
                            loss.backward()
                            
                            # Only step optimizer at accumulation boundaries
                            if (steps + 1) % gradient_accumulation_steps == 0 or batch_idx == len(dataloaders[phase]) - 1:
                                # Apply gradient clipping if configured
                                if gradient_clipping > 0:
                                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clipping)
                                
                                # Update weights
                                optimizer.step()
                        
                        timing['backward'] += time.time() - backward_start
                    else:
                        timing['forward'] += time.time() - forward_start
                
                # Collect softmax outputs for AUC calculation
                softmax_outputs = torch.nn.functional.softmax(outputs, dim=1)
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(softmax_outputs[:, 1].detach().cpu().numpy())  # Class 1 probabilities
                
                # Update running statistics
                batch_size = inputs.size(0)
                full_loss = loss.item() * gradient_accumulation_steps if phase == 'train' else loss.item()
                running['loss'] += full_loss * batch_size
                running['corrects'] += torch.sum(preds == labels.data).item()
                
                # Update confusion matrix
                predicted_positive = (preds == 1)
                actual_positive = (labels == 1)
                
                running['tp'] += torch.sum((predicted_positive) & (actual_positive)).item()
                running['fp'] += torch.sum((predicted_positive) & (~actual_positive)).item()
                running['fn'] += torch.sum((~predicted_positive) & (actual_positive)).item()
                running['tn'] += torch.sum((~predicted_positive) & (~actual_positive)).item()
                
                running['samples'] += batch_size
                steps += 1
                
                # Log memory stats periodically
                if memory_monitor and batch_idx % 30 == 0:
                    memory_monitor.log_memory_stats(phase=phase, step=batch_idx)
                
                # Prepare for next batch
                batch_start = time.time()
            
            # Update learning rate scheduler at end of training phase
            if phase == 'train':
                scheduler.step()
            
            # Calculate epoch metrics for this process
            processed_samples = running['samples']
            epoch_metrics = {
                'loss': running['loss'] / processed_samples if processed_samples > 0 else 0,
                'acc': running['corrects'] / processed_samples if processed_samples > 0 else 0
            }
            
            # Aggregate metrics across all processes
            if world_size > 1:
                # Gather confusion matrix values
                confusion_values = torch.tensor(
                    [running['tp'], running['fp'], running['fn'], running['tn'], processed_samples],
                    dtype=torch.float64, device=device
                )
                dist.all_reduce(confusion_values, op=dist.ReduceOp.SUM)
                
                # Gather loss and accuracy
                metrics_tensor = torch.tensor([epoch_metrics['loss'], epoch_metrics['acc'], processed_samples], device=device)
                dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
                
                # Compute weighted averages
                total_samples = metrics_tensor[2].item()
                if total_samples > 0:
                    epoch_metrics['loss'] = metrics_tensor[0].item() / world_size
                    epoch_metrics['acc'] = metrics_tensor[1].item() / world_size
                
                # Extract aggregated confusion matrix values
                tp, fp, fn, tn = [confusion_values[i].item() for i in range(4)]
            else:
                tp, fp, fn, tn = running['tp'], running['fp'], running['fn'], running['tn']
            
            # Compute additional metrics from confusion matrix
            cm_metrics = compute_metrics_from_confusion_matrix(tp, fp, fn, tn)
            epoch_metrics.update(cm_metrics)
            
            # Calculate AUC-ROC 
            epoch_metrics['auc'] = 0.0
            if len(set(all_labels)) > 1:  # Make sure we have both classes
                # Collect predictions from all processes if distributed
                if world_size > 1:
                    # Convert to tensors for gathering
                    all_labels_tensor = torch.tensor(all_labels, device=device)
                    all_probs_tensor = torch.tensor(all_probs, device=device)
                    
                    # Get sizes for padding - ensure consistent data type (int64)
                    local_size = torch.tensor([len(all_labels)], dtype=torch.int64, device=device)
                    sizes = [torch.zeros(1, dtype=torch.int64, device=device) for _ in range(world_size)]
                    dist.all_gather(sizes, local_size)
                    
                    # Pad tensors to maximum size - ensure consistent data types
                    max_size = int(max(sizes).item())
                    padded_labels = torch.zeros(max_size, dtype=torch.float32, device=device)
                    padded_probs = torch.zeros(max_size, dtype=torch.float32, device=device)
                    
                    # Copy data to padded tensors
                    padded_labels[:len(all_labels)] = all_labels_tensor
                    padded_probs[:len(all_probs)] = all_probs_tensor
                    
                    # Gather padded tensors - ensure consistent data types
                    gathered_labels = [torch.zeros(max_size, dtype=torch.float32, device=device) for _ in range(world_size)]
                    gathered_probs = [torch.zeros(max_size, dtype=torch.float32, device=device) for _ in range(world_size)]
                    
                    dist.all_gather(gathered_labels, padded_labels)
                    dist.all_gather(gathered_probs, padded_probs)
                    
                    # Combine and remove padding
                    combined_labels, combined_probs = [], []
                    for i in range(world_size):
                        size = int(sizes[i].item())
                        combined_labels.extend(gathered_labels[i][:size].cpu().numpy())
                        combined_probs.extend(gathered_probs[i][:size].cpu().numpy())
                    
                    # Calculate AUC-ROC
                    try:
                        from custom_metrics import roc_auc_score
                        epoch_metrics['auc'] = roc_auc_score(combined_labels, combined_probs)
                    except (ValueError, ImportError) as e:
                        if rank == 0:
                            logging.warning(f"Error calculating AUC-ROC: {e}")
                else:
                    # Single process calculation
                    try:
                        from custom_metrics import roc_auc_score
                        epoch_metrics['auc'] = roc_auc_score(all_labels, all_probs)
                    except (ValueError, ImportError) as e:
                        if rank == 0:
                            logging.warning(f"Error calculating AUC-ROC: {e}")
            
            # Use the accurate computed accuracy
            epoch_metrics['acc'] = cm_metrics['accuracy']
            
            # Record phase time
            phase_time = time.time() - phase_start_time
            
            # Store confusion matrix for reference
            epoch_metrics['confusion_matrix'] = {'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn}
            epoch_metrics['time'] = phase_time
            
            # Store results for this phase
            phase_results[phase] = epoch_metrics
            
            # Log results (rank 0 only)
            if rank == 0:
                logging.info(f'{phase} Loss: {epoch_metrics["loss"]:.4f} Acc: {epoch_metrics["acc"]:.4f} '
                            f'F1: {epoch_metrics["f1_score"]:.4f}')
                logging.info(f'{phase} Balanced Acc: {epoch_metrics["balanced_accuracy"]:.4f} '
                            f'AUC-ROC: {epoch_metrics["auc"]:.4f}')
                logging.info(f'{phase} Sensitivity: {epoch_metrics["sensitivity"]:.4f} '
                            f'Specificity: {epoch_metrics["specificity"]:.4f}')
                logging.info(f'{phase} Confusion Matrix: TP={tp}, FP={fp}, FN={fn}, TN={tn}')
                logging.info(f'{phase} completed in {phase_time:.2f}s')
                
                if phase == 'train':
                    logging.info(f'Data loading time: {timing["data"]:.2f}s, Forward time: {timing["forward"]:.2f}s, '
                                f'Backward time: {timing["backward"]:.2f}s')
                
                # Log cache stats if available
                try:
                    if hasattr(dataloaders[phase].dataset, 'get_cache_stats'):
                        stats = dataloaders[phase].dataset.get_cache_stats()
                        logging.info(f'{phase} cache: {stats["cache_size"]}/{stats["cache_size"] + 1} items, '
                                    f'hit rate: {stats["hit_rate"]:.2f}%')
                except Exception as e:
                    logging.debug(f"Error getting cache stats: {str(e)}")
        
        # Check early stopping based on validation metrics
        if early_stopping and 'val' in phase_results and rank == 0:
            metric_name = early_stopping_params.get('metric', 'balanced_accuracy')
            metric_value = phase_results['val'].get(metric_name, 0.0)
            
            # Call early stopping to check if we should stop
            if early_stopping(epoch, metric_value):
                logging.info(f"Early stopping triggered at epoch {epoch}")
                break
        
        # Save checkpoints and update best models (rank 0 only)
        if rank == 0 and 'val' in phase_results and checkpoint_dir:
            val_metrics = phase_results['val']
            
            # Check if any metrics improved
            metrics_improved = {}
            best_models = {}
            
            # Create mapping for metric names in the val_metrics dictionary
            metric_mapping = {
                'acc': 'accuracy',
                'f1': 'f1_score',
                'balanced_acc': 'balanced_accuracy',
                'auc': 'auc'
            }
            
            # Update each metric if improved
            for metric in best_metrics:
                metric_key = metric_mapping.get(metric, metric)
                if val_metrics[metric_key] > best_metrics[metric]:
                    best_metrics[metric] = val_metrics[metric_key]
                    metrics_improved[metric] = True
                    
                    # Create checkpoint file path
                    best_models[metric] = os.path.join(checkpoint_dir, f"best_{metric}_model.pth")
                    
                    # Also save as general best model if it's balanced_acc
                    if metric == 'balanced_acc':
                        best_models['general'] = os.path.join(checkpoint_dir, "best_model.pth")
            
            # Save improved models
            for metric, path in best_models.items():
                try:
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        **{f'best_{k}': v for k, v in best_metrics.items()},
                        'metrics': {
                            'accuracy': val_metrics['accuracy'],
                            'sensitivity': val_metrics['sensitivity'],
                            'specificity': val_metrics['specificity'],
                            'f1_score': val_metrics['f1_score'],
                            'balanced_accuracy': val_metrics['balanced_accuracy'],
                            'auc': val_metrics['auc']
                        }
                    }
                    
                    torch.save(checkpoint, path)
                    if metric != 'general':
                        metric_name = 'accuracy' if metric == 'acc' else metric
                        logging.info(f"Saved new best {metric_name} model with {metric_name}: "
                                   f"{best_metrics[metric]:.4f} at {path}")
                    else:
                        logging.info(f"Also saved as general best model at {path}")
                except Exception as e:
                    logging.error(f"Error saving best {metric} model: {str(e)}")
            
            # Save epoch checkpoint
            epoch_path = os.path.join(checkpoint_dir, f"epoch_{epoch}.pth")
            try:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    **{f'best_{k}': v for k, v in best_metrics.items()},
                    **{f'train_{k}': v for k, v in phase_results['train'].items() if k != 'confusion_matrix' and k != 'time'},
                    **{f'val_{k}': v for k, v in phase_results['val'].items() if k != 'confusion_matrix' and k != 'time'}
                }
                torch.save(checkpoint, epoch_path)
                logging.info(f"Saved epoch checkpoint at {epoch_path}")
            except Exception as e:
                logging.error(f"Error saving epoch checkpoint: {str(e)}")
        
        # Log epoch summary (rank 0 only)
        if rank == 0:
            epoch_time = time.time() - epoch_start_time
            logging.info(f"Epoch {epoch} summary:")
            for phase in ['train', 'val']:
                metrics = phase_results[phase]
                logging.info(f" {phase.capitalize()} Loss: {metrics['loss']:.4f}, "
                            f"{phase.capitalize()} Acc: {metrics['accuracy']:.4f}, "
                            f"{phase.capitalize()} F1: {metrics['f1_score']:.4f}")
                logging.info(f" {phase.capitalize()} Balanced Acc: {metrics['balanced_accuracy']:.4f}, "
                            f"{phase.capitalize()} AUC: {metrics['auc']:.4f}")
                logging.info(f" {phase.capitalize()} Sensitivity: {metrics['sensitivity']:.4f}, "
                            f"{phase.capitalize()} Specificity: {metrics['specificity']:.4f}")
            logging.info(f" Total Epoch Time: {epoch_time:.2f}s")
        
        # Synchronize processes at end of epoch
        if world_size > 1:
            dist.barrier()
    
    # Final summary (rank 0 only)
    if rank == 0:
        time_elapsed = time.time() - since
        hours, remainder = divmod(time_elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)
        logging.info("========== Training Completed ==========")
        logging.info(f'Training complete in {int(hours)}h {int(minutes)}m {seconds:.2f}s')
        logging.info(f'Best val Acc: {best_metrics["acc"]:.6f}, Best val F1: {best_metrics["f1"]:.6f}')
        logging.info(f'Best val Balanced Acc: {best_metrics["balanced_acc"]:.6f}, Best val AUC: {best_metrics["auc"]:.6f}')
        
        # Load best model weights based on balanced accuracy
        if checkpoint_dir:
            best_balanced_acc_path = os.path.join(checkpoint_dir, "best_balanced_acc_model.pth")
            best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
            
            if os.path.exists(best_balanced_acc_path):
                best_checkpoint = torch.load(best_balanced_acc_path)
                model.load_state_dict(best_checkpoint['model_state_dict'])
                logging.info(f"Loaded best balanced accuracy model with balanced acc: "
                           f"{best_checkpoint.get('best_balanced_acc', 0.0):.6f}")
            elif os.path.exists(best_model_path):
                best_checkpoint = torch.load(best_model_path)
                model.load_state_dict(best_checkpoint['model_state_dict'])
                logging.info("Loaded best model weights")
    
    # Broadcast best model to all processes
    if world_size > 1:
        # Wait for rank 0 to load best model
        dist.barrier()
        
        # Broadcast model parameters
        for param in model.parameters():
            dist.broadcast(param.data, src=0)
    
    return model
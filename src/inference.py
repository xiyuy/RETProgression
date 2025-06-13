import os
import argparse
import torch
import logging
import numpy as np
import pandas as pd
import time
import sys
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import JoslinData, get_transforms
from custom_metrics import roc_auc_score, balanced_accuracy_score, confusion_matrix_with_stats
from timm import create_model

def configure_logging(log_dir):
    """Configure logging with log file clearing"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'inference_results.log')
    
    # Clear previous log file
    try:
        with open(log_file, 'w') as f:
            f.write(f"=== New Inference Run: {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n\n")
    except Exception as e:
        print(f"Warning: Could not clear log file: {e}")
    
    # Set up logging
    logger = logging.getLogger()
    logger.handlers = []
    
    # Create handlers
    fh = logging.FileHandler(log_file, mode='a')
    ch = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s] - %(message)s', '%Y-%m-%d %H:%M:%S')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.setLevel(logging.INFO)
    
    # Basic system info
    if torch.cuda.is_available():
        logger.info(f"System: {sys.platform}, Python: {sys.version.split()[0]}, PyTorch: {torch.__version__}, GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.info(f"System: {sys.platform}, Python: {sys.version.split()[0]}, PyTorch: {torch.__version__}, CUDA: Not available")
    
    return logger

def load_model(checkpoint_path, model_name="vit_base_patch16_224", num_classes=2, device="cuda"):
    """Load model from checkpoint"""
    model = create_model(model_name, pretrained=False, num_classes=num_classes, img_size=1000)
    
    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    # Handle DDP prefix
    if all(k.startswith('module.') for k in model_state_dict.keys()):
        model_state_dict = {k[7:]: v for k, v in model_state_dict.items()}
    
    model.load_state_dict(model_state_dict)
    model = model.to(device).eval()
    
    # Log model details
    logging.info(f"Loaded model from {checkpoint_path}")
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Model parameters: {total_params:,} total, {total_params:,} trainable")
    
    # Log checkpoint metrics if available
    for key in ['val_acc', 'val_balanced_acc', 'val_f1', 'val_auc']:
        if isinstance(checkpoint, dict) and key in checkpoint:
            logging.info(f"Checkpoint {key}: {checkpoint[key]:.4f}")
    
    return model

def load_test_dataset(data_dir, annotations_file, resolution=1000):
    """Load test dataset"""
    transforms_dict = get_transforms('none', resolution=resolution)
    
    # Create dataset
    test_dataset = JoslinData(
        data_dir=data_dir,
        annotations_file=annotations_file,
        img_dir="Exports_02052025",
        transform=transforms_dict['val']
    )
    
    logging.info(f"JoslinData: Loaded {len(test_dataset)} samples from {annotations_file}")
    
    return test_dataset

def evaluate_model(model, test_loader, device, normalization_transform=None):
    """Evaluate model on test set with progress tracking"""
    model.eval()
    all_predictions = []
    all_probabilities = []
    all_targets = []
    
    # Calculate total number of samples
    total_samples = len(test_loader.dataset)
    processed_samples = 0
    total_batches = len(test_loader)
    
    # Start time
    start_time = time.time()
    last_update_time = start_time
    update_interval = 2.0  # Update every 2 seconds
    
    logging.info(f"Starting evaluation on {total_samples} test samples ({total_batches} batches)...")
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            batch_size = inputs.size(0)
            
            # Process batch
            inputs = inputs.to(device, non_blocking=True)
            if normalization_transform:
                inputs = normalization_transform(inputs)
            
            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            
            # Store results
            all_predictions.append(preds.cpu())
            all_probabilities.append(probs[:, 1].cpu())  # Store probability of positive class
            all_targets.append(targets)
            
            # Update counters
            processed_samples += batch_size
            percent_complete = (processed_samples / total_samples) * 100
            
            # Update progress periodically
            current_time = time.time()
            if (current_time - last_update_time > update_interval) or (batch_idx == len(test_loader) - 1):
                elapsed = current_time - start_time
                samples_per_sec = processed_samples / elapsed if elapsed > 0 else 0
                remaining = (total_samples - processed_samples) / samples_per_sec if samples_per_sec > 0 else 0
                
                # Format times
                elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
                remaining_str = time.strftime("%H:%M:%S", time.gmtime(remaining))
                
                # Progress bar
                bar_len = 30
                filled_len = int(bar_len * processed_samples / total_samples)
                bar = '█' * filled_len + '░' * (bar_len - filled_len)
                
                # Log progress
                msg = (f"Progress: [{bar}] {processed_samples}/{total_samples} samples ({percent_complete:.1f}%) | "
                       f"Batch: {batch_idx+1}/{total_batches} | Elapsed: {elapsed_str} | "
                       f"Remaining: {remaining_str} | Speed: {samples_per_sec:.1f} samples/sec")
                print(msg)
                logging.info(msg)
                last_update_time = current_time
    
    # Concat results
    all_predictions = torch.cat(all_predictions).numpy()
    all_probabilities = torch.cat(all_probabilities).numpy()
    all_targets = torch.cat(all_targets).numpy()
    
    # Log completion
    total_time = time.time() - start_time
    logging.info(f"Evaluation completed on {len(all_targets)} samples in {time.strftime('%H:%M:%S', time.gmtime(total_time))}")
    
    return all_targets, all_predictions, all_probabilities

def calculate_optimal_threshold(y_true, y_prob):
    """Calculate optimal threshold based on balanced accuracy"""
    thresholds = np.linspace(0.01, 0.99, 99)
    best_threshold = 0.5
    best_balanced_acc = 0.0
    
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        balanced_acc = balanced_accuracy_score(y_true, y_pred)
        
        if balanced_acc > best_balanced_acc:
            best_balanced_acc = balanced_acc
            best_threshold = threshold
    
    return best_threshold, best_balanced_acc

def log_metrics(y_true, y_pred, y_prob):
    """Log all relevant metrics"""
    # Calculate confusion matrix with stats
    metrics = confusion_matrix_with_stats(y_true, y_pred)
    
    # Calculate ROC AUC
    try:
        auc = roc_auc_score(y_true, y_prob)
        metrics['auc'] = auc
    except Exception as e:
        logging.warning(f"Error calculating ROC AUC: {e}")
        metrics['auc'] = 0.0
    
    # Log metrics
    logging.info("\nPerformance Metrics:")
    logging.info("====================")
    for metric in ['accuracy', 'balanced_accuracy', 'sensitivity', 'specificity', 
                   'precision', 'f1_score', 'auc']:
        logging.info(f"{metric.capitalize()}: {metrics[metric]:.4f}")
    
    # Log confusion matrix
    tp, fp = metrics['TP'], metrics['FP']
    fn, tn = metrics['FN'], metrics['TN']
    
    cm_str = f"\nConfusion Matrix:\n"
    cm_str += f"  | Pred Negative | Pred Positive\n"
    cm_str += f"--+---------------+-------------\n"
    cm_str += f"Actual Negative | {tn:13d} | {fp:13d}\n"
    cm_str += f"Actual Positive | {fn:13d} | {tp:13d}\n"
    logging.info(cm_str)
    
    return metrics

def evaluate_with_optimal_threshold(y_true, y_prob):
    """Evaluate with optimal threshold for balanced accuracy"""
    # Find optimal threshold
    optimal_threshold, optimal_bal_acc = calculate_optimal_threshold(y_true, y_prob)
    
    # Get predictions with optimal threshold
    y_pred_optimal = (y_prob >= optimal_threshold).astype(int)
    
    logging.info(f"\nOptimal threshold: {optimal_threshold:.4f} (balanced accuracy: {optimal_bal_acc:.4f})")
    
    # Log metrics with optimal threshold
    logging.info("\nMetrics at optimal threshold:")
    metrics_optimal = log_metrics(y_true, y_pred_optimal, y_prob)
    
    return optimal_threshold, metrics_optimal

def generate_error_analysis(y_true, y_pred, y_prob, output_dir):
    """Generate error analysis for false positives and false negatives"""
    # Create error analysis dataframe
    df = pd.DataFrame({
        'true_label': y_true,
        'predicted_label': y_pred,
        'probability': y_prob
    })
    
    # Identify false positives and false negatives
    df['false_positive'] = (df['true_label'] == 0) & (df['predicted_label'] == 1)
    df['false_negative'] = (df['true_label'] == 1) & (df['predicted_label'] == 0)
    
    # Create error analysis directory and save
    error_dir = os.path.join(output_dir, 'error_analysis')
    os.makedirs(error_dir, exist_ok=True)
    error_file = os.path.join(error_dir, 'error_analysis.csv')
    df.to_csv(error_file, index=False)
    
    # Log error statistics
    num_fp = df['false_positive'].sum()
    num_fn = df['false_negative'].sum()
    logging.info(f"\nError Analysis: {num_fp} false positives, {num_fn} false negatives")
    
    # Log probability distributions for errors
    if num_fp > 0:
        fp_probs = df[df['false_positive']]['probability']
        logging.info(f"FP probabilities - Min: {fp_probs.min():.4f}, Max: {fp_probs.max():.4f}, Mean: {fp_probs.mean():.4f}")
    
    if num_fn > 0:
        fn_probs = df[df['false_negative']]['probability']
        logging.info(f"FN probabilities - Min: {fn_probs.min():.4f}, Max: {fn_probs.max():.4f}, Mean: {fn_probs.mean():.4f}")
    
    logging.info(f"Error analysis saved to {error_file}")
    return df

def main():
    parser = argparse.ArgumentParser(description='Model Inference on Test Set')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to data directory')
    parser.add_argument('--annotations_file', type=str, default='referable_img_grades_test.csv',
                        help='Name of test annotations file')
    parser.add_argument('--output_dir', type=str, default='inference_results',
                        help='Directory to save inference results')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for inference')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run inference on')
    parser.add_argument('--resolution', type=int, default=1000, help='Image resolution')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Configure logging
    logger = configure_logging(args.output_dir)
    
    # Log arguments
    logging.info("Inference Parameters:")
    for arg, value in sorted(vars(args).items()):
        logging.info(f"  {arg}: {value}")
    
    # Check CUDA availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        logging.warning("CUDA is not available, falling back to CPU")
        args.device = 'cpu'
    
    device = torch.device(args.device)
    logging.info(f"Using device: {device}")
    
    try:
        # Load model and dataset
        model = load_model(args.checkpoint, device=device)
        test_dataset = load_test_dataset(args.data_dir, args.annotations_file, resolution=args.resolution)
        
        # Create data loader
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )
        
        # Define normalization transform
        normalization_transform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        # Evaluate model
        logging.info("\nEvaluating model on test set...")
        y_true, y_pred, y_prob = evaluate_model(model, test_loader, device, normalization_transform)
        
        # Log metrics with default threshold (0.5)
        logging.info("\nResults with default threshold (0.5):")
        metrics_default = log_metrics(y_true, y_pred, y_prob)
        
        # Evaluate with optimal threshold
        optimal_threshold, metrics_optimal = evaluate_with_optimal_threshold(y_true, y_prob)
        
        # Generate error analysis
        error_df = generate_error_analysis(y_true, y_pred, y_prob, args.output_dir)
        
        # Save results to CSV
        results = {
            'default_threshold': 0.5,
            'optimal_threshold': optimal_threshold,
            'default_accuracy': metrics_default['accuracy'],
            'default_balanced_accuracy': metrics_default['balanced_accuracy'],
            'default_sensitivity': metrics_default['sensitivity'],
            'default_specificity': metrics_default['specificity'],
            'default_precision': metrics_default['precision'],
            'default_f1': metrics_default['f1_score'],
            'default_auc': metrics_default['auc'],
            'optimal_accuracy': metrics_optimal['accuracy'],
            'optimal_balanced_accuracy': metrics_optimal['balanced_accuracy'],
            'optimal_sensitivity': metrics_optimal['sensitivity'],
            'optimal_specificity': metrics_optimal['specificity'],
            'optimal_precision': metrics_optimal['precision'],
            'optimal_f1': metrics_optimal['f1_score'],
            'optimal_auc': metrics_optimal['auc'],
        }
        
        # Save results
        results_df = pd.DataFrame([results])
        results_file = os.path.join(args.output_dir, 'inference_metrics.csv')
        results_df.to_csv(results_file, index=False)
        logging.info(f"\nSummary metrics saved to {results_file}")
        
        # Final summary
        logging.info("\nInference completed successfully")
        logging.info(f"Best metrics (at threshold={optimal_threshold:.4f}):")
        logging.info(f"  Balanced Accuracy: {metrics_optimal['balanced_accuracy']:.4f}")
        logging.info(f"  Sensitivity: {metrics_optimal['sensitivity']:.4f}")
        logging.info(f"  Specificity: {metrics_optimal['specificity']:.4f}")
        logging.info(f"  F1 Score: {metrics_optimal['f1_score']:.4f}")
        
    except Exception as e:
        logging.error(f"Error during inference: {e}")
        import traceback
        logging.error(traceback.format_exc())

if __name__ == '__main__':
    main()
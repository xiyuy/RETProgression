
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
    log_file = os.path.join(log_dir, 'validation_inference.log')
    
    # Clear previous log file
    try:
        with open(log_file, 'w') as f:
            f.write(f"=== New Validation Inference Run: {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n\n")
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
    
    return logger

def load_model(checkpoint_path, model_name, num_classes=2, img_size=1024, device="cuda"):
    """Load model from checkpoint"""
    # Create model with the same architecture used in training
    model = create_model(model_name, pretrained=False, num_classes=num_classes, img_size=img_size)
    
    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    # Handle DDP prefix if present
    if all(k.startswith('module.') for k in model_state_dict.keys()):
        model_state_dict = {k[7:]: v for k, v in model_state_dict.items()}
    
    model.load_state_dict(model_state_dict)
    model = model.to(device).eval()
    
    # Log model details
    logging.info(f"Loaded model from {checkpoint_path}")
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Model parameters: {total_params:,} total")
    
    # Log checkpoint metrics if available
    for key in ['val_acc', 'val_balanced_acc', 'val_f1', 'val_auc']:
        if isinstance(checkpoint, dict) and key in checkpoint:
            logging.info(f"Checkpoint {key}: {checkpoint[key]:.4f}")
    
    return model

def load_validation_dataset(data_dir, annotations_file, img_dir, resolution=1024):
    """Load validation dataset"""
    transforms_dict = get_transforms('none', resolution=resolution)

    # argparse additions
    parser.add_argument('--img_dir', type=str, default='clean_dataset_0701202',
                    help='Relative image folder used by the dataset')

    val_dataset = JoslinData(
        data_dir=data_dir,
        annotations_file=annotations_file,
        img_dir=img_dir,  # <- was hardcoded
        transform=transforms_dict['val']
    )

    logging.info(f"Loaded {len(val_dataset)} validation samples from {annotations_file} (img_dir={img_dir})")
    return val_dataset

def evaluate_validation_set(model, val_loader, device, normalization_transform=None):
    """Evaluate model on validation set and return detailed predictions"""
    model.eval()
    
    # Store results with image info
    results = []
    
    # Calculate total number of samples
    total_samples = len(val_loader.dataset)
    processed_samples = 0
    total_batches = len(val_loader)
    
    # Start time
    start_time = time.time()
    last_update_time = start_time
    update_interval = 2.0  # Update every 2 seconds
    
    logging.info(f"Starting validation evaluation on {total_samples} samples ({total_batches} batches)...")
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            batch_size = inputs.size(0)
            
            # Process batch
            inputs = inputs.to(device, non_blocking=True)
            if normalization_transform:
                inputs = normalization_transform(inputs)
            
            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            
            # Store results for each sample in the batch
            for i in range(batch_size):
                sample_idx = batch_idx * val_loader.batch_size + i
                
                # Get image filename from dataset
                try:
                    # Access the underlying dataset to get image info
                    if hasattr(val_loader.dataset, 'dataset'):
                        # If it's a cached dataset, get the underlying dataset
                        base_dataset = val_loader.dataset.dataset
                    else:
                        base_dataset = val_loader.dataset
                    
                    # Get image filename from the annotations
                    img_filename = base_dataset.img_labels.iloc[sample_idx, 0]
                    
                    results.append({
                        'image_filename': img_filename,
                        'true_label': targets[i].cpu().item(),
                        'predicted_label': preds[i].cpu().item(),
                        'probability_class_0': probs[i, 0].cpu().item(),
                        'probability_class_1': probs[i, 1].cpu().item(),
                        'prediction_correct': (preds[i] == targets[i]).cpu().item()
                    })
                except Exception as e:
                    logging.warning(f"Error getting image info for sample {sample_idx}: {e}")
                    results.append({
                        'image_filename': f'sample_{sample_idx}',
                        'true_label': targets[i].cpu().item(),
                        'predicted_label': preds[i].cpu().item(),
                        'probability_class_0': probs[i, 0].cpu().item(),
                        'probability_class_1': probs[i, 1].cpu().item(),
                        'prediction_correct': (preds[i] == targets[i]).cpu().item()
                    })
            
            # Update counters
            processed_samples += batch_size
            percent_complete = (processed_samples / total_samples) * 100
            
            # Update progress periodically
            current_time = time.time()
            if (current_time - last_update_time > update_interval) or (batch_idx == len(val_loader) - 1):
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
    
    # Log completion
    total_time = time.time() - start_time
    logging.info(f"Validation evaluation completed on {len(results)} samples in {time.strftime('%H:%M:%S', time.gmtime(total_time))}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Validation Set Inference with Detailed Output')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint (e.g., best_f1_model.pth)')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to data directory')
    parser.add_argument('--annotations_file', type=str, default='clean_gradable_dr_val.csv',
                        help='Name of validation annotations file')
    parser.add_argument('--output_dir', type=str, default='validation_results',
                        help='Directory to save results')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for inference')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run inference on')
    parser.add_argument('--resolution', type=int, default=1024, help='Image resolution')
    parser.add_argument('--model_name', type=str, default='swinv2_large_window12to16_192to256.ms_in22k_ft_in1k',
                        help='Model architecture name')
    parser.add_argument('--img_size', type=int, default=1024, help='Model input image size')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Configure logging
    logger = configure_logging(args.output_dir)
    
    # Log arguments
    logging.info("Validation Inference Parameters:")
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
        model = load_model(args.checkpoint, args.model_name, num_classes=2, img_size=args.img_size, device=device)
        val_dataset = load_validation_dataset(args.data_dir, args.annotations_file, resolution=args.resolution)
        
        # Create data loader
        val_loader = DataLoader(
            val_dataset,
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
        logging.info("\nEvaluating model on validation set...")
        results = evaluate_validation_set(model, val_loader, device, normalization_transform)
        
        # Convert results to DataFrame
        df = pd.DataFrame(results)
        
        # Add label mapping for better readability
        df['true_label_name'] = df['true_label'].map({0: 'Non-Gradable', 1: 'Gradable'})
        df['predicted_label_name'] = df['predicted_label'].map({0: 'Non-Gradable', 1: 'Gradable'})
        
        # Reorder columns for better readability
        df = df[['image_filename', 'true_label', 'true_label_name', 'predicted_label', 'predicted_label_name', 
                 'probability_class_0', 'probability_class_1', 'prediction_correct']]
        
        # Save detailed results
        detailed_results_file = os.path.join(args.output_dir, 'validation_detailed_results.csv')
        df.to_csv(detailed_results_file, index=False)
        logging.info(f"Detailed results saved to {detailed_results_file}")
        
        # Calculate and log summary metrics
        true_labels = df['true_label'].values
        predicted_labels = df['predicted_label'].values
        probabilities = df['probability_class_1'].values
        
        # Calculate metrics
        metrics = confusion_matrix_with_stats(true_labels, predicted_labels)
        auc = roc_auc_score(true_labels, probabilities)
        
        # Log summary metrics
        logging.info("\nValidation Set Performance Summary:")
        logging.info("=" * 50)
        logging.info(f"Total samples: {len(df)}")
        logging.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logging.info(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
        logging.info(f"Sensitivity (Recall): {metrics['sensitivity']:.4f}")
        logging.info(f"Specificity: {metrics['specificity']:.4f}")
        logging.info(f"Precision: {metrics['precision']:.4f}")
        logging.info(f"F1 Score: {metrics['f1_score']:.4f}")
        logging.info(f"AUC-ROC: {auc:.4f}")
        
        # Log confusion matrix
        logging.info("\nConfusion Matrix:")
        logging.info(f"True Positives: {metrics['TP']}")
        logging.info(f"False Positives: {metrics['FP']}")
        logging.info(f"False Negatives: {metrics['FN']}")
        logging.info(f"True Negatives: {metrics['TN']}")
        
        # Create summary metrics file
        summary_metrics = {
            'total_samples': len(df),
            'accuracy': metrics['accuracy'],
            'balanced_accuracy': metrics['balanced_accuracy'],
            'sensitivity': metrics['sensitivity'],
            'specificity': metrics['specificity'],
            'precision': metrics['precision'],
            'f1_score': metrics['f1_score'],
            'auc_roc': auc,
            'true_positives': metrics['TP'],
            'false_positives': metrics['FP'],
            'false_negatives': metrics['FN'],
            'true_negatives': metrics['TN']
        }
        
        summary_df = pd.DataFrame([summary_metrics])
        summary_file = os.path.join(args.output_dir, 'validation_summary_metrics.csv')
        summary_df.to_csv(summary_file, index=False)
        logging.info(f"Summary metrics saved to {summary_file}")
        
        # Create error analysis
        errors_df = df[df['prediction_correct'] == 0].copy()
        if len(errors_df) > 0:
            errors_file = os.path.join(args.output_dir, 'validation_errors.csv')
            errors_df.to_csv(errors_file, index=False)
            logging.info(f"Error analysis saved to {errors_file} ({len(errors_df)} errors)")
            
            # Log error breakdown
            fp_count = len(errors_df[errors_df['true_label'] == 0])
            fn_count = len(errors_df[errors_df['true_label'] == 1])
            logging.info(f"False Positives: {fp_count}, False Negatives: {fn_count}")
        
        logging.info("\nValidation inference completed successfully!")
        
    except Exception as e:
        logging.error(f"Error during validation inference: {e}")
        import traceback
        logging.error(traceback.format_exc())

if __name__ == '__main__':
    main()
#!/usr/bin/env python
"""
Fixed prediction script that uses the correct ID column from the CSV.
"""

import os
import sys
import time
import argparse
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from timm import create_model
from PIL import Image

# Import custom metrics
from custom_metrics import confusion_matrix_with_stats, roc_auc_score

class FixedJoslinData(Dataset):
    """Dataset that reads from the correct ID column"""
    
    def __init__(self, data_dir, annotations_file, img_dir, transform=None, img_size=1024, 
                 id_column='ID', label_column='gradable_binary_DR'):
        """
        Initialize dataset with correct column specification
        
        Args:
            data_dir: Base directory containing the data
            annotations_file: CSV file with annotations
            img_dir: Directory containing the images
            transform: Transformations to apply to the images
            img_size: Size for dummy tensors when files are missing
            id_column: Name of the column containing image IDs (default: 'ID')
            label_column: Name of the column containing labels (default: 'gradable_binary_DR')
        """
        self.img_dir = os.path.join(data_dir, img_dir)
        
        # Load CSV
        csv_path = os.path.join(data_dir, annotations_file)
        self.df = pd.read_csv(csv_path)
        
        logging.info(f"CSV loaded: {len(self.df)} rows")
        logging.info(f"Columns: {list(self.df.columns)}")
        
        # Check if specified columns exist
        if id_column not in self.df.columns:
            raise ValueError(f"ID column '{id_column}' not found. Available columns: {list(self.df.columns)}")
        if label_column not in self.df.columns:
            raise ValueError(f"Label column '{label_column}' not found. Available columns: {list(self.df.columns)}")
        
        # Extract IDs and labels
        self.ids = self.df[id_column].values
        self.labels = self.df[label_column].values
        
        self.transform = transform
        self.img_size = img_size
        
        # Pre-check which files exist
        logging.info("Checking file availability...")
        self.valid_indices = []
        self.missing_files = []
        
        # Show first few IDs for debugging
        logging.info(f"First 5 IDs from '{id_column}' column: {self.ids[:5].tolist()}")
        
        for idx in range(len(self.df)):
            img_id = str(self.ids[idx])
            
            # Handle float IDs (e.g., "12345.0" -> "12345")
            if '.' in img_id and img_id.replace('.', '').replace('-', '').isdigit():
                img_id = img_id.split('.')[0]
            
            # Add extension if not present
            if not img_id.endswith(('.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG')):
                img_name = f"{img_id}.jpg"
            else:
                img_name = img_id
            
            img_path = os.path.join(self.img_dir, img_name)
            
            if os.path.exists(img_path):
                self.valid_indices.append(idx)
            else:
                self.missing_files.append((idx, img_id, img_path))
        
        # Report results
        logging.info(f"Found {len(self.valid_indices)} valid images out of {len(self.df)} total")
        
        if self.missing_files:
            logging.warning(f"Missing {len(self.missing_files)} files")
            # Show first few missing as examples
            for i, (idx, img_id, path) in enumerate(self.missing_files[:5]):
                logging.debug(f"Missing example {i+1}: ID={img_id}, Path={path}")
        
        # Show class distribution
        if len(self.valid_indices) > 0:
            valid_labels = self.labels[self.valid_indices]
            unique_labels, counts = np.unique(valid_labels, return_counts=True)
            logging.info("Class distribution in valid samples:")
            for label, count in zip(unique_labels, counts):
                pct = 100 * count / len(self.valid_indices)
                logging.info(f"  Class {label}: {count} samples ({pct:.1f}%)")
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        """Get image and label for a given index"""
        # Map to actual index in the dataframe
        actual_idx = self.valid_indices[idx]
        
        # Get image ID and construct path
        img_id = str(self.ids[actual_idx])
        
        # Handle float IDs
        if '.' in img_id and img_id.replace('.', '').replace('-', '').isdigit():
            img_id = img_id.split('.')[0]
        
        # Add extension if needed
        if not img_id.endswith(('.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG')):
            img_name = f"{img_id}.jpg"
        else:
            img_name = img_id
        
        img_path = os.path.join(self.img_dir, img_name)
        
        try:
            # Open and transform image
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image_tensor = self.transform(image)
            else:
                image_tensor = transforms.ToTensor()(image)
            
            # Get label
            label = int(self.labels[actual_idx])
            label_tensor = torch.tensor(label, dtype=torch.long)
            
            return image_tensor, label_tensor, actual_idx
            
        except Exception as e:
            logging.error(f"Error loading {img_path}: {str(e)}")
            # Return dummy data
            if self.transform:
                dummy_image = Image.new('RGB', (self.img_size, self.img_size), color='black')
                dummy_tensor = self.transform(dummy_image)
            else:
                dummy_tensor = torch.zeros((3, self.img_size, self.img_size))
            
            label = int(self.labels[actual_idx]) if actual_idx < len(self.labels) else 0
            return dummy_tensor, torch.tensor(label, dtype=torch.long), actual_idx

def setup_logging(output_dir):
    """Setup logging configuration"""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, 'predictions.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Starting predictions - {time.strftime('%Y-%m-%d %H:%M:%S')}")

def load_model(checkpoint_path, model_name, num_classes=2, img_size=1024, device='cuda'):
    """Load model from checkpoint"""
    logging.info(f"Loading model: {model_name}")
    
    model = create_model(
        model_name,
        pretrained=False,
        num_classes=num_classes,
        img_size=img_size
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    # Handle DDP prefix
    if all(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    
    logging.info(f"Model loaded from: {checkpoint_path}")
    
    # Log checkpoint metrics if available
    if isinstance(checkpoint, dict):
        for metric in ['val_acc', 'val_balanced_accuracy', 'val_f1_score', 'val_auc']:
            if metric in checkpoint:
                logging.info(f"Checkpoint {metric}: {checkpoint[metric]:.4f}")
    
    return model

def run_predictions(model, dataloader, device, dataset, normalize=True):
    """Run model predictions"""
    model.eval()
    results = []
    
    norm_transform = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ) if normalize else None
    
    total_batches = len(dataloader)
    total_samples = len(dataloader.dataset)
    processed = 0
    
    logging.info(f"Running predictions on {total_samples} samples...")
    start_time = time.time()
    
    with torch.no_grad():
        for batch_idx, (inputs, labels, indices) in enumerate(dataloader):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            if norm_transform:
                inputs = norm_transform(inputs)
            
            # Forward pass
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            # Store results
            for i in range(len(inputs)):
                actual_idx = indices[i].item()
                img_id = str(dataset.ids[actual_idx])
                
                # Clean up ID for display
                if '.' in img_id and img_id.replace('.', '').replace('-', '').isdigit():
                    img_id = img_id.split('.')[0]
                
                result = {
                    'image_id': img_id,
                    'original_index': actual_idx,
                    'true_label': labels[i].cpu().item(),
                    'predicted_label': preds[i].cpu().item(),
                    'prob_class_0': probs[i, 0].cpu().item(),
                    'prob_class_1': probs[i, 1].cpu().item(),
                    'logit_0': outputs[i, 0].cpu().item(),
                    'logit_1': outputs[i, 1].cpu().item(),
                    'logit_margin': (outputs[i, 1] - outputs[i, 0]).cpu().item(),
                    'correct': (preds[i] == labels[i]).cpu().item()
                }
                results.append(result)
            
            processed += len(inputs)
            
            # Progress update
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
                elapsed = time.time() - start_time
                progress = 100.0 * processed / total_samples
                speed = processed / elapsed if elapsed > 0 else 0
                remaining = (total_samples - processed) / speed if speed > 0 else 0
                
                logging.info(f"Progress: {progress:.1f}% ({processed}/{total_samples}) | "
                           f"Batch: {batch_idx + 1}/{total_batches} | "
                           f"Speed: {speed:.1f} samples/sec | "
                           f"ETA: {remaining:.0f}s")
    
    logging.info(f"Predictions completed in {time.time() - start_time:.1f} seconds")
    return results

def save_results(results, output_dir, checkpoint_name):
    """Save results and compute metrics"""
    
    df = pd.DataFrame(results)
    
    # Save predictions
    pred_file = os.path.join(output_dir, f'predictions_{checkpoint_name}.csv')
    df.to_csv(pred_file, index=False)
    logging.info(f"Predictions saved to: {pred_file}")
    
    # Compute metrics
    if len(np.unique(df['true_label'])) > 1:
        y_true = df['true_label'].values
        y_pred = df['predicted_label'].values
        y_prob = df['prob_class_1'].values
        
        metrics = confusion_matrix_with_stats(y_true, y_pred)
        
        try:
            auc = roc_auc_score(y_true, y_prob)
            metrics['auc_roc'] = auc
        except Exception as e:
            logging.warning(f"Could not compute AUC: {e}")
            metrics['auc_roc'] = np.nan
        
        # Create summary
        summary = pd.DataFrame([{
            'total_samples': len(df),
            'accuracy': metrics['accuracy'],
            'balanced_accuracy': metrics['balanced_accuracy'],
            'sensitivity': metrics['sensitivity'],
            'specificity': metrics['specificity'],
            'precision': metrics['precision'],
            'f1_score': metrics['f1_score'],
            'auc_roc': metrics.get('auc_roc', np.nan),
            'TP': metrics['TP'],
            'FP': metrics['FP'],
            'FN': metrics['FN'],
            'TN': metrics['TN']
        }])
        
        summary_file = os.path.join(output_dir, f'metrics_{checkpoint_name}.csv')
        summary.to_csv(summary_file, index=False)
        logging.info(f"Metrics saved to: {summary_file}")
        
        # Log metrics
        logging.info("\n" + "="*50)
        logging.info("PERFORMANCE METRICS:")
        logging.info("="*50)
        for key in ['accuracy', 'balanced_accuracy', 'sensitivity', 'specificity', 
                   'precision', 'f1_score', 'auc_roc']:
            if key in summary.columns:
                value = summary[key].iloc[0]
                if not np.isnan(value):
                    logging.info(f"{key:20s}: {value:.4f}")
        
        logging.info(f"\nConfusion Matrix:")
        logging.info(f"  TP={metrics['TP']}, FP={metrics['FP']}")
        logging.info(f"  FN={metrics['FN']}, TN={metrics['TN']}")

def main():
    parser = argparse.ArgumentParser(description='Fixed prediction script with correct ID column')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--model_name', default='swinv2_large_window12to16_192to256.ms_in22k_ft_in1k')
    parser.add_argument('--data_dir', required=True, help='Root data directory')
    parser.add_argument('--annotations_file', required=True, help='CSV file')
    parser.add_argument('--img_dir', required=True, help='Image directory')
    parser.add_argument('--id_column', default='ID', help='Column name for image IDs')
    parser.add_argument('--label_column', default='gradable_binary_DR', help='Column name for labels')
    parser.add_argument('--img_size', type=int, default=1024, help='Image size')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=0, help='Workers')
    parser.add_argument('--output_dir', default='fixed_predictions', help='Output directory')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--no_normalize', action='store_true', help='Skip normalization')
    parser.add_argument('--no_resize', action='store_true', help='No resize')
    
    args = parser.parse_args()
    
    setup_logging(args.output_dir)
    device = torch.device(args.device)
    logging.info(f"Using device: {device}")
    
    checkpoint_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
    
    try:
        # Load model
        model = load_model(
            args.checkpoint,
            args.model_name,
            num_classes=2,
            img_size=args.img_size,
            device=device
        )
        
        # Create transform
        if args.no_resize:
            transform = transforms.Compose([transforms.ToTensor()])
        else:
            transform = transforms.Compose([
                transforms.Resize((args.img_size, args.img_size)),
                transforms.ToTensor()
            ])
        
        # Load dataset with correct column names
        dataset = FixedJoslinData(
            data_dir=args.data_dir,
            annotations_file=args.annotations_file,
            img_dir=args.img_dir,
            transform=transform,
            img_size=args.img_size,
            id_column=args.id_column,
            label_column=args.label_column
        )
        
        if len(dataset) == 0:
            logging.error("No valid samples found!")
            sys.exit(1)
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True if args.device == 'cuda' else False
        )
        
        # Run predictions
        results = run_predictions(
            model, 
            dataloader, 
            device,
            dataset,
            normalize=not args.no_normalize
        )
        
        # Save results
        save_results(results, args.output_dir, checkpoint_name)
        
        logging.info("\n" + "="*50)
        logging.info("COMPLETED SUCCESSFULLY")
        logging.info("="*50)
        
    except Exception as e:
        logging.error(f"Error: {e}")
        import traceback
        logging.error(traceback.format_exc())
        sys.exit(1)

if __name__ == '__main__':
    main()
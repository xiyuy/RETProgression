"""
Ensemble predictions from cross-validation folds for retinal image classification.
Creates ensemble predictions for the complete dataset by combining fold predictions.
Includes threshold tuning and conservative voting options.
"""
import os
import argparse
import logging
import time
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import json

# Import custom metrics
from custom_metrics import (
    roc_auc_score, balanced_accuracy_score, 
    confusion_matrix_with_stats
)

def setup_logging(output_dir):
    """Configure logging"""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, 'ensemble_crossval.log')
    
    # Clear previous log
    with open(log_file, 'w') as f:
        f.write(f"=== Cross-Validation Ensemble Run: {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n\n")
    
    logger = logging.getLogger()
    logger.handlers = []
    
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s] - %(message)s', '%Y-%m-%d %H:%M:%S')
    
    fh = logging.FileHandler(log_file, mode='a')
    fh.setFormatter(formatter)
    
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.setLevel(logging.INFO)
    
    return logger

def load_fold_predictions(pred_files):
    """
    Load prediction files from all folds
    
    Args:
        pred_files: List of paths to prediction CSV files
        
    Returns:
        Dictionary mapping sample IDs to list of predictions across folds
    """
    logging.info(f"Loading {len(pred_files)} fold predictions...")
    
    # Store predictions by sample ID
    predictions_by_sample = defaultdict(lambda: {
        'true_label': None,
        'predictions': [],
        'probabilities': [],
        'logits': []
    })
    
    fold_info = []
    
    for i, pred_file in enumerate(pred_files, 1):
        if not os.path.exists(pred_file):
            logging.error(f"File not found: {pred_file}")
            continue
            
        df = pd.read_csv(pred_file)
        logging.info(f"Fold {i}: Loaded {len(df)} samples from {os.path.basename(pred_file)}")
        
        # Try to detect the correct column names
        # Support multiple naming conventions
        id_col = None
        for possible_id in ['image_id', 'id', 'ID', 'image_filename']:
            if possible_id in df.columns:
                id_col = possible_id
                break
        
        true_label_col = None
        for possible_label in ['true_label', 'y_true', 'label']:
            if possible_label in df.columns:
                true_label_col = possible_label
                break
        
        pred_label_col = None
        for possible_pred in ['predicted_label', 'y_pred', 'prediction']:
            if possible_pred in df.columns:
                pred_label_col = possible_pred
                break
        
        prob_col = None
        for possible_prob in ['prob_class_1', 'y_prob_pos', 'probability_class_1']:
            if possible_prob in df.columns:
                prob_col = possible_prob
                break
        
        # Check if all required columns found
        if not all([id_col, true_label_col, pred_label_col, prob_col]):
            logging.error(f"Missing required columns in {pred_file}")
            logging.error(f"Available columns: {list(df.columns)}")
            logging.error(f"Found - ID: {id_col}, Label: {true_label_col}, Pred: {pred_label_col}, Prob: {prob_col}")
            continue
        
        has_logits = 'logit_0' in df.columns and 'logit_1' in df.columns
        
        logging.info(f"  Using columns - ID: {id_col}, Label: {true_label_col}, Pred: {pred_label_col}, Prob: {prob_col}")
        
        # Aggregate by sample ID
        for _, row in df.iterrows():
            sample_id = str(row[id_col])
            
            # Store true label (should be consistent across folds for same sample)
            if predictions_by_sample[sample_id]['true_label'] is None:
                predictions_by_sample[sample_id]['true_label'] = int(row[true_label_col])
            elif predictions_by_sample[sample_id]['true_label'] != int(row[true_label_col]):
                logging.warning(f"Label mismatch for sample {sample_id}")
            
            # Store predictions
            predictions_by_sample[sample_id]['predictions'].append(int(row[pred_label_col]))
            predictions_by_sample[sample_id]['probabilities'].append(float(row[prob_col]))
            
            if has_logits:
                predictions_by_sample[sample_id]['logits'].append([
                    float(row['logit_0']), 
                    float(row['logit_1'])
                ])
        
        fold_info.append({
            'fold': i,
            'file': os.path.basename(pred_file),
            'samples': len(df),
            'has_logits': has_logits
        })
    
    logging.info(f"Total unique samples: {len(predictions_by_sample)}")
    
    # Log prediction distribution
    pred_counts = [len(v['predictions']) for v in predictions_by_sample.values()]
    logging.info(f"Predictions per sample - Min: {min(pred_counts)}, Max: {max(pred_counts)}, Mean: {np.mean(pred_counts):.2f}")
    
    return predictions_by_sample, fold_info

def ensemble_average(predictions_by_sample):
    """Simple average of probabilities across folds"""
    logging.info("Computing Simple Average ensemble...")
    
    results = []
    for sample_id, data in predictions_by_sample.items():
        avg_prob = np.mean(data['probabilities'])
        
        results.append({
            'id': sample_id,
            'y_true': data['true_label'],
            'y_prob_pos': avg_prob,
            'y_pred': int(avg_prob >= 0.5),
            'n_folds': len(data['predictions']),
            'std_prob': np.std(data['probabilities'])
        })
    
    return pd.DataFrame(results)

def ensemble_logit_average(predictions_by_sample):
    """Average logits then apply softmax"""
    logging.info("Computing Logit Average ensemble...")
    
    results = []
    has_logits = False
    
    for sample_id, data in predictions_by_sample.items():
        if not data['logits']:
            # Fallback to probability average if no logits
            avg_prob = np.mean(data['probabilities'])
        else:
            has_logits = True
            # Average logits
            logits = np.array(data['logits'])
            avg_logits = np.mean(logits, axis=0)
            
            # Softmax
            exp_logits = np.exp(avg_logits - np.max(avg_logits))
            probs = exp_logits / np.sum(exp_logits)
            avg_prob = probs[1]
        
        results.append({
            'id': sample_id,
            'y_true': data['true_label'],
            'y_prob_pos': avg_prob,
            'y_pred': int(avg_prob >= 0.5),
            'n_folds': len(data['predictions'])
        })
    
    if not has_logits:
        logging.warning("No logits found - logit averaging fell back to probability averaging")
    
    return pd.DataFrame(results)

def ensemble_median(predictions_by_sample):
    """Median of probabilities - robust to outliers"""
    logging.info("Computing Median ensemble...")
    
    results = []
    for sample_id, data in predictions_by_sample.items():
        median_prob = np.median(data['probabilities'])
        
        results.append({
            'id': sample_id,
            'y_true': data['true_label'],
            'y_prob_pos': median_prob,
            'y_pred': int(median_prob >= 0.5),
            'n_folds': len(data['predictions']),
            'iqr_prob': np.percentile(data['probabilities'], 75) - np.percentile(data['probabilities'], 25)
        })
    
    return pd.DataFrame(results)

def ensemble_weighted_by_confidence(predictions_by_sample):
    """Weight predictions by their confidence (distance from 0.5)"""
    logging.info("Computing Confidence-Weighted ensemble...")
    
    results = []
    for sample_id, data in predictions_by_sample.items():
        probs = np.array(data['probabilities'])
        
        # Calculate confidence weights (distance from 0.5)
        confidences = np.abs(probs - 0.5)
        
        if np.sum(confidences) > 0:
            weights = confidences / np.sum(confidences)
            weighted_prob = np.sum(probs * weights)
        else:
            weighted_prob = np.mean(probs)
        
        results.append({
            'id': sample_id,
            'y_true': data['true_label'],
            'y_prob_pos': weighted_prob,
            'y_pred': int(weighted_prob >= 0.5),
            'n_folds': len(data['predictions']),
            'mean_confidence': np.mean(confidences)
        })
    
    return pd.DataFrame(results)

def ensemble_max_confidence(predictions_by_sample):
    """Use most confident prediction for each sample"""
    logging.info("Computing Max Confidence ensemble...")
    
    results = []
    for sample_id, data in predictions_by_sample.items():
        probs = np.array(data['probabilities'])
        
        # Find most confident prediction
        confidences = np.abs(probs - 0.5)
        most_confident_idx = np.argmax(confidences)
        
        results.append({
            'id': sample_id,
            'y_true': data['true_label'],
            'y_prob_pos': probs[most_confident_idx],
            'y_pred': data['predictions'][most_confident_idx],
            'n_folds': len(data['predictions']),
            'max_confidence': confidences[most_confident_idx]
        })
    
    return pd.DataFrame(results)

def ensemble_majority_vote(predictions_by_sample, vote_threshold=0.5):
    """
    Majority voting with adjustable threshold
    
    Args:
        vote_threshold: Fraction of models that must agree (0.5=simple majority, higher=more conservative)
    """
    logging.info(f"Computing Majority Vote ensemble (threshold={vote_threshold})...")
    
    results = []
    for sample_id, data in predictions_by_sample.items():
        predictions = np.array(data['predictions'])
        n_models = len(predictions)
        
        # Calculate vote fraction
        positive_votes = np.sum(predictions)
        vote_fraction = positive_votes / n_models
        
        # Predict positive only if enough models agree
        final_prediction = int(vote_fraction >= vote_threshold)
        
        # Use average probability for scoring
        avg_prob = np.mean(data['probabilities'])
        
        results.append({
            'id': sample_id,
            'y_true': data['true_label'],
            'y_pred': final_prediction,
            'y_prob_pos': avg_prob,
            'vote_fraction': vote_fraction,
            'positive_votes': int(positive_votes),
            'n_folds': n_models
        })
    
    return pd.DataFrame(results)

def evaluate_ensemble(result_df, method_name):
    """Calculate metrics for ensemble predictions"""
    y_true = result_df['y_true'].values
    y_pred = result_df['y_pred'].values
    y_prob = result_df['y_prob_pos'].values
    
    # Calculate metrics
    metrics = confusion_matrix_with_stats(y_true, y_pred)
    
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception as e:
        logging.warning(f"Could not compute AUC for {method_name}: {e}")
        auc = np.nan
    
    logging.info(f"\n{'='*60}")
    logging.info(f"{method_name} Results:")
    logging.info(f"{'='*60}")
    logging.info(f"  Accuracy:          {metrics['accuracy']:.4f}")
    logging.info(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    logging.info(f"  Sensitivity:       {metrics['sensitivity']:.4f}")
    logging.info(f"  Specificity:       {metrics['specificity']:.4f}")
    logging.info(f"  Precision:         {metrics['precision']:.4f}")
    logging.info(f"  F1 Score:          {metrics['f1_score']:.4f}")
    logging.info(f"  AUC-ROC:           {auc:.4f}")
    logging.info(f"  Confusion Matrix:  TP={metrics['TP']}, FP={metrics['FP']}, FN={metrics['FN']}, TN={metrics['TN']}")
    
    return {
        'method': method_name,
        'accuracy': metrics['accuracy'],
        'balanced_accuracy': metrics['balanced_accuracy'],
        'sensitivity': metrics['sensitivity'],
        'specificity': metrics['specificity'],
        'precision': metrics['precision'],
        'f1_score': metrics['f1_score'],
        'auc': auc if not np.isnan(auc) else 0.0,
        'TP': metrics['TP'],
        'FP': metrics['FP'],
        'FN': metrics['FN'],
        'TN': metrics['TN']
    }

def find_optimal_threshold(df, output_dir):
    """Find optimal threshold to balance sensitivity and specificity"""
    logging.info("\n" + "="*60)
    logging.info("THRESHOLD OPTIMIZATION")
    logging.info("="*60)
    
    y_true = df['y_true'].values
    y_prob = df['y_prob_pos'].values
    
    thresholds = np.linspace(0.1, 0.9, 81)
    results = []
    
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        metrics = confusion_matrix_with_stats(y_true, y_pred)
        
        results.append({
            'threshold': threshold,
            'balanced_accuracy': metrics['balanced_accuracy'],
            'sensitivity': metrics['sensitivity'],
            'specificity': metrics['specificity'],
            'f1_score': metrics['f1_score'],
            'FP': metrics['FP'],
            'FN': metrics['FN']
        })
    
    results_df = pd.DataFrame(results)
    
    # Find key operating points
    best_balanced = results_df.loc[results_df['balanced_accuracy'].idxmax()]
    
    # Find best specificity while keeping sensitivity >= 0.95
    high_sens = results_df[results_df['sensitivity'] >= 0.95]
    if len(high_sens) > 0:
        best_spec_high_sens = high_sens.loc[high_sens['specificity'].idxmax()]
    else:
        best_spec_high_sens = None
    
    # Find equal sensitivity and specificity
    results_df['sens_spec_diff'] = np.abs(results_df['sensitivity'] - results_df['specificity'])
    equal_point = results_df.loc[results_df['sens_spec_diff'].idxmin()]
    
    # Log recommendations
    logging.info(f"\n1. Maximum Balanced Accuracy (threshold={best_balanced['threshold']:.3f}):")
    logging.info(f"   Balanced Acc: {best_balanced['balanced_accuracy']:.4f}")
    logging.info(f"   Sensitivity:  {best_balanced['sensitivity']:.4f}")
    logging.info(f"   Specificity:  {best_balanced['specificity']:.4f}")
    logging.info(f"   FP: {int(best_balanced['FP'])}, FN: {int(best_balanced['FN'])}")
    
    if best_spec_high_sens is not None:
        logging.info(f"\n2. Best Specificity with Sensitivity ≥ 0.95 (threshold={best_spec_high_sens['threshold']:.3f}):")
        logging.info(f"   Balanced Acc: {best_spec_high_sens['balanced_accuracy']:.4f}")
        logging.info(f"   Sensitivity:  {best_spec_high_sens['sensitivity']:.4f}")
        logging.info(f"   Specificity:  {best_spec_high_sens['specificity']:.4f}")
        logging.info(f"   FP: {int(best_spec_high_sens['FP'])}, FN: {int(best_spec_high_sens['FN'])}")
    
    logging.info(f"\n3. Equal Sensitivity/Specificity (threshold={equal_point['threshold']:.3f}):")
    logging.info(f"   Balanced Acc: {equal_point['balanced_accuracy']:.4f}")
    logging.info(f"   Sensitivity:  {equal_point['sensitivity']:.4f}")
    logging.info(f"   Specificity:  {equal_point['specificity']:.4f}")
    logging.info(f"   FP: {int(equal_point['FP'])}, FN: {int(equal_point['FN'])}")
    
    # Save threshold analysis
    threshold_file = os.path.join(output_dir, 'threshold_analysis.csv')
    results_df.to_csv(threshold_file, index=False)
    logging.info(f"\nFull threshold analysis saved to: {threshold_file}")
    
    return results_df

def main():
    parser = argparse.ArgumentParser(
        description='Ensemble predictions from cross-validation folds',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--predictions', nargs='+', required=True,
                      help='List of prediction CSV files')
    parser.add_argument('--output_dir', required=True,
                       help='Output directory for ensemble results')
    parser.add_argument('--methods', nargs='+',
                       default=['average', 'logit', 'median', 'confidence', 'max_confidence'],
                       help='Ensemble methods to try')
    parser.add_argument('--min_folds', type=int, default=1,
                       help='Minimum number of folds required per sample')
    parser.add_argument('--tune_threshold', action='store_true',
                       help='Find optimal threshold for best ensemble method')
    parser.add_argument('--conservative_voting', action='store_true',
                       help='Try conservative voting thresholds (0.6, 0.8)')
    
    args = parser.parse_args()
    
    # Handle 'all' methods option
    if 'all' in args.methods:
        args.methods = ['average', 'logit', 'median', 'confidence', 'max_confidence']
    
    # Setup logging
    logger = setup_logging(args.output_dir)
    
    logging.info("="*60)
    logging.info("CROSS-VALIDATION ENSEMBLE")
    logging.info("="*60)
    
    logging.info(f"Found {len(args.predictions)} prediction files:")
    for f in args.predictions:
        logging.info(f"  - {f}")
    
    # Load predictions
    predictions_by_sample, fold_info = load_fold_predictions(args.predictions)
    
    if not predictions_by_sample:
        logging.error("No valid predictions loaded!")
        return
    
    # Filter by minimum folds if specified
    if args.min_folds > 1:
        original_count = len(predictions_by_sample)
        predictions_by_sample = {
            k: v for k, v in predictions_by_sample.items() 
            if len(v['predictions']) >= args.min_folds
        }
        filtered_count = original_count - len(predictions_by_sample)
        if filtered_count > 0:
            logging.info(f"Filtered {filtered_count} samples with < {args.min_folds} folds")
    
    # Save fold info
    fold_info_df = pd.DataFrame(fold_info)
    fold_info_df.to_csv(os.path.join(args.output_dir, 'fold_info.csv'), index=False)
    
    # Run ensemble methods
    all_metrics = []
    
    method_functions = {
        'average': lambda: ensemble_average(predictions_by_sample),
        'logit': lambda: ensemble_logit_average(predictions_by_sample),
        'median': lambda: ensemble_median(predictions_by_sample),
        'confidence': lambda: ensemble_weighted_by_confidence(predictions_by_sample),
        'max_confidence': lambda: ensemble_max_confidence(predictions_by_sample)
    }
    
    # Add conservative voting if requested
    if args.conservative_voting:
        method_functions['majority_60'] = lambda: ensemble_majority_vote(predictions_by_sample, 0.6)
        method_functions['majority_80'] = lambda: ensemble_majority_vote(predictions_by_sample, 0.8)
        method_functions['unanimous'] = lambda: ensemble_majority_vote(predictions_by_sample, 1.0)
    
    best_ensemble_df = None
    best_method = None
    
    for method in args.methods + (['majority_60', 'majority_80', 'unanimous'] if args.conservative_voting else []):
        if method not in method_functions:
            logging.warning(f"Unknown method: {method}")
            continue
        
        try:
            # Compute ensemble
            result_df = method_functions[method]()
            
            # Evaluate
            metrics = evaluate_ensemble(result_df, method.capitalize())
            all_metrics.append(metrics)
            
            # Save predictions
            output_file = os.path.join(args.output_dir, f'ensemble_{method}.csv')
            result_df.to_csv(output_file, index=False)
            logging.info(f"Saved {method} predictions to {output_file}")
            
            # Track best for threshold tuning
            if best_ensemble_df is None or metrics['balanced_accuracy'] > all_metrics[0]['balanced_accuracy']:
                best_ensemble_df = result_df
                best_method = method
            
        except Exception as e:
            logging.error(f"Error with {method} ensemble: {e}")
            import traceback
            logging.error(traceback.format_exc())
    
    # Create comparison table
    if all_metrics:
        comparison_df = pd.DataFrame(all_metrics)
        comparison_df = comparison_df.sort_values('balanced_accuracy', ascending=False)
        
        comparison_file = os.path.join(args.output_dir, 'ensemble_comparison.csv')
        comparison_df.to_csv(comparison_file, index=False)
        
        logging.info(f"\n{'='*60}")
        logging.info("ENSEMBLE COMPARISON (sorted by balanced accuracy)")
        logging.info(f"{'='*60}")
        print("\n" + comparison_df.to_string(index=False))
        
        # Highlight best method
        best = comparison_df.iloc[0]
        logging.info(f"\n🏆 Best Ensemble Method: {best['method']}")
        logging.info(f"   Balanced Accuracy: {best['balanced_accuracy']:.4f}")
        logging.info(f"   Sensitivity: {best['sensitivity']:.4f}")
        logging.info(f"   Specificity: {best['specificity']:.4f}")
        logging.info(f"   F1 Score: {best['f1_score']:.4f}")
        logging.info(f"   AUC-ROC: {best['auc']:.4f}")
        
        # Threshold tuning if requested
        if args.tune_threshold and best_ensemble_df is not None:
            logging.info(f"\nPerforming threshold optimization on best method ({best_method})...")
            threshold_results = find_optimal_threshold(best_ensemble_df, args.output_dir)
        
        # Save summary
        summary = {
            'n_folds': len(args.predictions),
            'total_samples': len(predictions_by_sample),
            'best_method': best['method'],
            'best_balanced_acc': float(best['balanced_accuracy']),
            'best_auc': float(best['auc']),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(os.path.join(args.output_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
    
    logging.info(f"\n{'='*60}")
    logging.info("Ensemble complete! Results saved to:")
    logging.info(f"  {args.output_dir}")
    logging.info(f"{'='*60}")

if __name__ == '__main__':
    main()

#python ensemble_cross_validation.py     --predictions         /projects/retprogression/rgarridogarcia/gradable_dr_new_5k_clean_test_cropped_centered_wt_allR_1/predictions_best_balanced_acc_model.csv         /projects/retprogression/rgarridogarcia/gradable_dr_new_5k_clean_test_cropped_centered_wt_allR_2/predictions_best_balanced_acc_model.csv         /projects/retprogression/rgarridogarcia/gradable_dr_new_5k_clean_test_cropped_centered_wt_allR_3/predictions_best_balanced_acc_model.csv         /projects/retprogression/rgarridogarcia/gradable_dr_new_5k_clean_test_cropped_centered_wt_allR_4/predictions_best_balanced_acc_model.csv         /projects/retprogression/rgarridogarcia/gradable_dr_new_5k_clean_test_cropped_centered_wt_allR_5/predictions_best_balanced_acc_model.csv     --output_dir /projects/retprogression/rgarridogarcia/ensemble/ensemble_all_methods_Paper     --methods all     --conservative_voting     --tune_threshold
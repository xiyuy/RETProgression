"""
Compare ensemble performance vs individual model performance
Uses calibrated thresholds for each ensemble method
Includes validation metrics and comprehensive summary tables
"""
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import argparse
import glob
from custom_metrics import confusion_matrix_with_stats, roc_auc_score

def load_calibrated_thresholds(threshold_file):
    """Load calibrated thresholds from CSV"""
    df = pd.read_csv(threshold_file)
    
    # Create dictionary mapping method -> threshold
    thresholds = {}
    for _, row in df.iterrows():
        thresholds[row['method']] = float(row['threshold'])
    
    return thresholds

def load_validation_metrics_from_checkpoints(checkpoint_paths):
    """Load validation metrics from model checkpoint files"""
    import torch
    
    print("\n" + "="*80)
    print("VALIDATION SET PERFORMANCE (from training checkpoints)")
    print("="*80)
    
    val_metrics = []
    
    for i, ckpt_path in enumerate(checkpoint_paths, 1):
        if not os.path.exists(ckpt_path):
            print(f"⚠️  Checkpoint not found: {ckpt_path}")
            continue
        
        try:
            # Load checkpoint
            checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            
            # Extract partition number for naming
            dir_name = os.path.basename(os.path.dirname(ckpt_path))
            if 'allR_' in dir_name:
                partition = dir_name.split('allR_')[1].split('_')[0]
                model_name = f"Model {partition}"
            else:
                model_name = f"Model {i}"
            
            # Initialize metrics dict
            metrics_dict = {
                'model': model_name,
                'checkpoint': os.path.basename(ckpt_path),
                'epoch': checkpoint.get('epoch', 'unknown')
            }
            
            # Try different naming conventions for metrics
            for metric in ['accuracy', 'acc', 'balanced_accuracy', 'balanced_acc', 
                          'sensitivity', 'specificity', 'f1_score', 'f1', 'auc']:
                # Try various key combinations
                possible_keys = [
                    metric,
                    f'val_{metric}',
                    f'best_{metric}'
                ]
                
                for key in possible_keys:
                    if key in checkpoint:
                        # Standardize metric names with val_ prefix
                        if 'acc' in metric and 'balanced' not in metric:
                            standard_name = 'val_accuracy'
                        elif 'balanced' in metric:
                            standard_name = 'val_balanced_accuracy'
                        elif 'f1' in metric:
                            standard_name = 'val_f1_score'
                        else:
                            standard_name = f'val_{metric}'
                        
                        metrics_dict[standard_name] = checkpoint[key]
                        break
            
            # Check if we have a metrics sub-dictionary
            if 'metrics' in checkpoint and isinstance(checkpoint['metrics'], dict):
                for key, value in checkpoint['metrics'].items():
                    # Store with val_ prefix for consistency
                    if not key.startswith('val_'):
                        metrics_dict[f'val_{key}'] = value
                    else:
                        metrics_dict[key] = value
            
            # Try to extract confusion matrix values
            for cm_key in ['TP', 'FP', 'FN', 'TN']:
                found = False
                # Try various locations
                for prefix in ['val_', '']:
                    full_key = f'{prefix}{cm_key}'
                    if full_key in checkpoint:
                        metrics_dict[f'val_{cm_key}'] = checkpoint[full_key]
                        found = True
                        break
                
                # Try in metrics sub-dict
                if not found and 'metrics' in checkpoint and isinstance(checkpoint['metrics'], dict):
                    if cm_key in checkpoint['metrics']:
                        metrics_dict[f'val_{cm_key}'] = checkpoint['metrics'][cm_key]
                    elif f'val_{cm_key}' in checkpoint['metrics']:
                        metrics_dict[f'val_{cm_key}'] = checkpoint['metrics'][f'val_{cm_key}']
            
            val_metrics.append(metrics_dict)
            
        except Exception as e:
            print(f"❌ Error loading checkpoint {ckpt_path}: {e}")
    
    if val_metrics:
        val_df = pd.DataFrame(val_metrics)
        
        # Display as a nice table with confusion matrix
        print(f"\n{'Model':<10} {'Epoch':>7} {'Bal Acc':>10} {'Sens':>10} {'Spec':>10} {'F1':>10} {'AUC':>10} {'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5}")
        print("-"*125)
        
        for _, row in val_df.iterrows():
            epoch = row.get('epoch', 'N/A')
            bal_acc = row.get('val_balanced_accuracy', row.get('val_balanced_acc', np.nan))
            sens = row.get('val_sensitivity', np.nan)
            spec = row.get('val_specificity', np.nan)
            f1 = row.get('val_f1_score', row.get('val_f1', np.nan))
            auc = row.get('val_auc', np.nan)
            tp = row.get('val_TP', np.nan)
            fp = row.get('val_FP', np.nan)
            fn = row.get('val_FN', np.nan)
            tn = row.get('val_TN', np.nan)
            
            # Format confusion matrix values
            tp_str = f"{int(tp)}" if not np.isnan(tp) else "-"
            fp_str = f"{int(fp)}" if not np.isnan(fp) else "-"
            fn_str = f"{int(fn)}" if not np.isnan(fn) else "-"
            tn_str = f"{int(tn)}" if not np.isnan(tn) else "-"
            
            print(f"{row['model']:<10} {str(epoch):>7} {bal_acc:>10.4f} {sens:>10.4f} {spec:>10.4f} "
                  f"{f1:>10.4f} {auc:>10.4f} {tp_str:>5} {fp_str:>5} {fn_str:>5} {tn_str:>5}")
        
        # Note if confusion matrix is missing
        if 'val_TP' not in val_df.columns or val_df['val_TP'].isna().all():
            print("\n⚠️  Note: Confusion matrix (TP/FP/FN/TN) not saved in checkpoints")
            print("   This is normal - only metrics were saved during training")
        
        return val_df
    else:
        return pd.DataFrame()

def evaluate_predictions(df, model_name, threshold=0.5):
    """Evaluate predictions at a given threshold"""
    # Detect column names
    true_col = next((c for c in ['y_true', 'true_label', 'label'] if c in df.columns), None)
    prob_col = next((c for c in ['y_prob_pos', 'prob_class_1', 'probability_class_1'] if c in df.columns), None)
    
    if true_col is None or prob_col is None:
        raise ValueError(f"Missing required columns in {model_name}. Available: {list(df.columns)}")
    
    y_true = df[true_col].values
    y_prob = df[prob_col].values
    
    # Apply threshold
    y_pred = (y_prob >= threshold).astype(int)
    
    # Calculate metrics
    metrics = confusion_matrix_with_stats(y_true, y_pred)
    
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception as e:
        auc = np.nan
    
    return {
        'model': model_name,
        'threshold': threshold,
        'n_samples': len(df),
        'accuracy': metrics['accuracy'],
        'balanced_accuracy': metrics['balanced_accuracy'],
        'sensitivity': metrics['sensitivity'],
        'specificity': metrics['specificity'],
        'precision': metrics['precision'],
        'f1_score': metrics['f1_score'],
        'auc': auc,
        'TP': metrics['TP'],
        'FP': metrics['FP'],
        'FN': metrics['FN'],
        'TN': metrics['TN']
    }

def load_individual_predictions(prediction_files, threshold=0.5):
    """Load and evaluate individual model predictions"""
    results = []
    
    print("\n" + "="*80)
    print(f"INDIVIDUAL MODEL PERFORMANCE (threshold={threshold:.2f})")
    print("="*80)
    
    for i, pred_file in enumerate(prediction_files, 1):
        if not os.path.exists(pred_file):
            print(f"⚠️  File not found: {pred_file}")
            continue
        
        try:
            df = pd.read_csv(pred_file)
            
            # Create short model name
            dir_name = os.path.basename(os.path.dirname(pred_file))
            if 'allR_' in dir_name:
                partition = dir_name.split('allR_')[1].split('_')[0]
                model_name = f"Model {partition}"
            else:
                model_name = f"Model {i}"
            
            metrics = evaluate_predictions(df, model_name, threshold)
            results.append(metrics)
            
            print(f"\n{model_name}:")
            print(f"  Balanced Acc: {metrics['balanced_accuracy']:.4f}")
            print(f"  Sensitivity:  {metrics['sensitivity']:.4f}")
            print(f"  Specificity:  {metrics['specificity']:.4f}")
            print(f"  AUC:          {metrics['auc']:.4f}")
            print(f"  Confusion:    TP={metrics['TP']}, FP={metrics['FP']}, FN={metrics['FN']}, TN={metrics['TN']}")
            
        except Exception as e:
            print(f"❌ Error loading {pred_file}: {e}")
    
    return pd.DataFrame(results)

def load_ensemble_predictions(ensemble_dir, calibrated_thresholds=None):
    """Load and evaluate ensemble predictions with calibrated thresholds"""
    ensemble_files = glob.glob(os.path.join(ensemble_dir, "ensemble_*.csv"))
    ensemble_files = [f for f in ensemble_files if 'comparison' not in f and 'analysis' not in f]
    
    if not ensemble_files:
        print(f"⚠️  No ensemble files found in {ensemble_dir}")
        return pd.DataFrame()
    
    results = []
    
    print("\n" + "="*80)
    print("ENSEMBLE PERFORMANCE (with calibrated thresholds)")
    print("="*80)
    
    for ensemble_file in sorted(ensemble_files):
        method_name = os.path.basename(ensemble_file).replace('ensemble_', '').replace('.csv', '')
        
        # Get calibrated threshold if available
        if calibrated_thresholds and method_name in calibrated_thresholds:
            threshold = calibrated_thresholds[method_name]
            print(f"\n{method_name}: Using calibrated threshold {threshold:.2f}")
        else:
            threshold = 0.5
            print(f"\n{method_name}: Using default threshold {threshold:.2f}")
        
        try:
            df = pd.read_csv(ensemble_file)
            display_name = f"Ensemble ({method_name})"
            
            metrics = evaluate_predictions(df, display_name, threshold)
            results.append(metrics)
            
            print(f"  Balanced Acc: {metrics['balanced_accuracy']:.4f}")
            print(f"  Sensitivity:  {metrics['sensitivity']:.4f}")
            print(f"  Specificity:  {metrics['specificity']:.4f}")
            print(f"  AUC:          {metrics['auc']:.4f}")
            print(f"  Confusion:    TP={metrics['TP']}, FP={metrics['FP']}, FN={metrics['FN']}, TN={metrics['TN']}")
            
        except Exception as e:
            print(f"❌ Error loading {ensemble_file}: {e}")
    
    return pd.DataFrame(results)

def compare_all(individual_df, ensemble_df):
    """Create comprehensive comparison"""
    
    # Combine dataframes
    all_results = pd.concat([individual_df, ensemble_df], ignore_index=True)
    
    # Sort by balanced accuracy
    all_results = all_results.sort_values('balanced_accuracy', ascending=False)
    
    print("\n" + "="*80)
    print(f"COMPLETE COMPARISON (Individual: 0.50, Ensemble: calibrated)")
    print("="*80)
    print("\nSorted by Balanced Accuracy:")
    print("-"*125)
    
    # Display table with confusion matrix
    print(f"{'Model':<40} {'Thresh':>8} {'Bal Acc':>10} {'Sens':>10} {'Spec':>10} {'F1':>10} {'AUC':>10} {'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5}")
    print("-"*125)
    
    for _, row in all_results.iterrows():
        model_display = row['model'][:38] if len(row['model']) > 38 else row['model']
        print(f"{model_display:<40} {row['threshold']:>8.2f} {row['balanced_accuracy']:>10.4f} {row['sensitivity']:>10.4f} "
              f"{row['specificity']:>10.4f} {row['f1_score']:>10.4f} {row['auc']:>10.4f} "
              f"{int(row['TP']):>5} {int(row['FP']):>5} {int(row['FN']):>5} {int(row['TN']):>5}")
    
    # Highlight best
    best = all_results.iloc[0]
    print("\n" + "="*80)
    print(f"🏆 BEST PERFORMER: {best['model']} (threshold={best['threshold']:.2f})")
    print("="*80)
    print(f"  Balanced Accuracy: {best['balanced_accuracy']:.4f}")
    print(f"  Sensitivity:       {best['sensitivity']:.4f}")
    print(f"  Specificity:       {best['specificity']:.4f}")
    print(f"  F1 Score:          {best['f1_score']:.4f}")
    print(f"  AUC:               {best['auc']:.4f}")
    print(f"  Confusion Matrix:  TP={int(best['TP'])}, FP={int(best['FP'])}, FN={int(best['FN'])}, TN={int(best['TN'])}")
    
    # Check if ensemble is best
    if not individual_df.empty and not ensemble_df.empty:
        best_individual = individual_df.loc[individual_df['balanced_accuracy'].idxmax()]
        best_ensemble = ensemble_df.loc[ensemble_df['balanced_accuracy'].idxmax()]
        
        print("\n" + "="*80)
        print("ENSEMBLE vs BEST INDIVIDUAL MODEL")
        print("="*80)
        
        print(f"\nBest Individual Model: {best_individual['model']} (threshold={best_individual['threshold']:.2f})")
        print(f"  Balanced Accuracy: {best_individual['balanced_accuracy']:.4f}")
        print(f"  Sensitivity:       {best_individual['sensitivity']:.4f}")
        print(f"  Specificity:       {best_individual['specificity']:.4f}")
        
        print(f"\nBest Ensemble Method: {best_ensemble['model']} (threshold={best_ensemble['threshold']:.2f})")
        print(f"  Balanced Accuracy: {best_ensemble['balanced_accuracy']:.4f}")
        print(f"  Sensitivity:       {best_ensemble['sensitivity']:.4f}")
        print(f"  Specificity:       {best_ensemble['specificity']:.4f}")
        
        improvement = (best_ensemble['balanced_accuracy'] - best_individual['balanced_accuracy']) * 100
        if improvement > 0:
            print(f"\n✅ Ensemble improves by: +{improvement:.2f}% balanced accuracy")
        elif improvement < 0:
            print(f"\n⚠️  Best individual model outperforms ensemble by: {abs(improvement):.2f}%")
        else:
            print(f"\n➡️  Ensemble and best individual model perform equally")
        
        # Show metrics comparison
        print(f"\nDetailed Comparison:")
        print(f"  {'Metric':<20} {'Individual':>12} {'Ensemble':>12} {'Difference':>12}")
        print(f"  {'-'*60}")
        for metric in ['balanced_accuracy', 'sensitivity', 'specificity', 'f1_score', 'auc']:
            ind_val = best_individual[metric]
            ens_val = best_ensemble[metric]
            diff = (ens_val - ind_val) * 100
            sign = '+' if diff >= 0 else ''
            print(f"  {metric:<20} {ind_val:>12.4f} {ens_val:>12.4f} {sign}{diff:>11.2f}%")
    
    return all_results

def create_comprehensive_summary_table(individual_df, ensemble_df, val_metrics_df, ensemble_dir, calibrated_thresholds):
    """
    Create comprehensive summary with all three conditions:
    Models:    [Validation | Test@0.5 | NaN]
    Ensembles: [NaN | Test@0.5 | Test@Optimal]
    """
    print("\n" + "="*80)
    print("COMPREHENSIVE SUMMARY TABLE")
    print("="*80)
    print("Format per metric: Value1 / Value2 / Value3")
    print("  Individual Models: Validation / Test@0.5 / NaN")
    print("  Ensembles:         NaN / Test@0.5 / Test@Optimal_Threshold")
    print("="*80)
    
    summary_rows = []
    
    # Process individual models
    for _, test_row in individual_df.iterrows():
        model_name = test_row['model']
        
        # Find validation metrics
        val_row = None
        if not val_metrics_df.empty:
            val_match = val_metrics_df[val_metrics_df['model'] == model_name]
            if len(val_match) > 0:
                val_row = val_match.iloc[0]
        
        # Format: Validation / Test@0.5 / NaN
        row_data = {'model': model_name, 'type': 'Individual'}
        
        for metric in ['balanced_accuracy', 'sensitivity', 'specificity', 'f1_score', 'auc', 'TP', 'FP', 'FN', 'TN']:
            val_key = f'val_{metric}'
            test_val = test_row.get(metric, np.nan)
            
            if val_row is not None:
                val_val = val_row.get(val_key, np.nan)
            else:
                val_val = np.nan
            
            row_data[metric] = f"{val_val:.4f} / {test_val:.4f} / NaN" if metric not in ['TP', 'FP', 'FN', 'TN'] else f"{int(val_val) if not np.isnan(val_val) else '-'} / {int(test_val)} / -"
        
        summary_rows.append(row_data)
    
    # Process ensembles - need both default (0.5) and optimal threshold
    ensemble_files = glob.glob(os.path.join(ensemble_dir, "ensemble_*.csv"))
    ensemble_files = [f for f in ensemble_files if 'comparison' not in f and 'analysis' not in f]
    
    for ensemble_file in sorted(ensemble_files):
        method_name = os.path.basename(ensemble_file).replace('ensemble_', '').replace('.csv', '')
        
        try:
            df = pd.read_csv(ensemble_file)
            
            # Get default threshold (0.5) metrics
            default_metrics = evaluate_predictions(df, f"Ensemble ({method_name})", 0.5)
            
            # Get optimal threshold metrics
            if calibrated_thresholds and method_name in calibrated_thresholds:
                optimal_threshold = calibrated_thresholds[method_name]
                optimal_metrics = evaluate_predictions(df, f"Ensemble ({method_name})", optimal_threshold)
            else:
                optimal_threshold = 0.5
                optimal_metrics = default_metrics
            
            # Format: NaN / Test@0.5 / Test@Optimal
            row_data = {'model': f"Ensemble ({method_name})", 'type': 'Ensemble'}
            
            for metric in ['balanced_accuracy', 'sensitivity', 'specificity', 'f1_score', 'auc', 'TP', 'FP', 'FN', 'TN']:
                default_val = default_metrics.get(metric, np.nan)
                optimal_val = optimal_metrics.get(metric, np.nan)
                
                if metric not in ['TP', 'FP', 'FN', 'TN']:
                    row_data[metric] = f"NaN / {default_val:.4f} / {optimal_val:.4f}"
                else:
                    row_data[metric] = f"- / {int(default_val)} / {int(optimal_val)}"
            
            summary_rows.append(row_data)
            
        except Exception as e:
            print(f"⚠️  Error processing ensemble {method_name}: {e}")
    
    # Create DataFrame
    summary_df = pd.DataFrame(summary_rows)
    
    # Display table
    print(f"\n{'Model':<30} {'Type':>12} {'Bal Acc (Val/Test@0.5/Optimal)':>40}")
    print("-"*125)
    
    for _, row in summary_df.iterrows():
        print(f"{row['model']:<30} {row['type']:>12} {row['balanced_accuracy']:>40}")
    
    print(f"\n{'Model':<30} {'Sensitivity (Val/Test@0.5/Optimal)':>50}")
    print("-"*125)
    for _, row in summary_df.iterrows():
        print(f"{row['model']:<30} {row['sensitivity']:>50}")
    
    print(f"\n{'Model':<30} {'Specificity (Val/Test@0.5/Optimal)':>50}")
    print("-"*125)
    for _, row in summary_df.iterrows():
        print(f"{row['model']:<30} {row['specificity']:>50}")
    
    print(f"\n{'Model':<30} {'F1 Score (Val/Test@0.5/Optimal)':>50}")
    print("-"*125)
    for _, row in summary_df.iterrows():
        print(f"{row['model']:<30} {row['f1_score']:>50}")
    
    print(f"\n{'Model':<30} {'TP (Val/Test@0.5/Optimal)':>40}")
    print("-"*125)
    for _, row in summary_df.iterrows():
        print(f"{row['model']:<30} {row['TP']:>40}")
    
    print(f"\n{'Model':<30} {'FP (Val/Test@0.5/Optimal)':>40}")
    print("-"*125)
    for _, row in summary_df.iterrows():
        print(f"{row['model']:<30} {row['FP']:>40}")
    
    print(f"\n{'Model':<30} {'FN (Val/Test@0.5/Optimal)':>40}")
    print("-"*125)
    for _, row in summary_df.iterrows():
        print(f"{row['model']:<30} {row['FN']:>40}")
    
    print(f"\n{'Model':<30} {'TN (Val/Test@0.5/Optimal)':>40}")
    print("-"*125)
    for _, row in summary_df.iterrows():
        print(f"{row['model']:<30} {row['TN']:>40}")
    
    # Save to CSV
    summary_file = os.path.join(ensemble_dir, 'comprehensive_summary_table.csv')
    summary_df.to_csv(summary_file, index=False)
    print(f"\n✅ Comprehensive summary saved to: {summary_file}")
    
    return summary_df

def main():
    parser = argparse.ArgumentParser(
        description='Compare ensemble vs individual model performance with calibrated thresholds',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--predictions', nargs='+', required=True,
                       help='Individual model prediction files')
    parser.add_argument('--ensemble_dir', required=True,
                       help='Directory containing ensemble results')
    parser.add_argument('--calibration_file',
                       help='Path to best_thresholds_summary.csv (optional, uses calibrated thresholds)')
    parser.add_argument('--checkpoints', nargs='+',
                       help='Model checkpoint files (.pth) to extract validation metrics')
    parser.add_argument('--output', help='Output CSV file for comparison table')
    parser.add_argument('--individual_threshold', type=float, default=0.5,
                       help='Decision threshold for individual models')
    
    args = parser.parse_args()
    
    print("="*80)
    print("ENSEMBLE vs INDIVIDUAL MODEL COMPARISON")
    print("="*80)
    print(f"Individual model threshold: {args.individual_threshold:.2f}")
    print(f"Ensemble thresholds: {'Calibrated' if args.calibration_file else 'Default (0.5)'}")
    print(f"Individual models: {len(args.predictions)}")
    print(f"Ensemble directory: {args.ensemble_dir}")
    
    # Load validation metrics from checkpoints if provided
    val_metrics_df = pd.DataFrame()
    if args.checkpoints:
        val_metrics_df = load_validation_metrics_from_checkpoints(args.checkpoints)
    
    # Load calibrated thresholds if provided
    calibrated_thresholds = None
    if args.calibration_file:
        if os.path.exists(args.calibration_file):
            print(f"\nLoading calibrated thresholds from: {args.calibration_file}")
            calibrated_thresholds = load_calibrated_thresholds(args.calibration_file)
            print(f"Loaded {len(calibrated_thresholds)} calibrated thresholds:")
            for method, thresh in calibrated_thresholds.items():
                print(f"  {method}: {thresh:.2f}")
        else:
            print(f"⚠️  Calibration file not found: {args.calibration_file}")
            print(f"  Using default threshold 0.5 for all ensembles")
    
    # Load individual model predictions (TEST SET)
    individual_df = load_individual_predictions(args.predictions, args.individual_threshold)
    
    if individual_df.empty:
        print("\n❌ No individual model predictions loaded!")
        return
    
    # Load ensemble predictions with calibrated thresholds
    ensemble_df = load_ensemble_predictions(args.ensemble_dir, calibrated_thresholds)
    
    if ensemble_df.empty:
        print("\n❌ No ensemble predictions loaded!")
        return
    
    # Compare all (TEST SET)
    all_results = compare_all(individual_df, ensemble_df)
    
    # Create comprehensive summary table
    comprehensive_summary = create_comprehensive_summary_table(
        individual_df, ensemble_df, val_metrics_df, args.ensemble_dir, calibrated_thresholds
    )
    
    # Validation vs Test comparison for individual models
    if not val_metrics_df.empty:
        print("\n" + "="*80)
        print("VALIDATION vs TEST SET COMPARISON (Individual Models)")
        print("="*80)
        
        print(f"\n{'Model':<15} {'Val Bal Acc':>12} {'Test Bal Acc':>13} {'Difference':>12} {'Status':>10}")
        print("-"*80)
        
        for _, test_row in individual_df.iterrows():
            val_row = val_metrics_df[val_metrics_df['model'] == test_row['model']]
            
            if len(val_row) > 0:
                val_row = val_row.iloc[0]
                val_bal_acc = val_row.get('val_balanced_accuracy', val_row.get('val_balanced_acc', np.nan))
                test_bal_acc = test_row['balanced_accuracy']
                
                if not np.isnan(val_bal_acc):
                    diff = (test_bal_acc - val_bal_acc) * 100
                    sign = '+' if diff >= 0 else ''
                    
                    # Check for overfitting/underfitting
                    if diff < -2:
                        status = "⚠️ Overfit"
                    elif diff > 2:
                        status = "✓ Better"
                    else:
                        status = "✓ Stable"
                    
                    print(f"{test_row['model']:<15} {val_bal_acc:>12.4f} {test_bal_acc:>13.4f} {sign}{diff:>11.2f}% {status:>10}")
        
        # Save validation comparison
        val_test_comparison = os.path.join(args.ensemble_dir, 'validation_vs_test_comparison.csv')
        
        comparison_rows = []
        for _, test_row in individual_df.iterrows():
            val_row = val_metrics_df[val_metrics_df['model'] == test_row['model']]
            if len(val_row) > 0:
                val_row = val_row.iloc[0]
                comparison_rows.append({
                    'model': test_row['model'],
                    'val_balanced_acc': val_row.get('val_balanced_accuracy', val_row.get('val_balanced_acc', np.nan)),
                    'val_sensitivity': val_row.get('val_sensitivity', np.nan),
                    'val_specificity': val_row.get('val_specificity', np.nan),
                    'test_balanced_acc': test_row['balanced_accuracy'],
                    'test_sensitivity': test_row['sensitivity'],
                    'test_specificity': test_row['specificity'],
                    'balanced_acc_diff': test_row['balanced_accuracy'] - val_row.get('val_balanced_accuracy', val_row.get('val_balanced_acc', 0))
                })
        
        if comparison_rows:
            pd.DataFrame(comparison_rows).to_csv(val_test_comparison, index=False)
            print(f"\n✅ Validation vs Test comparison saved to: {val_test_comparison}")
    
    # Save main results
    if args.output:
        all_results.to_csv(args.output, index=False)
        print(f"\n✅ Comparison table saved to: {args.output}")
    
    output_file = os.path.join(args.ensemble_dir, 'ensemble_vs_individual_comparison.csv')
    all_results.to_csv(output_file, index=False)
    print(f"✅ Comparison table saved to: {output_file}")
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)
    
    print("\nGenerated Files:")
    print(f"  1. ensemble_vs_individual_comparison.csv - Performance comparison")
    print(f"  2. comprehensive_summary_table.csv - Val/Test@0.5/Optimal format")
    if not val_metrics_df.empty:
        print(f"  3. validation_vs_test_comparison.csv - Validation vs Test")

if __name__ == '__main__':
    main()


# python bestT_compare_ensemble_vs_individual.py \
#     --predictions /projects/retprogression/rgarridogarcia/gradable_dr_new_5k_clean_test_cropped_centered_wt_allR_*/predictions*.csv \
#     --ensemble_dir /projects/retprogression/rgarridogarcia/ensemble/ensemble_results_20251120_113255 \
#     --calibration_file /projects/retprogression/rgarridogarcia/ensemble/ensemble_results_20251120_113255/best_thresholds_summary.csv \
#     --checkpoints /projects/retprogression/rgarridogarcia/checkpoints/cropped1024_cropped_centered_wt_allR_*/best_balanced_acc_model.pth
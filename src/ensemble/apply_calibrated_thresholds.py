"""
Apply calibrated thresholds to NEW dataset ensemble predictions
Uses thresholds from best_thresholds_summary.csv to evaluate performance on new data
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import argparse
import glob
import json
from custom_metrics import confusion_matrix_with_stats, roc_auc_score

def load_calibrated_thresholds(threshold_file):
    """Load calibrated thresholds from CSV"""
    df = pd.read_csv(threshold_file)
    
    # Create dictionary mapping method -> threshold
    thresholds = {}
    for _, row in df.iterrows():
        thresholds[row['method']] = {
            'threshold': row['threshold'],
            'calibration_balanced_acc': row['balanced_accuracy'],
            'calibration_sensitivity': row['sensitivity'],
            'calibration_specificity': row['specificity']
        }
    
    return thresholds

def evaluate_with_threshold(df, method_name, threshold):
    """Evaluate predictions with a specific threshold"""
    y_true = df['y_true'].values
    y_prob = df['y_prob_pos'].values
    
    # Apply threshold
    y_pred = (y_prob >= threshold).astype(int)
    
    # Calculate metrics
    metrics = confusion_matrix_with_stats(y_true, y_pred)
    
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception as e:
        auc = np.nan
    
    return {
        'method': method_name,
        'threshold': threshold,
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

def apply_thresholds_to_new_data(new_ensemble_dir, calibrated_thresholds, output_dir):
    """Apply calibrated thresholds to new ensemble predictions"""
    
    # Find all ensemble files in new directory
    ensemble_files = glob.glob(os.path.join(new_ensemble_dir, "ensemble_*.csv"))
    ensemble_files = [f for f in ensemble_files if 'comparison' not in f and 'analysis' not in f]
    
    if not ensemble_files:
        print(f"❌ No ensemble files found in {new_ensemble_dir}")
        return None
    
    print("="*80)
    print("APPLYING CALIBRATED THRESHOLDS TO VALIDATION DATASET")
    print("="*80)
    print(f"Validation ensemble directory: {new_ensemble_dir}")
    print(f"Found {len(ensemble_files)} ensemble methods")
    print("="*80)
    
    results = []
    
    for ensemble_file in sorted(ensemble_files):
        method_name = os.path.basename(ensemble_file).replace('ensemble_', '').replace('.csv', '')
        
        # Check if we have a calibrated threshold for this method
        if method_name not in calibrated_thresholds:
            print(f"\n⚠️  No calibrated threshold for {method_name}, using default 0.5")
            threshold = 0.5
            calibration_info = None
        else:
            threshold = calibrated_thresholds[method_name]['threshold']
            calibration_info = calibrated_thresholds[method_name]
            print(f"\n{method_name.upper()}: Using calibrated threshold {threshold:.2f}")
        
        try:
            df = pd.read_csv(ensemble_file)
            
            # Check for required columns
            if 'y_true' not in df.columns or 'y_prob_pos' not in df.columns:
                print(f"  ⚠️  Skipping {method_name} - missing required columns")
                continue
            
            # Evaluate with calibrated threshold
            metrics = evaluate_with_threshold(df, method_name, threshold)
            results.append(metrics)
            
            # Display results
            print(f"  Validation Set Performance:")
            print(f"    Accuracy:     {metrics['accuracy']:.4f}")
            print(f"    Bal Acc:      {metrics['balanced_accuracy']:.4f}")
            print(f"    Sensitivity:  {metrics['sensitivity']:.4f}")
            print(f"    Specificity:  {metrics['specificity']:.4f}")
            print(f"    F1 Score:     {metrics['f1_score']:.4f}")
            print(f"    AUC:          {metrics['auc']:.4f}")
            print(f"    Confusion:    TP={metrics['TP']}, FP={metrics['FP']}, FN={metrics['FN']}, TN={metrics['TN']}")
            
            # Compare to calibration set if available
            if calibration_info:
                bal_diff = (metrics['balanced_accuracy'] - calibration_info['calibration_balanced_acc']) * 100
                sens_diff = (metrics['sensitivity'] - calibration_info['calibration_sensitivity']) * 100
                spec_diff = (metrics['specificity'] - calibration_info['calibration_specificity']) * 100
                
                print(f"  Calibration Set Performance:")
                print(f"    Bal Acc:      {calibration_info['calibration_balanced_acc']:.4f} (diff: {bal_diff:+.2f}%)")
                print(f"    Sensitivity:  {calibration_info['calibration_sensitivity']:.4f} (diff: {sens_diff:+.2f}%)")
                print(f"    Specificity:  {calibration_info['calibration_specificity']:.4f} (diff: {spec_diff:+.2f}%)")
            
            # Save predictions with applied threshold
            df_with_threshold = df.copy()
            df_with_threshold['y_pred'] = (df_with_threshold['y_prob_pos'] >= threshold).astype(int)
            df_with_threshold['threshold_used'] = threshold
            
            output_file = os.path.join(output_dir, f'validation_{method_name}_thresh{threshold:.2f}.csv')
            df_with_threshold.to_csv(output_file, index=False)
            
        except Exception as e:
            print(f"❌ Error processing {method_name}: {e}")
            import traceback
            traceback.print_exc()
    
    return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser(
        description='Apply calibrated thresholds to validation dataset ensemble predictions',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--calibration_file', required=True,
                       help='Path to best_thresholds_summary.csv from calibration')
    parser.add_argument('--new_ensemble_dir', required=True,
                       help='Directory containing ensemble predictions on VALIDATION dataset')
    parser.add_argument('--output_dir', required=True,
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load calibrated thresholds
    print("Loading calibrated thresholds...")
    calibrated_thresholds = load_calibrated_thresholds(args.calibration_file)
    
    print(f"Loaded {len(calibrated_thresholds)} calibrated thresholds:")
    for method, info in calibrated_thresholds.items():
        print(f"  {method}: threshold={info['threshold']:.2f}")
    
    # Apply to validation data
    results_df = apply_thresholds_to_new_data(
        args.new_ensemble_dir,
        calibrated_thresholds,
        args.output_dir
    )
    
    if results_df is None or results_df.empty:
        print("\n❌ No results generated!")
        return
    
    # Sort by performance on validation dataset
    results_df = results_df.sort_values('balanced_accuracy', ascending=False)
    
    # Display final comparison
    print("\n" + "="*80)
    print("FINAL COMPARISON ON VALIDATION DATASET (with calibrated thresholds)")
    print("="*80)
    print(f"\n{'Method':<20} {'Thresh':>8} {'Bal Acc':>10} {'Sens':>10} {'Spec':>10} {'F1':>10} {'AUC':>10} {'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5}")
    print("-"*125)
    
    for _, row in results_df.iterrows():
        print(f"{row['method']:<20} {row['threshold']:>8.2f} {row['balanced_accuracy']:>10.4f} "
              f"{row['sensitivity']:>10.4f} {row['specificity']:>10.4f} "
              f"{row['f1_score']:>10.4f} {row['auc']:>10.4f} "
              f"{int(row['TP']):>5} {int(row['FP']):>5} {int(row['FN']):>5} {int(row['TN']):>5}")
    
    # Highlight best on validation dataset
    best = results_df.iloc[0]
    print("\n" + "="*80)
    print(f"🏆 BEST ON VALIDATION DATASET: {best['method']} (threshold={best['threshold']:.2f})")
    print("="*80)
    print(f"  Balanced Accuracy: {best['balanced_accuracy']:.4f}")
    print(f"  Sensitivity:       {best['sensitivity']:.4f}")
    print(f"  Specificity:       {best['specificity']:.4f}")
    print(f"  F1 Score:          {best['f1_score']:.4f}")
    print(f"  AUC:               {best['auc']:.4f}")
    print(f"  Confusion Matrix:  TP={int(best['TP'])}, FP={int(best['FP'])}, FN={int(best['FN'])}, TN={int(best['TN'])}")
    
    # Save comparison
    comparison_file = os.path.join(args.output_dir, 'validation_performance_with_calibrated_thresholds.csv')
    results_df.to_csv(comparison_file, index=False)
    print(f"\n✅ Saved comparison to: {comparison_file}")
    
    # Save summary
    summary = {
        'best_method': best['method'],
        'best_threshold': float(best['threshold']),
        'balanced_accuracy': float(best['balanced_accuracy']),
        'sensitivity': float(best['sensitivity']),
        'specificity': float(best['specificity']),
        'f1_score': float(best['f1_score']),
        'auc': float(best['auc'])
    }
    
    summary_file = os.path.join(args.output_dir, 'validation_best_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Saved summary to: {summary_file}")
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)
    print(f"\nResults saved to: {args.output_dir}")

if __name__ == '__main__':
    main()
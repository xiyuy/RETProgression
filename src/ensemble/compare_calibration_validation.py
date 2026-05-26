"""
Compare individual models vs ensembles (calibrated and non-calibrated) 
on both calibration and validation sets
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import argparse
import glob
from custom_metrics import confusion_matrix_with_stats, roc_auc_score

def load_calibrated_thresholds(threshold_file):
    """Load calibrated thresholds from CSV"""
    df = pd.read_csv(threshold_file)
    thresholds = {}
    for _, row in df.iterrows():
        thresholds[row['method']] = float(row['threshold'])
    return thresholds

def evaluate_predictions(df, model_name, threshold=0.5):
    """Evaluate predictions at a given threshold"""
    # Detect column names
    true_col = next((c for c in ['y_true', 'true_label', 'label'] if c in df.columns), None)
    prob_col = next((c for c in ['y_prob_pos', 'prob_class_1', 'probability_class_1'] if c in df.columns), None)
    
    if true_col is None or prob_col is None:
        raise ValueError(f"Missing required columns. Available: {list(df.columns)}")
    
    y_true = df[true_col].values
    y_prob = df[prob_col].values
    y_pred = (y_prob >= threshold).astype(int)
    
    metrics = confusion_matrix_with_stats(y_true, y_pred)
    
    try:
        auc = roc_auc_score(y_true, y_prob)
    except:
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

def evaluate_dataset(pred_files, ensemble_dir, calibrated_thresholds, dataset_name, individual_threshold=0.5):
    """Evaluate individual models and ensembles on a dataset"""
    
    print("\n" + "="*80)
    print(f"{dataset_name.upper()} SET EVALUATION")
    print("="*80)
    
    results = []
    
    # Evaluate individual models
    print(f"\nIndividual Models (threshold={individual_threshold:.2f}):")
    for i, pred_file in enumerate(pred_files, 1):
        if not os.path.exists(pred_file):
            continue
        
        df = pd.read_csv(pred_file)
        model_name = f"Model {i}"
        
        metrics = evaluate_predictions(df, model_name, individual_threshold)
        metrics['type'] = 'Individual'
        metrics['dataset'] = dataset_name
        results.append(metrics)
        
        print(f"  {model_name}: Bal Acc={metrics['balanced_accuracy']:.4f}, "
              f"Sens={metrics['sensitivity']:.4f}, Spec={metrics['specificity']:.4f}")
    
    # Evaluate ensembles at default threshold (0.5)
    print(f"\nEnsembles at Default Threshold (0.50):")
    ensemble_files = glob.glob(os.path.join(ensemble_dir, "ensemble_*.csv"))
    ensemble_files = [f for f in ensemble_files if 'comparison' not in f and 'analysis' not in f]
    
    for ensemble_file in sorted(ensemble_files):
        method_name = os.path.basename(ensemble_file).replace('ensemble_', '').replace('.csv', '')
        
        try:
            df = pd.read_csv(ensemble_file)
            display_name = f"Ensemble ({method_name})"
            
            metrics = evaluate_predictions(df, display_name, 0.5)
            metrics['type'] = 'Ensemble_Default'
            metrics['dataset'] = dataset_name
            results.append(metrics)
            
            print(f"  {method_name}: Bal Acc={metrics['balanced_accuracy']:.4f}, "
                  f"Sens={metrics['sensitivity']:.4f}, Spec={metrics['specificity']:.4f}")
        except Exception as e:
            print(f"  ⚠️ Error with {method_name}: {e}")
    
    # Evaluate ensembles with calibrated thresholds
    if calibrated_thresholds:
        print(f"\nEnsembles with Calibrated Thresholds:")
        
        for ensemble_file in sorted(ensemble_files):
            method_name = os.path.basename(ensemble_file).replace('ensemble_', '').replace('.csv', '')
            
            if method_name not in calibrated_thresholds:
                continue
            
            try:
                df = pd.read_csv(ensemble_file)
                threshold = calibrated_thresholds[method_name]
                display_name = f"Ensemble ({method_name})"
                
                metrics = evaluate_predictions(df, display_name, threshold)
                metrics['type'] = 'Ensemble_Calibrated'
                metrics['dataset'] = dataset_name
                results.append(metrics)
                
                print(f"  {method_name} @ {threshold:.2f}: Bal Acc={metrics['balanced_accuracy']:.4f}, "
                      f"Sens={metrics['sensitivity']:.4f}, Spec={metrics['specificity']:.4f}")
            except Exception as e:
                print(f"  ⚠️ Error with {method_name}: {e}")
    
    return pd.DataFrame(results)

def analyze_error_overlap(pred_files, ensemble_dir, calibrated_thresholds, dataset_name, output_dir, individual_threshold=0.5):
    """
    Analyze which specific samples are FP/FN across models
    Shows if models make the same mistakes
    """
    print("\n" + "="*80)
    print(f"{dataset_name.upper()} SET - ERROR OVERLAP ANALYSIS")
    print("="*80)
    
    # Load all predictions and track errors by sample
    all_predictions = {}
    
    # Individual models
    for i, pred_file in enumerate(pred_files, 1):
        if not os.path.exists(pred_file):
            continue
        
        df = pd.read_csv(pred_file)
        
        # Detect columns
        id_col = next((c for c in ['image_id', 'id', 'ID', 'image_filename'] if c in df.columns), None)
        true_col = next((c for c in ['y_true', 'true_label'] if c in df.columns), None)
        prob_col = next((c for c in ['y_prob_pos', 'prob_class_1'] if c in df.columns), None)
        
        if not all([id_col, true_col, prob_col]):
            continue
        
        for _, row in df.iterrows():
            sample_id = str(row[id_col])
            y_true = int(row[true_col])
            y_pred = int(row[prob_col] >= individual_threshold)
            
            if sample_id not in all_predictions:
                all_predictions[sample_id] = {
                    'true_label': y_true,
                    'models': {},
                    'ensembles_default': {},
                    'ensembles_calibrated': {}
                }
            
            all_predictions[sample_id]['models'][f'Model_{i}'] = y_pred
    
    # Ensemble predictions - default threshold
    ensemble_files = glob.glob(os.path.join(ensemble_dir, "ensemble_*.csv"))
    ensemble_files = [f for f in ensemble_files if 'comparison' not in f and 'analysis' not in f]
    
    for ensemble_file in sorted(ensemble_files):
        method = os.path.basename(ensemble_file).replace('ensemble_', '').replace('.csv', '')
        
        try:
            df = pd.read_csv(ensemble_file)
            id_col = next((c for c in ['id', 'image_id', 'ID'] if c in df.columns), None)
            prob_col = next((c for c in ['y_prob_pos', 'prob_class_1'] if c in df.columns), None)
            
            if not all([id_col, prob_col]):
                continue
            
            for _, row in df.iterrows():
                sample_id = str(row[id_col])
                y_pred_default = int(row[prob_col] >= 0.5)
                
                if sample_id in all_predictions:
                    all_predictions[sample_id]['ensembles_default'][method] = y_pred_default
                    
                    # Calibrated threshold
                    if calibrated_thresholds and method in calibrated_thresholds:
                        threshold = calibrated_thresholds[method]
                        y_pred_cal = int(row[prob_col] >= threshold)
                        all_predictions[sample_id]['ensembles_calibrated'][method] = y_pred_cal
        except:
            continue
    
    # Analyze negative samples (TN vs FP)
    negative_samples = {sid: data for sid, data in all_predictions.items() if data['true_label'] == 0}
    
    if len(negative_samples) > 0:
        print(f"\nNegative Samples (Non-Gradable): {len(negative_samples)} total")
        print(f"{'='*80}")
        
        # Count how many models correctly identify each negative sample
        fp_analysis = []
        
        for sample_id, data in negative_samples.items():
            # Count correct predictions (y_pred = 0 for negatives)
            models_correct = sum(1 for pred in data['models'].values() if pred == 0)
            total_models = len(data['models'])
            
            ensembles_default_correct = sum(1 for pred in data['ensembles_default'].values() if pred == 0)
            total_ensembles_default = len(data['ensembles_default'])
            
            ensembles_cal_correct = sum(1 for pred in data['ensembles_calibrated'].values() if pred == 0)
            total_ensembles_cal = len(data['ensembles_calibrated'])
            
            # Determine if this is TN or FP
            is_fp_models = models_correct < total_models  # At least one model wrong
            is_fp_ensembles = ensembles_default_correct < total_ensembles_default
            
            fp_analysis.append({
                'sample_id': sample_id,
                'models_correct': models_correct,
                'total_models': total_models,
                'ensembles_default_correct': ensembles_default_correct,
                'ensembles_calibrated_correct': ensembles_cal_correct,
                'all_models_FP': models_correct == 0,
                'all_models_TN': models_correct == total_models,
                'all_ensembles_default_FP': ensembles_default_correct == 0,
                'all_ensembles_calibrated_FP': ensembles_cal_correct == 0
            })
        
        fp_df = pd.DataFrame(fp_analysis)
        
        # Summary statistics
        print(f"\nNegative Sample Analysis:")
        print(f"  All models correct (TN):          {fp_df['all_models_TN'].sum()} samples")
        print(f"  All models wrong (FP):            {fp_df['all_models_FP'].sum()} samples")
        print(f"  Models disagree:                  {len(fp_df) - fp_df['all_models_TN'].sum() - fp_df['all_models_FP'].sum()} samples")
        
        print(f"\n  All ensembles (default) FP:       {fp_df['all_ensembles_default_FP'].sum()} samples")
        print(f"  All ensembles (calibrated) FP:    {fp_df['all_ensembles_calibrated_FP'].sum()} samples")
        
        # Show the problematic false positives
        all_models_fp = fp_df[fp_df['all_models_FP']]
        if len(all_models_fp) > 0:
            print(f"\n❌ Samples where ALL models predict FP ({len(all_models_fp)} total):")
            print(f"{'Sample ID':<30} {'Models Correct':>15} {'Ens Default Correct':>20} {'Ens Calibrated Correct':>23}")
            print("-"*100)
            for _, row in all_models_fp.head(20).iterrows():
                print(f"{str(row['sample_id'])[:28]:<30} {row['models_correct']:>15} "
                      f"{row['ensembles_default_correct']:>20} {row['ensembles_calibrated_correct']:>23}")
        
        # Show samples where ensembles help
        ensembles_better = fp_df[
            (fp_df['models_correct'] < fp_df['total_models']) & 
            (fp_df['ensembles_calibrated_correct'] > fp_df['models_correct'])
        ]
        
        if len(ensembles_better) > 0:
            print(f"\n✅ Samples where calibrated ensembles improve over individual models ({len(ensembles_better)} total):")
            print(f"{'Sample ID':<30} {'Models Correct':>15} {'Ens Calibrated Correct':>23}")
            print("-"*80)
            for _, row in ensembles_better.head(10).iterrows():
                print(f"{str(row['sample_id'])[:28]:<30} {row['models_correct']:>15} {row['ensembles_calibrated_correct']:>23}")
        
        # Save detailed analysis
        fp_csv = os.path.join(output_dir, f'{dataset_name.lower()}_negative_samples_analysis.csv')
        fp_df.to_csv(fp_csv, index=False)
        print(f"\n✅ Detailed negative sample analysis saved to: {fp_csv}")
        
        return fp_df
    else:
        print(f"\n⚠️ No negative samples found in {dataset_name} set")
        return None

def create_comparison_tables(cal_results, val_results, output_dir):
    """Create comprehensive comparison tables"""
    
    # Combine datasets
    all_results = pd.concat([cal_results, val_results], ignore_index=True)
    
    # Save complete results
    all_results.to_csv(os.path.join(output_dir, 'complete_results.csv'), index=False)
    
    # Create calibration table
    print("\n" + "="*80)
    print("CALIBRATION SET COMPARISON")
    print("="*80)
    print(f"\n{'Model':<35} {'Type':>20} {'Thresh':>8} {'Bal Acc':>10} {'Sens':>10} {'Spec':>10} {'F1':>10} {'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5}")
    print("-"*140)
    
    cal_sorted = cal_results.sort_values('balanced_accuracy', ascending=False)
    for _, row in cal_sorted.iterrows():
        model = row['model'][:33]
        type_str = row['type'].replace('_', ' ')
        print(f"{model:<35} {type_str:>20} {row['threshold']:>8.2f} {row['balanced_accuracy']:>10.4f} "
              f"{row['sensitivity']:>10.4f} {row['specificity']:>10.4f} {row['f1_score']:>10.4f} "
              f"{int(row['TP']):>5} {int(row['FP']):>5} {int(row['FN']):>5} {int(row['TN']):>5}")
    
    # Create validation table
    print("\n" + "="*80)
    print("VALIDATION SET COMPARISON")
    print("="*80)
    print(f"\n{'Model':<35} {'Type':>20} {'Thresh':>8} {'Bal Acc':>10} {'Sens':>10} {'Spec':>10} {'F1':>10} {'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5}")
    print("-"*140)
    
    val_sorted = val_results.sort_values('balanced_accuracy', ascending=False)
    for _, row in val_sorted.iterrows():
        model = row['model'][:33]
        type_str = row['type'].replace('_', ' ')
        print(f"{model:<35} {type_str:>20} {row['threshold']:>8.2f} {row['balanced_accuracy']:>10.4f} "
              f"{row['sensitivity']:>10.4f} {row['specificity']:>10.4f} {row['f1_score']:>10.4f} "
              f"{int(row['TP']):>5} {int(row['FP']):>5} {int(row['FN']):>5} {int(row['TN']):>5}")
    
    # Save individual tables
    cal_results.to_csv(os.path.join(output_dir, 'calibration_results.csv'), index=False)
    val_results.to_csv(os.path.join(output_dir, 'validation_results.csv'), index=False)
    
    # Highlight best in each category
    print("\n" + "="*80)
    print("BEST PERFORMERS")
    print("="*80)
    
    # Best on calibration
    best_cal = cal_sorted.iloc[0]
    print(f"\nCalibration Set:")
    print(f"  🏆 {best_cal['model']} ({best_cal['type']}, thresh={best_cal['threshold']:.2f})")
    print(f"     Bal Acc: {best_cal['balanced_accuracy']:.4f}, Sens: {best_cal['sensitivity']:.4f}, Spec: {best_cal['specificity']:.4f}")
    
    # Best on validation
    best_val = val_sorted.iloc[0]
    print(f"\nValidation Set:")
    print(f"  🏆 {best_val['model']} ({best_val['type']}, thresh={best_val['threshold']:.2f})")
    print(f"     Bal Acc: {best_val['balanced_accuracy']:.4f}, Sens: {best_val['sensitivity']:.4f}, Spec: {best_val['specificity']:.4f}")
    
    # Compare calibration vs validation for best ensemble
    best_ensemble_cal = cal_results[cal_results['type'] == 'Ensemble_Calibrated'].sort_values('balanced_accuracy', ascending=False).iloc[0]
    
    # Find same method in validation
    best_method = best_ensemble_cal['model']
    best_ensemble_val = val_results[val_results['model'] == best_method]
    
    if len(best_ensemble_val) > 0:
        best_ensemble_val = best_ensemble_val[val_results['type'] == 'Ensemble_Calibrated'].iloc[0]
        
        print(f"\n" + "="*80)
        print(f"GENERALIZATION: {best_method}")
        print("="*80)
        print(f"  Calibration: Bal Acc={best_ensemble_cal['balanced_accuracy']:.4f}, "
              f"Sens={best_ensemble_cal['sensitivity']:.4f}, Spec={best_ensemble_cal['specificity']:.4f}")
        print(f"  Validation:  Bal Acc={best_ensemble_val['balanced_accuracy']:.4f}, "
              f"Sens={best_ensemble_val['sensitivity']:.4f}, Spec={best_ensemble_val['specificity']:.4f}")
        
        bal_diff = (best_ensemble_val['balanced_accuracy'] - best_ensemble_cal['balanced_accuracy']) * 100
        if abs(bal_diff) < 2:
            status = "✅ Good generalization"
        elif bal_diff < 0:
            status = "⚠️ Performance dropped (possible overfitting)"
        else:
            status = "✓ Performance improved"
        
        print(f"  Difference: {bal_diff:+.2f}% - {status}")

def main():
    parser = argparse.ArgumentParser(
        description='Compare models and ensembles on calibration and validation sets',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--cal_predictions', nargs='+', required=True,
                       help='Individual model predictions on CALIBRATION set')
    parser.add_argument('--val_predictions', nargs='+', required=True,
                       help='Individual model predictions on VALIDATION set')
    parser.add_argument('--cal_ensemble_dir', required=True,
                       help='Directory with calibration set ensembles')
    parser.add_argument('--val_ensemble_dir', required=True,
                       help='Directory with validation set ensembles')
    parser.add_argument('--calibration_file', required=True,
                       help='Path to best_thresholds_summary.csv')
    parser.add_argument('--output_dir', required=True,
                       help='Output directory for comparison tables')
    parser.add_argument('--individual_threshold', type=float, default=0.5,
                       help='Threshold for individual models')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*80)
    print("CALIBRATION vs VALIDATION SET COMPARISON")
    print("="*80)
    print(f"Individual model threshold: {args.individual_threshold:.2f}")
    print(f"Ensemble thresholds: Calibrated from {os.path.basename(args.calibration_file)}")
    
    # Load calibrated thresholds
    calibrated_thresholds = load_calibrated_thresholds(args.calibration_file)
    print(f"\nLoaded {len(calibrated_thresholds)} calibrated thresholds")
    
    # Evaluate calibration set
    cal_results = evaluate_dataset(
        args.cal_predictions,
        args.cal_ensemble_dir,
        calibrated_thresholds,
        'Calibration',
        args.individual_threshold
    )
    
    # Analyze error overlap on calibration set
    cal_error_analysis = analyze_error_overlap(
        args.cal_predictions,
        args.cal_ensemble_dir,
        calibrated_thresholds,
        'Calibration',
        args.output_dir,
        args.individual_threshold
    )
    
    # Evaluate validation set
    val_results = evaluate_dataset(
        args.val_predictions,
        args.val_ensemble_dir,
        calibrated_thresholds,
        'Validation',
        args.individual_threshold
    )
    
    # Analyze error overlap on validation set
    val_error_analysis = analyze_error_overlap(
        args.val_predictions,
        args.val_ensemble_dir,
        calibrated_thresholds,
        'Validation',
        args.output_dir,
        args.individual_threshold
    )
    
    # Create comparison tables
    create_comparison_tables(cal_results, val_results, args.output_dir)
    
    print(f"\n✅ Results saved to: {args.output_dir}")
    print(f"  - calibration_results.csv")
    print(f"  - validation_results.csv")
    print(f"  - complete_results.csv")

if __name__ == '__main__':
    main()
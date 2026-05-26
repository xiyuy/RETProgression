"""
Plot ROC curves comparing best individual model, best default ensemble, 
and best calibrated ensemble on both calibration and validation sets
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import glob
from custom_metrics import roc_curve, roc_auc_score

def load_predictions(pred_file):
    """Load predictions and extract y_true and y_prob"""
    df = pd.read_csv(pred_file)
    
    # Detect columns
    true_col = next((c for c in ['y_true', 'true_label', 'label'] if c in df.columns), None)
    prob_col = next((c for c in ['y_prob_pos', 'prob_class_1'] if c in df.columns), None)
    
    if not all([true_col, prob_col]):
        raise ValueError(f"Missing required columns. Available: {list(df.columns)}")
    
    return df[true_col].values, df[prob_col].values

def find_best_files(pred_files, ensemble_dir, metric='balanced_accuracy'):
    """
    Find best individual model and best ensemble files
    
    Returns:
        best_individual_file, best_ensemble_default_file, ensemble method name
    """
    from custom_metrics import confusion_matrix_with_stats
    
    # Evaluate individual models
    best_individual = None
    best_individual_score = -1
    
    for pred_file in pred_files:
        if not os.path.exists(pred_file):
            continue
        
        y_true, y_prob = load_predictions(pred_file)
        y_pred = (y_prob >= 0.5).astype(int)
        
        metrics = confusion_matrix_with_stats(y_true, y_pred)
        score = metrics[metric]
        
        if score > best_individual_score:
            best_individual_score = score
            best_individual = pred_file
    
    # Evaluate ensembles at default threshold
    ensemble_files = glob.glob(os.path.join(ensemble_dir, "ensemble_*.csv"))
    ensemble_files = [f for f in ensemble_files if 'comparison' not in f and 'analysis' not in f]
    
    best_ensemble = None
    best_ensemble_score = -1
    best_ensemble_method = None
    
    for ensemble_file in ensemble_files:
        method = os.path.basename(ensemble_file).replace('ensemble_', '').replace('.csv', '')
        
        try:
            y_true, y_prob = load_predictions(ensemble_file)
            y_pred = (y_prob >= 0.5).astype(int)
            
            metrics = confusion_matrix_with_stats(y_true, y_pred)
            score = metrics[metric]
            
            if score > best_ensemble_score:
                best_ensemble_score = score
                best_ensemble = ensemble_file
                best_ensemble_method = method
        except:
            continue
    
    return best_individual, best_ensemble, best_ensemble_method

def plot_roc_comparison(cal_pred_files, val_pred_files, cal_ensemble_dir, val_ensemble_dir, 
                       calibrated_thresholds, output_dir):
    """Create ROC curve comparison plot"""
    
    print("\n" + "="*80)
    print("GENERATING ROC CURVES")
    print("="*80)
    
    # Find best files for calibration set
    print("\nFinding best performers on calibration set...")
    best_ind_cal, best_ens_cal, best_method = find_best_files(cal_pred_files, cal_ensemble_dir)
    print(f"  Best individual: {os.path.basename(best_ind_cal)}")
    print(f"  Best ensemble: {best_method}")
    
    # Find best files for validation set (use same method for consistency)
    print("\nUsing same models for validation set...")
    best_ind_val = best_ind_cal.replace('_calibration.csv', '_validation.csv')
    best_ens_val = os.path.join(val_ensemble_dir, f'ensemble_{best_method}.csv')
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # === CALIBRATION SET ===
    ax1.set_title('ROC Curves - Calibration Set', fontsize=14, fontweight='bold')
    
    # Individual model
    y_true, y_prob = load_predictions(best_ind_cal)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    ax1.plot(fpr, tpr, linewidth=2.5, label=f'Best Individual (AUC={auc:.3f})', 
            color='#1f77b4', linestyle='-')
    
    # Ensemble - default threshold
    y_true, y_prob = load_predictions(best_ens_cal)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    ax1.plot(fpr, tpr, linewidth=2.5, label=f'Ensemble {best_method} - Default (AUC={auc:.3f})', 
            color='#ff7f0e', linestyle='--')
    
    # Ensemble - calibrated threshold (same probabilities, just different operating point)
    # Plot full ROC but mark the calibrated threshold point
    if calibrated_thresholds and best_method in calibrated_thresholds:
        optimal_threshold = calibrated_thresholds[best_method]
        y_pred_optimal = (y_prob >= optimal_threshold).astype(int)
        
        # Calculate TPR and FPR at optimal threshold
        tp = np.sum((y_pred_optimal == 1) & (y_true == 1))
        fp = np.sum((y_pred_optimal == 1) & (y_true == 0))
        fn = np.sum((y_pred_optimal == 0) & (y_true == 1))
        tn = np.sum((y_pred_optimal == 0) & (y_true == 0))
        
        tpr_optimal = tp / (tp + fn)
        fpr_optimal = fp / (fp + tn)
        
        ax1.plot(fpr, tpr, linewidth=2.5, label=f'Ensemble {best_method} - ROC (AUC={auc:.3f})', 
                color='#2ca02c', linestyle='-', alpha=0.7)
        ax1.scatter([fpr_optimal], [tpr_optimal], s=150, c='red', marker='*', 
                   label=f'Calibrated Threshold ({optimal_threshold:.2f})', zorder=5, edgecolors='black', linewidths=1.5)
    
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC=0.500)', alpha=0.3)
    ax1.set_xlabel('False Positive Rate', fontsize=12)
    ax1.set_ylabel('True Positive Rate', fontsize=12)
    ax1.set_xlim([-0.02, 1.02])
    ax1.set_ylim([-0.02, 1.02])
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # === VALIDATION SET ===
    ax2.set_title('ROC Curves - Validation Set', fontsize=14, fontweight='bold')
    
    # Individual model
    if os.path.exists(best_ind_val):
        y_true, y_prob = load_predictions(best_ind_val)
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        ax2.plot(fpr, tpr, linewidth=2.5, label=f'Best Individual (AUC={auc:.3f})', 
                color='#1f77b4', linestyle='-')
    
    # Ensemble - default threshold
    if os.path.exists(best_ens_val):
        y_true, y_prob = load_predictions(best_ens_val)
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        ax2.plot(fpr, tpr, linewidth=2.5, label=f'Ensemble {best_method} - Default (AUC={auc:.3f})', 
                color='#ff7f0e', linestyle='--')
        
        # Ensemble - calibrated threshold
        if calibrated_thresholds and best_method in calibrated_thresholds:
            optimal_threshold = calibrated_thresholds[best_method]
            y_pred_optimal = (y_prob >= optimal_threshold).astype(int)
            
            tp = np.sum((y_pred_optimal == 1) & (y_true == 1))
            fp = np.sum((y_pred_optimal == 1) & (y_true == 0))
            fn = np.sum((y_pred_optimal == 0) & (y_true == 1))
            tn = np.sum((y_pred_optimal == 0) & (y_true == 0))
            
            tpr_optimal = tp / (tp + fn)
            fpr_optimal = fp / (fp + tn)
            
            ax2.plot(fpr, tpr, linewidth=2.5, label=f'Ensemble {best_method} - ROC (AUC={auc:.3f})', 
                    color='#2ca02c', linestyle='-', alpha=0.7)
            ax2.scatter([fpr_optimal], [tpr_optimal], s=150, c='red', marker='*', 
                       label=f'Calibrated Threshold ({optimal_threshold:.2f})', zorder=5, edgecolors='black', linewidths=1.5)
    
    ax2.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC=0.500)', alpha=0.3)
    ax2.set_xlabel('False Positive Rate', fontsize=12)
    ax2.set_ylabel('True Positive Rate', fontsize=12)
    ax2.set_xlim([-0.02, 1.02])
    ax2.set_ylim([-0.02, 1.02])
    ax2.legend(loc='lower right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_file = os.path.join(output_dir, 'roc_curves_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✅ ROC curves saved to: {output_file}")
    
    plt.show()
    
    return fig

def main():
    parser = argparse.ArgumentParser(description='Plot ROC curves for calibration and validation sets')
    
    parser.add_argument('--cal_predictions', nargs='+', required=True,
                       help='Individual model predictions on calibration set')
    parser.add_argument('--val_predictions', nargs='+', required=True,
                       help='Individual model predictions on validation set')
    parser.add_argument('--cal_ensemble_dir', required=True,
                       help='Directory with calibration ensembles')
    parser.add_argument('--val_ensemble_dir', required=True,
                       help='Directory with validation ensembles')
    parser.add_argument('--calibration_file', required=True,
                       help='Path to best_thresholds_summary.csv')
    parser.add_argument('--output_dir', required=True,
                       help='Output directory')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*80)
    print("ROC CURVE COMPARISON")
    print("="*80)
    
    # Load calibrated thresholds
    threshold_df = pd.read_csv(args.calibration_file)
    calibrated_thresholds = {}
    for _, row in threshold_df.iterrows():
        calibrated_thresholds[row['method']] = float(row['threshold'])
    
    print(f"Loaded {len(calibrated_thresholds)} calibrated thresholds")
    
    # Create ROC plot
    plot_roc_comparison(
        args.cal_predictions,
        args.val_predictions,
        args.cal_ensemble_dir,
        args.val_ensemble_dir,
        calibrated_thresholds,
        args.output_dir
    )
    
    print("\n" + "="*80)
    print("ROC curve generation complete!")
    print("="*80)

if __name__ == '__main__':
    main()
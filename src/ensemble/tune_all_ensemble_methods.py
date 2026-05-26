"""
Tune thresholds for ALL ensemble methods
Run from: src/ensemble/ directory
"""
import sys
import os

# Add parent directory (src/) to Python path to import custom_metrics
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import glob
import argparse
from custom_metrics import confusion_matrix_with_stats

def find_optimal_threshold(df, method_name):
    """Find optimal threshold for a single method"""
    y_true = df['y_true'].values
    y_prob = df['y_prob_pos'].values
    
    thresholds = np.linspace(0.1, 0.9, 81)
    results = []
    
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        metrics = confusion_matrix_with_stats(y_true, y_pred)
        
        results.append({
            'method': method_name,
            'threshold': threshold,
            'balanced_accuracy': metrics['balanced_accuracy'],
            'sensitivity': metrics['sensitivity'],
            'specificity': metrics['specificity'],
            'f1_score': metrics['f1_score'],
            'TP': metrics['TP'],
            'FP': metrics['FP'],
            'FN': metrics['FN'],
            'TN': metrics['TN']
        })
    
    results_df = pd.DataFrame(results)
    
    # Find best threshold for this method - return as dict with all columns
    best_idx = results_df['balanced_accuracy'].idxmax()
    best = results_df.loc[best_idx].to_dict()
    
    return results_df, best

def main():
    parser = argparse.ArgumentParser(description='Tune thresholds for all ensemble methods')
    parser.add_argument('--ensemble_dir', required=True, 
                       help='Directory containing ensemble results (e.g., /path/to/ensemble_results_20251120_143022)')
    parser.add_argument('--target_specificity', type=float, 
                       help='Optional: Find threshold for target specificity (e.g., 0.85)')
    
    args = parser.parse_args()
    
    # Find all ensemble CSV files
    ensemble_files = glob.glob(os.path.join(args.ensemble_dir, "ensemble_*.csv"))
    
    # Exclude comparison and analysis files
    ensemble_files = [f for f in ensemble_files if 'comparison' not in f and 'analysis' not in f]
    
    if not ensemble_files:
        print(f"❌ No ensemble files found in {args.ensemble_dir}")
        print(f"Looking for: {args.ensemble_dir}/ensemble_*.csv")
        return
    
    print("="*80)
    print("THRESHOLD OPTIMIZATION FOR ALL ENSEMBLE METHODS")
    print("="*80)
    print(f"Analyzing {len(ensemble_files)} ensemble methods from:")
    print(f"  {args.ensemble_dir}")
    print("="*80)
    
    all_results = []
    best_thresholds = []
    
    for ensemble_file in sorted(ensemble_files):
        method_name = os.path.basename(ensemble_file).replace('ensemble_', '').replace('.csv', '')
        
        try:
            df = pd.read_csv(ensemble_file)
            
            # Check for required columns
            if 'y_true' not in df.columns or 'y_prob_pos' not in df.columns:
                print(f"⚠️  Skipping {method_name} - missing required columns")
                continue
            
            # Tune threshold
            results_df, best = find_optimal_threshold(df, method_name)
            all_results.append(results_df)
            best_thresholds.append(best)
            
            # Get default (0.5) performance
            default_row = results_df[results_df['threshold'] == 0.5].iloc[0]
            
            print(f"\n{method_name.upper()}:")
            print(f"  Default (0.50): Bal Acc = {default_row['balanced_accuracy']:.4f}, "
                  f"Sens = {default_row['sensitivity']:.4f}, "
                  f"Spec = {default_row['specificity']:.4f}, "
                  f"FP = {int(default_row['FP'])}, FN = {int(default_row['FN'])}")
            print(f"  Optimal ({best['threshold']:.2f}): Bal Acc = {best['balanced_accuracy']:.4f}, "
                  f"Sens = {best['sensitivity']:.4f}, "
                  f"Spec = {best['specificity']:.4f}, "
                  f"FP = {int(best['FP'])}, FN = {int(best['FN'])}")
            
            improvement = (best['balanced_accuracy'] - default_row['balanced_accuracy']) * 100
            if improvement > 0:
                print(f"  Improvement: +{improvement:.2f}% balanced accuracy")
            else:
                print(f"  No improvement (optimal = default)")
            
            # Check if target specificity is achievable
            if args.target_specificity:
                target_rows = results_df[results_df['specificity'] >= args.target_specificity]
                if len(target_rows) > 0:
                    # Pick the one with best balanced accuracy
                    target_best = target_rows.loc[target_rows['balanced_accuracy'].idxmax()]
                    print(f"  Target Spec ≥{args.target_specificity:.2f} ({target_best['threshold']:.2f}): "
                          f"Bal Acc = {target_best['balanced_accuracy']:.4f}, "
                          f"Sens = {target_best['sensitivity']:.4f}, "
                          f"Spec = {target_best['specificity']:.4f}")
                else:
                    max_spec = results_df['specificity'].max()
                    print(f"  ⚠️  Cannot reach {args.target_specificity:.2f} specificity (max = {max_spec:.4f})")
            
        except Exception as e:
            print(f"❌ Error processing {method_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary comparison
    if best_thresholds:
        print("\n" + "="*80)
        print("COMPARISON: BEST THRESHOLD FOR EACH METHOD")
        print("="*80)
        
        comparison_df = pd.DataFrame(best_thresholds)
        comparison_df = comparison_df.sort_values('balanced_accuracy', ascending=False)
        
        print(f"\n{'Method':<20} {'Thresh':>8} {'Bal Acc':>10} {'Sens':>10} {'Spec':>10} {'F1':>10} {'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5}")
        print("-"*115)
        
        for _, row in comparison_df.iterrows():
            print(f"{row['method']:<20} {row['threshold']:>8.2f} {row['balanced_accuracy']:>10.4f} "
                  f"{row['sensitivity']:>10.4f} {row['specificity']:>10.4f} {row['f1_score']:>10.4f} "
                  f"{int(row['TP']):>5} {int(row['FP']):>5} {int(row['FN']):>5} {int(row['TN']):>5}")
        
        # Highlight best
        best = comparison_df.iloc[0]
        print(f"\n🏆 Best: {best['method']} at threshold {best['threshold']:.2f}")
        print(f"   Balanced Acc: {best['balanced_accuracy']:.4f}, Sens: {best['sensitivity']:.4f}, Spec: {best['specificity']:.4f}, F1: {best['f1_score']:.4f}")
        print(f"   Confusion: TP={int(best['TP'])}, FP={int(best['FP'])}, FN={int(best['FN'])}, TN={int(best['TN'])}")
        
        # Save results
        # Save all thresholds
        all_thresholds_df = pd.concat(all_results, ignore_index=True)
        all_thresholds_file = os.path.join(args.ensemble_dir, 'all_methods_threshold_analysis.csv')
        all_thresholds_df.to_csv(all_thresholds_file, index=False)
        print(f"\n✅ Saved complete threshold analysis to:")
        print(f"   {all_thresholds_file}")
        
        # Save best thresholds summary
        summary_file = os.path.join(args.ensemble_dir, 'best_thresholds_summary.csv')
        comparison_df.to_csv(summary_file, index=False)
        print(f"✅ Saved best thresholds summary to:")
        print(f"   {summary_file}")
    
    else:
        print("\n⚠️  No valid ensemble methods found for threshold optimization")
    
    print("\n" + "="*80)

if __name__ == '__main__':
    main()


# python tune_all_ensemble_methods.py     --ensemble_dir /projects/retprogression/rgarridogarcia/ensemble/ensemble_results_20251120_113255
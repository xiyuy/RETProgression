"""
Compare ensemble performance vs individual model performance
Shows whether ensemble outperforms the best single model
Includes detailed error analysis to see where each model makes mistakes
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

def evaluate_predictions(df, model_name, threshold=0.5):
    """
    Evaluate predictions at a given threshold
    
    Args:
        df: DataFrame with prediction columns
        model_name: Name to display
        threshold: Decision threshold (default 0.5)
    
    Returns:
        Dictionary of metrics
    """
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

def create_error_analysis(prediction_files, ensemble_dir, threshold=0.5):
    """
    Create detailed error analysis showing where each model makes mistakes
    
    Returns:
        DataFrame with one row per sample, columns for each model's prediction
    """
    print("\n" + "="*80)
    print("DETAILED ERROR ANALYSIS - Per Sample")
    print("="*80)
    
    # Load all predictions and merge by sample ID
    all_data = []
    
    for i, pred_file in enumerate(prediction_files, 1):
        if not os.path.exists(pred_file):
            continue
        
        df = pd.read_csv(pred_file)
        
        # Detect column names
        id_col = next((c for c in ['image_id', 'id', 'ID', 'image_filename'] if c in df.columns), None)
        true_col = next((c for c in ['y_true', 'true_label', 'label'] if c in df.columns), None)
        prob_col = next((c for c in ['y_prob_pos', 'prob_class_1', 'probability_class_1'] if c in df.columns), None)
        
        if not all([id_col, true_col, prob_col]):
            print(f"⚠️  Skipping {pred_file} - missing columns")
            continue
        
        # Shorten IDs for readability (remove extensions, take last part)
        df['sample_id'] = df[id_col].astype(str).apply(lambda x: os.path.splitext(str(x))[0])
        
        # Select and rename columns
        model_df = df[[id_col, 'sample_id', true_col, prob_col]].copy()
        model_df.columns = ['original_id', 'sample_id', 'true_label', f'model_{i}_prob']
        
        # Add prediction at threshold
        model_df[f'model_{i}_pred'] = (model_df[f'model_{i}_prob'] >= threshold).astype(int)
        model_df[f'model_{i}_correct'] = (model_df[f'model_{i}_pred'] == model_df['true_label']).astype(int)
        
        all_data.append(model_df)
    
    # Merge all models by sample_id
    if not all_data:
        print("❌ No valid prediction files to analyze")
        return None
    
    merged = all_data[0][['sample_id', 'true_label']].copy()
    
    for i, model_df in enumerate(all_data, 1):
        # Merge on sample_id, keeping only the prediction columns
        cols_to_merge = ['sample_id', f'model_{i}_prob', f'model_{i}_pred', f'model_{i}_correct']
        merged = merged.merge(
            model_df[cols_to_merge],
            on='sample_id',
            how='outer'
        )
    
    # Add ensemble predictions if available
    ensemble_files = glob.glob(os.path.join(ensemble_dir, "ensemble_*.csv"))
    ensemble_files = [f for f in ensemble_files if 'average' in f or 'logit' in f]
    
    for ensemble_file in sorted(ensemble_files)[:2]:  # Just add average and logit
        method_name = os.path.basename(ensemble_file).replace('ensemble_', '').replace('.csv', '')
        
        try:
            ens_df = pd.read_csv(ensemble_file)
            
            # Detect columns
            id_col = next((c for c in ['image_id', 'id', 'ID'] if c in ens_df.columns), None)
            prob_col = next((c for c in ['y_prob_pos', 'prob_class_1'] if c in ens_df.columns), None)
            
            if id_col and prob_col:
                ens_df['sample_id'] = ens_df[id_col].astype(str).apply(lambda x: os.path.splitext(str(x))[0])
                ens_df[f'ensemble_{method_name}_prob'] = ens_df[prob_col]
                ens_df[f'ensemble_{method_name}_pred'] = (ens_df[prob_col] >= threshold).astype(int)
                
                merged = merged.merge(
                    ens_df[['sample_id', f'ensemble_{method_name}_prob', f'ensemble_{method_name}_pred']],
                    on='sample_id',
                    how='left'
                )
        except Exception as e:
            print(f"⚠️  Could not add ensemble {method_name}: {e}")
    
    # Add error flags for ensemble
    if 'ensemble_average_pred' in merged.columns:
        merged['ensemble_average_correct'] = (merged['ensemble_average_pred'] == merged['true_label']).astype(int)
    
    # Count how many models got each sample correct
    correct_cols = [col for col in merged.columns if col.endswith('_correct')]
    merged['n_models_correct'] = merged[correct_cols].sum(axis=1)
    
    # Calculate agreement (std of probabilities)
    prob_cols = [col for col in merged.columns if 'model_' in col and col.endswith('_prob')]
    if prob_cols:
        merged['model_prob_std'] = merged[prob_cols].std(axis=1)
        merged['model_prob_mean'] = merged[prob_cols].mean(axis=1)
    
    # Identify error types
    merged['all_models_wrong'] = (merged['n_models_correct'] == 0).astype(int)
    merged['all_models_correct'] = (merged['n_models_correct'] == len(prob_cols)).astype(int)
    merged['models_disagree'] = ((merged['n_models_correct'] > 0) & 
                                  (merged['n_models_correct'] < len(prob_cols))).astype(int)
    
    # Sort by cases where ensemble helps most
    if 'ensemble_average_correct' in merged.columns:
        merged['ensemble_fixed_error'] = ((merged['n_models_correct'] < len(prob_cols)) & 
                                          (merged['ensemble_average_correct'] == 1)).astype(int)
        merged['ensemble_created_error'] = ((merged['n_models_correct'] == len(prob_cols)) & 
                                            (merged['ensemble_average_correct'] == 0)).astype(int)
    
    print(f"\nCreated detailed analysis with {len(merged)} samples")
    print(f"Columns: {list(merged.columns)}")
    
    return merged

def load_individual_predictions(prediction_files, threshold=0.5):
    """Load and evaluate individual model predictions"""
    results = []
    
    print("="*80)
    print("INDIVIDUAL MODEL PERFORMANCE")
    print("="*80)
    
    for i, pred_file in enumerate(prediction_files, 1):
        if not os.path.exists(pred_file):
            print(f"⚠️  File not found: {pred_file}")
            continue
        
        try:
            df = pd.read_csv(pred_file)
            
            # Create short model name
            dir_name = os.path.basename(os.path.dirname(pred_file))
            # Extract partition number if present
            if 'allR_' in dir_name:
                partition = dir_name.split('allR_')[1].split('_')[0]
                model_name = f"Model {partition}"
            else:
                model_name = f"Model {i}"
            
            metrics = evaluate_predictions(df, model_name, threshold)
            results.append(metrics)
            
            print(f"\n{model_name} ({dir_name[:50]}...):")
            print(f"  Samples:      {metrics['n_samples']}")
            print(f"  Accuracy:     {metrics['accuracy']:.4f}")
            print(f"  Balanced Acc: {metrics['balanced_accuracy']:.4f}")
            print(f"  Sensitivity:  {metrics['sensitivity']:.4f}")
            print(f"  Specificity:  {metrics['specificity']:.4f}")
            print(f"  F1 Score:     {metrics['f1_score']:.4f}")
            print(f"  AUC:          {metrics['auc']:.4f}")
            print(f"  Confusion:    TP={metrics['TP']}, FP={metrics['FP']}, FN={metrics['FN']}, TN={metrics['TN']}")
            
        except Exception as e:
            print(f"❌ Error loading {pred_file}: {e}")
    
    return pd.DataFrame(results)

def load_ensemble_predictions(ensemble_dir, threshold=0.5):
    """Load and evaluate ensemble predictions"""
    ensemble_files = glob.glob(os.path.join(ensemble_dir, "ensemble_*.csv"))
    ensemble_files = [f for f in ensemble_files if 'comparison' not in f and 'analysis' not in f]
    
    if not ensemble_files:
        print(f"⚠️  No ensemble files found in {ensemble_dir}")
        return pd.DataFrame()
    
    results = []
    
    print("\n" + "="*80)
    print("ENSEMBLE PERFORMANCE")
    print("="*80)
    
    for ensemble_file in sorted(ensemble_files):
        method_name = os.path.basename(ensemble_file).replace('ensemble_', '').replace('.csv', '')
        
        try:
            df = pd.read_csv(ensemble_file)
            display_name = f"Ensemble ({method_name})"
            
            metrics = evaluate_predictions(df, display_name, threshold)
            results.append(metrics)
            
            print(f"\n{display_name}:")
            print(f"  Samples:      {metrics['n_samples']}")
            print(f"  Accuracy:     {metrics['accuracy']:.4f}")
            print(f"  Balanced Acc: {metrics['balanced_accuracy']:.4f}")
            print(f"  Sensitivity:  {metrics['sensitivity']:.4f}")
            print(f"  Specificity:  {metrics['specificity']:.4f}")
            print(f"  F1 Score:     {metrics['f1_score']:.4f}")
            print(f"  AUC:          {metrics['auc']:.4f}")
            
        except Exception as e:
            print(f"❌ Error loading {ensemble_file}: {e}")
    
    return pd.DataFrame(results)

def compare_all(individual_df, ensemble_df, threshold):
    """Create comprehensive comparison"""
    
    # Combine dataframes
    all_results = pd.concat([individual_df, ensemble_df], ignore_index=True)
    
    # Sort by balanced accuracy
    all_results = all_results.sort_values('balanced_accuracy', ascending=False)
    
    print("\n" + "="*80)
    print(f"COMPLETE COMPARISON (Threshold = {threshold:.2f})")
    print("="*80)
    print("\nSorted by Balanced Accuracy:")
    print("-"*80)
    
    # Display table with confusion matrix
    print(f"{'Model':<40} {'Bal Acc':>10} {'Sens':>10} {'Spec':>10} {'F1':>10} {'AUC':>10} {'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5}")
    print("-"*125)
    
    for _, row in all_results.iterrows():
        model_display = row['model'][:38] if len(row['model']) > 38 else row['model']
        print(f"{model_display:<40} {row['balanced_accuracy']:>10.4f} {row['sensitivity']:>10.4f} "
              f"{row['specificity']:>10.4f} {row['f1_score']:>10.4f} {row['auc']:>10.4f} "
              f"{int(row['TP']):>5} {int(row['FP']):>5} {int(row['FN']):>5} {int(row['TN']):>5}")
    
    # Highlight best
    best = all_results.iloc[0]
    print("\n" + "="*80)
    print(f"🏆 BEST PERFORMER: {best['model']}")
    print("="*80)
    print(f"  Balanced Accuracy: {best['balanced_accuracy']:.4f}")
    print(f"  Sensitivity:       {best['sensitivity']:.4f}")
    print(f"  Specificity:       {best['specificity']:.4f}")
    print(f"  F1 Score:          {best['f1_score']:.4f}")
    print(f"  AUC:               {best['auc']:.4f}")
    
    # Check if ensemble is best
    best_individual = individual_df.loc[individual_df['balanced_accuracy'].idxmax()]
    best_ensemble = ensemble_df.loc[ensemble_df['balanced_accuracy'].idxmax()]
    
    print("\n" + "="*80)
    print("ENSEMBLE vs BEST INDIVIDUAL MODEL")
    print("="*80)
    
    print(f"\nBest Individual Model: {best_individual['model']}")
    print(f"  Balanced Accuracy: {best_individual['balanced_accuracy']:.4f}")
    
    print(f"\nBest Ensemble Method: {best_ensemble['model']}")
    print(f"  Balanced Accuracy: {best_ensemble['balanced_accuracy']:.4f}")
    
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

def analyze_error_patterns(error_df, output_dir):
    """Analyze error patterns and disagreements"""
    
    print("\n" + "="*80)
    print("ERROR PATTERN ANALYSIS")
    print("="*80)
    
    # Overall statistics
    total_samples = len(error_df)
    
    # Debug: check for NaN values in n_models_correct
    if 'n_models_correct' in error_df.columns:
        nan_count = error_df['n_models_correct'].isna().sum()
        if nan_count > 0:
            print(f"⚠️  Warning: {nan_count} samples have NaN for n_models_correct")
            # Fill NaN with 0 for counting
            error_df['n_models_correct'] = error_df['n_models_correct'].fillna(0)
    
    # Count error categories
    all_wrong = int(error_df['all_models_wrong'].sum()) if 'all_models_wrong' in error_df.columns else 0
    all_correct = int(error_df['all_models_correct'].sum()) if 'all_models_correct' in error_df.columns else 0
    disagree = int(error_df['models_disagree'].sum()) if 'models_disagree' in error_df.columns else 0
    
    # Verify categories are mutually exclusive and exhaustive
    categorized = all_wrong + all_correct + disagree
    uncategorized = total_samples - categorized
    
    print(f"\nSample Categories (out of {total_samples} total):")
    print(f"  All models correct:  {all_correct:4d} ({100*all_correct/total_samples:.1f}%)")
    print(f"  Models disagree:     {disagree:4d} ({100*disagree/total_samples:.1f}%)")
    print(f"  All models wrong:    {all_wrong:4d} ({100*all_wrong/total_samples:.1f}%)")
    
    if uncategorized > 0:
        print(f"  ⚠️  Uncategorized:    {uncategorized:4d} ({100*uncategorized/total_samples:.1f}%)")
        print(f"\n  Debugging uncategorized samples:")
        uncategorized_samples = error_df[
            (error_df['all_models_correct'] == 0) & 
            (error_df['models_disagree'] == 0) & 
            (error_df['all_models_wrong'] == 0)
        ]
        if len(uncategorized_samples) > 0:
            print(f"  First few uncategorized samples:")
            print(uncategorized_samples[['sample_id', 'true_label', 'n_models_correct']].head(10))
    else:
        print(f"  ✅ All samples categorized: {categorized:4d} ({100*categorized/total_samples:.1f}%)")
    
    if 'ensemble_fixed_error' in error_df.columns:
        ensemble_fixed = error_df['ensemble_fixed_error'].sum()
        ensemble_broke = error_df['ensemble_created_error'].sum()
        
        print(f"\nEnsemble Impact:")
        print(f"  Fixed errors (ensemble correct when ≥1 model wrong): {ensemble_fixed:4d}")
        print(f"  Created errors (ensemble wrong when all models correct): {ensemble_broke:4d}")
        print(f"  Net improvement: {ensemble_fixed - ensemble_broke:+4d} samples")
    
    # Analyze high disagreement cases
    if 'model_prob_std' in error_df.columns:
        high_disagreement = error_df.nlargest(10, 'model_prob_std')
        
        print(f"\n📊 Top 10 Samples with Highest Model Disagreement:")
        print(f"{'Sample ID':<20} {'True':>6} {'Prob Mean':>10} {'Prob Std':>10} {'Models Correct':>15}")
        print("-"*80)
        
        for _, row in high_disagreement.iterrows():
            sample_id = str(row['sample_id'])[:18]
            print(f"{sample_id:<20} {int(row['true_label']):>6} {row['model_prob_mean']:>10.3f} "
                  f"{row['model_prob_std']:>10.3f} {int(row['n_models_correct']):>15}")
    
    # Find samples where all models failed
    if all_wrong > 0:
        all_wrong_samples = error_df[error_df['all_models_wrong'] == 1]
        
        print(f"\n❌ Samples Where ALL Models Failed ({all_wrong} total):")
        print(f"{'Sample ID':<20} {'True Label':>12} {'Mean Prob':>12}")
        print("-"*80)
        
        for _, row in all_wrong_samples.head(20).iterrows():
            sample_id = str(row['sample_id'])[:18]
            print(f"{sample_id:<20} {int(row['true_label']):>12} {row.get('model_prob_mean', 0):>12.3f}")
        
        if len(all_wrong_samples) > 20:
            print(f"... and {len(all_wrong_samples) - 20} more")
    
    # Save detailed error analysis
    error_file = os.path.join(output_dir, 'detailed_error_analysis.csv')
    error_df.to_csv(error_file, index=False)
    print(f"\n✅ Detailed error analysis saved to: {error_file}")
    
    # Save specific error categories
    if all_wrong > 0:
        all_wrong_file = os.path.join(output_dir, 'samples_all_models_wrong.csv')
        error_df[error_df['all_models_wrong'] == 1].to_csv(all_wrong_file, index=False)
        print(f"✅ Samples where all models failed saved to: {all_wrong_file}")
    
    if disagree > 0:
        disagree_file = os.path.join(output_dir, 'samples_models_disagree.csv')
        error_df[error_df['models_disagree'] == 1].to_csv(disagree_file, index=False)
        print(f"✅ Samples where models disagree saved to: {disagree_file}")
    
    if 'ensemble_fixed_error' in error_df.columns and error_df['ensemble_fixed_error'].sum() > 0:
        fixed_file = os.path.join(output_dir, 'samples_ensemble_fixed.csv')
        error_df[error_df['ensemble_fixed_error'] == 1].to_csv(fixed_file, index=False)
        print(f"✅ Samples where ensemble fixed errors saved to: {fixed_file}")

def main():
    parser = argparse.ArgumentParser(
        description='Compare ensemble vs individual model performance with detailed error analysis',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--predictions', nargs='+', required=True,
                       help='Individual model prediction files')
    parser.add_argument('--ensemble_dir', required=True,
                       help='Directory containing ensemble results')
    parser.add_argument('--output', help='Output CSV file for comparison table')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Decision threshold to use')
    
    args = parser.parse_args()
    
    print("="*80)
    print("ENSEMBLE vs INDIVIDUAL MODEL COMPARISON")
    print("="*80)
    print(f"Threshold: {args.threshold:.2f}")
    print(f"Individual models: {len(args.predictions)}")
    print(f"Ensemble directory: {args.ensemble_dir}")
    
    # Load individual model predictions
    individual_df = load_individual_predictions(args.predictions, args.threshold)
    
    if individual_df.empty:
        print("\n❌ No individual model predictions loaded!")
        return
    
    # Load ensemble predictions
    ensemble_df = load_ensemble_predictions(args.ensemble_dir, args.threshold)
    
    if ensemble_df.empty:
        print("\n❌ No ensemble predictions loaded!")
        return
    
    # Compare all
    all_results = compare_all(individual_df, ensemble_df, args.threshold)
    
    # Create detailed error analysis
    error_analysis = create_error_analysis(args.predictions, args.ensemble_dir, args.threshold)
    
    if error_analysis is not None:
        analyze_error_patterns(error_analysis, args.ensemble_dir)
    
    # Save results if output specified
    if args.output:
        all_results.to_csv(args.output, index=False)
        print(f"\n✅ Comparison table saved to: {args.output}")
    
    # Also save in ensemble directory
    output_file = os.path.join(args.ensemble_dir, 'ensemble_vs_individual_comparison.csv')
    all_results.to_csv(output_file, index=False)
    print(f"✅ Comparison table saved to: {output_file}")
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)
    
    # Summary of outputs
    print("\nGenerated Files:")
    print(f"  1. ensemble_vs_individual_comparison.csv - Performance comparison table")
    print(f"  2. detailed_error_analysis.csv - Per-sample predictions from all models")
    print(f"  3. samples_all_models_wrong.csv - Samples where all models failed")
    print(f"  4. samples_models_disagree.csv - Samples where models disagree")
    print(f"  5. samples_ensemble_fixed.csv - Samples where ensemble corrected errors")

if __name__ == '__main__':
    main()
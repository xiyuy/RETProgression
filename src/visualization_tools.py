import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import pandas as pd
import seaborn as sns
import logging
from custom_metrics import roc_curve, precision_recall_curve, auc, average_precision_score
from custom_metrics import confusion_matrix

class TrainingMonitor:
    """
    Class to monitor and visualize training progress.
    
    Tracks metrics over time and creates visualizations for:
    - Learning curves (loss, accuracy, etc.)
    - ROC curves
    - Precision-Recall curves
    - Confusion matrices
    """
    def __init__(self, output_dir, experiment_name):
        """
        Initialize monitor with output directory.
        
        Args:
            output_dir: Directory to save visualizations
            experiment_name: Name of the experiment (used in filenames)
        """
        self.output_dir = output_dir
        self.experiment_name = experiment_name
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize tracking dictionaries
        self.train_metrics = {
            'loss': [],
            'acc': [],
            'f1': [],
            'balanced_acc': [],
            'auc': [],
            'sensitivity': [],
            'specificity': []
        }
        self.val_metrics = {
            'loss': [],
            'acc': [],
            'f1': [],
            'balanced_acc': [],
            'auc': [],
            'sensitivity': [],
            'specificity': []
        }
        
        # For learning rate tracking
        self.learning_rates = []
        self.lr_steps = []
        
        # Store all predictions and labels for ROC/PR curves
        self.epoch_predictions = {}
        self.epoch_targets = {}
        
        # Store confusion matrices
        self.confusion_matrices = {}
        
    def update_metrics(self, epoch, phase_results):
        """
        Update metrics with results from an epoch.
        
        Args:
            epoch: Current epoch number
            phase_results: Dictionary with 'train' and 'val' results
        """
        try:
            # Update training metrics
            if 'train' in phase_results:
                train_results = phase_results['train']
                for metric in self.train_metrics:
                    if metric in train_results:
                        self.train_metrics[metric].append(train_results[metric])
            
            # Update validation metrics
            if 'val' in phase_results:
                val_results = phase_results['val']
                for metric in self.val_metrics:
                    if metric in val_results:
                        self.val_metrics[metric].append(val_results[metric])
        except Exception as e:
            logging.warning(f"Error updating metrics: {str(e)}")
        
    def update_predictions(self, epoch, phase, outputs, targets):
        """
        Store predictions and targets for later curve generation.
        
        Args:
            epoch: Current epoch number
            phase: 'train' or 'val'
            outputs: Model outputs (logits or probabilities)
            targets: Ground truth labels
        """
        try:
            # Safety check for empty inputs
            if outputs is None or targets is None or len(outputs) == 0 or len(targets) == 0:
                logging.warning(f"Empty outputs or targets for {phase} epoch {epoch}")
                return
                
            # Convert to probabilities if logits
            if outputs.dim() > 1 and outputs.size(1) > 1:
                probs = torch.nn.functional.softmax(outputs, dim=1)
                # For binary classification, we want the probability of class 1
                if probs.size(1) == 2:
                    probs = probs[:, 1]
            else:
                # Already probabilities or binary logits
                probs = torch.sigmoid(outputs) if outputs.dim() == 1 else outputs
            
            # Convert to numpy arrays
            probs_np = probs.detach().cpu().numpy()
            targets_np = targets.detach().cpu().numpy()
            
            # Store
            key = f"{phase}_epoch{epoch}"
            self.epoch_predictions[key] = probs_np
            self.epoch_targets[key] = targets_np
        except Exception as e:
            logging.warning(f"Error updating predictions: {str(e)}")
        
    def update_confusion_matrix(self, epoch, phase, cm_values):
        """
        Store confusion matrix values.
        
        Args:
            epoch: Current epoch number
            phase: 'train' or 'val'
            cm_values: Dictionary with TP, FP, FN, TN counts
        """
        try:
            if not isinstance(cm_values, dict):
                logging.warning(f"Invalid confusion matrix format for {phase} epoch {epoch}")
                return
                
            key = f"{phase}_epoch{epoch}"
            self.confusion_matrices[key] = cm_values
        except Exception as e:
            logging.warning(f"Error updating confusion matrix: {str(e)}")
        
    def update_learning_rate(self, lr, step):
        """
        Track learning rate changes.
        
        Args:
            lr: Current learning rate
            step: Current step (can be global step or epoch)
        """
        try:
            self.learning_rates.append(float(lr))
            self.lr_steps.append(int(step))
        except Exception as e:
            logging.warning(f"Error updating learning rate: {str(e)}")
        
    def plot_learning_curves(self, save=True, display=False):
        """
        Plot learning curves for various metrics.
        
        Args:
            save: Whether to save plots to disk
            display: Whether to display plots
        """
        try:
            # Ensure we have at least some data to plot
            if not self.train_metrics['loss']:
                logging.info("No metrics data available yet for plotting learning curves")
                return
                
            epochs = range(1, len(self.train_metrics['loss']) + 1)
            
            # Group related metrics together
            metric_groups = [
                ('loss', 'Loss'),
                (('acc', 'balanced_acc'), 'Accuracy Metrics'),
                (('sensitivity', 'specificity'), 'Sensitivity/Specificity'),
                (('f1', 'auc'), 'F1/AUC Metrics')
            ]
            
            for metrics, title in metric_groups:
                plt.figure(figsize=(10, 6))
                has_data = False
                
                if isinstance(metrics, str):
                    # Single metric - check if we have data for this metric
                    metric = metrics
                    if metric in self.train_metrics and len(self.train_metrics[metric]) > 0:
                        if len(self.train_metrics[metric]) != len(epochs):
                            logging.warning(f"Length mismatch: epochs {len(epochs)}, train_{metric} {len(self.train_metrics[metric])}")
                            continue
                        plt.plot(epochs, self.train_metrics[metric], 'b-', label=f'Train {metric}')
                        has_data = True
                    if metric in self.val_metrics and len(self.val_metrics[metric]) > 0:
                        if len(self.val_metrics[metric]) != len(epochs):
                            logging.warning(f"Length mismatch: epochs {len(epochs)}, val_{metric} {len(self.val_metrics[metric])}")
                            continue
                        plt.plot(epochs, self.val_metrics[metric], 'r-', label=f'Val {metric}')
                        has_data = True
                else:
                    # Multiple metrics in same plot
                    for metric in metrics:
                        if metric in self.train_metrics and len(self.train_metrics[metric]) > 0:
                            if len(self.train_metrics[metric]) != len(epochs):
                                logging.warning(f"Length mismatch: epochs {len(epochs)}, train_{metric} {len(self.train_metrics[metric])}")
                                continue
                            plt.plot(epochs, self.train_metrics[metric], marker='o', linestyle='-',
                                    label=f'Train {metric}')
                            has_data = True
                        if metric in self.val_metrics and len(self.val_metrics[metric]) > 0:
                            if len(self.val_metrics[metric]) != len(epochs):
                                logging.warning(f"Length mismatch: epochs {len(epochs)}, val_{metric} {len(self.val_metrics[metric])}")
                                continue
                            plt.plot(epochs, self.val_metrics[metric], marker='s', linestyle='-',
                                    label=f'Val {metric}')
                            has_data = True
                
                # Skip empty plots
                if not has_data:
                    plt.close()
                    continue
                    
                plt.title(title)
                plt.xlabel('Epoch')
                plt.ylabel('Value')
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.legend()
                
                # Add horizontal line at 0.5 for accuracy-related metrics
                if title in ['Accuracy Metrics', 'Sensitivity/Specificity', 'F1/AUC Metrics']:
                    plt.axhline(y=0.5, color='k', linestyle=':', alpha=0.5)
                
                plt.tight_layout()
                
                if save:
                    filename = os.path.join(self.output_dir,
                                         f"{self.experiment_name}_{title.replace('/', '_')}.png")
                    plt.savefig(filename, dpi=300, bbox_inches='tight')
                    logging.info(f"Saved {title} plot to {filename}")
                
                if display:
                    plt.show()
                else:
                    plt.close()
        except Exception as e:
            logging.error(f"Error plotting learning curves: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            plt.close('all')  # Close any open figures
        
    def plot_learning_rate(self, save=True, display=False):
        """
        Plot learning rate over time.
        
        Args:
            save: Whether to save plots to disk
            display: Whether to display plots
        """
        try:
            if not self.learning_rates or not self.lr_steps:
                logging.info("No learning rate data available yet")
                return
                
            if len(self.learning_rates) != len(self.lr_steps):
                logging.warning(f"Length mismatch: steps {len(self.lr_steps)}, lr values {len(self.learning_rates)}")
                return
            
            plt.figure(figsize=(10, 6))
            plt.plot(self.lr_steps, self.learning_rates, 'b-')
            plt.title('Learning Rate Schedule')
            plt.xlabel('Step')
            plt.ylabel('Learning Rate')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.yscale('log')  # Log scale for better visualization
            
            plt.tight_layout()
            
            if save:
                filename = os.path.join(self.output_dir,
                                     f"{self.experiment_name}_learning_rate.png")
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                logging.info(f"Saved learning rate plot to {filename}")
            
            if display:
                plt.show()
            else:
                plt.close()
        except Exception as e:
            logging.error(f"Error plotting learning rate: {str(e)}")
            plt.close('all')  # Close any open figures
        
    def plot_roc_curves(self, save=True, display=False):
        """
        Plot ROC curves for each epoch.
        
        Args:
            save: Whether to save plots to disk
            display: Whether to display plots
        """
        try:
            # Only plot validation ROCs
            val_epochs = sorted([int(k.split('epoch')[1]) for k in self.epoch_predictions.keys() 
                               if k.startswith('val_')])
            
            if not val_epochs:
                logging.info("No validation predictions available yet for ROC curves")
                return
            
            plt.figure(figsize=(10, 8))
            has_data = False
            
            # Plot random baseline
            plt.plot([0, 1], [0, 1], 'k--', label='Random', alpha=0.5)
            
            # Plot ROC for each epoch or selected epochs
            # If too many epochs, plot only a subset (e.g., first, last, and some in between)
            if len(val_epochs) > 10:
                plot_epochs = [val_epochs[0]] + \
                            [val_epochs[i] for i in range(len(val_epochs)) if i % 3 == 0 and i > 0] + \
                            [val_epochs[-1]]
                plot_epochs = sorted(set(plot_epochs))
            else:
                plot_epochs = val_epochs
            
            for epoch in plot_epochs:
                key = f"val_epoch{epoch}"
                if key in self.epoch_predictions and key in self.epoch_targets:
                    # Verify data is not empty
                    if len(self.epoch_predictions[key]) == 0 or len(self.epoch_targets[key]) == 0:
                        logging.warning(f"Empty predictions or targets for {key}")
                        continue
                        
                    # Verify lengths match
                    if len(self.epoch_predictions[key]) != len(self.epoch_targets[key]):
                        logging.warning(f"Length mismatch in {key}: predictions {len(self.epoch_predictions[key])}, targets {len(self.epoch_targets[key])}")
                        continue
                    
                    y_pred = self.epoch_predictions[key]
                    y_true = self.epoch_targets[key]
                    
                    # Check if we have both classes
                    if len(np.unique(y_true)) < 2:
                        logging.warning(f"Only one class present in targets for {key}, skipping ROC")
                        continue
                    
                    try:
                        # Compute ROC curve and ROC area
                        fpr, tpr, _ = roc_curve(y_true, y_pred)
                        roc_auc = auc(fpr, tpr)
                        
                        plt.plot(fpr, tpr, lw=2, label=f'Epoch {epoch} (AUC = {roc_auc:.3f})')
                        has_data = True
                    except Exception as curve_error:
                        logging.warning(f"Error computing ROC curve for epoch {epoch}: {str(curve_error)}")
                        continue
            
            # Skip if no valid data
            if not has_data:
                plt.close()
                return
                
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curves')
            plt.legend(loc="lower right")
            plt.grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            
            if save:
                filename = os.path.join(self.output_dir,
                                     f"{self.experiment_name}_roc_curves.png")
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                logging.info(f"Saved ROC curves plot to {filename}")
            
            if display:
                plt.show()
            else:
                plt.close()
        except Exception as e:
            logging.error(f"Error plotting ROC curves: {str(e)}")
            plt.close('all')  # Close any open figures
        
    def plot_pr_curves(self, save=True, display=False):
        """
        Plot Precision-Recall curves for each epoch.
        
        Args:
            save: Whether to save plots to disk
            display: Whether to display plots
        """
        try:
            # Only plot validation PR curves
            val_epochs = sorted([int(k.split('epoch')[1]) for k in self.epoch_predictions.keys() 
                               if k.startswith('val_')])
            
            if not val_epochs:
                logging.info("No validation predictions available yet for PR curves")
                return
            
            plt.figure(figsize=(10, 8))
            has_data = False
            
            # If too many epochs, plot only a subset
            if len(val_epochs) > 10:
                plot_epochs = [val_epochs[0]] + \
                            [val_epochs[i] for i in range(len(val_epochs)) if i % 3 == 0 and i > 0] + \
                            [val_epochs[-1]]
                plot_epochs = sorted(set(plot_epochs))
            else:
                plot_epochs = val_epochs
            
            # Calculate class ratio for baseline if possible
            baseline_plotted = False
            for epoch in val_epochs:
                key = f"val_epoch{epoch}"
                if key in self.epoch_targets and len(self.epoch_targets[key]) > 0:
                    y_true = self.epoch_targets[key]
                    baseline = np.sum(y_true) / len(y_true)
                    plt.axhline(y=baseline, color='k', linestyle='--',
                             label=f'Baseline ({baseline:.3f})', alpha=0.5)
                    baseline_plotted = True
                    break
            
            for epoch in plot_epochs:
                key = f"val_epoch{epoch}"
                if key in self.epoch_predictions and key in self.epoch_targets:
                    # Verify data is not empty
                    if len(self.epoch_predictions[key]) == 0 or len(self.epoch_targets[key]) == 0:
                        logging.warning(f"Empty predictions or targets for {key}")
                        continue
                        
                    # Verify lengths match
                    if len(self.epoch_predictions[key]) != len(self.epoch_targets[key]):
                        logging.warning(f"Length mismatch in {key}: predictions {len(self.epoch_predictions[key])}, targets {len(self.epoch_targets[key])}")
                        continue
                    
                    y_pred = self.epoch_predictions[key]
                    y_true = self.epoch_targets[key]
                    
                    # Log prediction distribution stats
                    logging.info(f"Epoch {epoch} predictions - min: {np.min(y_pred):.4f}, max: {np.max(y_pred):.4f}, "
                                f"mean: {np.mean(y_pred):.4f}, std: {np.std(y_pred):.4f}")
                    logging.info(f"Epoch {epoch} unique prediction values: {len(np.unique(y_pred))}")
                    logging.info(f"Epoch {epoch} positive samples: {np.sum(y_true)}/{len(y_true)} "
                                f"({np.sum(y_true)/len(y_true)*100:.2f}%)")

                    # Check if we have positive samples
                    if np.sum(y_true) == 0:
                        logging.warning(f"No positive samples in targets for {key}, skipping PR curve")
                        continue
                    
                    try:
                        # Compute Precision-Recall curve
                        precision, recall, _ = precision_recall_curve(y_true, y_pred)
                        average_precision = average_precision_score(y_true, y_pred)
                        
                        plt.plot(recall, precision, lw=2,
                               label=f'Epoch {epoch} (AP = {average_precision:.3f})')
                        has_data = True
                    except Exception as curve_error:
                        logging.warning(f"Error computing PR curve for epoch {epoch}: {str(curve_error)}")
                        continue
            
            # Skip if no valid data
            if not has_data:
                plt.close()
                return
                
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curves')
            plt.legend(loc="lower left")
            plt.grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            
            if save:
                filename = os.path.join(self.output_dir,
                                     f"{self.experiment_name}_pr_curves.png")
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                logging.info(f"Saved PR curves plot to {filename}")
            
            if display:
                plt.show()
            else:
                plt.close()
        except Exception as e:
            logging.error(f"Error plotting PR curves: {str(e)}")
            plt.close('all')  # Close any open figures
        
    def plot_confusion_matrices(self, save=True, display=False):
        """
        Plot confusion matrices for selected epochs.
        
        Args:
            save: Whether to save plots to disk
            display: Whether to display plots
        """
        try:
            # Only plot validation confusion matrices
            val_epochs = sorted([int(k.split('epoch')[1]) for k in self.confusion_matrices.keys() 
                            if k.startswith('val_')])
            
            if not val_epochs:
                logging.info("No validation confusion matrices available yet")
                return
            
            # If too many epochs, plot only first, last, and best (by balanced accuracy)
            if len(val_epochs) > 6:
                # Find epoch with best balanced accuracy if metric exists and has data
                if (self.val_metrics['balanced_acc'] and 
                    len(self.val_metrics['balanced_acc']) > 0 and 
                    min(val_epochs) < len(self.val_metrics['balanced_acc'])):
                    
                    # Only consider epochs that have data in val_metrics
                    available_epochs = [e for e in val_epochs if e-1 < len(self.val_metrics['balanced_acc'])]
                    if available_epochs:
                        available_accs = [self.val_metrics['balanced_acc'][e-1] for e in available_epochs if e-1 < len(self.val_metrics['balanced_acc'])]
                        if available_accs:
                            best_idx = np.argmax(available_accs)
                            best_epoch = available_epochs[best_idx]
                            plot_epochs = [val_epochs[0], best_epoch, val_epochs[-1]]
                            # Remove duplicates and sort
                            plot_epochs = sorted(set([e for e in plot_epochs if e in val_epochs]))
                        else:
                            plot_epochs = [val_epochs[0], val_epochs[-1]]
                    else:
                        plot_epochs = [val_epochs[0], val_epochs[-1]]
                else:
                    # If no balanced accuracy metric, just use first and last
                    plot_epochs = [val_epochs[0], val_epochs[-1]]
                    # Add middle epoch if we have more than 2 epochs
                    if len(val_epochs) > 2:
                        mid_idx = len(val_epochs) // 2
                        plot_epochs = [val_epochs[0], val_epochs[mid_idx], val_epochs[-1]]
            else:
                plot_epochs = val_epochs
            
            logging.info(f"Plotting confusion matrices for epochs: {plot_epochs}")
            
            for epoch in plot_epochs:
                key = f"val_epoch{epoch}"
                if key in self.confusion_matrices:
                    cm = self.confusion_matrices[key]
                    
                    # Log debug info
                    logging.debug(f"Confusion matrix for {key}: {cm}")
                    
                    # Verify confusion matrix has required keys
                    required_keys = ['TP', 'FP', 'FN', 'TN']
                    if not all(k in cm for k in required_keys):
                        logging.warning(f"Missing required keys in confusion matrix for {key}")
                        continue
                    
                    # Extract confusion matrix values
                    tn = cm.get('TN', 0)
                    fp = cm.get('FP', 0)
                    fn = cm.get('FN', 0)
                    tp = cm.get('TP', 0)
                    
                    # Create 2x2 confusion matrix
                    cm_array = np.array([[tn, fp], [fn, tp]])
                    
                    # Safety check for division by zero
                    row_sums = cm_array.sum(axis=1)
                    if np.any(row_sums == 0):
                        cm_norm = np.zeros_like(cm_array, dtype=float)
                        for i in range(len(row_sums)):
                            if row_sums[i] > 0:
                                cm_norm[i, :] = cm_array[i, :] / row_sums[i]
                    else:
                        # Normalize to get percentages
                        cm_norm = cm_array.astype('float') / row_sums[:, np.newaxis]
                    
                    cm_norm = np.nan_to_num(cm_norm)  # Replace NaNs with 0
                    
                    plt.figure(figsize=(8, 6))
                    
                    # Plot both raw counts and percentages
                    labels = [f"{cm_array[i, j]}\n{cm_norm[i, j]*100:.1f}%" 
                            for i in range(2) for j in range(2)]
                    labels = np.array(labels).reshape(2, 2)
                    
                    sns.heatmap(cm_array, annot=labels, fmt='', cmap='Blues',
                            xticklabels=['Negative', 'Positive'],
                            yticklabels=['Negative', 'Positive'])
                    
                    plt.title(f'Confusion Matrix - Epoch {epoch}')
                    plt.ylabel('True Label')
                    plt.xlabel('Predicted Label')
                    
                    # Calculate metrics directly from confusion matrix - this should always work
                    try:
                        # Calculate metrics from confusion matrix values
                        total = tp + tn + fp + fn
                        accuracy = (tp + tn) / max(total, 1e-15)
                        sensitivity = tp / max(tp + fn, 1e-15)
                        specificity = tn / max(tn + fp, 1e-15)
                        precision = tp / max(tp + fp, 1e-15)
                        f1 = 2 * precision * sensitivity / max(precision + sensitivity, 1e-15)
                        balanced_acc = (sensitivity + specificity) / 2

                        # Always use calculated metrics for early epochs
                        # Only try to use stored metrics if we're sure they exist
                        use_stored_metrics = False
                        
                        epoch_idx = epoch - 1  # Convert to 0-indexed
                        if (epoch_idx < len(self.val_metrics['acc']) and
                            epoch_idx < len(self.val_metrics['balanced_acc']) and
                            epoch_idx < len(self.val_metrics['sensitivity']) and
                            epoch_idx < len(self.val_metrics['specificity']) and
                            epoch_idx < len(self.val_metrics['f1']) and
                            epoch_idx < len(self.val_metrics['auc'])):
                            use_stored_metrics = True
                            logging.debug(f"Using stored metrics for epoch {epoch}")
                        else:
                            logging.debug(f"Using calculated metrics for epoch {epoch}")
                        
                        # Use stored metrics if available, otherwise use calculated metrics
                        if use_stored_metrics:
                            try:
                                metrics_text = (
                                    f"Accuracy: {self.val_metrics['acc'][epoch_idx]:.4f}, "
                                    f"Balanced Accuracy: {self.val_metrics['balanced_acc'][epoch_idx]:.4f}\n"
                                    f"Sensitivity: {self.val_metrics['sensitivity'][epoch_idx]:.4f}, "
                                    f"Specificity: {self.val_metrics['specificity'][epoch_idx]:.4f}\n"
                                    f"F1 Score: {self.val_metrics['f1'][epoch_idx]:.4f}, "
                                    f"AUC: {self.val_metrics['auc'][epoch_idx]:.4f}"
                                )
                            except Exception as idx_error:
                                logging.warning(f"Failed to use stored metrics: {str(idx_error)}")
                                # Fall back to calculated metrics
                                use_stored_metrics = False
                        
                        if not use_stored_metrics:
                            # Use calculated metrics (note: AUC not calculable from confusion matrix alone)
                            metrics_text = (
                                f"Accuracy: {accuracy:.4f}, "
                                f"Balanced Accuracy: {balanced_acc:.4f}\n"
                                f"Sensitivity: {sensitivity:.4f}, "
                                f"Specificity: {specificity:.4f}\n"
                                f"F1 Score: {f1:.4f}"
                            )
                        
                        plt.figtext(0.5, 0.01, metrics_text, ha='center', fontsize=10,
                                bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
                    except Exception as metrics_error:
                        logging.warning(f"Error computing metrics for confusion matrix: {str(metrics_error)}")
                        # Just show the confusion matrix without metrics text
                    
                    plt.tight_layout()
                    
                    if save:
                        filename = os.path.join(self.output_dir,
                                            f"{self.experiment_name}_cm_epoch{epoch}.png")
                        plt.savefig(filename, dpi=300, bbox_inches='tight')
                        logging.info(f"Saved confusion matrix plot for epoch {epoch} to {filename}")
                    
                    if display:
                        plt.show()
                    else:
                        plt.close()
        except Exception as e:
            logging.error(f"Error plotting confusion matrices: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            plt.close('all')  # Close any open figures

    def generate_all_plots(self):
        """
        Generate all available plots.
        """
        try:
            logging.info("Generating all plots...")
            
            self.plot_learning_curves()
            self.plot_learning_rate()
            self.plot_roc_curves()
            self.plot_pr_curves()
            self.plot_confusion_matrices()
            
            logging.info(f"All plots saved to {self.output_dir}")
        except Exception as e:
            logging.error(f"Error generating plots: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            plt.close('all')  # Close any open figures
"""
Model Evaluation and Metrics Module
Provides comprehensive evaluation capabilities for fraud detection models.
"""
import numpy as np
import logging
from typing import Dict, Tuple, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve, classification_report
)

logger = logging.getLogger(__name__)


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (for ROC-AUC, PR-AUC)
    
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Basic classification metrics
    metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
    metrics['precision'] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics['recall'] = float(recall_score(y_true, y_pred, zero_division=0))
    metrics['f1_score'] = float(f1_score(y_true, y_pred, zero_division=0))
    
    # Confusion matrix components
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['true_positives'] = int(tp)
    metrics['false_positives'] = int(fp)
    metrics['true_negatives'] = int(tn)
    metrics['false_negatives'] = int(fn)
    
    # Derived metrics
    metrics['fpr'] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    metrics['fnr'] = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    metrics['specificity'] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    
    # Probability-based metrics
    if y_proba is not None:
        try:
            metrics['roc_auc'] = float(roc_auc_score(y_true, y_proba))
        except Exception as e:
            logger.warning(f"Could not compute ROC-AUC: {e}")
            metrics['roc_auc'] = 0.0
        
        try:
            metrics['pr_auc'] = float(average_precision_score(y_true, y_proba))
        except Exception as e:
            logger.warning(f"Could not compute PR-AUC: {e}")
            metrics['pr_auc'] = 0.0
    else:
        metrics['roc_auc'] = 0.0
        metrics['pr_auc'] = 0.0
    
    return metrics


def optimize_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    metric: str = 'f1'
) -> Tuple[float, float]:
    """
    Find optimal classification threshold.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        metric: Metric to optimize ('f1', 'youden', 'precision_recall')
    
    Returns:
        optimal_threshold, best_metric_value
    """
    if metric == 'f1':
        # Optimize F1 score
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        
        # Calculate F1 for each threshold
        with np.errstate(divide='ignore', invalid='ignore'):
            f1_scores = 2 * (precision * recall) / (precision + recall)
            f1_scores = np.nan_to_num(f1_scores, 0.0)
        
        # Find best threshold
        best_idx = np.argmax(f1_scores)
        best_f1 = f1_scores[best_idx]
        
        # Handle threshold array length
        if best_idx < len(thresholds):
            best_threshold = thresholds[best_idx]
        else:
            best_threshold = 1.0
        
        return float(best_threshold), float(best_f1)
    
    elif metric == 'youden':
        # Youden's J statistic (sensitivity + specificity - 1)
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        best_threshold = thresholds[best_idx]
        best_j = j_scores[best_idx]
        
        return float(best_threshold), float(best_j)
    
    else:
        raise ValueError(f"Unknown metric: {metric}")


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model"
):
    """Print detailed classification report."""
    print(f"\n{model_name} Classification Report:")
    print("="*60)
    print(classification_report(y_true, y_pred, target_names=['Legitimate', 'Fraud']))
    print("="*60)


def save_evaluation_report(
    metrics: Dict[str, float],
    model_name: str,
    output_path: str
):
    """Save evaluation metrics to file."""
    import json
    
    with open(output_path, 'w') as f:
        json.dump({
            'model': model_name,
            'metrics': metrics
        }, f, indent=2)
    
    logger.info(f"Evaluation report saved: {output_path}")

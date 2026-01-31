from __future__ import annotations
from typing import Dict, Optional
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, RocCurveDisplay, roc_curve
import matplotlib.pyplot as plt


def compute_metrics(y_true, y_pred, y_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Compute classification metrics.
    
    NVIDIA Insight: AUC-ROC is the primary metric for fraud detection.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional, for ROC-AUC)
        
    Returns:
        Dictionary of metric names and values
    """
    # Ensure predictions are binary (handle multiclass output from quantum models)
    y_pred_binary = np.where(y_pred > 0.5, 1, 0) if y_pred.dtype == float else y_pred
    
    # Check if we have more than 2 classes in predictions
    unique_classes = np.unique(y_pred_binary)
    if len(unique_classes) > 2:
        # Force to binary by taking modulo 2 or thresholding
        y_pred_binary = np.where(y_pred_binary >= 1, 1, 0)
    
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred_binary)),
        "precision": float(precision_score(y_true, y_pred_binary, zero_division=0, average='binary')),
        "recall": float(recall_score(y_true, y_pred_binary, zero_division=0, average='binary')),
        "f1_score": float(f1_score(y_true, y_pred_binary, zero_division=0, average='binary')),
    }
    
    # Calculate False Positive Rate (FPR)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
    metrics["fpr"] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    
    # AUC-ROC: Primary metric for fraud detection (NVIDIA insight)
    if y_proba is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        except Exception as e:
            print(f"Warning: Could not compute ROC-AUC: {e}")
            pass
    
    return metrics


def save_pr_curve(estimator, X_test, y_test, out_path: str) -> None:
    """Generate and save Precision-Recall curve plot.
    
    Args:
        estimator: Trained classifier
        X_test: Test features
        y_test: Test labels
        out_path: Output file path for the figure
    """
    try:
        from sklearn.metrics import PrecisionRecallDisplay
        PrecisionRecallDisplay.from_estimator(estimator, X_test, y_test)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.gcf().tight_layout()
        plt.title("Precision-Recall Curve")
        plt.savefig(out_path)
        plt.close()
    except Exception as e:
        print(f"Warning: Could not save PR Curve: {e}")
        pass


def save_confusion_matrix(y_true, y_pred, out_path: str) -> None:
    """Generate and save confusion matrix plot.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        out_path: Output file path for the figure
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    fig.colorbar(im)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_roc_curve(estimator, X_test, y_test, out_path: str) -> None:
    """Generate and save ROC curve plot.
    
    Args:
        estimator: Trained classifier with predict_proba method
        X_test: Test features
        y_test: Test labels
        out_path: Output file path for the figure
    """
    try:
        RocCurveDisplay.from_estimator(estimator, X_test, y_test)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.gcf().tight_layout()
        plt.savefig(out_path)
        plt.close()
    except Exception:
        # Some estimators may not support ROC directly
        pass

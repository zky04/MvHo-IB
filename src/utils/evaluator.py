"""Model evaluation utilities.

Provides comprehensive evaluation metrics for classification tasks.
"""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from typing import Dict, List, Optional, Tuple, Union
import logging


def compute_metrics(predictions: Union[List, np.ndarray, torch.Tensor],
                   targets: Union[List, np.ndarray, torch.Tensor],
                   num_classes: int = None,
                   average: str = 'weighted') -> Dict[str, float]:
    """Compute classification task evaluation metrics.
    
    Calculate common classification metrics including accuracy, precision, recall, and F1-score.
    
    Args:
        predictions: Model prediction results
        targets: True labels
        num_classes: Number of classes, auto-inferred if None
        average: Multi-class metric averaging method, options: 'micro', 'macro', 'weighted'
        
    Returns:
        Dictionary containing various evaluation metrics
        
    Raises:
        ValueError: When prediction and label lengths don't match
    """
    # Convert to numpy arrays
    if torch.is_tensor(predictions):
        predictions = predictions.cpu().numpy()
    if torch.is_tensor(targets):
        targets = targets.cpu().numpy()
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    if len(predictions) != len(targets):
        raise ValueError(f"Prediction and label length mismatch: {len(predictions)} vs {len(targets)}")
    
    # Auto-infer number of classes
    if num_classes is None:
        num_classes = len(np.unique(targets))
    
    # Calculate basic metrics
    metrics = {
        'accuracy': accuracy_score(targets, predictions),
        'precision': precision_score(targets, predictions, average=average, zero_division=0),
        'recall': recall_score(targets, predictions, average=average, zero_division=0),
        'f1_score': f1_score(targets, predictions, average=average, zero_division=0)
    }
    
    # For binary classification tasks, calculate AUC
    if num_classes == 2:
        try:
            # Assuming predictions are class labels, probabilities needed for AUC
            # This is simplified processing, probability values should be passed in actual use
            auc_score = roc_auc_score(targets, predictions)
            metrics['auc'] = auc_score
        except ValueError:
            # Skip if AUC calculation fails
            pass
    
    # Calculate per-class metrics
    precision_per_class = precision_score(targets, predictions, average=None, zero_division=0)
    recall_per_class = recall_score(targets, predictions, average=None, zero_division=0)
    f1_per_class = f1_score(targets, predictions, average=None, zero_division=0)
    
    for i in range(len(precision_per_class)):
        metrics[f'precision_class_{i}'] = precision_per_class[i]
        metrics[f'recall_class_{i}'] = recall_per_class[i]
        metrics[f'f1_class_{i}'] = f1_per_class[i]
    
    return metrics


def evaluate_model(predictions: Union[List, np.ndarray, torch.Tensor],
                  targets: Union[List, np.ndarray, torch.Tensor],
                  phase: str = "Evaluation",
                  class_names: Optional[List[str]] = None,
                  print_results: bool = True) -> Dict[str, Union[float, np.ndarray]]:
    """Comprehensive model evaluation.
    
    Calculate various evaluation metrics and optionally print detailed results.
    
    Args:
        predictions: Model prediction results
        targets: True labels
        phase: Evaluation phase name (e.g., "Training", "Validation", "Test")
        class_names: List of class names for display
        print_results: Whether to print evaluation results
        
    Returns:
        Dictionary containing evaluation metrics and confusion matrix
    """
    # Convert to numpy arrays
    if torch.is_tensor(predictions):
        predictions = predictions.cpu().numpy()
    if torch.is_tensor(targets):
        targets = targets.cpu().numpy()
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # Calculate basic metrics
    metrics = compute_metrics(predictions, targets)
    
    # Calculate confusion matrix
    cm = confusion_matrix(targets, predictions)
    metrics['confusion_matrix'] = cm
    
    # Generate classification report
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(len(np.unique(targets)))]
    
    report = classification_report(
        targets, predictions, 
        target_names=class_names, 
        digits=4,
        output_dict=True
    )
    
    if print_results:
        _print_evaluation_results(metrics, cm, report, phase, class_names)
    
    # Add classification report to metrics
    metrics['classification_report'] = report
    
    return metrics


def _print_evaluation_results(metrics: Dict[str, float],
                            confusion_matrix: np.ndarray,
                            classification_report: Dict,
                            phase: str,
                            class_names: List[str]) -> None:
    """Print formatted evaluation results.
    
    Args:
        metrics: Evaluation metrics dictionary
        confusion_matrix: Confusion matrix
        classification_report: Classification report
        phase: Evaluation phase name
        class_names: List of class names
    """
    print(f"\n{'=' * 50}")
    print(f"{phase} Evaluation Results")
    print(f"{'=' * 50}")
    
    # Print main metrics
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    
    if 'auc' in metrics:
        print(f"AUC: {metrics['auc']:.4f}")
    
    # Print confusion matrix
    print(f"\nConfusion Matrix:")
    print("Predicted\\Actual", end="")
    for name in class_names:
        print(f"\t{name}", end="")
    print()
    
    for i, name in enumerate(class_names):
        print(f"{name}", end="")
        for j in range(len(class_names)):
            print(f"\t{confusion_matrix[i, j]}", end="")
        print()
    
    # Print detailed classification report
    print(f"\nDetailed Classification Report:")
    print(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
    print("-" * 60)
    
    for class_name in class_names:
        if class_name in classification_report:
            report = classification_report[class_name]
            print(f"{class_name:<15} {report['precision']:<10.4f} "
                  f"{report['recall']:<10.4f} {report['f1-score']:<10.4f} "
                  f"{report['support']:<10}")
    
    # Print macro and weighted averages
    if 'macro avg' in classification_report:
        macro_avg = classification_report['macro avg']
        print(f"{'Macro Avg':<15} {macro_avg['precision']:<10.4f} "
              f"{macro_avg['recall']:<10.4f} {macro_avg['f1-score']:<10.4f} "
              f"{macro_avg['support']:<10}")
    
    if 'weighted avg' in classification_report:
        weighted_avg = classification_report['weighted avg']
        print(f"{'Weighted Avg':<15} {weighted_avg['precision']:<10.4f} "
              f"{weighted_avg['recall']:<10.4f} {weighted_avg['f1-score']:<10.4f} "
              f"{weighted_avg['support']:<10}")
    
    print(f"{'=' * 50}\n")


def calculate_class_weights(labels: Union[List, np.ndarray, torch.Tensor]) -> torch.Tensor:
    """Calculate class weights for handling sample imbalance.
    
    Args:
        labels: Training set labels
        
    Returns:
        Class weight tensor
    """
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()
    
    labels = np.array(labels)
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    # Calculate inverse frequency weights
    total_samples = len(labels)
    num_classes = len(unique_labels)
    
    weights = total_samples / (num_classes * counts)
    
    # Create weight tensor
    class_weights = torch.zeros(num_classes)
    for i, label in enumerate(unique_labels):
        class_weights[label] = weights[i]
    
    return class_weights 
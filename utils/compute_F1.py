"""Clustering metric implementation (pairwise and graph-based)."""
from typing import Tuple
import numpy as np
from scipy.sparse import base
from sklearn.metrics import cluster


def pairwise_precision(y_true, y_pred):
    """Computes pairwise precision of two clusterings.

    Args:
      y_true: An [n] int ground-truth cluster vector.
      y_pred: An [n] int predicted cluster vector.

    Returns:
      Precision value computed from the true/false positives and negatives.
    """
    true_positives, false_positives, _, _ = _pairwise_confusion(y_true, y_pred)
    return true_positives / (true_positives + false_positives)


def pairwise_recall(y_true, y_pred):
    """Computes pairwise recall of two clusterings.

    Args:
      y_true: An (n,) int ground-truth cluster vector.
      y_pred: An (n,) int predicted cluster vector.

    Returns:
      Recall value computed from the true/false positives and negatives.
    """
    true_positives, _, false_negatives, _ = _pairwise_confusion(y_true, y_pred)
    return true_positives / (true_positives + false_negatives)


def pairwise_accuracy(y_true, y_pred):
    """Computes pairwise accuracy of two clusterings.

    Args:
      y_true: An (n,) int ground-truth cluster vector.
      y_pred: An (n,) int predicted cluster vector.

    Returns:
      Accuracy value computed from the true/false positives and negatives.
    """
    true_pos, false_pos, false_neg, true_neg = _pairwise_confusion(y_true, y_pred)
    return (true_pos + false_pos) / (true_pos + false_pos + false_neg + true_neg)


def _pairwise_confusion(
        y_true,
        y_pred):
    """Computes pairwise confusion matrix of two clusterings.

    Args:
      y_true: An (n,) int ground-truth cluster vector.
      y_pred: An (n,) int predicted cluster vector.

    Returns:
      True positive, false positive, true negative, and false negative values.
    """
    contingency = cluster.contingency_matrix(y_true, y_pred)
    same_class_true = np.max(contingency, 1)
    same_class_pred = np.max(contingency, 0)
    diff_class_true = contingency.sum(axis=1) - same_class_true
    diff_class_pred = contingency.sum(axis=0) - same_class_pred
    total = contingency.sum()

    true_positives = (same_class_true * (same_class_true - 1)).sum()
    false_positives = (diff_class_true * same_class_true * 2).sum()
    false_negatives = (diff_class_pred * same_class_pred * 2).sum()
    true_negatives = total * (
            total - 1) - true_positives - false_positives - false_negatives

    return true_positives, false_positives, false_negatives, true_negatives

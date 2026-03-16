"""
Evaluation metrics for Dwarpala.
Implements ACER, TAR@FAR, and fairness metrics.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import roc_curve, confusion_matrix


def compute_tar_at_far(
    genuine_scores: np.ndarray,
    impostor_scores: np.ndarray,
    target_far: float = 1e-4,
) -> Tuple[float, float]:
    """
    Compute True Accept Rate (TAR) at a specific False Accept Rate (FAR).

    Args:
        genuine_scores: Similarity scores for genuine pairs.
        impostor_scores: Similarity scores for impostor pairs.
        target_far: Target FAR (e.g., 1e-4 = 1 in 10,000).

    Returns:
        Tuple of (TAR, threshold at target FAR).
    """
    labels = np.concatenate([
        np.ones(len(genuine_scores)),
        np.zeros(len(impostor_scores)),
    ])
    scores = np.concatenate([genuine_scores, impostor_scores])

    fpr, tpr, thresholds = roc_curve(labels, scores)

    # Find threshold where FAR is closest to target
    idx = np.argmin(np.abs(fpr - target_far))
    tar = tpr[idx]
    threshold = thresholds[idx]

    return float(tar), float(threshold)


def compute_acer(
    predictions: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    """
    Compute Average Classification Error Rate (ACER).
    ACER = (APCER + BPCER) / 2

    Args:
        predictions: Binary predictions (0=spoof, 1=live).
        labels: Ground truth labels (0=spoof, 1=live).

    Returns:
        Dictionary with APCER, BPCER, ACER values.
    """
    # APCER: Attack Presentation Classification Error Rate
    # "How many fakes got in?" (False Accept of spoofs)
    spoof_mask = labels == 0
    if spoof_mask.sum() > 0:
        apcer = (predictions[spoof_mask] == 1).sum() / spoof_mask.sum()
    else:
        apcer = 0.0

    # BPCER: Bona Fide Presentation Classification Error Rate
    # "How many real humans were rejected?" (False Reject of genuine)
    genuine_mask = labels == 1
    if genuine_mask.sum() > 0:
        bpcer = (predictions[genuine_mask] == 0).sum() / genuine_mask.sum()
    else:
        bpcer = 0.0

    acer = (apcer + bpcer) / 2.0

    return {
        "apcer": float(apcer),
        "bpcer": float(bpcer),
        "acer": float(acer),
    }


def compute_demographic_parity(
    scores: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    """
    Compute FMR/FRR differential across demographic groups.

    Args:
        scores: Predicted similarity/liveness scores.
        labels: Ground truth labels.
        groups: Demographic group identifiers.
        threshold: Decision threshold.

    Returns:
        Dictionary with per-group FMR/FRR and max differential.
    """
    predictions = (scores >= threshold).astype(int)
    unique_groups = np.unique(groups)

    group_metrics = {}
    fmr_values = []
    frr_values = []

    for group in unique_groups:
        mask = groups == group
        group_preds = predictions[mask]
        group_labels = labels[mask]

        # FMR: False Match Rate (False Positives / Total Negatives)
        neg_mask = group_labels == 0
        if neg_mask.sum() > 0:
            fmr = (group_preds[neg_mask] == 1).sum() / neg_mask.sum()
        else:
            fmr = 0.0

        # FRR: False Reject Rate (False Negatives / Total Positives)
        pos_mask = group_labels == 1
        if pos_mask.sum() > 0:
            frr = (group_preds[pos_mask] == 0).sum() / pos_mask.sum()
        else:
            frr = 0.0

        group_metrics[str(group)] = {"fmr": float(fmr), "frr": float(frr)}
        fmr_values.append(fmr)
        frr_values.append(frr)

    # Max differential: ratio of highest to lowest (avoid div by zero)
    fmr_values = [v for v in fmr_values if v > 0]
    frr_values = [v for v in frr_values if v > 0]

    max_fmr_diff = max(fmr_values) / min(fmr_values) if len(fmr_values) >= 2 else 1.0
    max_frr_diff = max(frr_values) / min(frr_values) if len(frr_values) >= 2 else 1.0

    return {
        "per_group": group_metrics,
        "max_fmr_differential": float(max_fmr_diff),
        "max_frr_differential": float(max_frr_diff),
    }

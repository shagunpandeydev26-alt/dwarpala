"""
Pure benchmark metric helpers (no models, no I/O) so they can be unit-tested in CI.

- Verification (LFW): accuracy at best threshold, ROC AUC, EER, TAR@FAR.
- Liveness (PAD): APCER / BPCER / ACER per layer, reusing dwarpala.utils.metrics.
"""

from typing import Dict, List, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from dwarpala.utils.metrics import compute_acer, compute_tar_at_far


def verification_metrics(
    genuine_scores: Sequence[float],
    impostor_scores: Sequence[float],
    far_targets: Sequence[float] = (1e-2, 1e-3),
) -> Dict[str, object]:
    """
    Compute face-verification metrics from genuine/impostor similarity scores.

    Args:
        genuine_scores: cosine similarities for same-person pairs.
        impostor_scores: cosine similarities for different-person pairs.
        far_targets: FAR operating points to report TAR + threshold at.

    Returns:
        Dict with accuracy, best_threshold, roc_auc, eer, eer_threshold,
        tar_at_far (per target), and the pair counts.
    """
    genuine = np.asarray(genuine_scores, dtype=np.float64)
    impostor = np.asarray(impostor_scores, dtype=np.float64)
    if len(genuine) == 0 or len(impostor) == 0:
        raise ValueError("Need at least one genuine and one impostor score.")

    labels = np.concatenate([np.ones(len(genuine)), np.zeros(len(impostor))])
    scores = np.concatenate([genuine, impostor])

    fpr, tpr, thresholds = roc_curve(labels, scores)
    auc = float(roc_auc_score(labels, scores))

    # Best-accuracy threshold over the ROC sweep.
    n_pos, n_neg = len(genuine), len(impostor)
    accuracy = (tpr * n_pos + (1.0 - fpr) * n_neg) / (n_pos + n_neg)
    best_idx = int(np.argmax(accuracy))

    # Equal Error Rate: where FPR == FNR (1 - TPR).
    fnr = 1.0 - tpr
    eer_idx = int(np.argmin(np.abs(fpr - fnr)))
    eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)

    tar_at_far = {}
    for target in far_targets:
        tar, thr = compute_tar_at_far(genuine, impostor, target_far=target)
        tar_at_far[f"far={target:g}"] = {"tar": tar, "threshold": thr}

    return {
        "n_genuine": n_pos,
        "n_impostor": n_neg,
        "accuracy": float(accuracy[best_idx]),
        "best_threshold": float(thresholds[best_idx]),
        "roc_auc": auc,
        "eer": eer,
        "eer_threshold": float(thresholds[eer_idx]),
        "tar_at_far": tar_at_far,
    }


def liveness_layer_metrics(
    scores: Sequence[float],
    labels: Sequence[int],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    APCER / BPCER / ACER for one liveness signal at a decision threshold.

    Args:
        scores: liveness scores in [0,1] (higher = more live).
        labels: ground truth, 1=live (bona fide), 0=spoof (attack).
        threshold: score >= threshold is predicted live.

    Returns:
        Dict with apcer, bpcer, acer, threshold.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)
    predictions = (scores >= threshold).astype(int)
    out = compute_acer(predictions, labels)
    out["threshold"] = float(threshold)
    return out


def per_attack_breakdown(
    scores: Sequence[float],
    labels: Sequence[int],
    attack_types: Sequence[str],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    APCER per attack type (e.g. 'print', 'screen') — the fraction of each attack
    that was wrongly accepted as live. Bona fide rows are ignored here.

    Returns:
        Dict mapping attack_type -> APCER for that type.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)
    attack_types = np.asarray(attack_types, dtype=object)
    predictions = (scores >= threshold).astype(int)

    result: Dict[str, float] = {}
    for atype in sorted(set(a for a, lbl in zip(attack_types, labels) if lbl == 0)):
        mask = (attack_types == atype) & (labels == 0)
        if mask.sum() > 0:
            result[atype] = float((predictions[mask] == 1).sum() / mask.sum())
    return result


def confusion_by_attack(
    scores: Sequence[float],
    labels: Sequence[int],
    attack_types: Sequence[str],
    threshold: float = 0.5,
) -> Dict[str, Dict[str, int]]:
    """
    Raw confusion counts, the honest companion to rates on a small sample.

    For bona fide rows: how many were correctly ACCEPTED as live.
    For each attack type: how many were correctly REJECTED as spoof.

    Args:
        scores: liveness scores in [0,1] (>= threshold predicted live).
        labels: 1=live (bona fide), 0=spoof (attack).
        attack_types: attack label per sample ('bona_fide' for live rows).
        threshold: decision threshold.

    Returns:
        Dict keyed by 'bona_fide' and each attack type. Bona fide carries
        {'accepted', 'total'}; attacks carry {'rejected', 'total'}.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)
    attack_types = np.asarray(attack_types, dtype=object)
    pred_live = scores >= threshold

    out: Dict[str, Dict[str, int]] = {}
    live_mask = labels == 1
    if live_mask.sum() > 0:
        out["bona_fide"] = {
            "accepted": int(pred_live[live_mask].sum()),
            "total": int(live_mask.sum()),
        }
    for atype in sorted(set(a for a, lbl in zip(attack_types, labels) if lbl == 0)):
        mask = (attack_types == atype) & (labels == 0)
        out[atype] = {
            "rejected": int((~pred_live[mask]).sum()),
            "total": int(mask.sum()),
        }
    return out


def markdown_table(headers: Sequence[str], rows: List[Sequence[object]]) -> str:
    """Render a simple GitHub-flavored markdown table."""
    head = "| " + " | ".join(str(h) for h in headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = "\n".join("| " + " | ".join(str(c) for c in row) + " |" for row in rows)
    return "\n".join([head, sep, body])


def fmt_pct(x: float) -> str:
    """Format a [0,1] rate as a percentage string."""
    return f"{x * 100:.2f}%"

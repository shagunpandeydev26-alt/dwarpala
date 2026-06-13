"""
LFW face-verification benchmark.

Runs the FULL Kavach -> Swarupa inference path (SCRFD detect -> ArcFace-template
align -> buffalo_l R50 embedding -> cosine) over the standard LFW verification
pairs and reports verification accuracy, ROC AUC, EER, and TAR at fixed FAR
operating points. Plots the ROC curve to ``benchmarks/results/lfw_roc.png``.

Usage:
    python -m benchmarks.run_lfw                       # full `test` subset (1000 pairs)
    python -m benchmarks.run_lfw --max-pairs 200       # quick smoke
    python -m benchmarks.run_lfw --subset 10_folds     # standard 10-fold protocol

Notes:
- LFW is downloaded once by scikit-learn into ~/scikit_learn_data (~200MB) and
  cached thereafter; subsequent runs are offline. We load the full funneled
  250x250 images (no aggressive crop) so SCRFD has a real face to detect.
- Detection failures are counted, reported separately, and EXCLUDED from the
  model's accuracy/ROC. We ALSO report an all-pairs system accuracy where an
  undetected pair is treated as a rejection (non-match) — the honest end-to-end
  number — and explain the gap.
- If accuracy on detected pairs is far below ~99%, suspect the alignment
  template or detector, NOT the model. Investigate before reporting.
- Deterministic: pair order and embeddings are fixed; seeds are pinned so the
  table and plot regenerate identically.
"""

import argparse
import json
import random
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

from benchmarks.metrics import fmt_pct, markdown_table, verification_metrics
from dwarpala.kavach import FaceAligner, FaceDetector
from dwarpala.swarupa.embedding import EmbeddingExtractor
from dwarpala.utils.logger import get_logger

logger = get_logger("benchmarks.lfw")

_SEED = 1729
_RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    """LFW images come as float; normalize to uint8 RGB (H, W, 3)."""
    arr = np.asarray(image)
    if arr.ndim == 2:  # grayscale -> 3-channel
        arr = np.stack([arr] * 3, axis=-1)
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0 + 1e-6:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _embed(
    detector: FaceDetector,
    aligner: FaceAligner,
    extractor: EmbeddingExtractor,
    image: np.ndarray,
) -> Optional[np.ndarray]:
    """Detect -> align -> embed one RGB image. Returns None if no face found."""
    det = detector.detect_largest(image)
    if det is None:
        return None
    aligned = aligner.align(image, det.landmarks)
    return extractor.extract(aligned)


def _plot_roc(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc: float,
    eer: float,
    out_path: Path,
) -> None:
    """Render the ROC curve to a PNG (no GUI backend)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color="#1f77b4", lw=2, label=f"ROC (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], color="#999999", lw=1, ls="--", label="Chance")
    # EER reference line (FPR == FNR).
    ax.plot([0, 1], [1, 0], color="#d62728", lw=0.8, ls=":", label=f"EER = {eer * 100:.2f}%")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False Accept Rate (FAR)")
    ax.set_ylabel("True Accept Rate (TAR)")
    ax.set_title("Dwarpala LFW Verification ROC\n(SCRFD → buffalo_l R50, real pipeline path)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    logger.info(f"Wrote ROC plot to {out_path}")


def run(
    max_pairs: int = 0,
    subset: str = "test",
    out_path: Optional[str] = None,
    plot_path: Optional[str] = None,
) -> dict:
    """
    Run the LFW benchmark and return a results dict.

    Args:
        max_pairs: Number of pairs to score (0 = all in the subset).
        subset: LFW subset ('test', 'train', or '10_folds').
        out_path: JSON output path (default benchmarks/results/lfw.json).
        plot_path: ROC PNG path (default benchmarks/results/lfw_roc.png).

    Returns:
        Results dict with detection accounting and verification metrics.
    """
    random.seed(_SEED)
    np.random.seed(_SEED)

    from sklearn.datasets import fetch_lfw_pairs
    from sklearn.metrics import roc_curve

    logger.info(f"Loading LFW pairs (subset={subset})...")
    # Full funneled image (slice_=None, resize=1.0) so the detector sees a real face.
    lfw = fetch_lfw_pairs(subset=subset, color=True, resize=1.0, slice_=None, funneled=True)
    pairs, targets = lfw.pairs, lfw.target  # pairs: (N, 2, H, W, 3); target: 1=same
    n = min(max_pairs, len(pairs)) if max_pairs else len(pairs)
    logger.info(f"Scoring {n} of {len(pairs)} pairs (CPU path; this can take minutes).")

    detector = FaceDetector(backend="scrfd", max_faces=1)
    aligner = FaceAligner(output_size=(112, 112))
    extractor = EmbeddingExtractor(backend="insightface")

    genuine: List[float] = []
    impostor: List[float] = []
    # Detection-failure accounting, split by ground-truth label so we can model
    # the end-to-end (all-pairs) accuracy honestly.
    fail_genuine = 0
    fail_impostor = 0
    t0 = time.time()

    for i in range(n):
        img_a = _to_uint8_rgb(pairs[i][0])
        img_b = _to_uint8_rgb(pairs[i][1])
        emb_a = _embed(detector, aligner, extractor, img_a)
        emb_b = _embed(detector, aligner, extractor, img_b)
        if emb_a is None or emb_b is None:
            if targets[i] == 1:
                fail_genuine += 1
            else:
                fail_impostor += 1
            continue
        sim = float(np.dot(emb_a, emb_b))  # both L2-normalized -> cosine
        (genuine if targets[i] == 1 else impostor).append(sim)
        if (i + 1) % 50 == 0:
            logger.info(f"  {i + 1}/{n} pairs ({time.time() - t0:.0f}s)")

    detect_fail = fail_genuine + fail_impostor
    scored = len(genuine) + len(impostor)
    if scored == 0:
        raise RuntimeError("Every pair failed detection — pipeline/detector is broken.")

    metrics = verification_metrics(genuine, impostor, far_targets=(1e-2, 1e-3))
    best_thr = metrics["best_threshold"]

    # A target FAR finer than 1/n_impostor cannot be resolved by this split; the
    # ROC sweep reports threshold=inf / TAR=0 there. Mark it honestly rather than
    # printing a misleading "0.00%". (e.g. ~494 impostors -> finest FAR ~0.2%.)
    far_resolution = 1.0 / len(impostor)
    for key, point in metrics["tar_at_far"].items():
        target = float(key.split("=")[1])
        if not np.isfinite(point["threshold"]) or target < far_resolution:
            point["tar"] = None
            point["threshold"] = None
            point["measurable"] = False
        else:
            point["measurable"] = True

    # ── All-pairs (end-to-end) accuracy ──────────────────────────────────────
    # Policy: an undetected pair is a system REJECT (non-match). That makes an
    # undetected GENUINE pair a false reject (wrong) and an undetected IMPOSTOR
    # pair a correct reject. This is the honest end-to-end number; the "detected"
    # accuracy above isolates the model from detector recall.
    g = np.asarray(genuine)
    im = np.asarray(impostor)
    correct_detected = int((g >= best_thr).sum() + (im < best_thr).sum())
    all_correct = correct_detected + fail_impostor  # genuine fails count as wrong
    all_pairs_accuracy = all_correct / n

    # ── ROC for the plot (detected pairs) ────────────────────────────────────
    labels = np.concatenate([np.ones(len(genuine)), np.zeros(len(impostor))])
    scores = np.concatenate([g, im])
    fpr, tpr, _ = roc_curve(labels, scores)

    elapsed = time.time() - t0

    far01 = metrics["tar_at_far"]["far=0.001"]
    far1 = metrics["tar_at_far"]["far=0.01"]

    def _tar_cell(point: dict, far_label: str) -> str:
        if not point.get("measurable", True):
            return f"unmeasurable (need >{int(round(1 / far_resolution))} impostors)"
        return f"{fmt_pct(point['tar'])} (thr {point['threshold']:.4f})"

    result = {
        "dataset": f"LFW {subset}",
        "date": time.strftime("%Y-%m-%d"),
        "pairs_requested": n,
        "pairs_scored": scored,
        "detection_failures": detect_fail,
        "detection_failures_genuine": fail_genuine,
        "detection_failures_impostor": fail_impostor,
        "detection_rate": scored / n,
        "far_resolution": far_resolution,
        "mean_genuine_sim": float(np.mean(genuine)),
        "mean_impostor_sim": float(np.mean(impostor)),
        "accuracy_detected": metrics["accuracy"],
        "accuracy_all_pairs": all_pairs_accuracy,
        "elapsed_seconds": round(elapsed, 1),
        **metrics,
    }

    table = markdown_table(
        ["Metric", "Value"],
        [
            ["Pairs scored / requested", f"{scored} / {n}"],
            [
                "Detection failures",
                f"{detect_fail} ({fail_genuine} genuine, {fail_impostor} impostor)",
            ],
            ["Detection rate", fmt_pct(result["detection_rate"])],
            ["Accuracy (detected pairs)", fmt_pct(metrics["accuracy"])],
            ["Accuracy (all pairs, fail→reject)", fmt_pct(all_pairs_accuracy)],
            ["Best threshold (cosine)", f"{best_thr:.4f}"],
            ["ROC AUC", f"{metrics['roc_auc']:.4f}"],
            ["EER", fmt_pct(metrics["eer"])],
            ["Mean genuine similarity", f"{result['mean_genuine_sim']:.4f}"],
            ["Mean impostor similarity", f"{result['mean_impostor_sim']:.4f}"],
            ["TAR @ FAR=1%", _tar_cell(far1, "1%")],
            ["TAR @ FAR=0.1%", _tar_cell(far01, "0.1%")],
        ],
    )
    print("\n" + table + "\n")
    result["markdown_table"] = table

    plot_path = plot_path or str(_RESULTS_DIR / "lfw_roc.png")
    _plot_roc(fpr, tpr, metrics["roc_auc"], metrics["eer"], Path(plot_path))
    result["roc_plot"] = plot_path

    out_path = out_path or str(_RESULTS_DIR / "lfw.json")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Wrote results to {out_path}")

    return result


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="LFW face-verification benchmark.")
    p.add_argument("--max-pairs", type=int, default=0, help="Pairs to score (0 = all).")
    p.add_argument("--subset", default="test", choices=["train", "test", "10_folds"])
    p.add_argument("--out", default=None, help="JSON output path.")
    p.add_argument("--plot", default=None, help="ROC PNG output path.")
    args = p.parse_args(argv)
    run(max_pairs=args.max_pairs, subset=args.subset, out_path=args.out, plot_path=args.plot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

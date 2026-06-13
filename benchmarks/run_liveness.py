"""
Liveness (presentation-attack-detection) benchmark — per-layer vs fused.

Scores each labeled sample through the REAL pipeline liveness path
(``DwarpalaPipeline.liveness_only``, which feeds MiniFASNet BGR and texture RGB
correctly) and reports, for each independent liveness layer AND the fused gate:
APCER / BPCER / ACER at the operating threshold, plus RAW confusion counts and a
per-attack-type breakdown. The point is to show, honestly, where fusion catches
attacks a single layer misses — including cases where a layer fails.

Data layout (``--data-dir``):
    DIR/real/*.jpg                 # bona fide captures (label = live)
    DIR/spoof/<attack_type>/*.jpg  # attacks, sub-foldered by type (print/screen/...)
    DIR/spoof/*.jpg                # attacks of type "spoof" if not sub-foldered

With no ``--data-dir`` it falls back to the tiny built-in fixture set
(``tests/testimg``): a SMALL-SAMPLE (n=4) indicative demonstration, NOT a dataset
benchmark. See BENCHMARKS.md for the honest limitations.

IMPORTANT — temporal & rPPG layers: these need a VIDEO (multiple frames). On the
still-image fixture set they are NOT_APPLICABLE and are reported as N/A rather
than scored. Run with a video dataset (frames per sample) to evaluate them.

Usage:
    python -m benchmarks.run_liveness
    python -m benchmarks.run_liveness --data-dir /path/to/pad_set
"""

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from benchmarks.metrics import (
    confusion_by_attack,
    fmt_pct,
    liveness_layer_metrics,
    markdown_table,
)
from dwarpala.utils.logger import get_logger

logger = get_logger("benchmarks.liveness")

_SEED = 1729
_FIXTURE_DIR = Path(__file__).resolve().parents[1] / "tests" / "testimg"
_RESULTS_DIR = Path(__file__).resolve().parent / "results"
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _builtin_manifest() -> List[Tuple[Path, int, str]]:
    """Tiny default set from tests/testimg (label 1=live, 0=spoof; + attack type)."""
    return [
        (_FIXTURE_DIR / "selfie1.jpeg", 1, "bona_fide"),
        (_FIXTURE_DIR / "selfie2.jpeg", 1, "bona_fide"),
        (_FIXTURE_DIR / "printed.jpeg", 0, "print"),
        (_FIXTURE_DIR / "screencapture.jpeg", 0, "screen"),
    ]


def _scan_data_dir(data_dir: Path) -> List[Tuple[Path, int, str]]:
    """Build a manifest from DIR/real and DIR/spoof[/attack_type]."""
    manifest: List[Tuple[Path, int, str]] = []
    real_dir = data_dir / "real"
    spoof_dir = data_dir / "spoof"
    for p in sorted(real_dir.rglob("*")):
        if p.suffix.lower() in _IMG_EXTS:
            manifest.append((p, 1, "bona_fide"))
    for p in sorted(spoof_dir.rglob("*")):
        if p.suffix.lower() in _IMG_EXTS:
            # attack type = immediate sub-folder under spoof/, else "spoof"
            rel = p.relative_to(spoof_dir)
            attack = rel.parts[0] if len(rel.parts) > 1 else "spoof"
            manifest.append((p, 0, attack))
    return manifest


def _layer_block(
    name: str,
    records: List[dict],
    score_key: str,
    threshold: float,
) -> Optional[dict]:
    """
    Build the metrics block for one layer from per-sample records, using ONLY
    samples where that layer actually ran (score is not None). Returns None if
    the layer ran on nothing (e.g. temporal/rPPG on still images).
    """
    rows = [r for r in records if r[score_key] is not None]
    if not rows:
        return None
    scores = [r[score_key] for r in rows]
    labels = [r["label"] for r in rows]
    attacks = [r["attack"] for r in rows]
    m = liveness_layer_metrics(scores, labels, threshold=threshold)
    m["n_evaluated"] = len(rows)
    m["confusion"] = confusion_by_attack(scores, labels, attacks, threshold=threshold)
    return m


def run(data_dir: Optional[str] = None, out_path: Optional[str] = None) -> dict:
    """Run the liveness benchmark and return a results dict."""
    random.seed(_SEED)
    np.random.seed(_SEED)

    from dwarpala.yantra.pipeline import DwarpalaPipeline

    if data_dir:
        manifest = _scan_data_dir(Path(data_dir))
        source = data_dir
        is_fixture = False
    else:
        manifest = _builtin_manifest()
        source = "tests/testimg (built-in SMALL-SAMPLE smoke set, n=4)"
        is_fixture = True

    manifest = [(p, lbl, a) for (p, lbl, a) in manifest if p.exists()]
    if not manifest:
        raise FileNotFoundError(f"No images found for liveness benchmark (source={source}).")

    n_real = sum(1 for _, lbl, _ in manifest if lbl == 1)
    n_spoof = sum(1 for _, lbl, _ in manifest if lbl == 0)
    logger.info(
        f"Liveness set: {len(manifest)} images ({n_real} live, {n_spoof} spoof) from {source}"
    )

    pipeline = DwarpalaPipeline()
    threshold = pipeline.liveness_threshold  # config / pipeline default (0.5)

    records: List[dict] = []
    detect_fail = 0
    # Track whether the video-only layers ever ran (they cannot on stills).
    temporal_ran = False
    rppg_ran = False

    for path, label, attack in sorted(manifest, key=lambda r: str(r[0])):
        bgr = cv2.imread(str(path))
        if bgr is None:
            logger.warning(f"Unreadable image skipped: {path}")
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        result = pipeline.liveness_only(rgb)
        lv = result.liveness_verdict
        if lv is None:
            detect_fail += 1
            logger.warning(f"No face / liveness unavailable: {path.name}")
            continue
        mf = lv.minifas_result.score if lv.minifas_result else None
        tx = lv.texture_result.score if lv.texture_result else None
        tm = lv.temporal_result.score if lv.temporal_result else None
        rp = lv.rppg_result.score if lv.rppg_result else None
        temporal_ran = temporal_ran or tm is not None
        rppg_ran = rppg_ran or rp is not None
        records.append(
            {
                "name": path.name,
                "label": label,
                "attack": attack,
                "minifas": mf,
                "texture": tx,
                "temporal": tm,
                "rppg": rp,
                "fused": lv.score,
            }
        )
        logger.info(
            f"  {path.name:18s} label={'live' if label else 'spoof':5s} "
            f"minifas={mf if mf is None else round(mf, 3)} "
            f"texture={tx if tx is None else round(tx, 3)} "
            f"fused={lv.score:.3f}"
        )

    if not records:
        raise RuntimeError("No samples scored — every image failed face detection.")

    # ── Per-layer metric blocks (independent layers vs the fused gate) ────────
    layer_keys = [
        ("MiniFASNet", "minifas"),
        ("Texture (LBP+FFT)", "texture"),
        ("Temporal", "temporal"),
        ("rPPG", "rppg"),
        ("Fused gate", "fused"),
    ]
    layer_results: Dict[str, dict] = {}
    layer_rows = []
    for name, key in layer_keys:
        block = _layer_block(name, records, key, threshold)
        if block is None:
            layer_results[name] = {
                "status": "N/A",
                "reason": "needs video frames (still-image set)",
            }
            layer_rows.append([name, "N/A", "N/A", "N/A", "—"])
        else:
            layer_results[name] = block
            layer_rows.append(
                [
                    name,
                    fmt_pct(block["apcer"]),
                    fmt_pct(block["bpcer"]),
                    fmt_pct(block["acer"]),
                    str(block["n_evaluated"]),
                ]
            )

    rate_table = markdown_table(["Layer", "APCER ↓", "BPCER ↓", "ACER ↓", "n"], layer_rows)

    # ── Per-attack RAW confusion table (rejected/total per layer per type) ────
    attack_types = sorted(set(r["attack"] for r in records if r["label"] == 0))
    headers = ["Layer", "live accepted"] + [f"{a} rejected" for a in attack_types]
    count_rows = []
    for name, key in layer_keys:
        block = layer_results[name]
        if "confusion" not in block:
            count_rows.append([name] + ["N/A"] * (1 + len(attack_types)))
            continue
        conf = block["confusion"]
        bf = conf.get("bona_fide", {"accepted": 0, "total": 0})
        row = [name, f"{bf['accepted']}/{bf['total']}"]
        for a in attack_types:
            c = conf.get(a, {"rejected": 0, "total": 0})
            row.append(f"{c['rejected']}/{c['total']}")
        count_rows.append(row)
    count_table = markdown_table(headers, count_rows)

    print("\n=== APCER/BPCER/ACER per layer (threshold %.2f) ===\n" % threshold)
    print(rate_table)
    print("\n=== Raw confusion counts (correct decisions / total) ===\n")
    print(count_table)
    print()

    result = {
        "source": source,
        "is_small_sample_fixture": is_fixture,
        "date": time.strftime("%Y-%m-%d"),
        "n_images": len(records),
        "n_live": sum(1 for r in records if r["label"] == 1),
        "n_spoof": sum(1 for r in records if r["label"] == 0),
        "attack_types": attack_types,
        "detection_failures": detect_fail,
        "threshold": threshold,
        "temporal_evaluated": temporal_ran,
        "rppg_evaluated": rppg_ran,
        "per_sample": records,
        "layers": layer_results,
        "rate_table": rate_table,
        "count_table": count_table,
    }

    out_path = out_path or str(_RESULTS_DIR / "liveness.json")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Wrote results to {out_path}")
    return result


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Liveness (PAD) benchmark: per-layer vs fused.")
    p.add_argument("--data-dir", default=None, help="Dir with real/ and spoof/ subfolders.")
    p.add_argument("--out", default=None, help="JSON output path.")
    args = p.parse_args(argv)
    run(data_dir=args.data_dir, out_path=args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

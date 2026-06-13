# Dwarpala Development Progress

## Phase 0 — Audit & Environment (2026-06-13)

**Status: ✅ COMPLETE**

Read all 28 source files. Ran full test suite — 30/30 tests pass on Python 3.14.5. Created virtualenv with pinned dependencies. Installed insightface (1.0.1), fastapi (0.136.3), gradio (6.18.0), and all required packages. Key finding: the pipeline runs end-to-end but produces meaningless match scores because the ViT backbone uses ImageNet pretrained weights (not face-specific). Identified 4 minor bugs (walrus operator misuse in quality_assessor, logger handler removal issue, non-gated random fallback in backbone, README null bytes). Environment is clean and ready for Phase 1.

**Commit:** `phase0: codebase audit, environment setup, docs/AUDIT.md`

---

## Phase 1 — Real Embeddings (2026-06-13)

**Status: ✅ COMPLETE**

Added InsightFaceEmbedder wrapping buffalo_l ArcFace R50 (w600k_r50.onnx) via onnxruntime. Created ModelManager with SHA256 verification and insightface cache fallback. Rewrote EmbeddingExtractor with dual backend support (insightface default, vit for research). Gated random ViT fallback behind `allow_untrained=True`. Updated pipeline with separate ID quality thresholds (looser for low-res ID photos) and `to_dict()` for API serialization. Verified on LFW: same-person similarity 0.49–0.74, different-person -0.07 to +0.07. All 30 tests pass.

**Commit:** `phase1: real embeddings via InsightFace buffalo_l ArcFace R50`

---

## Phase 2 — MiniFASNet Liveness Integration (2026-06-13)

**Status: ✅ COMPLETE**

Created `prana/minifas_analyzer.py` — wraps MiniFASNetV2 (scale 2.7) and MiniFASNetV1SE (scale 4.0) pretrained models via PyTorch. These are lightweight CNNs (~600KB each) that analyze 80×80 face crops with their own bbox-scaling preprocessing (NOT the ArcFace 112×112 crop). Extended `fusion_gate.py` to accept 4 signals with default weights: minifas=0.40, texture=0.25, temporal=0.15, rppg=0.20. Added early-exit logic (texture OR MiniFASNet confident spoof → skip deeper checks), NOT_APPLICABLE renormalization (single-image mode skips temporal+rPPG), and rPPG override (valid heartbeat boosts score). Updated pipeline to pass original image + bbox for MiniFASNet cropping. Added model registry entries with download URLs. 13 new tests covering MiniFASNet preprocessing, no-model fallback, 4-signal fusion, single-image mode, weight renormalization, and early-exit. All 43 tests pass.

**Commit:** `phase2: integrate MiniFASNet into Prana fusion gate`

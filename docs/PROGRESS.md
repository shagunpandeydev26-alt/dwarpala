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

**Status: ✅ COMPLETE — liveness now discriminates live vs spoof**

Created `prana/minifas_analyzer.py` — wraps MiniFASNetV2 (scale 2.7) and MiniFASNetV1SE (scale 4.0) pretrained models via PyTorch. These are lightweight CNNs (~1.8MB each) that analyze 80×80 face crops with bbox-scaling preprocessing (NOT the ArcFace 112×112 crop). Extended `fusion_gate.py` for 4-signal fusion with weights: minifas=0.40, texture=0.25, temporal=0.15, rppg=0.20. Added early-exit, NOT_APPLICABLE renormalization, and rPPG override.

### Bug fix (the "class-2 pinned for every image" failure)

The earlier integration produced class-2-dominant output for **every** image, real or spoof, and was wrongly attributed to "weight quality." It was an implementation bug, in two parts, both now fixed against the minivision reference source:

1. **Input normalization (root cause).** The reference `src/data_io/functional.py::to_tensor` has its `.div(255)` **commented out** — the pretrained weights expect RAW `[0,255]` BGR pixels, with no normalization. The integration was dividing by 255 (and an earlier version used `(x−127.5)/128`). Feeding `[0,1]` collapsed every activation and saturated the 3-class head to one class regardless of input. Fix: feed `float32` pixels in `[0,255]`, no division, no BGR→RGB swap.

2. **Architecture (hand-rebuilt → faithful port).** The previous net was reverse-engineered from state-dict shapes and diverged from the reference in ways that load cleanly (shapes match) but compute the wrong thing: `conv1` used stride 1 (should be 2); `conv_45` used stride 1 (should be 2); `conv_6_dw` had a PReLU and padding 2 (should be a `Linear_block` — Conv+BN only — kernel (5,5), padding 0); a global-average-pool replaced the reference `Flatten` of the 1×1 map; the SE block was missing its inter-FC `ReLU`; and `linear`/`prob` had stray biases not present in the weights. `prana/minifas_analyzer.py` is now a faithful port of `src/model_lib/MiniFASNet.py`, loading with **strict=True, 0 missing / 0 unexpected** for both models. Crop now matches the reference `CropImage` (scale clamp + box **shift**, not clip). Channel order confirmed BGR (reference keeps cv2 order; ToTensor does not swap).

Validated against the official minivision repo run on the same images (identical `.pth` SHA256): our outputs match its live/spoof verdicts.

### Acceptance results (tests/testimg/ — 2 selfies, 1 print, 1 screen cap)

Summed-softmax argmax and class-1 (live) probability, with the production insightface bbox:

| image | argmax | live score | verdict |
|---|---|---|---|
| selfie1 | 1 | 0.999 | LIVE |
| selfie2 | 1 | 1.000 | LIVE |
| printed | 2 | 0.049 | SPOOF |
| screencapture | 2 | 0.305 | SPOOF |

Both selfies land on class 1 (live), both spoofs on class 2 (fake) — clean separation at threshold 0.5.

- 47 tests pass (43 existing + 4 acceptance tests; `test_preprocess` updated to assert the correct `[0,255]` range).

---

## Phase 3 — FastAPI REST Server (2026-06-13)

**Status: ✅ COMPLETE**

Built `dwarpala/api/` exposing the existing `DwarpalaPipeline` over REST. The API is a thin transport layer — all detection, embedding, matching, and liveness fusion stay in the pipeline (single source of truth shared with the Phase 4 Gradio demo). No parallel verification logic was written.

**Endpoints** (OpenAPI docs at `/docs`):
- `POST /verify` — multipart `id_image` (image) + `selfie` (image or video) → full result schema (verdict, match_score, liveness_score, liveness_breakdown, signal_status, quality{id,selfie}, explanation, latency_ms, request_id).
- `POST /liveness` — `selfie` only (image/video) → liveness verdict + breakdown.
- `POST /match` — two images → match result only (no liveness).
- `GET /health` — 200 only when models are loaded (model versions + uptime); 503 envelope before load.

**Engineering:**
- Models load **once** at startup via a FastAPI lifespan handler. `create_app(pipeline_factory=...)` lets tests inject a mock so CI never downloads weights.
- Inference is serialized behind a single process-wide lock (`app.state.lock`); endpoints are sync `def` so FastAPI runs them in its threadpool and the event loop is never blocked. (Documented as the conservative default; a per-worker pipeline pool can replace it later.)
- Validation: 415 on non-image/video MIME, 413 on >15 MB (read-capped, never buffers oversize files), 422 on bytes that don't decode as image/video. Images decoded to RGB ndarrays; videos written to a temp file so the pipeline does frame extraction (one source of truth), then cleaned up.
- Consistent error envelope `{error:{code,message,request_id}}` for all 4xx/5xx; full stack traces logged server-side only, never returned to clients.
- Structured logging only (request_id, verdict, scores, latency, file sizes, MIME) — no image bytes (enforced by a regression test that greps captured logs).
- Optional SQLite audit log (`AUDIT_LOG=sqlite`), off by default, stores results only (no images); schema documented in `api/audit.py`.
- CORS permissive in dev, configurable, with a lock-it-down warning for production.
- All settings (host/port/upload cap/CORS/audit/model_dir) come from `configs/inference_config.yaml` (`api:` block) with env-var overrides — nothing hardcoded.

**CLI:** added `dwarpala/cli.py` with `serve` (config/env-driven, CLI flags win) and `download-models`; registered `dwarpala` console script in `pyproject.toml`.

**Pipeline changes (additive only — verify()/fusion algorithm untouched):**
- Added `match_only()` and `liveness_only()` methods that reuse the same detector/aligner/extractor/matcher/fusion-gate components as `verify()`, so `/match` and `/liveness` share one code path. New lightweight `MatchOnlyResult` / `LivenessOnlyResult` dataclasses.
- Extended `VerificationResult.to_dict()` with `signal_status`, structured `quality`, and a stable 4-key `liveness_breakdown` (null for signals that didn't run). `verify()` now also stores `*_quality_report` dicts in `details`. Added `QualityReport.to_dict()`.

**Bugs fixed (pre-existing, broken code per the master brief):**
- `VerificationResult.__str__` had a malformed f-string (`{score:.4f if ... else 'N/A'}`) that raised on every full `verify()` that reached its final log line, turning real ACCEPTs into `System error` REJECTs. It was latent because earlier code paths rejected before that line; the live `/verify` smoke surfaced it. Fixed by computing the value before the f-string.
- `quality_assessor._check_occlusion` walrus misuse (`if image_ndim := face_image.ndim == 3`, AUDIT bug #1) and an unused `cv2` import in `pipeline.py` — fixed to keep ruff clean.

**Acceptance verified:** `dwarpala serve` boots, loads models once, `/health`→200; live `curl /verify` on the two real selfies → `ACCEPT` (match 0.85, liveness 0.92, temporal/rppg `NOT_APPLICABLE`) in ~300 ms (< 3 s); `/liveness` → LIVE 0.92 (selfie) / SPOOF 0.41 (print); `/match` → MATCH 0.85; 415/413/422 and the error envelope confirmed; audit rows written; no image bytes in logs.

- 61 tests pass (47 prior + 14 new API tests). Endpoint tests run with a mocked pipeline (no model downloads); `requires_models` covers a real `/verify` integration test and the MiniFASNet `[0,255]` normalization regression guard (selfies > 0.5 live, spoofs < 0.5).

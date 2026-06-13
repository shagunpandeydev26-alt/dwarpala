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

---

## Pre-Phase 4 — Default detector → SCRFD with Haar fallback (2026-06-13)

Default `detector_backend` changed from OpenCV Haar to InsightFace **SCRFD** (pipeline constructor default + `configs/inference_config.yaml`; the YAML `kavach` block is not yet wired at runtime, so the constructor default is the effective one). Haar remains an explicit graceful fallback: if SCRFD/InsightFace fails to load, the detector falls back to Haar and logs a loud `DETECTION QUALITY IS DEGRADED` warning (never silent). **Landmark-order check confirmed:** SCRFD's `face.kps` order is byte-identical to the aligner's `ARCFACE_REFERENCE_112` (left eye, right eye, nose, left mouth, right mouth) — same as insightface's canonical `arcface_dst` template — so no remapping is needed and embeddings are unaffected. MiniFASNet still receives the original-image bbox (x,y,w,h) unchanged; liveness scores held within tolerance (regression guard green, integration `/verify` ACCEPTs with liveness ~0.92). The integration test now drives the positive ACCEPT path (matching selfie pair) instead of asserting only a "sane verdict". Added a fallback unit test (forces SCRFD unavailable → asserts Haar fallback + degradation warning). 62 tests pass; ruff + black clean.

---

## Phase 4 — Gradio Demo UI (2026-06-13)

**Status: ✅ COMPLETE**

Built `dwarpala/ui/app.py` — a two-tab Gradio demo (Verify + Liveness Lab) that is pure presentation over the proven pipeline. The callbacks call the SAME `DwarpalaPipeline.verify` / `liveness_only` methods the REST API uses, in-process (one verification code path across API + UI). No verification/matching/liveness logic lives in the UI; the display helpers (verdict colors, score bars, breakdown rows, rPPG figure) are pure functions, unit-tested directly with synthetic inputs.

### 🔴 Critical bug found and fixed: RGB→BGR seam to MiniFASNet (security-relevant)

While prepping the RGB/BGR parity guard, found that the pipeline feeds MiniFASNet the wrong channel order. The pipeline is RGB-native and the fusion gate forwarded `original_image` (RGB) straight to `MiniFASAnalyzer` (which expects **BGR**, the order its weights were trained on). Effect was severe and asymmetric: a screen-replay spoof scored **0.95 (LIVE)** instead of **0.30 (SPOOF)** — a spoof passing as live through the real pipeline. (MiniFASNet was validated standalone in Phase 2 with `cv2.imread` BGR, so the bug was invisible until traced through the pipeline.) **Fix:** `fusion_gate.analyze` now converts `original_image` RGB→BGR before the MiniFASNet call only (texture/temporal/rPPG stay RGB) — input marshaling, not scoring logic. After the fix, pipeline MiniFASNet scores match the validated BGR values exactly (selfie 0.999, print 0.049, screencapture 0.305). Guarded by two `requires_models` tests: an RGB-array-vs-file parity test and a screencapture-spoof regression.

### Tabs (every number/plot is real pipeline output)
- **Verify:** ID + selfie (upload/webcam) → `pipeline.verify`; colored verdict banner (green ACCEPT / red REJECT / amber MANUAL_REVIEW), match + liveness score bars, a 4-layer liveness breakdown table (minifas/texture/temporal/rppg, each with score + signal_status), the pipeline's `explanation`, and latency. Single-image selfies show temporal+rPPG as **NOT_APPLICABLE** (never a misleading 0). No-face/quality REJECTs render the pipeline's reason without crashing.
- **Liveness Lab:** webcam/uploaded video → `pipeline.liveness_only`; a **real matplotlib rPPG waveform** with estimated BPM when a confident heartbeat exists, otherwise an honest "insufficient signal — need ≥5s stable video" message (gated on the analyzer's `has_valid_heartbeat`, never a fabricated trace); the 4-layer breakdown (weak rPPG surfaced as **LOW_CONFIDENCE**); and a real **FFT texture map** artifact.

### Additive analyzer/pipeline accessors (no scoring changes)
- `TextureAnalyzer.get_fft_spectrum()` — read-only; recomputes the exact log-magnitude FFT the FFT-liveness score already uses, for the texture viz.
- `DwarpalaPipeline.load_selfie_frames()` — read-only; returns the RGB frame list via the existing `_load_selfie` handling so the UI extracts frames once and feeds the same list to `liveness_only` and `rppg_analyzer.get_rppg_waveform` (the plotted pulse is exactly the scored signal). rPPG already exposed `get_rppg_waveform()` (used as-is).
- Also removed a pre-existing dead assignment in `texture_analyzer._compute_lbp` to keep ruff clean.

### Engineering
- Pipeline loaded once in `build_demo` and closed over by callbacks (no per-interaction reload); `pipeline_factory` injection lets tests mock it.
- RGB/BGR discipline: UI passes gradio RGB arrays straight to the RGB-native pipeline; a `requires_models` parity test asserts file-load vs gradio-array give identical scores.
- `demo` CLI subcommand + `demo:` config block (host/port/share; `share=False` by default with a third-party-tunnel warning). matplotlib added to the `demo` extra.

End-to-end smoke (real pipeline): Tab 1 → ACCEPT in ~740 ms with real breakdown; Tab 2 on a 5s clip → temporal OK, rPPG LOW_CONFIDENCE with the honest plot message, real FFT map. 73 tests pass (62 prior + 11 UI); ruff + black clean.

---

## Phase 5 — Benchmarks + `BENCHMARKS.md` (2026-06-13)

**Status: ✅ COMPLETE — honest numbers, no pipeline/scoring changes.**

Two reproducible benchmark scripts (`benchmarks/run_lfw.py`, `benchmarks/run_liveness.py`) over the **real** product pipeline, a pure-metrics module (`benchmarks/metrics.py`) unit-tested in CI, and an honest `BENCHMARKS.md`. Benchmarks are `requires_models`/network and do **not** run in CI; results (JSON + ROC plot) committed under `benchmarks/results/`, raw datasets git-ignored (LFW lives in `~/scikit_learn_data`). Seeds pinned for determinism.

### Face matching — LFW (sklearn `fetch_lfw_pairs`, DevTest, funneled, 989/1000 scored)
- **97.98% accuracy on detected pairs**, AUC **0.9801**, EER **3.64%**, TAR@FAR=1% **95.93%** (thr 0.162); mean genuine 0.631 vs impostor 0.003. 11 detection failures (8 genuine, 3 impostor) counted separately → all-pairs accuracy 97.20% (the 0.78 pp gap is exactly the 8 undetected genuine pairs forced to reject). ROC plotted to `results/lfw_roc.png`.
- TAR@FAR=0.1% reported as **unmeasurable** (only ~497 impostors → finest resolvable FAR ≈ 0.2%), not a misleading 0%.
- **Investigated the sub-99% (per the brief's guard-rail) before reporting.** Read-only diagnostics, no product code touched: (1) channel order/normalization is a non-factor — current RGB→BGR, RGB÷127.5, RGB÷128 all give AUC ≈ 0.982; (2) our pipeline **matches the reference** — on identical 200 pairs, ours AUC 0.963 vs insightface's native `FaceAnalysis` 0.958. Conclusion: the gap from buffalo_l's published ~99.7% is the **evaluation protocol** (funneled DevTest + re-detection), not a Dwarpala alignment/embedding defect.

### Liveness — per-layer vs fused (FALLBACK: `tests/testimg`, n=4 stills, clearly marked indicative)
- **No PAD dataset was obtainable headlessly** (CelebA-Spoof is Drive/Baidu-gated ~70 GB; OULU-NPU/SiW/CASIA need institutional access). Fell back to the 4 fixtures (2 live, 1 print, 1 screen) — reported with raw confusion counts, since n=4 rates are meaningless alone. **Temporal & rPPG are N/A on stills (need video) — reported as N/A, not fabricated.**
- Headline finding (honest, including a layer that FAILS): **Texture alone APCER 100%** (accepts both spoofs: print 1.000, screen 0.793), **MiniFASNet alone 0%**, **Fused gate 0%** — fusion is *robust to the failed texture layer*. ⚠️ thin margin: screen-replay clears the gate by only 0.008 (fused 0.492 vs 0.500) — surfaced as a fragility, not hidden.

### Threshold calibration — measured, then deliberately NOT changed
- Default match threshold 0.45 is conservative on LFW (FRR 8.5% @ FAR 0%; max impostor sim 0.145). **Declined to recalibrate**: the ~165–497 impostor splits can't validate FAR at the 0.1% a KYC threshold needs, and LFW selfie-selfie is easier than production ID↔selfie — lowering a security threshold on that evidence would be dishonest. Liveness threshold likewise unchanged (n=4 far too small). No calibration commit; rationale documented in `BENCHMARKS.md §2.2`.

`BENCHMARKS.md` carries full methodology (hardware/Python 3.14.5/lib + model versions + SHA-256), both result tables, the ROC plot, a prominent Limitations section (small liveness n; temporal/rPPG unmeasured; rPPG webcam unreliability per Phase 4; LFW 1:1≠1:N and FAR-0.1% unresolvable; buffalo_l research license; no iBeta/ISO 30107 PAD), and an honest verdict. 9 new CI-safe metrics tests (82 total); ruff + black clean. No pipeline, fusion, or scoring logic was modified in this phase.

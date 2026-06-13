# Dwarpala Phase 0 — Codebase Audit

> Audit date: 2026-06-13
> Auditor: Automated Phase 0 analysis
> Commit: 53131c0f (initial clone)

## Environment

| Component | Version |
|---|---|
| Python | 3.14.5 |
| pip | 26.1.2 |
| torch | 2.12.0+cu130 |
| torchvision | 0.27.0 |
| timm | 1.0.27 |
| numpy | 2.4.6 |
| opencv-python | 4.13.0.92 |
| scipy | 1.17.1 |
| scikit-learn | 1.9.0 |
| onnxruntime | 1.26.0 |
| insightface | 1.0.1 |
| loguru | 0.7.3 |
| omegaconf | 2.3.1 |
| fastapi | 0.136.3 |
| gradio | 6.18.0 |
| uvicorn | 0.49.0 |
| pytest | 9.0.3 |
| ruff | 0.15.17 |
| black | 26.5.1 |

**Note:** numpy 2.4.6 installed without issues alongside insightface 1.0.1 (newer insightface no longer requires numpy<2).

## Test Suite Status

**All 30 existing tests PASS (3.25s)**

```
tests/test_kavach.py    — 12 tests — ALL PASS
tests/test_prana.py     — 10 tests — ALL PASS
tests/test_swarupa.py   —  8 tests — ALL PASS
```

No test failures, no import errors, no deprecation warnings blocking execution.

---

## Module-by-Module Status

### ✅ Kavach (Pre-processing) — FULLY FUNCTIONAL

| File | Status | Notes |
|---|---|---|
| `face_detector.py` | ✅ Works | OpenCV Haar fallback works. InsightFace SCRFD also loads correctly with insightface 1.0.1. Landmark estimation from Haar is geometric approximation (adequate for fallback). |
| `face_aligner.py` | ✅ Works | Uses correct ArcFace 112×112 reference landmarks `[[38.2946,51.6963],...]`. Umeyama similarity transform implementation is correct. |
| `quality_assessor.py` | ✅ Works | Blur (Laplacian variance), brightness, face size, confidence, and occlusion checks all functional. **Issue:** `_check_occlusion` line 165 has a walrus operator bug: `if image_ndim := face_image.ndim == 3:` — this assigns `True/False` to `image_ndim`, not the ndim value. However, this doesn't affect functionality because the gray conversion still happens correctly (the condition evaluates to True for 3-channel images). Minor cosmetic bug. |

### ⚠️ Swarupa (Identity) — STRUCTURALLY SOUND, FUNCTIONALLY USELESS

| File | Status | Notes |
|---|---|---|
| `backbone.py` | ⚠️ Loads but produces random embeddings for face tasks | Uses `timm.create_model("vit_base_patch16_224", pretrained=True)` — downloads 346MB ImageNet weights. These are NOT trained for face recognition. The `_fallback_features` returns `torch.randn()` — literally random. |
| `embedding.py` | ⚠️ Works mechanically, outputs meaningless | `EmbeddingExtractor` preprocesses 112×112 → resize to 224×224 → normalize → ViT forward. Produces 512-D L2-normalized vectors. But since the backbone isn't face-trained, embeddings have no discriminative power for faces. |
| `arcface_loss.py` | ✅ Code correct | Sub-Center ArcFace with K=3 implemented correctly. Not used at inference — training-only. |
| `siamese_net.py` | ✅ Code correct | Siamese wrapper around ViT backbone. Not used at inference — training-only. |
| `matcher.py` | ✅ Works | Cosine similarity + thresholds + review band. Purely mathematical, works correctly regardless of embedding quality. |

**Critical finding:** The default inference path uses ImageNet ViT weights. Same-person similarity will be essentially random. This is the #1 problem to fix (Phase 1: InsightFace buffalo_l).

### ✅ Prana (Liveness) — FUNCTIONAL, NEEDS MiniFASNet

| File | Status | Notes |
|---|---|---|
| `texture_analyzer.py` | ✅ Works | Custom LBP implementation (not using skimage). Multi-scale radii [1,2,3]. FFT periodic peak detection. Runs correctly on any 112×112 face crop. |
| `temporal_analyzer.py` | ✅ Works | Micro-saccade detection from landmark displacement. Tremor analysis via Welch PSD. Falls back to optical flow when landmarks unavailable. Returns "uncertain" for <15 frames. |
| `rppg_analyzer.py` | ✅ Works | CHROM method correctly implemented. Butterworth bandpass 0.7–4.0 Hz. FFT heart rate estimation. Returns "uncertain" for <90 frames (3s at 30fps). |
| `fusion_gate.py` | ✅ Works | 3-signal weighted fusion (0.35/0.30/0.35). Early exit on confident spoof. Weight renormalization when signals missing. **Missing:** MiniFASNet as 4th signal (Phase 2). |

### ✅ Dharma (Fairness) — FUNCTIONAL

| File | Status | Notes |
|---|---|---|
| `bias_auditor.py` | ✅ Works | FMR/FRR differential computation, sample imbalance detection. |
| `demographic_classifier.py` | ✅ Works | MLP on embeddings → age/gender/ethnicity. Uses random weights (no trained model). |
| `parity_loss.py` | ✅ Works | Training-time regularization. Not used at inference. |

### ✅ Yantra (Pipeline) — FUNCTIONAL

| File | Status | Notes |
|---|---|---|
| `pipeline.py` | ✅ Works end-to-end | 7-step pipeline: load → detect → quality → align → embed → liveness → decision. Tested: creates pipeline, processes images, returns VerificationResult. Rejects "no face" correctly. Handles video files and frame lists. |

### ✅ Utils — FUNCTIONAL

| File | Status | Notes |
|---|---|---|
| `image_utils.py` | ✅ Works | load_image, resize, normalize, crop_face_roi, blur/brightness computation. |
| `video_utils.py` | ✅ Works | read_video_frames, frames_to_rgb, capture_webcam_frames. |
| `logger.py` | ⚠️ Works but has a design issue | Every call to `get_logger()` calls `logger.remove()` which removes ALL handlers, then adds a new one. If multiple modules call it, earlier handlers get removed. In practice this works because loguru's global logger still receives all messages. |
| `metrics.py` | ✅ Works | TAR@FAR, ACER, demographic parity computation. |

### Demo

| File | Status | Notes |
|---|---|---|
| `demo/webcam_demo.py` | ✅ Works | OpenCV webcam demo with fancy bounding boxes. Requires display. Only runs texture analysis on snapshot — doesn't run full pipeline. |

### Configs

| File | Status | Notes |
|---|---|---|
| `configs/model_config.yaml` | ✅ Valid YAML | All parameters reasonable. Not loaded by any code currently (pipeline uses hardcoded defaults). |
| `configs/inference_config.yaml` | ✅ Valid YAML | Same — exists but not loaded. |

---

## What's Missing (Required for v1.0)

| Gap | Priority | Phase |
|---|---|---|
| Real face recognition model (InsightFace buffalo_l ONNX) | 🔴 Critical | Phase 1 |
| InsightFaceEmbedder implementing EmbeddingExtractor interface | 🔴 Critical | Phase 1 |
| MiniFASNet liveness model integration | 🔴 Critical | Phase 2 |
| 4-signal fusion gate (texture + temporal + rppg + minifas) | 🔴 Critical | Phase 2 |
| Single-image liveness mode (no video → skip temporal + rppg) | 🔴 Critical | Phase 2 |
| FastAPI REST server (`/verify`, `/liveness`, `/match`, `/health`) | 🔴 Critical | Phase 3 |
| Gradio web demo (Verify tab + Liveness Lab tab) | 🟡 High | Phase 4 |
| LFW benchmark script | 🟡 High | Phase 5 |
| Liveness benchmark script | 🟡 High | Phase 5 |
| BENCHMARKS.md with real numbers | 🟡 High | Phase 5 |
| Dockerfile + docker-compose.yml | 🟡 High | Phase 6 |
| GitHub Actions CI | 🟡 High | Phase 6 |
| CLI (download-models, verify, demo, serve) | 🟡 High | Phase 6 |
| README overhaul | 🟡 High | Phase 6 |
| LICENSES_THIRD_PARTY.md | 🟡 High | Phase 6 |
| Model download system with SHA256 checks | 🟡 High | Phase 1 |
| Separate quality thresholds for ID vs selfie | 🟢 Medium | Phase 1 |
| Config loading from YAML (currently hardcoded defaults) | 🟢 Medium | Phase 1 |

---

## Known Bugs Found

1. **`quality_assessor.py:165`** — Walrus operator misuse: `if image_ndim := face_image.ndim == 3:` assigns boolean, not ndim. Cosmetic, doesn't affect behavior.

2. **`logger.py`** — `logger.remove()` called on every `get_logger()` invocation removes all previously added handlers. Should only configure once.

3. **`backbone.py:123`** — `_fallback_features` returns `torch.randn()` — non-deterministic random noise. When `allow_untrained=True` is not explicitly gated, this silently produces garbage in the inference path if timm fails to import.

4. **`README.md:153-155`** — Contains null-byte garbage characters at the end of the file.

---

## Conclusion

The codebase is **architecturally well-designed** and **mechanically functional**. All 30 existing tests pass. The pipeline runs end-to-end without crashes. The single critical gap is that **no real face recognition model is in the inference path** — the ViT backbone uses ImageNet weights, making identity matching non-functional. This is exactly what Phase 1 addresses.

The environment installs cleanly on Python 3.14 with all dependencies. No numpy compatibility issues found (insightface 1.0.1 works with numpy 2.x).

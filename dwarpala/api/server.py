"""
Dwarpala FastAPI REST server (Phase 3).

Thin transport layer over ``DwarpalaPipeline``. All verification, matching, and
liveness computation happens in the pipeline (single source of truth shared with
the Gradio demo); this module only handles HTTP concerns: upload validation,
serialization, error envelopes, structured logging, and the optional audit log.

Thread-safety: the pipeline mixes PyTorch + onnxruntime + OpenCV state. To stay
unambiguously safe we serialize all inference behind a single process-wide lock
(``app.state.lock``). Endpoints are sync ``def`` so FastAPI runs them in its
worker threadpool — the event loop is never blocked, and concurrent requests
queue on the lock. For higher throughput this can later be swapped for a small
pool of per-worker pipeline instances; the lock is the conservative default.
"""

import os
import time
import uuid
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from typing import Callable, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from dwarpala.api.audit import maybe_create_audit
from dwarpala.api.config import APISettings, load_api_settings
from dwarpala.api.schemas import (
    HealthResponse,
    LivenessResponse,
    MatchResponse,
    VerifyResponse,
)
from dwarpala.utils.logger import get_logger

logger = get_logger("api.server")

_VIDEO_SUFFIXES = (".mp4", ".avi", ".mov", ".webm")


class APIError(Exception):
    """Error that maps to a clean error-envelope response."""

    def __init__(self, status: int, code: str, message: str):
        self.status = status
        self.code = code
        self.message = message
        super().__init__(message)


# ──────────────────────────────────────────────────────────────────────────
# Upload validation helpers (no image bytes are ever logged)
# ──────────────────────────────────────────────────────────────────────────
def _read_capped(upload: UploadFile, limit: int) -> bytes:
    """Read at most ``limit`` bytes; raise 413 if the file exceeds the cap."""
    f = upload.file
    f.seek(0)
    data = f.read(limit + 1)
    if len(data) > limit:
        raise APIError(
            413,
            "PAYLOAD_TOO_LARGE",
            f"File '{upload.filename}' exceeds the {limit // (1024 * 1024)} MB limit.",
        )
    return data


def _classify_mime(upload: UploadFile, allow_video: bool) -> str:
    """Return 'image' or 'video', or raise 415 for anything else."""
    ct = (upload.content_type or "").lower()
    if ct.startswith("image/"):
        return "image"
    if allow_video and ct.startswith("video/"):
        return "video"
    raise APIError(
        415,
        "UNSUPPORTED_MEDIA_TYPE",
        f"Unsupported content type '{ct or 'unknown'}' for '{upload.filename}'. "
        f"Expected an image{' or video' if allow_video else ''}.",
    )


def _decode_image_rgb(data: bytes, filename: Optional[str]) -> np.ndarray:
    """Decode raw bytes to an RGB ndarray, or raise 422 if undecodable."""
    arr = np.frombuffer(data, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise APIError(
            422,
            "UNDECODABLE_IMAGE",
            f"Uploaded file '{filename}' could not be decoded as an image.",
        )
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _save_temp_video(data: bytes, filename: Optional[str]) -> str:
    """Persist video bytes to a temp file and confirm at least one frame decodes."""
    suffix = Path(filename or "").suffix.lower()
    if suffix not in _VIDEO_SUFFIXES:
        suffix = ".mp4"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        tmp.write(data)
        tmp.flush()
    finally:
        tmp.close()

    cap = cv2.VideoCapture(tmp.name)
    ok, _ = cap.read()
    cap.release()
    if not ok:
        os.unlink(tmp.name)
        raise APIError(
            422,
            "UNDECODABLE_VIDEO",
            f"Uploaded file '{filename}' could not be decoded as a video.",
        )
    return tmp.name


def _prepare_image(upload: UploadFile, settings: APISettings):
    """Validate + decode an image-only upload. Returns (rgb_array, size_bytes)."""
    _classify_mime(upload, allow_video=False)
    data = _read_capped(upload, settings.max_upload_bytes)
    return _decode_image_rgb(data, upload.filename), len(data)


def _prepare_selfie(upload: UploadFile, settings: APISettings):
    """
    Validate + decode a selfie upload that may be image OR video.

    Returns (selfie_input, temp_path_or_None, size_bytes) where selfie_input is
    an RGB ndarray (image) or a filesystem path (video, for the pipeline to read
    frames). temp_path must be unlinked by the caller when not None.
    """
    kind = _classify_mime(upload, allow_video=True)
    data = _read_capped(upload, settings.max_upload_bytes)
    if kind == "image":
        return _decode_image_rgb(data, upload.filename), None, len(data)
    path = _save_temp_video(data, upload.filename)
    return path, path, len(data)


# ──────────────────────────────────────────────────────────────────────────
# Response shaping (reads pipeline result objects; schema lives in pipeline)
# ──────────────────────────────────────────────────────────────────────────
def _verify_payload(result, request_id: str) -> dict:
    d = result.to_dict()
    return {
        "request_id": request_id,
        "verdict": d["verdict"],
        "match_score": d.get("match_score", 0.0),
        "liveness_score": d.get("liveness_score"),
        "liveness_breakdown": d.get(
            "liveness_breakdown",
            {"minifas": None, "texture": None, "temporal": None, "rppg": None},
        ),
        "signal_status": d.get("signal_status", {}),
        "quality": d.get("quality", {"id": None, "selfie": None}),
        "explanation": d["explanation"],
        "latency_ms": d["latency_ms"],
    }


def _liveness_payload(result, request_id: str) -> dict:
    d = result.to_dict()
    return {
        "request_id": request_id,
        "verdict": "LIVE" if d.get("is_live") else "SPOOF",
        "is_live": bool(d.get("is_live", False)),
        "liveness_score": d.get("liveness_score"),
        "liveness_breakdown": d.get("liveness_breakdown", {}),
        "signal_status": d.get("signal_status", {}),
        "quality": d.get("quality", {"selfie": None}),
        "explanation": d["explanation"],
        "latency_ms": d["latency_ms"],
    }


def _match_payload(result, request_id: str) -> dict:
    d = result.to_dict()
    if not d.get("is_match"):
        verdict = "NO_MATCH"
    elif d.get("match_needs_review"):
        verdict = "MANUAL_REVIEW"
    else:
        verdict = "MATCH"
    return {
        "request_id": request_id,
        "verdict": verdict,
        "is_match": bool(d.get("is_match", False)),
        "match_score": d.get("match_score", 0.0),
        "match_confidence": d.get("match_confidence", "none"),
        "match_needs_review": bool(d.get("match_needs_review", False)),
        "quality": d.get("quality", {"id": None, "selfie": None}),
        "explanation": d["explanation"],
        "latency_ms": d["latency_ms"],
    }


def _default_pipeline_factory(settings: APISettings) -> Callable[[], object]:
    """Build the real pipeline. Imported lazily so app import stays light."""

    def factory():
        from dwarpala.yantra.pipeline import DwarpalaPipeline

        model_dir = Path(settings.model_dir) if settings.model_dir else None
        return DwarpalaPipeline(model_dir=model_dir)

    return factory


def create_app(
    pipeline_factory: Optional[Callable[[], object]] = None,
    settings: Optional[APISettings] = None,
) -> FastAPI:
    """
    Build the FastAPI app.

    Args:
        pipeline_factory: Zero-arg callable returning a pipeline-like object with
            ``verify`` / ``match_only`` / ``liveness_only``. Defaults to the real
            DwarpalaPipeline. Tests inject a mock so no models are downloaded.
        settings: APISettings. Defaults to ``load_api_settings()``.

    Returns:
        Configured FastAPI application.
    """
    settings = settings or load_api_settings()
    factory = pipeline_factory or _default_pipeline_factory(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = settings
        app.state.start_time = time.time()
        app.state.lock = Lock()
        app.state.pipeline = None
        app.state.models_loaded = False
        app.state.model_versions = {}
        app.state.audit = None
        try:
            logger.info("Loading pipeline / models (once, at startup)...")
            app.state.pipeline = factory()
            app.state.models_loaded = True
            app.state.model_versions = {
                "recognition": "insightface/buffalo_l (w600k_r50)",
                "liveness_v2": "2.7_80x80_MiniFASNetV2",
                "liveness_v1se": "4_0_0_80x80_MiniFASNetV1SE",
            }
            logger.info("Models loaded; API ready.")
        except Exception:
            logger.exception("Model load failed; /health will report 503.")
            app.state.models_loaded = False
        app.state.audit = maybe_create_audit(settings)
        try:
            yield
        finally:
            if app.state.audit is not None:
                app.state.audit.close()

    app = FastAPI(
        title="Dwarpala Verification API",
        version="0.1.0",
        description="Biometric verification (identity + liveness) over REST.",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── request_id + structured access log middleware ──
    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        start = time.time()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        latency_ms = (time.time() - start) * 1000
        logger.info(
            f"access path={request.url.path} method={request.method} "
            f"status={response.status_code} request_id={request_id} "
            f"latency_ms={latency_ms:.1f}"
        )
        return response

    # ── error envelope handlers (no stack traces / bytes to client) ──
    def _envelope(status: int, code: str, message: str, request_id: str):
        return JSONResponse(
            status_code=status,
            content={"error": {"code": code, "message": message, "request_id": request_id}},
        )

    @app.exception_handler(APIError)
    async def _api_error(request: Request, exc: APIError):
        rid = getattr(request.state, "request_id", str(uuid.uuid4()))
        logger.warning(
            f"api_error code={exc.code} status={exc.status} " f"request_id={rid} msg={exc.message}"
        )
        return _envelope(exc.status, exc.code, exc.message, rid)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError):
        rid = getattr(request.state, "request_id", str(uuid.uuid4()))
        return _envelope(
            422, "VALIDATION_ERROR", "Request is missing required fields or is malformed.", rid
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        rid = getattr(request.state, "request_id", str(uuid.uuid4()))
        logger.exception(f"unhandled error request_id={rid}")  # full trace server-side
        return _envelope(
            500, "INTERNAL_ERROR", "An internal error occurred while processing the request.", rid
        )

    def _require_pipeline(request: Request):
        if (
            not getattr(request.app.state, "models_loaded", False)
            or request.app.state.pipeline is None
        ):
            raise APIError(503, "MODELS_NOT_READY", "Models are still loading or failed to load.")
        return request.app.state.pipeline

    def _run_locked(request: Request, fn, *args):
        """Serialize inference behind the process-wide lock."""
        with request.app.state.lock:
            return fn(*args)

    def _audit(
        request: Request,
        endpoint: str,
        verdict: str,
        latency_ms: float,
        match_score=None,
        liveness_score=None,
    ):
        audit = getattr(request.app.state, "audit", None)
        if audit is not None:
            audit.record(
                request.state.request_id, endpoint, verdict, latency_ms, match_score, liveness_score
            )

    # ──────────────────────────────────────────────────────────────────
    # Endpoints
    # ──────────────────────────────────────────────────────────────────
    @app.get(
        "/health",
        response_model=HealthResponse,
        responses={503: {"description": "Models not loaded"}},
    )
    def health(request: Request):
        st = request.app.state
        uptime = round(time.time() - st.start_time, 2)
        if not getattr(st, "models_loaded", False):
            raise APIError(503, "MODELS_NOT_READY", "Models are not loaded yet.")
        return HealthResponse(
            status="ok",
            models_loaded=True,
            model_versions=st.model_versions,
            uptime_seconds=uptime,
        )

    @app.post("/verify", response_model=VerifyResponse)
    def verify(request: Request, id_image: UploadFile = File(...), selfie: UploadFile = File(...)):
        pipeline = _require_pipeline(request)
        id_arr, id_size = _prepare_image(id_image, settings)
        selfie_input, temp_path, selfie_size = _prepare_selfie(selfie, settings)
        rid = request.state.request_id
        logger.info(
            f"verify request_id={rid} id_mime={id_image.content_type} "
            f"id_bytes={id_size} selfie_mime={selfie.content_type} "
            f"selfie_bytes={selfie_size}"
        )
        try:
            result = _run_locked(request, pipeline.verify, id_arr, selfie_input)
        finally:
            if temp_path:
                _safe_unlink(temp_path)
        payload = _verify_payload(result, rid)
        _audit(
            request,
            "verify",
            payload["verdict"],
            payload["latency_ms"],
            payload.get("match_score"),
            payload.get("liveness_score"),
        )
        logger.info(
            f"verify_done request_id={rid} verdict={payload['verdict']} "
            f"latency_ms={payload['latency_ms']:.1f}"
        )
        return payload

    @app.post("/liveness", response_model=LivenessResponse)
    def liveness(request: Request, selfie: UploadFile = File(...)):
        pipeline = _require_pipeline(request)
        selfie_input, temp_path, selfie_size = _prepare_selfie(selfie, settings)
        rid = request.state.request_id
        logger.info(
            f"liveness request_id={rid} selfie_mime={selfie.content_type} "
            f"selfie_bytes={selfie_size}"
        )
        try:
            result = _run_locked(request, pipeline.liveness_only, selfie_input)
        finally:
            if temp_path:
                _safe_unlink(temp_path)
        payload = _liveness_payload(result, rid)
        _audit(
            request,
            "liveness",
            payload["verdict"],
            payload["latency_ms"],
            liveness_score=payload.get("liveness_score"),
        )
        logger.info(
            f"liveness_done request_id={rid} verdict={payload['verdict']} "
            f"latency_ms={payload['latency_ms']:.1f}"
        )
        return payload

    @app.post("/match", response_model=MatchResponse)
    def match(request: Request, id_image: UploadFile = File(...), selfie: UploadFile = File(...)):
        pipeline = _require_pipeline(request)
        id_arr, id_size = _prepare_image(id_image, settings)
        selfie_arr, selfie_size = _prepare_image(selfie, settings)
        rid = request.state.request_id
        logger.info(
            f"match request_id={rid} id_mime={id_image.content_type} "
            f"id_bytes={id_size} selfie_mime={selfie.content_type} "
            f"selfie_bytes={selfie_size}"
        )
        result = _run_locked(request, pipeline.match_only, id_arr, selfie_arr)
        payload = _match_payload(result, rid)
        _audit(
            request,
            "match",
            payload["verdict"],
            payload["latency_ms"],
            match_score=payload.get("match_score"),
        )
        logger.info(
            f"match_done request_id={rid} verdict={payload['verdict']} "
            f"latency_ms={payload['latency_ms']:.1f}"
        )
        return payload

    return app


def _safe_unlink(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:  # pragma: no cover
        pass


# Module-level app for `uvicorn dwarpala.api.server:app`. Importing this module
# does NOT load any models — the pipeline is built in the lifespan at startup.
app = create_app()

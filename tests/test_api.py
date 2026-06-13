"""
Tests for the Dwarpala FastAPI REST server (Phase 3).

Endpoint tests use a MOCKED pipeline (via the create_app pipeline_factory), so
they run in CI with no model downloads. Tests that need real weights are marked
``requires_models`` and skipped unless models are present locally.
"""

import io
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from dwarpala.api.config import APISettings
from dwarpala.api.server import create_app

FIXTURES = Path(__file__).parent / "fixtures" / "lfw_samples"
TESTIMG = Path(__file__).parent / "testimg"


# ── helpers ────────────────────────────────────────────────────────────────
class _Res:
    """Minimal stand-in for a pipeline result with a to_dict()."""

    def __init__(self, d):
        self._d = d

    def to_dict(self):
        return self._d


def _verify_dict(verdict="ACCEPT"):
    return {
        "verdict": verdict,
        "match_score": 0.82,
        "liveness_score": 0.97,
        "liveness_breakdown": {"minifas": 0.99, "texture": 0.81, "temporal": None, "rppg": None},
        "signal_status": {
            "minifas": "OK",
            "texture": "OK",
            "temporal": "NOT_APPLICABLE",
            "rppg": "NOT_APPLICABLE",
        },
        "quality": {"id": None, "selfie": None},
        "explanation": "Face match confirmed. Live subject confirmed.",
        "latency_ms": 12.3,
    }


def _liveness_dict(is_live=True):
    return {
        "is_live": is_live,
        "liveness_score": 0.97 if is_live else 0.04,
        "liveness_breakdown": {"minifas": 0.99, "texture": 0.81, "temporal": None, "rppg": None},
        "signal_status": {"minifas": "OK", "texture": "OK"},
        "quality": {"selfie": None},
        "explanation": "Subject appears to be a live person.",
        "latency_ms": 8.1,
    }


def _match_dict(is_match=True, needs_review=False):
    return {
        "is_match": is_match,
        "match_score": 0.82,
        "match_confidence": "high",
        "match_needs_review": needs_review,
        "quality": {"id": None, "selfie": None},
        "explanation": "Face match confirmed (similarity=0.820).",
        "latency_ms": 9.0,
    }


def make_pipeline():
    p = MagicMock()
    p.verify.return_value = _Res(_verify_dict())
    p.match_only.return_value = _Res(_match_dict())
    p.liveness_only.return_value = _Res(_liveness_dict())
    return p


def png_bytes(color=(120, 130, 140), size=64):
    img = np.full((size, size, 3), color, np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


@pytest.fixture
def app_and_pipeline():
    pipeline = make_pipeline()
    settings = APISettings(max_upload_mb=15, cors_origins=["*"], audit_log="off")
    app = create_app(pipeline_factory=lambda: pipeline, settings=settings)
    return app, pipeline


# ── happy paths ─────────────────────────────────────────────────────────────
def test_verify_happy_path(app_and_pipeline):
    app, pipeline = app_and_pipeline
    data = png_bytes()
    with TestClient(app) as client:
        r = client.post(
            "/verify",
            files={
                "id_image": ("id.png", data, "image/png"),
                "selfie": ("selfie.png", data, "image/png"),
            },
        )
    assert r.status_code == 200
    body = r.json()
    for key in (
        "request_id",
        "verdict",
        "match_score",
        "liveness_score",
        "liveness_breakdown",
        "signal_status",
        "quality",
        "explanation",
        "latency_ms",
    ):
        assert key in body
    assert body["verdict"] == "ACCEPT"
    assert set(body["liveness_breakdown"]) == {"minifas", "texture", "temporal", "rppg"}
    assert r.headers["X-Request-ID"] == body["request_id"]
    pipeline.verify.assert_called_once()


def test_liveness_happy_path(app_and_pipeline):
    app, _ = app_and_pipeline
    data = png_bytes()
    with TestClient(app) as client:
        r = client.post("/liveness", files={"selfie": ("s.png", data, "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert body["verdict"] == "LIVE"
    assert body["is_live"] is True
    assert "liveness_breakdown" in body and "request_id" in body


def test_match_happy_path(app_and_pipeline):
    app, _ = app_and_pipeline
    data = png_bytes()
    with TestClient(app) as client:
        r = client.post(
            "/match",
            files={
                "id_image": ("id.png", data, "image/png"),
                "selfie": ("selfie.png", data, "image/png"),
            },
        )
    assert r.status_code == 200
    body = r.json()
    assert body["verdict"] == "MATCH"
    assert body["is_match"] is True
    assert body["match_score"] == 0.82
    assert "liveness_score" not in body  # match endpoint never returns liveness


# ── validation ───────────────────────────────────────────────────────────────
def test_wrong_mime_returns_415(app_and_pipeline):
    app, _ = app_and_pipeline
    with TestClient(app) as client:
        r = client.post(
            "/verify",
            files={
                "id_image": ("id.txt", b"hello", "text/plain"),
                "selfie": ("s.png", png_bytes(), "image/png"),
            },
        )
    assert r.status_code == 415
    assert r.json()["error"]["code"] == "UNSUPPORTED_MEDIA_TYPE"


def test_oversize_returns_413():
    pipeline = make_pipeline()
    settings = APISettings(max_upload_mb=0)  # 0 bytes cap → any file is too big
    app = create_app(pipeline_factory=lambda: pipeline, settings=settings)
    with TestClient(app) as client:
        r = client.post(
            "/verify",
            files={
                "id_image": ("id.png", png_bytes(), "image/png"),
                "selfie": ("s.png", png_bytes(), "image/png"),
            },
        )
    assert r.status_code == 413
    assert r.json()["error"]["code"] == "PAYLOAD_TOO_LARGE"


def test_undecodable_returns_422(app_and_pipeline):
    app, _ = app_and_pipeline
    with TestClient(app) as client:
        r = client.post(
            "/verify",
            files={
                "id_image": ("id.png", b"this is not an image", "image/png"),
                "selfie": ("s.png", png_bytes(), "image/png"),
            },
        )
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "UNDECODABLE_IMAGE"


def test_missing_field_returns_envelope(app_and_pipeline):
    app, _ = app_and_pipeline
    with TestClient(app) as client:
        r = client.post(
            "/verify",
            files={
                "id_image": ("id.png", png_bytes(), "image/png"),
            },
        )
    assert r.status_code == 422
    assert "error" in r.json() and "request_id" in r.json()["error"]


# ── health ───────────────────────────────────────────────────────────────────
def test_health_503_before_load_and_200_after(app_and_pipeline):
    app, _ = app_and_pipeline
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["models_loaded"] is True

        client.app.state.models_loaded = False
        r = client.get("/health")
        assert r.status_code == 503
        assert r.json()["error"]["code"] == "MODELS_NOT_READY"


# ── error envelope on forced pipeline exception ──────────────────────────────
def test_pipeline_exception_returns_500_envelope(app_and_pipeline):
    app, pipeline = app_and_pipeline
    pipeline.verify.side_effect = RuntimeError("boom")
    data = png_bytes()
    with TestClient(app, raise_server_exceptions=False) as client:
        r = client.post(
            "/verify",
            files={
                "id_image": ("id.png", data, "image/png"),
                "selfie": ("s.png", data, "image/png"),
            },
        )
    assert r.status_code == 500
    body = r.json()
    assert body["error"]["code"] == "INTERNAL_ERROR"
    assert "boom" not in json.dumps(body)  # no internal detail leaked


# ── no image bytes in logs ───────────────────────────────────────────────────
def test_no_image_bytes_in_logs(app_and_pipeline):
    from loguru import logger as loguru_logger

    app, _ = app_and_pipeline
    buffer = io.StringIO()
    sink_id = loguru_logger.add(buffer, level="DEBUG")
    try:
        data = png_bytes(color=(7, 199, 3))
        with TestClient(app) as client:
            r = client.post(
                "/verify",
                files={
                    "id_image": ("id.png", data, "image/png"),
                    "selfie": ("s.png", data, "image/png"),
                },
            )
        assert r.status_code == 200
        rid = r.json()["request_id"]
    finally:
        loguru_logger.remove(sink_id)

    logs = buffer.getvalue()
    assert rid in logs  # structured logging happened
    assert "verify_done" in logs
    # The raw image bytes must never appear in any log line.
    marker = data[:24].decode("latin-1")
    assert marker not in logs


# ── concurrency smoke test (mocked pipeline) ─────────────────────────────────
def test_concurrent_requests(app_and_pipeline):
    app, _ = app_and_pipeline
    data = png_bytes()
    with TestClient(app) as client:

        def fire(_):
            return client.post(
                "/verify",
                files={
                    "id_image": ("id.png", data, "image/png"),
                    "selfie": ("s.png", data, "image/png"),
                },
            ).status_code

        with ThreadPoolExecutor(max_workers=5) as ex:
            statuses = list(ex.map(fire, range(5)))
    assert statuses == [200] * 5


def test_audit_log_sqlite(tmp_path):
    db = tmp_path / "audit.db"
    pipeline = make_pipeline()
    settings = APISettings(audit_log="sqlite", audit_db=str(db))
    app = create_app(pipeline_factory=lambda: pipeline, settings=settings)
    data = png_bytes()
    with TestClient(app) as client:
        r = client.post(
            "/verify",
            files={
                "id_image": ("id.png", data, "image/png"),
                "selfie": ("s.png", data, "image/png"),
            },
        )
        assert r.status_code == 200
    import sqlite3

    conn = sqlite3.connect(str(db))
    rows = conn.execute("SELECT endpoint, verdict FROM audit").fetchall()
    conn.close()
    assert rows and rows[0][0] == "verify" and rows[0][1] == "ACCEPT"


# ── integration tests requiring real models ──────────────────────────────────
@pytest.mark.requires_models
def test_verify_integration_real_models():
    """
    Real /verify on a matching pair of real selfies (same person, both live).
    With the SCRFD default detector both faces are detected and the full
    match + liveness path runs, so the verdict must be ACCEPT (was previously a
    false 'no face' REJECT under the weak Haar default).
    """
    id_path = TESTIMG / "selfie1.jpeg"
    selfie_path = TESTIMG / "selfie2.jpeg"
    if not id_path.exists() or not selfie_path.exists():
        pytest.skip("fixtures not available")

    try:
        app = create_app(settings=APISettings(audit_log="off"))
        with TestClient(app) as client:
            if not client.app.state.models_loaded:
                pytest.skip("models not available")
            with open(id_path, "rb") as f1, open(selfie_path, "rb") as f2:
                r = client.post(
                    "/verify",
                    files={
                        "id_image": ("id.jpeg", f1.read(), "image/jpeg"),
                        "selfie": ("selfie.jpeg", f2.read(), "image/jpeg"),
                    },
                )
    except Exception as e:
        pytest.skip(f"pipeline unavailable: {e}")

    assert r.status_code == 200
    body = r.json()
    assert body["verdict"] == "ACCEPT", f"expected ACCEPT, got {body}"
    assert body["match_score"] > 0.0
    assert body["liveness_score"] is not None
    assert isinstance(body["latency_ms"], (int, float))
    assert body["request_id"]


@pytest.mark.requires_models
def test_minifas_normalization_regression():
    """
    Regression guard for the [0,255] normalization bug: real selfies must score
    > 0.5 live and spoofs < 0.5. Mirrors scripts/diag_minifas.py's core check.
    """
    from dwarpala.prana.minifas_analyzer import MiniFASAnalyzer

    try:
        analyzer = MiniFASAnalyzer()
        if not analyzer.models_loaded:
            pytest.skip("MiniFASNet models not available")
        from insightface.app import FaceAnalysis

        face_app = FaceAnalysis(name="buffalo_l")
        face_app.prepare(ctx_id=-1, det_size=(640, 640))
    except Exception as e:
        pytest.skip(f"models/detector unavailable: {e}")

    expectations = {
        "selfie1": "live",
        "selfie2": "live",
        "printed": "spoof",
        "screencapture": "spoof",
    }
    for name, expected in expectations.items():
        path = TESTIMG / f"{name}.jpeg"
        if not path.exists():
            pytest.skip(f"missing {path}")
        img = cv2.imread(str(path))
        faces = face_app.get(img)
        assert faces, f"no face detected in {name}"
        x1, y1, x2, y2 = faces[0].bbox.astype(int)
        result = analyzer.analyze(img, (x1, y1, x2 - x1, y2 - y1))
        if expected == "live":
            assert result.score > 0.5, f"{name}: expected live, got {result.score:.3f}"
        else:
            assert result.score < 0.5, f"{name}: expected spoof, got {result.score:.3f}"

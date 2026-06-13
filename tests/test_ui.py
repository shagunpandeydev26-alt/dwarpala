"""
Tests for the Gradio demo UI (Phase 4).

Pure display/callback logic is tested directly with synthetic inputs and a
mocked pipeline — no browser, webcam, or model downloads needed. Tests that
need a real forward pass are marked ``requires_models``.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from matplotlib.figure import Figure

from dwarpala.ui.app import (
    format_breakdown,
    fft_to_image,
    run_liveness_lab,
    run_verify,
    rppg_figure,
    verdict_banner_html,
    verdict_color,
)

TESTIMG = Path(__file__).parent / "testimg"


# ── verdict banner color mapping ─────────────────────────────────────────────
def test_verdict_color_mapping():
    assert verdict_color("ACCEPT") == "#1a7f37"  # green
    assert verdict_color("LIVE") == "#1a7f37"  # green
    assert verdict_color("REJECT") == "#c1121f"  # red
    assert verdict_color("SPOOF") == "#c1121f"  # red
    assert verdict_color("MANUAL_REVIEW") == "#b45309"  # amber
    # banner HTML carries the same color
    assert "#1a7f37" in verdict_banner_html("ACCEPT")
    assert "#c1121f" in verdict_banner_html("REJECT")
    assert "#b45309" in verdict_banner_html("MANUAL_REVIEW")


# ── breakdown table rendering ────────────────────────────────────────────────
def test_breakdown_not_applicable_single_image():
    """Single-image case: temporal+rppg must show NOT_APPLICABLE, not 0."""
    breakdown = {"minifas": 0.99, "texture": 0.81, "temporal": None, "rppg": None}
    status = {
        "minifas": "OK",
        "texture": "OK",
        "temporal": "NOT_APPLICABLE",
        "rppg": "NOT_APPLICABLE",
    }
    rows = format_breakdown(breakdown, status, rppg_valid=None)
    as_dict = {r[0]: (r[1], r[2]) for r in rows}
    assert as_dict["minifas"] == ("0.990", "OK")
    assert as_dict["temporal"] == ("—", "NOT_APPLICABLE")
    assert as_dict["rppg"] == ("—", "NOT_APPLICABLE")
    # crucially: not rendered as a misleading 0.000
    assert as_dict["temporal"][0] != "0.000"


def test_breakdown_rppg_low_confidence():
    """rPPG ran (OK) but no valid heartbeat → surfaced as LOW_CONFIDENCE."""
    breakdown = {"minifas": 0.6, "texture": 0.5, "temporal": 0.5, "rppg": 0.3}
    status = {"minifas": "OK", "texture": "OK", "temporal": "OK", "rppg": "OK"}
    rows = format_breakdown(breakdown, status, rppg_valid=False)
    as_dict = {r[0]: r[2] for r in rows}
    assert as_dict["rppg"] == "LOW_CONFIDENCE"
    # a valid heartbeat keeps it OK
    rows_ok = format_breakdown(breakdown, status, rppg_valid=True)
    assert {r[0]: r[2] for r in rows_ok}["rppg"] == "OK"


# ── rPPG figure fallback ─────────────────────────────────────────────────────
def test_rppg_figure_insufficient_signal():
    fig = rppg_figure(None, None, None, has_valid=False, message="Insufficient signal — need ≥5s")
    assert isinstance(fig, Figure)
    texts = " ".join(t.get_text() for t in fig.axes[0].texts).lower()
    assert "insufficient" in texts


def test_rppg_figure_real_trace_has_no_message():
    t = np.linspace(0, 5, 150)
    sig = np.sin(2 * np.pi * 1.2 * t)
    fig = rppg_figure(t, sig, bpm=72, has_valid=True)
    assert isinstance(fig, Figure)
    # a real trace draws a line and a BPM title, not a centered message
    assert len(fig.axes[0].lines) == 1
    assert "72" in fig.axes[0].get_title()


def test_fft_to_image_roundtrip():
    spec = np.linspace(0, 1, 64 * 64, dtype=np.float32).reshape(64, 64)
    img = fft_to_image(spec)
    assert img.dtype == np.uint8 and img.shape == (64, 64)
    assert img.max() <= 255 and img.min() >= 0
    assert fft_to_image(None) is None


# ── callbacks with a mocked pipeline ─────────────────────────────────────────
class _Res:
    def __init__(self, d, liveness_verdict=None):
        self._d = d
        self.liveness_verdict = liveness_verdict

    def to_dict(self):
        return self._d


def test_run_verify_accept_path():
    pipeline = MagicMock()
    pipeline.verify.return_value = _Res(
        {
            "verdict": "ACCEPT",
            "match_score": 0.85,
            "liveness_score": 0.92,
            "liveness_breakdown": {
                "minifas": 0.99,
                "texture": 0.8,
                "temporal": None,
                "rppg": None,
            },
            "signal_status": {
                "minifas": "OK",
                "texture": "OK",
                "temporal": "NOT_APPLICABLE",
                "rppg": "NOT_APPLICABLE",
            },
            "explanation": "Face match confirmed. Live subject confirmed.",
            "latency_ms": 210.0,
        }
    )
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    banner, match_html, live_html, rows, expl, latency = run_verify(pipeline, img, img)
    assert "#1a7f37" in banner  # green ACCEPT
    assert "0.850" in match_html and "0.920" in live_html
    assert len(rows) == 4
    assert latency == 210.0
    pipeline.verify.assert_called_once()


def test_run_verify_missing_inputs_no_crash():
    pipeline = MagicMock()
    banner, _, _, rows, expl, latency = run_verify(pipeline, None, None)
    assert "#b45309" in banner  # amber prompt
    assert latency == 0.0
    pipeline.verify.assert_not_called()


def test_run_liveness_lab_insufficient_signal():
    """Single-frame 'video' → rPPG insufficient: honest message, SPOOF banner, no fake trace."""
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    pipeline = MagicMock()
    pipeline.load_selfie_frames.return_value = [frame]
    pipeline.liveness_only.return_value = _Res(
        {
            "is_live": False,
            "liveness_score": 0.40,
            "liveness_breakdown": {
                "minifas": 0.40,
                "texture": 0.5,
                "temporal": None,
                "rppg": None,
            },
            "signal_status": {
                "minifas": "OK",
                "texture": "OK",
                "temporal": "NOT_APPLICABLE",
                "rppg": "NOT_APPLICABLE",
            },
            "explanation": "Spoof suspected.",
            "latency_ms": 50.0,
        },
        liveness_verdict=SimpleNamespace(rppg_result=None),
    )
    pipeline.detector.detect_largest.return_value = None  # no FFT viz

    fig, rows, fft_img, banner = run_liveness_lab(pipeline, "fake_video.mp4")
    assert isinstance(fig, Figure)
    texts = " ".join(t.get_text() for t in fig.axes[0].texts).lower()
    assert "insufficient" in texts
    assert fft_img is None
    assert "#c1121f" in banner  # red SPOOF
    # rPPG analyzer waveform must NOT be requested when signal is insufficient
    pipeline.liveness.rppg_analyzer.get_rppg_waveform.assert_not_called()


# ── RGB/BGR seam + channel-fix regression (real models) ──────────────────────
@pytest.mark.requires_models
def test_rgb_bgr_parity_file_vs_gradio_array():
    """
    A file-loaded image and the same image as a Gradio-style RGB array must
    produce identical liveness scores. Guards the exact channel-order seam that
    bit MiniFASNet (the UI feeds RGB; the pipeline is RGB-native).
    """
    import cv2

    from dwarpala.yantra.pipeline import DwarpalaPipeline

    path = TESTIMG / "selfie1.jpeg"
    if not path.exists():
        pytest.skip("fixture missing")
    try:
        pipe = DwarpalaPipeline()
    except Exception as e:
        pytest.skip(f"pipeline unavailable: {e}")

    rgb_array = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)  # gradio-style
    r_array = pipe.liveness_only(rgb_array)
    r_path = pipe.liveness_only(str(path))  # pipeline loads as RGB internally

    s_array = r_array.liveness_verdict.score
    s_path = r_path.liveness_verdict.score
    assert abs(s_array - s_path) < 1e-6, f"RGB array {s_array} != file {s_path}"
    assert s_array > 0.5  # and the (correct) value: a real selfie reads live


@pytest.mark.requires_models
def test_screencapture_spoof_through_pipeline():
    """
    Channel-fix regression: a screen-replay spoof must score SPOOF through the
    real pipeline. Before the RGB→BGR fix in the fusion gate it scored ~0.95
    (LIVE) because MiniFASNet was fed RGB instead of BGR.
    """
    import cv2

    from dwarpala.yantra.pipeline import DwarpalaPipeline

    path = TESTIMG / "screencapture.jpeg"
    if not path.exists():
        pytest.skip("fixture missing")
    try:
        pipe = DwarpalaPipeline()
    except Exception as e:
        pytest.skip(f"pipeline unavailable: {e}")

    rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    result = pipe.liveness_only(rgb)
    minifas = result.liveness_verdict.minifas_result.score
    assert minifas < 0.5, f"screencapture MiniFASNet should be spoof, got {minifas:.3f}"

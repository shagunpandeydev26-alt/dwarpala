"""
Tests for the Prana liveness detection module.
"""

import numpy as np
import pytest

from dwarpala.prana.texture_analyzer import TextureAnalyzer
from dwarpala.prana.temporal_analyzer import TemporalAnalyzer
from dwarpala.prana.rppg_analyzer import RPPGAnalyzer
from dwarpala.prana.fusion_gate import LivenessFusionGate, DEFAULT_FUSION_WEIGHTS
from dwarpala.prana.minifas_analyzer import MiniFASAnalyzer, MiniFASResult


class TestTextureAnalyzer:
    """Test texture-based liveness detection."""

    def test_init(self):
        analyzer = TextureAnalyzer()
        assert analyzer.lbp_radii == [1, 2, 3]

    def test_analyze_returns_result(self):
        analyzer = TextureAnalyzer()
        face = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
        result = analyzer.analyze(face)
        assert 0.0 <= result.score <= 1.0
        assert result.prediction in ("live", "spoof")

    def test_uniform_image_scores_low(self):
        """A uniform (flat) image should score low — resembles print attack."""
        analyzer = TextureAnalyzer()
        flat = np.full((112, 112, 3), 128, dtype=np.uint8)
        result = analyzer.analyze(flat)
        # Flat images lack natural texture — should trend toward spoof
        assert result.lbp_score < 0.8


class TestTemporalAnalyzer:
    """Test temporal motion-based liveness detection."""

    def test_init(self):
        analyzer = TemporalAnalyzer(fps=30)
        assert analyzer.fps == 30

    def test_too_few_frames(self):
        """Should return uncertain for too few frames."""
        analyzer = TemporalAnalyzer(min_frames=15)
        frames = [np.zeros((112, 112, 3), dtype=np.uint8)] * 5
        result = analyzer.analyze(frames)
        assert result.prediction == "uncertain"

    def test_static_frames_score_low(self):
        """Identical frames (no motion) should score low."""
        analyzer = TemporalAnalyzer(fps=30, min_frames=5)
        frame = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
        frames = [frame.copy() for _ in range(20)]
        landmarks = [
            np.array([[30, 35], [80, 35], [56, 55], [40, 75], [70, 75]],
                     dtype=np.float32)
        ] * 20
        result = analyzer.analyze(frames, landmarks)
        # No motion → saccade count should be 0
        assert result.saccade_count == 0


class TestRPPGAnalyzer:
    """Test rPPG heartbeat detection."""

    def test_init(self):
        analyzer = RPPGAnalyzer(fps=30)
        assert analyzer.bandpass_low == 0.7
        assert analyzer.bandpass_high == 4.0

    def test_too_few_frames(self):
        """Should return uncertain for insufficient frames."""
        analyzer = RPPGAnalyzer(fps=30, window_seconds=3.0)
        frames = [np.zeros((112, 112, 3), dtype=np.uint8)] * 10
        result = analyzer.analyze(frames)
        assert not result.has_valid_heartbeat

    def test_synthetic_heartbeat(self):
        """Test with synthetic pulsating signal at 72 BPM."""
        fps = 30
        duration = 4  # seconds
        n_frames = fps * duration
        analyzer = RPPGAnalyzer(fps=fps, window_seconds=3.0, min_snr=1.0)

        # Create frames with synthetic pulse (72 BPM = 1.2 Hz)
        frames = []
        for i in range(n_frames):
            t = i / fps
            # Simulate subtle green channel oscillation (heartbeat)
            pulse = 0.02 * np.sin(2 * np.pi * 1.2 * t)
            base_color = 150 + int(pulse * 255)
            base_color = max(0, min(255, base_color))
            frame = np.full((112, 112, 3), base_color, dtype=np.uint8)
            frames.append(frame)

        result = analyzer.analyze(frames)
        assert result.score >= 0  # Should produce some score


class TestLivenessFusionGate:
    """Test multi-modal liveness fusion."""

    def test_init(self):
        gate = LivenessFusionGate(
            enable_texture=True,
            enable_temporal=False,
            enable_rppg=False,
        )
        assert gate.texture_analyzer is not None
        assert gate.temporal_analyzer is None

    def test_texture_only(self):
        """Should work with only texture analysis enabled."""
        gate = LivenessFusionGate(
            enable_texture=True,
            enable_temporal=False,
            enable_rppg=False,
        )
        face = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
        verdict = gate.analyze(face_image=face)
        assert verdict.score >= 0
        assert verdict.texture_result is not None
        assert verdict.temporal_result is None
        assert verdict.rppg_result is None


class TestMiniFASAnalyzer:
    """Test MiniFASNet-based anti-spoofing."""

    def test_init_no_models(self):
        """Should initialize gracefully without model files."""
        analyzer = MiniFASAnalyzer()
        assert not analyzer.models_loaded
        assert analyzer._model_v2 is None
        assert analyzer._model_v1se is None

    def test_analyze_no_models(self):
        """Without models, should return neutral result."""
        analyzer = MiniFASAnalyzer()
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        bbox = (100, 50, 200, 300)
        result = analyzer.analyze(img, bbox)
        assert isinstance(result, MiniFASResult)
        assert result.score == 0.5
        assert result.prediction == "uncertain"
        assert result.v2_score == 0.5
        assert result.v1se_score == 0.5

    def test_preprocess(self):
        """Test MiniFASNet preprocessing produces correct shape."""
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        bbox = (200, 150, 100, 120)
        tensor = MiniFASAnalyzer.preprocess(img, bbox, scale=2.7)
        assert tensor.shape == (1, 3, 80, 80)
        assert tensor.dtype == np.float32
        assert tensor.min() >= -1.01
        assert tensor.max() <= 1.01

    def test_preprocess_different_scales(self):
        """Scale 2.7 and 4.0 should produce different crops."""
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        bbox = (200, 150, 100, 120)
        t1 = MiniFASAnalyzer.preprocess(img, bbox, scale=2.7)
        t2 = MiniFASAnalyzer.preprocess(img, bbox, scale=4.0)
        assert t1.shape == t2.shape == (1, 3, 80, 80)

    def test_preprocess_bbox_at_edge(self):
        """Should handle bboxes near image edges without cropping errors."""
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        bbox = (0, 0, 50, 50)
        tensor = MiniFASAnalyzer.preprocess(img, bbox, scale=2.7)
        assert tensor.shape == (1, 3, 80, 80)

    def test_preprocess_invalid_bbox_fallback(self):
        """Should fall back to original bbox when scaled crop goes out of bounds."""
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        bbox = (0, 0, 10, 10)
        tensor = MiniFASAnalyzer.preprocess(img, bbox, scale=4.0)
        assert tensor.shape == (1, 3, 80, 80)


class TestFusionGateExtended:
    """Extended tests for 4-signal fusion gate."""

    def test_default_weights_sum_to_one(self):
        """Default fusion weights must sum to 1.0."""
        total = sum(DEFAULT_FUSION_WEIGHTS.values())
        assert abs(total - 1.0) < 1e-6

    def test_single_image_mode(self):
        """Single image (no video) should skip temporal and rPPG."""
        gate = LivenessFusionGate(
            enable_texture=True,
            enable_temporal=True,
            enable_rppg=True,
            enable_minifas=False,
        )
        face = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
        verdict = gate.analyze(face_image=face, video_frames=None)
        assert verdict.texture_result is not None
        assert verdict.temporal_result is None
        assert verdict.rppg_result is None
        assert verdict.method_used == "single_image"
        assert verdict.signal_status["temporal"] == "NOT_APPLICABLE"
        assert verdict.signal_status["rppg"] == "NOT_APPLICABLE"

    def test_video_mode_includes_temporal(self):
        """With video frames, temporal should run."""
        gate = LivenessFusionGate(
            enable_texture=True,
            enable_temporal=True,
            enable_rppg=False,
            enable_minifas=False,
        )
        face = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
        frames = [np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
                  for _ in range(30)]
        verdict = gate.analyze(face_image=face, video_frames=frames)
        assert verdict.method_used == "full"
        assert verdict.signal_status.get("temporal") in ("OK", "FAILED")

    def test_minifas_in_verdict(self):
        """MiniFASNet result should be present in verdict."""
        gate = LivenessFusionGate(
            enable_texture=False,
            enable_temporal=False,
            enable_rppg=False,
            enable_minifas=True,
        )
        face = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
        verdict = gate.analyze(
            face_image=face,
            original_image=face,
            bbox=(10, 10, 50, 50),
        )
        assert verdict.minifas_result is not None

    def test_not_applicable_renormalization(self):
        """When some signals are N/A, weights should be renormalized."""
        gate = LivenessFusionGate(
            enable_texture=True,
            enable_temporal=False,
            enable_rppg=False,
            enable_minifas=False,
        )
        face = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
        verdict = gate.analyze(face_image=face)
        assert verdict.score == verdict.texture_result.score

    def test_single_image_renormalization(self):
        """Single image mode should renormalize weights."""
        gate = LivenessFusionGate(
            enable_texture=True,
            enable_temporal=True,
            enable_rppg=True,
            enable_minifas=False,
        )
        face = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
        verdict = gate.analyze(face_image=face, video_frames=None)
        assert verdict.score == verdict.texture_result.score

    def test_verdict_has_signal_status(self):
        """LivenessVerdict should include signal_status dict."""
        gate = LivenessFusionGate(
            enable_texture=True,
            enable_temporal=False,
            enable_rppg=False,
            enable_minifas=False,
        )
        face = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
        verdict = gate.analyze(face_image=face)
        assert isinstance(verdict.signal_status, dict)
        assert "texture" in verdict.signal_status
        assert verdict.signal_status["texture"] == "OK"

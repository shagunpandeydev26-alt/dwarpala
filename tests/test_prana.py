"""
Tests for the Prana liveness detection module.
"""

import numpy as np
import pytest

from dwarpala.prana.texture_analyzer import TextureAnalyzer
from dwarpala.prana.temporal_analyzer import TemporalAnalyzer
from dwarpala.prana.rppg_analyzer import RPPGAnalyzer
from dwarpala.prana.fusion_gate import LivenessFusionGate


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
        verdict = gate.analyze(face)
        assert verdict.score >= 0
        assert verdict.texture_result is not None
        assert verdict.temporal_result is None
        assert verdict.rppg_result is None

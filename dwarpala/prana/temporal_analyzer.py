"""
Temporal Analysis for liveness detection (Layer 2 — Motion).

Detects involuntary physiological movements that are impossible to
perfectly replicate in a static image or simple video replay:
1. Micro-saccades: Involuntary rapid eye movements (1-2 per second)
2. Head tremor: Physiological tremor at 8-12 Hz

A still photo or a replayed video will lack these subtle, bio-specific
motion patterns.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple

from dwarpala.utils.logger import get_logger

logger = get_logger("prana.temporal")


@dataclass
class TemporalResult:
    """Result from temporal/motion-based liveness analysis."""

    score: float  # 0.0 = spoof, 1.0 = live
    saccade_score: float
    tremor_score: float
    saccade_count: int  # Number of detected micro-saccades
    tremor_dominant_freq: float  # Hz
    prediction: str  # "live" or "spoof"

    def __str__(self):
        status = "🟢 LIVE" if self.prediction == "live" else "🔴 SPOOF"
        return (
            f"Temporal {status} | score={self.score:.3f} "
            f"(saccades={self.saccade_count}, tremor={self.tremor_dominant_freq:.1f}Hz)"
        )


class TemporalAnalyzer:
    """
    Layer 2 of Prana liveness detection.

    Analyzes video frames for involuntary physiological micro-movements:
    - Micro-saccades: tiny, rapid eye movements (0.1-0.5° amplitude)
    - Physiological head tremor: subtle 8-12 Hz head oscillation

    These signals are nearly impossible to perfectly fake:
    - Still photos have zero motion → instant spoof detection
    - Video replays may have motion, but lack the characteristic
      frequency signature of real physiological tremor

    Usage:
        analyzer = TemporalAnalyzer(fps=30)
        result = analyzer.analyze(list_of_rgb_frames, landmarks_per_frame)
    """

    def __init__(
        self,
        fps: float = 30.0,
        saccade_threshold: float = 0.3,
        tremor_freq_range: Tuple[float, float] = (8.0, 12.0),
        min_frames: int = 15,
        spoof_threshold: float = 0.5,
    ):
        """
        Args:
            fps: Frame rate of input video.
            saccade_threshold: Pixel displacement threshold for saccade detection.
            tremor_freq_range: Expected physiological tremor band (Hz).
            min_frames: Minimum frames required for analysis.
            spoof_threshold: Below this score = spoof.
        """
        self.fps = fps
        self.saccade_threshold = saccade_threshold
        self.tremor_freq_range = tremor_freq_range
        self.min_frames = min_frames
        self.spoof_threshold = spoof_threshold

        logger.info(
            f"TemporalAnalyzer: fps={fps}, tremor_band={tremor_freq_range}Hz"
        )

    def analyze(
        self,
        frames: List[np.ndarray],
        landmarks_per_frame: Optional[List[np.ndarray]] = None,
    ) -> TemporalResult:
        """
        Analyze temporal motion patterns for liveness.

        Args:
            frames: List of face images (RGB) across time.
            landmarks_per_frame: Optional 5-point landmarks for each frame.
                If None, will try to track using optical flow.

        Returns:
            TemporalResult with motion-based liveness assessment.
        """
        if len(frames) < self.min_frames:
            logger.warning(
                f"Only {len(frames)} frames (need {self.min_frames}). Skipping temporal."
            )
            return TemporalResult(
                score=0.5,
                saccade_score=0.5,
                tremor_score=0.5,
                saccade_count=0,
                tremor_dominant_freq=0.0,
                prediction="uncertain",
            )

        # Track eye and head positions across frames
        if landmarks_per_frame is not None:
            eye_positions = self._extract_eye_positions(landmarks_per_frame)
            head_positions = self._extract_head_positions(landmarks_per_frame)
        else:
            eye_positions, head_positions = self._track_with_optical_flow(frames)

        # Analyze micro-saccades
        saccade_score, saccade_count = self._analyze_saccades(eye_positions)

        # Analyze head tremor
        tremor_score, dominant_freq = self._analyze_tremor(head_positions)

        # Combined score
        score = 0.5 * saccade_score + 0.5 * tremor_score
        prediction = "live" if score >= self.spoof_threshold else "spoof"

        result = TemporalResult(
            score=float(score),
            saccade_score=float(saccade_score),
            tremor_score=float(tremor_score),
            saccade_count=saccade_count,
            tremor_dominant_freq=float(dominant_freq),
            prediction=prediction,
        )

        logger.info(str(result))
        return result

    def _extract_eye_positions(
        self, landmarks_list: List[np.ndarray]
    ) -> np.ndarray:
        """Extract eye center positions across frames."""
        positions = np.zeros((len(landmarks_list), 2))
        for i, lm in enumerate(landmarks_list):
            # Eye center = midpoint of left eye and right eye landmarks
            left_eye = lm[0]
            right_eye = lm[1]
            center = (left_eye + right_eye) / 2.0
            positions[i] = center
        return positions

    def _extract_head_positions(
        self, landmarks_list: List[np.ndarray]
    ) -> np.ndarray:
        """Extract head position (nose tip) across frames."""
        positions = np.zeros((len(landmarks_list), 2))
        for i, lm in enumerate(landmarks_list):
            positions[i] = lm[2]  # Nose tip
        return positions

    def _track_with_optical_flow(
        self, frames: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Track eye and head positions using Lucas-Kanade optical flow.
        Used when per-frame landmarks are not available.
        """
        gray_frames = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
        h, w = gray_frames[0].shape

        # Approximate eye region (top-center of face)
        eye_roi = (int(0.3 * w), int(0.25 * h))
        # Nose/head center
        head_roi = (int(0.5 * w), int(0.5 * h))

        # Initialize tracking points
        pts = np.array([[eye_roi], [head_roi]], dtype=np.float32)

        eye_positions = np.zeros((len(frames), 2))
        head_positions = np.zeros((len(frames), 2))
        eye_positions[0] = eye_roi
        head_positions[0] = head_roi

        lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )

        for i in range(1, len(frames)):
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                gray_frames[i - 1], gray_frames[i], pts, None, **lk_params
            )
            if next_pts is not None and status is not None:
                if status[0][0] == 1:
                    eye_positions[i] = next_pts[0][0]
                else:
                    eye_positions[i] = eye_positions[i - 1]
                if status[1][0] == 1:
                    head_positions[i] = next_pts[1][0]
                else:
                    head_positions[i] = head_positions[i - 1]
                pts = next_pts
            else:
                eye_positions[i] = eye_positions[i - 1]
                head_positions[i] = head_positions[i - 1]

        return eye_positions, head_positions

    def _analyze_saccades(
        self, eye_positions: np.ndarray
    ) -> Tuple[float, int]:
        """
        Detect micro-saccades from eye position timeseries.

        Micro-saccades are rapid, involuntary eye movements:
        - Amplitude: 0.1-0.5° (translates to ~0.3-1.5 pixels at typical resolution)
        - Frequency: 1-2 per second
        - Duration: 15-30ms

        Returns:
            (liveness_score, saccade_count)
        """
        if len(eye_positions) < 3:
            return 0.5, 0

        # Compute frame-to-frame displacement
        dx = np.diff(eye_positions[:, 0])
        dy = np.diff(eye_positions[:, 1])
        displacement = np.sqrt(dx**2 + dy**2)

        # Detect saccade events (sudden jumps above threshold)
        saccade_mask = displacement > self.saccade_threshold
        saccade_count = int(np.sum(saccade_mask))

        # Expected saccade rate: 1-2 per second
        duration_seconds = len(eye_positions) / self.fps
        expected_saccades = duration_seconds * 1.5  # ~1.5 per second

        # Score based on presence and rate of saccades
        if saccade_count == 0:
            # No micro-movements → likely a still image or frozen replay
            saccade_score = 0.1
        elif saccade_count > expected_saccades * 3:
            # Too many "saccades" → might be noisy/shaky replay
            saccade_score = 0.4
        else:
            # Reasonable saccade rate
            ratio = saccade_count / max(1, expected_saccades)
            saccade_score = min(1.0, 0.5 + 0.5 * min(ratio, 1.0))

        return float(saccade_score), saccade_count

    def _analyze_tremor(
        self, head_positions: np.ndarray
    ) -> Tuple[float, float]:
        """
        Detect physiological head tremor from position timeseries.

        Real humans have involuntary head tremor at 8-12 Hz due to
        the vestibular system and muscle micro-contractions.

        Returns:
            (liveness_score, dominant_frequency_hz)
        """
        if len(head_positions) < self.min_frames:
            return 0.5, 0.0

        # Remove trend (slow head movements)
        from scipy import signal as sp_signal

        x_detrended = sp_signal.detrend(head_positions[:, 0])
        y_detrended = sp_signal.detrend(head_positions[:, 1])

        # Compute power spectral density
        nperseg = min(len(x_detrended), 32)

        freq_x, psd_x = sp_signal.welch(
            x_detrended, fs=self.fps, nperseg=nperseg
        )
        freq_y, psd_y = sp_signal.welch(
            y_detrended, fs=self.fps, nperseg=nperseg
        )

        # Combined PSD
        psd_combined = psd_x + psd_y

        # Find energy in tremor band
        low, high = self.tremor_freq_range
        tremor_mask = (freq_x >= low) & (freq_x <= high)

        total_energy = np.sum(psd_combined)
        tremor_energy = np.sum(psd_combined[tremor_mask]) if tremor_mask.any() else 0

        # Dominant frequency
        dominant_idx = np.argmax(psd_combined)
        dominant_freq = float(freq_x[dominant_idx])

        # Score based on tremor band energy presence
        if total_energy == 0:
            tremor_score = 0.1  # No motion at all → spoof
        else:
            tremor_ratio = tremor_energy / total_energy

            # Real person: tremor band should have appreciable energy
            if tremor_ratio > 0.05 and low <= dominant_freq <= high:
                tremor_score = min(1.0, 0.6 + tremor_ratio * 2)
            elif tremor_ratio > 0.02:
                tremor_score = 0.5 + tremor_ratio * 5
            else:
                tremor_score = 0.2 + tremor_ratio * 10

        return float(tremor_score), dominant_freq

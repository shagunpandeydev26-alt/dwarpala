"""
Adaptive Fusion Gate for multi-modal liveness decision.

Combines scores from the three liveness layers:
1. Texture (LBP + FFT) — static analysis
2. Temporal (saccades + tremor) — motion analysis
3. rPPG (heartbeat) — physiological analysis

Supports multiple fusion strategies:
- Weighted average with configurable weights
- Early exit: if texture analysis is highly confident, skip deeper checks
- Score-level fusion with learned weights (planned)

The key insight: each modality catches different types of attacks.
Texture catches prints/screens, temporal catches static replays,
rPPG catches even high-quality deepfake videos.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional

from dwarpala.prana.texture_analyzer import TextureAnalyzer, TextureResult
from dwarpala.prana.temporal_analyzer import TemporalAnalyzer, TemporalResult
from dwarpala.prana.rppg_analyzer import RPPGAnalyzer, RPPGResult
from dwarpala.utils.logger import get_logger

logger = get_logger("prana.fusion")


@dataclass
class LivenessVerdict:
    """Final liveness decision from multi-modal fusion."""

    is_live: bool
    score: float  # Fused liveness score [0, 1]
    texture_result: Optional[TextureResult]
    temporal_result: Optional[TemporalResult]
    rppg_result: Optional[RPPGResult]
    method_used: str  # "full" | "early_exit_texture" | etc.
    explanation: str

    def __str__(self):
        status = "🟢 LIVE" if self.is_live else "🔴 SPOOF"
        return (
            f"Liveness {status} | score={self.score:.3f} | method={self.method_used}\n"
            f"  {self.explanation}"
        )


class LivenessFusionGate:
    """
    Fuses liveness scores from texture, temporal, and rPPG analyzers
    into a single liveness verdict.

    Supports early exit for efficiency:
    - If texture analysis gives very high confidence (score > 0.99 spoof),
      skip temporal and rPPG (saves ~200ms)
    - If rPPG detects a valid heartbeat, this is the strongest live signal

    Usage:
        fusion = LivenessFusionGate()
        verdict = fusion.analyze(face_image, video_frames, landmarks)
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        early_exit_threshold: float = 0.99,
        liveness_threshold: float = 0.5,
        enable_texture: bool = True,
        enable_temporal: bool = True,
        enable_rppg: bool = True,
        fps: float = 30.0,
    ):
        """
        Args:
            weights: Fusion weights {"texture": w1, "temporal": w2, "rppg": w3}.
            early_exit_threshold: If texture spoof confidence > this, skip others.
            liveness_threshold: Below this fused score = spoof.
            enable_texture: Enable texture analysis.
            enable_temporal: Enable temporal analysis.
            enable_rppg: Enable rPPG analysis.
            fps: Video frame rate.
        """
        self.weights = weights or {"texture": 0.35, "temporal": 0.30, "rppg": 0.35}
        self.early_exit_threshold = early_exit_threshold
        self.liveness_threshold = liveness_threshold
        self.fps = fps

        # Initialize analyzers
        self.texture_analyzer = TextureAnalyzer() if enable_texture else None
        self.temporal_analyzer = TemporalAnalyzer(fps=fps) if enable_temporal else None
        self.rppg_analyzer = RPPGAnalyzer(fps=fps) if enable_rppg else None

        active = []
        if enable_texture:
            active.append("texture")
        if enable_temporal:
            active.append("temporal")
        if enable_rppg:
            active.append("rppg")

        logger.info(
            f"LivenessFusionGate: active={active}, "
            f"weights={self.weights}, threshold={liveness_threshold}"
        )

    def analyze(
        self,
        face_image: np.ndarray,
        video_frames: list = None,
        landmarks_per_frame: list = None,
    ) -> LivenessVerdict:
        """
        Run multi-modal liveness analysis.

        Args:
            face_image: Single aligned face image (RGB) for texture analysis.
            video_frames: List of face images across time for temporal + rPPG.
            landmarks_per_frame: Optional per-frame landmarks.

        Returns:
            LivenessVerdict with fused decision.
        """
        texture_result = None
        temporal_result = None
        rppg_result = None
        method_used = "full"

        # ═══ Layer 1: Texture Analysis (Static) ═══
        if self.texture_analyzer is not None:
            texture_result = self.texture_analyzer.analyze(face_image)

            # Early exit: if texture analysis is very confident it's a spoof
            if texture_result.score < (1 - self.early_exit_threshold):
                logger.info("Early exit: texture analysis detected clear spoof")
                return LivenessVerdict(
                    is_live=False,
                    score=texture_result.score * self.weights["texture"],
                    texture_result=texture_result,
                    temporal_result=None,
                    rppg_result=None,
                    method_used="early_exit_spoof",
                    explanation=(
                        f"Texture analysis detected spoof characteristics "
                        f"(LBP={texture_result.lbp_score:.3f}, "
                        f"FFT={texture_result.fft_score:.3f}). "
                        f"Likely a printed photo or screen replay."
                    ),
                )

        # ═══ Layer 2: Temporal Analysis (Motion) ═══
        if self.temporal_analyzer is not None and video_frames:
            temporal_result = self.temporal_analyzer.analyze(
                video_frames, landmarks_per_frame
            )

        # ═══ Layer 3: rPPG Analysis (Physiological) ═══
        if self.rppg_analyzer is not None and video_frames:
            rppg_result = self.rppg_analyzer.analyze(
                video_frames, landmarks_per_frame
            )

            # rPPG override: valid heartbeat is very strong live evidence
            if rppg_result.has_valid_heartbeat and rppg_result.signal_quality > 5.0:
                logger.info(
                    f"rPPG detected valid heartbeat: "
                    f"{rppg_result.heart_rate_bpm:.0f} BPM "
                    f"(SNR={rppg_result.signal_quality:.1f})"
                )

        # ═══ Fusion ═══
        fused_score = self._fuse_scores(
            texture_result, temporal_result, rppg_result
        )

        is_live = fused_score >= self.liveness_threshold

        # Generate explanation
        explanation = self._generate_explanation(
            is_live, fused_score, texture_result, temporal_result, rppg_result
        )

        verdict = LivenessVerdict(
            is_live=is_live,
            score=float(fused_score),
            texture_result=texture_result,
            temporal_result=temporal_result,
            rppg_result=rppg_result,
            method_used=method_used,
            explanation=explanation,
        )

        logger.info(str(verdict))
        return verdict

    def _fuse_scores(
        self,
        texture_result: Optional[TextureResult],
        temporal_result: Optional[TemporalResult],
        rppg_result: Optional[RPPGResult],
    ) -> float:
        """
        Weighted fusion of liveness scores.
        Adapts weights if some modalities are unavailable.
        """
        scores = {}
        weights = {}

        if texture_result is not None:
            scores["texture"] = texture_result.score
            weights["texture"] = self.weights["texture"]

        if temporal_result is not None:
            scores["temporal"] = temporal_result.score
            weights["temporal"] = self.weights["temporal"]

        if rppg_result is not None:
            scores["rppg"] = rppg_result.score
            weights["rppg"] = self.weights["rppg"]

        if not scores:
            return 0.5  # No analysis possible

        # Normalize weights to sum to 1
        total_weight = sum(weights.values())
        fused = sum(
            scores[k] * weights[k] / total_weight for k in scores
        )

        return float(fused)

    def _generate_explanation(
        self,
        is_live: bool,
        fused_score: float,
        texture_result: Optional[TextureResult],
        temporal_result: Optional[TemporalResult],
        rppg_result: Optional[RPPGResult],
    ) -> str:
        """Generate human-readable explanation of the liveness verdict."""
        parts = []

        if is_live:
            parts.append(f"Subject appears to be a live person (confidence: {fused_score:.1%}).")
        else:
            parts.append(
                f"Spoof attack detected (confidence: {1 - fused_score:.1%})."
            )

        if texture_result:
            if texture_result.prediction == "live":
                parts.append("Skin texture appears natural.")
            else:
                parts.append("Texture patterns suggest a reproduced image.")

        if temporal_result:
            if temporal_result.saccade_count > 0:
                parts.append(
                    f"Detected {temporal_result.saccade_count} micro-saccades."
                )
            else:
                parts.append("No involuntary eye movements detected.")

        if rppg_result:
            if rppg_result.has_valid_heartbeat:
                parts.append(
                    f"Valid cardiac rhythm detected: "
                    f"{rppg_result.heart_rate_bpm:.0f} BPM."
                )
            else:
                parts.append("No valid cardiac signal detected.")

        return " ".join(parts)

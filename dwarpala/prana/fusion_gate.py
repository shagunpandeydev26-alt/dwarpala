"""
Adaptive Fusion Gate for multi-modal liveness decision.

Combines scores from four liveness layers:
1. MiniFASNet — Passive anti-spoofing CNN (most reliable single signal)
2. Texture (LBP + FFT) — Static texture analysis
3. Temporal (saccades + tremor) — Motion analysis
4. rPPG (heartbeat) — Physiological analysis

Fusion strategy:
- Weighted fusion with configurable weights (defaults calibrated from benchmark)
- Early exit: if texture OR MiniFASNet is very confident spoof, skip deeper checks
- rPPG override: strong valid heartbeat signal boosts liveness confidence
- NOT_APPLICABLE handling: when a signal is unavailable (e.g., single image),
  weights are renormalized among available signals
"""

import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

from dwarpala.prana.texture_analyzer import TextureAnalyzer, TextureResult
from dwarpala.prana.temporal_analyzer import TemporalAnalyzer, TemporalResult
from dwarpala.prana.rppg_analyzer import RPPGAnalyzer, RPPGResult
from dwarpala.prana.minifas_analyzer import MiniFASAnalyzer, MiniFASResult
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
    minifas_result: Optional[MiniFASResult]
    method_used: str  # "full" | "early_exit" | "single_image" etc.
    explanation: str
    # Per-signal status for API serialization
    signal_status: Dict[str, str] = None  # "OK" | "NOT_APPLICABLE" | "FAILED"

    def __post_init__(self):
        if self.signal_status is None:
            self.signal_status = {}

    def __str__(self):
        status = "LIVE" if self.is_live else "SPOOF"
        mf = f"{self.minifas_result.score:.3f}" if self.minifas_result else "N/A"
        tx = f"{self.texture_result.score:.3f}" if self.texture_result else "N/A"
        tm = f"{self.temporal_result.score:.3f}" if self.temporal_result else "N/A"
        rp = f"{self.rppg_result.score:.3f}" if self.rppg_result else "N/A"
        signals = f"MiniFAS={mf}, Texture={tx}, Temporal={tm}, rPPG={rp}"
        return (
            f"Liveness {status} | score={self.score:.3f} | method={self.method_used}\n"
            f"  {signals}\n"
            f"  {self.explanation}"
        )


# Default fusion weights calibrated from liveness benchmark.
# MiniFASNet is weighted highest as the most reliable passive signal.
# See BENCHMARKS.md for calibration methodology.
DEFAULT_FUSION_WEIGHTS: Dict[str, float] = {
    "minifas": 0.40,
    "texture": 0.25,
    "temporal": 0.15,
    "rppg": 0.20,
}


class LivenessFusionGate:
    """
    Fuses liveness scores from MiniFASNet, texture, temporal, and rPPG
    analyzers into a single liveness verdict.

    Supports:
    - Weighted fusion with NOT_APPLICABLE renormalization
    - Early exit: if texture or MiniFASNet detects clear spoof, skip deeper checks
    - rPPG override: strong valid heartbeat is strong live evidence
    - Single-image mode: no video frames → skip temporal + rPPG

    Usage:
        fusion = LivenessFusionGate()
        verdict = fusion.analyze(face_image, original_image, bbox, video_frames)
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        early_exit_threshold: float = 0.99,
        liveness_threshold: float = 0.5,
        enable_texture: bool = True,
        enable_temporal: bool = True,
        enable_rppg: bool = True,
        enable_minifas: bool = True,
        fps: float = 30.0,
        model_dir: Optional[Path] = None,
    ):
        """
        Args:
            weights: Fusion weights dict. Defaults to calibrated weights.
            early_exit_threshold: If any signal's spoof confidence > this, exit.
            liveness_threshold: Below this fused score = spoof.
            enable_texture: Enable texture analysis.
            enable_temporal: Enable temporal analysis.
            enable_rppg: Enable rPPG analysis.
            enable_minifas: Enable MiniFASNet analysis.
            fps: Video frame rate.
            model_dir: Model directory for MiniFASNet weights.
        """
        self.weights = (
            weights if weights is not None else dict(DEFAULT_FUSION_WEIGHTS)
        )
        self.early_exit_threshold = early_exit_threshold
        self.liveness_threshold = liveness_threshold
        self.fps = fps

        # Initialize analyzers
        self.texture_analyzer = TextureAnalyzer() if enable_texture else None
        self.temporal_analyzer = TemporalAnalyzer(fps=fps) if enable_temporal else None
        self.rppg_analyzer = RPPGAnalyzer(fps=fps) if enable_rppg else None
        self.minifas_analyzer = (
            MiniFASAnalyzer(model_dir=model_dir) if enable_minifas else None
        )

        active = []
        if enable_texture:
            active.append("texture")
        if enable_temporal:
            active.append("temporal")
        if enable_rppg:
            active.append("rppg")
        if enable_minifas:
            active.append("minifas")

        logger.info(
            f"LivenessFusionGate: active={active}, "
            f"weights={self.weights}, threshold={liveness_threshold}"
        )

    def analyze(
        self,
        face_image: np.ndarray,
        original_image: Optional[np.ndarray] = None,
        bbox: Optional[Tuple[int, int, int, int]] = None,
        video_frames: list = None,
        landmarks_per_frame: list = None,
    ) -> LivenessVerdict:
        """
        Run multi-modal liveness analysis.

        Args:
            face_image: Single aligned face image (RGB) for texture analysis.
            original_image: Full original image (RGB) for MiniFASNet crop.
            bbox: (x, y, w, h) face bbox for MiniFASNet crop.
            video_frames: List of face images across time for temporal + rPPG.
            landmarks_per_frame: Optional per-frame landmarks.

        Returns:
            LivenessVerdict with fused decision.
        """
        texture_result = None
        temporal_result = None
        rppg_result = None
        minifas_result = None
        signal_status = {}

        # ═══ Layer 1: MiniFASNet (Passive Anti-Spoofing CNN) ═══
        if self.minifas_analyzer is not None and original_image is not None and bbox is not None:
            try:
                minifas_result = self.minifas_analyzer.analyze(original_image, bbox)
                signal_status["minifas"] = "OK"

                # Early exit: MiniFASNet very confident spoof
                if minifas_result.score < (1 - self.early_exit_threshold):
                    logger.info(
                        "Early exit: MiniFASNet detected clear spoof "
                        f"(score={minifas_result.score:.3f})"
                    )
                    return LivenessVerdict(
                        is_live=False,
                        score=minifas_result.score * self.weights.get("minifas", 0.40),
                        texture_result=None,
                        temporal_result=None,
                        rppg_result=None,
                        minifas_result=minifas_result,
                        method_used="early_exit_minifas",
                        signal_status=signal_status,
                        explanation=(
                            f"MiniFASNet anti-spoofing detected reproduction artifacts "
                            f"(score={minifas_result.score:.2f}). "
                            f"Likely a printed photo or screen replay."
                        ),
                    )
            except Exception as e:
                logger.warning(f"MiniFASNet analysis failed: {e}")
                signal_status["minifas"] = "FAILED"
        elif self.minifas_analyzer is not None:
            signal_status["minifas"] = "NOT_APPLICABLE"
            logger.debug("MiniFASNet skipped: no original image or bbox")

        # ═══ Layer 2: Texture Analysis (Static) ═══
        if self.texture_analyzer is not None:
            try:
                texture_result = self.texture_analyzer.analyze(face_image)
                signal_status["texture"] = "OK"

                # Early exit: texture analysis very confident spoof
                if texture_result.score < (1 - self.early_exit_threshold):
                    logger.info("Early exit: texture analysis detected clear spoof")
                    return LivenessVerdict(
                        is_live=False,
                        score=texture_result.score * self.weights.get("texture", 0.25),
                        texture_result=texture_result,
                        temporal_result=None,
                        rppg_result=None,
                        minifas_result=minifas_result,
                        method_used="early_exit_texture",
                        signal_status=signal_status,
                        explanation=(
                            f"Texture analysis detected spoof characteristics "
                            f"(LBP={texture_result.lbp_score:.3f}, "
                            f"FFT={texture_result.fft_score:.3f}). "
                            f"Likely a printed photo or screen replay."
                        ),
                    )
            except Exception as e:
                logger.warning(f"Texture analysis failed: {e}")
                signal_status["texture"] = "FAILED"
        else:
            signal_status["texture"] = "NOT_APPLICABLE"

        # Determine if video frames are available for temporal + rPPG
        has_video = video_frames is not None and len(video_frames) > 1

        # ═══ Layer 3: Temporal Analysis (Motion) ═══
        if self.temporal_analyzer is not None and has_video:
            try:
                temporal_result = self.temporal_analyzer.analyze(
                    video_frames, landmarks_per_frame
                )
                signal_status["temporal"] = "OK"
            except Exception as e:
                logger.warning(f"Temporal analysis failed: {e}")
                signal_status["temporal"] = "FAILED"
        elif self.temporal_analyzer is not None:
            signal_status["temporal"] = "NOT_APPLICABLE"
            logger.debug("Temporal analysis skipped: single image (no video frames)")

        # ═══ Layer 4: rPPG Analysis (Physiological) ═══
        rppg_override = False
        if self.rppg_analyzer is not None and has_video:
            try:
                rppg_result = self.rppg_analyzer.analyze(
                    video_frames, landmarks_per_frame
                )
                signal_status["rppg"] = "OK"

                # rPPG override: valid heartbeat is strong live evidence
                if rppg_result.has_valid_heartbeat and rppg_result.signal_quality > 5.0:
                    rppg_override = True
                    logger.info(
                        f"rPPG override: valid heartbeat "
                        f"{rppg_result.heart_rate_bpm:.0f} BPM "
                        f"(SNR={rppg_result.signal_quality:.1f})"
                    )
            except Exception as e:
                logger.warning(f"rPPG analysis failed: {e}")
                signal_status["rppg"] = "FAILED"
        elif self.rppg_analyzer is not None:
            signal_status["rppg"] = "NOT_APPLICABLE"
            logger.debug("rPPG analysis skipped: single image (no video frames)")

        # Determine method description
        if has_video:
            method_used = "full"
        else:
            method_used = "single_image"

        # ═══ Fusion ═══
        fused_score = self._fuse_scores(
            minifas_result, texture_result, temporal_result, rppg_result,
        )

        # Apply rPPG override: if we detected a valid heartbeat, boost score
        if rppg_override and fused_score < 0.7:
            boost = min(0.25, rppg_result.score * 0.3)
            fused_score = min(1.0, fused_score + boost)
            logger.info(f"rPPG override applied: fused score boosted by {boost:.3f}")

        is_live = fused_score >= self.liveness_threshold

        explanation = self._generate_explanation(
            is_live, fused_score, minifas_result, texture_result,
            temporal_result, rppg_result,
        )

        verdict = LivenessVerdict(
            is_live=is_live,
            score=float(fused_score),
            texture_result=texture_result,
            temporal_result=temporal_result,
            rppg_result=rppg_result,
            minifas_result=minifas_result,
            method_used=method_used,
            signal_status=signal_status,
            explanation=explanation,
        )

        logger.info(str(verdict))
        return verdict

    def _fuse_scores(
        self,
        minifas_result: Optional[MiniFASResult],
        texture_result: Optional[TextureResult],
        temporal_result: Optional[TemporalResult],
        rppg_result: Optional[RPPGResult],
    ) -> float:
        """
        Weighted fusion of liveness scores.
        Auto-renormalizes weights when signals are unavailable.
        """
        scores = {}
        weights = {}

        if minifas_result is not None:
            scores["minifas"] = minifas_result.score
            weights["minifas"] = self.weights.get("minifas", 0.40)

        if texture_result is not None:
            scores["texture"] = texture_result.score
            weights["texture"] = self.weights.get("texture", 0.25)

        if temporal_result is not None:
            scores["temporal"] = temporal_result.score
            weights["temporal"] = self.weights.get("temporal", 0.15)

        if rppg_result is not None:
            scores["rppg"] = rppg_result.score
            weights["rppg"] = self.weights.get("rppg", 0.20)

        if not scores:
            return 0.5  # No analysis possible — return neutral

        # Renormalize weights to sum to 1
        total_weight = sum(weights.values())
        if total_weight <= 0:
            return 0.5

        fused = sum(scores[k] * weights[k] / total_weight for k in scores)
        return float(fused)

    def _generate_explanation(
        self,
        is_live: bool,
        fused_score: float,
        minifas_result: Optional[MiniFASResult],
        texture_result: Optional[TextureResult],
        temporal_result: Optional[TemporalResult],
        rppg_result: Optional[RPPGResult],
    ) -> str:
        """Generate human-readable explanation."""
        parts = []

        if is_live:
            parts.append(
                f"Subject appears to be a live person "
                f"(confidence: {fused_score:.1%})."
            )
        else:
            parts.append(
                f"Spoof attack detected "
                f"(confidence: {1 - fused_score:.1%})."
            )

        if minifas_result:
            if minifas_result.prediction == "live":
                parts.append("MiniFASNet anti-spoofing passes.")
            else:
                parts.append("MiniFASNet detected reproduction artifacts.")

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

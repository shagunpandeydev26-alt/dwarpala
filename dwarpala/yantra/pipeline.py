"""
Dwarpala Unified Verification Pipeline.

Orchestrates the full biometric verification flow:
1. Kavach (Pre-processing) → Detect, align, quality-check faces
2. Swarupa (Identity) → Extract embeddings, compute similarity
3. Prana (Liveness) → Multi-modal spoof detection
4. Decision → Accept / Reject / Manual Review

Single API call for the entire verification:
    result = pipeline.verify(id_image, selfie_video)
"""

import time
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

from dwarpala.kavach import FaceDetector, FaceAligner, QualityAssessor
from dwarpala.swarupa.embedding import EmbeddingExtractor
from dwarpala.swarupa.matcher import FaceMatcher, MatchResult
from dwarpala.prana.fusion_gate import LivenessFusionGate, LivenessVerdict
from dwarpala.utils.image_utils import load_image
from dwarpala.utils.video_utils import read_video_frames, frames_to_rgb
from dwarpala.utils.logger import get_logger

logger = get_logger("yantra.pipeline")


@dataclass
class VerificationResult:
    """Final verification verdict from the Dwarpala pipeline."""

    verdict: str  # "ACCEPT" | "REJECT" | "MANUAL_REVIEW"
    match_score: float
    match_result: Optional[MatchResult]
    liveness_verdict: Optional[LivenessVerdict]
    latency_ms: float
    explanation: str
    details: Dict = field(default_factory=dict)

    def __str__(self):
        icon = {"ACCEPT": "✅", "REJECT": "❌", "MANUAL_REVIEW": "⚠️"}.get(self.verdict, "❓")
        liveness = f"{self.liveness_verdict.score:.4f}" if self.liveness_verdict else "N/A"
        return (
            f"\n{'═' * 60}\n"
            f"  {icon} DWARPALA VERDICT: {self.verdict}\n"
            f"{'═' * 60}\n"
            f"  Match Score:    {self.match_score:.4f}\n"
            f"  Liveness Score: {liveness}\n"
            f"  Latency:        {self.latency_ms:.0f}ms\n"
            f"  Explanation:    {self.explanation}\n"
            f"{'═' * 60}"
        )

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary."""
        result = {
            "verdict": self.verdict,
            "match_score": round(self.match_score, 4),
            "latency_ms": round(self.latency_ms, 1),
            "explanation": self.explanation,
        }
        if self.liveness_verdict:
            lv = self.liveness_verdict
            result["liveness_score"] = round(lv.score, 4)
            # Stable 4-key breakdown; absent signals are null (e.g. single image).
            breakdown = {"minifas": None, "texture": None, "temporal": None, "rppg": None}
            if getattr(lv, "minifas_result", None):
                breakdown["minifas"] = round(lv.minifas_result.score, 4)
            if lv.texture_result:
                breakdown["texture"] = round(lv.texture_result.score, 4)
            if lv.temporal_result:
                breakdown["temporal"] = round(lv.temporal_result.score, 4)
            if lv.rppg_result:
                breakdown["rppg"] = round(lv.rppg_result.score, 4)
            result["liveness_breakdown"] = breakdown
            result["signal_status"] = dict(lv.signal_status or {})
        if self.match_result:
            result["match_confidence"] = self.match_result.confidence
            result["match_needs_review"] = self.match_result.needs_review
        # Structured per-image quality (populated by verify(); may be absent).
        result["quality"] = {
            "id": self.details.get("id_quality_report"),
            "selfie": self.details.get("selfie_quality_report"),
        }
        result["details"] = self.details
        return result


@dataclass
class MatchOnlyResult:
    """Identity-match-only result (no liveness). Used by the /match endpoint."""

    match_result: Optional[MatchResult]
    match_score: float
    latency_ms: float
    explanation: str
    details: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        out = {
            "match_score": round(self.match_score, 4),
            "latency_ms": round(self.latency_ms, 1),
            "explanation": self.explanation,
        }
        if self.match_result:
            out["is_match"] = bool(self.match_result.is_match)
            out["match_confidence"] = self.match_result.confidence
            out["match_needs_review"] = bool(self.match_result.needs_review)
        else:
            out["is_match"] = False
            out["match_confidence"] = "none"
            out["match_needs_review"] = False
        out["quality"] = {
            "id": self.details.get("id_quality_report"),
            "selfie": self.details.get("selfie_quality_report"),
        }
        return out


@dataclass
class LivenessOnlyResult:
    """Liveness-only result (no identity match). Used by the /liveness endpoint."""

    liveness_verdict: Optional[LivenessVerdict]
    latency_ms: float
    explanation: str
    details: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        out = {
            "latency_ms": round(self.latency_ms, 1),
            "explanation": self.explanation,
        }
        lv = self.liveness_verdict
        if lv is not None:
            out["is_live"] = bool(lv.is_live)
            out["liveness_score"] = round(lv.score, 4)
            breakdown = {"minifas": None, "texture": None, "temporal": None, "rppg": None}
            if getattr(lv, "minifas_result", None):
                breakdown["minifas"] = round(lv.minifas_result.score, 4)
            if lv.texture_result:
                breakdown["texture"] = round(lv.texture_result.score, 4)
            if lv.temporal_result:
                breakdown["temporal"] = round(lv.temporal_result.score, 4)
            if lv.rppg_result:
                breakdown["rppg"] = round(lv.rppg_result.score, 4)
            out["liveness_breakdown"] = breakdown
            out["signal_status"] = dict(lv.signal_status or {})
        else:
            out["is_live"] = False
            out["liveness_score"] = None
            out["liveness_breakdown"] = {
                "minifas": None,
                "texture": None,
                "temporal": None,
                "rppg": None,
            }
            out["signal_status"] = {}
        out["quality"] = {"selfie": self.details.get("selfie_quality_report")}
        return out


class DwarpalaPipeline:
    """
    The complete Dwarpala verification pipeline.

    Usage:
        pipeline = DwarpalaPipeline()

        # Verify with image files
        result = pipeline.verify(
            id_image="path/to/id_card.jpg",
            selfie_source="path/to/selfie.mp4",
        )
        print(result.verdict)  # "ACCEPT"

        # Verify with numpy arrays
        result = pipeline.verify(
            id_image=id_array,
            selfie_source=selfie_frames,
        )
    """

    def __init__(
        self,
        detector_backend: str = "scrfd",
        embedding_backend: str = "insightface",
        match_threshold: float = 0.45,
        liveness_threshold: float = 0.5,
        review_band: float = 0.1,
        enable_liveness: bool = True,
        weights_path: Optional[str] = None,
        device: str = "auto",
        allow_untrained: bool = False,
        model_dir: Optional[Path] = None,
        # ID image quality thresholds (looser — ID photos are low-res, glossy)
        id_min_blur: float = 20.0,
        id_min_brightness: float = 30.0,
        id_max_brightness: float = 240.0,
        id_min_face_size: int = 40,
    ):
        """
        Args:
            detector_backend: Face detection backend.
            embedding_backend: 'insightface' (default) or 'vit'.
            match_threshold: Face similarity threshold.
            liveness_threshold: Liveness score threshold.
            review_band: Uncertainty band for MANUAL_REVIEW.
            enable_liveness: Whether to run liveness checks.
            weights_path: Path to model weights.
            device: Compute device.
            allow_untrained: Allow ViT backend without face weights.
            model_dir: Directory for model files.
            id_min_blur: Looser blur threshold for ID photos.
            id_min_brightness: Looser brightness min for ID photos.
            id_max_brightness: Looser brightness max for ID photos.
            id_min_face_size: Looser face size min for ID photos.
        """
        logger.info("Initializing Dwarpala Pipeline...")
        init_start = time.time()

        # Phase 1: Kavach (Shield — Pre-processing)
        self.detector = FaceDetector(
            backend=detector_backend,
            max_faces=1,
        )
        self.aligner = FaceAligner(output_size=(112, 112))
        # Standard quality gate for selfie
        self.quality = QualityAssessor()
        # Looser quality gate for ID document images
        # Rationale: ID card photos are low-res, often glossy/grainy,
        # and may have moiré patterns. Calibrated from empirical testing.
        self.id_quality = QualityAssessor(
            min_blur_score=id_min_blur,
            min_brightness=id_min_brightness,
            max_brightness=id_max_brightness,
            min_face_size=id_min_face_size,
            min_landmark_confidence=0.5,
        )

        # Phase 2: Swarupa (True Form — Identity)
        self.extractor = EmbeddingExtractor(
            backend=embedding_backend,
            weights_path=weights_path,
            device=device,
            allow_untrained=allow_untrained,
            model_dir=model_dir,
        )
        self.matcher = FaceMatcher(
            threshold=match_threshold,
            review_band=review_band,
        )

        # Phase 3: Prana (Life Force — Liveness)
        self.enable_liveness = enable_liveness
        if enable_liveness:
            self.liveness = LivenessFusionGate(
                enable_texture=True,
                enable_temporal=True,
                enable_rppg=True,
                enable_minifas=True,
                model_dir=model_dir,
            )
        else:
            self.liveness = None

        # Thresholds
        self.match_threshold = match_threshold
        self.liveness_threshold = liveness_threshold
        self.review_band = review_band

        init_time = (time.time() - init_start) * 1000
        logger.info(f"Pipeline initialized in {init_time:.0f}ms")

    def verify(
        self,
        id_image: Union[str, Path, np.ndarray],
        selfie_source: Union[str, Path, np.ndarray, List[np.ndarray]],
    ) -> VerificationResult:
        """
        Run the full verification pipeline.

        Args:
            id_image: ID document image (path or RGB array).
            selfie_source: Selfie image/video (path, RGB array, or list of frames).

        Returns:
            VerificationResult with verdict and detailed scores.
        """
        start_time = time.time()
        details = {}

        try:
            # ═══ STEP 1: Load & Preprocess ═══
            id_img = self._load_image(id_image)
            selfie_img, selfie_frames = self._load_selfie(selfie_source)

            # ═══ STEP 2: Face Detection ═══
            id_detection = self.detector.detect_largest(id_img)
            if id_detection is None:
                return self._make_reject(start_time, "No face detected in ID document.")

            selfie_detection = self.detector.detect_largest(selfie_img)
            if selfie_detection is None:
                return self._make_reject(start_time, "No face detected in selfie.")

            details["id_confidence"] = id_detection.confidence
            details["selfie_confidence"] = selfie_detection.confidence

            # ═══ STEP 3: Quality Check ═══
            # Use LOOSER thresholds for ID image (low-res, glossy, moiré)
            id_quality = self.id_quality.assess(id_img, id_detection)
            selfie_quality = self.quality.assess(selfie_img, selfie_detection)

            details["id_quality"] = str(id_quality)
            details["selfie_quality"] = str(selfie_quality)
            # Structured copies for API serialization (non-blocking, additive).
            details["id_quality_report"] = id_quality.to_dict()
            details["selfie_quality_report"] = selfie_quality.to_dict()

            if not selfie_quality.is_acceptable:
                return self._make_reject(
                    start_time,
                    f"Selfie quality insufficient: {', '.join(selfie_quality.issues)}",
                )

            # ID quality is logged but NOT used to reject — ID photos are
            # often low-quality by nature. The matching model handles this.
            if not id_quality.is_acceptable:
                logger.warning(
                    f"ID quality below threshold (non-blocking): " f"{', '.join(id_quality.issues)}"
                )

            # ═══ STEP 4: Face Alignment ═══
            id_aligned = self.aligner.align(id_img, id_detection.landmarks)
            selfie_aligned = self.aligner.align(selfie_img, selfie_detection.landmarks)

            # ═══ STEP 5: Identity Matching (Swarupa) ═══
            id_embedding = self.extractor.extract(id_aligned)
            selfie_embedding = self.extractor.extract(selfie_aligned)
            match_result = self.matcher.match(id_embedding, selfie_embedding)

            details["match_similarity"] = match_result.similarity

            # ═══ STEP 6: Liveness Detection (Prana) ═══
            liveness_verdict = None
            if self.enable_liveness and self.liveness is not None:
                # Compute bbox in (x, y, w, h) format from detection (x1, y1, x2, y2)
                bbox_xywh = None
                if selfie_detection.bbox is not None:
                    x1, y1, x2, y2 = selfie_detection.bbox.astype(int)
                    bbox_xywh = (x1, y1, max(1, x2 - x1), max(1, y2 - y1))

                liveness_verdict = self.liveness.analyze(
                    face_image=selfie_aligned,
                    original_image=selfie_img,
                    bbox=bbox_xywh,
                    video_frames=selfie_frames,
                )
                details["liveness_score"] = liveness_verdict.score

            # ═══ STEP 7: Final Decision ═══
            verdict, explanation = self._make_decision(match_result, liveness_verdict)

            latency_ms = (time.time() - start_time) * 1000

            result = VerificationResult(
                verdict=verdict,
                match_score=match_result.similarity,
                match_result=match_result,
                liveness_verdict=liveness_verdict,
                latency_ms=latency_ms,
                explanation=explanation,
                details=details,
            )

            logger.info(str(result))
            return result

        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            return self._make_reject(start_time, f"System error: {str(e)}")

    def match_only(
        self,
        id_image: Union[str, Path, np.ndarray],
        selfie_image: Union[str, Path, np.ndarray],
    ) -> MatchOnlyResult:
        """
        Run identity matching only (detect → quality → align → embed → match),
        with NO liveness. Reuses the exact same components as ``verify`` so there
        is a single source of truth for detection, embedding, and matching.

        Args:
            id_image: ID document image (path or RGB array).
            selfie_image: Selfie image (path or RGB array). Image only.

        Returns:
            MatchOnlyResult with the match score and per-image quality.
        """
        start_time = time.time()
        details: Dict = {}

        id_img = self._load_image(id_image)
        selfie_img = self._load_image(selfie_image)

        id_detection = self.detector.detect_largest(id_img)
        if id_detection is None:
            return MatchOnlyResult(
                None,
                0.0,
                (time.time() - start_time) * 1000,
                "No face detected in ID document.",
                details,
            )
        selfie_detection = self.detector.detect_largest(selfie_img)
        if selfie_detection is None:
            return MatchOnlyResult(
                None,
                0.0,
                (time.time() - start_time) * 1000,
                "No face detected in selfie.",
                details,
            )

        details["id_quality_report"] = self.id_quality.assess(id_img, id_detection).to_dict()
        details["selfie_quality_report"] = self.quality.assess(
            selfie_img, selfie_detection
        ).to_dict()

        id_aligned = self.aligner.align(id_img, id_detection.landmarks)
        selfie_aligned = self.aligner.align(selfie_img, selfie_detection.landmarks)
        id_embedding = self.extractor.extract(id_aligned)
        selfie_embedding = self.extractor.extract(selfie_aligned)
        match_result = self.matcher.match(id_embedding, selfie_embedding)

        if match_result.is_match:
            explanation = f"Face match confirmed (similarity={match_result.similarity:.3f})."
        else:
            explanation = (
                f"Face mismatch (similarity={match_result.similarity:.3f}, "
                f"threshold={self.match_threshold})."
            )

        return MatchOnlyResult(
            match_result=match_result,
            match_score=match_result.similarity,
            latency_ms=(time.time() - start_time) * 1000,
            explanation=explanation,
            details=details,
        )

    def liveness_only(
        self,
        selfie_source: Union[str, Path, np.ndarray, List[np.ndarray]],
    ) -> LivenessOnlyResult:
        """
        Run liveness detection only (no identity match). Reuses the exact same
        detector, aligner, and fusion gate as ``verify`` — single source of truth.

        Args:
            selfie_source: Selfie image/video (path, RGB array, or frame list).

        Returns:
            LivenessOnlyResult with the fused liveness verdict and breakdown.
        """
        start_time = time.time()
        details: Dict = {}

        selfie_img, selfie_frames = self._load_selfie(selfie_source)
        if selfie_img is None:
            return LivenessOnlyResult(
                None,
                (time.time() - start_time) * 1000,
                "No usable selfie frame.",
                details,
            )

        selfie_detection = self.detector.detect_largest(selfie_img)
        if selfie_detection is None:
            return LivenessOnlyResult(
                None,
                (time.time() - start_time) * 1000,
                "No face detected in selfie.",
                details,
            )

        details["selfie_quality_report"] = self.quality.assess(
            selfie_img, selfie_detection
        ).to_dict()

        if not (self.enable_liveness and self.liveness is not None):
            return LivenessOnlyResult(
                None,
                (time.time() - start_time) * 1000,
                "Liveness detection is disabled on this pipeline.",
                details,
            )

        selfie_aligned = self.aligner.align(selfie_img, selfie_detection.landmarks)
        bbox_xywh = None
        if selfie_detection.bbox is not None:
            x1, y1, x2, y2 = selfie_detection.bbox.astype(int)
            bbox_xywh = (x1, y1, max(1, x2 - x1), max(1, y2 - y1))

        liveness_verdict = self.liveness.analyze(
            face_image=selfie_aligned,
            original_image=selfie_img,
            bbox=bbox_xywh,
            video_frames=selfie_frames,
        )

        return LivenessOnlyResult(
            liveness_verdict=liveness_verdict,
            latency_ms=(time.time() - start_time) * 1000,
            explanation=liveness_verdict.explanation,
            details=details,
        )

    def _load_image(self, source: Union[str, Path, np.ndarray]) -> np.ndarray:
        """Load image from path or pass through if already numpy."""
        if isinstance(source, np.ndarray):
            return source
        return load_image(source, color="rgb")

    def _load_selfie(self, source: Union[str, Path, np.ndarray, List[np.ndarray]]) -> tuple:
        """Load selfie: returns (first_frame, list_of_frames_or_None)."""
        if isinstance(source, list):
            # List of frames
            frames = source
            first_frame = frames[0] if frames else None
            return first_frame, frames
        elif isinstance(source, np.ndarray):
            return source, None
        else:
            path = Path(source)
            if path.suffix.lower() in (".mp4", ".avi", ".mov", ".webm"):
                frames, _ = read_video_frames(str(path), max_frames=30)
                if not frames:
                    raise ValueError(
                        f"No frames read from video: {path}. "
                        f"File may be corrupted or use an unsupported codec."
                    )
                rgb_frames = frames_to_rgb(frames)
                return rgb_frames[0] if rgb_frames else None, rgb_frames
            else:
                img = load_image(path, color="rgb")
                return img, None

    def _make_decision(
        self,
        match_result: MatchResult,
        liveness_verdict: Optional[LivenessVerdict],
    ) -> tuple:
        """
        Make final Accept/Reject/Review decision.
        """
        explanations = []

        # Identity check
        if not match_result.is_match:
            explanations.append(
                f"Face mismatch (similarity={match_result.similarity:.3f}, "
                f"threshold={self.match_threshold})"
            )
            return "REJECT", ". ".join(explanations)

        explanations.append(f"Face match confirmed (similarity={match_result.similarity:.3f})")

        # Liveness check
        if liveness_verdict is not None:
            if not liveness_verdict.is_live:
                explanations.append(
                    f"Spoof detected (liveness={liveness_verdict.score:.3f}). "
                    f"{liveness_verdict.explanation}"
                )
                return "REJECT", ". ".join(explanations)
            explanations.append(f"Live subject confirmed (liveness={liveness_verdict.score:.3f})")

        # Check if needs manual review
        if match_result.needs_review:
            explanations.append("Borderline match score — recommend manual review")
            return "MANUAL_REVIEW", ". ".join(explanations)

        return "ACCEPT", ". ".join(explanations)

    def _make_reject(self, start_time: float, reason: str) -> VerificationResult:
        """Create a reject result."""
        return VerificationResult(
            verdict="REJECT",
            match_score=0.0,
            match_result=None,
            liveness_verdict=None,
            latency_ms=(time.time() - start_time) * 1000,
            explanation=reason,
        )

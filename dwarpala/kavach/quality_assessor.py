"""
Quality Assessment module for Dwarpala.
Evaluates whether a face image is of sufficient quality for verification.

Checks: blur, brightness, face size, landmark confidence, occlusion.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional

from dwarpala.utils.logger import get_logger

logger = get_logger("kavach.quality")


@dataclass
class QualityReport:
    """Quality assessment results for a face image."""

    is_acceptable: bool
    blur_score: float  # Higher = sharper
    brightness: float  # Mean pixel value (0-255)
    face_width: int
    face_height: int
    landmark_confidence: float
    issues: list  # List of quality issue strings

    def __str__(self):
        status = "✅ PASS" if self.is_acceptable else "❌ FAIL"
        return (
            f"Quality {status} | blur={self.blur_score:.1f} "
            f"bright={self.brightness:.0f} face={self.face_width}x{self.face_height} "
            f"conf={self.landmark_confidence:.2f}"
            + (f" | Issues: {', '.join(self.issues)}" if self.issues else "")
        )


class QualityAssessor:
    """
    Assesses whether a detected face meets minimum quality requirements
    for reliable biometric verification.

    Poor quality images (blurry, too dark, too small) degrade both
    face matching and liveness detection accuracy.

    Usage:
        assessor = QualityAssessor()
        report = assessor.assess(face_image, detection)
    """

    def __init__(
        self,
        min_blur_score: float = 50.0,
        min_brightness: float = 40.0,
        max_brightness: float = 220.0,
        min_face_size: int = 80,
        min_landmark_confidence: float = 0.8,
    ):
        """
        Args:
            min_blur_score: Minimum Laplacian variance (below = too blurry).
            min_brightness: Minimum mean brightness (below = too dark).
            max_brightness: Maximum mean brightness (above = overexposed).
            min_face_size: Minimum face width in pixels.
            min_landmark_confidence: Minimum detection confidence.
        """
        self.min_blur_score = min_blur_score
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.min_face_size = min_face_size
        self.min_landmark_confidence = min_landmark_confidence

        logger.info(
            f"QualityAssessor: blur>{min_blur_score}, "
            f"bright=[{min_brightness},{max_brightness}], "
            f"min_face={min_face_size}px"
        )

    def assess(
        self,
        face_image: np.ndarray,
        detection=None,
    ) -> QualityReport:
        """
        Run quality assessment on a face image.

        Args:
            face_image: Face image (RGB, can be cropped or aligned).
            detection: Optional FaceDetection with confidence and bbox info.

        Returns:
            QualityReport with detailed scores and pass/fail verdict.
        """
        issues = []

        # 1. Blur detection (Laplacian variance)
        blur_score = self._compute_blur(face_image)
        if blur_score < self.min_blur_score:
            issues.append(f"Too blurry (score={blur_score:.1f}, min={self.min_blur_score})")

        # 2. Brightness check
        brightness = self._compute_brightness(face_image)
        if brightness < self.min_brightness:
            issues.append(f"Too dark (brightness={brightness:.0f})")
        elif brightness > self.max_brightness:
            issues.append(f"Overexposed (brightness={brightness:.0f})")

        # 3. Face size check
        face_h, face_w = face_image.shape[:2]
        if detection is not None:
            face_w, face_h = detection.face_size
        if min(face_w, face_h) < self.min_face_size:
            issues.append(f"Face too small ({face_w}x{face_h}px)")

        # 4. Landmark confidence
        confidence = detection.confidence if detection else 1.0
        if confidence < self.min_landmark_confidence:
            issues.append(f"Low confidence ({confidence:.2f})")

        # 5. Simple occlusion check (eye region darkness asymmetry)
        occlusion_issue = self._check_occlusion(face_image)
        if occlusion_issue:
            issues.append(occlusion_issue)

        is_acceptable = len(issues) == 0

        report = QualityReport(
            is_acceptable=is_acceptable,
            blur_score=blur_score,
            brightness=brightness,
            face_width=face_w,
            face_height=face_h,
            landmark_confidence=confidence,
            issues=issues,
        )

        logger.info(str(report))
        return report

    def _compute_blur(self, image: np.ndarray) -> float:
        """Compute blur score using Laplacian variance."""
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def _compute_brightness(self, image: np.ndarray) -> float:
        """Compute average brightness."""
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        return float(np.mean(gray))

    def _check_occlusion(self, face_image: np.ndarray) -> Optional[str]:
        """
        Simple occlusion detection.
        Checks if major face regions have unexpected uniformity
        (could indicate sunglasses, mask, hand covering face).
        """
        h, w = face_image.shape[:2]
        if image_ndim := face_image.ndim == 3:
            gray = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
        else:
            gray = face_image

        # Check eye region
        eye_region = gray[int(0.2 * h) : int(0.45 * h), int(0.1 * w) : int(0.9 * w)]
        eye_std = np.std(eye_region)

        # Very low std in eye region suggests sunglasses/occlusion
        if eye_std < 10.0:
            return "Possible eye region occlusion (sunglasses?)"

        # Check lower face (mouth/nose)
        lower_region = gray[int(0.55 * h) : int(0.9 * h), int(0.2 * w) : int(0.8 * w)]
        lower_std = np.std(lower_region)

        if lower_std < 8.0:
            return "Possible lower face occlusion (mask?)"

        return None

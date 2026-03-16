"""
Face Detection module for Dwarpala.
Wraps InsightFace's SCRFD/RetinaFace detector for robust face detection.

The detector identifies face bounding boxes and 5-point landmarks
(left eye, right eye, nose tip, left mouth corner, right mouth corner).
"""

import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from dwarpala.utils.logger import get_logger

logger = get_logger("kavach.detector")


@dataclass
class FaceDetection:
    """Result from face detection."""

    bbox: np.ndarray  # (x1, y1, x2, y2)
    landmarks: np.ndarray  # (5, 2) — 5 facial landmarks
    confidence: float
    face_size: Tuple[int, int] = field(init=False)  # (width, height)

    def __post_init__(self):
        x1, y1, x2, y2 = self.bbox.astype(int)
        self.face_size = (x2 - x1, y2 - y1)

    @property
    def area(self) -> int:
        return self.face_size[0] * self.face_size[1]


class FaceDetector:
    """
    Face detector using InsightFace SCRFD or a fallback OpenCV Haar cascade.

    Usage:
        detector = FaceDetector(backend="scrfd")
        detections = detector.detect(image)
        for det in detections:
            print(det.bbox, det.confidence, det.landmarks)
    """

    SUPPORTED_BACKENDS = ("scrfd", "retinaface", "opencv")

    def __init__(
        self,
        backend: str = "scrfd",
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.4,
        max_faces: int = 1,
    ):
        """
        Initialize the face detector.

        Args:
            backend: Detection backend ('scrfd', 'retinaface', or 'opencv' fallback).
            confidence_threshold: Minimum confidence to accept a detection.
            nms_threshold: Non-max suppression IoU threshold.
            max_faces: Maximum number of faces to return.
        """
        if backend not in self.SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unknown backend: {backend}. Supported: {self.SUPPORTED_BACKENDS}"
            )

        self.backend = backend
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.max_faces = max_faces
        self._model = None

        self._initialize_backend()

    def _initialize_backend(self):
        """Load the detection model."""
        if self.backend in ("scrfd", "retinaface"):
            try:
                from insightface.app import FaceAnalysis

                self._model = FaceAnalysis(
                    name="buffalo_l",
                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                )
                self._model.prepare(
                    ctx_id=0,
                    det_size=(640, 640),
                    det_thresh=self.confidence_threshold,
                )
                logger.info(f"Loaded InsightFace ({self.backend}) backend")
            except (ImportError, Exception) as e:
                logger.warning(
                    f"InsightFace not available ({e}), falling back to OpenCV"
                )
                self.backend = "opencv"
                self._initialize_opencv()
        else:
            self._initialize_opencv()

    def _initialize_opencv(self):
        """Initialize OpenCV Haar cascade as fallback detector."""
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(cascade_path)

        # Eye cascade for approximate landmark estimation
        eye_cascade_path = cv2.data.haarcascades + "haarcascade_eye.xml"
        self._eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

        logger.info("Loaded OpenCV Haar cascade fallback detector")

    def detect(self, image: np.ndarray) -> List[FaceDetection]:
        """
        Detect faces in an image.

        Args:
            image: Input image in RGB format (H, W, 3).

        Returns:
            List of FaceDetection objects, sorted by face area (largest first).
        """
        if image is None or image.size == 0:
            return []

        if self.backend in ("scrfd", "retinaface") and self._model is not None:
            return self._detect_insightface(image)
        else:
            return self._detect_opencv(image)

    def _detect_insightface(self, image: np.ndarray) -> List[FaceDetection]:
        """Detect using InsightFace."""
        # InsightFace expects BGR
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        faces = self._model.get(bgr)

        detections = []
        for face in faces:
            if face.det_score < self.confidence_threshold:
                continue

            det = FaceDetection(
                bbox=face.bbox.astype(np.float32),
                landmarks=face.kps.astype(np.float32),
                confidence=float(face.det_score),
            )
            detections.append(det)

        # Sort by area (largest first) and limit
        detections.sort(key=lambda d: d.area, reverse=True)
        detections = detections[: self.max_faces]

        logger.info(f"Detected {len(detections)} face(s) via InsightFace")
        return detections

    def _detect_opencv(self, image: np.ndarray) -> List[FaceDetection]:
        """Detect using OpenCV Haar cascade (fallback)."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        faces_rect = self._cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80),
        )

        detections = []
        for x, y, w, h in faces_rect:
            bbox = np.array([x, y, x + w, y + h], dtype=np.float32)

            # Estimate approximate 5-point landmarks from face geometry
            landmarks = self._estimate_landmarks(gray, x, y, w, h)

            det = FaceDetection(
                bbox=bbox,
                landmarks=landmarks,
                confidence=0.9,  # Haar cascade doesn't give explicit confidence
            )
            detections.append(det)

        detections.sort(key=lambda d: d.area, reverse=True)
        detections = detections[: self.max_faces]

        logger.info(f"Detected {len(detections)} face(s) via OpenCV")
        return detections

    def _estimate_landmarks(
        self, gray: np.ndarray, x: int, y: int, w: int, h: int
    ) -> np.ndarray:
        """
        Estimate 5-point landmarks from face geometry when proper
        landmark detector is not available.

        Landmarks: left_eye, right_eye, nose, left_mouth, right_mouth
        """
        # Try eye detection for better estimates
        face_roi = gray[y : y + h, x : x + w]
        eyes = self._eye_cascade.detectMultiScale(
            face_roi,
            scaleFactor=1.1,
            minNeighbors=10,
            minSize=(20, 20),
        )

        if len(eyes) >= 2:
            # Sort eyes by x-coordinate
            eyes = sorted(eyes, key=lambda e: e[0])
            left_eye = (
                x + eyes[0][0] + eyes[0][2] // 2,
                y + eyes[0][1] + eyes[0][3] // 2,
            )
            right_eye = (
                x + eyes[1][0] + eyes[1][2] // 2,
                y + eyes[1][1] + eyes[1][3] // 2,
            )
        else:
            # Geometric estimation
            left_eye = (x + int(0.3 * w), y + int(0.35 * h))
            right_eye = (x + int(0.7 * w), y + int(0.35 * h))

        nose = (x + int(0.5 * w), y + int(0.55 * h))
        left_mouth = (x + int(0.35 * w), y + int(0.75 * h))
        right_mouth = (x + int(0.65 * w), y + int(0.75 * h))

        return np.array(
            [left_eye, right_eye, nose, left_mouth, right_mouth],
            dtype=np.float32,
        )

    def detect_largest(self, image: np.ndarray) -> Optional[FaceDetection]:
        """
        Detect the single largest face in an image.

        Args:
            image: Input image in RGB format.

        Returns:
            Largest FaceDetection, or None if no face found.
        """
        detections = self.detect(image)
        return detections[0] if detections else None

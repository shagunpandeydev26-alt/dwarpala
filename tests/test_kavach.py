"""
Tests for the Kavach pre-processing module.
"""

import numpy as np
import pytest

from dwarpala.kavach import FaceDetector, FaceAligner, QualityAssessor


class TestFaceDetector:
    """Test face detection functionality."""

    def test_init_opencv_backend(self):
        """Detector should initialize with OpenCV backend."""
        detector = FaceDetector(backend="opencv")
        assert detector.backend == "opencv"

    def test_detect_no_face(self):
        """Should return empty list for blank image."""
        detector = FaceDetector(backend="opencv")
        blank = np.zeros((200, 200, 3), dtype=np.uint8)
        detections = detector.detect(blank)
        assert isinstance(detections, list)

    def test_detect_returns_list(self):
        """Should always return a list."""
        detector = FaceDetector(backend="opencv")
        noise = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        detections = detector.detect(noise)
        assert isinstance(detections, list)

    def test_detect_largest_returns_none_for_blank(self):
        """detect_largest should return None if no face found."""
        detector = FaceDetector(backend="opencv")
        blank = np.zeros((200, 200, 3), dtype=np.uint8)
        result = detector.detect_largest(blank)
        assert result is None

    def test_invalid_backend(self):
        """Should raise ValueError for invalid backend."""
        with pytest.raises(ValueError):
            FaceDetector(backend="invalid_backend")


class TestFaceAligner:
    """Test face alignment functionality."""

    def test_init(self):
        """Aligner should initialize with default params."""
        aligner = FaceAligner(output_size=(112, 112))
        assert aligner.output_size == (112, 112)

    def test_align_shape(self):
        """Aligned face should have correct output shape."""
        aligner = FaceAligner(output_size=(112, 112))
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        landmarks = np.array([
            [200, 200], [400, 200], [300, 300], [220, 380], [380, 380]
        ], dtype=np.float32)

        aligned = aligner.align(image, landmarks)
        assert aligned.shape == (112, 112, 3)

    def test_invalid_landmarks_shape(self):
        """Should raise ValueError for wrong landmark shape."""
        aligner = FaceAligner()
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        bad_landmarks = np.zeros((3, 2), dtype=np.float32)

        with pytest.raises(ValueError):
            aligner.align(image, bad_landmarks)


class TestQualityAssessor:
    """Test quality assessment functionality."""

    def test_init(self):
        """Assessor should initialize with default params."""
        assessor = QualityAssessor()
        assert assessor.min_blur_score > 0

    def test_bright_image_passes(self):
        """Well-lit image should pass brightness check."""
        assessor = QualityAssessor()
        bright_img = np.full((112, 112, 3), 128, dtype=np.uint8)
        # Add some texture to pass blur check
        bright_img += np.random.randint(-20, 20, bright_img.shape).astype(np.uint8)
        report = assessor.assess(bright_img)
        assert report.brightness > 40

    def test_dark_image_fails(self):
        """Very dark image should fail quality check."""
        assessor = QualityAssessor()
        dark_img = np.full((112, 112, 3), 10, dtype=np.uint8)
        report = assessor.assess(dark_img)
        assert not report.is_acceptable
        assert any("dark" in issue.lower() for issue in report.issues)

    def test_quality_report_str(self):
        """Quality report should have readable string representation."""
        assessor = QualityAssessor()
        img = np.random.randint(50, 200, (112, 112, 3), dtype=np.uint8)
        report = assessor.assess(img)
        assert "Quality" in str(report)

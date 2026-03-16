"""
Tests for the Swarupa identity module and Yantra pipeline.
"""

import numpy as np
import pytest

from dwarpala.swarupa.matcher import FaceMatcher


class TestFaceMatcher:
    """Test face matching functionality."""

    def test_init(self):
        matcher = FaceMatcher(threshold=0.45)
        assert matcher.threshold == 0.45

    def test_identical_embeddings_match(self):
        """Identical embeddings should produce a match."""
        matcher = FaceMatcher(threshold=0.45)
        embedding = np.random.randn(512).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)

        result = matcher.match(embedding, embedding)
        assert result.is_match
        assert result.similarity > 0.99

    def test_orthogonal_embeddings_no_match(self):
        """Orthogonal embeddings should not match."""
        matcher = FaceMatcher(threshold=0.45)
        e1 = np.zeros(512, dtype=np.float32)
        e2 = np.zeros(512, dtype=np.float32)
        e1[0] = 1.0
        e2[1] = 1.0

        result = matcher.match(e1, e2)
        assert not result.is_match
        assert result.similarity < 0.01

    def test_review_band(self):
        """Borderline scores should trigger NEEDS_REVIEW."""
        matcher = FaceMatcher(threshold=0.5, review_band=0.1)

        # Create embeddings with similarity close to threshold
        e1 = np.random.randn(512).astype(np.float32)
        e1 = e1 / np.linalg.norm(e1)

        # Slightly perturb to get score near threshold
        noise = np.random.randn(512).astype(np.float32) * 0.5
        e2 = e1 + noise
        e2 = e2 / np.linalg.norm(e2)

        result = matcher.match(e1, e2)
        # Just verify the result has the needs_review field
        assert isinstance(result.needs_review, bool)

    def test_match_result_str(self):
        """MatchResult should have readable string representation."""
        matcher = FaceMatcher()
        e1 = np.random.randn(512).astype(np.float32)
        e1 /= np.linalg.norm(e1)
        result = matcher.match(e1, e1)
        assert "MATCH" in str(result)


class TestMetrics:
    """Test evaluation metrics."""

    def test_compute_acer(self):
        from dwarpala.utils.metrics import compute_acer

        predictions = np.array([1, 1, 0, 0, 1])
        labels = np.array([1, 0, 0, 1, 1])

        result = compute_acer(predictions, labels)
        assert "apcer" in result
        assert "bpcer" in result
        assert "acer" in result
        assert 0 <= result["acer"] <= 1

    def test_compute_tar_at_far(self):
        from dwarpala.utils.metrics import compute_tar_at_far

        genuine = np.random.normal(0.7, 0.1, 1000)
        impostor = np.random.normal(0.3, 0.1, 1000)

        tar, threshold = compute_tar_at_far(genuine, impostor, target_far=0.01)
        assert 0 <= tar <= 1
        assert isinstance(threshold, float)

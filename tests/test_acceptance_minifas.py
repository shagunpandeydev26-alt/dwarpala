"""
Acceptance test for MiniFASNet liveness detection.

Verifies that the model loads, produces different scores for different images,
and ranks live images above spoof images.
"""

import pytest
import cv2
import numpy as np
from pathlib import Path

from dwarpala.prana.minifas_analyzer import MiniFASAnalyzer


TEST_IMAGES = Path(__file__).parent / "testimg"


@pytest.fixture(scope="module")
def analyzer():
    try:
        return MiniFASAnalyzer()
    except Exception:
        pytest.skip("MiniFASNet models not available")


@pytest.fixture(scope="module")
def test_data():
    """Load test images and detect faces."""
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=-1, det_size=(640, 640))

    samples = {}
    for name in ["selfie1", "selfie2", "printed", "screencapture"]:
        path = TEST_IMAGES / f"{name}.jpeg"
        if not path.exists():
            pytest.skip(f"Test image not found: {path}")
        img = cv2.imread(str(path))
        faces = app.get(img)
        if not faces:
            pytest.skip(f"No face detected in {name}")
        f = faces[0]
        x1, y1, x2, y2 = f.bbox.astype(int)
        samples[name] = (img, (x1, y1, x2 - x1, y2 - y1))
    return samples


class TestMiniFASAcceptance:
    """Real-world acceptance tests for MiniFASNet."""

    def test_models_loaded(self, analyzer):
        assert analyzer.models_loaded

    def test_scores_differ_per_image(self, analyzer, test_data):
        """Each image should produce a different score."""
        scores = {}
        for name, (img, bbox) in test_data.items():
            result = analyzer.analyze(img, bbox)
            scores[name] = result.score
        unique = set(round(s, 4) for s in scores.values())
        assert len(unique) > 1, (
            f"All images produced same score: {scores}"
        )

    def test_selfie_scores_higher_than_spoof(self, analyzer, test_data):
        """Live selfies should score higher than printed/screen captures."""
        selfie_scores = []
        spoof_scores = []
        for name, (img, bbox) in test_data.items():
            result = analyzer.analyze(img, bbox)
            if "selfie" in name:
                selfie_scores.append(result.score)
            else:
                spoof_scores.append(result.score)

        avg_selfie = float(np.mean(selfie_scores))
        avg_spoof = float(np.mean(spoof_scores))
        assert avg_selfie > avg_spoof, (
            f"Selfie scores ({avg_selfie:.3f}) should exceed "
            f"spoof scores ({avg_spoof:.3f})"
        )

    def test_score_range(self, analyzer, test_data):
        """All scores should be in [0, 1]."""
        for name, (img, bbox) in test_data.items():
            result = analyzer.analyze(img, bbox)
            assert 0.0 <= result.score <= 1.0, (
                f"{name}: score {result.score} out of range"
            )

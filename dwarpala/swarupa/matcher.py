"""
Face Matcher module for Dwarpala.
Compares face embeddings and produces match/no-match decisions
with confidence scores and explanations.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from dwarpala.utils.logger import get_logger

logger = get_logger("swarupa.matcher")


@dataclass
class MatchResult:
    """Result of a face matching comparison."""

    is_match: bool
    similarity: float  # Cosine similarity [0, 1]
    confidence: str  # "high", "medium", "low"
    needs_review: bool  # True if score is in the uncertain band

    def __str__(self):
        status = "✅ MATCH" if self.is_match else "❌ NO MATCH"
        return (
            f"{status} | similarity={self.similarity:.4f} "
            f"| confidence={self.confidence}"
            + (" ⚠️ NEEDS REVIEW" if self.needs_review else "")
        )


class FaceMatcher:
    """
    Compares face embeddings to determine if they belong to the same person.

    Uses cosine similarity with configurable thresholds and a
    "review band" for borderline cases that should be escalated
    to manual verification.

    Usage:
        matcher = FaceMatcher(threshold=0.45)
        result = matcher.match(embedding_a, embedding_b)
        print(result.is_match, result.similarity)
    """

    def __init__(
        self,
        threshold: float = 0.45,
        high_confidence_threshold: float = 0.65,
        review_band: float = 0.1,
    ):
        """
        Args:
            threshold: Minimum similarity for a match.
            high_confidence_threshold: Above this = high confidence match.
            review_band: If similarity is within this range of threshold,
                         flag for manual review.
        """
        self.threshold = threshold
        self.high_confidence_threshold = high_confidence_threshold
        self.review_band = review_band

        logger.info(
            f"FaceMatcher: threshold={threshold}, "
            f"high_conf={high_confidence_threshold}, "
            f"review_band=±{review_band}"
        )

    def match(
        self,
        embedding_a: np.ndarray,
        embedding_b: np.ndarray,
    ) -> MatchResult:
        """
        Compare two face embeddings.

        Args:
            embedding_a: First face embedding (512-D, L2-normalized).
            embedding_b: Second face embedding (512-D, L2-normalized).

        Returns:
            MatchResult with verdict and confidence assessment.
        """
        similarity = self._cosine_similarity(embedding_a, embedding_b)

        is_match = similarity >= self.threshold

        # Determine confidence level
        if similarity >= self.high_confidence_threshold:
            confidence = "high"
        elif similarity >= self.threshold:
            confidence = "medium"
        elif similarity >= self.threshold - self.review_band:
            confidence = "low"
        else:
            confidence = "low"

        # Flag for review if in the uncertainty band
        needs_review = abs(similarity - self.threshold) < self.review_band

        result = MatchResult(
            is_match=is_match,
            similarity=float(similarity),
            confidence=confidence,
            needs_review=needs_review,
        )

        logger.info(str(result))
        return result

    def _cosine_similarity(
        self,
        vec_a: np.ndarray,
        vec_b: np.ndarray,
    ) -> float:
        """
        Compute cosine similarity between two vectors.
        Embeddings should already be L2-normalized, but we handle
        the general case.
        """
        dot = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(dot / (norm_a * norm_b))

    def find_best_match(
        self,
        query_embedding: np.ndarray,
        gallery_embeddings: np.ndarray,
        gallery_ids: Optional[list] = None,
    ) -> dict:
        """
        Find the best matching identity from a gallery.

        Args:
            query_embedding: Query face embedding (512-D).
            gallery_embeddings: Gallery embeddings (N, 512).
            gallery_ids: Optional list of identity IDs.

        Returns:
            Dictionary with best match info.
        """
        similarities = gallery_embeddings @ query_embedding

        best_idx = int(np.argmax(similarities))
        best_sim = float(similarities[best_idx])

        result = self.match(query_embedding, gallery_embeddings[best_idx])

        return {
            "best_index": best_idx,
            "best_id": gallery_ids[best_idx] if gallery_ids else best_idx,
            "similarity": best_sim,
            "match_result": result,
        }

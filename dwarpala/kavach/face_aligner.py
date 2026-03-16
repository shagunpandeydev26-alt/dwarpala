"""
Face Alignment module for Dwarpala.
Performs affine transformation using 5-point landmarks to produce
a normalized, aligned face image suitable for embedding extraction.
"""

import cv2
import numpy as np
from typing import Tuple

from dwarpala.utils.logger import get_logger

logger = get_logger("kavach.aligner")

# Standard reference landmarks for ArcFace-style alignment (112x112)
# These coordinates define the "ideal" face geometry
ARCFACE_REFERENCE_112 = np.array(
    [
        [38.2946, 51.6963],  # Left eye
        [73.5318, 51.5014],  # Right eye
        [56.0252, 71.7366],  # Nose tip
        [41.5493, 92.3655],  # Left mouth corner
        [70.7299, 92.2041],  # Right mouth corner
    ],
    dtype=np.float32,
)


class FaceAligner:
    """
    Aligns detected faces to a canonical pose using affine transformation.
    This is critical for face recognition — alignment ensures structural
    consistency regardless of head pose, camera angle, or image resolution.

    Usage:
        aligner = FaceAligner(output_size=(112, 112))
        aligned = aligner.align(image, landmarks)
    """

    def __init__(
        self,
        output_size: Tuple[int, int] = (112, 112),
        reference: str = "arcface",
    ):
        """
        Args:
            output_size: (width, height) of the aligned output image.
            reference: Reference landmark set ('arcface').
        """
        self.output_size = output_size
        self._reference = self._get_reference_landmarks(reference, output_size)
        logger.info(f"FaceAligner initialized: output={output_size}, ref={reference}")

    def _get_reference_landmarks(
        self,
        name: str,
        output_size: Tuple[int, int],
    ) -> np.ndarray:
        """
        Get reference landmark coordinates scaled to output size.
        """
        if name == "arcface":
            ref = ARCFACE_REFERENCE_112.copy()
            # Scale if output size is different from 112x112
            scale_x = output_size[0] / 112.0
            scale_y = output_size[1] / 112.0
            ref[:, 0] *= scale_x
            ref[:, 1] *= scale_y
            return ref
        else:
            raise ValueError(f"Unknown reference: {name}")

    def align(
        self,
        image: np.ndarray,
        landmarks: np.ndarray,
    ) -> np.ndarray:
        """
        Align a face using 5-point landmarks.

        Args:
            image: Input image (H, W, C) in RGB.
            landmarks: 5-point landmarks array of shape (5, 2).

        Returns:
            Aligned face image of shape (output_h, output_w, C).
        """
        if landmarks.shape != (5, 2):
            raise ValueError(
                f"Expected landmarks shape (5, 2), got {landmarks.shape}"
            )

        # Estimate similarity transform (rotation, scale, translation)
        transform_matrix = self._estimate_similarity_transform(
            landmarks, self._reference
        )

        # Apply affine warp
        aligned = cv2.warpAffine(
            image,
            transform_matrix,
            self.output_size,
            borderMode=cv2.BORDER_REPLICATE,
        )

        return aligned

    def _estimate_similarity_transform(
        self,
        src_points: np.ndarray,
        dst_points: np.ndarray,
    ) -> np.ndarray:
        """
        Estimate 2D similarity transformation matrix.
        Uses Umeyama's method for a least-squares estimate.

        Args:
            src_points: Source landmarks (N, 2).
            dst_points: Destination landmarks (N, 2).

        Returns:
            2x3 affine transformation matrix.
        """
        num = src_points.shape[0]
        dim = 2

        # Center points
        src_mean = np.mean(src_points, axis=0)
        dst_mean = np.mean(dst_points, axis=0)

        src_demean = src_points - src_mean
        dst_demean = dst_points - dst_mean

        # Variance
        src_var = np.mean(np.sum(src_demean**2, axis=1))

        # Covariance matrix
        cov = (dst_demean.T @ src_demean) / num

        # SVD
        U, S, Vt = np.linalg.svd(cov)

        # Handle reflection
        det = np.linalg.det(U) * np.linalg.det(Vt)
        d = np.ones(dim, dtype=np.float64)
        if det < 0:
            d[dim - 1] = -1

        # Rotation
        T = np.eye(dim + 1, dtype=np.float64)
        R = U @ np.diag(d) @ Vt

        # Scale
        scale = np.sum(S * d) / src_var

        # Translation
        T[:dim, :dim] = scale * R
        T[:dim, dim] = dst_mean - scale * R @ src_mean

        return T[:2, :]  # Return 2x3 for cv2.warpAffine

    def align_from_detection(
        self,
        image: np.ndarray,
        detection,
    ) -> np.ndarray:
        """
        Convenience method: align using a FaceDetection object.

        Args:
            image: Input image (RGB).
            detection: FaceDetection from face_detector.

        Returns:
            Aligned face image.
        """
        return self.align(image, detection.landmarks)

"""
Texture Analysis for liveness detection (Layer 1 — Static).

Uses Local Binary Patterns (LBP) and Frequency Domain analysis (FFT)
to detect the "pixel-grid" noise inherent in screens or the "flatness"
of printed paper attacks.

Real skin has characteristic micro-texture patterns (pores, fine lines)
that are absent in printed photos and have different frequency signatures
than digital screens.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List

from dwarpala.utils.logger import get_logger

logger = get_logger("prana.texture")


@dataclass
class TextureResult:
    """Result from texture-based liveness analysis."""

    score: float  # 0.0 = spoof, 1.0 = live
    lbp_score: float
    fft_score: float
    prediction: str  # "live" or "spoof"

    def __str__(self):
        status = "🟢 LIVE" if self.prediction == "live" else "🔴 SPOOF"
        return (
            f"Texture {status} | score={self.score:.3f} "
            f"(LBP={self.lbp_score:.3f}, FFT={self.fft_score:.3f})"
        )


class TextureAnalyzer:
    """
    Layer 1 of Prana liveness detection.

    Analyzes face texture using two complementary methods:
    1. Multi-scale LBP: Captures micro-texture patterns different between
       real skin and reproduced images (prints, screens)
    2. FFT Analysis: Detects periodic high-frequency patterns characteristic
       of screens (pixel grid) and printers (halftone dots)

    Usage:
        analyzer = TextureAnalyzer()
        result = analyzer.analyze(face_image_rgb)
    """

    def __init__(
        self,
        lbp_radii: List[int] = None,
        lbp_points_multiplier: int = 8,
        fft_threshold: float = 0.15,
        spoof_threshold: float = 0.5,
    ):
        """
        Args:
            lbp_radii: List of LBP radii for multi-scale analysis.
            lbp_points_multiplier: Points = radius * this value.
            fft_threshold: Threshold for FFT periodic pattern detection.
            spoof_threshold: Below this score = spoof.
        """
        self.lbp_radii = lbp_radii or [1, 2, 3]
        self.lbp_points_multiplier = lbp_points_multiplier
        self.fft_threshold = fft_threshold
        self.spoof_threshold = spoof_threshold

        logger.info(f"TextureAnalyzer: radii={self.lbp_radii}, " f"fft_thresh={fft_threshold}")

    def analyze(self, face_image: np.ndarray) -> TextureResult:
        """
        Analyze face texture for liveness.

        Args:
            face_image: Aligned face image (H, W, 3) in RGB.

        Returns:
            TextureResult with liveness scores.
        """
        gray = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)

        # Multi-scale LBP analysis
        lbp_score = self._compute_lbp_liveness(gray)

        # Frequency domain analysis
        fft_score = self._compute_fft_liveness(gray)

        # Combined score (weighted average)
        score = 0.6 * lbp_score + 0.4 * fft_score
        prediction = "live" if score >= self.spoof_threshold else "spoof"

        result = TextureResult(
            score=float(score),
            lbp_score=float(lbp_score),
            fft_score=float(fft_score),
            prediction=prediction,
        )

        logger.info(str(result))
        return result

    def get_fft_spectrum(self, face_image: np.ndarray) -> np.ndarray:
        """
        Read-only accessor for visualization: return the log-magnitude FFT
        spectrum the analyzer computes internally for the FFT liveness score.

        This recomputes the exact `magnitude` array used by
        `_compute_fft_liveness` (windowed 2D FFT, fftshift, log1p, max-normalized)
        WITHOUT affecting any score. Screens/prints show concentrated periodic
        peaks here; real skin shows a smoother natural fall-off.

        Args:
            face_image: Aligned face image (H, W, 3) in RGB.

        Returns:
            2D float32 magnitude spectrum in [0, 1], shape (H, W).
        """
        gray = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
        rows, cols = gray.shape
        window = np.outer(np.hanning(rows), np.hanning(cols))
        windowed = gray.astype(np.float64) * window
        f_shift = np.fft.fftshift(np.fft.fft2(windowed))
        magnitude = np.log1p(np.abs(f_shift))
        magnitude = magnitude / (magnitude.max() + 1e-10)
        return magnitude.astype(np.float32)

    def _compute_lbp_liveness(self, gray: np.ndarray) -> float:
        """
        Compute LBP-based liveness score.

        Real faces have rich, varied LBP patterns due to skin texture.
        Printed/screen faces have more uniform, periodic patterns.

        Higher entropy of LBP histogram = more texture variety = more likely real.
        """
        all_histograms = []

        for radius in self.lbp_radii:
            n_points = radius * self.lbp_points_multiplier
            lbp = self._compute_lbp(gray, radius, n_points)

            # Compute normalized histogram
            hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2, range=(0, n_points + 2))
            hist = hist.astype(np.float64)
            hist = hist / (hist.sum() + 1e-10)
            all_histograms.append(hist)

        # Concatenate multi-scale histograms
        feature = np.concatenate(all_histograms)

        # Compute entropy as liveness indicator
        entropy = -np.sum(feature * np.log2(feature + 1e-10))

        # Normalize entropy to [0, 1] range
        # Real faces typically have entropy > 4.0, spoofs < 3.5
        max_entropy = np.log2(len(feature))
        normalized_entropy = min(1.0, entropy / max_entropy)

        # Additional: variance of LBP values (real faces are more varied)
        lbp_variance = np.var(feature)
        variance_score = min(1.0, lbp_variance * 1000)

        return 0.7 * normalized_entropy + 0.3 * variance_score

    def _compute_lbp(self, gray: np.ndarray, radius: int, n_points: int) -> np.ndarray:
        """
        Compute Local Binary Pattern image.

        For each pixel, compare with n_points neighbors at given radius.
        The binary comparison result forms the LBP code.
        """
        rows, cols = gray.shape
        lbp = np.zeros_like(gray, dtype=np.uint32)

        for i in range(n_points):
            angle = 2.0 * np.pi * i / n_points
            dy = -radius * np.cos(angle)
            dx = radius * np.sin(angle)

            # Bilinear interpolation for sub-pixel positions
            fy, fx = int(np.floor(dy)), int(np.floor(dx))
            cy, cx = fy + 1, fx + 1

            # Boundary-safe indexing
            r_start = max(0, -min(fy, cy))
            r_end = min(rows, rows - max(fy, cy))
            c_start = max(0, -min(fx, cx))
            c_end = min(cols, cols - max(fx, cx))

            if r_start >= r_end or c_start >= c_end:
                continue

            # Interpolated neighbor value
            center = gray[r_start:r_end, c_start:c_end].astype(np.float64)

            rr = slice(r_start + fy, r_end + fy)
            rc = slice(c_start + fx, c_end + fx)

            # Simplified: use nearest neighbor for speed
            if 0 <= r_start + round(dy) < rows and 0 <= c_start + round(dx) < cols:
                rr = slice(
                    max(0, r_start + round(dy)),
                    min(rows, r_end + round(dy)),
                )
                rc = slice(
                    max(0, c_start + round(dx)),
                    min(cols, c_end + round(dx)),
                )

                if rr.stop - rr.start == r_end - r_start and rc.stop - rc.start == c_end - c_start:
                    neighbor = gray[rr, rc].astype(np.float64)
                    lbp[r_start:r_end, c_start:c_end] += (neighbor >= center).astype(np.uint32) << i

        return lbp

    def _compute_fft_liveness(self, gray: np.ndarray) -> float:
        """
        Compute FFT-based liveness score.

        Screens show characteristic periodic patterns in the frequency domain
        (pixel grid creates peaks at specific frequencies).
        Printed photos show halftone dot patterns.
        Real faces have a natural fall-off in high frequencies.

        Score: higher = more likely real.
        """
        # Apply windowing to reduce edge artifacts
        rows, cols = gray.shape
        window = np.outer(np.hanning(rows), np.hanning(cols))
        windowed = gray.astype(np.float64) * window

        # 2D FFT
        f_transform = np.fft.fft2(windowed)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.log1p(np.abs(f_shift))

        # Normalize magnitude spectrum
        magnitude = magnitude / (magnitude.max() + 1e-10)

        # Analyze high-frequency energy ratio
        center_r, center_c = rows // 2, cols // 2
        r_quarter = rows // 4
        c_quarter = cols // 4

        # Low frequency region (center)
        low_freq = magnitude[
            center_r - r_quarter : center_r + r_quarter,
            center_c - c_quarter : center_c + c_quarter,
        ]

        # High frequency region (periphery)
        total_energy = np.sum(magnitude)
        low_energy = np.sum(low_freq)
        high_energy = total_energy - low_energy

        # Ratio of high to total frequency energy
        # Real faces: natural texture → moderate high-freq energy
        # Screens: pixel grid → concentrated peaks in high freq
        # Prints: less high-freq energy (smoother)
        high_freq_ratio = high_energy / (total_energy + 1e-10)

        # Detect periodic peaks (characteristic of screens)
        # Real faces don't have sharp periodic peaks
        peak_score = self._detect_periodic_peaks(magnitude)

        # Combine: penalize both too-low (print) and peaked (screen) patterns
        # Ideal range for real faces: moderate high_freq_ratio, low peak_score
        if peak_score > self.fft_threshold:
            # Screen-like periodic pattern detected
            fft_score = max(0.0, 1.0 - peak_score * 3)
        elif high_freq_ratio < 0.1:
            # Too smooth (print-like)
            fft_score = high_freq_ratio * 5
        else:
            # Natural looking
            fft_score = min(1.0, 0.5 + high_freq_ratio)

        return float(fft_score)

    def _detect_periodic_peaks(self, magnitude: np.ndarray) -> float:
        """
        Detect periodic peaks in frequency spectrum.
        Screens produce characteristic peaks due to pixel grid repetition.

        Returns a score: higher = more periodic peaks detected.
        """
        rows, cols = magnitude.shape
        center_r, center_c = rows // 2, cols // 2

        # Analyze radial frequency distribution
        max_radius = min(center_r, center_c)
        radial_profile = np.zeros(max_radius)

        for r in range(max_radius):
            # Sample points on circle of radius r
            theta = np.linspace(0, 2 * np.pi, max(8, r * 4), endpoint=False)
            y_coords = (center_r + r * np.sin(theta)).astype(int)
            x_coords = (center_c + r * np.cos(theta)).astype(int)

            # Boundary check
            valid = (y_coords >= 0) & (y_coords < rows) & (x_coords >= 0) & (x_coords < cols)
            if valid.sum() > 0:
                radial_profile[r] = np.mean(magnitude[y_coords[valid], x_coords[valid]])

        # Detect peaks in radial profile
        if len(radial_profile) < 5:
            return 0.0

        # Compute local maxima strength
        smoothed = np.convolve(radial_profile, np.ones(5) / 5, mode="same")
        residual = radial_profile - smoothed
        peak_strength = np.max(np.abs(residual[5:])) if len(residual) > 5 else 0.0

        return float(peak_strength)

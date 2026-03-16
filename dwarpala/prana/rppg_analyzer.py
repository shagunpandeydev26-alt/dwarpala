"""
Remote Photoplethysmography (rPPG) Analyzer — Layer 3: The Chaitanya Check.

The most "divine" liveness check. By analyzing subtle skin color fluctuations
invisible to the human eye, the system detects the cardiac rhythm (heartbeat),
proving the subject has blood flowing through their veins.

Even a perfect 4K deepfake replay cannot reproduce the real-time blood flow
pattern of a live person, because:
1. The color changes are sub-pixel level (0.1-0.5% intensity variation)
2. They are synchronized with the person's actual heart rate
3. Replayed video has its own capture artifacts that corrupt the rPPG signal

Method:
    1. Select ROI (forehead + cheeks — richest capillary beds)
    2. Extract green channel mean over time (green has best SNR for rPPG)
    3. Bandpass filter: 0.7-4.0 Hz (42-240 BPM)
    4. Detect periodicity → extract heart rate
    5. Valid periodicity = live person

References:
    - De Haan & Jeanne, "Robust Pulse Rate from Chrominance-based rPPG", 2013
    - Wang et al., "Algorithmic Principles of Remote PPG", 2017
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple

from dwarpala.utils.logger import get_logger

logger = get_logger("prana.rppg")


@dataclass
class RPPGResult:
    """Result from rPPG-based liveness analysis."""

    score: float  # 0.0 = spoof, 1.0 = live
    heart_rate_bpm: float  # Estimated heart rate
    signal_quality: float  # SNR of the rPPG signal
    has_valid_heartbeat: bool
    prediction: str  # "live" or "spoof"

    def __str__(self):
        status = "🟢 LIVE" if self.prediction == "live" else "🔴 SPOOF"
        hr_str = f"{self.heart_rate_bpm:.0f} BPM" if self.has_valid_heartbeat else "N/A"
        return (
            f"rPPG {status} | score={self.score:.3f} "
            f"| HR={hr_str} | SNR={self.signal_quality:.1f}"
        )


class RPPGAnalyzer:
    """
    Layer 3 of Prana liveness detection — The Chaitanya (Consciousness) Check.

    Extracts cardiovascular pulse signal from subtle skin color changes
    in video frames. A valid, periodic pulse signal proves the subject
    is a living human being.

    Usage:
        analyzer = RPPGAnalyzer(fps=30)
        result = analyzer.analyze(frames_rgb, face_landmarks_per_frame)
    """

    def __init__(
        self,
        fps: float = 30.0,
        window_seconds: float = 3.0,
        bandpass_low: float = 0.7,
        bandpass_high: float = 4.0,
        min_snr: float = 3.0,
        valid_bpm_range: Tuple[float, float] = (45, 200),
        spoof_threshold: float = 0.5,
    ):
        """
        Args:
            fps: Video frame rate.
            window_seconds: Seconds of video to analyze.
            bandpass_low: Lower bandpass frequency (Hz). 0.7 Hz = 42 BPM.
            bandpass_high: Upper bandpass frequency (Hz). 4.0 Hz = 240 BPM.
            min_snr: Minimum signal-to-noise ratio for valid heartbeat.
            valid_bpm_range: Acceptable heart rate range (BPM).
            spoof_threshold: Below this = spoof.
        """
        self.fps = fps
        self.window_seconds = window_seconds
        self.bandpass_low = bandpass_low
        self.bandpass_high = bandpass_high
        self.min_snr = min_snr
        self.valid_bpm_range = valid_bpm_range
        self.spoof_threshold = spoof_threshold

        logger.info(
            f"RPPGAnalyzer: fps={fps}, window={window_seconds}s, "
            f"bandpass=[{bandpass_low}-{bandpass_high}]Hz"
        )

    def analyze(
        self,
        frames: List[np.ndarray],
        landmarks_per_frame: Optional[List[np.ndarray]] = None,
    ) -> RPPGResult:
        """
        Analyze video frames for rPPG cardiac signal.

        Args:
            frames: List of face images (RGB) across time.
            landmarks_per_frame: Optional 5-point landmarks per frame
                for precise ROI selection.

        Returns:
            RPPGResult with heartbeat detection and liveness assessment.
        """
        min_frames = int(self.fps * self.window_seconds)

        if len(frames) < min_frames:
            logger.warning(
                f"Only {len(frames)} frames (need {min_frames}). Insufficient for rPPG."
            )
            return RPPGResult(
                score=0.5,
                heart_rate_bpm=0.0,
                signal_quality=0.0,
                has_valid_heartbeat=False,
                prediction="uncertain",
            )

        # Step 1: Extract rPPG signal from skin ROIs
        raw_signal = self._extract_rppg_signal(frames, landmarks_per_frame)

        # Step 2: Bandpass filter to cardiac frequency range
        filtered_signal = self._bandpass_filter(raw_signal)

        # Step 3: Detect heart rate and assess signal quality
        heart_rate, snr = self._estimate_heart_rate(filtered_signal)

        # Step 4: Determine liveness
        has_valid_heartbeat = (
            snr >= self.min_snr
            and self.valid_bpm_range[0] <= heart_rate <= self.valid_bpm_range[1]
        )

        # Compute liveness score
        if has_valid_heartbeat:
            # Scale score based on SNR confidence
            snr_confidence = min(1.0, snr / (self.min_snr * 2))
            score = 0.6 + 0.4 * snr_confidence
        else:
            score = max(0.0, min(0.4, snr / (self.min_snr * 3)))

        prediction = "live" if score >= self.spoof_threshold else "spoof"

        result = RPPGResult(
            score=float(score),
            heart_rate_bpm=float(heart_rate),
            signal_quality=float(snr),
            has_valid_heartbeat=has_valid_heartbeat,
            prediction=prediction,
        )

        logger.info(str(result))
        return result

    def _extract_rppg_signal(
        self,
        frames: List[np.ndarray],
        landmarks_per_frame: Optional[List[np.ndarray]],
    ) -> np.ndarray:
        """
        Extract raw rPPG signal using the CHROM method.

        CHROM (Chrominance-based method) uses a linear combination of
        color channels to eliminate specular reflection and enhance
        the blood volume pulse signal.

        Signal = 3R - 2G (simplified chrominance projection)
        """
        n_frames = len(frames)
        # Store mean RGB values per frame
        mean_rgb = np.zeros((n_frames, 3), dtype=np.float64)

        for i, frame in enumerate(frames):
            # Get ROI based on landmarks or default face region
            roi = self._get_skin_roi(frame, landmarks_per_frame, i)

            if roi is not None and roi.size > 0:
                # Mean of each color channel across the ROI
                mean_rgb[i, 0] = np.mean(roi[:, :, 0])  # R
                mean_rgb[i, 1] = np.mean(roi[:, :, 1])  # G
                mean_rgb[i, 2] = np.mean(roi[:, :, 2])  # B
            else:
                # Fallback: use center of image
                h, w = frame.shape[:2]
                center = frame[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
                mean_rgb[i, 0] = np.mean(center[:, :, 0])
                mean_rgb[i, 1] = np.mean(center[:, :, 1])
                mean_rgb[i, 2] = np.mean(center[:, :, 2])

        # Normalize each channel by its mean (remove DC component)
        for c in range(3):
            channel_mean = np.mean(mean_rgb[:, c])
            if channel_mean > 0:
                mean_rgb[:, c] = mean_rgb[:, c] / channel_mean

        # CHROM method: project onto chrominance plane
        # X_s = 3R - 2G
        # Y_s = 1.5R + G - 1.5B
        x_s = 3.0 * mean_rgb[:, 0] - 2.0 * mean_rgb[:, 1]
        y_s = 1.5 * mean_rgb[:, 0] + mean_rgb[:, 1] - 1.5 * mean_rgb[:, 2]

        # Adaptive combination using standard deviation ratio
        x_std = np.std(x_s)
        y_std = np.std(y_s)

        if y_std > 0:
            alpha = x_std / y_std
        else:
            alpha = 1.0

        rppg_signal = x_s - alpha * y_s

        # Normalize to zero mean
        rppg_signal = rppg_signal - np.mean(rppg_signal)

        return rppg_signal

    def _get_skin_roi(
        self,
        frame: np.ndarray,
        landmarks_per_frame: Optional[List[np.ndarray]],
        frame_idx: int,
    ) -> Optional[np.ndarray]:
        """
        Get skin Region of Interest for rPPG extraction.
        Forehead and cheek regions have the richest blood flow.
        """
        h, w = frame.shape[:2]

        if landmarks_per_frame is not None and frame_idx < len(landmarks_per_frame):
            lm = landmarks_per_frame[frame_idx]

            # Forehead ROI: above eye line, between eyes
            left_eye = lm[0].astype(int)
            right_eye = lm[1].astype(int)
            nose = lm[2].astype(int)

            # Forehead: centered above eyes
            eye_center_x = (left_eye[0] + right_eye[0]) // 2
            eye_dist = abs(right_eye[0] - left_eye[0])

            forehead_x1 = max(0, eye_center_x - eye_dist // 2)
            forehead_x2 = min(w, eye_center_x + eye_dist // 2)
            forehead_y1 = max(0, left_eye[1] - eye_dist // 2)
            forehead_y2 = max(0, left_eye[1] - eye_dist // 8)

            if forehead_y2 > forehead_y1 and forehead_x2 > forehead_x1:
                return frame[forehead_y1:forehead_y2, forehead_x1:forehead_x2]

        # Default: center-upper region of face (approximate forehead + cheeks)
        roi = frame[int(0.1 * h) : int(0.4 * h), int(0.2 * w) : int(0.8 * w)]
        return roi

    def _bandpass_filter(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply bandpass filter to isolate cardiac frequency range.
        """
        from scipy.signal import butter, filtfilt

        nyquist = self.fps / 2.0
        low = self.bandpass_low / nyquist
        high = self.bandpass_high / nyquist

        # Clamp to valid range
        low = max(0.01, min(low, 0.99))
        high = max(low + 0.01, min(high, 0.99))

        # 4th order Butterworth bandpass
        b, a = butter(4, [low, high], btype="band")

        # Apply zero-phase filtering
        try:
            filtered = filtfilt(b, a, signal)
        except ValueError:
            logger.warning("Bandpass filter failed, returning raw signal")
            filtered = signal

        return filtered

    def _estimate_heart_rate(
        self, filtered_signal: np.ndarray
    ) -> Tuple[float, float]:
        """
        Estimate heart rate from filtered rPPG signal using FFT.

        Returns:
            (heart_rate_bpm, signal_to_noise_ratio)
        """
        n = len(filtered_signal)
        if n < 8:
            return 0.0, 0.0

        # Apply window to reduce spectral leakage
        window = np.hanning(n)
        windowed = filtered_signal * window

        # FFT
        fft_vals = np.fft.rfft(windowed)
        fft_freqs = np.fft.rfftfreq(n, d=1.0 / self.fps)
        power = np.abs(fft_vals) ** 2

        # Find cardiac frequency range
        cardiac_mask = (fft_freqs >= self.bandpass_low) & (
            fft_freqs <= self.bandpass_high
        )

        if not cardiac_mask.any():
            return 0.0, 0.0

        cardiac_power = power[cardiac_mask]
        cardiac_freqs = fft_freqs[cardiac_mask]

        # Find dominant frequency
        peak_idx = np.argmax(cardiac_power)
        dominant_freq = cardiac_freqs[peak_idx]
        peak_power = cardiac_power[peak_idx]

        # Convert Hz to BPM
        heart_rate = dominant_freq * 60.0

        # Compute SNR: peak power / mean of non-peak power
        non_peak_power = np.delete(cardiac_power, peak_idx)
        noise_floor = np.mean(non_peak_power) if len(non_peak_power) > 0 else 1e-10
        snr = peak_power / (noise_floor + 1e-10)
        snr_db = 10 * np.log10(snr + 1e-10)

        return float(heart_rate), float(max(0, snr_db))

    def get_rppg_waveform(
        self,
        frames: List[np.ndarray],
        landmarks_per_frame: Optional[List[np.ndarray]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the rPPG waveform for visualization.

        Returns:
            (time_axis, filtered_signal)
        """
        raw = self._extract_rppg_signal(frames, landmarks_per_frame)
        filtered = self._bandpass_filter(raw)
        time_axis = np.arange(len(filtered)) / self.fps
        return time_axis, filtered

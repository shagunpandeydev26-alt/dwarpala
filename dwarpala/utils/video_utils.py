"""
Video utility functions for Dwarpala.
Handles video reading, frame extraction, and webcam capture for liveness analysis.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Generator, List, Optional, Tuple, Union

from dwarpala.utils.logger import get_logger

logger = get_logger("utils.video")


def read_video_frames(
    source: Union[str, Path, int],
    max_frames: int = 30,
    target_fps: Optional[int] = None,
) -> Tuple[List[np.ndarray], float]:
    """
    Read frames from a video file or webcam.

    Args:
        source: Video file path or camera index (0 for default webcam).
        max_frames: Maximum number of frames to read.
        target_fps: Resample to this FPS. None = use original.

    Returns:
        Tuple of (list of BGR frames, actual FPS).

    Raises:
        ValueError: If the video source cannot be opened.
    """
    if isinstance(source, (str, Path)):
        source = str(source)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video source: {source}")

    original_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = 1
    if target_fps and target_fps < original_fps:
        frame_interval = int(original_fps / target_fps)

    frames = []
    frame_count = 0

    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frames.append(frame)

        frame_count += 1

    cap.release()

    actual_fps = target_fps if target_fps else original_fps
    logger.info(f"Read {len(frames)} frames at {actual_fps:.1f} FPS from {source}")

    return frames, actual_fps


def frames_to_rgb(frames: List[np.ndarray]) -> List[np.ndarray]:
    """
    Convert a list of BGR frames to RGB.

    Args:
        frames: List of BGR frames.

    Returns:
        List of RGB frames.
    """
    return [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]


def extract_roi_timeseries(
    frames: List[np.ndarray],
    roi: Tuple[int, int, int, int],
    channel: int = 1,
) -> np.ndarray:
    """
    Extract the mean pixel value of a specific channel from an ROI across frames.
    Used for rPPG signal extraction.

    Args:
        frames: List of BGR or RGB frames.
        roi: (x1, y1, x2, y2) region of interest.
        channel: Channel index to extract (0=B/R, 1=G, 2=R/B).

    Returns:
        1D array of mean values over time.
    """
    x1, y1, x2, y2 = roi
    signal = np.zeros(len(frames), dtype=np.float64)

    for i, frame in enumerate(frames):
        roi_patch = frame[y1:y2, x1:x2, channel]
        signal[i] = np.mean(roi_patch)

    return signal


def capture_webcam_frames(
    num_frames: int = 30,
    camera_id: int = 0,
    show_preview: bool = True,
) -> Tuple[List[np.ndarray], float]:
    """
    Capture frames from the webcam with an optional live preview.

    Args:
        num_frames: Number of frames to capture.
        camera_id: Webcam device ID.
        show_preview: Whether to show the capture window.

    Returns:
        Tuple of (list of BGR frames, FPS).
    """
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise ValueError(f"Cannot open webcam: {camera_id}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []

    logger.info(f"Capturing {num_frames} frames from webcam {camera_id}...")

    while len(frames) < num_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frames.append(frame.copy())

        if show_preview:
            display = frame.copy()
            cv2.putText(
                display,
                f"Capturing: {len(frames)}/{num_frames}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            cv2.imshow("Dwarpala - Capture", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if show_preview:
        cv2.destroyAllWindows()

    logger.info(f"Captured {len(frames)} frames at {fps:.1f} FPS")
    return frames, fps

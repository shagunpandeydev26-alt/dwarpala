"""
Image utility functions for Dwarpala.
Handles image loading, color conversion, resizing, and normalization.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Union

from dwarpala.utils.logger import get_logger

logger = get_logger("utils.image")


def load_image(
    path: Union[str, Path],
    color: str = "rgb",
) -> np.ndarray:
    """
    Load an image from disk.

    Args:
        path: Path to the image file.
        color: Color space — 'rgb', 'bgr', or 'gray'.

    Returns:
        Image as numpy array in the requested color space.

    Raises:
        FileNotFoundError: If the image path does not exist.
        ValueError: If the image cannot be read.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")

    if color == "rgb":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif color == "gray":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif color == "bgr":
        pass  # OpenCV loads as BGR by default
    else:
        raise ValueError(f"Unknown color space: {color}")

    return img


def resize_image(
    image: np.ndarray,
    target_size: Tuple[int, int],
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """
    Resize an image to target dimensions.

    Args:
        image: Input image array.
        target_size: (width, height) tuple.
        interpolation: OpenCV interpolation method.

    Returns:
        Resized image.
    """
    return cv2.resize(image, target_size, interpolation=interpolation)


def normalize_image(
    image: np.ndarray,
    mean: Tuple[float, ...] = (0.5, 0.5, 0.5),
    std: Tuple[float, ...] = (0.5, 0.5, 0.5),
) -> np.ndarray:
    """
    Normalize an image to zero mean and unit variance.

    Args:
        image: Input image (H, W, C) in range [0, 255].
        mean: Per-channel mean for normalization.
        std: Per-channel standard deviation.

    Returns:
        Normalized image as float32.
    """
    img = image.astype(np.float32) / 255.0
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    return (img - mean) / std


def image_to_tensor(image: np.ndarray) -> np.ndarray:
    """
    Convert HWC image to CHW format for PyTorch.

    Args:
        image: Input image (H, W, C).

    Returns:
        Transposed image (C, H, W).
    """
    if image.ndim == 2:
        return image[np.newaxis, ...]  # Add channel dim for grayscale
    return np.transpose(image, (2, 0, 1))


def crop_face_roi(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    margin: float = 0.2,
) -> np.ndarray:
    """
    Crop a face region from an image with margin.

    Args:
        image: Full image array.
        bbox: (x1, y1, x2, y2) bounding box.
        margin: Fractional margin to add around the face.

    Returns:
        Cropped face image.
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    face_w, face_h = x2 - x1, y2 - y1

    # Add margin
    margin_w = int(face_w * margin)
    margin_h = int(face_h * margin)

    x1 = max(0, x1 - margin_w)
    y1 = max(0, y1 - margin_h)
    x2 = min(w, x2 + margin_w)
    y2 = min(h, y2 + margin_h)

    return image[y1:y2, x1:x2]


def compute_blur_score(image: np.ndarray) -> float:
    """
    Compute blur score using Laplacian variance.
    Higher = sharper image.

    Args:
        image: Input image (grayscale or color).

    Returns:
        Blur score (Laplacian variance).
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_brightness(image: np.ndarray) -> float:
    """
    Compute average brightness of an image.

    Args:
        image: Input image.

    Returns:
        Mean pixel value.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    return float(np.mean(gray))

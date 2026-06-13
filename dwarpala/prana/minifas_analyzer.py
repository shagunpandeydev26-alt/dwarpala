"""
MiniFASNet Anti-Spoofing Analyzer — Layer 4 of Prana.

Integrates MiniFASNetV2 and MiniFASNetV1SE (from Minivision's
Silent-Face-Anti-Spoofing) as a passive liveness signal alongside
texture, temporal, and rPPG analyzers.

MiniFASNet is a lightweight CNN (~600KB) that detects spoof attacks
by analyzing face texture at two different crop scales:
  - Scale 2.7 → MiniFASNetV2 (80×80 crop)
  - Scale 4.0 → MiniFASNetV1SE (80×80 crop, with SE blocks)

Reference:
  - https://github.com/minivision-ai/Silent-Face-Anti-Spoofing
  - License: Apache-2.0

IMPORTANT: MiniFASNet expects its OWN crop (scaled around the face bbox,
resized to 80×80), NOT the 112×112 ArcFace-aligned crop used by Swarupa.
"""

import numpy as np
import cv2
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

from dwarpala.utils.logger import get_logger
from dwarpala.utils.model_manager import ModelManager

logger = get_logger("prana.minifas")


@dataclass
class MiniFASResult:
    """Result from MiniFASNet anti-spoofing analysis."""

    score: float          # 0.0 = spoof, 1.0 = live (fused from both models)
    v2_score: float       # MiniFASNetV2 score
    v1se_score: float     # MiniFASNetV1SE score
    prediction: str       # "live" or "spoof"

    def __str__(self):
        status = "LIVE" if self.prediction == "live" else "SPOOF"
        return (
            f"MiniFAS {status} | score={self.score:.3f} "
            f"(V2={self.v2_score:.3f}, V1SE={self.v1se_score:.3f})"
        )


def _build_minifas_model(arch: str):
    """
    Build a MiniFASNet PyTorch model and load pretrained weights.
    Uses PyTorch's nn.Module for correct, fast inference.

    Args:
        arch: 'V2' for MiniFASNetV2 or 'V1SE' for MiniFASNetV1SE.

    Returns:
        A loaded torch.nn.Module in eval mode, or None if torch unavailable.
    """
    import torch
    import torch.nn as nn

    class ConvBNReLU(nn.Module):
        def __init__(self, in_c, out_c, k=3, s=1, p=1):
            super().__init__()
            self.conv = nn.Conv2d(in_c, out_c, k, s, p, bias=False)
            self.bn = nn.BatchNorm2d(out_c)
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x):
            return self.relu(self.bn(self.conv(x)))

    class LinearBlock(nn.Module):
        def __init__(self, in_c, out_c):
            super().__init__()
            self.depthwise = nn.Conv2d(in_c, in_c, 3, 1, 1, groups=in_c, bias=False)
            self.pointwise = nn.Conv2d(in_c, out_c, 1, 1, 0, bias=False)
            self.bn = nn.BatchNorm2d(out_c)
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x):
            return self.relu(self.bn(self.pointwise(self.depthwise(x))))

    class SELayer(nn.Module):
        def __init__(self, in_c, reduction=4):
            super().__init__()
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.fc1 = nn.Linear(in_c, max(in_c // reduction, 1))
            self.relu = nn.ReLU(inplace=True)
            self.fc2 = nn.Linear(max(in_c // reduction, 1), in_c)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            b, c, _, _ = x.size()
            y = self.avg_pool(x).view(b, c)
            y = self.fc1(y)
            y = self.relu(y)
            y = self.fc2(y)
            y = self.sigmoid(y).view(b, c, 1, 1)
            return x * y

    class MiniFASNetV2(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = ConvBNReLU(3, 32, 3, 1, 1)
            self.pool1 = nn.MaxPool2d(2, 2)
            self.conv2 = ConvBNReLU(32, 64, 3, 1, 1)
            self.pool2 = nn.MaxPool2d(2, 2)
            self.conv3 = ConvBNReLU(64, 128, 3, 1, 1)
            self.pool3 = nn.MaxPool2d(2, 2)
            self.conv4 = ConvBNReLU(128, 128, 3, 1, 1)
            self.pool4 = nn.MaxPool2d(2, 2)
            self.conv5 = ConvBNReLU(128, 64, 3, 1, 1)
            self.pool5 = nn.MaxPool2d(2, 2)
            self.linear1 = LinearBlock(64, 64)
            self.linear2 = LinearBlock(64, 64)
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.flatten = nn.Flatten()
            self.dropout = nn.Dropout(0.5)
            self.fc = nn.Linear(64, 2)

        def forward(self, x):
            x = self.pool1(self.conv1(x))
            x = self.pool2(self.conv2(x))
            x = self.pool3(self.conv3(x))
            x = self.pool4(self.conv4(x))
            x = self.pool5(self.conv5(x))
            x = self.linear1(x)
            x = self.linear2(x)
            x = self.avgpool(x)
            x = self.flatten(x)
            x = self.dropout(x)
            return self.fc(x)

    class MiniFASNetV1SE(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = ConvBNReLU(3, 32, 3, 1, 1)
            self.pool1 = nn.MaxPool2d(2, 2)
            self.se1 = SELayer(32)
            self.conv2 = ConvBNReLU(32, 64, 3, 1, 1)
            self.pool2 = nn.MaxPool2d(2, 2)
            self.se2 = SELayer(64)
            self.conv3 = ConvBNReLU(64, 128, 3, 1, 1)
            self.pool3 = nn.MaxPool2d(2, 2)
            self.se3 = SELayer(128)
            self.conv4 = ConvBNReLU(128, 128, 3, 1, 1)
            self.pool4 = nn.MaxPool2d(2, 2)
            self.se4 = SELayer(128)
            self.conv5 = ConvBNReLU(128, 64, 3, 1, 1)
            self.pool5 = nn.MaxPool2d(2, 2)
            self.se5 = SELayer(64)
            self.linear1 = LinearBlock(64, 64)
            self.se6 = SELayer(64)
            self.linear2 = LinearBlock(64, 64)
            self.se7 = SELayer(64)
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.flatten = nn.Flatten()
            self.dropout = nn.Dropout(0.5)
            self.fc = nn.Linear(64, 2)

        def forward(self, x):
            x = self.pool1(self.conv1(x))
            x = self.se1(x)
            x = self.pool2(self.conv2(x))
            x = self.se2(x)
            x = self.pool3(self.conv3(x))
            x = self.se3(x)
            x = self.pool4(self.conv4(x))
            x = self.se4(x)
            x = self.pool5(self.conv5(x))
            x = self.se5(x)
            x = self.linear1(x)
            x = self.se6(x)
            x = self.linear2(x)
            x = self.se7(x)
            x = self.avgpool(x)
            x = self.flatten(x)
            x = self.dropout(x)
            return self.fc(x)

    model_map = {"V2": MiniFASNetV2, "V1SE": MiniFASNetV1SE}
    if arch not in model_map:
        raise ValueError(f"Unknown MiniFASNet architecture: {arch}")
    return model_map[arch]()


class MiniFASAnalyzer:
    """
    MiniFASNet anti-spoofing analyzer for Dwarpala Prana.

    Loads both MiniFASNetV2 (scale 2.7) and MiniFASNetV1SE (scale 4.0)
    pretrained models and fuses their predictions.

    IMPORTANT: Uses its OWN crop from the original image (scaled around
    the face bounding box, resized to 80×80). Does NOT use the
    112×112 ArcFace-aligned crop used by Swarupa.

    Usage:
        analyzer = MiniFASAnalyzer()
        result = analyzer.analyze(original_image_rgb, face_bbox)
    """

    INPUT_SIZE = 80

    def __init__(
        self,
        model_dir: Optional[Path] = None,
        v2_path: Optional[Union[str, Path]] = None,
        v1se_path: Optional[Union[str, Path]] = None,
        spoof_threshold: float = 0.5,
    ):
        self.spoof_threshold = spoof_threshold
        self._models_loaded = False

        if v2_path is not None and v1se_path is not None:
            v2_path = Path(v2_path)
            v1se_path = Path(v1se_path)
        else:
            manager = ModelManager(model_dir=model_dir)
            try:
                v2_path = manager.get_model_path("minifas_v2_2_7")
            except FileNotFoundError:
                v2_path = None
            try:
                v1se_path = manager.get_model_path("minifas_v1se_4_0")
            except FileNotFoundError:
                v1se_path = None

        self.v2_path = v2_path
        self.v1se_path = v1se_path

        import torch

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if v2_path and v2_path.exists():
            try:
                self._model_v2 = self._load_model(v2_path, "V2")
                logger.info(f"Loaded MiniFASNetV2 from {v2_path}")
                self._models_loaded = True
            except Exception as e:
                logger.warning(f"Failed to load MiniFASNetV2: {e}")
                self._model_v2 = None
        else:
            self._model_v2 = None

        if v1se_path and v1se_path.exists():
            try:
                self._model_v1se = self._load_model(v1se_path, "V1SE")
                logger.info(f"Loaded MiniFASNetV1SE from {v1se_path}")
                self._models_loaded = True
            except Exception as e:
                logger.warning(f"Failed to load MiniFASNetV1SE: {e}")
                self._model_v1se = None
        else:
            self._model_v1se = None

        if self._models_loaded:
            logger.info(
                f"MiniFASAnalyzer: V2={'loaded' if self._model_v2 else 'N/A'}, "
                f"V1SE={'loaded' if self._model_v1se else 'N/A'}, "
                f"threshold={spoof_threshold}, device={self.device}"
            )
        else:
            logger.warning(
                "MiniFASAnalyzer: no models loaded. "
                "Run 'dwarpala download-models' to download MiniFASNet weights."
            )

    def _load_model(self, path: Path, arch: str):
        """Load a .pth state dict into a PyTorch model."""
        import torch

        model = _build_minifas_model(arch)
        state_dict = torch.load(str(path), map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        return model

    @staticmethod
    def preprocess(original_image: np.ndarray, bbox: Tuple[int, int, int, int],
                   scale: float, input_size: int = 80) -> np.ndarray:
        """
        Preprocess face crop for MiniFASNet inference.

        Args:
            original_image: Full image (H, W, 3) in RGB.
            bbox: (x, y, w, h) face bounding box.
            scale: Bbox enlargement factor (2.7 for V2, 4.0 for V1SE).
            input_size: Target crop size (default 80).

        Returns:
            Preprocessed crop as (1, 3, input_size, input_size) float32
            normalized to [-1, 1].
        """
        x, y, w, h = bbox
        cx, cy = x + w // 2, y + h // 2
        new_w = int(w * scale)
        new_h = int(h * scale)

        x1 = max(0, cx - new_w // 2)
        y1 = max(0, cy - new_h // 2)
        x2 = min(original_image.shape[1], cx + new_w // 2)
        y2 = min(original_image.shape[0], cy + new_h // 2)

        if x2 <= x1 or y2 <= y1:
            x1, y1 = x, y
            x2 = min(original_image.shape[1], x + w)
            y2 = min(original_image.shape[0], y + h)

        crop = original_image[y1:y2, x1:x2]
        if crop.size == 0:
            raise ValueError("Empty crop — bbox out of image bounds")

        crop = cv2.resize(crop, (input_size, input_size),
                         interpolation=cv2.INTER_LINEAR)
        crop = crop.astype(np.float32) / 255.0
        crop = (crop - 0.5) * 2.0
        crop = np.transpose(crop, (2, 0, 1))
        crop = np.expand_dims(crop, axis=0)
        return crop

    def analyze(
        self,
        original_image: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> MiniFASResult:
        """
        Run MiniFASNet anti-spoofing analysis.

        Args:
            original_image: Full image (H, W, 3) in RGB.
            bbox: (x, y, w, h) face bounding box.

        Returns:
            MiniFASResult with fused prediction.
        """
        if not self._models_loaded:
            return MiniFASResult(
                score=0.5, v2_score=0.5, v1se_score=0.5, prediction="uncertain",
            )

        import torch

        v2_score = 0.5
        v1se_score = 0.5

        with torch.no_grad():
            if self._model_v2 is not None:
                try:
                    tensor = self.preprocess(original_image, bbox, scale=2.7)
                    t = torch.from_numpy(tensor).to(self.device)
                    out = self._model_v2(t)
                    prob = torch.softmax(out, dim=1)
                    v2_score = float(prob[0, 1].cpu().numpy())
                except Exception as e:
                    logger.warning(f"MiniFASNetV2 inference failed: {e}")

            if self._model_v1se is not None:
                try:
                    tensor = self.preprocess(original_image, bbox, scale=4.0)
                    t = torch.from_numpy(tensor).to(self.device)
                    out = self._model_v1se(t)
                    prob = torch.softmax(out, dim=1)
                    v1se_score = float(prob[0, 1].cpu().numpy())
                except Exception as e:
                    logger.warning(f"MiniFASNetV1SE inference failed: {e}")

        available = []
        if self._model_v2 is not None:
            available.append(v2_score)
        if self._model_v1se is not None:
            available.append(v1se_score)

        fused = float(np.mean(available)) if available else 0.5
        prediction = "live" if fused >= self.spoof_threshold else "spoof"

        result = MiniFASResult(
            score=fused,
            v2_score=v2_score,
            v1se_score=v1se_score,
            prediction=prediction,
        )
        logger.info(str(result))
        return result

    @property
    def models_loaded(self) -> bool:
        """Whether any models were loaded successfully."""
        return self._models_loaded

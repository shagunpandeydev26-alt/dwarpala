"""
MiniFASNet Anti-Spoofing Analyzer — Layer 4 of Prana.

Integrates MiniFASNetV2 and MiniFASNetV1SE (from Minivision's
Silent-Face-Anti-Spoofing) as a passive liveness signal alongside
texture, temporal, and rPPG analyzers.

MiniFASNet is a lightweight CNN (~1.8MB) that detects spoof attacks
by analyzing face texture at two different crop scales:
  - Scale 2.7 → MiniFASNetV2  (80×80 crop, no SE)
  - Scale 4.0 → MiniFASNetV1SE (80×80 crop, with SE blocks)

Reference:
  - https://github.com/minivision-ai/Silent-Face-Anti-Spoofing
  - License: Apache-2.0

This module is a faithful port of the reference architecture
(src/model_lib/MiniFASNet.py), preprocessing (CropImage in
src/generate_patches.py), and inference transform (ToTensor only,
BGR preserved) so that the published pretrained weights produce the
same outputs they do in the original repo.

IMPORTANT: MiniFASNet expects its OWN crop (the reference CropImage
bbox-scaled crop resized to 80×80), NOT the 112×112 ArcFace-aligned
crop used by Swarupa.
"""

import numpy as np
import cv2
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

from dwarpala.utils.logger import get_logger
from dwarpala.utils.model_manager import ModelManager

logger = get_logger("prana.minifas")


# ── Reference channel configs (keep_dict from minivision MiniFASNet.py) ──
# V2  (2.7_80x80_MiniFASNetV2.pth)   → MiniFASNet   with keep '1.8M_'
# V1SE(4_0_0_80x80_MiniFASNetV1SE.pth)→ MiniFASNetSE with keep '1.8M'
_KEEP_DICT = {
    "1.8M": [32, 32, 103, 103, 64, 13, 13, 64, 26, 26,
             64, 13, 13, 64, 52, 52, 64, 231, 231, 128,
             154, 154, 128, 52, 52, 128, 26, 26, 128, 52,
             52, 128, 26, 26, 128, 26, 26, 128, 308, 308,
             128, 26, 26, 128, 26, 26, 128, 512, 512],
    "1.8M_": [32, 32, 103, 103, 64, 13, 13, 64, 13, 13, 64, 13,
              13, 64, 13, 13, 64, 231, 231, 128, 231, 231, 128, 52,
              52, 128, 26, 26, 128, 77, 77, 128, 26, 26, 128, 26, 26,
              128, 308, 308, 128, 26, 26, 128, 26, 26, 128, 512, 512],
}

# get_kernel(80, 80) = ((80+15)//16, (80+15)//16) = (5, 5)
_CONV6_KERNEL = (5, 5)


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
    Build a MiniFASNet model as a faithful port of the minivision reference
    (src/model_lib/MiniFASNet.py). Module structure and names match the
    pretrained state dict exactly, so weights load with strict=True.

    Args:
        arch: 'V2' for MiniFASNetV2 (no SE) or 'V1SE' for MiniFASNetV1SE.

    Returns:
        A torch.nn.Module (not yet loaded; weights applied separately).
    """
    import torch
    from torch.nn import (
        Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid,
        AdaptiveAvgPool2d, Sequential, Module, Dropout,
    )

    class Flatten(Module):
        def forward(self, x):
            return x.view(x.size(0), -1)

    class Conv_block(Module):
        def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1),
                     padding=(0, 0), groups=1):
            super().__init__()
            self.conv = Conv2d(in_c, out_c, kernel_size=kernel, groups=groups,
                               stride=stride, padding=padding, bias=False)
            self.bn = BatchNorm2d(out_c)
            self.prelu = PReLU(out_c)

        def forward(self, x):
            return self.prelu(self.bn(self.conv(x)))

    class Linear_block(Module):
        """Conv + BN, NO activation (minivision 'Linear_block')."""
        def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1),
                     padding=(0, 0), groups=1):
            super().__init__()
            self.conv = Conv2d(in_c, out_c, kernel_size=kernel, groups=groups,
                               stride=stride, padding=padding, bias=False)
            self.bn = BatchNorm2d(out_c)

        def forward(self, x):
            return self.bn(self.conv(x))

    class Depth_Wise(Module):
        def __init__(self, c1, c2, c3, residual=False, kernel=(3, 3),
                     stride=(2, 2), padding=(1, 1), groups=1):
            super().__init__()
            c1_in, c1_out = c1
            c2_in, c2_out = c2
            c3_in, c3_out = c3
            self.conv = Conv_block(c1_in, c1_out, kernel=(1, 1),
                                   padding=(0, 0), stride=(1, 1))
            self.conv_dw = Conv_block(c2_in, c2_out, groups=c2_in, kernel=kernel,
                                      padding=padding, stride=stride)
            self.project = Linear_block(c3_in, c3_out, kernel=(1, 1),
                                        padding=(0, 0), stride=(1, 1))
            self.residual = residual

        def forward(self, x):
            short_cut = x if self.residual else None
            x = self.conv(x)
            x = self.conv_dw(x)
            x = self.project(x)
            return short_cut + x if self.residual else x

    class SEModule(Module):
        def __init__(self, channels, reduction):
            super().__init__()
            self.avg_pool = AdaptiveAvgPool2d(1)
            self.fc1 = Conv2d(channels, channels // reduction, kernel_size=1,
                              padding=0, bias=False)
            self.bn1 = BatchNorm2d(channels // reduction)
            self.relu = ReLU(inplace=True)
            self.fc2 = Conv2d(channels // reduction, channels, kernel_size=1,
                              padding=0, bias=False)
            self.bn2 = BatchNorm2d(channels)
            self.sigmoid = Sigmoid()

        def forward(self, x):
            module_input = x
            x = self.avg_pool(x)
            x = self.fc1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.fc2(x)
            x = self.bn2(x)
            x = self.sigmoid(x)
            return module_input * x

    class Depth_Wise_SE(Module):
        def __init__(self, c1, c2, c3, residual=False, kernel=(3, 3),
                     stride=(2, 2), padding=(1, 1), groups=1, se_reduct=8):
            super().__init__()
            c1_in, c1_out = c1
            c2_in, c2_out = c2
            c3_in, c3_out = c3
            self.conv = Conv_block(c1_in, c1_out, kernel=(1, 1),
                                   padding=(0, 0), stride=(1, 1))
            self.conv_dw = Conv_block(c2_in, c2_out, groups=c2_in, kernel=kernel,
                                      padding=padding, stride=stride)
            self.project = Linear_block(c3_in, c3_out, kernel=(1, 1),
                                        padding=(0, 0), stride=(1, 1))
            self.residual = residual
            self.se_module = SEModule(c3_out, se_reduct)

        def forward(self, x):
            short_cut = x if self.residual else None
            x = self.conv(x)
            x = self.conv_dw(x)
            x = self.project(x)
            if self.residual:
                x = self.se_module(x)
                return short_cut + x
            return x

    class Residual(Module):
        def __init__(self, c1, c2, c3, num_block, groups, kernel=(3, 3),
                     stride=(1, 1), padding=(1, 1)):
            super().__init__()
            modules = []
            for i in range(num_block):
                modules.append(Depth_Wise(
                    c1[i], c2[i], c3[i], residual=True, kernel=kernel,
                    padding=padding, stride=stride, groups=groups))
            self.model = Sequential(*modules)

        def forward(self, x):
            return self.model(x)

    class ResidualSE(Module):
        def __init__(self, c1, c2, c3, num_block, groups, kernel=(3, 3),
                     stride=(1, 1), padding=(1, 1), se_reduct=4):
            super().__init__()
            modules = []
            for i in range(num_block):
                if i == num_block - 1:
                    modules.append(Depth_Wise_SE(
                        c1[i], c2[i], c3[i], residual=True, kernel=kernel,
                        padding=padding, stride=stride, groups=groups,
                        se_reduct=se_reduct))
                else:
                    modules.append(Depth_Wise(
                        c1[i], c2[i], c3[i], residual=True, kernel=kernel,
                        padding=padding, stride=stride, groups=groups))
            self.model = Sequential(*modules)

        def forward(self, x):
            return self.model(x)

    class MiniFASNet(Module):
        def __init__(self, keep, embedding_size=128, conv6_kernel=(5, 5),
                     drop_p=0.2, num_classes=3, img_channel=3, use_se=False):
            super().__init__()
            self.embedding_size = embedding_size
            res_cls = ResidualSE if use_se else Residual

            self.conv1 = Conv_block(img_channel, keep[0], kernel=(3, 3),
                                    stride=(2, 2), padding=(1, 1))
            self.conv2_dw = Conv_block(keep[0], keep[1], kernel=(3, 3),
                                       stride=(1, 1), padding=(1, 1),
                                       groups=keep[1])

            self.conv_23 = Depth_Wise(
                (keep[1], keep[2]), (keep[2], keep[3]), (keep[3], keep[4]),
                kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=keep[3])

            c1 = [(keep[4], keep[5]), (keep[7], keep[8]),
                  (keep[10], keep[11]), (keep[13], keep[14])]
            c2 = [(keep[5], keep[6]), (keep[8], keep[9]),
                  (keep[11], keep[12]), (keep[14], keep[15])]
            c3 = [(keep[6], keep[7]), (keep[9], keep[10]),
                  (keep[12], keep[13]), (keep[15], keep[16])]
            self.conv_3 = res_cls(c1, c2, c3, num_block=4, groups=keep[4],
                                  kernel=(3, 3), stride=(1, 1), padding=(1, 1))

            self.conv_34 = Depth_Wise(
                (keep[16], keep[17]), (keep[17], keep[18]), (keep[18], keep[19]),
                kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=keep[19])

            c1 = [(keep[19], keep[20]), (keep[22], keep[23]), (keep[25], keep[26]),
                  (keep[28], keep[29]), (keep[31], keep[32]), (keep[34], keep[35])]
            c2 = [(keep[20], keep[21]), (keep[23], keep[24]), (keep[26], keep[27]),
                  (keep[29], keep[30]), (keep[32], keep[33]), (keep[35], keep[36])]
            c3 = [(keep[21], keep[22]), (keep[24], keep[25]), (keep[27], keep[28]),
                  (keep[30], keep[31]), (keep[33], keep[34]), (keep[36], keep[37])]
            self.conv_4 = res_cls(c1, c2, c3, num_block=6, groups=keep[19],
                                  kernel=(3, 3), stride=(1, 1), padding=(1, 1))

            self.conv_45 = Depth_Wise(
                (keep[37], keep[38]), (keep[38], keep[39]), (keep[39], keep[40]),
                kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=keep[40])

            c1 = [(keep[40], keep[41]), (keep[43], keep[44])]
            c2 = [(keep[41], keep[42]), (keep[44], keep[45])]
            c3 = [(keep[42], keep[43]), (keep[45], keep[46])]
            self.conv_5 = res_cls(c1, c2, c3, num_block=2, groups=keep[40],
                                  kernel=(3, 3), stride=(1, 1), padding=(1, 1))

            self.conv_6_sep = Conv_block(keep[46], keep[47], kernel=(1, 1),
                                         stride=(1, 1), padding=(0, 0))
            self.conv_6_dw = Linear_block(keep[47], keep[48], groups=keep[48],
                                          kernel=conv6_kernel, stride=(1, 1),
                                          padding=(0, 0))
            self.conv_6_flatten = Flatten()
            self.linear = Linear(512, embedding_size, bias=False)
            self.bn = BatchNorm1d(embedding_size)
            self.drop = Dropout(p=drop_p)
            self.prob = Linear(embedding_size, num_classes, bias=False)

        def forward(self, x):
            out = self.conv1(x)
            out = self.conv2_dw(out)
            out = self.conv_23(out)
            out = self.conv_3(out)
            out = self.conv_34(out)
            out = self.conv_4(out)
            out = self.conv_45(out)
            out = self.conv_5(out)
            out = self.conv_6_sep(out)
            out = self.conv_6_dw(out)
            out = self.conv_6_flatten(out)
            if self.embedding_size != 512:
                out = self.linear(out)
            out = self.bn(out)
            out = self.drop(out)
            out = self.prob(out)
            return out

    if arch == "V2":
        return MiniFASNet(_KEEP_DICT["1.8M_"], conv6_kernel=_CONV6_KERNEL,
                          drop_p=0.2, num_classes=3, use_se=False)
    elif arch == "V1SE":
        return MiniFASNet(_KEEP_DICT["1.8M"], conv6_kernel=_CONV6_KERNEL,
                          drop_p=0.75, num_classes=3, use_se=True)
    raise ValueError(f"Unknown arch: {arch}")


def _load_minifas_weights(model, path: Path):
    """
    Load the pretrained state dict. The reference saves with a 'module.'
    prefix (DataParallel); strip it. Module names otherwise match exactly,
    so we load with strict=True and surface any mismatch loudly.
    """
    import torch

    state_dict = torch.load(str(path), map_location="cpu", weights_only=True)

    first_key = next(iter(state_dict))
    if first_key.startswith("module."):
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=True)
    return model


class MiniFASAnalyzer:
    """
    MiniFASNet anti-spoofing analyzer for Dwarpala Prana.

    Loads both MiniFASNetV2 (scale 2.7) and MiniFASNetV1SE (scale 4.0)
    pretrained models and fuses their predictions.

    IMPORTANT: Uses the reference CropImage crop from the original image
    (bbox scaled and shifted to stay in-frame, resized to 80×80). Does NOT
    use the 112×112 ArcFace-aligned crop used by Swarupa.

    The model outputs 3-class logits. Index 1 = live (real); indices 0 and 2
    are spoof types. The liveness score is the softmax probability of class 1,
    averaged across the two models (equivalent to the reference pipeline,
    which sums the two softmax vectors and reads class 1).
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

        import torch

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if v2_path and v2_path.exists():
            try:
                model = _build_minifas_model("V2")
                self._model_v2 = _load_minifas_weights(model, v2_path)
                self._model_v2.to(self.device)
                self._model_v2.eval()
                logger.info(f"Loaded MiniFASNetV2 from {v2_path}")
                self._models_loaded = True
            except Exception as e:
                logger.warning(f"Failed to load MiniFASNetV2: {e}")
                self._model_v2 = None
        else:
            self._model_v2 = None

        if v1se_path and v1se_path.exists():
            try:
                model = _build_minifas_model("V1SE")
                self._model_v1se = _load_minifas_weights(model, v1se_path)
                self._model_v1se.to(self.device)
                self._model_v1se.eval()
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

    @staticmethod
    def _get_new_box(src_w: int, src_h: int,
                     bbox: Tuple[int, int, int, int], scale: float):
        """
        Reference CropImage._get_new_box: clamp the scale so the enlarged box
        fits the image, then SHIFT (not clip) any out-of-bounds edge inward so
        the crop keeps its full requested size.
        """
        x, y, box_w, box_h = bbox
        scale = min((src_h - 1) / box_h, min((src_w - 1) / box_w, scale))

        new_width = box_w * scale
        new_height = box_h * scale
        center_x, center_y = box_w / 2 + x, box_h / 2 + y

        left_top_x = center_x - new_width / 2
        left_top_y = center_y - new_height / 2
        right_bottom_x = center_x + new_width / 2
        right_bottom_y = center_y + new_height / 2

        if left_top_x < 0:
            right_bottom_x -= left_top_x
            left_top_x = 0
        if left_top_y < 0:
            right_bottom_y -= left_top_y
            left_top_y = 0
        if right_bottom_x > src_w - 1:
            left_top_x -= right_bottom_x - src_w + 1
            right_bottom_x = src_w - 1
        if right_bottom_y > src_h - 1:
            left_top_y -= right_bottom_y - src_h + 1
            right_bottom_y = src_h - 1

        return (int(left_top_x), int(left_top_y),
                int(right_bottom_x), int(right_bottom_y))

    @staticmethod
    def preprocess(original_image: np.ndarray, bbox: Tuple[int, int, int, int],
                   scale: float, input_size: int = 80) -> np.ndarray:
        """
        Preprocess a face crop for MiniFASNet inference, matching the
        reference exactly: CropImage bbox-scaled crop (BGR) → resize 80×80
        → channels-first float32 in [0, 255].

        The reference transform (src/data_io/functional.py::to_tensor) has the
        `.div(255)` deliberately commented out — the pretrained weights expect
        RAW [0,255] BGR pixels with NO normalization and NO BGR→RGB swap.
        Dividing by 255 (or applying (x-127.5)/128) collapses every activation
        and makes the 3-class head saturate to one class for all inputs.

        Args:
            original_image: Full image (H, W, 3) in BGR (OpenCV default).
            bbox: (x, y, w, h) face bounding box.
            scale: Bbox enlargement factor (2.7 for V2, 4.0 for V1SE).
            input_size: Target crop size (default 80).

        Returns:
            Preprocessed crop as (1, 3, input_size, input_size) float32 in [0, 255].
        """
        src_h, src_w = original_image.shape[:2]
        x1, y1, x2, y2 = MiniFASAnalyzer._get_new_box(src_w, src_h, bbox, scale)

        crop = original_image[y1:y2 + 1, x1:x2 + 1]
        if crop.size == 0:
            raise ValueError("Empty crop — bbox out of image bounds")

        crop = cv2.resize(crop, (input_size, input_size))
        # Reference to_tensor: HWC [0,255] BGR → CHW float32, NO /255, NO swap.
        crop = crop.astype(np.float32)
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

        Runs both V2 (scale 2.7) and V1SE (scale 4.0) on 80×80 crops and
        returns the class-1 (live) softmax probability, averaged across the
        two models (the reference minivision fusion).

        Args:
            original_image: Full image (H, W, 3) in BGR.
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
        return self._models_loaded

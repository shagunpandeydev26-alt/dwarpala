"""
Embedding extraction pipeline for Dwarpala.
Takes an aligned face image and produces a 512-D normalized embedding
using the ViT backbone.
"""

import torch
import numpy as np
import cv2
from typing import Optional, Union
from pathlib import Path

from dwarpala.swarupa.backbone import ViTBackbone
from dwarpala.utils.logger import get_logger

logger = get_logger("swarupa.embedding")


class EmbeddingExtractor:
    """
    Extracts face embeddings from aligned face images.
    Handles image preprocessing, model inference, and optional ONNX runtime.

    Usage:
        extractor = EmbeddingExtractor(weights_path="weights/swarupa.pth")
        embedding = extractor.extract(aligned_face_rgb)  # np.ndarray (512,)
    """

    def __init__(
        self,
        weights_path: Optional[Union[str, Path]] = None,
        model_name: str = "vit_base_patch16_224",
        embedding_dim: int = 512,
        device: str = "auto",
        use_onnx: bool = False,
    ):
        """
        Args:
            weights_path: Path to trained model weights (.pth or .onnx).
            model_name: timm model name for the backbone.
            embedding_dim: Output embedding dimension.
            device: 'auto', 'cpu', 'cuda', or 'cuda:0'.
            use_onnx: Whether to use ONNX Runtime for inference.
        """
        self.embedding_dim = embedding_dim
        self.use_onnx = use_onnx

        # Resolve device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if use_onnx and weights_path and str(weights_path).endswith(".onnx"):
            self._init_onnx(weights_path)
        else:
            self._init_pytorch(model_name, embedding_dim, weights_path)

        logger.info(
            f"EmbeddingExtractor: device={self.device}, "
            f"onnx={use_onnx}, dim={embedding_dim}"
        )

    def _init_pytorch(
        self,
        model_name: str,
        embedding_dim: int,
        weights_path: Optional[Union[str, Path]],
    ):
        """Initialize PyTorch backbone."""
        self.model = ViTBackbone(
            model_name=model_name,
            pretrained=True,
            embedding_dim=embedding_dim,
        )

        if weights_path and Path(weights_path).exists():
            state_dict = torch.load(
                weights_path, map_location=self.device, weights_only=True
            )
            self.model.load_state_dict(state_dict, strict=False)
            logger.info(f"Loaded weights from {weights_path}")

        self.model.to(self.device)
        self.model.eval()

    def _init_onnx(self, onnx_path: Union[str, Path]):
        """Initialize ONNX Runtime session."""
        import onnxruntime as ort

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.ort_session = ort.InferenceSession(str(onnx_path), providers=providers)
        self.input_name = self.ort_session.get_inputs()[0].name
        logger.info(f"Loaded ONNX model from {onnx_path}")

    def preprocess(self, face_image: np.ndarray) -> np.ndarray:
        """
        Preprocess aligned face image for the backbone.

        Args:
            face_image: Aligned face (H, W, 3) in RGB, uint8.

        Returns:
            Preprocessed tensor as numpy array (1, 3, 224, 224).
        """
        # Resize to 224x224 for ViT input
        img = cv2.resize(face_image, (224, 224), interpolation=cv2.INTER_LINEAR)

        # Normalize to [-1, 1]
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5

        # HWC → CHW
        img = np.transpose(img, (2, 0, 1))

        # Add batch dimension
        img = np.expand_dims(img, axis=0)

        return img

    @torch.no_grad()
    def extract(self, face_image: np.ndarray) -> np.ndarray:
        """
        Extract embedding from a single aligned face image.

        Args:
            face_image: Aligned face (H, W, 3) in RGB, uint8.

        Returns:
            L2-normalized embedding (embedding_dim,).
        """
        preprocessed = self.preprocess(face_image)

        if self.use_onnx and hasattr(self, "ort_session"):
            outputs = self.ort_session.run(None, {self.input_name: preprocessed})
            embedding = outputs[0][0]
        else:
            tensor = torch.from_numpy(preprocessed).to(self.device)
            embedding = self.model(tensor).cpu().numpy()[0]

        # Ensure L2 normalization
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    @torch.no_grad()
    def extract_batch(self, face_images: list) -> np.ndarray:
        """
        Extract embeddings for a batch of aligned face images.

        Args:
            face_images: List of aligned face images (H, W, 3) in RGB.

        Returns:
            L2-normalized embeddings (N, embedding_dim).
        """
        preprocessed = np.concatenate(
            [self.preprocess(img) for img in face_images], axis=0
        )

        if self.use_onnx and hasattr(self, "ort_session"):
            outputs = self.ort_session.run(None, {self.input_name: preprocessed})
            embeddings = outputs[0]
        else:
            tensor = torch.from_numpy(preprocessed).to(self.device)
            embeddings = self.model(tensor).cpu().numpy()

        # L2 normalize each embedding
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        embeddings = embeddings / norms

        return embeddings

    def export_onnx(self, output_path: Union[str, Path]):
        """
        Export the backbone to ONNX format for optimized inference.

        Args:
            output_path: Path to save the ONNX model.
        """
        if self.use_onnx:
            logger.warning("Cannot export from ONNX session")
            return

        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
        torch.onnx.export(
            self.model,
            dummy_input,
            str(output_path),
            opset_version=17,
            input_names=["input"],
            output_names=["embedding"],
            dynamic_axes={"input": {0: "batch_size"}, "embedding": {0: "batch_size"}},
        )
        logger.info(f"Exported ONNX model to {output_path}")

"""
Embedding extraction pipeline for Dwarpala.

Supports two backends:
1. **insightface** (default): Uses buffalo_l ArcFace R50 (w600k_r50.onnx)
   via onnxruntime. Produces real, discriminative 512-D face embeddings.
   Requires model download via `dwarpala download-models`.

2. **vit**: Uses the ViT backbone from timm with ImageNet pretrained weights.
   This is the TRAINING-READY research path — the ViT backbone is intended
   to be fine-tuned with ArcFace loss on a face dataset. With ImageNet-only
   weights, embeddings are NOT face-discriminative.
"""

import numpy as np
import cv2
from typing import Optional, Union
from pathlib import Path

from dwarpala.utils.logger import get_logger

logger = get_logger("swarupa.embedding")


class EmbeddingExtractor:
    """
    Extracts face embeddings from aligned face images.

    The default backend is InsightFace (buffalo_l ArcFace R50), which provides
    real face recognition capability (~99.8% on LFW). The ViT backend is
    available for research/training purposes but requires fine-tuning.

    Usage:
        # Default: InsightFace (recommended)
        extractor = EmbeddingExtractor(backend="insightface")
        embedding = extractor.extract(aligned_face_rgb)  # np.ndarray (512,)

        # Research: ViT (requires fine-tuning for face tasks)
        extractor = EmbeddingExtractor(backend="vit", allow_untrained=True)
    """

    SUPPORTED_BACKENDS = ("insightface", "vit")

    def __init__(
        self,
        backend: str = "insightface",
        weights_path: Optional[Union[str, Path]] = None,
        model_name: str = "vit_base_patch16_224",
        embedding_dim: int = 512,
        device: str = "auto",
        use_onnx: bool = False,
        allow_untrained: bool = False,
        model_dir: Optional[Path] = None,
    ):
        """
        Args:
            backend: 'insightface' (default, real model) or 'vit' (research).
            weights_path: Path to model weights (for ViT: .pth, for InsightFace: .onnx).
            model_name: timm model name (ViT backend only).
            embedding_dim: Output embedding dimension.
            device: 'auto', 'cpu', 'cuda' (ViT backend only).
            use_onnx: Use ONNX runtime (ViT backend only).
            allow_untrained: Allow ViT backend without face-specific weights.
                Must be True to use ViT with ImageNet-only weights.
            model_dir: Model directory for InsightFace model lookup.
        """
        if backend not in self.SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unknown backend: {backend}. "
                f"Supported: {self.SUPPORTED_BACKENDS}"
            )

        self.backend = backend
        self.embedding_dim = embedding_dim

        if backend == "insightface":
            self._init_insightface(weights_path, model_dir)
        else:
            self._init_vit(
                model_name, embedding_dim, weights_path, device,
                use_onnx, allow_untrained
            )

        logger.info(
            f"EmbeddingExtractor: backend={backend}, dim={embedding_dim}"
        )

    def _init_insightface(
        self,
        weights_path: Optional[Union[str, Path]],
        model_dir: Optional[Path],
    ) -> None:
        """Initialize InsightFace ArcFace R50 backend."""
        from dwarpala.swarupa.insightface_embedder import InsightFaceEmbedder

        self._embedder = InsightFaceEmbedder(
            model_path=weights_path,
            model_dir=model_dir,
        )
        self.embedding_dim = self._embedder.embedding_dim

    def _init_vit(
        self,
        model_name: str,
        embedding_dim: int,
        weights_path: Optional[Union[str, Path]],
        device: str,
        use_onnx: bool,
        allow_untrained: bool,
    ) -> None:
        """Initialize ViT backbone (research/training path)."""
        import torch
        from dwarpala.swarupa.backbone import ViTBackbone

        # Gate: ViT without face weights is NOT suitable for inference
        has_face_weights = weights_path and Path(weights_path).exists()
        if not has_face_weights and not allow_untrained:
            raise RuntimeError(
                "ViT backend requires face-specific trained weights for inference.\n"
                "Either:\n"
                "  1. Use backend='insightface' (recommended for inference)\n"
                "  2. Provide weights_path to a face-trained ViT checkpoint\n"
                "  3. Set allow_untrained=True (for testing/research only — "
                "produces random embeddings)"
            )

        # Resolve device
        if device == "auto":
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        self.use_onnx = use_onnx

        if use_onnx and weights_path and str(weights_path).endswith(".onnx"):
            import onnxruntime as ort
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self.ort_session = ort.InferenceSession(
                str(weights_path), providers=providers
            )
            self.input_name = self.ort_session.get_inputs()[0].name
            logger.info(f"Loaded ViT ONNX model from {weights_path}")
        else:
            self.model = ViTBackbone(
                model_name=model_name,
                pretrained=True,
                embedding_dim=embedding_dim,
            )
            if has_face_weights:
                state_dict = torch.load(
                    weights_path, map_location=self.device, weights_only=True
                )
                self.model.load_state_dict(state_dict, strict=False)
                logger.info(f"Loaded face-trained ViT weights from {weights_path}")
            elif allow_untrained:
                logger.warning(
                    "ViT backend using ImageNet-only weights — "
                    "embeddings are NOT face-discriminative. "
                    "This is only suitable for testing."
                )

            self.model.to(self.device)
            self.model.eval()

        self._embedder = None  # Mark that we're using ViT path

    def preprocess(self, face_image: np.ndarray) -> np.ndarray:
        """
        Preprocess aligned face image for the active backend.

        Args:
            face_image: Aligned face (H, W, 3) in RGB, uint8.

        Returns:
            Preprocessed tensor as numpy array.
        """
        if self.backend == "insightface":
            return self._embedder.preprocess(face_image)

        # ViT preprocessing: resize 224×224, normalize [-1, 1]
        img = cv2.resize(face_image, (224, 224), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        img = np.transpose(img, (2, 0, 1))  # HWC → CHW
        img = np.expand_dims(img, axis=0)
        return img

    def extract(self, face_image: np.ndarray) -> np.ndarray:
        """
        Extract embedding from a single aligned face image.

        Args:
            face_image: Aligned face (H, W, 3) in RGB, uint8.

        Returns:
            L2-normalized embedding (embedding_dim,).
        """
        if self.backend == "insightface":
            return self._embedder.extract(face_image)

        # ViT path
        import torch

        preprocessed = self.preprocess(face_image)

        if self.use_onnx and hasattr(self, "ort_session"):
            outputs = self.ort_session.run(
                None, {self.input_name: preprocessed}
            )
            embedding = outputs[0][0]
        else:
            with torch.no_grad():
                tensor = torch.from_numpy(preprocessed).to(self.device)
                embedding = self.model(tensor).cpu().numpy()[0]

        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def extract_batch(self, face_images: list) -> np.ndarray:
        """
        Extract embeddings for a batch of aligned face images.

        Args:
            face_images: List of aligned face images (H, W, 3) in RGB.

        Returns:
            L2-normalized embeddings (N, embedding_dim).
        """
        if self.backend == "insightface":
            return self._embedder.extract_batch(face_images)

        # ViT path
        import torch

        preprocessed = np.concatenate(
            [self.preprocess(img) for img in face_images], axis=0
        )

        if self.use_onnx and hasattr(self, "ort_session"):
            outputs = self.ort_session.run(
                None, {self.input_name: preprocessed}
            )
            embeddings = outputs[0]
        else:
            with torch.no_grad():
                tensor = torch.from_numpy(preprocessed).to(self.device)
                embeddings = self.model(tensor).cpu().numpy()

        # L2 normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        embeddings = embeddings / norms

        return embeddings

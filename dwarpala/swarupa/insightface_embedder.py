"""
InsightFace-based embedding extractor for Dwarpala.

Uses the buffalo_l ArcFace R50 recognition model (w600k_r50.onnx) directly
via onnxruntime for face embedding extraction. This is the default inference
backend, providing ~99.8% accuracy on LFW.

The detector is NOT loaded here — face detection is handled by Kavach.
We only load the recognition ONNX and feed it Kavach's aligned 112×112 crops.

License: buffalo_l is for non-commercial research use only (InsightFace license).
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Union

from dwarpala.utils.logger import get_logger
from dwarpala.utils.model_manager import ModelManager

logger = get_logger("swarupa.insightface_embedder")


class InsightFaceEmbedder:
    """
    Face embedding extractor using InsightFace's ArcFace R50 model.

    Loads w600k_r50.onnx directly via onnxruntime — lightweight and fast.
    Produces 512-D L2-normalized embeddings compatible with cosine similarity.

    Preprocessing follows the ArcFace convention:
        - Input: 112×112 BGR image
        - Normalization: (pixel - 127.5) / 128.0
        - Output: 512-D L2-normalized embedding

    Usage:
        embedder = InsightFaceEmbedder()
        embedding = embedder.extract(aligned_face_rgb)  # np.ndarray (512,)
    """

    # Embedding dimensionality produced by ArcFace R50
    EMBEDDING_DIM = 512

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        model_dir: Optional[Path] = None,
    ):
        """
        Args:
            model_path: Explicit path to w600k_r50.onnx. If None, uses ModelManager.
            model_dir: Model directory for ModelManager. Defaults to ~/.dwarpala/models/.
        """
        self.embedding_dim = self.EMBEDDING_DIM

        if model_path is not None:
            onnx_path = Path(model_path)
        else:
            manager = ModelManager(model_dir=model_dir)
            onnx_path = manager.get_model_path("buffalo_l_recognition")

        self._load_model(onnx_path)

    def _load_model(self, onnx_path: Path) -> None:
        """Load the ONNX recognition model."""
        import onnxruntime as ort

        if not onnx_path.exists():
            raise FileNotFoundError(
                f"Recognition model not found: {onnx_path}\n"
                f"Run 'dwarpala download-models' to download required models."
            )

        # Prefer CPU for deterministic results; use CUDA if available
        providers = ["CPUExecutionProvider"]
        try:
            available = ort.get_available_providers()
            if "CUDAExecutionProvider" in available:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        except Exception:
            pass

        self.session = ort.InferenceSession(str(onnx_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name

        # Verify output shape
        output_shape = self.session.get_outputs()[0].shape
        logger.info(
            f"Loaded InsightFace recognition model: {onnx_path.name} "
            f"(input={self.input_name}, output_shape={output_shape})"
        )

    def preprocess(self, face_image: np.ndarray) -> np.ndarray:
        """
        Preprocess an aligned face image for ArcFace inference.

        Args:
            face_image: Aligned face (H, W, 3) in RGB, uint8.
                Expected size: 112×112 from Kavach aligner.

        Returns:
            Preprocessed tensor as numpy array (1, 3, 112, 112) float32.
        """
        img = face_image.copy()

        # Resize to 112×112 if not already (should be from Kavach)
        if img.shape[0] != 112 or img.shape[1] != 112:
            img = cv2.resize(img, (112, 112), interpolation=cv2.INTER_LINEAR)

        # RGB → BGR (ArcFace convention)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # ArcFace normalization: (pixel - 127.5) / 128.0
        # This maps [0, 255] → [-0.9961, 0.9961]
        img = img.astype(np.float32)
        img = (img - 127.5) / 128.0

        # HWC → CHW
        img = np.transpose(img, (2, 0, 1))

        # Add batch dimension: (1, 3, 112, 112)
        img = np.expand_dims(img, axis=0)

        return img

    def extract(self, face_image: np.ndarray) -> np.ndarray:
        """
        Extract embedding from a single aligned face image.

        Args:
            face_image: Aligned face (H, W, 3) in RGB, uint8.
                Must be 112×112 from Kavach aligner with ArcFace reference landmarks.

        Returns:
            L2-normalized embedding (512,) as float32.
        """
        preprocessed = self.preprocess(face_image)
        outputs = self.session.run(None, {self.input_name: preprocessed})
        embedding = outputs[0][0]  # (512,)

        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding.astype(np.float32)

    def extract_batch(self, face_images: list) -> np.ndarray:
        """
        Extract embeddings for a batch of aligned face images.

        Args:
            face_images: List of aligned face images (H, W, 3) in RGB.

        Returns:
            L2-normalized embeddings (N, 512) as float32.
        """
        preprocessed = np.concatenate(
            [self.preprocess(img) for img in face_images], axis=0
        )

        outputs = self.session.run(None, {self.input_name: preprocessed})
        embeddings = outputs[0]  # (N, 512)

        # L2 normalize each embedding
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        embeddings = embeddings / norms

        return embeddings.astype(np.float32)

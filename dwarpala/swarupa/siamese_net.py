"""
Siamese Network for cross-domain face verification.

The Siamese architecture uses weight-shared twin branches to process
the ID photo and live selfie through the same ViT backbone, producing
comparable embeddings despite the domain gap.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple

from dwarpala.swarupa.backbone import ViTBackbone
from dwarpala.utils.logger import get_logger

logger = get_logger("swarupa.siamese")


class SiameseNetwork(nn.Module):
    """
    Siamese network for face verification (1:1 matching).

    Both the ID document photo and the live selfie pass through the
    SAME backbone (shared weights), producing embeddings that can
    be compared via cosine similarity.

    Architecture:
        ID Image ──→ [ViT Backbone] ──→ Embedding A ──┐
                                                        ├→ Cosine Similarity → Score
        Selfie ───→ [ViT Backbone] ──→ Embedding B ──┘
              (shared weights)

    Usage:
        model = SiameseNetwork()
        result = model(id_image_tensor, selfie_tensor)
        print(result["similarity"])  # 0.87
    """

    def __init__(
        self,
        backbone_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        embedding_dim: int = 512,
        drop_rate: float = 0.1,
    ):
        super().__init__()

        # Single shared backbone (this IS the Siamese property)
        self.backbone = ViTBackbone(
            model_name=backbone_name,
            pretrained=pretrained,
            embedding_dim=embedding_dim,
            drop_rate=drop_rate,
        )

        self.embedding_dim = embedding_dim
        logger.info(f"SiameseNetwork initialized: backbone={backbone_name}")

    def forward(
        self,
        image_a: torch.Tensor,
        image_b: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Process two face images and compute their similarity.

        Args:
            image_a: First face image batch (B, 3, 224, 224) — typically ID photo.
            image_b: Second face image batch (B, 3, 224, 224) — typically selfie.

        Returns:
            Dictionary with:
                - embedding_a: (B, embedding_dim)
                - embedding_b: (B, embedding_dim)
                - similarity: (B,) cosine similarity scores
        """
        # Same backbone, shared weights
        embedding_a = self.backbone(image_a)
        embedding_b = self.backbone(image_b)

        # Cosine similarity (embeddings are already L2-normalized)
        similarity = torch.sum(embedding_a * embedding_b, dim=1)

        return {
            "embedding_a": embedding_a,
            "embedding_b": embedding_b,
            "similarity": similarity,
        }

    def extract_embedding(self, image: torch.Tensor) -> torch.Tensor:
        """
        Extract embedding for a single image.

        Args:
            image: Face image batch (B, 3, 224, 224).

        Returns:
            Embedding (B, embedding_dim).
        """
        return self.backbone(image)

    def compute_similarity(
        self,
        embedding_a: torch.Tensor,
        embedding_b: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute cosine similarity between precomputed embeddings.

        Args:
            embedding_a: First embedding (B, embedding_dim).
            embedding_b: Second embedding (B, embedding_dim).

        Returns:
            Similarity scores (B,).
        """
        return torch.sum(embedding_a * embedding_b, dim=1)

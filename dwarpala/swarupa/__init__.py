"""
🔱 Swarupa — True Form
Identity verification engine for Dwarpala.
Cross-domain face matching using Siamese ViT with ArcFace.
Default inference backend: InsightFace buffalo_l (ArcFace R50).
"""

from dwarpala.swarupa.backbone import ViTBackbone
from dwarpala.swarupa.siamese_net import SiameseNetwork
from dwarpala.swarupa.arcface_loss import SubCenterArcFace
from dwarpala.swarupa.embedding import EmbeddingExtractor
from dwarpala.swarupa.insightface_embedder import InsightFaceEmbedder
from dwarpala.swarupa.matcher import FaceMatcher

__all__ = [
    "ViTBackbone",
    "SiameseNetwork",
    "SubCenterArcFace",
    "EmbeddingExtractor",
    "InsightFaceEmbedder",
    "FaceMatcher",
]

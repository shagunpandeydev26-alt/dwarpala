"""
🔱 Swarupa — True Form
Identity verification engine for Dwarpala.
Cross-domain face matching using Siamese ViT with ArcFace.
"""

from dwarpala.swarupa.backbone import ViTBackbone
from dwarpala.swarupa.siamese_net import SiameseNetwork
from dwarpala.swarupa.arcface_loss import SubCenterArcFace
from dwarpala.swarupa.embedding import EmbeddingExtractor
from dwarpala.swarupa.matcher import FaceMatcher

__all__ = [
    "ViTBackbone",
    "SiameseNetwork",
    "SubCenterArcFace",
    "EmbeddingExtractor",
    "FaceMatcher",
]

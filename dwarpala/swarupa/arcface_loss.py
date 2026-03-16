"""
Sub-Center ArcFace Loss with Demographic Parity Regularization.

Standard ArcFace enforces angular margin between classes for discriminative
embeddings. Sub-center ArcFace adds K sub-centers per identity to handle
intra-class variation (e.g., same person with/without glasses, aging).

The demographic parity term penalizes FMR/FRR differentials across
demographic groups during training.

References:
    - ArcFace: Deng et al., "ArcFace: Additive Angular Margin Loss", CVPR 2019
    - Sub-Center ArcFace: Deng et al., "Sub-center ArcFace", ECCV 2020
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

from dwarpala.utils.logger import get_logger

logger = get_logger("swarupa.arcface")


class SubCenterArcFace(nn.Module):
    """
    Sub-Center ArcFace angular margin loss.

    For each class, maintains K sub-centers. During training, only the
    closest sub-center is used for the angular margin penalty, allowing
    the model to learn robust embeddings that handle intra-class variation.

    Loss = -log(e^(s * cos(θ_yi + m)) / (e^(s * cos(θ_yi + m)) + Σ e^(s * cos(θ_j))))

    Where:
        s = scale factor (64)
        m = angular margin (0.5 radians ≈ 28.6°)
        θ_yi = angle between embedding and closest sub-center of true class
        θ_j = angle between embedding and other class centers

    Usage:
        loss_fn = SubCenterArcFace(embedding_dim=512, num_classes=85742, sub_centers=3)
        logits = loss_fn(embeddings, labels)
        loss = F.cross_entropy(logits, labels)
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        num_classes: int = 85742,
        scale: float = 64.0,
        margin: float = 0.50,
        sub_centers: int = 3,
        easy_margin: bool = False,
    ):
        """
        Args:
            embedding_dim: Dimension of face embeddings.
            num_classes: Number of identity classes.
            scale: Scale factor s.
            margin: Angular margin m (radians).
            sub_centers: Number of sub-centers K per class.
            easy_margin: Use easy margin variant.
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.scale = scale
        self.margin = margin
        self.sub_centers = sub_centers
        self.easy_margin = easy_margin

        # Weight matrix: (num_classes, sub_centers, embedding_dim)
        # Each class has K sub-center weight vectors
        self.weight = nn.Parameter(
            torch.FloatTensor(num_classes * sub_centers, embedding_dim)
        )
        nn.init.xavier_uniform_(self.weight)

        # Precompute margin values
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.threshold = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

        logger.info(
            f"SubCenterArcFace: classes={num_classes}, K={sub_centers}, "
            f"s={scale}, m={margin:.2f}"
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute ArcFace logits.

        Args:
            embeddings: L2-normalized face embeddings (B, embedding_dim).
            labels: Class labels (B,).

        Returns:
            Scaled logits (B, num_classes) for cross-entropy loss.
        """
        # Normalize weights
        weight_norm = F.normalize(self.weight, p=2, dim=1)

        # Cosine similarity: (B, num_classes * sub_centers)
        cosine_all = F.linear(embeddings, weight_norm)

        # Reshape to (B, num_classes, sub_centers)
        cosine_all = cosine_all.view(-1, self.num_classes, self.sub_centers)

        # Select maximum cosine across sub-centers (closest sub-center)
        cosine, _ = torch.max(cosine_all, dim=2)  # (B, num_classes)

        # Apply angular margin to the target class
        sine = torch.sqrt(1.0 - torch.clamp(cosine * cosine, 0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m  # cos(θ + m)

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.threshold, phi, cosine - self.mm)

        # One-hot encoding of target class
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        # Apply margin only to target class, keep others unchanged
        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        logits *= self.scale

        return logits


class DemographicParityLoss(nn.Module):
    """
    Regularization loss that penalizes score differentials across
    demographic groups, promoting fairness.

    L_parity = Σ_g |mean_score_g - mean_score_global|²

    This encourages the model to produce similar confidence distributions
    across all demographic subgroups.
    """

    def __init__(self, lambda_fairness: float = 0.1):
        """
        Args:
            lambda_fairness: Weight of the parity loss.
        """
        super().__init__()
        self.lambda_fairness = lambda_fairness

    def forward(
        self,
        scores: torch.Tensor,
        groups: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute demographic parity regularization.

        Args:
            scores: Predicted scores (B,).
            groups: Demographic group labels (B,).

        Returns:
            Parity loss scalar.
        """
        global_mean = scores.mean()
        unique_groups = torch.unique(groups)

        parity_loss = torch.tensor(0.0, device=scores.device)

        for g in unique_groups:
            mask = groups == g
            if mask.sum() > 0:
                group_mean = scores[mask].mean()
                parity_loss += (group_mean - global_mean) ** 2

        return self.lambda_fairness * parity_loss

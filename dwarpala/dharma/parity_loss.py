"""
Demographic Parity Regularization for training fairness.
Penalizes FMR/FRR differentials across demographic groups during training.
"""

import torch
import torch.nn as nn
from typing import Optional

from dwarpala.utils.logger import get_logger

logger = get_logger("dharma.parity")


class DemographicParityRegularizer(nn.Module):
    """
    Training-time regularization that ensures similar error rates
    across demographic groups.

    L_parity = Σ_g (mean_confidence_g - mean_confidence_global)²

    This is added to the main training loss:
    L_total = L_arcface + λ * L_parity

    The effect: the model is penalized if it's significantly more
    or less confident for any particular demographic group, forcing
    it to learn equally discriminative features across all populations.

    Usage:
        regularizer = DemographicParityRegularizer(lambda_fairness=0.1)
        parity_loss = regularizer(similarity_scores, demographic_groups)
        total_loss = arcface_loss + parity_loss
    """

    def __init__(
        self,
        lambda_fairness: float = 0.1,
        max_fmr_differential: float = 5.0,
    ):
        """
        Args:
            lambda_fairness: Weight of parity regularization.
            max_fmr_differential: Maximum allowed FMR ratio for alerting.
        """
        super().__init__()
        self.lambda_fairness = lambda_fairness
        self.max_fmr_differential = max_fmr_differential

        logger.info(
            f"DemographicParityRegularizer: λ={lambda_fairness}, "
            f"max_diff={max_fmr_differential}x"
        )

    def forward(
        self,
        scores: torch.Tensor,
        groups: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute demographic parity loss.

        Args:
            scores: Predicted confidence/similarity scores (B,).
            groups: Demographic group labels (B,).
            labels: Optional ground truth for FMR/FRR specific penalties.

        Returns:
            Parity regularization loss.
        """
        global_mean = scores.mean()
        unique_groups = torch.unique(groups)

        parity_loss = torch.tensor(0.0, device=scores.device, requires_grad=True)

        for g in unique_groups:
            mask = groups == g
            if mask.sum() < 2:  # Need at least 2 samples
                continue

            group_mean = scores[mask].mean()
            group_var = scores[mask].var()

            # Penalize mean difference (different confidence levels)
            mean_penalty = (group_mean - global_mean) ** 2

            # Penalize variance difference (different consistency)
            global_var = scores.var()
            var_penalty = (group_var - global_var) ** 2

            parity_loss = parity_loss + mean_penalty + 0.5 * var_penalty

        return self.lambda_fairness * parity_loss

    def compute_differential(
        self,
        scores: torch.Tensor,
        groups: torch.Tensor,
    ) -> dict:
        """
        Compute the actual FMR differential for monitoring.
        Does not contribute to gradient — for logging only.
        """
        with torch.no_grad():
            unique_groups = torch.unique(groups)
            group_means = {}

            for g in unique_groups:
                mask = groups == g
                group_means[int(g)] = float(scores[mask].mean())

            if len(group_means) >= 2:
                values = list(group_means.values())
                max_diff = max(values) / (min(values) + 1e-10)
            else:
                max_diff = 1.0

            return {
                "group_means": group_means,
                "max_differential": max_diff,
                "exceeds_threshold": max_diff > self.max_fmr_differential,
            }

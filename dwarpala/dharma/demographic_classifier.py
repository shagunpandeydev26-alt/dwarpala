"""
Demographic classifier for fairness analysis.
Estimates age group, gender, and perceived ethnicity from face embeddings
for use in bias auditing — NOT for discrimination.

This classifier's purpose is EXCLUSIVELY to measure and mitigate bias
in the verification system. It ensures Dharma (fairness) is upheld.
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import Optional

from dwarpala.utils.logger import get_logger

logger = get_logger("dharma.classifier")

# FairFace categories
AGE_GROUPS = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60+"]
GENDERS = ["Male", "Female"]
ETHNICITIES = [
    "East Asian", "Indian", "Black", "White",
    "Middle Eastern", "Latino/Hispanic", "Southeast Asian",
]


@dataclass
class DemographicPrediction:
    """Predicted demographic attributes for bias auditing."""

    age_group: str
    age_group_confidence: float
    gender: str
    gender_confidence: float
    ethnicity: str
    ethnicity_confidence: float

    def __str__(self):
        return (
            f"Demographics: age={self.age_group}({self.age_group_confidence:.0%}), "
            f"gender={self.gender}({self.gender_confidence:.0%}), "
            f"ethnicity={self.ethnicity}({self.ethnicity_confidence:.0%})"
        )

    @property
    def group_key(self) -> str:
        """Unique key for this demographic subgroup."""
        return f"{self.ethnicity}_{self.gender}_{self.age_group}"


class DemographicClassifier(nn.Module):
    """
    Lightweight demographic classifier for bias measurement.

    Uses a simple MLP on top of face embeddings (from the Swarupa backbone)
    to predict demographic attributes. This avoids the need for a separate
    heavy model.

    ⚠️ ETHICAL NOTE: This classifier exists ONLY for measuring and
    reducing bias in the verification system. It should never be used
    for profiling, discrimination, or any purpose beyond fairness auditing.

    Usage:
        classifier = DemographicClassifier(embedding_dim=512)
        prediction = classifier.predict(face_embedding)
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        hidden_dim: int = 256,
    ):
        super().__init__()

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # Task-specific heads
        self.age_head = nn.Linear(hidden_dim, len(AGE_GROUPS))
        self.gender_head = nn.Linear(hidden_dim, len(GENDERS))
        self.ethnicity_head = nn.Linear(hidden_dim, len(ETHNICITIES))

        logger.info(
            f"DemographicClassifier: dim={embedding_dim}, "
            f"ages={len(AGE_GROUPS)}, genders={len(GENDERS)}, "
            f"ethnicities={len(ETHNICITIES)}"
        )

    def forward(self, embedding: torch.Tensor) -> dict:
        """
        Predict demographic attributes from face embedding.

        Args:
            embedding: Face embedding (B, embedding_dim).

        Returns:
            Dictionary of logits for each attribute.
        """
        features = self.shared(embedding)

        return {
            "age": self.age_head(features),
            "gender": self.gender_head(features),
            "ethnicity": self.ethnicity_head(features),
        }

    @torch.no_grad()
    def predict(self, embedding: np.ndarray) -> DemographicPrediction:
        """
        Predict demographics from a single embedding.

        Args:
            embedding: Face embedding (embedding_dim,).

        Returns:
            DemographicPrediction with predicted attributes.
        """
        self.eval()
        tensor = torch.from_numpy(embedding).unsqueeze(0).float()

        if next(self.parameters(), None) is not None:
            device = next(self.parameters()).device
            tensor = tensor.to(device)

        logits = self.forward(tensor)

        age_probs = torch.softmax(logits["age"], dim=1)[0]
        gender_probs = torch.softmax(logits["gender"], dim=1)[0]
        ethnicity_probs = torch.softmax(logits["ethnicity"], dim=1)[0]

        age_idx = int(torch.argmax(age_probs))
        gender_idx = int(torch.argmax(gender_probs))
        ethnicity_idx = int(torch.argmax(ethnicity_probs))

        return DemographicPrediction(
            age_group=AGE_GROUPS[age_idx],
            age_group_confidence=float(age_probs[age_idx]),
            gender=GENDERS[gender_idx],
            gender_confidence=float(gender_probs[gender_idx]),
            ethnicity=ETHNICITIES[ethnicity_idx],
            ethnicity_confidence=float(ethnicity_probs[ethnicity_idx]),
        )

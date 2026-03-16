"""
⚖️ Dharma — Righteousness
Fairness and demographic integrity module for Dwarpala.
Ensures equal treatment across all populations.
"""

from dwarpala.dharma.demographic_classifier import DemographicClassifier
from dwarpala.dharma.parity_loss import DemographicParityRegularizer
from dwarpala.dharma.bias_auditor import BiasAuditor

__all__ = ["DemographicClassifier", "DemographicParityRegularizer", "BiasAuditor"]

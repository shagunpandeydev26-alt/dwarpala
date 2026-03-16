"""
Bias Auditor for post-deployment fairness monitoring.
Analyzes FMR/FRR differentials across demographic subgroups
and generates audit reports.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from collections import defaultdict

from dwarpala.utils.logger import get_logger

logger = get_logger("dharma.auditor")


@dataclass
class BiasReport:
    """Comprehensive bias audit report."""

    total_samples: int
    num_groups: int
    per_group_metrics: Dict[str, dict]
    max_fmr_differential: float
    max_frr_differential: float
    fairness_passed: bool
    issues: List[str] = field(default_factory=list)

    def __str__(self):
        status = "✅ FAIR" if self.fairness_passed else "⚠️ BIASED"
        return (
            f"Bias Audit {status} | groups={self.num_groups} "
            f"| FMR_diff={self.max_fmr_differential:.2f}x "
            f"| FRR_diff={self.max_frr_differential:.2f}x"
            + (f"\n  Issues: {'; '.join(self.issues)}" if self.issues else "")
        )


class BiasAuditor:
    """
    Post-deployment bias auditor.

    Collects verification results tagged with demographic information
    and produces fairness reports showing FMR/FRR differentials
    across demographic subgroups.

    According to NIST FRVT:
    - FMR can vary 10x-100x across demographics in poor systems
    - Our target: < 5x differential

    Usage:
        auditor = BiasAuditor(max_differential=5.0)
        auditor.add_result(prediction=True, ground_truth=True, group="White_Male_20-29")
        # ... add many more results ...
        report = auditor.generate_report()
    """

    def __init__(self, max_differential: float = 5.0):
        """
        Args:
            max_differential: Maximum acceptable FMR/FRR ratio.
        """
        self.max_differential = max_differential

        # Collect results per demographic group
        self._results: Dict[str, List[dict]] = defaultdict(list)
        self._total = 0

        logger.info(f"BiasAuditor: max_differential={max_differential}x")

    def add_result(
        self,
        prediction: bool,
        ground_truth: bool,
        group: str,
        score: float = 0.0,
    ):
        """
        Record a single verification result.

        Args:
            prediction: System's prediction (True=accept, False=reject).
            ground_truth: Actual label (True=genuine, False=impostor).
            group: Demographic group key (e.g., "White_Male_20-29").
            score: Raw similarity/liveness score.
        """
        self._results[group].append({
            "prediction": prediction,
            "ground_truth": ground_truth,
            "score": score,
        })
        self._total += 1

    def generate_report(self) -> BiasReport:
        """
        Generate a comprehensive bias audit report.

        Returns:
            BiasReport with per-group metrics and fairness assessment.
        """
        per_group = {}
        fmr_values = []
        frr_values = []
        issues = []

        for group, results in self._results.items():
            n = len(results)
            if n == 0:
                continue

            preds = np.array([r["prediction"] for r in results])
            truths = np.array([r["ground_truth"] for r in results])
            scores_arr = np.array([r["score"] for r in results])

            # False Match Rate (impostor accepted)
            impostors = truths == False
            if impostors.sum() > 0:
                fmr = float(preds[impostors].sum() / impostors.sum())
            else:
                fmr = 0.0

            # False Reject Rate (genuine rejected)
            genuines = truths == True
            if genuines.sum() > 0:
                frr = float((~preds[genuines]).sum() / genuines.sum())
            else:
                frr = 0.0

            per_group[group] = {
                "count": n,
                "fmr": fmr,
                "frr": frr,
                "mean_score": float(np.mean(scores_arr)),
                "std_score": float(np.std(scores_arr)),
            }

            if fmr > 0:
                fmr_values.append(fmr)
            if frr > 0:
                frr_values.append(frr)

        # Compute differentials
        max_fmr_diff = 1.0
        max_frr_diff = 1.0

        if len(fmr_values) >= 2:
            max_fmr_diff = max(fmr_values) / min(fmr_values)
        if len(frr_values) >= 2:
            max_frr_diff = max(frr_values) / min(frr_values)

        # Check for issues
        if max_fmr_diff > self.max_differential:
            issues.append(
                f"FMR differential {max_fmr_diff:.1f}x exceeds "
                f"threshold {self.max_differential}x"
            )

        if max_frr_diff > self.max_differential:
            issues.append(
                f"FRR differential {max_frr_diff:.1f}x exceeds "
                f"threshold {self.max_differential}x"
            )

        # Check for underrepresented groups
        if per_group:
            counts = [g["count"] for g in per_group.values()]
            min_count = min(counts)
            max_count = max(counts)
            if max_count > 10 * min_count:
                issues.append(
                    f"Severe sample imbalance: "
                    f"min={min_count}, max={max_count}"
                )

        fairness_passed = len(issues) == 0

        report = BiasReport(
            total_samples=self._total,
            num_groups=len(per_group),
            per_group_metrics=per_group,
            max_fmr_differential=max_fmr_diff,
            max_frr_differential=max_frr_diff,
            fairness_passed=fairness_passed,
            issues=issues,
        )

        logger.info(str(report))
        return report

    def reset(self):
        """Clear all collected results."""
        self._results.clear()
        self._total = 0

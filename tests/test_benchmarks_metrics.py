"""
Unit tests for benchmarks/metrics.py — pure metric helpers, no models or network,
so they run in CI (the benchmark scripts themselves are requires_models / network
and do NOT run in CI).
"""

import numpy as np

from benchmarks.metrics import (
    confusion_by_attack,
    fmt_pct,
    liveness_layer_metrics,
    markdown_table,
    per_attack_breakdown,
    verification_metrics,
)


class TestVerificationMetrics:
    def test_perfect_separation(self):
        genuine = [0.9, 0.85, 0.95, 0.8]
        impostor = [0.1, 0.05, 0.2, 0.0]
        m = verification_metrics(genuine, impostor, far_targets=(1e-2,))
        assert m["roc_auc"] == 1.0
        assert m["accuracy"] == 1.0
        assert m["eer"] == 0.0
        # A threshold between 0.2 and 0.8 separates the two clouds.
        assert 0.2 < m["best_threshold"] <= 0.8

    def test_tar_at_far_keys_and_threshold(self):
        rng = np.random.default_rng(0)
        genuine = list(rng.normal(0.6, 0.05, 200))
        impostor = list(rng.normal(0.0, 0.05, 200))
        m = verification_metrics(genuine, impostor, far_targets=(1e-2, 1e-3))
        assert "far=0.01" in m["tar_at_far"]
        assert "far=0.001" in m["tar_at_far"]
        assert 0.0 <= m["tar_at_far"]["far=0.01"]["tar"] <= 1.0

    def test_requires_both_classes(self):
        import pytest

        with pytest.raises(ValueError):
            verification_metrics([0.5], [])


class TestLivenessLayerMetrics:
    def test_all_correct(self):
        # 1=live, 0=spoof; scores cleanly separated around 0.5.
        scores = [0.9, 0.8, 0.1, 0.2]
        labels = [1, 1, 0, 0]
        m = liveness_layer_metrics(scores, labels, threshold=0.5)
        assert m["apcer"] == 0.0
        assert m["bpcer"] == 0.0
        assert m["acer"] == 0.0

    def test_one_spoof_accepted(self):
        # A spoof at 0.7 sneaks past threshold 0.5 -> APCER = 1/2.
        scores = [0.9, 0.8, 0.7, 0.2]
        labels = [1, 1, 0, 0]
        m = liveness_layer_metrics(scores, labels, threshold=0.5)
        assert m["apcer"] == 0.5
        assert m["bpcer"] == 0.0
        assert m["acer"] == 0.25


class TestConfusionByAttack:
    def test_counts(self):
        scores = [0.9, 0.8, 0.1, 0.7]  # last spoof wrongly accepted
        labels = [1, 1, 0, 0]
        attacks = ["bona_fide", "bona_fide", "print", "screen"]
        conf = confusion_by_attack(scores, labels, attacks, threshold=0.5)
        assert conf["bona_fide"] == {"accepted": 2, "total": 2}
        assert conf["print"] == {"rejected": 1, "total": 1}
        assert conf["screen"] == {"rejected": 0, "total": 1}


class TestPerAttackBreakdown:
    def test_apcer_per_type(self):
        scores = [0.1, 0.7]
        labels = [0, 0]
        attacks = ["print", "screen"]
        out = per_attack_breakdown(scores, labels, attacks, threshold=0.5)
        assert out["print"] == 0.0  # rejected -> not accepted
        assert out["screen"] == 1.0  # accepted -> APCER 100%


class TestFormatting:
    def test_fmt_pct(self):
        assert fmt_pct(0.9912) == "99.12%"

    def test_markdown_table(self):
        t = markdown_table(["A", "B"], [[1, 2], [3, 4]])
        lines = t.splitlines()
        assert lines[0] == "| A | B |"
        assert lines[1] == "| --- | --- |"
        assert lines[2] == "| 1 | 2 |"

"""Tests for src.monitoring.drift — KS-test drift detection."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.monitoring.drift import compute_feature_drift, compute_prediction_drift, summarize_drift


class TestComputeFeatureDrift:
    def test_identical_distributions_no_drift(self):
        rng = np.random.RandomState(42)
        data = rng.randn(500)
        ref = pd.DataFrame({"feat_a": data, "feat_b": data * 2})
        cur = pd.DataFrame({"feat_a": data, "feat_b": data * 2})
        result = compute_feature_drift(ref, cur, ["feat_a", "feat_b"])
        assert result["drift_detected"] == False  # noqa: E712
        assert result["n_drifted"] == 0

    def test_different_distributions_detect_drift(self):
        rng = np.random.RandomState(42)
        ref = pd.DataFrame({"feat_a": rng.randn(500)})
        cur = pd.DataFrame({"feat_a": rng.randn(500) + 5})  # shifted mean
        result = compute_feature_drift(ref, cur, ["feat_a"])
        assert result["drift_detected"] == True  # noqa: E712
        assert result["n_drifted"] == 1
        assert result["feature_results"][0]["drifted"] == True  # noqa: E712

    def test_missing_feature_skipped(self):
        ref = pd.DataFrame({"feat_a": [1, 2, 3]})
        cur = pd.DataFrame({"feat_a": [1, 2, 3]})
        result = compute_feature_drift(ref, cur, ["feat_a", "feat_missing"])
        assert len(result["feature_results"]) == 1

    def test_non_numeric_feature_skipped(self):
        ref = pd.DataFrame({"feat_a": ["a", "b", "c"]})
        cur = pd.DataFrame({"feat_a": ["a", "b", "c"]})
        result = compute_feature_drift(ref, cur, ["feat_a"])
        assert len(result["feature_results"]) == 0

    def test_custom_p_threshold(self):
        rng = np.random.RandomState(42)
        ref = pd.DataFrame({"feat_a": rng.randn(100)})
        cur = pd.DataFrame({"feat_a": rng.randn(100) + 0.3})  # slight shift
        strict = compute_feature_drift(ref, cur, ["feat_a"], p_threshold=0.50)
        lenient = compute_feature_drift(ref, cur, ["feat_a"], p_threshold=0.001)
        # Strict threshold should detect drift more easily
        assert strict["n_drifted"] >= lenient["n_drifted"]


class TestComputePredictionDrift:
    def test_identical_no_drift(self):
        probas = np.array([0.1, 0.5, 0.9])
        result = compute_prediction_drift(probas, probas)
        assert result["drifted"] == False  # noqa: E712

    def test_different_drift(self):
        rng = np.random.RandomState(42)
        ref = rng.uniform(0, 0.3, 500)
        cur = rng.uniform(0.7, 1.0, 500)
        result = compute_prediction_drift(ref, cur)
        assert result["drifted"] == True  # noqa: E712

    def test_empty_arrays(self):
        result = compute_prediction_drift(np.array([]), np.array([]))
        assert result["drifted"] == False  # noqa: E712
        assert result["p_value"] == 1.0


class TestSummarizeDrift:
    def test_no_drift_summary(self):
        feat = {"drift_detected": False, "n_drifted": 0, "feature_results": []}
        pred = {"drifted": False, "ks_statistic": 0.01, "p_value": 0.95}
        summary = summarize_drift(feat, pred)
        assert summary["any_drift"] is False
        assert "No drift detected" in summary["summary"]

    def test_drift_summary(self):
        feat = {
            "drift_detected": True,
            "n_drifted": 1,
            "feature_results": [{"feature": "feat_a", "drifted": True}],
        }
        pred = {"drifted": True, "ks_statistic": 0.85, "p_value": 0.001}
        summary = summarize_drift(feat, pred)
        assert summary["any_drift"] is True
        assert "feat_a" in summary["summary"]

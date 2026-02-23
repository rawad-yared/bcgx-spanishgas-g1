"""Tests for src.models — preprocessing, churn model, scorer."""

import numpy as np
import pandas as pd
import pytest

from src.models.churn_model import evaluate_model, get_model_definitions, pick_threshold
from src.models.preprocessing import build_preprocessing_pipeline
from src.models.scorer import assign_risk_tiers


@pytest.fixture
def sample_X():
    np.random.seed(42)
    return pd.DataFrame({
        "tenure_months": np.random.uniform(1, 60, 200),
        "monthly_elec_kwh": np.random.uniform(50, 500, 200),
        "is_dual_fuel": np.random.choice([0, 1], 200),
        "segment": np.random.choice(["Residential", "SME", "Corporate"], 200),
    })


@pytest.fixture
def sample_y():
    np.random.seed(42)
    return pd.Series(np.random.choice([0, 1], 200, p=[0.9, 0.1]))


class TestPreprocessing:
    def test_builds_column_transformer(self, sample_X):
        ct = build_preprocessing_pipeline(sample_X, scale_numeric=True)
        assert ct is not None
        assert len(ct.transformers) == 2  # num + cat

    def test_fit_transform(self, sample_X):
        ct = build_preprocessing_pipeline(sample_X, scale_numeric=False)
        result = ct.fit_transform(sample_X)
        assert result.shape[0] == len(sample_X)


class TestModelDefinitions:
    def test_returns_models(self, sample_y):
        models = get_model_definitions(sample_y)
        assert "logistic_regression" in models
        assert "random_forest" in models

    def test_xgboost_included_when_installed(self, sample_y):
        pytest.importorskip("xgboost")
        models = get_model_definitions(sample_y)
        assert "xgboost" in models


class TestPickThreshold:
    def test_threshold_in_valid_range(self):
        y_true = np.array([0]*90 + [1]*10)
        y_proba = np.random.RandomState(42).random(100)
        threshold = pick_threshold(y_true, y_proba, target_recall=0.50)
        assert 0 <= threshold <= 1


class TestEvaluateModel:
    def test_returns_all_metrics(self):
        y_true = np.array([0]*80 + [1]*20)
        y_proba = np.random.RandomState(42).random(100)
        metrics = evaluate_model(y_true, y_proba, threshold=0.5)
        assert "roc_auc" in metrics
        assert "pr_auc" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "tp" in metrics


class TestAssignRiskTiers:
    def test_tier_labels(self):
        scored = pd.DataFrame({
            "customer_id": ["C1", "C2", "C3", "C4"],
            "churn_proba": [0.10, 0.45, 0.75, 0.90],
        })
        result = assign_risk_tiers(scored)
        assert result.loc[0, "risk_tier"] == "Low (<40%)"
        assert result.loc[1, "risk_tier"] == "Medium (40-60%)"
        assert result.loc[2, "risk_tier"] == "High (60-80%)"
        assert result.loc[3, "risk_tier"] == "Critical (>80%)"

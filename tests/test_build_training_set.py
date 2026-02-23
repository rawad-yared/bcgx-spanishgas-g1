"""Tests for src.data.build_training_set."""

import numpy as np
import pandas as pd
import pytest

from src.data.build_training_set import build_model_matrix, create_train_test_split


@pytest.fixture
def gold_df():
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "customer_id": [f"C{i:04d}" for i in range(n)],
        "churn": np.random.choice([0, 1], size=n, p=[0.9, 0.1]),
        "tenure_months": np.random.uniform(1, 60, n),
        "monthly_elec_kwh": np.random.uniform(50, 500, n),
        "is_dual_fuel": np.random.choice([0, 1], n),
        "last_interaction_days_ago": np.where(
            np.random.random(n) > 0.3, np.random.uniform(1, 365, n), np.nan
        ),
        "sentiment_label": np.where(
            np.random.random(n) > 0.4,
            np.random.choice(["Positive", "Negative", "Neutral"], n),
            None,
        ),
        "segment": np.random.choice(["Residential", "SME", "Corporate"], n),
    })


class TestBuildModelMatrix:
    def test_returns_x_y_cids(self, gold_df):
        features = ["tenure_months", "monthly_elec_kwh", "is_dual_fuel"]
        X, y, cids = build_model_matrix(gold_df, features)
        assert len(X) == len(gold_df)
        assert len(y) == len(gold_df)
        assert len(cids) == len(gold_df)
        assert list(X.columns) == features

    def test_structural_fill_numeric(self, gold_df):
        features = ["tenure_months", "last_interaction_days_ago"]
        X, _, _ = build_model_matrix(gold_df, features)
        # All NaN in last_interaction_days_ago should be 9999
        assert not X["last_interaction_days_ago"].isna().any()
        assert (X["last_interaction_days_ago"] == 9999).sum() > 0

    def test_structural_fill_categorical(self, gold_df):
        features = ["tenure_months", "sentiment_label"]
        X, _, _ = build_model_matrix(gold_df, features)
        assert not X["sentiment_label"].isna().any()
        assert (X["sentiment_label"] == "no_interaction").sum() > 0

    def test_missing_features_warned(self, gold_df):
        features = ["tenure_months", "nonexistent_feature"]
        with pytest.warns(UserWarning, match="not in gold master"):
            X, _, _ = build_model_matrix(gold_df, features)
        assert "nonexistent_feature" not in X.columns


class TestTrainTestSplit:
    def test_stratified_split(self, gold_df):
        features = ["tenure_months", "monthly_elec_kwh"]
        X, y, _ = build_model_matrix(gold_df, features)
        X_train, X_test, y_train, y_test = create_train_test_split(X, y)
        # Check sizes
        assert len(X_train) + len(X_test) == len(X)
        # Check stratification (churn rates similar)
        train_rate = y_train.mean()
        test_rate = y_test.mean()
        assert abs(train_rate - test_rate) < 0.1

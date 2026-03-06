"""Tests for src.models.artifacts — model save/load to S3."""

from __future__ import annotations

import boto3
import numpy as np
from moto import mock_aws
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.artifacts import load_model, save_model

BUCKET = "test-bucket"
REGION = "us-east-1"


def _make_pipeline():
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(random_state=42)),
    ])
    X = np.random.RandomState(42).randn(50, 3)
    y = np.random.RandomState(42).choice([0, 1], 50)
    pipe.fit(X, y)
    return pipe


class TestArtifactRoundTrip:
    def test_save_load_pipeline(self):
        with mock_aws():
            boto3.client("s3", region_name=REGION).create_bucket(Bucket=BUCKET)
            pipe = _make_pipeline()
            metrics = {"pr_auc": 0.82, "roc_auc": 0.90}

            save_model(pipe, 0.45, metrics, "test_model", BUCKET, "models/v1/", region=REGION)
            loaded_pipe, metadata = load_model(BUCKET, "models/v1/", region=REGION)

            assert metadata["model_name"] == "test_model"
            assert metadata["threshold"] == 0.45
            assert metadata["metrics"]["pr_auc"] == 0.82

            # Verify pipeline predictions match
            X_test = np.random.RandomState(99).randn(5, 3)
            np.testing.assert_array_equal(pipe.predict(X_test), loaded_pipe.predict(X_test))

    def test_metadata_fields(self):
        with mock_aws():
            boto3.client("s3", region_name=REGION).create_bucket(Bucket=BUCKET)
            pipe = _make_pipeline()

            save_model(pipe, 0.50, {"f1": 0.7}, "rf_model", BUCKET, "models/v2/", region=REGION)
            _, metadata = load_model(BUCKET, "models/v2/", region=REGION)

            assert "model_name" in metadata
            assert "threshold" in metadata
            assert "metrics" in metadata

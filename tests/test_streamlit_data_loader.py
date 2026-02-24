"""Tests for src.serving.ui.data_loader — cached data loading."""

from __future__ import annotations

import json

import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def _mock_streamlit(monkeypatch):
    """Replace st.cache_data with a passthrough decorator before importing data_loader."""
    import streamlit as st
    monkeypatch.setattr(st, "cache_data", lambda **kwargs: lambda fn: fn)


class TestLoadScoredData:
    def test_load_from_local_parquet(self, tmp_path):
        from src.serving.ui.data_loader import load_scored_data

        df = pd.DataFrame({"customer_id": ["C1"], "churn_proba": [0.5]})
        path = tmp_path / "scored.parquet"
        df.to_parquet(path, index=False)
        result = load_scored_data(source="local", path=str(path))
        assert len(result) == 1
        assert "customer_id" in result.columns

    def test_missing_file_returns_empty(self):
        from src.serving.ui.data_loader import load_scored_data

        result = load_scored_data(source="local", path="/nonexistent/file.parquet")
        assert isinstance(result, pd.DataFrame)
        assert result.empty


class TestLoadModelMetrics:
    def test_load_from_local_json(self, tmp_path):
        from src.serving.ui.data_loader import load_model_metrics

        metrics = {"pr_auc": 0.82, "roc_auc": 0.90}
        path = tmp_path / "metrics.json"
        path.write_text(json.dumps(metrics))
        result = load_model_metrics(source="local", path=str(path))
        assert result["pr_auc"] == 0.82

    def test_missing_file_returns_empty_dict(self):
        from src.serving.ui.data_loader import load_model_metrics

        result = load_model_metrics(source="local", path="/nonexistent/metrics.json")
        assert result == {}


class TestLoadDriftResults:
    def test_load_from_local_json(self, tmp_path):
        from src.serving.ui.data_loader import load_drift_results

        drift = {"any_drift": True, "n_features_drifted": 2}
        path = tmp_path / "drift.json"
        path.write_text(json.dumps(drift))
        result = load_drift_results(source="local", path=str(path))
        assert result["any_drift"] is True

    def test_missing_file_returns_empty_dict(self):
        from src.serving.ui.data_loader import load_drift_results

        result = load_drift_results(source="local", path="/nonexistent/drift.json")
        assert result == {}


class TestLoadPipelineRuns:
    def test_load_from_local_json(self, tmp_path):
        from src.serving.ui.data_loader import load_pipeline_runs

        runs = [{"run_id": "r1", "status": "completed"}, {"run_id": "r2", "status": "started"}]
        path = tmp_path / "runs.json"
        path.write_text(json.dumps(runs))
        result = load_pipeline_runs(source="local", path=str(path))
        assert len(result) == 2

    def test_missing_file_returns_empty_list(self):
        from src.serving.ui.data_loader import load_pipeline_runs

        result = load_pipeline_runs(source="local", path="/nonexistent/runs.json")
        assert result == []


class TestLoadGoldData:
    def test_load_from_local_parquet(self, tmp_path):
        from src.serving.ui.data_loader import load_gold_data

        df = pd.DataFrame({"customer_id": ["C1"], "churn": [1], "segment": ["SME"]})
        path = tmp_path / "gold.parquet"
        df.to_parquet(path, index=False)
        result = load_gold_data(source="local", path=str(path))
        assert len(result) == 1
        assert "segment" in result.columns

    def test_missing_file_returns_empty(self):
        from src.serving.ui.data_loader import load_gold_data

        result = load_gold_data(source="local", path="/nonexistent/gold.parquet")
        assert isinstance(result, pd.DataFrame)
        assert result.empty


class TestLoadRecommendations:
    def test_load_from_local_parquet(self, tmp_path):
        from src.serving.ui.data_loader import load_recommendations

        df = pd.DataFrame(
            {
                "customer_id": ["C1", "C2"],
                "risk_score": [0.9, 0.3],
                "action": ["offer_large", "no_offer"],
            }
        )
        path = tmp_path / "recommendations.parquet"
        df.to_parquet(path, index=False)
        result = load_recommendations(source="local", path=str(path))
        assert len(result) == 2
        assert "action" in result.columns

    def test_missing_file_returns_empty(self):
        from src.serving.ui.data_loader import load_recommendations

        result = load_recommendations(source="local", path="/nonexistent/reco.parquet")
        assert isinstance(result, pd.DataFrame)
        assert result.empty

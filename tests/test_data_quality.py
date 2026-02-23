"""Tests for src.monitoring.data_quality — data quality checks."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.monitoring.data_quality import check_data_quality


class TestCheckDataQuality:
    def test_clean_data_passes(self):
        df = pd.DataFrame({
            "customer_id": ["C1", "C2", "C3"],
            "value": [10.0, 20.0, 30.0],
        })
        result = check_data_quality(df, layer="gold")
        assert result["passed"] is True
        assert result["row_count"] == 3
        assert result["issues"] == []

    def test_high_nulls_flagged(self):
        df = pd.DataFrame({
            "customer_id": ["C1", "C2", "C3", "C4", "C5"],
            "value": [1.0, np.nan, np.nan, np.nan, np.nan],
        })
        result = check_data_quality(df, layer="gold")
        assert result["passed"] is False
        assert any("null rate" in issue.lower() for issue in result["issues"])

    def test_duplicate_keys_flagged(self):
        df = pd.DataFrame({
            "customer_id": ["C1", "C1", "C2"],
            "value": [10.0, 20.0, 30.0],
        })
        result = check_data_quality(df, layer="bronze")
        assert result["passed"] is False
        assert result["duplicate_keys"] > 0

    def test_empty_dataframe_flagged(self):
        df = pd.DataFrame({"customer_id": pd.Series(dtype="str"), "value": pd.Series(dtype="float64")})
        result = check_data_quality(df)
        assert result["passed"] is False
        assert result["row_count"] == 0

    def test_numeric_ranges_computed(self):
        df = pd.DataFrame({
            "customer_id": ["C1", "C2"],
            "amount": [100.0, 200.0],
        })
        result = check_data_quality(df)
        assert "amount" in result["numeric_ranges"]
        assert result["numeric_ranges"]["amount"]["min"] == 100.0
        assert result["numeric_ranges"]["amount"]["max"] == 200.0

    def test_schema_output(self):
        df = pd.DataFrame({"customer_id": ["C1"], "count": [5]})
        result = check_data_quality(df)
        assert len(result["schema"]) == 2
        assert result["schema"][0]["column"] == "customer_id"

    def test_layer_thresholds_differ(self):
        # raw layer allows up to 80% nulls, gold only 10%
        df = pd.DataFrame({
            "customer_id": ["C1", "C2", "C3", "C4", "C5"],
            "value": [1.0, np.nan, np.nan, np.nan, np.nan],
        })
        raw_result = check_data_quality(df, layer="raw")
        gold_result = check_data_quality(df, layer="gold")
        assert raw_result["passed"] is True
        assert gold_result["passed"] is False

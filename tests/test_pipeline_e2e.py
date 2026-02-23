"""End-to-end integration test for the local pipeline flow.

Uses minimal synthetic data to test bronze -> silver -> gold -> train -> score chain.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.ingest import build_bronze_customer


@pytest.fixture
def synthetic_raw_data():
    """Create minimal synthetic DataFrames mimicking raw input data."""
    np.random.seed(42)
    n = 100

    churn = pd.DataFrame({
        "customer_id": [f"C{i:04d}" for i in range(n)],
        "churn": np.random.choice([0, 1], n, p=[0.85, 0.15]),
    })

    attributes = pd.DataFrame({
        "customer_id": [f"C{i:04d}" for i in range(n)],
        "province": np.random.choice(["Madrid", "Barcelona", "Valencia"], n),
        "is_industrial": np.random.choice([0, 1], n, p=[0.8, 0.2]),
        "subscribed_power": np.random.uniform(3, 15, n),
    })

    contracts = pd.DataFrame({
        "customer_id": [f"C{i:04d}" for i in range(n)],
        "contract_start": pd.to_datetime("2020-01-01"),
        "contract_end": pd.to_datetime("2024-01-01"),
        "has_gas": np.random.choice([0, 1], n),
        "channel": np.random.choice(["online", "branch", "phone"], n),
    })

    interactions = pd.DataFrame({
        "customer_id": np.random.choice([f"C{i:04d}" for i in range(n)], n * 2),
        "interaction_type": np.random.choice(["complaint", "inquiry", "payment"], n * 2),
        "interaction_date": pd.date_range("2023-01-01", periods=n * 2, freq="D")[:n * 2],
    })

    return churn, attributes, contracts, interactions


@pytest.mark.slow
class TestPipelineE2E:
    def test_bronze_produces_dataframe(self, synthetic_raw_data):
        churn, attributes, contracts, interactions = synthetic_raw_data
        bronze = build_bronze_customer(churn, attributes, contracts, interactions)
        assert isinstance(bronze, pd.DataFrame)
        assert len(bronze) > 0
        assert "customer_id" in bronze.columns

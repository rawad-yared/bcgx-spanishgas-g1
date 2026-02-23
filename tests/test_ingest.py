"""Tests for src.data.ingest — raw loading and bronze table construction."""

import numpy as np
import pandas as pd
import pytest

from src.data.ingest import (
    build_bronze_customer,
    build_bronze_customer_month,
)


@pytest.fixture
def sample_churn():
    return pd.DataFrame({
        "customer_id": ["C001", "C002", "C003"],
        "churn": [0, 1, 0],
    })


@pytest.fixture
def sample_attributes():
    return pd.DataFrame({
        "customer_id": ["C001", "C002", "C003"],
        "province": ["Madrid", "Barcelona", "Valencia"],
        "is_industrial": [0, 0, 1],
        "contracted_power_kw": [3.5, 5.0, 15.0],
    })


@pytest.fixture
def sample_contracts():
    return pd.DataFrame({
        "customer_id": ["C001", "C002", "C003"],
        "contract_start_date": ["2023-01-01", "2023-06-01", "2022-01-01"],
        "next_renewal_date": ["2025-01-01", "2025-06-01", "2024-01-01"],
    })


@pytest.fixture
def sample_interactions():
    return pd.DataFrame({
        "customer_id": ["C001", "C002"],
        "interaction_type": ["billing", "complaint"],
        "sentiment_score": [-0.2, -0.8],
    })


@pytest.fixture
def sample_consumption():
    rng = pd.date_range("2024-01-01", periods=48, freq="h")
    return pd.DataFrame({
        "customer_id": ["C001"] * 48,
        "timestamp": rng,
        "consumption_elec_kwh": np.random.uniform(0, 5, 48),
        "consumption_gas_m3": np.random.uniform(0, 1, 48),
    })


class TestBuildBronzeCustomer:
    def test_one_row_per_customer(self, sample_churn, sample_attributes, sample_contracts, sample_interactions):
        bronze = build_bronze_customer(sample_churn, sample_attributes, sample_contracts, sample_interactions)
        assert len(bronze) == 3
        assert bronze["customer_id"].nunique() == 3

    def test_left_join_preserves_all_churn(self, sample_churn, sample_attributes, sample_contracts, sample_interactions):
        bronze = build_bronze_customer(sample_churn, sample_attributes, sample_contracts, sample_interactions)
        assert set(bronze["customer_id"]) == {"C001", "C002", "C003"}

    def test_duplicate_churn_raises(self, sample_attributes, sample_contracts, sample_interactions):
        churn_dup = pd.DataFrame({"customer_id": ["C001", "C001"], "churn": [0, 1]})
        with pytest.raises(ValueError, match="duplicate"):
            build_bronze_customer(churn_dup, sample_attributes, sample_contracts, sample_interactions)


class TestBuildBronzeCustomerMonth:
    def test_monthly_aggregation(self, sample_consumption):
        result = build_bronze_customer_month(sample_consumption)
        assert "monthly_elec_kwh" in result.columns
        assert "monthly_gas_m3" in result.columns
        assert result["customer_id"].iloc[0] == "C001"

    def test_negative_consumption_clamped(self):
        df = pd.DataFrame({
            "customer_id": ["C001"] * 2,
            "timestamp": ["2024-01-01 10:00", "2024-01-01 11:00"],
            "consumption_elec_kwh": [-5.0, 3.0],
            "consumption_gas_m3": [1.0, -2.0],
        })
        result = build_bronze_customer_month(df)
        assert result["monthly_elec_kwh"].iloc[0] >= 0
        assert result["monthly_gas_m3"].iloc[0] >= 0

    def test_tariff_tiers_present(self, sample_consumption):
        result = build_bronze_customer_month(sample_consumption)
        tier_cols = [c for c in result.columns if "tier_" in c]
        assert len(tier_cols) > 0

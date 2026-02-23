"""Tests for src.features.build_features."""

from datetime import datetime

import pandas as pd
import pytest

from src.features.build_features import (
    build_gold_master,
    build_lifecycle_features,
    build_market_core_features,
    build_market_risk_features,
    build_sentiment_features,
)


@pytest.fixture
def silver_customer():
    return pd.DataFrame({
        "customer_id": ["C001", "C002", "C003"],
        "churn": [0, 1, 0],
        "is_industrial": [0, 0, 1],
        "contracted_power_kw": [3.5, 5.0, 15.0],
        "is_second_residence": [0, 1, 0],
        "sales_channel": ["Comparison Website", "Telemarketing", "Office"],
        "customer_first_activation_date": ["2022-01-01", "2023-06-01", "2021-01-01"],
        "next_renewal_date": ["2025-03-01", "2025-08-01", "2024-06-01"],
        "is_high_competition_province": [1, 0, 1],
        "has_interaction": [1, 1, 0],
        "customer_intent": ["Pricing Offers", "Cancel", None],
        "sentiment_label": ["Negative", "Negative", None],
        "sentiment_neg": [0.7, 0.9, 0.0],
        "sentiment_pos": [0.1, 0.0, 0.0],
        "segment": ["Residential", "Residential", "Corporate"],
    })


@pytest.fixture
def silver_customer_month():
    return pd.DataFrame({
        "customer_id": ["C001", "C001", "C002", "C002", "C003", "C003"],
        "month": ["2024-01", "2024-02", "2024-01", "2024-02", "2024-01", "2024-02"],
        "monthly_elec_kwh": [100, 120, 80, 90, 500, 600],
        "monthly_gas_m3": [10, 12, 0, 0, 20, 25],
        "total_margin": [8.0, 9.0, 5.0, 6.0, 50.0, 55.0],
        "elec_margin": [6.0, 7.0, 5.0, 6.0, 40.0, 45.0],
        "gas_margin": [2.0, 2.0, 0.0, 0.0, 10.0, 10.0],
        "total_revenue": [20.0, 22.0, 12.0, 14.0, 100.0, 110.0],
        "gas_revenue_variable": [5.0, 6.0, 0.0, 0.0, 10.0, 12.5],
        "variable_price_tier1_eur_kwh": [0.15, 0.16, 0.14, 0.14, 0.12, 0.12],
        "gas_variable_price_eur_m3": [0.50, 0.52, 0.0, 0.0, 0.45, 0.45],
    })


class TestLifecycleFeatures:
    def test_produces_one_row_per_customer(self, silver_customer, silver_customer_month):
        result = build_lifecycle_features(silver_customer, silver_customer_month, datetime(2025, 1, 1))
        assert result["customer_id"].nunique() == 3
        assert len(result) == 3

    def test_tenure_computed(self, silver_customer, silver_customer_month):
        result = build_lifecycle_features(silver_customer, silver_customer_month, datetime(2025, 1, 1))
        assert "tenure_months" in result.columns
        # C001 activated 2022-01-01, as_of 2025-01-01 → ~36 months
        c001 = result[result["customer_id"] == "C001"]["tenure_months"].iloc[0]
        assert 35 <= c001 <= 37

    def test_dual_fuel_flag(self, silver_customer, silver_customer_month):
        result = build_lifecycle_features(silver_customer, silver_customer_month, datetime(2025, 1, 1))
        assert "is_dual_fuel" in result.columns
        # C001 has gas, C002 has no gas
        assert result[result["customer_id"] == "C001"]["is_dual_fuel"].iloc[0] == 1
        assert result[result["customer_id"] == "C002"]["is_dual_fuel"].iloc[0] == 0


class TestMarketCoreFeatures:
    def test_avg_consumption(self, silver_customer_month, silver_customer):
        result = build_market_core_features(silver_customer_month, silver_customer)
        c001 = result[result["customer_id"] == "C001"]
        assert c001["avg_monthly_elec_kwh"].iloc[0] == pytest.approx(110.0)


class TestMarketRiskFeatures:
    def test_volatility(self, silver_customer_month):
        result = build_market_risk_features(silver_customer_month)
        assert "elec_consumption_volatility" in result.columns
        assert len(result) == 3


class TestSentimentFeatures:
    def test_negative_flag(self, silver_customer):
        result = build_sentiment_features(silver_customer)
        assert "has_negative_sentiment" in result.columns
        assert result[result["customer_id"] == "C001"]["has_negative_sentiment"].iloc[0] == 1


class TestCompoundFeatures:
    def test_cross_features(self, silver_customer, silver_customer_month):
        gold = build_gold_master(silver_customer, silver_customer_month, datetime(2025, 1, 1))
        assert "intent_x_renewal_bucket" in gold.columns or "renewal_x_complaint" in gold.columns


class TestGoldMaster:
    def test_grain_one_per_customer(self, silver_customer, silver_customer_month):
        gold = build_gold_master(silver_customer, silver_customer_month, datetime(2025, 1, 1))
        assert gold["customer_id"].nunique() == len(gold)

    def test_churn_label_present(self, silver_customer, silver_customer_month):
        gold = build_gold_master(silver_customer, silver_customer_month, datetime(2025, 1, 1))
        assert "churn" in gold.columns

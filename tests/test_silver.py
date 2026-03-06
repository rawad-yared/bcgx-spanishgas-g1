"""Tests for src.data.silver — silver transforms."""

import numpy as np
import pandas as pd
import pytest

from src.data.silver import (
    clean_sales_channels,
    compute_margins,
    derive_customer_segments,
    impute_prices_hierarchical,
)


@pytest.fixture
def silver_customer():
    return pd.DataFrame({
        "customer_id": ["C001", "C002", "C003", "C004"],
        "is_industrial": [0, 0, 1, 1],
        "contracted_power_kw": [3.5, 5.0, 10.0, 50.0],
        "is_second_residence": [0, 1, 0, 0],
        "sales_channel": ["comparador", "telemarketing", "presencial_comercial", "web_propia"],
    })


@pytest.fixture
def silver_customer_month():
    return pd.DataFrame({
        "customer_id": ["C001", "C001", "C002", "C002"],
        "month": ["2024-01", "2024-02", "2024-01", "2024-02"],
        "monthly_elec_kwh": [100.0, 120.0, 80.0, 90.0],
        "monthly_gas_m3": [10.0, 12.0, 0.0, 0.0],
        "elec_kwh_tier_1_peak": [40.0, 50.0, 30.0, 35.0],
        "elec_kwh_tier_2_standard": [35.0, 40.0, 25.0, 30.0],
        "elec_kwh_tier_3_offpeak": [25.0, 30.0, 25.0, 25.0],
        "variable_price_tier1_eur_kwh": [0.15, np.nan, 0.14, 0.14],
        "variable_price_tier2_eur_kwh": [0.12, 0.12, 0.11, np.nan],
        "variable_price_tier3_eur_kwh": [0.08, 0.08, 0.07, 0.07],
        "gas_variable_price_eur_m3": [0.50, 0.52, np.nan, np.nan],
    })


class TestDeriveCustomerSegments:
    def test_residential_segment(self, silver_customer):
        result = derive_customer_segments(silver_customer)
        assert result.loc[result["customer_id"] == "C001", "segment"].iloc[0] == "Residential"

    def test_sme_segment(self, silver_customer):
        result = derive_customer_segments(silver_customer)
        assert result.loc[result["customer_id"] == "C003", "segment"].iloc[0] == "SME"

    def test_corporate_segment(self, silver_customer):
        result = derive_customer_segments(silver_customer)
        assert result.loc[result["customer_id"] == "C004", "segment"].iloc[0] == "Corporate"

    def test_residential_type(self, silver_customer):
        result = derive_customer_segments(silver_customer)
        assert result.loc[result["customer_id"] == "C002", "residential_type"].iloc[0] == "Second_Residence"


class TestCleanSalesChannels:
    def test_spanish_to_english(self, silver_customer):
        result = clean_sales_channels(silver_customer)
        assert result.loc[result["customer_id"] == "C001", "sales_channel"].iloc[0] == "Comparison Website"
        assert result.loc[result["customer_id"] == "C004", "sales_channel"].iloc[0] == "Own Website"


class TestImputePricesHierarchical:
    def test_fills_missing_prices(self, silver_customer_month, silver_customer):
        result = impute_prices_hierarchical(silver_customer_month, silver_customer)
        # C001 had NaN in tier1 for Feb — should be forward-filled from Jan
        c001_feb = result[(result["customer_id"] == "C001") & (result["month"] == "2024-02")]
        assert not c001_feb["variable_price_tier1_eur_kwh"].isna().any()

    def test_no_data_loss(self, silver_customer_month, silver_customer):
        result = impute_prices_hierarchical(silver_customer_month, silver_customer)
        assert len(result) == len(silver_customer_month)


class TestComputeMargins:
    def test_margin_columns_created(self):
        df = pd.DataFrame({
            "customer_id": ["C001"],
            "monthly_elec_kwh": [100.0],
            "monthly_gas_m3": [10.0],
            "elec_kwh_tier_1_peak": [40.0],
            "elec_kwh_tier_2_standard": [35.0],
            "elec_kwh_tier_3_offpeak": [25.0],
            "variable_price_tier1_eur_kwh": [0.15],
            "variable_price_tier2_eur_kwh": [0.12],
            "variable_price_tier3_eur_kwh": [0.08],
            "elec_fixed_fee_eur_month": [5.0],
            "elec_var_cost_eur_kwh": [0.05],
            "peaje_elec_eur_kwh": [0.02],
            "elec_fixed_cost_eur_month": [2.0],
            "gas_variable_price_eur_m3": [0.50],
            "gas_fixed_revenue_eur_year": [24.0],
            "gas_var_cost_eur_m3": [0.30],
            "gas_fixed_cost_eur_year": [12.0],
        })
        result = compute_margins(df)
        assert "elec_margin" in result.columns
        assert "gas_margin" in result.columns
        assert "total_margin" in result.columns
        assert result["total_margin"].iloc[0] == pytest.approx(
            result["total_revenue"].iloc[0] - result["total_cost"].iloc[0]
        )

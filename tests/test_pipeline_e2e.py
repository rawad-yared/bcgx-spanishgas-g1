"""End-to-end integration test for the local pipeline flow.

Uses minimal synthetic data to test bronze -> silver -> gold -> train -> score chain.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from src.data.ingest import build_bronze_customer, build_bronze_customer_month
from src.data.silver import build_silver_tables
from src.features.build_features import build_gold_master
from src.models.preprocessing import build_preprocessing_pipeline
from src.models.scorer import assign_risk_tiers, score_all_customers

# ---------------------------------------------------------------------------
# Shared fixtures — realistic synthetic data matching actual column schemas
# ---------------------------------------------------------------------------

AS_OF = datetime(2025, 1, 1)
N_CUSTOMERS = 50


@pytest.fixture
def synthetic_churn():
    """Churn labels (1 row per customer)."""
    np.random.seed(42)
    return pd.DataFrame({
        "customer_id": [f"C{i:04d}" for i in range(N_CUSTOMERS)],
        "churn": np.random.choice([0, 1], N_CUSTOMERS, p=[0.85, 0.15]),
    })


@pytest.fixture
def synthetic_attributes():
    """Customer attributes aligned with bronze_customer merge."""
    np.random.seed(42)
    return pd.DataFrame({
        "customer_id": [f"C{i:04d}" for i in range(N_CUSTOMERS)],
        "province": np.random.choice(["Madrid", "Barcelona", "Valencia", "Sevilla"], N_CUSTOMERS),
        "is_industrial": np.random.choice([0, 0, 0, 1], N_CUSTOMERS),
        "contracted_power_kw": np.random.choice([3.5, 5.0, 10.0, 50.0], N_CUSTOMERS),
        "is_second_residence": np.random.choice([0, 1], N_CUSTOMERS, p=[0.8, 0.2]),
        "subscribed_power": np.random.uniform(3, 15, N_CUSTOMERS),
        "sales_channel": np.random.choice(
            ["comparador", "telemarketing", "presencial_comercial", "web_propia"],
            N_CUSTOMERS,
        ),
        "customer_first_activation_date": pd.to_datetime(
            np.random.choice(pd.date_range("2020-01-01", "2023-12-01", freq="MS"), N_CUSTOMERS)
        ),
        "next_renewal_date": pd.to_datetime(
            np.random.choice(pd.date_range("2024-06-01", "2026-01-01", freq="MS"), N_CUSTOMERS)
        ),
        "is_high_competition_province": np.random.choice([0, 1], N_CUSTOMERS),
        "has_interaction": np.random.choice([0, 1], N_CUSTOMERS, p=[0.4, 0.6]),
        "customer_intent": np.random.choice(
            ["Pricing Offers", "Cancel", "General Inquiry", None], N_CUSTOMERS
        ),
        "sentiment_label": np.random.choice(["Negative", "Neutral", "Positive", None], N_CUSTOMERS),
        "sentiment_neg": np.random.uniform(0, 1, N_CUSTOMERS).round(2),
        "sentiment_pos": np.random.uniform(0, 1, N_CUSTOMERS).round(2),
        "sentiment_neu": np.random.uniform(0, 1, N_CUSTOMERS).round(2),
        "segment": None,  # will be derived in silver
    })


@pytest.fixture
def synthetic_contracts():
    """Contract data (1 row per customer)."""
    np.random.seed(42)
    return pd.DataFrame({
        "customer_id": [f"C{i:04d}" for i in range(N_CUSTOMERS)],
        "contract_start_date": pd.to_datetime("2020-01-01"),
        "contract_end_date": pd.to_datetime("2025-01-01"),
        "has_gas": np.random.choice([0, 1], N_CUSTOMERS),
        "channel": np.random.choice(["online", "branch", "phone"], N_CUSTOMERS),
    })


@pytest.fixture
def synthetic_interactions():
    """Interaction data (some customers have multiple, some have none)."""
    np.random.seed(42)
    ids = np.random.choice([f"C{i:04d}" for i in range(N_CUSTOMERS)], N_CUSTOMERS * 2)
    return pd.DataFrame({
        "customer_id": ids,
        "interaction_type": np.random.choice(["complaint", "inquiry", "payment"], len(ids)),
        "interaction_date": pd.date_range("2023-01-01", periods=len(ids), freq="D")[: len(ids)],
    })


@pytest.fixture
def synthetic_consumption():
    """Hourly consumption for a subset of customers (3 months of data)."""
    np.random.seed(42)
    hours = pd.date_range("2024-01-01", "2024-03-31 23:00", freq="h")
    # Only use first 10 customers for speed
    cust_ids = [f"C{i:04d}" for i in range(10)]
    rows = []
    for cid in cust_ids:
        for ts in hours:
            rows.append({
                "customer_id": cid,
                "timestamp": ts,
                "consumption_elec_kwh": max(0, np.random.normal(2.0, 0.8)),
                "consumption_gas_m3": max(0, np.random.normal(0.3, 0.15)),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Bronze fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def bronze_customer(synthetic_churn, synthetic_attributes, synthetic_contracts, synthetic_interactions):
    """Bronze customer table (1 row per customer)."""
    return build_bronze_customer(synthetic_churn, synthetic_attributes, synthetic_contracts, synthetic_interactions)


@pytest.fixture
def bronze_customer_month(synthetic_consumption):
    """Bronze customer-month table (aggregated from hourly consumption)."""
    return build_bronze_customer_month(synthetic_consumption)


# ---------------------------------------------------------------------------
# Silver fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def silver_tables(bronze_customer, bronze_customer_month):
    """Silver customer and silver customer-month tables."""
    return build_silver_tables(bronze_customer, bronze_customer_month)


@pytest.fixture
def silver_customer(silver_tables):
    return silver_tables[0]


@pytest.fixture
def silver_customer_month(silver_tables):
    return silver_tables[1]


# ---------------------------------------------------------------------------
# Gold fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def gold_master(silver_customer, silver_customer_month):
    """Gold master table (1 row per customer with all features)."""
    return build_gold_master(silver_customer, silver_customer_month, as_of_date=AS_OF)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestPipelineE2E:
    """Original bronze smoke test."""

    def test_bronze_produces_dataframe(self, bronze_customer):
        assert isinstance(bronze_customer, pd.DataFrame)
        assert len(bronze_customer) > 0
        assert "customer_id" in bronze_customer.columns


@pytest.mark.slow
class TestSilverTransformE2E:
    """Silver transform integration: bronze -> silver with cleaning, imputation, margins."""

    def test_silver_customer_has_segments(self, silver_customer):
        """derive_customer_segments should populate the segment column."""
        assert "segment" in silver_customer.columns
        # At least Residential and one industrial type should appear
        segments = set(silver_customer["segment"].dropna().unique())
        assert "Residential" in segments

    def test_silver_customer_channels_cleaned(self, silver_customer):
        """Spanish channel names should be translated to English."""
        assert "sales_channel" in silver_customer.columns
        channels = silver_customer["sales_channel"].dropna().unique()
        # None of the raw Spanish names should remain
        for ch in channels:
            assert ch not in ("comparador", "telemarketing", "presencial_comercial", "web_propia")

    def test_silver_customer_month_has_margins(self, silver_customer_month):
        """compute_margins should produce margin columns."""
        for col in ["elec_margin", "gas_margin", "total_margin", "total_revenue", "total_cost"]:
            assert col in silver_customer_month.columns, f"Missing column: {col}"

    def test_silver_customer_month_row_count_preserved(self, bronze_customer_month, silver_customer_month):
        """Silver transforms should not drop rows from the customer-month table."""
        assert len(silver_customer_month) == len(bronze_customer_month)

    def test_silver_customer_preserves_customer_count(self, bronze_customer, silver_customer):
        """Silver transforms should not drop customers."""
        assert silver_customer["customer_id"].nunique() == bronze_customer["customer_id"].nunique()


@pytest.mark.slow
class TestGoldFeaturesE2E:
    """Gold feature engineering integration: silver -> gold master."""

    def test_gold_one_row_per_customer(self, gold_master):
        """Gold master grain must be 1 row per customer."""
        assert gold_master["customer_id"].nunique() == len(gold_master)

    def test_gold_has_lifecycle_features(self, gold_master):
        """Tier 1A lifecycle features should be present."""
        expected = ["tenure_months", "is_dual_fuel"]
        for col in expected:
            assert col in gold_master.columns, f"Missing lifecycle feature: {col}"

    def test_gold_has_market_core_features(self, gold_master):
        """Tier MP_Core market features should be present."""
        expected = ["avg_monthly_elec_kwh", "avg_monthly_gas_m3"]
        for col in expected:
            assert col in gold_master.columns, f"Missing market core feature: {col}"

    def test_gold_has_market_risk_features(self, gold_master):
        """Tier MP_Risk volatility/trend features should be present."""
        expected = ["elec_consumption_volatility", "gas_consumption_volatility"]
        for col in expected:
            assert col in gold_master.columns, f"Missing market risk feature: {col}"

    def test_gold_has_churn_label(self, gold_master):
        """Churn label must be carried through to gold master."""
        assert "churn" in gold_master.columns
        # Should have both 0s and 1s (given seed and 15% churn rate)
        assert set(gold_master["churn"].dropna().unique()) == {0, 1}

    def test_gold_customer_count_matches_bronze(self, bronze_customer, gold_master):
        """Gold master should have features for all customers present in bronze.

        Note: gold only includes customers that appear in silver_customer, which
        is derived from bronze_customer. Customers without consumption data
        still appear via the left join in build_gold_master.
        """
        assert gold_master["customer_id"].nunique() == bronze_customer["customer_id"].nunique()


@pytest.mark.slow
class TestScoreE2E:
    """Scoring integration: gold master + trained model -> scored output with risk tiers."""

    @pytest.fixture
    def trained_pipeline_and_features(self, gold_master):
        """Train a simple model on gold master features for scoring tests.

        Uses RandomForest for a realistic but fast pipeline. Only uses numeric
        features to keep the fixture simple and deterministic.
        """
        feature_cols = [
            c for c in gold_master.columns
            if c not in ("customer_id", "churn") and gold_master[c].dtype in ("float64", "int64", "Int64")
        ]
        # Need at least a few features
        assert len(feature_cols) >= 2, f"Not enough numeric features: {feature_cols}"

        X = gold_master[feature_cols].copy()
        y = gold_master["churn"].copy()

        # Fill NaN for training (simple median fill)
        X = X.fillna(X.median())

        preprocessor = build_preprocessing_pipeline(X, scale_numeric=False, ohe_sparse=False)
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", DummyClassifier(strategy="stratified", random_state=42)),
        ])
        pipeline.fit(X, y)

        return pipeline, feature_cols

    def test_score_all_produces_expected_columns(self, gold_master, trained_pipeline_and_features):
        """score_all_customers should produce churn_proba, churn_pred columns."""
        pipeline, feature_cols = trained_pipeline_and_features

        # Fill NaN in gold for scoring (matches training)
        gold_filled = gold_master.copy()
        for col in feature_cols:
            if col in gold_filled.columns:
                gold_filled[col] = gold_filled[col].fillna(gold_filled[col].median())

        scored = score_all_customers(pipeline, gold_filled, feature_cols, threshold=0.5)

        assert "churn_proba" in scored.columns
        assert "churn_pred" in scored.columns
        assert "customer_id" in scored.columns

    def test_score_probabilities_valid_range(self, gold_master, trained_pipeline_and_features):
        """Churn probabilities must be between 0 and 1."""
        pipeline, feature_cols = trained_pipeline_and_features

        gold_filled = gold_master.copy()
        for col in feature_cols:
            if col in gold_filled.columns:
                gold_filled[col] = gold_filled[col].fillna(gold_filled[col].median())

        scored = score_all_customers(pipeline, gold_filled, feature_cols, threshold=0.5)

        assert scored["churn_proba"].min() >= 0.0
        assert scored["churn_proba"].max() <= 1.0

    def test_risk_tiers_assigned(self, gold_master, trained_pipeline_and_features):
        """assign_risk_tiers should add a risk_tier column with valid labels."""
        pipeline, feature_cols = trained_pipeline_and_features

        gold_filled = gold_master.copy()
        for col in feature_cols:
            if col in gold_filled.columns:
                gold_filled[col] = gold_filled[col].fillna(gold_filled[col].median())

        scored = score_all_customers(pipeline, gold_filled, feature_cols, threshold=0.5)
        scored = assign_risk_tiers(scored)

        assert "risk_tier" in scored.columns
        valid_tiers = {"Low (<40%)", "Medium (40-60%)", "High (60-80%)", "Critical (>80%)"}
        actual_tiers = set(scored["risk_tier"].dropna().unique())
        assert actual_tiers.issubset(valid_tiers)

    def test_scored_row_count_matches_gold(self, gold_master, trained_pipeline_and_features):
        """Every customer in gold should get a score."""
        pipeline, feature_cols = trained_pipeline_and_features

        gold_filled = gold_master.copy()
        for col in feature_cols:
            if col in gold_filled.columns:
                gold_filled[col] = gold_filled[col].fillna(gold_filled[col].median())

        scored = score_all_customers(pipeline, gold_filled, feature_cols, threshold=0.5)

        assert len(scored) == len(gold_master)


@pytest.mark.slow
class TestFullPipelineE2E:
    """Full pipeline chain: bronze -> silver -> gold -> score (pre-train).

    Validates data integrity is preserved throughout the pipeline and that
    each stage produces valid output for the next stage.
    """

    def test_bronze_to_gold_data_integrity(
        self, bronze_customer, bronze_customer_month, silver_customer, silver_customer_month, gold_master
    ):
        """Data shape and customer IDs should be consistent across all layers."""
        # Bronze customer count = silver customer count = gold customer count
        bronze_count = bronze_customer["customer_id"].nunique()
        silver_count = silver_customer["customer_id"].nunique()
        gold_count = gold_master["customer_id"].nunique()

        assert bronze_count == silver_count, "Customer count changed bronze -> silver"
        assert bronze_count == gold_count, "Customer count changed bronze/silver -> gold"

        # Bronze customer-month row count = silver customer-month row count
        assert len(bronze_customer_month) == len(silver_customer_month), (
            "Row count changed in customer-month table"
        )

    def test_no_duplicate_customer_ids_at_any_stage(
        self, bronze_customer, silver_customer, gold_master
    ):
        """Each stage should maintain 1-row-per-customer grain."""
        assert not bronze_customer.duplicated(subset=["customer_id"]).any(), "Dupes in bronze"
        assert not silver_customer.duplicated(subset=["customer_id"]).any(), "Dupes in silver"
        assert not gold_master.duplicated(subset=["customer_id"]).any(), "Dupes in gold"

    def test_churn_label_preserved_through_pipeline(
        self, synthetic_churn, bronze_customer, silver_customer, gold_master
    ):
        """Churn label should be carried from raw -> bronze -> silver -> gold without corruption."""
        # Check gold has churn
        assert "churn" in gold_master.columns

        # Verify churn distribution matches original
        original_churn_rate = synthetic_churn["churn"].mean()
        gold_churn_rate = gold_master["churn"].mean()
        assert abs(original_churn_rate - gold_churn_rate) < 0.01, (
            f"Churn rate drift: original={original_churn_rate:.3f}, gold={gold_churn_rate:.3f}"
        )

    def test_silver_enriches_bronze(self, bronze_customer, silver_customer):
        """Silver should add columns beyond what bronze has (segments, cleaned channels)."""
        silver_only_cols = set(silver_customer.columns) - set(bronze_customer.columns)
        # derive_customer_segments adds 'segment' and 'residential_type'
        assert "residential_type" in silver_only_cols or "segment" in silver_customer.columns

    def test_gold_has_more_features_than_silver(self, silver_customer, gold_master):
        """Gold master should have substantially more columns than silver customer."""
        assert len(gold_master.columns) > len(silver_customer.columns), (
            f"Gold ({len(gold_master.columns)} cols) should have more features than "
            f"silver ({len(silver_customer.columns)} cols)"
        )

    def test_full_chain_produces_scoreable_output(
        self, gold_master
    ):
        """Gold master should be scoreable — it must have numeric features and a target."""
        assert "churn" in gold_master.columns

        numeric_features = gold_master.select_dtypes(include=[np.number]).columns.tolist()
        # Remove target from feature count
        numeric_features = [c for c in numeric_features if c != "churn"]
        assert len(numeric_features) >= 5, (
            f"Gold master should have at least 5 numeric features for modeling, "
            f"got {len(numeric_features)}"
        )

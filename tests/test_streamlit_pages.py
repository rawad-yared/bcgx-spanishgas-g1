"""Tests for Streamlit UI page render functions."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd

# ---------------------------------------------------------------------------
# Model Performance page
# ---------------------------------------------------------------------------


class TestModelPerformancePage:
    @patch("src.serving.ui.pages.model_performance.load_model_metrics")
    @patch("src.serving.ui.pages.model_performance.st")
    def test_no_data(self, mock_st, mock_loader):
        """Empty metrics dict triggers a warning and early return."""
        mock_loader.return_value = {}

        from src.serving.ui.pages.model_performance import render

        render()

        mock_st.header.assert_called_once_with("Model Performance")
        mock_st.warning.assert_called_once()

    @patch("src.serving.ui.pages.model_performance.load_model_metrics")
    @patch("src.serving.ui.pages.model_performance.st")
    def test_with_data(self, mock_st, mock_loader):
        """With valid metrics, st.columns and col.metric are called with key values."""
        mock_loader.return_value = {
            "pr_auc": 0.82,
            "roc_auc": 0.90,
            "precision": 0.75,
            "recall": 0.70,
            "tp": 100,
            "fp": 20,
            "fn": 30,
            "tn": 850,
        }

        col1, col2, col3, col4 = MagicMock(), MagicMock(), MagicMock(), MagicMock()
        mock_st.columns.return_value = [col1, col2, col3, col4]

        from src.serving.ui.pages.model_performance import render

        render()

        mock_st.header.assert_called_once_with("Model Performance")
        mock_st.warning.assert_not_called()

        # Verify all four metric cards
        col1.metric.assert_called_once_with("PR-AUC", "0.8200")
        col2.metric.assert_called_once_with("ROC-AUC", "0.9000")
        col3.metric.assert_called_once_with("Precision", "0.7500")
        col4.metric.assert_called_once_with("Recall", "0.7000")

        # Confusion matrix should be rendered (tp+fp+fn+tn > 0)
        mock_st.subheader.assert_any_call("Confusion Matrix")
        mock_st.plotly_chart.assert_called_once()

        # All metrics JSON dump
        mock_st.subheader.assert_any_call("All Metrics")
        mock_st.json.assert_called_once()


# ---------------------------------------------------------------------------
# Drift Monitor page
# ---------------------------------------------------------------------------


class TestDriftMonitorPage:
    @patch("src.serving.ui.pages.drift_monitor.load_drift_results")
    @patch("src.serving.ui.pages.drift_monitor.st")
    def test_no_data(self, mock_st, mock_loader):
        """Empty drift dict triggers an info message and early return."""
        mock_loader.return_value = {}

        from src.serving.ui.pages.drift_monitor import render

        render()

        mock_st.header.assert_called_once_with("Drift Monitor")
        mock_st.info.assert_called_once()

    @patch("src.serving.ui.pages.drift_monitor.load_drift_results")
    @patch("src.serving.ui.pages.drift_monitor.st")
    def test_with_data(self, mock_st, mock_loader):
        """With drift results, metrics and feature table are rendered."""
        mock_loader.return_value = {
            "any_drift": True,
            "n_features_drifted": 2,
            "summary": "2 of 5 features drifted",
            "feature_drift": {
                "feature_results": [
                    {
                        "feature": "cons_12m",
                        "ks_statistic": 0.15,
                        "p_value": 0.001,
                        "drifted": True,
                    },
                    {
                        "feature": "tenure_months",
                        "ks_statistic": 0.04,
                        "p_value": 0.45,
                        "drifted": False,
                    },
                ],
            },
            "prediction_drift": {
                "ks_statistic": 0.08,
                "p_value": 0.02,
            },
        }

        col1, col2 = MagicMock(), MagicMock()
        mock_st.columns.return_value = [col1, col2]

        from src.serving.ui.pages.drift_monitor import render

        render()

        mock_st.header.assert_called_once_with("Drift Monitor")
        mock_st.info.assert_not_called()

        # Top-level drift metrics
        col1.metric.assert_called_once_with("Drift Detected", "Yes")
        col2.metric.assert_called_once_with("Features Drifted", 2)

        # Summary text
        mock_st.text.assert_called_once_with("2 of 5 features drifted")

        # Feature drift details table and chart
        mock_st.subheader.assert_any_call("Feature Drift Details")
        mock_st.dataframe.assert_called_once()
        mock_st.plotly_chart.assert_called_once()

        # Prediction drift metrics
        mock_st.subheader.assert_any_call("Prediction Drift")
        # Two st.metric calls for prediction drift (ks_statistic + p-value)
        assert mock_st.metric.call_count == 2


# ---------------------------------------------------------------------------
# Customer Risk page
# ---------------------------------------------------------------------------


class TestCustomerRiskPage:
    @patch("src.serving.ui.pages.customer_risk.load_scored_data")
    @patch("src.serving.ui.pages.customer_risk.st")
    def test_no_data(self, mock_st, mock_loader):
        """Empty DataFrame triggers a warning and early return."""
        mock_loader.return_value = pd.DataFrame()

        from src.serving.ui.pages.customer_risk import render

        render()

        mock_st.header.assert_called_once_with("Customer Risk Overview")
        mock_st.warning.assert_called_once()

    @patch("src.serving.ui.pages.customer_risk.load_scored_data")
    @patch("src.serving.ui.pages.customer_risk.st")
    def test_with_data(self, mock_st, mock_loader):
        """With scored data, metrics, charts, and customer table are rendered."""
        mock_loader.return_value = pd.DataFrame(
            {
                "customer_id": ["C001", "C002", "C003", "C004"],
                "churn_proba": [0.92, 0.75, 0.40, 0.10],
                "risk_tier": [
                    "Critical (>80%)",
                    "High (60-80%)",
                    "Medium (40-60%)",
                    "Low (<40%)",
                ],
                "segment": ["SME", "SME", "Residential", "Residential"],
                "expected_monthly_loss": [500.0, 300.0, 100.0, 20.0],
                "churn_pred": [1, 1, 0, 0],
            }
        )

        # st.columns(4) for key metrics row
        col1, col2, col3, col4 = MagicMock(), MagicMock(), MagicMock(), MagicMock()
        # st.columns(2) for risk tier distribution row
        col_a, col_b = MagicMock(), MagicMock()
        # st.columns(3) for filter row
        filter_col1, filter_col2, filter_col3 = MagicMock(), MagicMock(), MagicMock()

        mock_st.columns.side_effect = [
            [col1, col2, col3, col4],  # key metrics
            [col_a, col_b],            # risk tier charts
            [filter_col1, filter_col2, filter_col3],  # filters
        ]

        # selectbox returns "All" (no filtering)
        mock_st.selectbox.return_value = "All"
        # slider returns 50 (default)
        mock_st.slider.return_value = 50

        from src.serving.ui.pages.customer_risk import render

        render()

        mock_st.header.assert_called_once_with("Customer Risk Overview")
        mock_st.warning.assert_not_called()

        # Key metric cards
        col1.metric.assert_called_once_with("Total Customers", 4)
        col2.metric.assert_called_once_with("Critical Risk", 1)
        col3.metric.assert_called_once_with("High Risk", 1)
        col4.metric.assert_called_once()  # Expected Monthly Loss

        # Customer details table rendered
        mock_st.subheader.assert_any_call("Customer Details")


# ---------------------------------------------------------------------------
# Pipeline Status page
# ---------------------------------------------------------------------------


class TestPipelineStatusPage:
    @patch("src.serving.ui.pages.pipeline_status.load_pipeline_runs")
    @patch("src.serving.ui.pages.pipeline_status.st")
    def test_no_data(self, mock_st, mock_loader):
        """Empty runs list triggers an info message and early return."""
        mock_loader.return_value = []

        from src.serving.ui.pages.pipeline_status import render

        render()

        mock_st.header.assert_called_once_with("Pipeline Status")
        mock_st.info.assert_called_once()

    @patch("src.serving.ui.pages.pipeline_status.load_pipeline_runs")
    @patch("src.serving.ui.pages.pipeline_status.st")
    def test_with_data(self, mock_st, mock_loader):
        """With run data, metrics and run history table are rendered."""
        mock_loader.return_value = [
            {
                "run_id": "r-001",
                "file_key": "raw/data_2025_01.csv",
                "status": "completed",
                "started_at": "2025-01-15T10:00:00Z",
                "completed_at": "2025-01-15T10:15:00Z",
            },
            {
                "run_id": "r-002",
                "file_key": "raw/data_2025_02.csv",
                "status": "started",
                "started_at": "2025-02-01T08:00:00Z",
            },
            {
                "run_id": "r-003",
                "file_key": "raw/data_2025_03.csv",
                "status": "failed",
                "started_at": "2025-03-01T09:00:00Z",
                "completed_at": "2025-03-01T09:02:00Z",
            },
        ]

        # st.columns(4) for key metrics
        col1, col2, col3, col4 = MagicMock(), MagicMock(), MagicMock(), MagicMock()
        # st.columns([1, 3]) for filter row
        filter_col, spacer = MagicMock(), MagicMock()

        mock_st.columns.side_effect = [
            [col1, col2, col3, col4],   # key metrics
            [filter_col, spacer],       # filter row
        ]

        mock_st.selectbox.return_value = "All"

        from src.serving.ui.pages.pipeline_status import render

        render()

        mock_st.header.assert_called_once_with("Pipeline Status")
        mock_st.info.assert_not_called()

        # Key metric cards
        col1.metric.assert_called_once_with("Total Runs", 3)
        col2.metric.assert_called_once_with("Completed", 1)
        col3.metric.assert_called_once_with("In Progress", 1)
        col4.metric.assert_called_once_with("Failed", 1)

        # Run history table
        mock_st.subheader.assert_any_call("Run History")
        mock_st.dataframe.assert_called_once()


# ---------------------------------------------------------------------------
# Overview page
# ---------------------------------------------------------------------------


class TestOverviewPage:
    @patch("src.serving.ui.pages.overview.load_pipeline_runs")
    @patch("src.serving.ui.pages.overview.load_model_metrics")
    @patch("src.serving.ui.pages.overview.load_scored_data")
    @patch("src.serving.ui.pages.overview.st")
    def test_no_data(self, mock_st, mock_scored, mock_metrics, mock_runs):
        """Empty data results in em-dash placeholders."""
        mock_scored.return_value = pd.DataFrame()
        mock_metrics.return_value = {}
        mock_runs.return_value = []

        col = MagicMock()
        mock_st.columns.return_value = [col, col, col, col]

        from src.serving.ui.pages.overview import render

        render()

        mock_st.header.assert_called_once_with(
            "SpanishGas Churn Intelligence \u2014 Overview"
        )

    @patch("src.serving.ui.pages.overview.load_pipeline_runs")
    @patch("src.serving.ui.pages.overview.load_model_metrics")
    @patch("src.serving.ui.pages.overview.load_scored_data")
    @patch("src.serving.ui.pages.overview.st")
    def test_with_data(self, mock_st, mock_scored, mock_metrics, mock_runs):
        """With valid data, KPI metrics are rendered."""
        mock_scored.return_value = pd.DataFrame(
            {
                "customer_id": ["C001", "C002", "C003"],
                "risk_tier": [
                    "Critical (>80%)",
                    "High (60-80%)",
                    "Low (<40%)",
                ],
                "expected_monthly_loss": [500.0, 200.0, 10.0],
            }
        )
        mock_metrics.return_value = {
            "metrics": {"pr_auc": 0.75, "roc_auc": 0.88}
        }
        mock_runs.return_value = [
            {"run_id": "r1", "status": "completed"},
            {"run_id": "r2", "status": "started"},
        ]

        c1, c2, c3, c4 = MagicMock(), MagicMock(), MagicMock(), MagicMock()
        mock_st.columns.return_value = [c1, c2, c3, c4]

        from src.serving.ui.pages.overview import render

        render()

        mock_st.header.assert_called_once_with(
            "SpanishGas Churn Intelligence \u2014 Overview"
        )


# ---------------------------------------------------------------------------
# Recommendations page
# ---------------------------------------------------------------------------


class TestRecommendationsPage:
    @patch("src.serving.ui.pages.recommendations.load_recommendations")
    @patch("src.serving.ui.pages.recommendations.st")
    def test_no_data(self, mock_st, mock_loader):
        """Empty DataFrame triggers a warning and early return."""
        mock_loader.return_value = pd.DataFrame()

        from src.serving.ui.pages.recommendations import render

        render()

        mock_st.header.assert_called_once_with("Retention Recommendations")
        mock_st.warning.assert_called_once()

    @patch("src.serving.ui.pages.recommendations.load_recommendations")
    @patch("src.serving.ui.pages.recommendations.st")
    def test_with_data(self, mock_st, mock_loader):
        """With recommendations data, metrics, chart, and table are rendered."""
        mock_loader.return_value = pd.DataFrame(
            {
                "customer_id": ["C001", "C002", "C003", "C004"],
                "risk_tier": [
                    "Critical (>80%)",
                    "High (60-80%)",
                    "Medium (40-60%)",
                    "Low (<40%)",
                ],
                "risk_score": [0.92, 0.75, 0.45, 0.15],
                "segment": ["SME", "SME", "Residential", "Residential"],
                "action": [
                    "offer_large",
                    "offer_medium",
                    "offer_small",
                    "no_offer",
                ],
                "timing_window": [
                    "immediate",
                    "immediate",
                    "30_60_days",
                    "60_90_days",
                ],
                "expected_margin_impact": [500.0, 300.0, 100.0, 20.0],
                "reason_codes": [
                    ["critical_churn_risk"],
                    ["high_churn_risk"],
                    ["moderate_churn_risk"],
                    ["low_risk_monitoring"],
                ],
            }
        )

        col1, col2, col3, col4 = MagicMock(), MagicMock(), MagicMock(), MagicMock()
        filter_col1, filter_col2, filter_col3 = MagicMock(), MagicMock(), MagicMock()
        mock_st.columns.side_effect = [
            [col1, col2, col3, col4],                  # summary metrics
            [filter_col1, filter_col2, filter_col3],   # filters
        ]
        mock_st.selectbox.return_value = "All"
        mock_st.slider.return_value = 50

        from src.serving.ui.pages.recommendations import render

        render()

        mock_st.header.assert_called_once_with("Retention Recommendations")
        mock_st.warning.assert_not_called()

        # Summary metric
        col1.metric.assert_called_once_with("Total Recommendations", 4)
        col2.metric.assert_called_once_with("Distinct Actions", 4)

        # Bar chart
        mock_st.subheader.assert_any_call("Recommendations by Action Type")
        mock_st.plotly_chart.assert_called_once()

        # Filterable table
        mock_st.subheader.assert_any_call("Recommendation Details")
        mock_st.dataframe.assert_called_once()


# ---------------------------------------------------------------------------
# Customer Lookup page
# ---------------------------------------------------------------------------


class TestCustomerLookupPage:
    @patch("src.serving.ui.pages.customer_lookup.load_recommendations")
    @patch("src.serving.ui.pages.customer_lookup.load_scored_data")
    @patch("src.serving.ui.pages.customer_lookup.st")
    def test_no_data(self, mock_st, mock_scored, mock_reco):
        """Empty scored DataFrame triggers a warning and early return."""
        mock_scored.return_value = pd.DataFrame()
        mock_reco.return_value = pd.DataFrame()

        from src.serving.ui.pages.customer_lookup import render

        render()

        mock_st.header.assert_called_once_with("Customer Lookup")
        mock_st.warning.assert_called_once()

    @patch("src.serving.ui.pages.customer_lookup.load_recommendations")
    @patch("src.serving.ui.pages.customer_lookup.load_scored_data")
    @patch("src.serving.ui.pages.customer_lookup.st")
    def test_customer_found(self, mock_st, mock_scored, mock_reco):
        """Valid customer ID renders risk assessment and financial impact."""
        mock_scored.return_value = pd.DataFrame(
            {
                "customer_id": ["C001", "C002"],
                "churn_proba": [0.85, 0.20],
                "risk_tier": ["Critical (>80%)", "Low (<40%)"],
                "avg_monthly_margin": [100.0, 50.0],
                "segment": ["SME", "Residential"],
            }
        )
        mock_reco.return_value = pd.DataFrame(
            {
                "customer_id": ["C001", "C002"],
                "action": ["offer_large", "no_offer"],
                "timing_window": ["immediate", "60_90_days"],
                "reason_codes": [["critical_churn_risk"], ["low_risk_monitoring"]],
                "expected_margin_impact": [500.0, 10.0],
            }
        )

        # text_input returns valid customer ID
        mock_st.text_input.return_value = "C001"

        c1, c2, c3, c4 = MagicMock(), MagicMock(), MagicMock(), MagicMock()
        left, right = MagicMock(), MagicMock()
        mock_st.columns.side_effect = [
            [c1, c2, c3, c4],  # risk assessment
            [left, right],     # financial impact
        ]

        # Expander as context manager
        expander = MagicMock()
        mock_st.expander.return_value.__enter__ = MagicMock(return_value=expander)
        mock_st.expander.return_value.__exit__ = MagicMock(return_value=False)

        from src.serving.ui.pages.customer_lookup import render

        render()

        mock_st.header.assert_called_once_with("Customer Lookup")
        mock_st.error.assert_not_called()
        mock_st.warning.assert_not_called()

        # Risk assessment metrics rendered
        mock_st.subheader.assert_any_call("Risk Assessment")
        c2.metric.assert_called_once_with("Risk Tier", "Critical (>80%)")
        c3.metric.assert_called_once_with("Recommended Action", "Large Retention Offer")

        # Financial impact rendered
        mock_st.subheader.assert_any_call("Financial Impact")

    @patch("src.serving.ui.pages.customer_lookup.load_recommendations")
    @patch("src.serving.ui.pages.customer_lookup.load_scored_data")
    @patch("src.serving.ui.pages.customer_lookup.st")
    def test_customer_not_found(self, mock_st, mock_scored, mock_reco):
        """Invalid customer ID triggers error message."""
        mock_scored.return_value = pd.DataFrame(
            {
                "customer_id": ["C001"],
                "churn_proba": [0.5],
                "risk_tier": ["Medium (40-60%)"],
                "avg_monthly_margin": [80.0],
            }
        )
        mock_reco.return_value = pd.DataFrame()
        mock_st.text_input.return_value = "INVALID"

        from src.serving.ui.pages.customer_lookup import render

        render()

        mock_st.header.assert_called_once_with("Customer Lookup")
        mock_st.error.assert_called_once()


# ---------------------------------------------------------------------------
# Data Explorer page
# ---------------------------------------------------------------------------


class TestDataExplorerPage:
    @patch("src.serving.ui.pages.data_explorer.load_gold_data")
    @patch("src.serving.ui.pages.data_explorer.st")
    def test_no_data(self, mock_st, mock_loader):
        """Empty DataFrame triggers a warning and early return."""
        mock_loader.return_value = pd.DataFrame()

        from src.serving.ui.pages.data_explorer import render

        render()

        mock_st.header.assert_called_once_with("Data Explorer")
        mock_st.warning.assert_called_once()

    @patch("src.serving.ui.pages.data_explorer.load_gold_data")
    @patch("src.serving.ui.pages.data_explorer.st")
    def test_with_data(self, mock_st, mock_loader):
        """With gold master data, charts are rendered."""
        mock_loader.return_value = pd.DataFrame(
            {
                "customer_id": ["C001", "C002", "C003", "C004"],
                "segment": ["SME", "SME", "Residential", "Corporate"],
                "churn": [1, 0, 1, 0],
                "renewal_bucket": ["0-3m", "3-6m", "6-12m", "0-3m"],
                "sentiment_label": ["Negative", "Positive", None, "Neutral"],
                "avg_monthly_margin": [100.0, 50.0, 80.0, -20.0],
                "is_dual_fuel": [1, 0, 1, 0],
            }
        )

        col1, col2 = MagicMock(), MagicMock()
        mock_st.columns.return_value = [col1, col2]

        from src.serving.ui.pages.data_explorer import render

        render()

        mock_st.header.assert_called_once_with("Data Explorer")
        mock_st.warning.assert_not_called()

        # Segment breakdown
        mock_st.subheader.assert_any_call("Customer Segment Breakdown")
        # Churn by renewal
        mock_st.subheader.assert_any_call("Churn Rate by Renewal Proximity")
        # Sentiment impact
        mock_st.subheader.assert_any_call("Sentiment Impact on Churn")
        # Profitability
        mock_st.subheader.assert_any_call("Profitability by Segment")
        # Dual fuel
        mock_st.subheader.assert_any_call("Dual-Fuel Analysis")
        # Methodology
        mock_st.subheader.assert_any_call("Key Assumptions & Methodology")

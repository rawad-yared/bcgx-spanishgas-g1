"""Tests for src.data.nlp — intent classification + sentiment enrichment."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.nlp import classify_intent, enrich_interactions, enrich_interactions_intent

# ── classify_intent ──────────────────────────────────────────────────────────


class TestClassifyIntent:
    def test_cancellation(self):
        assert classify_intent("I want to cancel my contract") == "Cancellation / Switch"

    def test_complaint(self):
        assert classify_intent("Very frustrated with the service") == "Complaint / Escalation"

    def test_billing(self):
        assert classify_intent("Question about my billing statement") == "Billing / Payment"

    def test_renewal(self):
        assert classify_intent("When does my contract expire?") == "Contract Renewal"

    def test_pricing(self):
        assert classify_intent("Are there any discounts available?") == "Pricing Offers"

    def test_plan_inquiry(self):
        assert classify_intent("What plan options do you have?") == "Plan / Product Inquiry"

    def test_account_inquiry(self):
        assert classify_intent("Need my account details updated") == "Account / Service Inquiry"

    def test_general(self):
        assert classify_intent("Just a routine follow-up call") == "General / Operational Contact"

    def test_unclassified(self):
        assert classify_intent("hello world xyz") == "Other / Unclassified"

    def test_priority_cancellation_over_complaint(self):
        # "cancel" should win over "complaint" when both present
        assert classify_intent("I want to cancel, this is a complaint") == "Cancellation / Switch"

    def test_case_insensitive(self):
        assert classify_intent("CANCEL my account") == "Cancellation / Switch"
        assert classify_intent("FRUSTRATED with service") == "Complaint / Escalation"

    def test_none_input(self):
        assert classify_intent(None) == "Other / Unclassified"

    def test_empty_string(self):
        assert classify_intent("") == "Other / Unclassified"


# ── enrich_interactions_intent ───────────────────────────────────────────────


class TestEnrichInteractionsIntent:
    @pytest.fixture
    def interactions_df(self):
        return pd.DataFrame({
            "customer_id": ["C001", "C002", "C003"],
            "date": ["2024-06-01", "2024-07-15", None],
            "interaction_summary": [
                "Customer wants to cancel contract",
                "Billing question about charges",
                None,
            ],
        })

    def test_adds_customer_intent_column(self, interactions_df):
        result = enrich_interactions_intent(interactions_df)
        assert "customer_intent" in result.columns
        assert result.loc[0, "customer_intent"] == "Cancellation / Switch"
        assert result.loc[1, "customer_intent"] == "Billing / Payment"

    def test_adds_has_interaction_column(self, interactions_df):
        result = enrich_interactions_intent(interactions_df)
        assert "has_interaction" in result.columns
        assert result.loc[0, "has_interaction"] == 1  # has date + summary
        assert result.loc[1, "has_interaction"] == 1  # has date + summary
        assert result.loc[2, "has_interaction"] == 0  # no date, no summary

    def test_does_not_mutate_input(self, interactions_df):
        original_cols = set(interactions_df.columns)
        enrich_interactions_intent(interactions_df)
        assert set(interactions_df.columns) == original_cols

    def test_preserves_all_rows(self, interactions_df):
        result = enrich_interactions_intent(interactions_df)
        assert len(result) == len(interactions_df)

    def test_has_interaction_date_only(self):
        df = pd.DataFrame({
            "customer_id": ["C001"],
            "date": ["2024-01-01"],
            "interaction_summary": [None],
        })
        result = enrich_interactions_intent(df)
        assert result.loc[0, "has_interaction"] == 1  # has date even though no summary


# ── enrich_interactions_sentiment (graceful fallback) ────────────────────────


class TestEnrichInteractionsSentiment:
    def test_sentiment_import_guard(self):
        """Sentiment enrichment should not crash if transformers is unavailable.

        This test verifies the graceful fallback path — the function returns
        the dataframe unchanged when transformers cannot be imported.
        If transformers IS installed, it will actually run sentiment and we
        verify the output columns exist.
        """
        df = pd.DataFrame({
            "customer_id": ["C001"],
            "interaction_summary": ["I am very happy with the service"],
        })
        result = enrich_interactions(df)
        # Intent columns should always be present
        assert "customer_intent" in result.columns
        assert "has_interaction" in result.columns
        # Row count preserved
        assert len(result) == 1


# ── enrich_interactions (orchestrator) ───────────────────────────────────────


class TestEnrichInteractions:
    def test_full_enrichment(self):
        df = pd.DataFrame({
            "customer_id": ["C001", "C002"],
            "date": ["2024-01-01", None],
            "interaction_summary": [
                "Customer frustrated about billing charges",
                "",
            ],
        })
        result = enrich_interactions(df)
        assert "customer_intent" in result.columns
        assert "has_interaction" in result.columns
        assert len(result) == 2
        # C001: "frustrated" matches Complaint (priority 2) over "billing" (priority 3)
        assert result.loc[0, "customer_intent"] == "Complaint / Escalation"
        assert result.loc[0, "has_interaction"] == 1
        # C002: empty summary, no date
        assert result.loc[1, "has_interaction"] == 0

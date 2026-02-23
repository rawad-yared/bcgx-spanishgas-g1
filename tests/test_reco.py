"""Tests for src.reco — schema and recommendation engine."""

import pandas as pd
import pytest

from src.reco.engine import generate_recommendations
from src.reco.schema import Recommendation


class TestRecommendationSchema:
    def test_valid_recommendation(self):
        rec = Recommendation(
            customer_id="C001",
            risk_score=0.75,
            segment="Residential",
            action="offer_medium",
            timing_window="immediate",
            expected_margin_impact=10.0,
            reason_codes=["high_churn_risk"],
        )
        assert rec.action == "offer_medium"

    def test_empty_reason_codes_raises(self):
        with pytest.raises(ValueError, match="reason_codes must be non-empty"):
            Recommendation(
                customer_id="C001",
                risk_score=0.5,
                segment="Residential",
                action="offer_small",
                timing_window="30_60_days",
                expected_margin_impact=5.0,
                reason_codes=[],
            )

    def test_negative_margin_non_offer_raises(self):
        with pytest.raises(ValueError, match="Negative margin"):
            Recommendation(
                customer_id="C001",
                risk_score=0.6,
                segment="Residential",
                action="offer_medium",
                timing_window="immediate",
                expected_margin_impact=-5.0,
                reason_codes=["high_churn_risk"],
            )

    def test_negative_margin_no_offer_ok(self):
        rec = Recommendation(
            customer_id="C001",
            risk_score=0.6,
            segment="Residential",
            action="no_offer",
            timing_window="immediate",
            expected_margin_impact=-5.0,
            reason_codes=["negative_margin"],
        )
        assert rec.action == "no_offer"

    def test_invalid_risk_score_raises(self):
        with pytest.raises(ValueError, match="risk_score"):
            Recommendation(
                customer_id="C001",
                risk_score=1.5,
                segment="Residential",
                action="no_offer",
                timing_window="immediate",
                expected_margin_impact=0.0,
                reason_codes=["test"],
            )


class TestGenerateRecommendations:
    def test_produces_recommendations(self):
        scored = pd.DataFrame({
            "customer_id": ["C001", "C002", "C003"],
            "churn_proba": [0.10, 0.55, 0.85],
            "risk_tier": ["Low (<40%)", "Medium (40-60%)", "Critical (>80%)"],
            "segment": ["Residential", "SME", "Corporate"],
            "expected_monthly_loss": [1.0, 5.0, 50.0],
        })
        recs = generate_recommendations(scored)
        assert len(recs) == 3
        assert recs[0].action == "no_offer"
        assert recs[1].action == "offer_small"
        assert recs[2].action == "offer_large"

    def test_negative_margin_guardrail(self):
        scored = pd.DataFrame({
            "customer_id": ["C001"],
            "churn_proba": [0.85],
            "risk_tier": ["Critical (>80%)"],
            "segment": ["Residential"],
            "expected_monthly_loss": [-10.0],
        })
        recs = generate_recommendations(scored)
        assert recs[0].action == "no_offer"

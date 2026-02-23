"""Phase 1F: Recommendation engine — map risk tiers to retention actions."""

from __future__ import annotations

import pandas as pd

from src.reco.schema import Recommendation

# Action mapping by risk tier
TIER_ACTIONS = {
    "Low (<40%)": "no_offer",
    "Medium (40-60%)": "offer_small",
    "High (60-80%)": "offer_medium",
    "Critical (>80%)": "offer_large",
}

# Timing windows
TIER_TIMING = {
    "Low (<40%)": "60_90_days",
    "Medium (40-60%)": "30_60_days",
    "High (60-80%)": "immediate",
    "Critical (>80%)": "immediate",
}


def _build_reason_codes(row: pd.Series) -> list[str]:
    """Generate reason codes from scored row features."""
    reasons: list[str] = []

    if row.get("churn_proba", 0) >= 0.80:
        reasons.append("critical_churn_risk")
    elif row.get("churn_proba", 0) >= 0.60:
        reasons.append("high_churn_risk")
    elif row.get("churn_proba", 0) >= 0.40:
        reasons.append("moderate_churn_risk")

    tier = str(row.get("risk_tier", ""))
    if "Critical" in tier:
        reasons.append("churn_probability_above_80pct")
    elif "High" in tier:
        reasons.append("churn_probability_above_60pct")

    segment = str(row.get("segment", ""))
    if segment:
        reasons.append(f"segment_{segment.lower()}")

    if not reasons:
        reasons.append("low_risk_monitoring")

    return reasons


def generate_recommendations(scored: pd.DataFrame) -> list[Recommendation]:
    """Map scored customers to retention recommendations.

    Args:
        scored: DataFrame with columns [customer_id, churn_proba, risk_tier, segment,
                expected_monthly_loss, ...]

    Returns:
        List of Recommendation objects with policy guardrails enforced.
    """
    recommendations: list[Recommendation] = []

    for _, row in scored.iterrows():
        risk_tier = str(row.get("risk_tier", "Low (<40%)"))
        action = TIER_ACTIONS.get(risk_tier, "no_offer")
        timing = TIER_TIMING.get(risk_tier, "60_90_days")

        margin_impact = float(row.get("expected_monthly_loss", 0))

        # Guardrail: no negative-margin offers
        if margin_impact < 0:
            action = "no_offer"

        reason_codes = _build_reason_codes(row)

        rec = Recommendation(
            customer_id=str(row["customer_id"]),
            risk_score=float(row.get("churn_proba", 0)),
            segment=str(row.get("segment", "Unknown")),
            action=action,
            timing_window=timing,
            expected_margin_impact=margin_impact,
            reason_codes=reason_codes,
        )
        recommendations.append(rec)

    return recommendations

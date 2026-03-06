"""Customer Lookup page — enter a customer ID to see full risk profile."""

from __future__ import annotations

import streamlit as st

from src.serving.ui.data_loader import load_recommendations, load_scored_data

# Human-readable action labels
_ACTION_LABELS: dict[str, str] = {
    "offer_large": "Large Retention Offer",
    "offer_medium": "Medium Retention Offer",
    "offer_small": "Small Retention Offer",
    "no_offer": "No Action Needed",
}

# Offer cost as fraction of monthly margin
_OFFER_COST_PCT: dict[str, float] = {
    "offer_large": 0.25,
    "offer_medium": 0.15,
    "offer_small": 0.05,
    "no_offer": 0.0,
}


def _risk_color(proba: float) -> str:
    """Return a CSS color string based on churn probability."""
    if proba >= 0.80:
        return "#d32f2f"  # red
    if proba >= 0.60:
        return "#f57c00"  # orange
    if proba >= 0.40:
        return "#fbc02d"  # yellow
    return "#388e3c"  # green


def render() -> None:
    st.header("Customer Lookup")

    scored = load_scored_data()
    reco = load_recommendations()

    if scored.empty:
        st.warning("No scored data available. Run the scoring pipeline first.")
        return

    # Input
    customer_id = st.text_input("Enter Customer ID", placeholder="e.g. C001")

    if not customer_id:
        st.info("Enter a customer ID above to view their risk profile.")
        return

    # Lookup
    match = scored[scored["customer_id"] == customer_id]
    if match.empty:
        st.error(f"Customer '{customer_id}' not found in scored data.")
        return

    row = match.iloc[0]

    # Merge recommendation data if available
    reco_row = None
    if not reco.empty and "customer_id" in reco.columns:
        reco_match = reco[reco["customer_id"] == customer_id]
        if not reco_match.empty:
            reco_row = reco_match.iloc[0]

    # ------------------------------------------------------------------
    # Section 1: Risk Assessment
    # ------------------------------------------------------------------
    st.subheader("Risk Assessment")
    c1, c2, c3, c4 = st.columns(4)

    churn_proba = row.get("churn_proba", 0.0)
    color = _risk_color(churn_proba)
    c1.markdown(
        f"**Churn Probability**<br>"
        f"<span style='font-size:2em;color:{color}'>{churn_proba:.0%}</span>",
        unsafe_allow_html=True,
    )

    risk_tier = row.get("risk_tier", "N/A")
    c2.metric("Risk Tier", risk_tier)

    action = reco_row.get("action", "N/A") if reco_row is not None else "N/A"
    c3.metric("Recommended Action", _ACTION_LABELS.get(action, action))

    timing = reco_row.get("timing_window", "N/A") if reco_row is not None else "N/A"
    c4.metric("Timing Window", timing.replace("_", " ").title() if timing != "N/A" else "N/A")

    st.divider()

    # ------------------------------------------------------------------
    # Section 2: Financial Impact
    # ------------------------------------------------------------------
    st.subheader("Financial Impact")
    margin = row.get("avg_monthly_margin", 0.0)
    expected_monthly_loss = churn_proba * margin
    expected_annual_loss = expected_monthly_loss * 12
    offer_cost_pct = _OFFER_COST_PCT.get(action, 0.0)
    offer_cost = offer_cost_pct * margin
    net_saved = expected_monthly_loss - offer_cost
    roi = (net_saved / offer_cost * 100) if offer_cost > 0 else float("inf")

    left, right = st.columns(2)
    with left:
        st.markdown("**If No Action**")
        st.metric("Expected Monthly Loss", f"\u20ac{expected_monthly_loss:,.2f}")
        st.metric("Expected Annual Loss", f"\u20ac{expected_annual_loss:,.2f}")
    with right:
        st.markdown("**With Recommended Action**")
        st.metric("Monthly Offer Cost", f"\u20ac{offer_cost:,.2f}")
        st.metric("Net Monthly Profit Saved", f"\u20ac{net_saved:,.2f}")
        roi_str = f"{roi:,.0f}%" if roi != float("inf") else "N/A (no cost)"
        st.metric("ROI of Intervention", roi_str)

    st.divider()

    # ------------------------------------------------------------------
    # Section 3: Why This Recommendation
    # ------------------------------------------------------------------
    with st.expander("Why This Recommendation", expanded=False):
        # Reason codes
        if reco_row is not None:
            raw_codes = reco_row.get("reason_codes", [])
            reason_codes = raw_codes.tolist() if hasattr(raw_codes, "tolist") else list(raw_codes) if raw_codes is not None else []
            if len(reason_codes) > 0:
                st.markdown("**Reason Codes:**")
                for code in reason_codes:
                    st.markdown(f"- {code.replace('_', ' ').title()}")

        # Customer profile
        st.markdown("**Customer Profile:**")
        profile_fields = [
            ("Segment", "segment"),
            ("Tenure (months)", "tenure_months"),
            ("Renewal Bucket", "renewal_bucket"),
            ("Sentiment", "sentiment_label"),
            ("Has Interaction", "has_interaction"),
        ]
        for label, col in profile_fields:
            val = row.get(col, None)
            if val is not None:
                st.markdown(f"- **{label}:** {val}")

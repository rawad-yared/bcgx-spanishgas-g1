"""Shared offer-policy constants and UI helper used by recommendations + customer lookup pages."""

from __future__ import annotations

import pandas as pd
import streamlit as st

# Human-readable action labels
ACTION_LABELS: dict[str, str] = {
    "offer_large": "Large Retention Offer",
    "offer_medium": "Medium Retention Offer",
    "offer_small": "Small Retention Offer",
    "no_offer": "No Action Needed",
}

# Offer cost as fraction of monthly margin
OFFER_COST_PCT: dict[str, float] = {
    "offer_large": 0.25,
    "offer_medium": 0.15,
    "offer_small": 0.05,
    "no_offer": 0.0,
}

_POLICY_TABLE = pd.DataFrame(
    {
        "Risk Tier": ["Critical (>80%)", "High (60-80%)", "Medium (40-60%)", "Low (<40%)"],
        "Action": ["Large Retention Offer", "Medium Retention Offer", "Small Retention Offer", "No Action Needed"],
        "Max Offer (% of Monthly Margin)": ["25%", "15%", "5%", "0%"],
        "Timing": ["Immediate", "Immediate", "30-60 days", "60-90 days (monitor)"],
    }
)

_DISCLAIMER = (
    "**Note:** Offer budgets are calculated as a simplified fixed percentage of each customer's "
    "average monthly margin. Future work will train a dedicated offer-optimization model to determine "
    "personalized retention spend (see report for details)."
)


def render_offer_policy_table() -> None:
    """Render the offer-policy reference table with disclaimer."""
    with st.expander("Offer Policy Reference", expanded=False):
        st.dataframe(_POLICY_TABLE, use_container_width=True, hide_index=True)
        st.caption(_DISCLAIMER)

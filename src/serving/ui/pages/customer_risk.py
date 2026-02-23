"""Customer Risk page — risk tier distribution, customer table."""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from src.serving.ui.data_loader import load_scored_data


def render() -> None:
    st.header("Customer Risk Overview")

    scored = load_scored_data()
    if scored.empty:
        st.warning("No scored data available. Run the scoring pipeline first.")
        return

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", len(scored))

    if "risk_tier" in scored.columns:
        critical = int((scored["risk_tier"] == "Critical (>80%)").sum())
        high = int((scored["risk_tier"] == "High (60-80%)").sum())
        col2.metric("Critical Risk", critical)
        col3.metric("High Risk", high)

    if "expected_monthly_loss" in scored.columns:
        total_loss = scored["expected_monthly_loss"].sum()
        col4.metric("Expected Monthly Loss", f"\u20ac{total_loss:,.0f}")

    st.divider()

    # Risk tier distribution
    if "risk_tier" in scored.columns:
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Risk Tier Distribution")
            tier_counts = scored["risk_tier"].value_counts().reset_index()
            tier_counts.columns = ["Risk Tier", "Count"]
            fig = px.pie(tier_counts, names="Risk Tier", values="Count",
                         color_discrete_sequence=px.colors.qualitative.Set2)
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            st.subheader("Risk Tier Breakdown")
            st.dataframe(tier_counts, use_container_width=True)

    st.divider()

    # Filterable customer table
    st.subheader("Customer Details")

    # Filters
    filter_cols = st.columns(3)
    tier_filter = None
    segment_filter = None

    if "risk_tier" in scored.columns:
        with filter_cols[0]:
            tiers = ["All"] + sorted(scored["risk_tier"].dropna().unique().tolist())
            tier_filter = st.selectbox("Risk Tier", tiers)

    if "segment" in scored.columns:
        with filter_cols[1]:
            segments = ["All"] + sorted(scored["segment"].dropna().unique().tolist())
            segment_filter = st.selectbox("Segment", segments)

    filtered = scored.copy()
    if tier_filter and tier_filter != "All":
        filtered = filtered[filtered["risk_tier"] == tier_filter]
    if segment_filter and segment_filter != "All":
        filtered = filtered[filtered["segment"] == segment_filter]

    # Show top-N
    n = st.slider("Show top N customers", 10, 500, 50)
    display_cols = [c for c in ["customer_id", "churn_proba", "risk_tier", "segment",
                                 "expected_monthly_loss", "churn_pred"] if c in filtered.columns]
    st.dataframe(filtered[display_cols].head(n), use_container_width=True)

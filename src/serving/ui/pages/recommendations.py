"""Recommendations page — retention actions by risk tier and customer."""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from src.serving.ui.data_loader import load_recommendations


def render() -> None:
    st.header("Retention Recommendations")

    reco = load_recommendations()
    if reco.empty:
        st.warning(
            "No recommendation data available. "
            "Run the scoring pipeline and generate recommendations first."
        )
        return

    # ------------------------------------------------------------------
    # Summary metrics
    # ------------------------------------------------------------------
    st.subheader("Summary")
    total = len(reco)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Recommendations", total)

    action_col = "action" if "action" in reco.columns else None
    if action_col:
        action_counts = reco[action_col].value_counts()
        col2.metric("Distinct Actions", len(action_counts))

        # Show top two actions as quick metrics
        top_actions = action_counts.head(2)
        for idx, (col, (action_name, count)) in enumerate(
            zip([col3, col4], top_actions.items())
        ):
            col.metric(action_name.replace("_", " ").title(), count)

    st.divider()

    # ------------------------------------------------------------------
    # Bar chart: recommendation counts by action type
    # ------------------------------------------------------------------
    if action_col:
        st.subheader("Recommendations by Action Type")
        action_df = action_counts.reset_index()
        action_df.columns = ["Action", "Count"]
        fig = px.bar(
            action_df,
            x="Action",
            y="Count",
            color="Action",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(showlegend=False, xaxis_title="Action", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ------------------------------------------------------------------
    # Filters
    # ------------------------------------------------------------------
    st.subheader("Recommendation Details")
    filter_cols = st.columns(3)

    filtered = reco.copy()

    # Filter by action type
    if action_col:
        with filter_cols[0]:
            actions = ["All"] + sorted(reco[action_col].dropna().unique().tolist())
            action_filter = st.selectbox("Action Type", actions)
        if action_filter != "All":
            filtered = filtered[filtered[action_col] == action_filter]

    # Filter by risk tier
    risk_col = "risk_tier" if "risk_tier" in reco.columns else None
    if risk_col:
        with filter_cols[1]:
            tiers = ["All"] + sorted(reco[risk_col].dropna().unique().tolist())
            tier_filter = st.selectbox("Risk Tier", tiers)
        if tier_filter != "All":
            filtered = filtered[filtered[risk_col] == tier_filter]

    # ------------------------------------------------------------------
    # Filterable table
    # ------------------------------------------------------------------
    display_columns = [
        c
        for c in [
            "customer_id",
            "risk_tier",
            "risk_score",
            "segment",
            "action",
            "timing_window",
            "expected_margin_impact",
            "reason_codes",
        ]
        if c in filtered.columns
    ]

    n = st.slider("Show top N recommendations", 10, 500, 50)
    st.dataframe(filtered[display_columns].head(n), use_container_width=True)

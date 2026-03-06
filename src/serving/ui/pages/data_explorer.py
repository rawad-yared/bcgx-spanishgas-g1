"""Data Explorer page — key EDA visualizations from the gold master."""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from src.serving.ui.data_loader import load_gold_data


def render() -> None:
    st.header("Data Explorer")

    gold = load_gold_data()
    if gold.empty:
        st.warning("No gold master data available. Run the pipeline first.")
        return

    # ------------------------------------------------------------------
    # 1. Customer Segment Breakdown
    # ------------------------------------------------------------------
    if "segment" in gold.columns:
        st.subheader("Customer Segment Breakdown")
        seg_counts = gold["segment"].value_counts().reset_index()
        seg_counts.columns = ["Segment", "Count"]
        seg_counts["Pct"] = (seg_counts["Count"] / seg_counts["Count"].sum() * 100).round(1)

        col1, col2 = st.columns(2)
        with col1:
            fig = px.pie(
                seg_counts,
                names="Segment",
                values="Count",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig.update_layout(title_text="Segment Distribution")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.dataframe(seg_counts, use_container_width=True)

        st.divider()

    # ------------------------------------------------------------------
    # 2. Churn Rate by Renewal Proximity
    # ------------------------------------------------------------------
    if "renewal_bucket" in gold.columns and "churn" in gold.columns:
        st.subheader("Churn Rate by Renewal Proximity")
        renewal_churn = (
            gold.groupby("renewal_bucket")["churn"]
            .mean()
            .mul(100)
            .round(1)
            .reset_index()
        )
        renewal_churn.columns = ["Renewal Bucket", "Churn Rate (%)"]
        fig = px.bar(
            renewal_churn,
            x="Renewal Bucket",
            y="Churn Rate (%)",
            color="Churn Rate (%)",
            color_continuous_scale="RdYlGn_r",
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        st.divider()

    # ------------------------------------------------------------------
    # 3. Sentiment Impact on Churn
    # ------------------------------------------------------------------
    if "sentiment_label" in gold.columns and "churn" in gold.columns:
        st.subheader("Sentiment Impact on Churn")
        sent_data = gold.dropna(subset=["sentiment_label"])
        if not sent_data.empty:
            sent_churn = (
                sent_data.groupby("sentiment_label")["churn"]
                .mean()
                .mul(100)
                .round(1)
                .reset_index()
            )
            sent_churn.columns = ["Sentiment", "Churn Rate (%)"]
            fig = px.bar(
                sent_churn,
                x="Sentiment",
                y="Churn Rate (%)",
                color="Churn Rate (%)",
                color_continuous_scale="RdYlGn_r",
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            no_interaction_pct = (
                gold["sentiment_label"].isna().mean() * 100
            )
            st.caption(
                f"Note: {no_interaction_pct:.0f}% of customers have no recorded interaction "
                "(sentiment unavailable)."
            )
        else:
            st.info("No sentiment data available in the gold master.")

        st.divider()

    # ------------------------------------------------------------------
    # 4. Profitability by Segment
    # ------------------------------------------------------------------
    if "segment" in gold.columns and "avg_monthly_margin" in gold.columns:
        st.subheader("Profitability by Segment")
        fig = px.box(
            gold,
            x="segment",
            y="avg_monthly_margin",
            color="segment",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(
            xaxis_title="Segment",
            yaxis_title="Avg Monthly Margin (\u20ac)",
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.divider()

    # ------------------------------------------------------------------
    # 5. Dual-Fuel Analysis
    # ------------------------------------------------------------------
    if "is_dual_fuel" in gold.columns and "churn" in gold.columns:
        st.subheader("Dual-Fuel Analysis")
        dual_churn = (
            gold.groupby("is_dual_fuel")["churn"]
            .mean()
            .mul(100)
            .round(1)
            .reset_index()
        )
        dual_churn.columns = ["Dual Fuel", "Churn Rate (%)"]
        dual_churn["Dual Fuel"] = dual_churn["Dual Fuel"].map(
            {0: "Single Fuel", 1: "Dual Fuel"}
        )
        fig = px.bar(
            dual_churn,
            x="Dual Fuel",
            y="Churn Rate (%)",
            color="Dual Fuel",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        st.divider()

    # ------------------------------------------------------------------
    # 6. Key Assumptions & Methodology
    # ------------------------------------------------------------------
    st.subheader("Key Assumptions & Methodology")
    st.markdown(
        """
- **Dataset:** 20,099 Spanish energy customers
- **Negative consumption** values set to 0
- **Pricing imputation:** hierarchical (customer → segment-month → national average)
- **Segment classification:** Industrial/Residential; SME < 10 kW vs Corporate
- **NLP sentiment:** cardiffnlp/twitter-roberta-base-sentiment-latest
- **68%** of customers have no recorded interaction
- **Data pipeline:** Medallion architecture (Bronze → Silver → Gold)
"""
    )

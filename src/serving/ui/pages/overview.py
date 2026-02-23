"""Overview page — executive KPI dashboard."""

from __future__ import annotations

import streamlit as st

from src.serving.ui.data_loader import load_model_metrics, load_pipeline_runs, load_scored_data


def render() -> None:
    st.header("SpanishGas Churn Intelligence — Overview")
    st.caption("Executive dashboard for the SpanishGas customer churn prediction system")

    # Load data
    scored = load_scored_data()
    metrics = load_model_metrics()
    runs = load_pipeline_runs()

    st.divider()

    # Row 1: Customer KPIs
    st.subheader("Customer Risk Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers Scored", len(scored) if not scored.empty else "\u2014")

    if not scored.empty and "risk_tier" in scored.columns:
        critical = int((scored["risk_tier"] == "Critical (>80%)").sum())
        high = int((scored["risk_tier"] == "High (60-80%)").sum())
        at_risk_pct = round((critical + high) / len(scored) * 100, 1) if len(scored) > 0 else 0
        c2.metric("Critical Risk", critical)
        c3.metric("High Risk", high)
        c4.metric("At-Risk %", f"{at_risk_pct}%")
    else:
        c2.metric("Critical Risk", "\u2014")
        c3.metric("High Risk", "\u2014")
        c4.metric("At-Risk %", "\u2014")

    st.divider()

    # Row 2: Financial + Model KPIs
    st.subheader("Model & Financial")
    m1, m2, m3, m4 = st.columns(4)

    model_metrics = metrics.get("metrics", metrics) if metrics else {}
    m1.metric(
        "PR-AUC",
        f"{model_metrics['pr_auc']:.4f}" if model_metrics.get("pr_auc") else "\u2014",
    )
    m2.metric(
        "ROC-AUC",
        f"{model_metrics['roc_auc']:.4f}" if model_metrics.get("roc_auc") else "\u2014",
    )

    if not scored.empty and "expected_monthly_loss" in scored.columns:
        total_loss = scored["expected_monthly_loss"].sum()
        m3.metric("Expected Monthly Loss", f"\u20ac{total_loss:,.0f}")
    else:
        m3.metric("Expected Monthly Loss", "\u2014")

    # Pipeline runs
    if runs:
        completed = sum(1 for r in runs if r.get("status") == "completed")
        m4.metric("Pipeline Runs (completed)", completed)
    else:
        m4.metric("Pipeline Runs", "\u2014")

    st.divider()

    # Row 3: Quick links
    st.subheader("Quick Navigation")
    col1, col2, col3, col4 = st.columns(4)
    col1.info("**Model Performance** -- PR-AUC, confusion matrix, metrics")
    col2.info("**Drift Monitor** -- Feature & prediction drift")
    col3.info("**Customer Risk** -- Risk tiers, customer details")
    col4.info("**Pipeline Status** -- Run history, step status")

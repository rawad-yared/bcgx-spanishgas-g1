"""Drift Monitor page — feature drift results, prediction distribution."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.serving.ui.data_loader import load_drift_results


def render() -> None:
    st.header("Drift Monitor")

    drift = load_drift_results()
    if not drift:
        st.info("No drift results available. Run the drift detection step first.")
        return

    any_drift = drift.get("any_drift", False)
    n_drifted = drift.get("n_features_drifted", 0)

    col1, col2 = st.columns(2)
    col1.metric("Drift Detected", "Yes" if any_drift else "No")
    col2.metric("Features Drifted", n_drifted)

    st.text(drift.get("summary", ""))

    st.divider()

    # Feature drift results table
    feature_drift = drift.get("feature_drift", {})
    feature_results = feature_drift.get("feature_results", [])

    if feature_results:
        st.subheader("Feature Drift Details")
        df = pd.DataFrame(feature_results)
        df["status"] = df["drifted"].map({True: "DRIFTED", False: "OK"})

        # Color-coded table
        st.dataframe(
            df[["feature", "ks_statistic", "p_value", "status"]].style.apply(
                lambda row: ["background-color: #ffcccc" if row["status"] == "DRIFTED" else "" for _ in row],
                axis=1,
            ),
            use_container_width=True,
        )

        # KS statistic bar chart
        fig = px.bar(
            df.sort_values("ks_statistic", ascending=False),
            x="feature", y="ks_statistic",
            color="drifted",
            color_discrete_map={True: "red", False: "green"},
            title="KS Statistic by Feature",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Prediction drift
    pred_drift = drift.get("prediction_drift", {})
    if pred_drift:
        st.subheader("Prediction Drift")
        st.metric("KS Statistic", f"{pred_drift.get('ks_statistic', 0):.4f}")
        st.metric("p-value", f"{pred_drift.get('p_value', 0):.6f}")

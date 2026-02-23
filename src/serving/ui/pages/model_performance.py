"""Model Performance page — PR-AUC, ROC-AUC, confusion matrix, feature importance."""

from __future__ import annotations

import numpy as np
import plotly.express as px
import streamlit as st

from src.serving.ui.data_loader import load_model_metrics


def render() -> None:
    st.header("Model Performance")

    metrics = load_model_metrics()
    if not metrics:
        st.warning("No model metrics available. Run a training pipeline first.")
        return

    model_metrics = metrics.get("metrics", metrics)

    # Key metric cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("PR-AUC", f"{model_metrics.get('pr_auc', 0):.4f}")
    col2.metric("ROC-AUC", f"{model_metrics.get('roc_auc', 0):.4f}")
    col3.metric("Precision", f"{model_metrics.get('precision', 0):.4f}")
    col4.metric("Recall", f"{model_metrics.get('recall', 0):.4f}")

    st.divider()

    # Confusion matrix
    tp = model_metrics.get("tp", 0)
    fp = model_metrics.get("fp", 0)
    fn = model_metrics.get("fn", 0)
    tn = model_metrics.get("tn", 0)

    if tp + fp + fn + tn > 0:
        st.subheader("Confusion Matrix")
        cm = np.array([[tn, fp], [fn, tp]])
        fig = px.imshow(
            cm,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=["No Churn", "Churn"],
            y=["No Churn", "Churn"],
            text_auto=True,
            color_continuous_scale="Blues",
        )
        fig.update_layout(width=400, height=400)
        st.plotly_chart(fig, use_container_width=False)

    # Additional metrics table
    st.subheader("All Metrics")
    st.json(model_metrics)

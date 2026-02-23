"""SpanishGas MLOps Dashboard — main entry point."""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="SpanishGas MLOps", layout="wide", page_icon="\u26fd")

st.sidebar.title("SpanishGas MLOps")

page = st.sidebar.radio(
    "Navigate",
    ["Model Performance", "Drift Monitor", "Customer Risk", "Pipeline Status"],
)

if page == "Model Performance":
    from src.serving.ui.pages.model_performance import render
    render()
elif page == "Drift Monitor":
    from src.serving.ui.pages.drift_monitor import render
    render()
elif page == "Customer Risk":
    from src.serving.ui.pages.customer_risk import render
    render()
elif page == "Pipeline Status":
    from src.serving.ui.pages.pipeline_status import render
    render()

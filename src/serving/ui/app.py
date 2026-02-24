"""SpanishGas MLOps Dashboard — main entry point."""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="SpanishGas MLOps", layout="wide", page_icon="\u26fd")

# Hide non-functional hamburger menu, footer, and header in production deployments
st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title("SpanishGas MLOps")

page = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "Data Explorer",
        "Model Performance",
        "Customer Risk",
        "Customer Lookup",
        "Recommendations",
        "Drift Monitor",
        "Pipeline Status",
    ],
)

if page == "Overview":
    from src.serving.ui.pages.overview import render

    render()
elif page == "Data Explorer":
    from src.serving.ui.pages.data_explorer import render

    render()
elif page == "Model Performance":
    from src.serving.ui.pages.model_performance import render

    render()
elif page == "Customer Risk":
    from src.serving.ui.pages.customer_risk import render

    render()
elif page == "Customer Lookup":
    from src.serving.ui.pages.customer_lookup import render

    render()
elif page == "Recommendations":
    from src.serving.ui.pages.recommendations import render

    render()
elif page == "Drift Monitor":
    from src.serving.ui.pages.drift_monitor import render

    render()
elif page == "Pipeline Status":
    from src.serving.ui.pages.pipeline_status import render

    render()

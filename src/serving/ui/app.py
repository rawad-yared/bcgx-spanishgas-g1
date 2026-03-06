"""SpanishGas MLOps Dashboard — main entry point."""

from __future__ import annotations

from pathlib import Path

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

_ASSETS = Path(__file__).parent / "assets"

import base64

def _img_to_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()

_ie_b64 = _img_to_b64(_ASSETS / "iesst_logo.jpeg")
_bcg_b64 = _img_to_b64(_ASSETS / "BCG_X.jpg")

st.sidebar.markdown(
    f"""
    <div style="display:flex; align-items:center; gap:18px; margin-bottom:12px;">
        <img src="data:image/jpeg;base64,{_ie_b64}" style="height:80px; width:auto;">
        <img src="data:image/jpeg;base64,{_bcg_b64}" style="height:80px; width:auto;">
    </div>
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

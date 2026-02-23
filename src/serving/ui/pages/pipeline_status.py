"""Pipeline Status page — run history, step status, manifest overview."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.serving.ui.data_loader import load_pipeline_runs


def render() -> None:
    st.header("Pipeline Status")

    runs = load_pipeline_runs()
    if not runs:
        st.info("No pipeline runs recorded yet. Trigger a pipeline run to see status here.")
        return

    df = pd.DataFrame(runs)

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Runs", len(df))

    if "status" in df.columns:
        completed = int((df["status"] == "completed").sum())
        started = int((df["status"] == "started").sum())
        failed = int((df["status"] == "failed").sum())
        col2.metric("Completed", completed)
        col3.metric("In Progress", started)
        col4.metric("Failed", failed)

    st.divider()

    # Status filter
    filter_col, _ = st.columns([1, 3])
    with filter_col:
        statuses = ["All"]
        if "status" in df.columns:
            statuses += sorted(df["status"].dropna().unique().tolist())
        status_filter = st.selectbox("Filter by Status", statuses)

    filtered = df.copy()
    if status_filter != "All" and "status" in df.columns:
        filtered = filtered[filtered["status"] == status_filter]

    # Run history table
    st.subheader("Run History")
    display_cols = [c for c in ["run_id", "file_key", "status", "started_at", "completed_at"]
                    if c in filtered.columns]
    if display_cols:
        styled = filtered[display_cols].copy()
        if "status" in styled.columns:
            styled["status"] = styled["status"].apply(_status_emoji)
        st.dataframe(styled, use_container_width=True)
    else:
        st.dataframe(filtered, use_container_width=True)


def _status_emoji(status: str) -> str:
    mapping = {
        "completed": "completed",
        "started": "started (in progress)",
        "failed": "FAILED",
    }
    return mapping.get(status, status)

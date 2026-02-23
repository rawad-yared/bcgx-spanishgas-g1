"""Cached data loading for Streamlit dashboard."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st


@st.cache_data(ttl=300)
def load_scored_data(source: str = "local", path: str = "data/scored/scored_customers.parquet", **kwargs) -> pd.DataFrame:
    """Load scored customer data from local parquet or S3."""
    if source == "s3":
        from src.pipelines.s3_io import read_parquet
        return read_parquet(kwargs["bucket"], kwargs["key"], kwargs.get("region", "eu-west-1"))

    p = Path(path)
    if p.exists():
        return pd.read_parquet(p)
    return pd.DataFrame()


@st.cache_data(ttl=300)
def load_model_metrics(source: str = "local", path: str = "data/models/latest/evaluation.json", **kwargs) -> dict:
    """Load model evaluation metrics."""
    if source == "s3":
        from src.pipelines.s3_io import read_json_s3
        return read_json_s3(kwargs["bucket"], kwargs["key"], kwargs.get("region", "eu-west-1"))

    p = Path(path)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}


@st.cache_data(ttl=300)
def load_drift_results(source: str = "local", path: str = "data/monitoring/drift_results.json", **kwargs) -> dict:
    """Load drift detection results."""
    if source == "s3":
        from src.pipelines.s3_io import read_json_s3
        return read_json_s3(kwargs["bucket"], kwargs["key"], kwargs.get("region", "eu-west-1"))

    p = Path(path)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}


@st.cache_data(ttl=60)
def load_pipeline_runs(source: str = "local", path: str = "data/monitoring/pipeline_runs.json", **kwargs) -> list[dict]:
    """Load pipeline run history.

    Each run dict has: run_id, file_key, status, started_at, completed_at (optional).
    In production reads from DynamoDB via scan; locally from JSON file.
    """
    if source == "dynamodb":
        import boto3

        region = kwargs.get("region", "eu-west-1")
        table_name = kwargs.get("table_name", "spanishgas-pipeline-manifest")
        table = boto3.resource("dynamodb", region_name=region).Table(table_name)
        resp = table.scan()
        items = resp.get("Items", [])
        while "LastEvaluatedKey" in resp:
            resp = table.scan(ExclusiveStartKey=resp["LastEvaluatedKey"])
            items.extend(resp.get("Items", []))
        return sorted(items, key=lambda x: x.get("started_at", ""), reverse=True)

    p = Path(path)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return []

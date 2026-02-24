"""Cached data loading for Streamlit dashboard."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Module-level configuration from environment variables
# ---------------------------------------------------------------------------
DATA_SOURCE = os.getenv("DATA_SOURCE", "local")
S3_BUCKET = os.getenv("S3_BUCKET", "spanishgas-data-dev")
AWS_REGION = os.getenv("AWS_REGION", "eu-west-1")
DYNAMODB_TABLE = os.getenv("DYNAMODB_MANIFEST_TABLE", "spanishgas-dev-pipeline-manifest")


@st.cache_data(ttl=300)
def load_scored_data(source: str | None = None, path: str = "data/scored/scored_customers.parquet", **kwargs) -> pd.DataFrame:
    """Load scored customer data from local parquet or S3."""
    source = source or DATA_SOURCE
    if source == "s3":
        from src.pipelines.s3_io import read_parquet

        bucket = kwargs.get("bucket", S3_BUCKET)
        key = kwargs.get("key", "scored/scored_customers.parquet")
        region = kwargs.get("region", AWS_REGION)
        return read_parquet(bucket, key, region)

    p = Path(path)
    if p.exists():
        return pd.read_parquet(p)
    return pd.DataFrame()


@st.cache_data(ttl=300)
def load_model_metrics(source: str | None = None, path: str = "data/models/latest/evaluation.json", **kwargs) -> dict:
    """Load model evaluation metrics."""
    source = source or DATA_SOURCE
    if source == "s3":
        from src.pipelines.s3_io import read_json_s3

        bucket = kwargs.get("bucket", S3_BUCKET)
        key = kwargs.get("key", "models/latest/evaluation.json")
        region = kwargs.get("region", AWS_REGION)
        return read_json_s3(bucket, key, region)

    p = Path(path)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}


@st.cache_data(ttl=300)
def load_drift_results(source: str | None = None, path: str = "data/monitoring/drift_results.json", **kwargs) -> dict:
    """Load drift detection results."""
    source = source or DATA_SOURCE
    if source == "s3":
        from src.pipelines.s3_io import read_json_s3

        bucket = kwargs.get("bucket", S3_BUCKET)
        key = kwargs.get("key", "monitoring/drift_results.json")
        region = kwargs.get("region", AWS_REGION)
        return read_json_s3(bucket, key, region)

    p = Path(path)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}


@st.cache_data(ttl=60)
def load_pipeline_runs(source: str | None = None, path: str = "data/monitoring/pipeline_runs.json", **kwargs) -> list[dict]:
    """Load pipeline run history.

    Each run dict has: run_id, file_key, status, started_at, completed_at (optional).
    In production reads from DynamoDB via scan; locally from JSON file.
    """
    source = source or DATA_SOURCE
    if source == "dynamodb" or (source == "s3" and kwargs.get("table_name")):
        import boto3

        region = kwargs.get("region", AWS_REGION)
        table_name = kwargs.get("table_name", DYNAMODB_TABLE)
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


@st.cache_data(ttl=300)
def load_gold_data(source: str | None = None, path: str = "data/gold/gold_master.parquet", **kwargs) -> pd.DataFrame:
    """Load gold master data for EDA exploration."""
    source = source or DATA_SOURCE
    if source == "s3":
        from src.pipelines.s3_io import read_parquet

        bucket = kwargs.get("bucket", S3_BUCKET)
        key = kwargs.get("key", "gold/gold_master.parquet")
        region = kwargs.get("region", AWS_REGION)
        return read_parquet(bucket, key, region)

    p = Path(path)
    if p.exists():
        return pd.read_parquet(p)
    return pd.DataFrame()


@st.cache_data(ttl=300)
def load_recommendations(source: str | None = None, path: str = "data/scored/recommendations.parquet", **kwargs) -> pd.DataFrame:
    """Load recommendation results."""
    source = source or DATA_SOURCE
    if source == "s3":
        from src.pipelines.s3_io import read_parquet

        bucket = kwargs.get("bucket", S3_BUCKET)
        key = kwargs.get("key", "scored/recommendations.parquet")
        region = kwargs.get("region", AWS_REGION)
        return read_parquet(bucket, key, region)

    p = Path(path)
    if p.exists():
        return pd.read_parquet(p)
    return pd.DataFrame()

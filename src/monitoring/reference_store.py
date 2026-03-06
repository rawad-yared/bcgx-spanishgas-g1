"""Save and load reference feature distributions for drift detection."""

from __future__ import annotations

import json
import logging

import pandas as pd

from src.pipelines.s3_io import get_s3_client

logger = logging.getLogger(__name__)


def save_reference(
    df: pd.DataFrame,
    features: list[str],
    bucket: str,
    key: str,
    region: str = "eu-west-1",
) -> None:
    """Compute and save reference distributions to S3 as JSON.

    Saves raw values, mean, std, and quantiles for each numeric feature.
    """
    ref: dict = {"features": {}}

    for feat in features:
        if feat not in df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[feat]):
            continue

        vals = df[feat].dropna()
        ref["features"][feat] = {
            "values": vals.tolist(),
            "mean": float(vals.mean()) if len(vals) > 0 else 0.0,
            "std": float(vals.std()) if len(vals) > 0 else 0.0,
            "quantiles": {
                "0.25": float(vals.quantile(0.25)) if len(vals) > 0 else 0.0,
                "0.50": float(vals.quantile(0.50)) if len(vals) > 0 else 0.0,
                "0.75": float(vals.quantile(0.75)) if len(vals) > 0 else 0.0,
            },
            "count": len(vals),
        }

    s3 = get_s3_client(region)
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(ref).encode("utf-8"),
    )
    logger.info("Saved reference distributions (%d features) to s3://%s/%s", len(ref["features"]), bucket, key)


def load_reference(
    bucket: str,
    key: str,
    region: str = "eu-west-1",
) -> dict:
    """Load reference distributions from S3."""
    s3 = get_s3_client(region)
    obj = s3.get_object(Bucket=bucket, Key=key)
    return json.loads(obj["Body"].read().decode("utf-8"))

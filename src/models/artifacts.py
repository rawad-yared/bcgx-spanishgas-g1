"""Model artifact persistence — save/load sklearn pipelines + metadata."""

from __future__ import annotations

import io
import json
import logging

import joblib

from src.pipelines.s3_io import get_s3_client

logger = logging.getLogger(__name__)


def save_model(
    pipeline,
    threshold: float,
    metrics: dict,
    model_name: str,
    bucket: str,
    key_prefix: str,
    region: str = "eu-west-1",
) -> None:
    """Save pipeline (joblib) and metadata JSON to S3."""
    s3 = get_s3_client(region)

    # Save pipeline as joblib
    buf = io.BytesIO()
    joblib.dump(pipeline, buf)
    buf.seek(0)
    s3.put_object(Bucket=bucket, Key=f"{key_prefix}model.joblib", Body=buf.getvalue())

    # Save metadata
    metadata = {
        "model_name": model_name,
        "threshold": threshold,
        "metrics": metrics,
    }
    s3.put_object(
        Bucket=bucket,
        Key=f"{key_prefix}metadata.json",
        Body=json.dumps(metadata).encode("utf-8"),
    )

    logger.info("Saved model artifacts to s3://%s/%s", bucket, key_prefix)


def load_model(
    bucket: str,
    key_prefix: str,
    region: str = "eu-west-1",
) -> tuple:
    """Load pipeline + metadata from S3.

    Returns (pipeline, metadata_dict) where metadata includes threshold, metrics, model_name.
    """
    s3 = get_s3_client(region)

    # Load pipeline
    obj = s3.get_object(Bucket=bucket, Key=f"{key_prefix}model.joblib")
    pipeline = joblib.load(io.BytesIO(obj["Body"].read()))

    # Load metadata
    obj = s3.get_object(Bucket=bucket, Key=f"{key_prefix}metadata.json")
    metadata = json.loads(obj["Body"].read().decode("utf-8"))

    logger.info("Loaded model from s3://%s/%s (model=%s)", bucket, key_prefix, metadata.get("model_name"))
    return pipeline, metadata

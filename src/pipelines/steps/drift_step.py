"""Drift detection step — SageMaker Processing Job entry point."""

from __future__ import annotations

import argparse
import logging
import os

import numpy as np
import pandas as pd

from botocore.exceptions import ClientError

from src.monitoring.alerts import publish_cloudwatch_metrics, publish_sns_alert
from src.monitoring.drift import compute_feature_drift, compute_prediction_drift, summarize_drift
from src.monitoring.reference_store import load_reference, save_reference
from src.pipelines.s3_io import read_parquet, write_json

logger = logging.getLogger(__name__)


def _build_reference_dataframe(ref_payload: dict) -> pd.DataFrame:
    """Reconstruct a DataFrame from saved reference distribution values."""
    features_data: dict[str, list] = {}
    for feat_name, dist in ref_payload.get("features", {}).items():
        features_data[feat_name] = dist.get("values", [])

    if not features_data:
        return pd.DataFrame()

    # Pad to equal length
    max_len = max(len(v) for v in features_data.values())
    for k in features_data:
        diff = max_len - len(features_data[k])
        if diff > 0:
            features_data[k] = features_data[k] + [np.nan] * diff

    return pd.DataFrame(features_data)


def run_drift_step(
    bucket: str,
    reference_key: str,
    scored_key: str,
    output_key: str = "monitoring/drift_results.json",
    region: str = "eu-west-1",
    namespace: str = "SpanishGas/MLOps",
    sns_topic_arn: str | None = None,
    p_threshold: float = 0.01,
) -> dict:
    """Execute drift detection: load reference + current, compute, alert."""
    logger.info("Drift step: ref=%s scored=%s", reference_key, scored_key)

    # Load scored data first (needed regardless)
    scored_df = read_parquet(bucket, scored_key, region)
    numeric_features = [c for c in scored_df.select_dtypes(include="number").columns
                        if c not in ("customer_id",)]

    # Load reference; on first run it won't exist yet
    try:
        ref_payload = load_reference(bucket, reference_key, region)
    except ClientError as exc:
        if exc.response["Error"]["Code"] == "NoSuchKey":
            logger.info("No reference found — first run. Saving current data as reference.")
            save_reference(scored_df, numeric_features, bucket, reference_key, region)
            first_run_result = {
                "any_drift": False,
                "n_features_drifted": 0,
                "summary": "First run — reference baseline saved, no drift comparison.",
                "feature_drift": {},
                "prediction_drift": {},
            }
            write_json(first_run_result, bucket, output_key, region)
            return first_run_result
        raise

    reference_df = _build_reference_dataframe(ref_payload)
    features = list(ref_payload.get("features", {}).keys())
    logger.info("Reference: %d features, Scored: %d rows", len(features), len(scored_df))

    # Feature drift
    feature_drift = compute_feature_drift(reference_df, scored_df, features, p_threshold)

    # Prediction drift
    ref_probas = np.array(ref_payload.get("features", {}).get("churn_proba", {}).get("values", []))
    cur_probas = scored_df["churn_proba"].values if "churn_proba" in scored_df.columns else np.array([])
    prediction_drift = compute_prediction_drift(ref_probas, cur_probas, p_threshold)

    drift_summary = summarize_drift(feature_drift, prediction_drift)
    logger.info("Drift: %s", drift_summary["summary"])

    # Publish CloudWatch metrics
    cw_metrics = [
        {"MetricName": "DriftDetected", "Value": 1.0 if drift_summary["any_drift"] else 0.0, "Unit": "Count"},
        {"MetricName": "FeaturesDrifted", "Value": float(drift_summary["n_features_drifted"]), "Unit": "Count"},
    ]
    try:
        publish_cloudwatch_metrics(namespace, cw_metrics, region)
    except Exception:
        logger.exception("Failed to publish CloudWatch metrics")

    # SNS alert if drift detected
    if drift_summary["any_drift"]:
        topic = sns_topic_arn or os.environ.get("SNS_TOPIC_ARN", "")
        if topic:
            try:
                publish_sns_alert(topic, "SpanishGas: Drift Detected", drift_summary["summary"], region)
            except Exception:
                logger.exception("Failed to publish SNS alert")

    # Save results to S3
    write_json(drift_summary, bucket, output_key, region)
    return drift_summary


def main() -> None:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"),
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Drift detection step")
    parser.add_argument("--bucket", default=os.environ.get("S3_BUCKET", ""))
    parser.add_argument("--reference-key", default="monitoring/reference.json")
    parser.add_argument("--scored-key", default="scored/scored_customers.parquet")
    parser.add_argument("--output-key", default="monitoring/drift_results.json")
    parser.add_argument("--region", default=os.environ.get("AWS_REGION", "eu-west-1"))
    parser.add_argument("--namespace", default="SpanishGas/MLOps")
    parser.add_argument("--sns-topic-arn", default=os.environ.get("SNS_TOPIC_ARN"))
    parser.add_argument("--p-threshold", type=float, default=0.01)
    args = parser.parse_args()

    if not args.bucket:
        parser.error("--bucket is required (or set S3_BUCKET env var)")

    run_drift_step(
        args.bucket, args.reference_key, args.scored_key,
        args.output_key, args.region, args.namespace,
        args.sns_topic_arn, args.p_threshold,
    )


if __name__ == "__main__":
    main()

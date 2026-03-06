"""Evaluation step — compare candidate model against promotion threshold."""

from __future__ import annotations

import argparse
import logging
import os

from src.data.build_training_set import build_model_matrix, create_train_test_split
from src.models.artifacts import load_model
from src.models.churn_model import evaluate_model
from src.pipelines.s3_io import read_json_s3, read_parquet, write_json

logger = logging.getLogger(__name__)

PROMOTION_THRESHOLD = 0.70  # PR-AUC minimum for promotion


def run_evaluate_step(
    bucket: str,
    models_prefix: str = "models/",
    gold_prefix: str = "gold/",
    region: str = "eu-west-1",
    pr_auc_threshold: float = PROMOTION_THRESHOLD,
) -> dict:
    """Load trained model, evaluate on holdout, decide promote/reject."""
    logger.info("Evaluate step: pr_auc_threshold=%.2f", pr_auc_threshold)

    model_key = f"{models_prefix}latest/"
    pipeline, metadata = load_model(bucket, model_key, region)
    threshold = metadata["threshold"]

    # Load gold and re-split to get the same test set
    gold = read_parquet(bucket, f"{gold_prefix}gold_master.parquet", region)
    features = metadata.get("features", [])
    if not features:
        eval_data = read_json_s3(bucket, f"{model_key}evaluation.json", region)
        features = eval_data.get("features", [])

    X, y, _ = build_model_matrix(gold, features)
    _, X_test, _, y_test = create_train_test_split(X, y)

    # Evaluate
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    metrics = evaluate_model(y_test, y_proba, threshold)

    promote = metrics["pr_auc"] >= pr_auc_threshold
    logger.info("PR-AUC=%.4f (threshold=%.2f) → %s",
                metrics["pr_auc"], pr_auc_threshold, "PROMOTE" if promote else "REJECT")

    result = {
        "metrics": metrics,
        "promote": promote,
        "pr_auc_threshold": pr_auc_threshold,
        "model_key": model_key,
    }

    write_json(result, bucket, f"{model_key}promote_decision.json", region)
    logger.info("Evaluate step complete")
    return result


def main() -> None:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"),
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Model evaluation step")
    parser.add_argument("--bucket", default=os.environ.get("S3_BUCKET", ""))
    parser.add_argument("--models-prefix", default="models/")
    parser.add_argument("--gold-prefix", default="gold/")
    parser.add_argument("--region", default=os.environ.get("AWS_REGION", "eu-west-1"))
    parser.add_argument("--pr-auc-threshold", type=float, default=PROMOTION_THRESHOLD)
    args = parser.parse_args()

    if not args.bucket:
        parser.error("--bucket is required (or set S3_BUCKET env var)")

    run_evaluate_step(
        args.bucket, args.models_prefix, args.gold_prefix,
        args.region, args.pr_auc_threshold,
    )


if __name__ == "__main__":
    main()

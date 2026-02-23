"""Training step — SageMaker Processing/Training Job entry point."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import yaml

from src.data.build_training_set import build_model_matrix, create_train_test_split
from src.models.artifacts import save_model
from src.models.churn_model import run_experiment
from src.pipelines.s3_io import read_parquet, write_json

logger = logging.getLogger(__name__)


def _load_feature_list(config_path: str = "configs/feature_tiers.yaml", experiment: str = "E5") -> list[str]:
    """Load feature list for the given experiment from config YAML."""
    path = Path(config_path)
    if not path.exists():
        # Fallback: try relative to module
        path = Path(__file__).resolve().parents[3] / config_path
    with open(path) as f:
        config = yaml.safe_load(f)

    experiments = config.get("experiments", {})
    exp_def = experiments.get(experiment, {})
    tier_names = exp_def.get("tiers", [])

    tiers = config.get("tiers", {})
    features: list[str] = []
    for tier_name in tier_names:
        features.extend(tiers.get(tier_name, []))
    return features


def run_train_step(
    bucket: str,
    gold_prefix: str = "gold/",
    models_prefix: str = "models/",
    region: str = "eu-west-1",
    model_name: str = "xgboost",
    experiment: str = "E5",
    target_recall: float = 0.70,
) -> dict:
    """Load gold, train model, save artifacts to S3."""
    logger.info("Train step: model=%s experiment=%s", model_name, experiment)

    gold = read_parquet(bucket, f"{gold_prefix}gold_master.parquet", region)
    logger.info("Gold master: %d rows", len(gold))

    features = _load_feature_list(experiment=experiment)
    logger.info("Feature list (%s): %d features", experiment, len(features))

    X, y, cids = build_model_matrix(gold, features)
    X_train, X_test, y_train, y_test = create_train_test_split(X, y)
    logger.info("Train: %d, Test: %d, Churn rate: %.3f", len(X_train), len(X_test), float(y.mean()))

    result = run_experiment(
        X_train, y_train, X_test, y_test,
        model_name=model_name,
        target_recall=target_recall,
    )

    metrics = result["metrics"]
    logger.info("Metrics: PR-AUC=%.4f ROC-AUC=%.4f Precision=%.4f Recall=%.4f",
                metrics["pr_auc"], metrics["roc_auc"], metrics["precision"], metrics["recall"])

    # Save model artifacts to S3
    run_key = f"{models_prefix}latest/"
    save_model(
        pipeline=result["pipeline"],
        threshold=result["threshold"],
        metrics=metrics,
        model_name=model_name,
        bucket=bucket,
        key_prefix=run_key,
        region=region,
    )

    # Save evaluation results
    eval_result = {
        "model_name": model_name,
        "experiment": experiment,
        "threshold": result["threshold"],
        "metrics": metrics,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "features": features,
    }
    write_json(eval_result, bucket, f"{run_key}evaluation.json", region)

    logger.info("Train step complete — artifacts at s3://%s/%s", bucket, run_key)
    return eval_result


def main() -> None:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"),
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Model training step")
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--gold-prefix", default="gold/")
    parser.add_argument("--models-prefix", default="models/")
    parser.add_argument("--region", default="eu-west-1")
    parser.add_argument("--model-name", default="xgboost")
    parser.add_argument("--experiment", default="E5")
    parser.add_argument("--target-recall", type=float, default=0.70)
    args = parser.parse_args()

    run_train_step(
        args.bucket, args.gold_prefix, args.models_prefix,
        args.region, args.model_name, args.experiment, args.target_recall,
    )


if __name__ == "__main__":
    main()

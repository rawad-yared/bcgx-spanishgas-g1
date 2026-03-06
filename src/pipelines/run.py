"""Local pipeline runner for dev/testing — runs bronze→silver→gold→train→score."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from src.data.build_training_set import build_model_matrix, create_train_test_split
from src.data.ingest import (
    build_bronze_customer,
    build_bronze_customer_month,
    load_or_convert_consumption,
    load_raw_datasets,
)
from src.data.silver import build_silver_tables
from src.features.build_features import build_gold_master
from src.models.churn_model import run_experiment
from src.models.scorer import assign_risk_tiers, score_all_customers

logger = logging.getLogger(__name__)


def run_local_pipeline(
    data_dir: str = "data",
    output_dir: str = "data",
    model_name: str = "random_forest",
    target_recall: float = 0.70,
) -> dict:
    """Run the full pipeline locally using filesystem I/O."""
    data_path = Path(data_dir)
    out_path = Path(output_dir)

    # Bronze
    logger.info("Loading raw datasets from %s", data_path)
    raw = load_raw_datasets(data_path)
    consumption = load_or_convert_consumption(data_path)

    bronze_customer = build_bronze_customer(
        raw["churn"], raw["attributes"], raw["contracts"], raw["interactions"],
    )
    bronze_customer_month = build_bronze_customer_month(
        consumption, raw.get("prices"), raw.get("costs"),
        province_lookup=raw.get("attributes"),
    )
    logger.info("Bronze: customer=%d, customer_month=%d",
                len(bronze_customer), len(bronze_customer_month))

    # Silver
    silver_customer, silver_customer_month = build_silver_tables(
        bronze_customer, bronze_customer_month,
    )
    logger.info("Silver: customer=%d, customer_month=%d",
                len(silver_customer), len(silver_customer_month))

    # Gold
    gold = build_gold_master(silver_customer, silver_customer_month)
    logger.info("Gold: %d rows, %d columns", len(gold), len(gold.columns))

    # Save intermediate outputs
    bronze_dir = out_path / "bronze"
    silver_dir = out_path / "silver"
    gold_dir = out_path / "gold"
    scored_dir = out_path / "scored"
    for d in [bronze_dir, silver_dir, gold_dir, scored_dir]:
        d.mkdir(parents=True, exist_ok=True)

    bronze_customer.to_parquet(bronze_dir / "bronze_customer.parquet", index=False)
    bronze_customer_month.to_parquet(bronze_dir / "bronze_customer_month.parquet", index=False)
    silver_customer.to_parquet(silver_dir / "silver_customer.parquet", index=False)
    silver_customer_month.to_parquet(silver_dir / "silver_customer_month.parquet", index=False)
    gold.to_parquet(gold_dir / "gold_master.parquet", index=False)

    # Train
    feature_cols = [c for c in gold.columns if c not in ("customer_id", "churn")]
    X, y, cids = build_model_matrix(gold, feature_cols)
    X_train, X_test, y_train, y_test = create_train_test_split(X, y)

    result = run_experiment(
        X_train, y_train, X_test, y_test,
        model_name=model_name,
        target_recall=target_recall,
    )
    metrics = result["metrics"]
    logger.info("Model: %s | PR-AUC=%.4f ROC-AUC=%.4f",
                model_name, metrics["pr_auc"], metrics["roc_auc"])

    # Score
    scored = score_all_customers(result["pipeline"], gold, feature_cols, result["threshold"])
    scored = assign_risk_tiers(scored)
    scored.to_parquet(scored_dir / "scored_customers.parquet", index=False)

    logger.info("Scored %d customers. Critical: %d, High: %d",
                len(scored),
                int((scored["risk_tier"] == "Critical (>80%)").sum()),
                int((scored["risk_tier"] == "High (60-80%)").sum()))

    return {
        "metrics": metrics,
        "n_customers": len(scored),
        "model_name": model_name,
        "threshold": result["threshold"],
    }


def main() -> None:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"),
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Local pipeline runner")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--model-name", default="random_forest")
    parser.add_argument("--target-recall", type=float, default=0.70)
    args = parser.parse_args()

    run_local_pipeline(args.data_dir, args.output_dir, args.model_name, args.target_recall)


if __name__ == "__main__":
    main()

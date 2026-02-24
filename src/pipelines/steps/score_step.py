"""Scoring step — batch score all customers with champion model."""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import asdict

import pandas as pd

from src.models.artifacts import load_model
from src.models.scorer import assign_risk_tiers, score_all_customers
from src.pipelines.s3_io import read_json_s3, read_parquet, write_parquet
from src.reco.engine import generate_recommendations

logger = logging.getLogger(__name__)


def run_score_step(
    bucket: str,
    models_prefix: str = "models/",
    gold_prefix: str = "gold/",
    scored_prefix: str = "scored/",
    region: str = "eu-west-1",
) -> None:
    """Load champion model, score all customers, assign risk tiers, write output."""
    logger.info("Score step: bucket=%s models=%s gold=%s scored=%s",
                bucket, models_prefix, gold_prefix, scored_prefix)

    model_key = f"{models_prefix}latest/"
    pipeline, metadata = load_model(bucket, model_key, region)
    threshold = metadata["threshold"]

    # Get feature list
    features = metadata.get("features", [])
    if not features:
        eval_data = read_json_s3(bucket, f"{model_key}evaluation.json", region)
        features = eval_data.get("features", [])

    # Load gold master
    gold = read_parquet(bucket, f"{gold_prefix}gold_master.parquet", region)
    logger.info("Gold master: %d customers", len(gold))

    # Score
    scored = score_all_customers(pipeline, gold, features, threshold)
    scored = assign_risk_tiers(scored)

    logger.info("Scored: %d customers, Critical: %d, High: %d",
                len(scored),
                int((scored["risk_tier"] == "Critical (>80%)").sum()),
                int((scored["risk_tier"] == "High (60-80%)").sum()))

    write_parquet(scored, bucket, f"{scored_prefix}scored_customers.parquet", region)

    # Generate retention recommendations
    logger.info("Generating recommendations for %d customers", len(scored))
    reco_list = generate_recommendations(scored)
    reco_df = pd.DataFrame([asdict(r) for r in reco_list])
    write_parquet(reco_df, bucket, f"{scored_prefix}recommendations.parquet", region)
    logger.info("Wrote %d recommendations", len(reco_df))

    logger.info("Score step complete")


def main() -> None:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"),
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Batch scoring step")
    parser.add_argument("--bucket", default=os.environ.get("S3_BUCKET", ""))
    parser.add_argument("--models-prefix", default="models/")
    parser.add_argument("--gold-prefix", default="gold/")
    parser.add_argument("--scored-prefix", default="scored/")
    parser.add_argument("--region", default=os.environ.get("AWS_REGION", "eu-west-1"))
    args = parser.parse_args()

    if not args.bucket:
        parser.error("--bucket is required (or set S3_BUCKET env var)")

    run_score_step(args.bucket, args.models_prefix, args.gold_prefix,
                   args.scored_prefix, args.region)


if __name__ == "__main__":
    main()

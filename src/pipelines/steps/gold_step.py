"""Gold feature engineering step — SageMaker Processing Job entry point."""

from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime

from src.features.build_features import build_gold_master
from src.pipelines.s3_io import read_parquet, write_parquet

logger = logging.getLogger(__name__)


def run_gold_step(
    bucket: str,
    silver_prefix: str = "silver/",
    gold_prefix: str = "gold/",
    region: str = "eu-west-1",
    as_of_date: str | None = None,
) -> None:
    """Read silver parquet, build gold master, write gold parquet."""
    logger.info("Gold step: bucket=%s silver=%s gold=%s", bucket, silver_prefix, gold_prefix)

    silver_customer = read_parquet(bucket, f"{silver_prefix}silver_customer.parquet", region)
    silver_customer_month = read_parquet(bucket, f"{silver_prefix}silver_customer_month.parquet", region)

    logger.info("Loaded silver: customer=%d, customer_month=%d",
                len(silver_customer), len(silver_customer_month))

    aod = datetime.fromisoformat(as_of_date) if as_of_date else None
    gold = build_gold_master(silver_customer, silver_customer_month, aod)

    logger.info("Gold master: %d rows, %d columns", len(gold), len(gold.columns))

    write_parquet(gold, bucket, f"{gold_prefix}gold_master.parquet", region)
    logger.info("Gold step complete")


def main() -> None:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"),
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Gold feature engineering step")
    parser.add_argument("--bucket", default=os.environ.get("S3_BUCKET", ""))
    parser.add_argument("--silver-prefix", default="silver/")
    parser.add_argument("--gold-prefix", default="gold/")
    parser.add_argument("--region", default=os.environ.get("AWS_REGION", "eu-west-1"))
    parser.add_argument("--as-of-date", default=None, help="ISO date for feature computation")
    args = parser.parse_args()

    if not args.bucket:
        parser.error("--bucket is required (or set S3_BUCKET env var)")

    run_gold_step(args.bucket, args.silver_prefix, args.gold_prefix, args.region, args.as_of_date)


if __name__ == "__main__":
    main()

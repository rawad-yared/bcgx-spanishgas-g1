"""Silver ETL step — SageMaker Processing Job entry point."""

from __future__ import annotations

import argparse
import logging
import os

from src.data.silver import build_silver_tables
from src.pipelines.s3_io import read_parquet, write_parquet

logger = logging.getLogger(__name__)


def run_silver_step(
    bucket: str,
    bronze_prefix: str = "bronze/",
    silver_prefix: str = "silver/",
    region: str = "eu-west-1",
) -> None:
    """Read bronze parquet from S3, run silver transforms, write silver parquet."""
    logger.info("Silver step: bucket=%s bronze=%s silver=%s", bucket, bronze_prefix, silver_prefix)

    bronze_customer = read_parquet(bucket, f"{bronze_prefix}bronze_customer.parquet", region)
    bronze_customer_month = read_parquet(bucket, f"{bronze_prefix}bronze_customer_month.parquet", region)

    logger.info("Loaded bronze: customer=%d, customer_month=%d",
                len(bronze_customer), len(bronze_customer_month))

    silver_customer, silver_customer_month = build_silver_tables(bronze_customer, bronze_customer_month)

    logger.info("Silver customer: %d rows, Silver customer-month: %d rows",
                len(silver_customer), len(silver_customer_month))

    write_parquet(silver_customer, bucket, f"{silver_prefix}silver_customer.parquet", region)
    write_parquet(silver_customer_month, bucket, f"{silver_prefix}silver_customer_month.parquet", region)

    logger.info("Silver step complete")


def main() -> None:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"),
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Silver ETL step")
    parser.add_argument("--bucket", default=os.environ.get("S3_BUCKET", ""))
    parser.add_argument("--bronze-prefix", default="bronze/")
    parser.add_argument("--silver-prefix", default="silver/")
    parser.add_argument("--region", default=os.environ.get("AWS_REGION", "eu-west-1"))
    args = parser.parse_args()

    if not args.bucket:
        parser.error("--bucket is required (or set S3_BUCKET env var)")

    run_silver_step(args.bucket, args.bronze_prefix, args.silver_prefix, args.region)


if __name__ == "__main__":
    main()

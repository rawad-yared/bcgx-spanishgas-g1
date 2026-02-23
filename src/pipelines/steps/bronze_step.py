"""Bronze ETL step — SageMaker Processing Job entry point."""

from __future__ import annotations

import argparse
import logging
import os

from src.data.ingest import build_bronze_customer, build_bronze_customer_month
from src.pipelines.s3_io import read_csv, read_parquet, write_parquet

logger = logging.getLogger(__name__)

RAW_FILES = {
    "churn": "churn_label.csv",
    "attributes": "customer_attributes.csv",
    "contracts": "customer_contracts.csv",
    "prices": "price_history.csv",
    "costs": "costs_by_province_month.csv",
    "interactions": "customer_interactions.json",
    "consumption": "consumption_hourly_2024.parquet",
}


def run_bronze_step(
    bucket: str,
    raw_prefix: str = "raw/",
    bronze_prefix: str = "bronze/",
    region: str = "eu-west-1",
) -> None:
    """Read raw files from S3, build bronze tables, write parquet."""
    logger.info("Bronze step: bucket=%s raw=%s bronze=%s", bucket, raw_prefix, bronze_prefix)

    # Load raw datasets from S3
    churn = read_csv(bucket, f"{raw_prefix}churn_label.csv", region)
    attributes = read_csv(bucket, f"{raw_prefix}customer_attributes.csv", region)
    contracts = read_csv(bucket, f"{raw_prefix}customer_contracts.csv", region)
    prices = read_csv(bucket, f"{raw_prefix}price_history.csv", region)
    costs = read_csv(bucket, f"{raw_prefix}costs_by_province_month.csv", region)

    # Interactions (JSON) — read as CSV workaround or use read_json_s3
    import pandas as pd

    from src.pipelines.s3_io import read_json_s3
    interactions_data = read_json_s3(bucket, f"{raw_prefix}customer_interactions.json", region)
    interactions = pd.DataFrame(interactions_data) if isinstance(interactions_data, list) else pd.json_normalize(interactions_data)

    # Consumption (parquet)
    consumption = read_parquet(bucket, f"{raw_prefix}consumption_hourly_2024.parquet", region)

    logger.info("Loaded raw datasets: churn=%d, consumption=%d", len(churn), len(consumption))

    # Build bronze tables
    bronze_customer = build_bronze_customer(churn, attributes, contracts, interactions)
    bronze_customer_month = build_bronze_customer_month(
        consumption, prices, costs, province_lookup=attributes,
    )

    logger.info("Bronze customer: %d rows, Bronze customer-month: %d rows",
                len(bronze_customer), len(bronze_customer_month))

    # Write to S3
    write_parquet(bronze_customer, bucket, f"{bronze_prefix}bronze_customer.parquet", region)
    write_parquet(bronze_customer_month, bucket, f"{bronze_prefix}bronze_customer_month.parquet", region)

    logger.info("Bronze step complete")


def main() -> None:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"),
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Bronze ETL step")
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--raw-prefix", default="raw/")
    parser.add_argument("--bronze-prefix", default="bronze/")
    parser.add_argument("--region", default="eu-west-1")
    args = parser.parse_args()

    run_bronze_step(args.bucket, args.raw_prefix, args.bronze_prefix, args.region)


if __name__ == "__main__":
    main()

"""Bronze ETL step — SageMaker Processing Job entry point."""

from __future__ import annotations

import argparse
import logging
import os

import numpy as np
import pandas as pd

from src.data.ingest import (
    GAS_KWH_PER_M3,
    KEY,
    _assign_tariff_tiers,
    build_bronze_customer,
)
from src.pipelines.s3_io import (
    read_csv,
    read_json_s3,
    read_parquet_batches,
    write_parquet,
)

logger = logging.getLogger(__name__)

CONSUMPTION_COLUMNS = [KEY, "timestamp", "consumption_elec_kwh", "consumption_gas_m3"]


def _aggregate_consumption_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a single chunk of hourly consumption to customer x month totals."""
    elec_col = "consumption_elec_kwh"
    gas_col = "consumption_gas_m3"
    ts_col = "timestamp"

    cons = chunk[[KEY, ts_col, elec_col, gas_col]].copy()
    cons[ts_col] = pd.to_datetime(cons[ts_col], errors="coerce")

    # Clean negative consumption
    cons[elec_col] = np.where(cons[elec_col] < 0, 0.0, cons[elec_col])
    cons[gas_col] = np.where(cons[gas_col] < 0, 0.0, cons[gas_col])
    cons["consumption_gas_kwh"] = cons[gas_col] * GAS_KWH_PER_M3

    cons["month"] = cons[ts_col].dt.to_period("M").astype(str)
    cons = _assign_tariff_tiers(cons, ts_col)

    # Tier splits
    for prefix, src in [
        ("elec_kwh", elec_col),
        ("gas_m3", gas_col),
        ("gas_kwh", "consumption_gas_kwh"),
    ]:
        for tier_name in ["tier_1_peak", "tier_2_standard", "tier_3_offpeak"]:
            cons[f"{prefix}_{tier_name}"] = np.where(cons["tier"] == tier_name, cons[src], 0.0)

    # Build aggregation map
    agg_map = {
        f"monthly_{elec_col}": (elec_col, "sum"),
        f"monthly_{gas_col}": (gas_col, "sum"),
        "monthly_gas_kwh": ("consumption_gas_kwh", "sum"),
    }
    for prefix in ["elec_kwh", "gas_m3", "gas_kwh"]:
        for tier_name in ["tier_1_peak", "tier_2_standard", "tier_3_offpeak"]:
            col = f"{prefix}_{tier_name}"
            agg_map[col] = (col, "sum")

    return cons.groupby([KEY, "month"], as_index=False).agg(**agg_map)


def _build_bronze_customer_month_chunked(
    bucket: str,
    raw_prefix: str,
    region: str,
    prices: pd.DataFrame | None = None,
    costs: pd.DataFrame | None = None,
    province_lookup: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build bronze_customer_month by reading consumption in memory-safe batches."""
    elec_col = "consumption_elec_kwh"
    gas_col = "consumption_gas_m3"

    partials: list[pd.DataFrame] = []
    total_rows = 0

    for chunk in read_parquet_batches(
        bucket,
        f"{raw_prefix}consumption_hourly_2024.parquet",
        region,
        columns=CONSUMPTION_COLUMNS,
        batch_size=2_000_000,
    ):
        total_rows += len(chunk)
        logger.info("Processing consumption batch: %d rows (total so far: %d)", len(chunk), total_rows)
        partials.append(_aggregate_consumption_chunk(chunk))

    logger.info("Finished reading consumption: %d total rows in %d batches", total_rows, len(partials))

    # Combine partial monthly aggregations and re-sum
    combined = pd.concat(partials, ignore_index=True)
    sum_cols = [c for c in combined.columns if c not in (KEY, "month")]
    monthly = combined.groupby([KEY, "month"], as_index=False)[sum_cols].sum()

    # Rename for consistency
    monthly = monthly.rename(columns={
        f"monthly_{elec_col}": "monthly_elec_kwh",
        f"monthly_{gas_col}": "monthly_gas_m3",
    })

    monthly = monthly.dropna(subset=[KEY])
    monthly = monthly[monthly[KEY] != ""]

    # Merge prices
    if prices is not None:
        prices_m = prices.copy()
        if "pricing_date" in prices_m.columns:
            prices_m["month"] = pd.to_datetime(
                prices_m["pricing_date"], errors="coerce"
            ).dt.to_period("M").astype(str)
            prices_m = prices_m.drop(columns=["pricing_date"], errors="ignore")
        monthly = monthly.merge(prices_m, on=[KEY, "month"], how="left")

    # Merge province lookup
    if province_lookup is not None:
        prov = province_lookup[[KEY, "province_code"]].drop_duplicates(subset=[KEY])
        monthly = monthly.merge(prov, on=KEY, how="left")

    # Merge costs
    if costs is not None and "province_code" in monthly.columns:
        costs_m = costs.copy()
        costs_m["month"] = costs_m["month"].astype(str).str[:7]
        monthly["month"] = monthly["month"].astype(str).str[:7]
        monthly = monthly.merge(
            costs_m,
            left_on=["province_code", "month"],
            right_on=["province", "month"],
            how="left",
        )
        monthly = monthly.drop(columns=["province"], errors="ignore")
        monthly = monthly.drop(columns=["province_code"], errors="ignore")

    return monthly


def run_bronze_step(
    bucket: str,
    raw_prefix: str = "raw/",
    bronze_prefix: str = "bronze/",
    region: str = "eu-west-1",
) -> None:
    """Read raw files from S3, build bronze tables, write parquet."""
    logger.info("Bronze step: bucket=%s raw=%s bronze=%s", bucket, raw_prefix, bronze_prefix)

    # Load small raw datasets from S3
    churn = read_csv(bucket, f"{raw_prefix}churn_label.csv", region)
    attributes = read_csv(bucket, f"{raw_prefix}customer_attributes.csv", region)
    contracts = read_csv(bucket, f"{raw_prefix}customer_contracts.csv", region)
    prices = read_csv(bucket, f"{raw_prefix}price_history.csv", region)
    costs = read_csv(bucket, f"{raw_prefix}costs_by_province_month.csv", region)

    interactions_data = read_json_s3(bucket, f"{raw_prefix}customer_interactions.json", region)
    interactions = (
        pd.DataFrame(interactions_data)
        if isinstance(interactions_data, list)
        else pd.json_normalize(interactions_data)
    )

    logger.info("Loaded small raw datasets: churn=%d", len(churn))

    # Build bronze customer (small, fits in memory)
    bronze_customer = build_bronze_customer(churn, attributes, contracts, interactions)

    # Build bronze customer-month (chunked — consumption is too large for single load)
    bronze_customer_month = _build_bronze_customer_month_chunked(
        bucket, raw_prefix, region,
        prices=prices, costs=costs, province_lookup=attributes,
    )

    logger.info(
        "Bronze customer: %d rows, Bronze customer-month: %d rows",
        len(bronze_customer), len(bronze_customer_month),
    )

    # Write to S3
    write_parquet(bronze_customer, bucket, f"{bronze_prefix}bronze_customer.parquet", region)
    write_parquet(bronze_customer_month, bucket, f"{bronze_prefix}bronze_customer_month.parquet", region)

    logger.info("Bronze step complete")


def main() -> None:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"),
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Bronze ETL step")
    parser.add_argument("--bucket", default=os.environ.get("S3_BUCKET", ""))
    parser.add_argument("--raw-prefix", default="raw/")
    parser.add_argument("--bronze-prefix", default="bronze/")
    parser.add_argument("--region", default=os.environ.get("AWS_REGION", "eu-west-1"))
    args = parser.parse_args()

    if not args.bucket:
        parser.error("--bucket is required (or set S3_BUCKET env var)")

    run_bronze_step(args.bucket, args.raw_prefix, args.bronze_prefix, args.region)


if __name__ == "__main__":
    main()

"""Dataset contracts for SpanishGas source and gold tables."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable


@dataclass(frozen=True)
class ColumnContract:
    """Column-level schema contract."""

    name: str
    dtype: str


@dataclass(frozen=True)
class DatasetContract:
    """Table-level schema contract."""

    dataset_name: str
    layer: str
    source_uri: str
    columns: tuple[ColumnContract, ...]
    primary_keys: tuple[str, ...]
    time_columns: tuple[str, ...]
    grain: str


SOURCE_CONTRACTS: Dict[str, DatasetContract] = {
    "customer_attributes.csv": DatasetContract(
        dataset_name="customer_attributes.csv",
        layer="source",
        source_uri="s3://spanishgas-data-g1/raw/customer_attributes.csv",
        columns=(
            ColumnContract("customer_id", "string"),
            ColumnContract("province", "string"),
            ColumnContract("customer_type", "string"),
            ColumnContract("tariff_type", "string"),
            ColumnContract("signup_date", "date"),
            ColumnContract("product_bundle", "string"),
        ),
        primary_keys=("customer_id",),
        time_columns=("signup_date",),
        grain="one row per customer",
    ),
    "customer_contracts.csv": DatasetContract(
        dataset_name="customer_contracts.csv",
        layer="source",
        source_uri="s3://spanishgas-data-g1/raw/customer_contracts.csv",
        columns=(
            ColumnContract("contract_id", "string"),
            ColumnContract("customer_id", "string"),
            ColumnContract("contract_start_date", "date"),
            ColumnContract("contract_end_date", "date"),
            ColumnContract("contract_status", "string"),
            ColumnContract("contract_term_months", "int"),
            ColumnContract("product_type", "string"),
        ),
        primary_keys=("contract_id",),
        time_columns=("contract_start_date", "contract_end_date"),
        grain="one row per contract",
    ),
    "price_history.csv": DatasetContract(
        dataset_name="price_history.csv",
        layer="source",
        source_uri="s3://spanishgas-data-g1/raw/price_history.csv",
        columns=(
            ColumnContract("price_date", "date"),
            ColumnContract("product_type", "string"),
            ColumnContract("tariff_type", "string"),
            ColumnContract("region_code", "string"),
            ColumnContract("price_eur_per_kwh", "float"),
            ColumnContract("market_benchmark_eur_per_kwh", "float"),
        ),
        primary_keys=("price_date", "product_type", "tariff_type", "region_code"),
        time_columns=("price_date",),
        grain="daily row per product/tariff/region",
    ),
    "consumption_hourly_2024.csv": DatasetContract(
        dataset_name="consumption_hourly_2024.csv",
        layer="source",
        source_uri="s3://spanishgas-data-g1/raw/consumption_hourly_2024.csv",
        columns=(
            ColumnContract("customer_id", "string"),
            ColumnContract("timestamp_utc", "timestamp"),
            ColumnContract("commodity", "string"),
            ColumnContract("consumption_kwh", "float"),
            ColumnContract("meter_id", "string"),
            ColumnContract("source_system", "string"),
        ),
        primary_keys=("customer_id", "timestamp_utc", "commodity"),
        time_columns=("timestamp_utc",),
        grain="hourly row per customer and commodity",
    ),
    "customer_interactions.json": DatasetContract(
        dataset_name="customer_interactions.json",
        layer="source",
        source_uri="s3://spanishgas-data-g1/raw/customer_interactions.json",
        columns=(
            ColumnContract("interaction_id", "string"),
            ColumnContract("customer_id", "string"),
            ColumnContract("interaction_ts", "timestamp"),
            ColumnContract("channel", "string"),
            ColumnContract("interaction_type", "string"),
            ColumnContract("sentiment_score", "float"),
            ColumnContract("resolution_status", "string"),
            ColumnContract("agent_id", "string"),
        ),
        primary_keys=("interaction_id",),
        time_columns=("interaction_ts",),
        grain="event row per interaction",
    ),
    "costs_by_province_month.csv": DatasetContract(
        dataset_name="costs_by_province_month.csv",
        layer="source",
        source_uri="s3://spanishgas-data-g1/raw/costs_by_province_month.csv",
        columns=(
            ColumnContract("cost_month", "date"),
            ColumnContract("province", "string"),
            ColumnContract("commodity", "string"),
            ColumnContract("variable_cost_eur_per_kwh", "float"),
            ColumnContract("fixed_cost_eur_month", "float"),
            ColumnContract("network_cost_eur_per_kwh", "float"),
        ),
        primary_keys=("cost_month", "province", "commodity"),
        time_columns=("cost_month",),
        grain="monthly row per province and commodity",
    ),
    "churn_label.csv": DatasetContract(
        dataset_name="churn_label.csv",
        layer="source",
        source_uri="s3://spanishgas-data-g1/raw/churn_label.csv",
        columns=(
            ColumnContract("customer_id", "string"),
            ColumnContract("label_date", "date"),
            ColumnContract("horizon_days", "int"),
            ColumnContract("churned_within_horizon", "int"),
            ColumnContract("churn_effective_date", "date"),
        ),
        primary_keys=("customer_id", "label_date", "horizon_days"),
        time_columns=("label_date", "churn_effective_date"),
        grain="one label row per customer, label date, and horizon",
    ),
}


GOLD_CONTRACTS: Dict[str, DatasetContract] = {
    "customer_snapshot_daily": DatasetContract(
        dataset_name="customer_snapshot_daily",
        layer="gold",
        source_uri="data/gold/customer_snapshot_daily/",
        columns=(
            ColumnContract("customer_id", "string"),
            ColumnContract("snapshot_date", "date"),
            ColumnContract("active_contract_count", "int"),
            ColumnContract("days_to_contract_end", "int"),
            ColumnContract("latest_price_eur_per_kwh", "float"),
            ColumnContract("consumption_30d_kwh", "float"),
            ColumnContract("interaction_count_30d", "int"),
        ),
        primary_keys=("customer_id", "snapshot_date"),
        time_columns=("snapshot_date",),
        grain="daily snapshot per customer",
    ),
    "customer_snapshot_monthly": DatasetContract(
        dataset_name="customer_snapshot_monthly",
        layer="gold",
        source_uri="data/gold/customer_snapshot_monthly/",
        columns=(
            ColumnContract("customer_id", "string"),
            ColumnContract("snapshot_month", "date"),
            ColumnContract("monthly_consumption_kwh", "float"),
            ColumnContract("avg_price_eur_per_kwh", "float"),
            ColumnContract("monthly_interaction_count", "int"),
            ColumnContract("active_contract_count", "int"),
        ),
        primary_keys=("customer_id", "snapshot_month"),
        time_columns=("snapshot_month",),
        grain="monthly snapshot per customer",
    ),
    "customer_features_asof_date": DatasetContract(
        dataset_name="customer_features_asof_date",
        layer="gold",
        source_uri="data/gold/customer_features_asof_date/asof_date=YYYY-MM-DD/",
        columns=(
            ColumnContract("customer_id", "string"),
            ColumnContract("asof_date", "date"),
            ColumnContract("feature_version", "string"),
            ColumnContract("tenure_days", "int"),
            ColumnContract("days_to_contract_end", "int"),
            ColumnContract("price_vs_benchmark_delta", "float"),
            ColumnContract("consumption_volatility_90d", "float"),
            ColumnContract("interaction_count_90d", "int"),
            ColumnContract("negative_consumption_flag", "int"),
        ),
        primary_keys=("customer_id", "asof_date", "feature_version"),
        time_columns=("asof_date",),
        grain="as-of snapshot per customer",
    ),
    "churn_training_dataset": DatasetContract(
        dataset_name="churn_training_dataset",
        layer="gold",
        source_uri="data/gold/churn_training_dataset/",
        columns=(
            ColumnContract("customer_id", "string"),
            ColumnContract("asof_date", "date"),
            ColumnContract("label_horizon_days", "int"),
            ColumnContract("churn_label", "int"),
            ColumnContract("split", "string"),
            ColumnContract("feature_version", "string"),
        ),
        primary_keys=("customer_id", "asof_date", "label_horizon_days"),
        time_columns=("asof_date",),
        grain="one training row per customer and as-of date",
    ),
    "recommendation_candidates": DatasetContract(
        dataset_name="recommendation_candidates",
        layer="gold",
        source_uri="data/gold/recommendation_candidates/",
        columns=(
            ColumnContract("customer_id", "string"),
            ColumnContract("run_date", "date"),
            ColumnContract("risk_score", "float"),
            ColumnContract("segment_id", "string"),
            ColumnContract("recommended_action", "string"),
            ColumnContract("timing_window", "string"),
            ColumnContract("expected_margin_impact_eur", "float"),
            ColumnContract("reason_codes", "array<string>"),
        ),
        primary_keys=("customer_id", "run_date"),
        time_columns=("run_date",),
        grain="one recommendation row per customer and run date",
    ),
}


def get_source_contracts() -> Dict[str, DatasetContract]:
    """Return source contracts keyed by dataset name."""

    return dict(SOURCE_CONTRACTS)


def get_gold_contracts() -> Dict[str, DatasetContract]:
    """Return gold contracts keyed by dataset name."""

    return dict(GOLD_CONTRACTS)


def load_contracts() -> Dict[str, DatasetContract]:
    """Load all contracts keyed by dataset name."""

    contracts = dict(SOURCE_CONTRACTS)
    contracts.update(GOLD_CONTRACTS)
    return contracts


def _format_columns(columns: Iterable[ColumnContract]) -> str:
    return ", ".join(f"{column.name}:{column.dtype}" for column in columns)


def render_contracts() -> str:
    """Render contracts as plain text for CLI verification."""

    contracts = load_contracts()
    lines = []
    for dataset_name in sorted(contracts):
        contract = contracts[dataset_name]
        lines.append(f"{contract.layer.upper()} | {contract.dataset_name}")
        lines.append(f"  source_uri: {contract.source_uri}")
        lines.append(f"  primary_keys: {', '.join(contract.primary_keys)}")
        lines.append(f"  time_columns: {', '.join(contract.time_columns)}")
        lines.append(f"  grain: {contract.grain}")
        lines.append(f"  columns: {_format_columns(contract.columns)}")
        lines.append("")
    return "\n".join(lines).rstrip()


def main() -> None:
    """Entry point for ticket verify step."""

    print(render_contracts())


if __name__ == "__main__":
    main()

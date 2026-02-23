"""Phase 1A: Data ingestion — load raw datasets and build bronze tables."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ── Constants ────────────────────────────────────────────────────────────────

RAW_FILES = {
    "churn": "churn_label.csv",
    "attributes": "customer_attributes.csv",
    "contracts": "customer_contracts.csv",
    "prices": "price_history.csv",
    "costs": "costs_by_province_month.csv",
    "interactions": "customer_interactions.json",
}

SPANISH_PUBLIC_HOLIDAYS_MD = {101, 106, 501, 815, 1012, 1101, 1206, 1208, 1225}

KEY = "customer_id"
GAS_KWH_PER_M3 = 11.0


# ── Raw loading ──────────────────────────────────────────────────────────────


def load_raw_datasets(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Load all 7 raw datasets from *data_dir* and return as a dict."""
    data_dir = Path(data_dir)
    dfs: dict[str, pd.DataFrame] = {}
    for key, filename in RAW_FILES.items():
        path = data_dir / filename
        if filename.endswith(".json"):
            dfs[key] = pd.read_json(path)
        else:
            dfs[key] = pd.read_csv(path)
    return dfs


def load_or_convert_consumption(data_dir: str | Path) -> pd.DataFrame:
    """Load consumption data, preferring parquet over CSV for speed."""
    data_dir = Path(data_dir)
    parquet_path = data_dir / "consumption_hourly_2024.parquet"
    csv_gz_path = data_dir / "consumption_hourly_2024.csv.gz"
    csv_path = data_dir / "consumption_hourly_2024.csv"

    if parquet_path.exists():
        return pd.read_parquet(parquet_path)

    source_path = csv_gz_path if csv_gz_path.exists() else (csv_path if csv_path.exists() else None)
    if source_path is None:
        raise FileNotFoundError("Consumption file not found in " + str(data_dir))

    df = pd.read_csv(source_path)
    df.to_parquet(parquet_path, index=False)
    return df


# ── Bronze: customer-level (1 row per customer) ─────────────────────────────


def build_bronze_customer(
    churn: pd.DataFrame,
    attributes: pd.DataFrame,
    contracts: pd.DataFrame,
    interactions: pd.DataFrame,
) -> pd.DataFrame:
    """Merge churn + attributes + contracts + interactions → 1 row / customer."""
    if churn.duplicated(subset=[KEY]).any():
        raise ValueError("churn has duplicate customer_id values")

    attr_1 = attributes.drop_duplicates(subset=[KEY], keep="first")
    con_1 = contracts.drop_duplicates(subset=[KEY], keep="first")
    int_1 = interactions.drop_duplicates(subset=[KEY], keep="first")

    bronze = (
        churn.copy()
        .merge(attr_1, on=KEY, how="left", validate="one_to_one")
        .merge(con_1, on=KEY, how="left", validate="one_to_one")
        .merge(int_1, on=KEY, how="left", validate="one_to_one")
    )

    if bronze.duplicated(subset=[KEY]).any():
        raise ValueError("bronze_customer has duplicate rows after merge")
    return bronze


# ── Bronze: customer-month level ─────────────────────────────────────────────


def _assign_tariff_tiers(cons: pd.DataFrame, ts_col: str = "timestamp") -> pd.DataFrame:
    """Add tariff tier column (peak/standard/offpeak) based on Spanish PVPC rules."""
    ts = cons[ts_col]
    h = ts.dt.hour.to_numpy(dtype="int8")
    weekday = ts.dt.weekday.to_numpy(dtype="int8")

    md_int = (ts.dt.month.to_numpy(dtype="int16") * 100 + ts.dt.day.to_numpy(dtype="int16"))
    is_holiday = np.isin(md_int, list(SPANISH_PUBLIC_HOLIDAYS_MD))
    is_weekend = np.isin(weekday, [5, 6])

    is_weekday_nonholiday = ~is_weekend & ~is_holiday

    peak = is_weekday_nonholiday & (((h >= 10) & (h < 14)) | ((h >= 18) & (h < 22)))
    standard = is_weekday_nonholiday & (
        ((h >= 8) & (h < 10)) | ((h >= 14) & (h < 18)) | ((h >= 22) & (h < 24))
    )

    tier = np.full(len(cons), "tier_3_offpeak", dtype=object)
    tier[standard] = "tier_2_standard"
    tier[peak] = "tier_1_peak"

    cons = cons.copy()
    cons["tier"] = pd.Categorical(tier)
    return cons


def build_bronze_customer_month(
    consumption: pd.DataFrame,
    prices: pd.DataFrame | None = None,
    costs: pd.DataFrame | None = None,
    province_lookup: pd.DataFrame | None = None,
    elec_col: str = "consumption_elec_kwh",
    gas_col: str = "consumption_gas_m3",
    ts_col: str = "timestamp",
) -> pd.DataFrame:
    """Aggregate hourly consumption to monthly, merge prices & costs.

    Steps:
      1. Clean negative consumption → 0
      2. Assign Spanish tariff tiers
      3. Aggregate to customer × month
      4. Merge prices and costs if provided
    """
    cons = consumption[[KEY, ts_col, elec_col, gas_col]].copy()
    cons[ts_col] = pd.to_datetime(cons[ts_col], errors="coerce")

    # Clean negative consumption
    cons[elec_col] = np.where(cons[elec_col] < 0, 0.0, cons[elec_col])
    cons[gas_col] = np.where(cons[gas_col] < 0, 0.0, cons[gas_col])

    # Gas kWh conversion
    cons["consumption_gas_kwh"] = cons[gas_col] * GAS_KWH_PER_M3

    # Month
    cons["month"] = cons[ts_col].dt.to_period("M").astype(str)

    # Tariff tiers
    cons = _assign_tariff_tiers(cons, ts_col)

    # Tier splits (vectorized for speed)
    for prefix, src in [("elec_kwh", elec_col), ("gas_m3", gas_col), ("gas_kwh", "consumption_gas_kwh")]:
        for tier_name in ["tier_1_peak", "tier_2_standard", "tier_3_offpeak"]:
            cons[f"{prefix}_{tier_name}"] = np.where(cons["tier"] == tier_name, cons[src], 0.0)

    # Monthly aggregation
    agg_map = {
        f"monthly_{elec_col}": (elec_col, "sum"),
        f"monthly_{gas_col}": (gas_col, "sum"),
        "monthly_gas_kwh": ("consumption_gas_kwh", "sum"),
    }
    for prefix in ["elec_kwh", "gas_m3", "gas_kwh"]:
        for tier_name in ["tier_1_peak", "tier_2_standard", "tier_3_offpeak"]:
            col = f"{prefix}_{tier_name}"
            agg_map[col] = (col, "sum")

    monthly = cons.groupby([KEY, "month"], as_index=False).agg(**agg_map)

    # Rename for consistency
    monthly = monthly.rename(columns={
        f"monthly_{elec_col}": "monthly_elec_kwh",
        f"monthly_{gas_col}": "monthly_gas_m3",
    })

    # Drop missing customer IDs
    monthly = monthly.dropna(subset=[KEY])
    monthly = monthly[monthly[KEY] != ""]

    # Merge prices if provided
    if prices is not None:
        prices_m = prices.copy()
        if "pricing_date" in prices_m.columns:
            prices_m["month"] = pd.to_datetime(prices_m["pricing_date"], errors="coerce").dt.to_period("M").astype(str)
            prices_m = prices_m.drop(columns=["pricing_date"], errors="ignore")
        monthly = monthly.merge(prices_m, on=[KEY, "month"], how="left")

    # Merge province lookup if provided
    if province_lookup is not None:
        prov = province_lookup[[KEY, "province_code"]].drop_duplicates(subset=[KEY])
        monthly = monthly.merge(prov, on=KEY, how="left")

    # Merge costs if provided
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

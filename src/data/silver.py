"""Phase 1B: Silver transforms — cleaning, imputation, segmentation, margins."""

from __future__ import annotations

import numpy as np
import pandas as pd

KEY = "customer_id"

# ── Sales channel Spanish → English mapping ──────────────────────────────────

CHANNEL_MAP = {
    "presencial_comercial": "In-Person Commercial",
    "comparador": "Comparison Website",
    "oficina": "Office",
    "telemarketing": "Telemarketing",
    "web_propia": "Own Website",
    "desconocido": "Unknown",
    "unknown": "Unknown",
}


# ── Price imputation ─────────────────────────────────────────────────────────


def impute_prices_hierarchical(
    scm: pd.DataFrame,
    silver_customer: pd.DataFrame,
    segment_col: str = "is_industrial",
) -> pd.DataFrame:
    """3-level hierarchical price imputation.

    For each price column where the corresponding consumption exists:
      1. Customer ffill/bfill
      2. Segment × month median
      3. National month median
    """
    scm = scm.copy()
    month_col = "month"

    # Bring segment if missing
    if segment_col not in scm.columns:
        seg_lookup = silver_customer[[KEY, segment_col]].drop_duplicates(subset=[KEY])
        scm = scm.merge(seg_lookup, on=KEY, how="left", validate="many_to_one")

    # Parse month to datetime for sorting
    scm[month_col] = pd.to_datetime(scm[month_col].astype(str).str[:7] + "-01", errors="coerce")
    scm = scm.sort_values([KEY, month_col])

    # Price columns and their corresponding consumption columns
    price_consumption_pairs = [
        ("variable_price_tier1_eur_kwh", "elec_kwh_tier_1_peak"),
        ("variable_price_tier2_eur_kwh", "elec_kwh_tier_2_standard"),
        ("variable_price_tier3_eur_kwh", "elec_kwh_tier_3_offpeak"),
        ("gas_variable_price_eur_m3", "monthly_gas_m3"),
        ("elec_fixed_fee_eur_month", "monthly_elec_kwh"),
        ("gas_fixed_revenue_eur_year", "monthly_gas_m3"),
    ]

    for price_col, cons_col in price_consumption_pairs:
        if price_col not in scm.columns or cons_col not in scm.columns:
            continue

        scm[price_col] = pd.to_numeric(scm[price_col], errors="coerce")
        scm.loc[scm[price_col] == 0, price_col] = np.nan

        mask = scm[cons_col].fillna(0) > 0

        _impute_single_column(scm, price_col, mask, segment_col, month_col)

    # Convert month back to YYYY-MM string
    scm[month_col] = scm[month_col].dt.to_period("M").astype(str)
    return scm


def _impute_single_column(
    scm: pd.DataFrame,
    col: str,
    mask: pd.Series,
    seg_col: str,
    month_col: str,
) -> None:
    """In-place 3-level imputation for a single price column."""
    # Level 1: customer ffill/bfill
    filled = (
        scm.groupby(KEY, sort=False)[col]
        .apply(lambda s: s.ffill().bfill())
        .reset_index(level=0, drop=True)
    )
    need = mask & scm[col].isna()
    if need.any():
        scm.loc[need, col] = filled.loc[need]

    # Level 2: segment × month median
    need2 = mask & scm[col].isna()
    if need2.any():
        seg_month_med = scm.loc[mask].groupby([seg_col, month_col], sort=False)[col].median()
        idx = list(zip(scm.loc[need2, seg_col], scm.loc[need2, month_col]))
        scm.loc[need2, col] = [seg_month_med.get(k, np.nan) for k in idx]

    # Level 3: national month median
    need3 = mask & scm[col].isna()
    if need3.any():
        nat_med = scm.loc[mask].groupby(month_col, sort=False)[col].median()
        scm.loc[need3, col] = scm.loc[need3, month_col].map(nat_med)


# ── Customer segmentation ────────────────────────────────────────────────────


def derive_customer_segments(sc: pd.DataFrame) -> pd.DataFrame:
    """Derive segment (Residential/SME/Corporate) and residential_type."""
    sc = sc.copy()
    sc["segment"] = None

    # Residential: not industrial
    sc.loc[sc["is_industrial"] == 0, "segment"] = "Residential"

    # Industrial: SME if contracted_power_kw == 10, Corporate if > 10
    sc.loc[
        (sc["is_industrial"] == 1) & (sc["contracted_power_kw"] == 10),
        "segment",
    ] = "SME"
    sc.loc[
        (sc["is_industrial"] == 1) & (sc["contracted_power_kw"] > 10),
        "segment",
    ] = "Corporate"

    # Residential sub-type
    sc["residential_type"] = None
    sc.loc[
        (sc["segment"] == "Residential") & (sc["is_second_residence"] == 1),
        "residential_type",
    ] = "Second_Residence"
    sc.loc[
        (sc["segment"] == "Residential") & (sc["is_second_residence"] == 0),
        "residential_type",
    ] = "Primary_Residence"

    return sc


# ── Sales channel cleaning ───────────────────────────────────────────────────


def clean_sales_channels(sc: pd.DataFrame) -> pd.DataFrame:
    """Translate Spanish sales channel names to English."""
    sc = sc.copy()
    if "sales_channel" not in sc.columns:
        return sc

    cleaned = sc["sales_channel"].astype(str).str.strip().str.lower()
    sc["sales_channel"] = cleaned.map(CHANNEL_MAP).fillna(cleaned)
    return sc


# ── Margin computation ───────────────────────────────────────────────────────


def compute_margins(scm: pd.DataFrame) -> pd.DataFrame:
    """Compute electricity margin, gas margin, and total margin per customer-month."""
    df = scm.copy()

    # ── Electricity P&L ──
    elec_cols = [
        "elec_kwh_tier_1_peak", "elec_kwh_tier_2_standard", "elec_kwh_tier_3_offpeak",
        "variable_price_tier1_eur_kwh", "variable_price_tier2_eur_kwh", "variable_price_tier3_eur_kwh",
        "elec_fixed_fee_eur_month", "monthly_elec_kwh",
        "elec_var_cost_eur_kwh", "peaje_elec_eur_kwh", "elec_fixed_cost_eur_month",
    ]
    for c in elec_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    df["elec_revenue_variable"] = (
        df.get("elec_kwh_tier_1_peak", 0) * df.get("variable_price_tier1_eur_kwh", 0)
        + df.get("elec_kwh_tier_2_standard", 0) * df.get("variable_price_tier2_eur_kwh", 0)
        + df.get("elec_kwh_tier_3_offpeak", 0) * df.get("variable_price_tier3_eur_kwh", 0)
    )
    df["elec_revenue_fixed"] = df.get("elec_fixed_fee_eur_month", 0)
    df["elec_cost_variable"] = df.get("monthly_elec_kwh", 0) * (
        df.get("elec_var_cost_eur_kwh", 0) + df.get("peaje_elec_eur_kwh", 0)
    )
    df["elec_cost_fixed"] = df.get("elec_fixed_cost_eur_month", 0)
    df["elec_margin"] = (
        df["elec_revenue_variable"] + df["elec_revenue_fixed"]
        - df["elec_cost_variable"] - df["elec_cost_fixed"]
    )

    # ── Gas P&L ──
    gas_cols = [
        "monthly_gas_m3", "gas_variable_price_eur_m3",
        "gas_fixed_revenue_eur_year", "gas_var_cost_eur_m3", "gas_fixed_cost_eur_year",
    ]
    for c in gas_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    df["gas_revenue_variable"] = df.get("monthly_gas_m3", 0) * df.get("gas_variable_price_eur_m3", 0)
    df["gas_revenue_fixed"] = df.get("gas_fixed_revenue_eur_year", 0) / 12
    df["gas_cost_variable"] = df.get("monthly_gas_m3", 0) * df.get("gas_var_cost_eur_m3", 0)
    df["gas_cost_fixed"] = df.get("gas_fixed_cost_eur_year", 0) / 12
    df["gas_margin"] = (
        df["gas_revenue_variable"] + df["gas_revenue_fixed"]
        - df["gas_cost_variable"] - df["gas_cost_fixed"]
    )

    # ── Totals ──
    df["total_revenue"] = (
        df["elec_revenue_variable"] + df["elec_revenue_fixed"]
        + df["gas_revenue_variable"] + df["gas_revenue_fixed"]
    )
    df["total_cost"] = (
        df["elec_cost_variable"] + df["elec_cost_fixed"]
        + df["gas_cost_variable"] + df["gas_cost_fixed"]
    )
    df["total_margin"] = df["total_revenue"] - df["total_cost"]

    return df


# ── Silver orchestration ─────────────────────────────────────────────────────


def build_silver_tables(
    bronze_customer: pd.DataFrame,
    bronze_customer_month: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build silver_customer and silver_customer_month from bronze tables."""
    silver_customer = bronze_customer.copy()
    silver_customer_month = bronze_customer_month.copy()

    # Customer-level transforms
    silver_customer = derive_customer_segments(silver_customer)
    silver_customer = clean_sales_channels(silver_customer)

    # Price imputation
    silver_customer_month = impute_prices_hierarchical(silver_customer_month, silver_customer)

    # Margin computation
    silver_customer_month = compute_margins(silver_customer_month)

    return silver_customer, silver_customer_month

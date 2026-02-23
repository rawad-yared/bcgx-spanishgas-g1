"""Phase 1C: Feature engineering — build gold master from silver tables."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

KEY = "customer_id"


# ── Tier 1A: Lifecycle + stickiness + dual fuel ─────────────────────────────


def build_lifecycle_features(
    silver_customer: pd.DataFrame,
    silver_customer_month: pd.DataFrame,
    as_of_date: datetime | None = None,
) -> pd.DataFrame:
    """Tier 1A — lifecycle timing, structural stickiness, dual fuel flag."""
    sc = silver_customer.copy()
    scm = silver_customer_month.copy()

    if as_of_date is None:
        as_of_date = pd.Timestamp.now()
    as_of = pd.Timestamp(as_of_date)

    features = sc[[KEY]].drop_duplicates().reset_index(drop=True)

    # Tenure
    if "customer_first_activation_date" in sc.columns:
        act = pd.to_datetime(sc["customer_first_activation_date"], errors="coerce")
        sc["tenure_months"] = ((as_of - act).dt.days / 30.44).round(0)
    elif "contract_start_date" in sc.columns:
        act = pd.to_datetime(sc["contract_start_date"], errors="coerce")
        sc["tenure_months"] = ((as_of - act).dt.days / 30.44).round(0)

    # Months to renewal
    if "next_renewal_date" in sc.columns:
        renewal = pd.to_datetime(sc["next_renewal_date"], errors="coerce")
        sc["months_to_renewal"] = ((renewal - as_of).dt.days / 30.44).round(1)

    # Renewal bucket
    if "months_to_renewal" in sc.columns:
        bins = [-np.inf, 0, 1, 3, 6, 12, np.inf]
        labels = ["expired", "0-1m", "1-3m", "3-6m", "6-12m", "12m+"]
        sc["renewal_bucket"] = pd.cut(sc["months_to_renewal"], bins=bins, labels=labels)
        sc["is_within_3m_of_renewal"] = (sc["months_to_renewal"].fillna(999) <= 3).astype("Int64")

    # Tenure bucket
    if "tenure_months" in sc.columns:
        bins_t = [0, 6, 12, 24, 60, np.inf]
        labels_t = ["0-6m", "6-12m", "1-2y", "2-5y", "5y+"]
        sc["tenure_bucket"] = pd.cut(sc["tenure_months"], bins=bins_t, labels=labels_t)

    # Merge lifecycle cols
    lifecycle_cols = [
        c for c in [
            KEY, "months_to_renewal", "renewal_bucket", "is_within_3m_of_renewal",
            "tenure_months", "tenure_bucket",
        ] if c in sc.columns
    ]
    features = features.merge(sc[lifecycle_cols].drop_duplicates(KEY), on=KEY, how="left")

    # Stickiness from silver_customer
    stickiness = [
        c for c in ["segment", "sales_channel", "is_high_competition_province",
                     "has_interaction", "is_second_residence"]
        if c in sc.columns
    ]
    if stickiness:
        features = features.merge(
            sc[[KEY] + stickiness].drop_duplicates(KEY), on=KEY, how="left"
        )

    # Expired contract flag
    if "renewal_bucket" in features.columns:
        features["is_expired_contract"] = (
            features["renewal_bucket"].astype("string") == "expired"
        ).astype("Int64")

    # Channel flags
    if "sales_channel" in features.columns:
        features["is_comparison_channel"] = (
            features["sales_channel"].astype("string") == "Comparison Website"
        ).astype("Int64")
        features["is_own_website_channel"] = (
            features["sales_channel"].astype("string") == "Own Website"
        ).astype("Int64")

    # Dual fuel
    fuel = scm.groupby(KEY, as_index=False).agg(
        total_elec=("monthly_elec_kwh", "sum"),
        total_gas=("monthly_gas_m3", "sum"),
    )
    fuel["is_dual_fuel"] = ((fuel["total_elec"] > 0) & (fuel["total_gas"] > 0)).astype("Int64")
    features = features.merge(fuel[[KEY, "is_dual_fuel"]], on=KEY, how="left")
    features["is_dual_fuel"] = features["is_dual_fuel"].fillna(0).astype("Int64")

    return features


# ── Tier MP_Core: Market & portfolio core ────────────────────────────────────


def build_market_core_features(
    silver_customer_month: pd.DataFrame,
    silver_customer: pd.DataFrame,
) -> pd.DataFrame:
    """Tier MP_Core — consumption averages, margins, portfolio type."""
    scm = silver_customer_month.copy()
    sc = silver_customer.copy()

    agg = scm.groupby(KEY, as_index=False).agg(
        avg_monthly_elec_kwh=("monthly_elec_kwh", "mean"),
        total_elec_kwh_2024=("monthly_elec_kwh", "sum"),
        avg_monthly_gas_m3=("monthly_gas_m3", "mean"),
        total_gas_m3_2024=("monthly_gas_m3", "sum"),
    )

    # Margin features
    if "total_margin" in scm.columns:
        margin_agg = scm.groupby(KEY, as_index=False).agg(
            avg_monthly_margin=("total_margin", "mean"),
            total_margin_2024=("total_margin", "sum"),
        )
        agg = agg.merge(margin_agg, on=KEY, how="left")

    # Digital channel flag
    if "sales_channel" in sc.columns:
        sc_ch = sc[[KEY, "sales_channel", "segment"]].drop_duplicates(KEY)
        sc_ch["is_digital_channel"] = (
            sc_ch["sales_channel"].astype(str).isin(["Comparison Website", "Own Website"])
        ).astype("Int64")
        agg = agg.merge(sc_ch[[KEY, "is_digital_channel", "segment"]], on=KEY, how="left")

    # Dual fuel + portfolio type
    fuel = scm.groupby(KEY, as_index=False).agg(_total_gas=("monthly_gas_m3", "sum"))
    fuel["is_dual_fuel"] = (fuel["_total_gas"] > 0).astype("Int64")

    if "segment" in agg.columns:
        fuel = fuel.merge(agg[[KEY, "segment"]], on=KEY, how="left")
        fuel["portfolio_type"] = (
            fuel["segment"].astype(str)
            + "_"
            + fuel["is_dual_fuel"].map({1: "DualFuel", 0: "SingleFuel"}).fillna("Unknown")
        )
        agg = agg.merge(fuel[[KEY, "is_dual_fuel", "portfolio_type"]], on=KEY, how="left")
    else:
        agg = agg.merge(fuel[[KEY, "is_dual_fuel"]], on=KEY, how="left")

    # Gas share of revenue
    if "total_revenue" in scm.columns and "gas_revenue_variable" in scm.columns:
        rev = scm.groupby(KEY, as_index=False).agg(
            _total_rev=("total_revenue", "sum"),
            _gas_rev=("gas_revenue_variable", "sum"),
        )
        rev["gas_share_of_revenue"] = np.where(
            rev["_total_rev"] > 0, rev["_gas_rev"] / rev["_total_rev"], 0.0
        )
        agg = agg.merge(rev[[KEY, "gas_share_of_revenue"]], on=KEY, how="left")

    # Provincial costs
    for cost_col, out_col in [
        ("elec_var_cost_eur_kwh", "province_avg_elec_cost_2024"),
        ("gas_var_cost_eur_m3", "province_avg_gas_cost_2024"),
    ]:
        if cost_col in scm.columns:
            cost_agg = scm.groupby(KEY, as_index=False).agg(**{out_col: (cost_col, "mean")})
            agg = agg.merge(cost_agg, on=KEY, how="left")

    # Price update count
    if "variable_price_tier1_eur_kwh" in scm.columns:
        p = scm.sort_values([KEY, "month"])
        p["_price_changed"] = p.groupby(KEY)["variable_price_tier1_eur_kwh"].diff().abs() > 0.001
        price_count = p.groupby(KEY, as_index=False).agg(price_update_count=("_price_changed", "sum"))
        agg = agg.merge(price_count, on=KEY, how="left")

    return agg


# ── Tier MP_Risk: Volatility, trends, margin stability ──────────────────────


def build_market_risk_features(
    silver_customer_month: pd.DataFrame,
) -> pd.DataFrame:
    """Tier MP_Risk — consumption volatility, price trends, margin stability."""
    scm = silver_customer_month.copy()

    risk = scm[[KEY]].drop_duplicates().reset_index(drop=True)

    # Consumption volatility (CV)
    cons_stats = scm.groupby(KEY).agg(
        _elec_mean=("monthly_elec_kwh", "mean"),
        _elec_std=("monthly_elec_kwh", "std"),
        _gas_mean=("monthly_gas_m3", "mean"),
        _gas_std=("monthly_gas_m3", "std"),
    ).reset_index()

    cons_stats["elec_consumption_volatility"] = np.where(
        cons_stats["_elec_mean"] > 0,
        cons_stats["_elec_std"] / cons_stats["_elec_mean"],
        0.0,
    )
    cons_stats["gas_consumption_volatility"] = np.where(
        cons_stats["_gas_mean"] > 0,
        cons_stats["_gas_std"] / cons_stats["_gas_mean"],
        0.0,
    )
    risk = risk.merge(
        cons_stats[[KEY, "elec_consumption_volatility", "gas_consumption_volatility"]],
        on=KEY, how="left",
    )

    # Price trend (slope of monthly prices via simple diff)
    if "variable_price_tier1_eur_kwh" in scm.columns:
        sorted_scm = scm.sort_values([KEY, "month"])
        price_trend = sorted_scm.groupby(KEY).agg(
            _first_price=("variable_price_tier1_eur_kwh", "first"),
            _last_price=("variable_price_tier1_eur_kwh", "last"),
            _n_months=("month", "count"),
        ).reset_index()
        price_trend["elec_price_trend"] = np.where(
            price_trend["_n_months"] > 1,
            (price_trend["_last_price"] - price_trend["_first_price"]) / price_trend["_n_months"],
            0.0,
        )
        risk = risk.merge(price_trend[[KEY, "elec_price_trend"]], on=KEY, how="left")

    if "gas_variable_price_eur_m3" in scm.columns:
        sorted_scm = scm.sort_values([KEY, "month"])
        gas_trend = sorted_scm.groupby(KEY).agg(
            _first=("gas_variable_price_eur_m3", "first"),
            _last=("gas_variable_price_eur_m3", "last"),
            _n=("month", "count"),
        ).reset_index()
        gas_trend["gas_price_trend"] = np.where(
            gas_trend["_n"] > 1,
            (gas_trend["_last"] - gas_trend["_first"]) / gas_trend["_n"],
            0.0,
        )
        risk = risk.merge(gas_trend[[KEY, "gas_price_trend"]], on=KEY, how="left")

    # Margin stability (std of total_margin)
    if "total_margin" in scm.columns:
        margin_std = scm.groupby(KEY, as_index=False).agg(
            elec_margin_stability=("elec_margin", "std") if "elec_margin" in scm.columns else ("total_margin", "std"),
            gas_margin_stability=("gas_margin", "std") if "gas_margin" in scm.columns else ("total_margin", "std"),
            total_margin_avg=("total_margin", "mean"),
        )
        risk = risk.merge(margin_std, on=KEY, how="left")

    return risk


# ── Tier 2A: Behavioral features ────────────────────────────────────────────


def build_behavioral_features(
    silver_customer: pd.DataFrame,
    as_of_date: datetime | None = None,
) -> pd.DataFrame:
    """Tier 2A — interaction counts, complaints, intent, recency."""
    sc = silver_customer.copy()
    if as_of_date is None:
        as_of_date = pd.Timestamp.now()
    as_of = pd.Timestamp(as_of_date)

    behav = sc[[KEY]].drop_duplicates().reset_index(drop=True)

    # Has interaction
    if "has_interaction" in sc.columns:
        behav = behav.merge(
            sc[[KEY, "has_interaction"]].drop_duplicates(KEY), on=KEY, how="left"
        )

    # Intent
    if "customer_intent" in sc.columns:
        behav = behav.merge(
            sc[[KEY, "customer_intent"]].drop_duplicates(KEY), on=KEY, how="left"
        )
        behav["intent_to_cancel"] = (
            behav["customer_intent"].astype("string").str.lower().str.contains("cancel", na=False)
        ).astype("Int64")

    # Complaints
    if "interaction_summary" in sc.columns:
        summary = sc[[KEY, "interaction_summary"]].drop_duplicates(KEY)
        summary["has_complaint"] = (
            summary["interaction_summary"].astype("string").str.lower().str.contains("complaint", na=False)
        ).astype("Int64")
        behav = behav.merge(summary[[KEY, "has_complaint"]], on=KEY, how="left")

    # Last interaction days ago
    if "date" in sc.columns:
        dates = sc[[KEY, "date"]].copy()
        dates["date"] = pd.to_datetime(dates["date"], errors="coerce")
        latest = dates.groupby(KEY, as_index=False).agg(last_date=("date", "max"))
        latest["last_interaction_days_ago"] = (as_of - latest["last_date"]).dt.days
        behav = behav.merge(latest[[KEY, "last_interaction_days_ago"]], on=KEY, how="left")

    return behav


# ── Tier 2B: Sentiment features ─────────────────────────────────────────────


def build_sentiment_features(silver_customer: pd.DataFrame) -> pd.DataFrame:
    """Tier 2B — sentiment label, negative sentiment flag, avg sentiment."""
    sc = silver_customer.copy()
    sent = sc[[KEY]].drop_duplicates().reset_index(drop=True)

    for col in ["sentiment_label", "sentiment_neg", "sentiment_pos", "sentiment_neu"]:
        if col in sc.columns:
            sent = sent.merge(sc[[KEY, col]].drop_duplicates(KEY), on=KEY, how="left")

    if "sentiment_label" in sent.columns:
        sent["has_negative_sentiment"] = (
            sent["sentiment_label"].astype("string").str.lower() == "negative"
        ).astype("Int64")

    if "sentiment_neg" in sent.columns:
        sent["avg_sentiment_score"] = (
            sent.get("sentiment_pos", 0) - sent.get("sentiment_neg", 0)
        )

    return sent


# ── Tier 3: Compound interaction features ────────────────────────────────────


def build_compound_features(
    features: pd.DataFrame,
) -> pd.DataFrame:
    """Tier 3 — cross-tier interaction flags."""
    df = features.copy()

    def _safe_cross(a: str, b: str) -> pd.Series:
        return (
            df[a].astype("string").fillna("Unknown")
            + "_x_"
            + df[b].astype("string").fillna("Unknown")
        )

    # Interaction string features
    if "customer_intent" in df.columns and "renewal_bucket" in df.columns:
        df["intent_x_renewal_bucket"] = _safe_cross("customer_intent", "renewal_bucket")

    if "customer_intent" in df.columns and "tenure_bucket" in df.columns:
        df["intent_x_tenure_bucket"] = _safe_cross("customer_intent", "tenure_bucket")

    if "sentiment_label" in df.columns and "renewal_bucket" in df.columns:
        df["sentiment_x_renewal_bucket"] = _safe_cross("sentiment_label", "renewal_bucket")

    if "customer_intent" in df.columns and "sentiment_label" in df.columns:
        df["intent_x_sentiment"] = _safe_cross("customer_intent", "sentiment_label")

    if "tenure_bucket" in df.columns and "renewal_bucket" in df.columns:
        df["tenure_x_renewal_bucket"] = _safe_cross("tenure_bucket", "renewal_bucket")

    # Binary interaction flags
    if "is_within_3m_of_renewal" in df.columns and "has_complaint" in df.columns:
        df["renewal_x_complaint"] = (
            (df["is_within_3m_of_renewal"].fillna(0) == 1)
            & (df["has_complaint"].fillna(0) == 1)
        ).astype("Int64")

    if "has_negative_sentiment" in df.columns and "is_within_3m_of_renewal" in df.columns:
        df["high_risk_x_negative_sentiment"] = (
            (df["is_within_3m_of_renewal"].fillna(0) == 1)
            & (df["has_negative_sentiment"].fillna(0) == 1)
        ).astype("Int64")

    # Price sensitivity
    if "customer_intent" in df.columns:
        df["is_price_sensitive"] = (
            df["customer_intent"].astype("string") == "Pricing Offers"
        ).astype("Int64")

    return df


# ── Gold master orchestration ────────────────────────────────────────────────


def build_gold_master(
    silver_customer: pd.DataFrame,
    silver_customer_month: pd.DataFrame,
    as_of_date: datetime | None = None,
) -> pd.DataFrame:
    """Orchestrate all feature tiers into a single gold master table (1 row/customer)."""
    tier_1a = build_lifecycle_features(silver_customer, silver_customer_month, as_of_date)
    tier_mp_core = build_market_core_features(silver_customer_month, silver_customer)
    tier_mp_risk = build_market_risk_features(silver_customer_month)
    tier_2a = build_behavioral_features(silver_customer, as_of_date)
    tier_2b = build_sentiment_features(silver_customer)

    # Merge all tiers onto tier_1a backbone
    gold = tier_1a.copy()
    for tier in [tier_mp_core, tier_mp_risk, tier_2a, tier_2b]:
        # Avoid duplicate columns
        new_cols = [c for c in tier.columns if c not in gold.columns or c == KEY]
        gold = gold.merge(tier[new_cols], on=KEY, how="left")

    # Build compound features on merged data
    gold = build_compound_features(gold)

    # Attach churn label
    if "churn" in silver_customer.columns:
        gold = gold.merge(
            silver_customer[[KEY, "churn"]].drop_duplicates(KEY), on=KEY, how="left"
        )

    # Validate grain
    if gold.duplicated(subset=[KEY]).any():
        raise ValueError("gold_master has duplicate customer_id rows")

    return gold

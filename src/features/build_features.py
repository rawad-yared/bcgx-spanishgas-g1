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

    # Renewal bucket (5 bins to match notebook)
    if "months_to_renewal" in sc.columns:
        bins = [-np.inf, 0, 3, 6, 12, np.inf]
        labels = ["expired", "0-3m", "3-6m", "6-12m", "12m+"]
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

    # Margin features (excluded from training but kept for expected_monthly_loss)
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


# ── Tier MP_Risk: Consumption std, price trends, margin stability ────────────


def build_market_risk_features(
    silver_customer_month: pd.DataFrame,
) -> pd.DataFrame:
    """Tier MP_Risk — consumption std, relative price trends, margin stability."""
    scm = silver_customer_month.copy()

    risk = scm[[KEY]].drop_duplicates().reset_index(drop=True)

    # Consumption standard deviation (raw std, not CV) + active months count
    cons_stats = scm.groupby(KEY).agg(
        std_monthly_elec_kwh=("monthly_elec_kwh", "std"),
        std_monthly_gas_m3=("monthly_gas_m3", "std"),
        active_months_count=("month", "nunique"),
    ).reset_index()
    risk = risk.merge(cons_stats, on=KEY, how="left")

    # Margin stats: std, min, count of negative months
    if "total_margin" in scm.columns:
        margin_stats = scm.groupby(KEY, as_index=False).agg(
            std_margin=("total_margin", "std"),
            min_monthly_margin=("total_margin", "min"),
        )
        risk = risk.merge(margin_stats, on=KEY, how="left")

        # max_negative_margin = count of months with negative margin
        neg_months = scm[scm["total_margin"] < 0].groupby(KEY, as_index=False).agg(
            max_negative_margin=("total_margin", "count"),
        )
        risk = risk.merge(neg_months, on=KEY, how="left")
        risk["max_negative_margin"] = risk["max_negative_margin"].fillna(0)

    # Electricity price trend (relative: (last - first) / first) + volatility
    if "variable_price_tier1_eur_kwh" in scm.columns:
        sorted_scm = scm.sort_values([KEY, "month"])
        price_agg = sorted_scm.groupby(KEY).agg(
            _first_price=("variable_price_tier1_eur_kwh", "first"),
            _last_price=("variable_price_tier1_eur_kwh", "last"),
            elec_price_volatility_12m=("variable_price_tier1_eur_kwh", "std"),
        ).reset_index()
        price_agg["elec_price_trend_12m"] = np.where(
            price_agg["_first_price"] > 0,
            (price_agg["_last_price"] - price_agg["_first_price"]) / price_agg["_first_price"],
            0.0,
        )
        price_agg["is_price_increase"] = (price_agg["elec_price_trend_12m"] > 0).astype("Int64")
        risk = risk.merge(
            price_agg[[KEY, "elec_price_trend_12m", "elec_price_volatility_12m", "is_price_increase"]],
            on=KEY, how="left",
        )

    # Gas price trend (relative)
    if "gas_variable_price_eur_m3" in scm.columns:
        sorted_scm = scm.sort_values([KEY, "month"])
        gas_agg = sorted_scm.groupby(KEY).agg(
            _first_gas=("gas_variable_price_eur_m3", "first"),
            _last_gas=("gas_variable_price_eur_m3", "last"),
        ).reset_index()
        gas_agg["gas_price_trend_12m"] = np.where(
            gas_agg["_first_gas"] > 0,
            (gas_agg["_last_gas"] - gas_agg["_first_gas"]) / gas_agg["_first_gas"],
            0.0,
        )
        risk = risk.merge(gas_agg[[KEY, "gas_price_trend_12m"]], on=KEY, how="left")

    # Province electricity cost trend (relative)
    if "elec_var_cost_eur_kwh" in scm.columns:
        sorted_scm = scm.sort_values([KEY, "month"])
        cost_agg = sorted_scm.groupby(KEY).agg(
            _first_cost=("elec_var_cost_eur_kwh", "first"),
            _last_cost=("elec_var_cost_eur_kwh", "last"),
        ).reset_index()
        cost_agg["province_elec_cost_trend"] = np.where(
            cost_agg["_first_cost"] > 0,
            (cost_agg["_last_cost"] - cost_agg["_first_cost"]) / cost_agg["_first_cost"],
            0.0,
        )
        risk = risk.merge(cost_agg[[KEY, "province_elec_cost_trend"]], on=KEY, how="left")

    # Price vs province cost spread (last price - avg province cost)
    if "variable_price_tier1_eur_kwh" in scm.columns and "elec_var_cost_eur_kwh" in scm.columns:
        sorted_scm = scm.sort_values([KEY, "month"])
        spread_agg = sorted_scm.groupby(KEY).agg(
            _last_price_s=("variable_price_tier1_eur_kwh", "last"),
            _avg_cost=("elec_var_cost_eur_kwh", "mean"),
        ).reset_index()
        spread_agg["elec_price_vs_province_cost_spread"] = (
            spread_agg["_last_price_s"] - spread_agg["_avg_cost"]
        )
        risk = risk.merge(
            spread_agg[[KEY, "elec_price_vs_province_cost_spread"]], on=KEY, how="left",
        )

    # Rolling margin trend (last 3 months avg - prior 3 months avg)
    if "total_margin" in scm.columns:
        sorted_scm = scm.sort_values([KEY, "month"])
        sorted_scm["_rank"] = sorted_scm.groupby(KEY).cumcount(ascending=False)
        last_3 = sorted_scm[sorted_scm["_rank"] < 3].groupby(KEY, as_index=False).agg(
            _last3_avg=("total_margin", "mean"),
        )
        prior_3 = sorted_scm[
            (sorted_scm["_rank"] >= 3) & (sorted_scm["_rank"] < 6)
        ].groupby(KEY, as_index=False).agg(
            _prior3_avg=("total_margin", "mean"),
        )
        margin_trend = last_3.merge(prior_3, on=KEY, how="left")
        margin_trend["rolling_margin_trend"] = (
            margin_trend["_last3_avg"] - margin_trend["_prior3_avg"].fillna(margin_trend["_last3_avg"])
        )
        risk = risk.merge(margin_trend[[KEY, "rolling_margin_trend"]], on=KEY, how="left")

    return risk


# ── Tier 2A: Behavioral features ────────────────────────────────────────────


def build_behavioral_features(
    silver_customer: pd.DataFrame,
    as_of_date: datetime | None = None,
) -> pd.DataFrame:
    """Tier 2A — interaction presence, intent flags, severity, recency, timing."""
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

    # Intent + intent-based flags
    if "customer_intent" in sc.columns:
        intent_df = sc[[KEY, "customer_intent"]].drop_duplicates(KEY)
        behav = behav.merge(intent_df, on=KEY, how="left")

        intent_str = behav["customer_intent"].astype("string").str.strip()
        behav["is_cancellation_intent"] = (
            intent_str == "Cancellation / Switch"
        ).astype("Int64")
        behav["is_complaint_intent"] = (
            intent_str == "Complaint / Escalation"
        ).astype("Int64")
        behav["recent_complaint_flag"] = (
            intent_str.isin(["Complaint / Escalation", "Cancellation / Switch"])
        ).astype("Int64")

        # Severity ordinal
        severity_map = {
            "Cancellation / Switch": 3,
            "Complaint / Escalation": 2,
            "Pricing Offers": 1,
        }
        behav["intent_severity_score"] = (
            behav["customer_intent"].map(severity_map).fillna(0).astype("Int64")
        )

    # Last interaction days ago
    if "date" in sc.columns:
        dates = sc[[KEY, "date"]].copy()
        dates["date"] = pd.to_datetime(dates["date"], errors="coerce")
        latest = dates.groupby(KEY, as_index=False).agg(last_date=("date", "max"))
        latest["last_interaction_days_ago"] = (as_of - latest["last_date"]).dt.days
        behav = behav.merge(latest[[KEY, "last_interaction_days_ago"]], on=KEY, how="left")

    # Interaction timing relative to renewal
    if "date" in sc.columns and "next_renewal_date" in sc.columns:
        timing = sc[[KEY, "date", "next_renewal_date"]].copy()
        timing["date"] = pd.to_datetime(timing["date"], errors="coerce")
        timing["next_renewal_date"] = pd.to_datetime(timing["next_renewal_date"], errors="coerce")
        timing["_months_to_renewal_at_interaction"] = (
            (timing["next_renewal_date"] - timing["date"]).dt.days / 30.44
        )
        # Take latest interaction per customer
        latest_timing = timing.sort_values("date").groupby(KEY, as_index=False).last()
        latest_timing["interaction_within_3m_of_renewal"] = (
            latest_timing["_months_to_renewal_at_interaction"].between(0, 3)
        ).astype("Int64")
        latest_timing["is_interaction_within_30d_of_renewal"] = (
            latest_timing["_months_to_renewal_at_interaction"].between(0, 1)
        ).astype("Int64")
        behav = behav.merge(
            latest_timing[[KEY, "interaction_within_3m_of_renewal", "is_interaction_within_30d_of_renewal"]],
            on=KEY, how="left",
        )

    # Complaint near renewal
    if "recent_complaint_flag" in behav.columns and "interaction_within_3m_of_renewal" in behav.columns:
        behav["complaint_near_renewal"] = (
            (behav["recent_complaint_flag"].fillna(0) == 1)
            & (behav["interaction_within_3m_of_renewal"].fillna(0) == 1)
        ).astype("Int64")

    # Months since last product change
    if "last_product_change_date" in sc.columns:
        change = sc[[KEY, "last_product_change_date"]].drop_duplicates(KEY)
        change["last_product_change_date"] = pd.to_datetime(
            change["last_product_change_date"], errors="coerce"
        )
        change["months_since_last_change"] = (
            (as_of - change["last_product_change_date"]).dt.days / 30.44
        ).round(1)
        behav = behav.merge(change[[KEY, "months_since_last_change"]], on=KEY, how="left")

    return behav


# ── Tier 2B: Sentiment features ─────────────────────────────────────────────


def build_sentiment_features(silver_customer: pd.DataFrame) -> pd.DataFrame:
    """Tier 2B — sentiment label, negative sentiment flag."""
    sc = silver_customer.copy()
    sent = sc[[KEY]].drop_duplicates().reset_index(drop=True)

    for col in ["sentiment_label", "sentiment_neg", "sentiment_pos", "sentiment_neu"]:
        if col in sc.columns:
            sent = sent.merge(sc[[KEY, col]].drop_duplicates(KEY), on=KEY, how="left")

    if "sentiment_label" in sent.columns:
        sent["is_negative_sentiment"] = (
            sent["sentiment_label"].astype("string").str.lower() == "negative"
        ).astype("Int64")

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

    # Interaction string features (Tier 1B)
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

    if "sales_channel" in df.columns and "renewal_bucket" in df.columns:
        df["sales_channel_x_renewal_bucket"] = _safe_cross("sales_channel", "renewal_bucket")

    if "has_interaction" in df.columns and "renewal_bucket" in df.columns:
        df["has_interaction_x_renewal_bucket"] = _safe_cross("has_interaction", "renewal_bucket")

    if "is_high_competition_province" in df.columns and "customer_intent" in df.columns:
        df["competition_x_intent"] = _safe_cross("is_high_competition_province", "customer_intent")

    # Binary compound flags (Tier 3)
    if "customer_intent" in df.columns:
        df["is_price_sensitive"] = (
            df["customer_intent"].astype("string") == "Pricing Offers"
        ).astype("Int64")

    if "renewal_bucket" in df.columns and "tenure_bucket" in df.columns:
        df["is_high_risk_lifecycle"] = (
            df["renewal_bucket"].astype("string").isin(["expired", "0-3m"])
            & df["tenure_bucket"].astype("string").isin(["0-6m", "6-12m"])
        ).astype("Int64")

    if "is_high_competition_province" in df.columns and "is_within_3m_of_renewal" in df.columns:
        df["is_competition_x_renewal"] = (
            (df["is_high_competition_province"].fillna(0) == 1)
            & (df["is_within_3m_of_renewal"].fillna(0) == 1)
        ).astype("Int64")

    if "is_dual_fuel" in df.columns and "is_within_3m_of_renewal" in df.columns:
        df["dual_fuel_x_renewal"] = (
            (df["is_dual_fuel"].fillna(0) == 1)
            & (df["is_within_3m_of_renewal"].fillna(0) == 1)
        ).astype("Int64")

    if "is_dual_fuel" in df.columns and "is_high_competition_province" in df.columns:
        df["dual_fuel_x_competition"] = (
            (df["is_dual_fuel"].fillna(0) == 1)
            & (df["is_high_competition_province"].fillna(0) == 1)
        ).astype("Int64")

    if "is_dual_fuel" in df.columns and "customer_intent" in df.columns:
        df["dual_fuel_x_intent"] = (
            (df["is_dual_fuel"].fillna(0) == 1)
            & (
                df["customer_intent"].astype("string").isin(
                    ["Cancellation / Switch", "Complaint / Escalation"]
                )
            )
        ).astype("Int64")

    if "recent_complaint_flag" in df.columns and "is_negative_sentiment" in df.columns:
        df["complaint_x_negative_sentiment"] = (
            (df["recent_complaint_flag"].fillna(0) == 1)
            & (df["is_negative_sentiment"].fillna(0) == 1)
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

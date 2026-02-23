"""Phase 1E: Score all customers and assign risk tiers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

KEY = "customer_id"


def score_all_customers(
    pipeline: Pipeline,
    gold: pd.DataFrame,
    features: list[str],
    threshold: float,
) -> pd.DataFrame:
    """Score every customer in the gold master, producing churn_proba and churn_pred."""
    available = [f for f in features if f in gold.columns]
    X_all = gold[available]
    cids = gold[KEY]
    y_all = gold["churn"] if "churn" in gold.columns else pd.Series(np.nan, index=gold.index)

    proba = pipeline.predict_proba(X_all)[:, 1]
    pred = (proba >= threshold).astype(int)

    scored = pd.DataFrame({
        KEY: cids.values,
        "churn_actual": y_all.values,
        "churn_proba": np.round(proba, 4),
        "churn_pred": pred,
    })

    # Attach business columns (not used in training)
    biz_cols = [c for c in ["avg_monthly_margin", "total_margin_2024", "segment"] if c in gold.columns]
    if biz_cols:
        scored = scored.merge(gold[[KEY] + biz_cols], on=KEY, how="left")

    # Expected monthly loss
    if "avg_monthly_margin" in scored.columns:
        scored["expected_monthly_loss"] = (
            scored["churn_proba"] * scored["avg_monthly_margin"].clip(lower=0)
        ).round(2)

    scored = scored.sort_values("churn_proba", ascending=False).reset_index(drop=True)
    return scored


def assign_risk_tiers(scored: pd.DataFrame) -> pd.DataFrame:
    """Assign risk tiers based on churn probability."""
    scored = scored.copy()
    scored["risk_tier"] = pd.cut(
        scored["churn_proba"],
        bins=[0, 0.40, 0.60, 0.80, 1.01],
        labels=["Low (<40%)", "Medium (40-60%)", "High (60-80%)", "Critical (>80%)"],
        right=False,
    )
    return scored

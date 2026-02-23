"""Phase 1D: Build model-ready training set from gold master."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

KEY = "customer_id"
TARGET = "churn"

# Structural fills applied before sklearn pipeline
NUMERIC_SENTINEL_COLS = ["last_interaction_days_ago"]
NUMERIC_SENTINEL_VALUE = 9999

CATEGORICAL_DEFAULT_COLS = [
    "sentiment_label",
    "customer_intent",
    "sentiment_x_renewal_bucket",
    "intent_x_renewal_bucket",
    "intent_x_tenure_bucket",
    "intent_x_sentiment",
    "tenure_x_renewal_bucket",
    "sales_channel_x_renewal_bucket",
    "has_interaction_x_renewal_bucket",
    "competition_x_intent",
]
CATEGORICAL_DEFAULT_VALUE = "no_interaction"


def build_model_matrix(
    gold_df: pd.DataFrame,
    feature_list: list[str],
    target_col: str = TARGET,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Build X, y, customer_id from gold master.

    Applies structural fills for nulls that have business meaning
    (e.g., no interaction → 9999 days ago, not missing data).

    Returns:
        (X, y, customer_ids)
    """
    df = gold_df.copy()

    # Structural fills: numeric sentinel
    for col in NUMERIC_SENTINEL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(NUMERIC_SENTINEL_VALUE)

    # Structural fills: categorical default
    for col in CATEGORICAL_DEFAULT_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(CATEGORICAL_DEFAULT_VALUE)

    # Binary interaction flags → 0 for missing
    binary_flags = [c for c in df.columns if c.startswith("is_") or c.endswith("_x_complaint")]
    for col in binary_flags:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Select features that exist
    available = [f for f in feature_list if f in df.columns]
    missing = [f for f in feature_list if f not in df.columns]
    if missing:
        import warnings
        warnings.warn(f"Features not in gold master (skipped): {missing}")

    X = df[available]
    y = df[target_col]
    cids = df[KEY]

    return X, y, cids


def create_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.20,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split preserving churn distribution."""
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

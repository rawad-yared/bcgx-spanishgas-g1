"""Data quality checks for each medallion layer."""

from __future__ import annotations

import pandas as pd

KEY = "customer_id"

# Null-rate thresholds per layer
_NULL_THRESHOLDS = {"raw": 0.80, "bronze": 0.50, "silver": 0.30, "gold": 0.10}


def check_data_quality(df: pd.DataFrame, layer: str = "bronze") -> dict:
    """Run data quality checks on a DataFrame.

    Returns dict with null_rates, duplicate_keys, row_count, column_count,
    numeric_ranges, schema, issues list, and passed boolean.
    """
    issues: list[str] = []
    null_threshold = _NULL_THRESHOLDS.get(layer, 0.50)

    # Row / column counts
    row_count = len(df)
    column_count = len(df.columns)

    if row_count == 0:
        issues.append("DataFrame is empty (0 rows)")

    # Null rates
    null_rates = {}
    for col in df.columns:
        rate = float(df[col].isna().mean()) if row_count > 0 else 0.0
        null_rates[col] = round(rate, 4)
        if rate > null_threshold:
            issues.append(f"High null rate in '{col}': {rate:.1%} (threshold: {null_threshold:.0%})")

    # Duplicate keys
    duplicate_keys = 0
    if KEY in df.columns and row_count > 0:
        dup_mask = df.duplicated(subset=[KEY], keep=False)
        duplicate_keys = int(dup_mask.sum())
        if duplicate_keys > 0:
            issues.append(f"Found {duplicate_keys} duplicate {KEY} rows")

    # Numeric ranges
    numeric_ranges = {}
    for col in df.select_dtypes(include="number").columns:
        numeric_ranges[col] = {
            "min": float(df[col].min()) if row_count > 0 else None,
            "max": float(df[col].max()) if row_count > 0 else None,
        }

    # Schema
    schema = [{"column": col, "dtype": str(df[col].dtype)} for col in df.columns]

    passed = len(issues) == 0

    return {
        "layer": layer,
        "null_rates": null_rates,
        "duplicate_keys": duplicate_keys,
        "row_count": row_count,
        "column_count": column_count,
        "numeric_ranges": numeric_ranges,
        "schema": schema,
        "issues": issues,
        "passed": passed,
    }

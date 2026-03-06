"""Feature and prediction drift detection using Kolmogorov-Smirnov test."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def compute_feature_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    features: list[str],
    p_threshold: float = 0.01,
) -> dict:
    """Compute KS test for each numeric feature between reference and current.

    Returns dict with feature_results, n_drifted, drift_detected.
    """
    results: list[dict] = []

    for feat in features:
        if feat not in reference.columns or feat not in current.columns:
            continue
        if not pd.api.types.is_numeric_dtype(reference[feat]):
            continue

        ref_vals = reference[feat].dropna().values
        cur_vals = current[feat].dropna().values

        if len(ref_vals) == 0 or len(cur_vals) == 0:
            continue

        ks_stat, p_value = stats.ks_2samp(ref_vals, cur_vals)
        results.append({
            "feature": feat,
            "ks_statistic": float(ks_stat),
            "p_value": float(p_value),
            "drifted": p_value < p_threshold,
        })

    n_drifted = sum(1 for r in results if r["drifted"])
    return {
        "feature_results": results,
        "n_drifted": n_drifted,
        "drift_detected": n_drifted > 0,
    }


def compute_prediction_drift(
    reference_probas: np.ndarray,
    current_probas: np.ndarray,
    p_threshold: float = 0.01,
) -> dict:
    """KS test on prediction probability distributions."""
    if len(reference_probas) == 0 or len(current_probas) == 0:
        return {"ks_statistic": 0.0, "p_value": 1.0, "drifted": False}

    ks_stat, p_value = stats.ks_2samp(reference_probas, current_probas)
    return {
        "ks_statistic": float(ks_stat),
        "p_value": float(p_value),
        "drifted": p_value < p_threshold,
    }


def summarize_drift(feature_drift: dict, prediction_drift: dict) -> dict:
    """Combine feature and prediction drift results into a summary."""
    any_drift = feature_drift["drift_detected"] or prediction_drift["drifted"]
    n_features_drifted = feature_drift["n_drifted"]

    drifted_names = [r["feature"] for r in feature_drift["feature_results"] if r["drifted"]]
    lines = []
    if drifted_names:
        lines.append(f"Feature drift detected in: {', '.join(drifted_names)}")
    if prediction_drift["drifted"]:
        lines.append(f"Prediction drift: KS={prediction_drift['ks_statistic']:.4f}")
    if not lines:
        lines.append("No drift detected.")

    return {
        "any_drift": any_drift,
        "n_features_drifted": n_features_drifted,
        "feature_drift": feature_drift,
        "prediction_drift": prediction_drift,
        "summary": "\n".join(lines),
    }

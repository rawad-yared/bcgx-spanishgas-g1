"""Phase 1E: Churn model definitions, threshold selection, evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from src.models.preprocessing import build_preprocessing_pipeline

TARGET_RECALL = 0.70


def get_model_definitions(y_train: pd.Series | None = None) -> dict[str, Any]:
    """Return dict of model name → sklearn estimator."""
    churn_rate = float(y_train.mean()) if y_train is not None else 0.10
    scale_weight = (1 - churn_rate) / max(churn_rate, 1e-6)

    models = {
        "dummy_stratified": DummyClassifier(strategy="stratified", random_state=42),
        "dummy_most_frequent": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }

    try:
        from xgboost import XGBClassifier

        models["xgboost"] = XGBClassifier(
            n_estimators=600,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_weight,
            eval_metric="aucpr",
            random_state=42,
            n_jobs=-1,
            use_label_encoder=False,
        )
    except ImportError:
        pass

    return models


def pick_threshold(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray,
    target_recall: float = TARGET_RECALL,
) -> float:
    """Select threshold maximising precision subject to recall >= target_recall."""
    precision_arr, recall_arr, thresholds = precision_recall_curve(y_true, y_proba)
    pr = precision_arr[:-1]
    re = recall_arr[:-1]

    valid_idx = np.where(re >= target_recall)[0]

    if len(valid_idx) == 0:
        # Fallback: best F1
        f1 = 2 * (pr * re) / (pr + re + 1e-12)
        best_idx = int(np.nanargmax(f1))
    else:
        best_idx = valid_idx[np.argmax(pr[valid_idx])]

    return float(thresholds[best_idx])


def evaluate_model(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Compute all evaluation metrics for a model."""
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "threshold": threshold,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def run_experiment(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "xgboost",
    scale_numeric: bool = False,
    target_recall: float = TARGET_RECALL,
) -> dict[str, Any]:
    """Run a single experiment: preprocess, train, threshold, evaluate.

    Returns dict with model pipeline, threshold, and metrics.
    """
    is_linear = model_name in ("logistic_regression",)
    preprocessor = build_preprocessing_pipeline(
        X_train, scale_numeric=is_linear, ohe_sparse=False
    )

    models = get_model_definitions(y_train)
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", models[model_name]),
    ])

    pipeline.fit(X_train, y_train)

    # Threshold on validation split from training data
    from sklearn.model_selection import train_test_split

    X_tr_sub, X_val, y_tr_sub, y_val = train_test_split(
        X_train, y_train, test_size=0.25, stratify=y_train, random_state=42
    )
    val_pipeline = Pipeline([
        ("preprocessor", build_preprocessing_pipeline(X_tr_sub, scale_numeric=is_linear)),
        ("model", models[model_name].__class__(**models[model_name].get_params())),
    ])
    val_pipeline.fit(X_tr_sub, y_tr_sub)
    val_proba = val_pipeline.predict_proba(X_val)[:, 1]
    threshold = pick_threshold(y_val, val_proba, target_recall)

    # Final evaluation on test set (refit on full training)
    test_proba = pipeline.predict_proba(X_test)[:, 1]
    metrics = evaluate_model(y_test, test_proba, threshold)

    return {
        "pipeline": pipeline,
        "threshold": threshold,
        "metrics": metrics,
        "model_name": model_name,
    }

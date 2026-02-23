"""Phase 1E: Preprocessing pipeline — numeric + categorical ColumnTransformer."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessing_pipeline(
    X: pd.DataFrame,
    scale_numeric: bool = True,
    ohe_sparse: bool = False,
) -> ColumnTransformer:
    """Build a ColumnTransformer that handles numeric and categorical features.

    Numeric:  SimpleImputer(median) → optional StandardScaler
    Categorical: SimpleImputer(constant="missing") → OneHotEncoder(handle_unknown=ignore)

    Args:
        X: Training feature DataFrame (used to detect dtypes).
        scale_numeric: Whether to apply StandardScaler to numeric features.
        ohe_sparse: Whether OHE output should be sparse (False for tree models).

    Returns:
        Fitted-ready ColumnTransformer.
    """
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()

    # Numeric pipeline
    num_steps: list[tuple] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(num_steps)

    # Categorical pipeline
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=ohe_sparse)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", cat_pipeline, categorical_cols),
        ],
        remainder="drop",
    )

    return preprocessor

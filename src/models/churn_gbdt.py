"""Train and evaluate a gradient-boosted churn model."""

from __future__ import annotations

import argparse
import json
import math
import pickle
from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any
from typing import Iterable
from typing import Protocol

import numpy as np
import yaml
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import average_precision_score
from sklearn.metrics import brier_score_loss


class S3ClientProtocol(Protocol):
    """Minimal S3 client protocol used by this module."""

    def get_object(self, **kwargs: Any) -> dict[str, Any]: ...

    def put_object(self, **kwargs: Any) -> dict[str, Any]: ...


@dataclass(frozen=True)
class GBDTConfig:
    training_dataset_path: str
    baseline_metrics_path: str
    model_output_dir: str
    comparison_report_path: str
    random_seed: int
    n_estimators: int
    learning_rate: float
    max_depth: int
    top_k_fraction: float
    calibration_bins: int
    max_brier_increase: float
    max_ece_increase: float
    top_feature_count: int
    max_shap_samples: int


@dataclass(frozen=True)
class SplitMetrics:
    split: str
    row_count: int
    positive_rate: float
    pr_auc: float
    recall_at_k: float
    precision_at_k: float
    brier_score: float
    expected_calibration_error: float
    avg_predicted_probability: float


@dataclass(frozen=True)
class TrainResult:
    model_path: str
    metrics_path: str
    report_path: str
    split_counts: dict[str, int]
    primary_eval_split: str
    champion_pass: bool
    improvement_dimension: str
    metrics: dict[str, SplitMetrics]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate gradient-boosted churn model."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to model GBDT config YAML.",
    )
    return parser.parse_args(argv)


def _is_s3_uri(path: str) -> bool:
    return path.startswith("s3://")


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    if not _is_s3_uri(uri):
        raise ValueError(f"Not an S3 URI: {uri}")
    bucket_and_key = uri[len("s3://") :]
    bucket, _, key = bucket_and_key.partition("/")
    if not bucket:
        raise ValueError(f"Invalid S3 URI, missing bucket: {uri}")
    return bucket, key.lstrip("/")


def _join_location(root: str, relative_path: str) -> str:
    if _is_s3_uri(root):
        return f"{root.rstrip('/')}/{relative_path.lstrip('/')}"
    return str(Path(root) / relative_path)


def _resolve_s3_client(
    paths: Iterable[str], s3_client: S3ClientProtocol | None = None
) -> S3ClientProtocol | None:
    if s3_client is not None:
        return s3_client

    if not any(_is_s3_uri(path) for path in paths):
        return None

    try:
        import boto3

        return boto3.client("s3")
    except ImportError:
        pass

    try:
        from botocore.session import Session

        return Session().create_client("s3")
    except Exception as exc:
        raise RuntimeError(
            "S3 model training requested but no S3 SDK is available. "
            "Install boto3 or use local paths."
        ) from exc


def _read_bytes(path: str, s3_client: S3ClientProtocol | None) -> bytes:
    if _is_s3_uri(path):
        if s3_client is None:
            raise RuntimeError("S3 path used without an S3 client.")
        bucket, key = _parse_s3_uri(path)
        try:
            response = s3_client.get_object(Bucket=bucket, Key=key)
        except Exception as exc:
            raise FileNotFoundError(f"Missing input file: {path}") from exc
        return response["Body"].read()

    local_path = Path(path)
    if not local_path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    return local_path.read_bytes()


def _write_bytes(path: str, payload: bytes, s3_client: S3ClientProtocol | None) -> None:
    if _is_s3_uri(path):
        if s3_client is None:
            raise RuntimeError("S3 path used without an S3 client.")
        bucket, key = _parse_s3_uri(path)
        s3_client.put_object(Bucket=bucket, Key=key, Body=payload)
        return

    local_path = Path(path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_bytes(payload)


def _read_json(path: str, s3_client: S3ClientProtocol | None) -> dict[str, Any]:
    payload = _read_bytes(path, s3_client)
    decoded = json.loads(payload.decode("utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError(f"JSON payload at {path} must be an object.")
    return decoded


def _read_jsonl(path: str, s3_client: S3ClientProtocol | None) -> list[dict[str, Any]]:
    payload = _read_bytes(path, s3_client)
    rows: list[dict[str, Any]] = []
    for line in payload.decode("utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        decoded = json.loads(stripped)
        if isinstance(decoded, dict):
            rows.append(decoded)
    return rows


def _load_config(config_path: str) -> GBDTConfig:
    raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Config must be a YAML object.")

    paths = raw.get("paths") or {}
    model = raw.get("model") or {}
    evaluation = raw.get("evaluation") or {}

    training_dataset_path = str(paths.get("training_dataset_path", "")).strip()
    baseline_metrics_path = str(paths.get("baseline_metrics_path", "")).strip()
    model_output_dir = str(paths.get("model_output_dir", "")).strip()
    comparison_report_path = str(paths.get("comparison_report_path", "")).strip()

    if (
        not training_dataset_path
        or not baseline_metrics_path
        or not model_output_dir
        or not comparison_report_path
    ):
        raise ValueError(
            "Config paths must include training_dataset_path, baseline_metrics_path, "
            "model_output_dir, and comparison_report_path."
        )

    random_seed = int(model.get("random_seed", 42))
    n_estimators = int(model.get("n_estimators", 200))
    learning_rate = float(model.get("learning_rate", 0.05))
    max_depth = int(model.get("max_depth", 3))

    top_k_fraction = float(evaluation.get("top_k_fraction", 0.2))
    calibration_bins = int(evaluation.get("calibration_bins", 10))
    max_brier_increase = float(evaluation.get("max_brier_increase", 0.03))
    max_ece_increase = float(evaluation.get("max_ece_increase", 0.03))
    top_feature_count = int(evaluation.get("top_feature_count", 10))
    max_shap_samples = int(evaluation.get("max_shap_samples", 300))

    if not (0 < top_k_fraction <= 1):
        raise ValueError("evaluation.top_k_fraction must be in (0, 1].")
    if calibration_bins <= 1:
        raise ValueError("evaluation.calibration_bins must be > 1.")
    if top_feature_count <= 0:
        raise ValueError("evaluation.top_feature_count must be > 0.")
    if max_shap_samples <= 0:
        raise ValueError("evaluation.max_shap_samples must be > 0.")

    return GBDTConfig(
        training_dataset_path=training_dataset_path,
        baseline_metrics_path=baseline_metrics_path,
        model_output_dir=model_output_dir,
        comparison_report_path=comparison_report_path,
        random_seed=random_seed,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        top_k_fraction=top_k_fraction,
        calibration_bins=calibration_bins,
        max_brier_increase=max_brier_increase,
        max_ece_increase=max_ece_increase,
        top_feature_count=top_feature_count,
        max_shap_samples=max_shap_samples,
    )


def _to_int_label(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _feature_columns(rows: list[dict[str, Any]]) -> list[str]:
    reserved = {
        "customer_id",
        "asof_date",
        "label_horizon_days",
        "churn_label",
        "split",
    }
    return sorted({key for row in rows for key in row if key not in reserved})


def _numeric_columns(rows: list[dict[str, Any]], feature_cols: list[str]) -> set[str]:
    numeric_cols: set[str] = set()
    for column in feature_cols:
        non_null_values = [row.get(column) for row in rows if row.get(column) is not None]
        if not non_null_values:
            continue
        if all(isinstance(value, (int, float)) for value in non_null_values):
            numeric_cols.add(column)
    return numeric_cols


def _normalize_feature_row(
    row: dict[str, Any], feature_cols: list[str], numeric_cols: set[str]
) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for column in feature_cols:
        value = row.get(column)
        if column in numeric_cols:
            normalized[column] = 0.0 if value is None else float(value)
        else:
            normalized[column] = "__missing__" if value is None else str(value)
    return normalized


def _recall_precision_at_k(
    y_true: np.ndarray, y_prob: np.ndarray, top_k_fraction: float
) -> tuple[float, float]:
    if y_true.size == 0:
        return 0.0, 0.0
    k = max(1, int(math.ceil(y_true.size * top_k_fraction)))
    ranked_idx = np.argsort(-y_prob)[:k]
    tp = int(np.sum(y_true[ranked_idx]))
    positives = int(np.sum(y_true))
    precision_at_k = tp / k if k > 0 else 0.0
    recall_at_k = tp / positives if positives > 0 else 0.0
    return recall_at_k, precision_at_k


def _expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, bins: int
) -> float:
    if y_true.size == 0:
        return 0.0

    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for idx in range(bins):
        left, right = edges[idx], edges[idx + 1]
        if idx == bins - 1:
            mask = (y_prob >= left) & (y_prob <= right)
        else:
            mask = (y_prob >= left) & (y_prob < right)
        if not np.any(mask):
            continue
        bin_true = y_true[mask]
        bin_prob = y_prob[mask]
        ece += (bin_true.size / y_true.size) * abs(
            float(np.mean(bin_true)) - float(np.mean(bin_prob))
        )
    return float(ece)


def _compute_split_metrics(
    split_name: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    top_k_fraction: float,
    calibration_bins: int,
) -> SplitMetrics:
    if y_true.size == 0:
        return SplitMetrics(
            split=split_name,
            row_count=0,
            positive_rate=0.0,
            pr_auc=0.0,
            recall_at_k=0.0,
            precision_at_k=0.0,
            brier_score=0.0,
            expected_calibration_error=0.0,
            avg_predicted_probability=0.0,
        )

    positive_rate = float(np.mean(y_true))
    unique_labels = set(y_true.tolist())
    if len(unique_labels) < 2:
        pr_auc = positive_rate
    else:
        pr_auc = float(average_precision_score(y_true, y_prob))

    recall_at_k, precision_at_k = _recall_precision_at_k(y_true, y_prob, top_k_fraction)
    brier = float(brier_score_loss(y_true, y_prob))
    ece = _expected_calibration_error(y_true, y_prob, calibration_bins)

    return SplitMetrics(
        split=split_name,
        row_count=int(y_true.size),
        positive_rate=positive_rate,
        pr_auc=pr_auc,
        recall_at_k=recall_at_k,
        precision_at_k=precision_at_k,
        brier_score=brier,
        expected_calibration_error=ece,
        avg_predicted_probability=float(np.mean(y_prob)),
    )


def _top_feature_importance(
    model: GradientBoostingClassifier,
    feature_names: list[str],
    top_k: int,
) -> list[dict[str, Any]]:
    importances = model.feature_importances_
    idx = np.argsort(-importances)[:top_k]
    return [
        {"feature": feature_names[i], "importance": float(importances[i])}
        for i in idx
        if importances[i] > 0
    ]


def _shap_summary(
    model: GradientBoostingClassifier,
    x_eval: np.ndarray,
    feature_names: list[str],
    y_prob: np.ndarray,
    random_seed: int,
    top_k: int,
    max_samples: int,
) -> dict[str, Any]:
    sample_size = min(x_eval.shape[0], max_samples)
    if sample_size <= 0:
        return {"method": "none", "top_features": []}

    rng = np.random.default_rng(random_seed)
    sampled_idx = rng.choice(x_eval.shape[0], size=sample_size, replace=False)
    x_sample = x_eval[sampled_idx]

    try:
        import shap  # type: ignore

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(x_sample)
        if isinstance(shap_values, list):
            shap_array = np.asarray(shap_values[-1])
        else:
            shap_array = np.asarray(shap_values)
        mean_abs = np.mean(np.abs(shap_array), axis=0)
        ranked = np.argsort(-mean_abs)[:top_k]
        return {
            "method": "shap",
            "top_features": [
                {"feature": feature_names[i], "mean_abs_shap": float(mean_abs[i])}
                for i in ranked
            ],
        }
    except Exception:
        # SHAP not installed/available: fallback proxy based on permutation effect
        base_probs = y_prob[sampled_idx]
        impacts = []
        for col_idx, feature_name in enumerate(feature_names):
            permuted = x_sample.copy()
            permuted[:, col_idx] = rng.permutation(permuted[:, col_idx])
            permuted_probs = model.predict_proba(permuted)[:, 1]
            impact = float(np.mean(np.abs(permuted_probs - base_probs)))
            impacts.append((feature_name, impact))
        impacts.sort(key=lambda item: item[1], reverse=True)
        return {
            "method": "permutation_proxy",
            "top_features": [
                {"feature": feature, "mean_abs_effect": impact}
                for feature, impact in impacts[:top_k]
            ],
        }


def train_gbdt(
    config: GBDTConfig,
    s3_client: S3ClientProtocol | None = None,
) -> TrainResult:
    """Train champion-candidate GBDT and compare against baseline."""

    resolved_s3_client = _resolve_s3_client(
        paths=[
            config.training_dataset_path,
            config.baseline_metrics_path,
            config.model_output_dir,
            config.comparison_report_path,
        ],
        s3_client=s3_client,
    )

    rows = _read_jsonl(config.training_dataset_path, resolved_s3_client)
    if not rows:
        raise ValueError("Training dataset is empty.")

    baseline_metrics_payload = _read_json(config.baseline_metrics_path, resolved_s3_client)
    baseline_metrics_map = baseline_metrics_payload.get("metrics")
    if not isinstance(baseline_metrics_map, dict):
        raise ValueError("Baseline metrics JSON must contain a metrics object.")

    split_rows: dict[str, list[dict[str, Any]]] = {"train": [], "valid": [], "test": []}
    for row in rows:
        split = str(row.get("split") or "").strip().lower()
        if split in split_rows:
            split_rows[split].append(row)

    if not split_rows["train"]:
        raise ValueError("Training dataset has no train rows.")

    feature_cols = _feature_columns(rows)
    if not feature_cols:
        raise ValueError("No feature columns available for training.")
    numeric_cols = _numeric_columns(rows, feature_cols)

    train_y: list[int] = []
    train_x: list[dict[str, Any]] = []
    for row in split_rows["train"]:
        label = _to_int_label(row.get("churn_label"))
        if label is None:
            continue
        train_y.append(1 if label == 1 else 0)
        train_x.append(_normalize_feature_row(row, feature_cols, numeric_cols))

    if not train_x:
        raise ValueError("No labeled train rows available.")
    if len(set(train_y)) < 2:
        raise ValueError("Train split requires both churn classes for GBDT.")

    vectorizer = DictVectorizer(sparse=False)
    x_train = vectorizer.fit_transform(train_x)

    model = GradientBoostingClassifier(
        random_state=config.random_seed,
        n_estimators=config.n_estimators,
        learning_rate=config.learning_rate,
        max_depth=config.max_depth,
    )
    model.fit(x_train, np.array(train_y, dtype=int))

    metrics: dict[str, SplitMetrics] = {}
    split_counts = {name: len(values) for name, values in split_rows.items()}
    split_probs: dict[str, np.ndarray] = {}
    split_matrix: dict[str, np.ndarray] = {}
    split_labels: dict[str, np.ndarray] = {}

    for split_name, split_data in split_rows.items():
        y_values: list[int] = []
        x_values: list[dict[str, Any]] = []
        for row in split_data:
            label = _to_int_label(row.get("churn_label"))
            if label is None:
                continue
            y_values.append(1 if label == 1 else 0)
            x_values.append(_normalize_feature_row(row, feature_cols, numeric_cols))

        if not y_values:
            metrics[split_name] = _compute_split_metrics(
                split_name=split_name,
                y_true=np.array([], dtype=int),
                y_prob=np.array([], dtype=float),
                top_k_fraction=config.top_k_fraction,
                calibration_bins=config.calibration_bins,
            )
            split_probs[split_name] = np.array([], dtype=float)
            split_matrix[split_name] = np.zeros((0, len(feature_cols)))
            split_labels[split_name] = np.array([], dtype=int)
            continue

        x_matrix = vectorizer.transform(x_values)
        y_true = np.array(y_values, dtype=int)
        y_prob = model.predict_proba(x_matrix)[:, 1]
        metrics[split_name] = _compute_split_metrics(
            split_name=split_name,
            y_true=y_true,
            y_prob=y_prob,
            top_k_fraction=config.top_k_fraction,
            calibration_bins=config.calibration_bins,
        )
        split_probs[split_name] = y_prob
        split_matrix[split_name] = x_matrix
        split_labels[split_name] = y_true

    if metrics["test"].row_count > 0:
        primary_eval_split = "test"
    elif metrics["valid"].row_count > 0:
        primary_eval_split = "valid"
    else:
        primary_eval_split = "train"

    baseline_split = (
        primary_eval_split
        if primary_eval_split in baseline_metrics_map
        else baseline_metrics_payload.get("primary_eval_split", "test")
    )
    baseline_split_metrics = baseline_metrics_map.get(baseline_split)
    if not isinstance(baseline_split_metrics, dict):
        raise ValueError(f"Baseline metrics missing split: {baseline_split}")

    current = metrics[primary_eval_split]
    baseline_pr_auc = float(baseline_split_metrics.get("pr_auc", 0.0))
    baseline_recall = float(baseline_split_metrics.get("recall_at_k", 0.0))
    baseline_brier = float(baseline_split_metrics.get("brier_score", 0.0))
    baseline_ece = float(baseline_split_metrics.get("expected_calibration_error", 0.0))

    improved_pr_auc = current.pr_auc > baseline_pr_auc
    improved_recall = current.recall_at_k > baseline_recall
    brier_increase = current.brier_score - baseline_brier
    ece_increase = current.expected_calibration_error - baseline_ece
    calibration_ok = (
        brier_increase <= config.max_brier_increase
        and ece_increase <= config.max_ece_increase
    )
    champion_pass = (improved_pr_auc or improved_recall) and calibration_ok
    if improved_pr_auc:
        improvement_dimension = "pr_auc"
    elif improved_recall:
        improvement_dimension = "recall_at_k"
    else:
        improvement_dimension = "none"

    feature_names = vectorizer.get_feature_names_out().tolist()
    top_importance = _top_feature_importance(
        model=model,
        feature_names=feature_names,
        top_k=config.top_feature_count,
    )
    shap_summary = _shap_summary(
        model=model,
        x_eval=split_matrix[primary_eval_split],
        feature_names=feature_names,
        y_prob=split_probs[primary_eval_split],
        random_seed=config.random_seed,
        top_k=config.top_feature_count,
        max_samples=config.max_shap_samples,
    )

    model_path = _join_location(config.model_output_dir, "model.pkl")
    metrics_path = _join_location(config.model_output_dir, "metrics.json")

    model_payload = {
        "trained_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "model": model,
        "vectorizer": vectorizer,
        "feature_columns": feature_cols,
        "numeric_columns": sorted(numeric_cols),
        "config": asdict(config),
    }
    _write_bytes(model_path, pickle.dumps(model_payload), resolved_s3_client)

    metrics_payload = {
        "primary_eval_split": primary_eval_split,
        "baseline_reference_split": baseline_split,
        "split_counts": split_counts,
        "metrics": {name: asdict(value) for name, value in metrics.items()},
        "comparison": {
            "baseline_pr_auc": baseline_pr_auc,
            "gbdt_pr_auc": current.pr_auc,
            "baseline_recall_at_k": baseline_recall,
            "gbdt_recall_at_k": current.recall_at_k,
            "baseline_brier_score": baseline_brier,
            "gbdt_brier_score": current.brier_score,
            "baseline_expected_calibration_error": baseline_ece,
            "gbdt_expected_calibration_error": current.expected_calibration_error,
            "brier_increase": brier_increase,
            "ece_increase": ece_increase,
            "improvement_dimension": improvement_dimension,
            "champion_pass": champion_pass,
        },
        "feature_importance": top_importance,
        "shap_summary": shap_summary,
    }
    _write_bytes(
        metrics_path,
        json.dumps(metrics_payload, indent=2, sort_keys=True).encode("utf-8"),
        resolved_s3_client,
    )

    report_lines = [
        "# Churn Model Comparison: Baseline vs GBDT",
        "",
        f"Primary evaluation split: `{primary_eval_split}`",
        f"Baseline comparison split: `{baseline_split}`",
        "",
        "## Comparison Metrics",
        "",
        "| Metric | Baseline | GBDT | Delta (GBDT - Baseline) |",
        "| --- | ---: | ---: | ---: |",
        f"| PR-AUC | {baseline_pr_auc:.4f} | {current.pr_auc:.4f} | {current.pr_auc - baseline_pr_auc:.4f} |",
        f"| Recall@K | {baseline_recall:.4f} | {current.recall_at_k:.4f} | {current.recall_at_k - baseline_recall:.4f} |",
        f"| Precision@K | {float(baseline_split_metrics.get('precision_at_k', 0.0)):.4f} | {current.precision_at_k:.4f} | {current.precision_at_k - float(baseline_split_metrics.get('precision_at_k', 0.0)):.4f} |",
        f"| Brier Score | {baseline_brier:.4f} | {current.brier_score:.4f} | {brier_increase:.4f} |",
        f"| Expected Calibration Error | {baseline_ece:.4f} | {current.expected_calibration_error:.4f} | {ece_increase:.4f} |",
        "",
        "## Champion Decision",
        "",
        f"- Improvement dimension: `{improvement_dimension}`",
        f"- Calibration guardrail pass: `{calibration_ok}`",
        f"- Champion criteria pass: `{champion_pass}`",
        "",
        "## Feature Importance (Top)",
        "",
    ]
    if top_importance:
        for item in top_importance:
            report_lines.append(f"- `{item['feature']}`: {item['importance']:.6f}")
    else:
        report_lines.append("- No non-zero feature importances were found.")

    report_lines.extend(["", "## SHAP Summary", ""])
    report_lines.append(f"- Method: `{shap_summary.get('method', 'unknown')}`")
    top_shap = shap_summary.get("top_features") or []
    if top_shap:
        for item in top_shap:
            score = item.get("mean_abs_shap")
            if score is None:
                score = item.get("mean_abs_effect", 0.0)
            report_lines.append(f"- `{item['feature']}`: {float(score):.6f}")
    else:
        report_lines.append("- No SHAP summary features available.")

    report_lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Model: `{model_path}`",
            f"- Metrics JSON: `{metrics_path}`",
        ]
    )
    _write_bytes(
        config.comparison_report_path,
        "\n".join(report_lines).encode("utf-8"),
        resolved_s3_client,
    )

    if not champion_pass:
        raise ValueError(
            "GBDT did not satisfy champion criteria: must improve baseline on "
            "PR-AUC or Recall@K without severe calibration loss."
        )

    return TrainResult(
        model_path=model_path,
        metrics_path=metrics_path,
        report_path=config.comparison_report_path,
        split_counts=split_counts,
        primary_eval_split=primary_eval_split,
        champion_pass=champion_pass,
        improvement_dimension=improvement_dimension,
        metrics=metrics,
    )


def _render_summary(result: TrainResult) -> str:
    primary = result.metrics[result.primary_eval_split]
    return "\n".join(
        [
            "GBDT churn training complete:",
            f"- primary_eval_split={result.primary_eval_split}",
            f"- pr_auc={primary.pr_auc:.4f}",
            f"- recall_at_k={primary.recall_at_k:.4f}",
            f"- precision_at_k={primary.precision_at_k:.4f}",
            f"- ece={primary.expected_calibration_error:.4f}",
            f"- champion_pass={result.champion_pass}",
            f"- improvement_dimension={result.improvement_dimension}",
            f"- model={result.model_path}",
            f"- report={result.report_path}",
        ]
    )


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    config = _load_config(args.config)
    result = train_gbdt(config=config)
    print(_render_summary(result))


if __name__ == "__main__":
    main()

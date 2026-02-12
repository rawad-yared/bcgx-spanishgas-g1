"""Train and evaluate a logistic-regression churn baseline."""

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
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.metrics import brier_score_loss


class S3ClientProtocol(Protocol):
    """Minimal S3 client protocol used by this module."""

    def get_object(self, **kwargs: Any) -> dict[str, Any]: ...

    def put_object(self, **kwargs: Any) -> dict[str, Any]: ...


@dataclass(frozen=True)
class BaselineConfig:
    training_dataset_path: str
    model_output_dir: str
    report_path: str
    random_seed: int
    max_iter: int
    regularization_c: float
    top_k_fraction: float
    calibration_bins: int


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
    metrics: dict[str, SplitMetrics]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate baseline churn logistic regression model."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to model config YAML.",
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


def _load_config(config_path: str) -> BaselineConfig:
    raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Config must be a YAML object.")

    paths = raw.get("paths") or {}
    model = raw.get("model") or {}
    evaluation = raw.get("evaluation") or {}

    training_dataset_path = str(paths.get("training_dataset_path", "")).strip()
    model_output_dir = str(paths.get("model_output_dir", "")).strip()
    report_path = str(paths.get("report_path", "")).strip()

    if not training_dataset_path or not model_output_dir or not report_path:
        raise ValueError(
            "Config must define paths.training_dataset_path, "
            "paths.model_output_dir, and paths.report_path."
        )

    random_seed = int(model.get("random_seed", 42))
    max_iter = int(model.get("max_iter", 1000))
    regularization_c = float(model.get("regularization_c", 1.0))
    top_k_fraction = float(evaluation.get("top_k_fraction", 0.2))
    calibration_bins = int(evaluation.get("calibration_bins", 10))

    if not (0 < top_k_fraction <= 1):
        raise ValueError("evaluation.top_k_fraction must be in (0, 1].")
    if calibration_bins <= 1:
        raise ValueError("evaluation.calibration_bins must be > 1.")

    return BaselineConfig(
        training_dataset_path=training_dataset_path,
        model_output_dir=model_output_dir,
        report_path=report_path,
        random_seed=random_seed,
        max_iter=max_iter,
        regularization_c=regularization_c,
        top_k_fraction=top_k_fraction,
        calibration_bins=calibration_bins,
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
    columns = sorted({key for row in rows for key in row if key not in reserved})
    return columns


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
            if value is None:
                normalized[column] = 0.0
            else:
                normalized[column] = float(value)
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


def train_baseline(
    config: BaselineConfig,
    s3_client: S3ClientProtocol | None = None,
) -> TrainResult:
    """Train baseline logistic regression and write artifacts."""

    resolved_s3_client = _resolve_s3_client(
        paths=[
            config.training_dataset_path,
            config.model_output_dir,
            config.report_path,
        ],
        s3_client=s3_client,
    )

    rows = _read_jsonl(config.training_dataset_path, resolved_s3_client)
    if not rows:
        raise ValueError("Training dataset is empty.")

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
        raise ValueError("Train split requires both churn classes for logistic regression.")

    vectorizer = DictVectorizer(sparse=False)
    x_train_matrix = vectorizer.fit_transform(train_x)
    model = LogisticRegression(
        random_state=config.random_seed,
        max_iter=config.max_iter,
        C=config.regularization_c,
        solver="lbfgs",
    )
    model.fit(x_train_matrix, np.array(train_y, dtype=int))

    metrics: dict[str, SplitMetrics] = {}
    split_counts = {name: len(values) for name, values in split_rows.items()}

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
            continue

        x_matrix = vectorizer.transform(x_values)
        y_prob = model.predict_proba(x_matrix)[:, 1]
        metrics[split_name] = _compute_split_metrics(
            split_name=split_name,
            y_true=np.array(y_values, dtype=int),
            y_prob=y_prob,
            top_k_fraction=config.top_k_fraction,
            calibration_bins=config.calibration_bins,
        )

    if metrics["test"].row_count > 0:
        primary_eval_split = "test"
    elif metrics["valid"].row_count > 0:
        primary_eval_split = "valid"
    else:
        primary_eval_split = "train"

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
        "split_counts": split_counts,
        "metrics": {name: asdict(value) for name, value in metrics.items()},
    }
    _write_bytes(
        metrics_path,
        json.dumps(metrics_payload, indent=2, sort_keys=True).encode("utf-8"),
        resolved_s3_client,
    )

    report_lines = [
        "# Churn Baseline Evaluation",
        "",
        f"Primary evaluation split: `{primary_eval_split}`",
        "",
        "## Metrics",
        "",
        "| Split | Rows | PR-AUC | Recall@K | Precision@K | Brier | ECE | Positive Rate | Avg Pred |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split_name in ("train", "valid", "test"):
        metric = metrics[split_name]
        report_lines.append(
            "| "
            f"{split_name} | {metric.row_count} | "
            f"{metric.pr_auc:.4f} | {metric.recall_at_k:.4f} | {metric.precision_at_k:.4f} | "
            f"{metric.brier_score:.4f} | {metric.expected_calibration_error:.4f} | "
            f"{metric.positive_rate:.4f} | {metric.avg_predicted_probability:.4f} |"
        )
    report_lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Model: `{model_path}`",
            f"- Metrics JSON: `{metrics_path}`",
            "",
            "## Reproducibility",
            "",
            f"- Random seed: `{config.random_seed}`",
            f"- Max iterations: `{config.max_iter}`",
            f"- Regularization C: `{config.regularization_c}`",
            f"- Top-K fraction: `{config.top_k_fraction}`",
            f"- Calibration bins: `{config.calibration_bins}`",
        ]
    )
    _write_bytes(
        config.report_path,
        "\n".join(report_lines).encode("utf-8"),
        resolved_s3_client,
    )

    return TrainResult(
        model_path=model_path,
        metrics_path=metrics_path,
        report_path=config.report_path,
        split_counts=split_counts,
        primary_eval_split=primary_eval_split,
        metrics=metrics,
    )


def _render_summary(result: TrainResult) -> str:
    primary = result.metrics[result.primary_eval_split]
    return "\n".join(
        [
            "Baseline churn training complete:",
            f"- primary_eval_split={result.primary_eval_split}",
            f"- pr_auc={primary.pr_auc:.4f}",
            f"- recall_at_k={primary.recall_at_k:.4f}",
            f"- precision_at_k={primary.precision_at_k:.4f}",
            f"- ece={primary.expected_calibration_error:.4f}",
            f"- model={result.model_path}",
            f"- report={result.report_path}",
        ]
    )


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    config = _load_config(args.config)
    result = train_baseline(config=config)
    print(_render_summary(result))


if __name__ == "__main__":
    main()

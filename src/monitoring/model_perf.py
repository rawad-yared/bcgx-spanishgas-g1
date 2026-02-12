"""Model performance monitoring with delayed labels."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any
from typing import Iterable
from typing import Mapping
from typing import Protocol

import yaml
from sklearn.metrics import average_precision_score


class S3ClientProtocol(Protocol):
    """Minimal S3 client protocol used by this module."""

    def get_object(self, **kwargs: Any) -> dict[str, Any]: ...

    def put_object(self, **kwargs: Any) -> dict[str, Any]: ...


@dataclass(frozen=True)
class MonitorConfig:
    scoring_path: str
    labels_path: str
    report_path: str
    score_customer_id_col: str
    score_value_col: str
    score_segment_col: str
    score_time_bucket_col: str
    label_customer_id_col: str
    label_value_col: str
    label_time_bucket_col: str
    label_horizon_days_col: str
    label_time_bucket_for_scores: str | None
    required_horizon_days: int | None
    top_k_fraction: float
    calibration_bins: int
    min_pr_auc: float
    min_recall_at_k: float
    max_segment_calibration_drift: float
    min_evaluation_rows: int


@dataclass(frozen=True)
class BucketMetrics:
    row_count: int
    positive_rate: float
    pr_auc: float
    recall_at_k: float
    precision_at_k: float
    expected_calibration_error: float


@dataclass(frozen=True)
class SegmentCalibrationDrift:
    segment: str
    current_bucket: str
    baseline_bucket: str
    current_ece: float
    baseline_ece: float
    calibration_drift: float


@dataclass(frozen=True)
class ModelPerfSummary:
    scoring_path: str
    labels_path: str
    report_path: str
    metrics_json_path: str
    evaluated_rows: int
    time_buckets: tuple[str, ...]
    current_bucket: str
    baseline_bucket: str
    retraining_trigger: bool
    retraining_reasons: tuple[str, ...]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute delayed-label model performance monitoring metrics."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to monitoring config YAML.",
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
            "S3 model performance monitoring requested but no S3 SDK is available. "
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


def _read_rows(path: str, s3_client: S3ClientProtocol | None) -> list[dict[str, Any]]:
    payload = _read_bytes(path, s3_client).decode("utf-8")
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        return [dict(row) for row in csv.DictReader(io.StringIO(payload))]

    rows: list[dict[str, Any]] = []
    for line in payload.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        decoded = json.loads(stripped)
        if isinstance(decoded, dict):
            rows.append(decoded)
    return rows


def _parse_iso_date(value: str) -> str:
    return datetime.strptime(value, "%Y-%m-%d").date().isoformat()


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _to_int(value: Any) -> int | None:
    numeric = _to_float(value)
    if numeric is None:
        return None
    return int(numeric)


def _clip_probability(value: float) -> float:
    return min(1.0, max(0.0, value))


def _recall_precision_at_k(
    y_true: list[int], y_prob: list[float], top_k_fraction: float
) -> tuple[float, float]:
    n = len(y_true)
    if n == 0:
        return 0.0, 0.0
    k = max(1, int(math.ceil(n * top_k_fraction)))
    ranked = sorted(range(n), key=lambda idx: y_prob[idx], reverse=True)[:k]
    true_positives = sum(y_true[idx] for idx in ranked)
    positives = sum(y_true)
    precision = true_positives / k if k > 0 else 0.0
    recall = true_positives / positives if positives > 0 else 0.0
    return recall, precision


def _expected_calibration_error(
    y_true: list[int], y_prob: list[float], bins: int
) -> float:
    n = len(y_true)
    if n == 0:
        return 0.0
    bins = max(2, bins)
    edges = [idx / bins for idx in range(bins + 1)]

    ece = 0.0
    for idx in range(bins):
        left, right = edges[idx], edges[idx + 1]
        bucket_indices = []
        for row_idx, probability in enumerate(y_prob):
            if idx == bins - 1:
                in_bucket = left <= probability <= right
            else:
                in_bucket = left <= probability < right
            if in_bucket:
                bucket_indices.append(row_idx)
        if not bucket_indices:
            continue

        bucket_true = [y_true[row_idx] for row_idx in bucket_indices]
        bucket_prob = [y_prob[row_idx] for row_idx in bucket_indices]
        avg_true = sum(bucket_true) / len(bucket_true)
        avg_prob = sum(bucket_prob) / len(bucket_prob)
        ece += (len(bucket_indices) / n) * abs(avg_true - avg_prob)
    return float(ece)


def _compute_bucket_metrics(
    y_true: list[int],
    y_prob: list[float],
    top_k_fraction: float,
    calibration_bins: int,
) -> BucketMetrics:
    if not y_true:
        return BucketMetrics(
            row_count=0,
            positive_rate=0.0,
            pr_auc=0.0,
            recall_at_k=0.0,
            precision_at_k=0.0,
            expected_calibration_error=0.0,
        )

    positive_rate = sum(y_true) / len(y_true)
    if len(set(y_true)) < 2:
        pr_auc = positive_rate
    else:
        pr_auc = float(average_precision_score(y_true, y_prob))
    recall_at_k, precision_at_k = _recall_precision_at_k(
        y_true=y_true,
        y_prob=y_prob,
        top_k_fraction=top_k_fraction,
    )
    ece = _expected_calibration_error(
        y_true=y_true,
        y_prob=y_prob,
        bins=calibration_bins,
    )
    return BucketMetrics(
        row_count=len(y_true),
        positive_rate=float(positive_rate),
        pr_auc=float(pr_auc),
        recall_at_k=float(recall_at_k),
        precision_at_k=float(precision_at_k),
        expected_calibration_error=float(ece),
    )


def _load_config(config_path: str) -> MonitorConfig:
    raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("Monitoring config must be a YAML object.")

    paths = raw.get("paths") or {}
    columns = raw.get("columns") or {}
    matching = raw.get("matching") or {}
    evaluation = raw.get("evaluation") or {}
    trigger_thresholds = raw.get("trigger_thresholds") or {}

    scoring_path = str(paths.get("scoring_path", "")).strip()
    labels_path = str(paths.get("labels_path", "")).strip()
    report_path = str(paths.get("report_path", "")).strip()
    if not scoring_path or not labels_path or not report_path:
        raise ValueError(
            "Config must include paths.scoring_path, paths.labels_path, and paths.report_path."
        )

    label_time_bucket_for_scores_raw = matching.get("label_time_bucket_for_scores")
    label_time_bucket_for_scores: str | None = None
    if label_time_bucket_for_scores_raw is not None:
        label_time_bucket_for_scores = _parse_iso_date(
            str(label_time_bucket_for_scores_raw).strip()
        )

    required_horizon_days_raw = matching.get("required_horizon_days")
    required_horizon_days = (
        int(required_horizon_days_raw) if required_horizon_days_raw is not None else None
    )

    top_k_fraction = float(evaluation.get("top_k_fraction", 0.2))
    calibration_bins = int(evaluation.get("calibration_bins", 10))
    if not (0 < top_k_fraction <= 1):
        raise ValueError("evaluation.top_k_fraction must be in (0, 1].")
    if calibration_bins <= 1:
        raise ValueError("evaluation.calibration_bins must be > 1.")

    return MonitorConfig(
        scoring_path=scoring_path,
        labels_path=labels_path,
        report_path=report_path,
        score_customer_id_col=str(columns.get("score_customer_id", "customer_id")).strip(),
        score_value_col=str(columns.get("score_value", "risk_score")).strip(),
        score_segment_col=str(columns.get("score_segment", "segment")).strip(),
        score_time_bucket_col=str(columns.get("score_time_bucket", "run_date")).strip(),
        label_customer_id_col=str(columns.get("label_customer_id", "customer_id")).strip(),
        label_value_col=str(columns.get("label_value", "churned_within_horizon")).strip(),
        label_time_bucket_col=str(columns.get("label_time_bucket", "label_date")).strip(),
        label_horizon_days_col=str(columns.get("label_horizon_days", "horizon_days")).strip(),
        label_time_bucket_for_scores=label_time_bucket_for_scores,
        required_horizon_days=required_horizon_days,
        top_k_fraction=top_k_fraction,
        calibration_bins=calibration_bins,
        min_pr_auc=float(trigger_thresholds.get("min_pr_auc", 0.60)),
        min_recall_at_k=float(trigger_thresholds.get("min_recall_at_k", 0.50)),
        max_segment_calibration_drift=float(
            trigger_thresholds.get("max_segment_calibration_drift", 0.05)
        ),
        min_evaluation_rows=int(trigger_thresholds.get("min_evaluation_rows", 50)),
    )


def run_model_performance_monitoring(
    config: MonitorConfig,
    s3_client: S3ClientProtocol | None = None,
) -> ModelPerfSummary:
    """Compute delayed-label performance metrics and retraining trigger."""

    resolved_s3_client = _resolve_s3_client(
        paths=[config.scoring_path, config.labels_path, config.report_path],
        s3_client=s3_client,
    )

    scored_rows = _read_rows(config.scoring_path, resolved_s3_client)
    label_rows = _read_rows(config.labels_path, resolved_s3_client)
    if not scored_rows:
        raise ValueError("Scoring input is empty.")
    if not label_rows:
        raise ValueError("Labels input is empty.")

    labels_by_key: dict[tuple[str, str], int] = {}
    for row in label_rows:
        customer_id = str(row.get(config.label_customer_id_col) or "").strip()
        time_bucket_raw = str(row.get(config.label_time_bucket_col) or "").strip()
        label_value_raw = _to_int(row.get(config.label_value_col))
        if not customer_id or not time_bucket_raw or label_value_raw is None:
            continue
        try:
            time_bucket = _parse_iso_date(time_bucket_raw)
        except ValueError:
            continue

        if config.required_horizon_days is not None:
            horizon = _to_int(row.get(config.label_horizon_days_col))
            if horizon != config.required_horizon_days:
                continue
        labels_by_key[(customer_id, time_bucket)] = 1 if label_value_raw >= 1 else 0

    evaluation_rows: list[dict[str, Any]] = []
    for row in scored_rows:
        customer_id = str(row.get(config.score_customer_id_col) or "").strip()
        time_bucket_raw = str(row.get(config.score_time_bucket_col) or "").strip()
        segment = str(row.get(config.score_segment_col) or "unknown").strip() or "unknown"
        score_value = _to_float(row.get(config.score_value_col))
        if not customer_id or not time_bucket_raw or score_value is None:
            continue
        score_bucket = time_bucket_raw
        try:
            _parse_iso_date(score_bucket)
        except ValueError:
            pass

        label_bucket = config.label_time_bucket_for_scores or score_bucket
        try:
            label_bucket = _parse_iso_date(label_bucket)
        except ValueError:
            pass

        label = labels_by_key.get((customer_id, label_bucket))
        if label is None:
            continue

        evaluation_rows.append(
            {
                "customer_id": customer_id,
                "time_bucket": score_bucket,
                "segment": segment,
                "score": _clip_probability(score_value),
                "label": label,
            }
        )

    if not evaluation_rows:
        raise ValueError("No scored rows could be matched with delayed labels.")

    buckets = sorted({str(row["time_bucket"]) for row in evaluation_rows})
    current_bucket = buckets[-1]
    baseline_bucket = buckets[-2] if len(buckets) > 1 else buckets[-1]

    metrics_by_bucket: dict[str, BucketMetrics] = {}
    for bucket in buckets:
        bucket_rows = [row for row in evaluation_rows if row["time_bucket"] == bucket]
        y_true = [int(row["label"]) for row in bucket_rows]
        y_prob = [float(row["score"]) for row in bucket_rows]
        metrics_by_bucket[bucket] = _compute_bucket_metrics(
            y_true=y_true,
            y_prob=y_prob,
            top_k_fraction=config.top_k_fraction,
            calibration_bins=config.calibration_bins,
        )

    current_rows = [row for row in evaluation_rows if row["time_bucket"] == current_bucket]
    baseline_rows = [row for row in evaluation_rows if row["time_bucket"] == baseline_bucket]

    segment_metrics_current: dict[str, BucketMetrics] = {}
    for segment in sorted({str(row["segment"]) for row in current_rows}):
        rows = [row for row in current_rows if row["segment"] == segment]
        segment_metrics_current[segment] = _compute_bucket_metrics(
            y_true=[int(row["label"]) for row in rows],
            y_prob=[float(row["score"]) for row in rows],
            top_k_fraction=config.top_k_fraction,
            calibration_bins=config.calibration_bins,
        )

    segment_metrics_baseline: dict[str, BucketMetrics] = {}
    for segment in sorted({str(row["segment"]) for row in baseline_rows}):
        rows = [row for row in baseline_rows if row["segment"] == segment]
        segment_metrics_baseline[segment] = _compute_bucket_metrics(
            y_true=[int(row["label"]) for row in rows],
            y_prob=[float(row["score"]) for row in rows],
            top_k_fraction=config.top_k_fraction,
            calibration_bins=config.calibration_bins,
        )

    segment_calibration_drift: list[SegmentCalibrationDrift] = []
    all_segments = sorted(set(segment_metrics_current) | set(segment_metrics_baseline))
    for segment in all_segments:
        current_ece = segment_metrics_current.get(
            segment,
            BucketMetrics(0, 0.0, 0.0, 0.0, 0.0, 0.0),
        ).expected_calibration_error
        baseline_ece = segment_metrics_baseline.get(
            segment,
            BucketMetrics(0, 0.0, 0.0, 0.0, 0.0, 0.0),
        ).expected_calibration_error
        segment_calibration_drift.append(
            SegmentCalibrationDrift(
                segment=segment,
                current_bucket=current_bucket,
                baseline_bucket=baseline_bucket,
                current_ece=current_ece,
                baseline_ece=baseline_ece,
                calibration_drift=current_ece - baseline_ece,
            )
        )

    current_overall = metrics_by_bucket[current_bucket]
    retraining_reasons: list[str] = []
    if len(current_rows) < config.min_evaluation_rows:
        retraining_reasons.append(
            "insufficient_evaluation_rows"
            f" ({len(current_rows)}<{config.min_evaluation_rows})"
        )
    if current_overall.pr_auc < config.min_pr_auc:
        retraining_reasons.append(
            f"pr_auc_below_threshold ({current_overall.pr_auc:.4f}<{config.min_pr_auc:.4f})"
        )
    if current_overall.recall_at_k < config.min_recall_at_k:
        retraining_reasons.append(
            "recall_at_k_below_threshold "
            f"({current_overall.recall_at_k:.4f}<{config.min_recall_at_k:.4f})"
        )
    max_drift = max(
        (abs(item.calibration_drift) for item in segment_calibration_drift),
        default=0.0,
    )
    if max_drift > config.max_segment_calibration_drift:
        retraining_reasons.append(
            "segment_calibration_drift_exceeded "
            f"({max_drift:.4f}>{config.max_segment_calibration_drift:.4f})"
        )

    retraining_trigger = bool(retraining_reasons)

    report_path = config.report_path
    report_parent = str(Path(report_path).parent)
    metrics_json_path = str(Path(report_parent) / "model_perf_metrics.json")
    if _is_s3_uri(report_path):
        report_dir = report_path.rsplit("/", 1)[0]
        metrics_json_path = f"{report_dir}/model_perf_metrics.json"

    metrics_payload = {
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "scoring_path": config.scoring_path,
        "labels_path": config.labels_path,
        "evaluated_rows": len(evaluation_rows),
        "time_buckets": buckets,
        "current_bucket": current_bucket,
        "baseline_bucket": baseline_bucket,
        "overall_by_time_bucket": {
            bucket: asdict(metrics_by_bucket[bucket]) for bucket in buckets
        },
        "segment_metrics_current_bucket": {
            segment: asdict(metric)
            for segment, metric in sorted(segment_metrics_current.items())
        },
        "segment_calibration_drift": [asdict(item) for item in segment_calibration_drift],
        "retraining_trigger": {
            "value": retraining_trigger,
            "reasons": retraining_reasons,
        },
        "thresholds": {
            "min_pr_auc": config.min_pr_auc,
            "min_recall_at_k": config.min_recall_at_k,
            "max_segment_calibration_drift": config.max_segment_calibration_drift,
            "min_evaluation_rows": config.min_evaluation_rows,
        },
    }
    _write_bytes(
        metrics_json_path,
        json.dumps(metrics_payload, indent=2, sort_keys=True).encode("utf-8"),
        resolved_s3_client,
    )

    report_lines = [
        "# Model Performance Monitoring",
        "",
        f"- Scoring path: `{config.scoring_path}`",
        f"- Labels path: `{config.labels_path}`",
        f"- Evaluated rows: `{len(evaluation_rows)}`",
        f"- Current bucket: `{current_bucket}`",
        f"- Baseline bucket: `{baseline_bucket}`",
        "",
        "## Overall Performance By Time Bucket",
        "",
        "| Time Bucket | Rows | Positive Rate | PR-AUC | Recall@K | Precision@K | ECE |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for bucket in buckets:
        metric = metrics_by_bucket[bucket]
        report_lines.append(
            f"| {bucket} | {metric.row_count} | {metric.positive_rate:.4f} | "
            f"{metric.pr_auc:.4f} | {metric.recall_at_k:.4f} | "
            f"{metric.precision_at_k:.4f} | {metric.expected_calibration_error:.4f} |"
        )

    report_lines.extend(
        [
            "",
            f"## Segment Metrics (Current Bucket: {current_bucket})",
            "",
            "| Segment | Rows | PR-AUC | Recall@K | ECE | Calibration Drift vs Baseline |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    drift_by_segment = {
        item.segment: item.calibration_drift for item in segment_calibration_drift
    }
    for segment, metric in sorted(segment_metrics_current.items()):
        report_lines.append(
            f"| {segment} | {metric.row_count} | {metric.pr_auc:.4f} | "
            f"{metric.recall_at_k:.4f} | {metric.expected_calibration_error:.4f} | "
            f"{drift_by_segment.get(segment, 0.0):.4f} |"
        )

    report_lines.extend(
        [
            "",
            "## Retraining Trigger",
            "",
            f"- Trigger: `{retraining_trigger}`",
        ]
    )
    if retraining_reasons:
        for reason in retraining_reasons:
            report_lines.append(f"- Reason: `{reason}`")
    else:
        report_lines.append("- Reason: `none`")

    _write_bytes(
        report_path,
        "\n".join(report_lines).encode("utf-8"),
        resolved_s3_client,
    )

    return ModelPerfSummary(
        scoring_path=config.scoring_path,
        labels_path=config.labels_path,
        report_path=report_path,
        metrics_json_path=metrics_json_path,
        evaluated_rows=len(evaluation_rows),
        time_buckets=tuple(buckets),
        current_bucket=current_bucket,
        baseline_bucket=baseline_bucket,
        retraining_trigger=retraining_trigger,
        retraining_reasons=tuple(retraining_reasons),
    )


def _render_summary(summary: ModelPerfSummary) -> str:
    reason_text = ", ".join(summary.retraining_reasons) if summary.retraining_reasons else "none"
    return "\n".join(
        [
            "Model performance monitoring complete:",
            f"- scoring_path={summary.scoring_path}",
            f"- labels_path={summary.labels_path}",
            f"- evaluated_rows={summary.evaluated_rows}",
            f"- current_bucket={summary.current_bucket}",
            f"- baseline_bucket={summary.baseline_bucket}",
            f"- retraining_trigger={summary.retraining_trigger}",
            f"- retraining_reasons={reason_text}",
            f"- metrics_json={summary.metrics_json_path}",
            f"- report={summary.report_path}",
        ]
    )


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    config = _load_config(args.config)
    summary = run_model_performance_monitoring(config=config)
    print(_render_summary(summary))


if __name__ == "__main__":
    main()

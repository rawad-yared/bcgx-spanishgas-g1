"""Data drift monitoring for feature snapshots."""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any
from typing import Iterable
from typing import Protocol


DEFAULT_KEY_FEATURES: tuple[str, ...] = (
    "tenure_days",
    "days_to_contract_end",
    "price_vs_benchmark_delta",
    "consumption_volatility_90d",
    "interaction_count_90d",
    "negative_consumption_flag",
)
EPSILON = 1e-6


class S3ClientProtocol(Protocol):
    """Minimal S3 client protocol used by this module."""

    def get_object(self, **kwargs: Any) -> dict[str, Any]: ...

    def put_object(self, **kwargs: Any) -> dict[str, Any]: ...

    def list_objects_v2(self, **kwargs: Any) -> dict[str, Any]: ...


@dataclass(frozen=True)
class FeatureDriftMetric:
    feature: str
    missingness_current: float
    missingness_baseline: float
    missingness_delta: float
    psi: float
    missingness_exceeds_threshold: bool
    psi_exceeds_threshold: bool
    threshold_exceeded: bool


@dataclass(frozen=True)
class DataDriftSummary:
    run_date: str
    current_asof_date: str
    baseline_asof_date: str
    baseline_source: str
    current_row_count: int
    baseline_row_count: int
    features_evaluated: int
    threshold_exceeded: bool
    metrics_json_path: str
    report_path: str


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run data drift checks for feature snapshots.")
    parser.add_argument(
        "--run-date",
        required=True,
        help="Run date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--gold-root",
        default=os.environ.get("SPANISHGAS_GOLD_ROOT", "data/gold/"),
        help="Gold root location (local path or s3:// URI).",
    )
    parser.add_argument(
        "--artifacts-root",
        default=os.environ.get("SPANISHGAS_ARTIFACTS_ROOT", "artifacts/"),
        help="Artifacts root location (local path or s3:// URI).",
    )
    parser.add_argument(
        "--current-asof-date",
        default=None,
        help="Optional explicit current as-of date YYYY-MM-DD.",
    )
    parser.add_argument(
        "--baseline-asof-date",
        default=None,
        help="Optional explicit baseline as-of date YYYY-MM-DD.",
    )
    parser.add_argument(
        "--baseline-run-date",
        default=None,
        help="Optional baseline run date for deriving baseline as-of date.",
    )
    parser.add_argument(
        "--features",
        default=None,
        help="Comma-separated feature list to monitor.",
    )
    parser.add_argument(
        "--psi-threshold",
        type=float,
        default=0.2,
        help="PSI threshold for drift flagging.",
    )
    parser.add_argument(
        "--missingness-delta-threshold",
        type=float,
        default=0.05,
        help="Absolute missingness delta threshold for drift flagging.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=10,
        help="Number of bins used for numeric PSI.",
    )
    return parser.parse_args(argv)


def _parse_iso_date(value: str) -> str:
    return datetime.strptime(value, "%Y-%m-%d").date().isoformat()


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
            "S3 data drift monitoring requested but no S3 SDK is available. "
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
    rows: list[dict[str, Any]] = []
    for line in _read_bytes(path, s3_client).decode("utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        decoded = json.loads(stripped)
        if isinstance(decoded, dict):
            rows.append(decoded)
    return rows


def _extract_asof_date_from_key(value: str) -> str | None:
    token = "asof_date="
    if token not in value:
        return None
    segment = value.split(token, 1)[1]
    candidate = segment.split("/", 1)[0]
    try:
        return _parse_iso_date(candidate)
    except ValueError:
        return None


def _available_feature_asof_dates(
    gold_root: str,
    s3_client: S3ClientProtocol | None,
) -> list[str]:
    if _is_s3_uri(gold_root):
        if s3_client is None:
            return []
        bucket, key_prefix = _parse_s3_uri(gold_root)
        prefix = (
            f"{key_prefix.rstrip('/')}/customer_features_asof_date/"
            if key_prefix
            else "customer_features_asof_date/"
        )
        contents: list[dict[str, Any]] = []
        continuation: str | None = None
        while True:
            kwargs: dict[str, Any] = {"Bucket": bucket, "Prefix": prefix}
            if continuation is not None:
                kwargs["ContinuationToken"] = continuation
            response = s3_client.list_objects_v2(**kwargs)
            contents.extend(response.get("Contents") or [])
            if not response.get("IsTruncated"):
                break
            continuation = response.get("NextContinuationToken")
            if not continuation:
                break
        asof_dates = {
            asof
            for item in contents
            for asof in [_extract_asof_date_from_key(str(item.get("Key", "")))]
            if asof is not None
        }
        return sorted(asof_dates)

    root = Path(gold_root) / "customer_features_asof_date"
    if not root.exists():
        return []
    asof_dates: set[str] = set()
    for child in root.iterdir():
        if not child.is_dir() or not child.name.startswith("asof_date="):
            continue
        candidate = child.name.split("=", 1)[1]
        try:
            asof_dates.add(_parse_iso_date(candidate))
        except ValueError:
            continue
    return sorted(asof_dates)


def _read_scoring_asof_date(
    run_date: str,
    gold_root: str,
    s3_client: S3ClientProtocol | None,
) -> str | None:
    path = _join_location(gold_root, f"scoring/run_date={run_date}/scores.jsonl")
    try:
        rows = _read_jsonl(path, s3_client)
    except FileNotFoundError:
        return None
    for row in rows:
        asof = str(row.get("asof_date") or "").strip()
        if not asof:
            continue
        try:
            return _parse_iso_date(asof)
        except ValueError:
            continue
    return None


def _resolve_current_asof_date(
    run_date: str,
    explicit_current_asof: str | None,
    gold_root: str,
    s3_client: S3ClientProtocol | None,
) -> str:
    if explicit_current_asof is not None:
        return _parse_iso_date(explicit_current_asof)

    scoring_asof = _read_scoring_asof_date(run_date, gold_root, s3_client)
    if scoring_asof is not None:
        return scoring_asof

    run_iso = _parse_iso_date(run_date)
    feature_dates = _available_feature_asof_dates(gold_root, s3_client)
    if not feature_dates:
        raise FileNotFoundError(
            f"No customer feature snapshots found under {gold_root}."
        )

    eligible = [value for value in feature_dates if value <= run_iso]
    if eligible:
        return eligible[-1]
    return feature_dates[-1]


def _resolve_baseline_asof_date(
    current_asof_date: str,
    explicit_baseline_asof: str | None,
    baseline_run_date: str | None,
    gold_root: str,
    s3_client: S3ClientProtocol | None,
) -> tuple[str, str]:
    if explicit_baseline_asof is not None:
        return _parse_iso_date(explicit_baseline_asof), "explicit_baseline_asof_date"

    if baseline_run_date is not None:
        baseline_from_run = _read_scoring_asof_date(
            _parse_iso_date(baseline_run_date),
            gold_root,
            s3_client,
        )
        if baseline_from_run is not None:
            return baseline_from_run, "baseline_run_date"

    feature_dates = _available_feature_asof_dates(gold_root, s3_client)
    prior = [value for value in feature_dates if value < current_asof_date]
    if prior:
        return prior[-1], "previous_feature_snapshot"

    return current_asof_date, "fallback_current_as_baseline"


def _feature_path(gold_root: str, asof_date: str) -> str:
    return _join_location(
        gold_root,
        (
            "customer_features_asof_date/"
            f"asof_date={asof_date}/customer_features_asof_date.jsonl"
        ),
    )


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def _to_float(value: Any) -> float | None:
    if _is_missing(value):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except ValueError:
        return None


def _non_missing(values: list[Any]) -> list[Any]:
    return [value for value in values if not _is_missing(value)]


def _is_numeric_feature(values: list[Any]) -> bool:
    observed = _non_missing(values)
    if not observed:
        return True
    return all(_to_float(value) is not None for value in observed)


def _quantile_edges(values: list[float], bins: int) -> list[float]:
    if not values:
        return []
    ordered = sorted(values)
    if len(ordered) == 1:
        return [ordered[0], ordered[0]]

    bins = max(1, bins)
    edges: list[float] = []
    for idx in range(bins + 1):
        position = idx * (len(ordered) - 1) / bins
        left_index = int(math.floor(position))
        right_index = int(math.ceil(position))
        if left_index == right_index:
            quantile = ordered[left_index]
        else:
            left_value = ordered[left_index]
            right_value = ordered[right_index]
            weight = position - left_index
            quantile = left_value + (right_value - left_value) * weight
        edges.append(float(quantile))

    unique = [edges[0]]
    for edge in edges[1:]:
        if edge > unique[-1]:
            unique.append(edge)
    if len(unique) == 1:
        unique.append(unique[0])
    return unique


def _bucket_index(value: float, edges: list[float]) -> int:
    if not edges:
        return 0
    if value <= edges[0]:
        return 0
    for idx in range(1, len(edges)):
        if value <= edges[idx]:
            return idx - 1
    return len(edges) - 2


def _distribution_numeric(values: list[float], edges: list[float]) -> list[float]:
    bucket_count = max(1, len(edges) - 1)
    counts = [0.0] * bucket_count
    if not values:
        return [0.0] * bucket_count
    for value in values:
        index = _bucket_index(value, edges)
        counts[index] += 1.0
    total = float(sum(counts))
    if total == 0:
        return [0.0] * bucket_count
    return [count / total for count in counts]


def _distribution_categorical(values: list[Any], categories: list[str]) -> list[float]:
    counts = {category: 0.0 for category in categories}
    if not values:
        return [0.0 for _ in categories]
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0.0) + 1.0
    total = float(sum(counts.values()))
    if total == 0:
        return [0.0 for _ in categories]
    return [counts.get(category, 0.0) / total for category in categories]


def _psi_from_distributions(
    baseline_distribution: list[float],
    current_distribution: list[float],
) -> float:
    total = 0.0
    for baseline_share, current_share in zip(
        baseline_distribution, current_distribution
    ):
        adjusted_baseline = baseline_share if baseline_share > 0 else EPSILON
        adjusted_current = current_share if current_share > 0 else EPSILON
        total += (adjusted_current - adjusted_baseline) * math.log(
            adjusted_current / adjusted_baseline
        )
    return float(total)


def _compute_feature_psi(
    baseline_values: list[Any],
    current_values: list[Any],
    bins: int,
) -> float:
    baseline_observed = _non_missing(baseline_values)
    current_observed = _non_missing(current_values)
    if not baseline_observed and not current_observed:
        return 0.0

    values_for_type = baseline_observed or current_observed
    if _is_numeric_feature(values_for_type):
        baseline_numeric = [
            numeric
            for numeric in (_to_float(value) for value in baseline_observed)
            if numeric is not None
        ]
        current_numeric = [
            numeric
            for numeric in (_to_float(value) for value in current_observed)
            if numeric is not None
        ]
        if not baseline_numeric and not current_numeric:
            return 0.0
        if not baseline_numeric:
            baseline_numeric = current_numeric

        edges = _quantile_edges(baseline_numeric, bins=bins)
        baseline_dist = _distribution_numeric(baseline_numeric, edges)
        current_dist = _distribution_numeric(current_numeric, edges)
        return _psi_from_distributions(baseline_dist, current_dist)

    categories = sorted({str(value) for value in baseline_observed + current_observed})
    baseline_dist = _distribution_categorical(baseline_observed, categories)
    current_dist = _distribution_categorical(current_observed, categories)
    return _psi_from_distributions(baseline_dist, current_dist)


def run_data_drift_monitoring(
    run_date: str,
    gold_root: str | Path = "data/gold/",
    artifacts_root: str | Path = "artifacts/",
    current_asof_date: str | None = None,
    baseline_asof_date: str | None = None,
    baseline_run_date: str | None = None,
    features: list[str] | None = None,
    psi_threshold: float = 0.2,
    missingness_delta_threshold: float = 0.05,
    bins: int = 10,
    s3_client: S3ClientProtocol | None = None,
) -> DataDriftSummary:
    """Compute feature drift between current and baseline snapshots."""

    run_date_iso = _parse_iso_date(str(run_date))
    gold_root_str = str(gold_root)
    artifacts_root_str = str(artifacts_root)

    resolved_s3_client = _resolve_s3_client(
        paths=[gold_root_str, artifacts_root_str],
        s3_client=s3_client,
    )

    current_asof = _resolve_current_asof_date(
        run_date=run_date_iso,
        explicit_current_asof=current_asof_date,
        gold_root=gold_root_str,
        s3_client=resolved_s3_client,
    )
    baseline_asof, baseline_source = _resolve_baseline_asof_date(
        current_asof_date=current_asof,
        explicit_baseline_asof=baseline_asof_date,
        baseline_run_date=baseline_run_date,
        gold_root=gold_root_str,
        s3_client=resolved_s3_client,
    )

    current_rows = _read_jsonl(_feature_path(gold_root_str, current_asof), resolved_s3_client)
    baseline_rows = _read_jsonl(_feature_path(gold_root_str, baseline_asof), resolved_s3_client)
    if not current_rows:
        raise ValueError("Current feature snapshot is empty.")
    if not baseline_rows:
        raise ValueError("Baseline feature snapshot is empty.")

    columns = sorted({key for row in current_rows + baseline_rows for key in row})
    feature_set = [name for name in DEFAULT_KEY_FEATURES if name in columns]
    if features:
        requested = [feature.strip() for feature in features if feature.strip()]
        feature_set = [feature for feature in requested if feature in columns]
    if not feature_set:
        feature_set = [
            column for column in columns if column not in {"customer_id", "asof_date", "feature_version"}
        ]

    metrics: list[FeatureDriftMetric] = []
    threshold_exceeded = False
    for feature in feature_set:
        current_values = [row.get(feature) for row in current_rows]
        baseline_values = [row.get(feature) for row in baseline_rows]

        missing_current = sum(1 for value in current_values if _is_missing(value))
        missing_baseline = sum(1 for value in baseline_values if _is_missing(value))
        missingness_current = missing_current / len(current_values) if current_values else 0.0
        missingness_baseline = (
            missing_baseline / len(baseline_values) if baseline_values else 0.0
        )
        missingness_delta = missingness_current - missingness_baseline

        psi_value = _compute_feature_psi(
            baseline_values=baseline_values,
            current_values=current_values,
            bins=bins,
        )
        missing_flag = abs(missingness_delta) > missingness_delta_threshold
        psi_flag = psi_value > psi_threshold
        feature_flag = missing_flag or psi_flag
        threshold_exceeded = threshold_exceeded or feature_flag

        metrics.append(
            FeatureDriftMetric(
                feature=feature,
                missingness_current=missingness_current,
                missingness_baseline=missingness_baseline,
                missingness_delta=missingness_delta,
                psi=psi_value,
                missingness_exceeds_threshold=missing_flag,
                psi_exceeds_threshold=psi_flag,
                threshold_exceeded=feature_flag,
            )
        )

    monitoring_root = _join_location(artifacts_root_str, "monitoring")
    metrics_json_path = _join_location(
        monitoring_root, f"data_drift_run_date={run_date_iso}.json"
    )
    report_path = _join_location(
        monitoring_root, f"data_drift_run_date={run_date_iso}.md"
    )

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "run_date": run_date_iso,
        "current_asof_date": current_asof,
        "baseline_asof_date": baseline_asof,
        "baseline_source": baseline_source,
        "thresholds": {
            "psi_threshold": psi_threshold,
            "missingness_delta_threshold": missingness_delta_threshold,
        },
        "row_counts": {
            "current": len(current_rows),
            "baseline": len(baseline_rows),
        },
        "threshold_exceeded": threshold_exceeded,
        "feature_metrics": [asdict(metric) for metric in metrics],
    }
    _write_bytes(
        metrics_json_path,
        json.dumps(payload, indent=2, sort_keys=True).encode("utf-8"),
        resolved_s3_client,
    )

    report_lines = [
        "# Data Drift Report",
        "",
        f"- Run date: `{run_date_iso}`",
        f"- Current as-of date: `{current_asof}`",
        f"- Baseline as-of date: `{baseline_asof}`",
        f"- Baseline source: `{baseline_source}`",
        f"- Current rows: `{len(current_rows)}`",
        f"- Baseline rows: `{len(baseline_rows)}`",
        f"- PSI threshold: `{psi_threshold}`",
        f"- Missingness delta threshold: `{missingness_delta_threshold}`",
        f"- Any threshold exceeded: `{threshold_exceeded}`",
        "",
        "## Feature Drift Metrics",
        "",
        "| Feature | Missingness Current | Missingness Baseline | Missingness Delta | PSI | Missing Flag | PSI Flag |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for metric in metrics:
        report_lines.append(
            f"| {metric.feature} | {metric.missingness_current:.6f} | "
            f"{metric.missingness_baseline:.6f} | {metric.missingness_delta:.6f} | "
            f"{metric.psi:.6f} | {metric.missingness_exceeds_threshold} | "
            f"{metric.psi_exceeds_threshold} |"
        )
    _write_bytes(
        report_path,
        "\n".join(report_lines).encode("utf-8"),
        resolved_s3_client,
    )

    return DataDriftSummary(
        run_date=run_date_iso,
        current_asof_date=current_asof,
        baseline_asof_date=baseline_asof,
        baseline_source=baseline_source,
        current_row_count=len(current_rows),
        baseline_row_count=len(baseline_rows),
        features_evaluated=len(metrics),
        threshold_exceeded=threshold_exceeded,
        metrics_json_path=metrics_json_path,
        report_path=report_path,
    )


def _render_summary(summary: DataDriftSummary) -> str:
    return "\n".join(
        [
            "Data drift monitoring complete:",
            f"- run_date={summary.run_date}",
            f"- current_asof_date={summary.current_asof_date}",
            f"- baseline_asof_date={summary.baseline_asof_date}",
            f"- baseline_source={summary.baseline_source}",
            f"- current_row_count={summary.current_row_count}",
            f"- baseline_row_count={summary.baseline_row_count}",
            f"- features_evaluated={summary.features_evaluated}",
            f"- threshold_exceeded={summary.threshold_exceeded}",
            f"- metrics_json={summary.metrics_json_path}",
            f"- report={summary.report_path}",
        ]
    )


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    feature_list = (
        [item.strip() for item in str(args.features).split(",")]
        if args.features is not None
        else None
    )
    summary = run_data_drift_monitoring(
        run_date=args.run_date,
        gold_root=args.gold_root,
        artifacts_root=args.artifacts_root,
        current_asof_date=args.current_asof_date,
        baseline_asof_date=args.baseline_asof_date,
        baseline_run_date=args.baseline_run_date,
        features=feature_list,
        psi_threshold=args.psi_threshold,
        missingness_delta_threshold=args.missingness_delta_threshold,
        bins=args.bins,
    )
    print(_render_summary(summary))


if __name__ == "__main__":
    main()

"""Offer simulation for profit-aware recommendation evaluation."""

from __future__ import annotations

import argparse
import csv
import io
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any
from typing import Iterable
from typing import Mapping
from typing import Protocol

import yaml


class S3ClientProtocol(Protocol):
    """Minimal S3 client protocol used by this module."""

    def get_object(self, **kwargs: Any) -> dict[str, Any]: ...

    def put_object(self, **kwargs: Any) -> dict[str, Any]: ...


@dataclass(frozen=True)
class RiskBucket:
    name: str
    min_inclusive: float
    max_inclusive: float


@dataclass(frozen=True)
class SimulationConfig:
    input_path: str
    metrics_output_path: str
    report_path: str
    customer_id_col: str
    segment_col: str
    churn_probability_col: str
    acceptance_probability_col: str
    margin_col: str
    discount_level_col: str
    default_acceptance_probability: float
    retention_given_acceptance: float
    discount_is_fraction: bool
    risk_buckets: tuple[RiskBucket, ...]


@dataclass(frozen=True)
class SimulationSummary:
    input_path: str
    report_path: str
    metrics_output_path: str
    total_rows: int
    simulated_rows: int
    skipped_rows: int
    overall_roi: float | None
    overall_incremental_margin: float


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run offer simulation and profiling.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to simulation config YAML.",
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
            "S3 simulation requested but no S3 SDK is available. "
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


def _read_records(path: str, s3_client: S3ClientProtocol | None) -> list[dict[str, Any]]:
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


def _clip_probability(value: float) -> float:
    return min(1.0, max(0.0, value))


def _risk_bucket_name(churn_probability: float, buckets: tuple[RiskBucket, ...]) -> str:
    for bucket in buckets:
        if bucket.min_inclusive <= churn_probability <= bucket.max_inclusive:
            return bucket.name
    return "unbucketed"


def _safe_div(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _load_config(config_path: str) -> SimulationConfig:
    raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("Config must be a YAML object.")

    paths = raw.get("paths") or {}
    columns = raw.get("columns") or {}
    assumptions = raw.get("assumptions") or {}
    risk_buckets_raw = raw.get("risk_buckets") or []

    input_path = str(paths.get("input_path", "")).strip()
    metrics_output_path = str(paths.get("metrics_output_path", "")).strip()
    report_path = str(paths.get("report_path", "")).strip()
    if not input_path or not metrics_output_path or not report_path:
        raise ValueError(
            "Config must include paths.input_path, paths.metrics_output_path, and paths.report_path."
        )

    customer_id_col = str(columns.get("customer_id", "customer_id")).strip()
    segment_col = str(columns.get("segment", "segment")).strip()
    churn_probability_col = str(columns.get("churn_probability", "churn_probability")).strip()
    acceptance_probability_col = str(
        columns.get("acceptance_probability", "acceptance_probability")
    ).strip()
    margin_col = str(columns.get("margin", "margin_eur")).strip()
    discount_level_col = str(columns.get("discount_level", "discount_level")).strip()

    default_acceptance_probability = float(
        assumptions.get("default_acceptance_probability", 0.25)
    )
    retention_given_acceptance = float(assumptions.get("retention_given_acceptance", 1.0))
    discount_is_fraction = bool(assumptions.get("discount_is_fraction", True))

    if not (0 <= default_acceptance_probability <= 1):
        raise ValueError("assumptions.default_acceptance_probability must be in [0, 1].")
    if not (0 <= retention_given_acceptance <= 1):
        raise ValueError("assumptions.retention_given_acceptance must be in [0, 1].")

    buckets: list[RiskBucket] = []
    for bucket in risk_buckets_raw:
        if not isinstance(bucket, Mapping):
            continue
        name = str(bucket.get("name", "")).strip()
        min_inclusive = _to_float(bucket.get("min"))
        max_inclusive = _to_float(bucket.get("max"))
        if not name or min_inclusive is None or max_inclusive is None:
            continue
        buckets.append(
            RiskBucket(
                name=name,
                min_inclusive=min_inclusive,
                max_inclusive=max_inclusive,
            )
        )
    if not buckets:
        buckets = [
            RiskBucket("low", 0.0, 0.33),
            RiskBucket("medium", 0.33, 0.66),
            RiskBucket("high", 0.66, 1.0),
        ]

    return SimulationConfig(
        input_path=input_path,
        metrics_output_path=metrics_output_path,
        report_path=report_path,
        customer_id_col=customer_id_col,
        segment_col=segment_col,
        churn_probability_col=churn_probability_col,
        acceptance_probability_col=acceptance_probability_col,
        margin_col=margin_col,
        discount_level_col=discount_level_col,
        default_acceptance_probability=default_acceptance_probability,
        retention_given_acceptance=retention_given_acceptance,
        discount_is_fraction=discount_is_fraction,
        risk_buckets=tuple(buckets),
    )


def run_offer_simulation(
    config: SimulationConfig,
    s3_client: S3ClientProtocol | None = None,
) -> SimulationSummary:
    """Run offer simulation and write metrics/report artifacts."""

    resolved_s3_client = _resolve_s3_client(
        paths=[config.input_path, config.metrics_output_path, config.report_path],
        s3_client=s3_client,
    )
    records = _read_records(config.input_path, resolved_s3_client)

    by_bucket: dict[tuple[str, str], dict[str, float]] = defaultdict(
        lambda: {
            "row_count": 0.0,
            "sum_churn_probability": 0.0,
            "sum_acceptance_probability": 0.0,
            "sum_margin": 0.0,
            "sum_discount_level": 0.0,
            "sum_expected_retained_margin": 0.0,
            "sum_expected_offer_cost": 0.0,
            "sum_incremental_margin": 0.0,
        }
    )

    skipped_rows = 0
    simulated_rows = 0

    for row in records:
        customer_id = str(row.get(config.customer_id_col) or "").strip()
        segment = str(row.get(config.segment_col) or "unknown").strip() or "unknown"
        churn_probability_raw = _to_float(row.get(config.churn_probability_col))
        margin_raw = _to_float(row.get(config.margin_col))
        discount_level_raw = _to_float(row.get(config.discount_level_col))

        if not customer_id or churn_probability_raw is None or margin_raw is None or discount_level_raw is None:
            skipped_rows += 1
            continue

        acceptance_probability_raw = _to_float(row.get(config.acceptance_probability_col))
        if acceptance_probability_raw is None:
            acceptance_probability = config.default_acceptance_probability
        else:
            acceptance_probability = _clip_probability(acceptance_probability_raw)

        churn_probability = _clip_probability(churn_probability_raw)
        margin = max(0.0, margin_raw)
        discount_level = max(0.0, discount_level_raw)

        if config.discount_is_fraction:
            discount_amount = min(margin, margin * discount_level)
        else:
            discount_amount = min(margin, discount_level)
        margin_after_discount = max(0.0, margin - discount_amount)

        # Expected retained margin from would-be churners retained due to accepted offer.
        expected_retained_margin = (
            churn_probability
            * acceptance_probability
            * config.retention_given_acceptance
            * margin_after_discount
        )

        # Expected offer discount cost for any accepted offer.
        expected_offer_cost = acceptance_probability * discount_amount

        # Incremental margin against no-offer baseline.
        expected_non_churn_discount_cost = (
            (1 - churn_probability) * acceptance_probability * discount_amount
        )
        incremental_margin = expected_retained_margin - expected_non_churn_discount_cost

        risk_bucket = _risk_bucket_name(churn_probability, config.risk_buckets)
        key = (segment, risk_bucket)
        bucket = by_bucket[key]
        bucket["row_count"] += 1
        bucket["sum_churn_probability"] += churn_probability
        bucket["sum_acceptance_probability"] += acceptance_probability
        bucket["sum_margin"] += margin
        bucket["sum_discount_level"] += discount_level
        bucket["sum_expected_retained_margin"] += expected_retained_margin
        bucket["sum_expected_offer_cost"] += expected_offer_cost
        bucket["sum_incremental_margin"] += incremental_margin
        simulated_rows += 1

    overall = {
        "row_count": 0.0,
        "sum_expected_retained_margin": 0.0,
        "sum_expected_offer_cost": 0.0,
        "sum_incremental_margin": 0.0,
    }
    bucket_metrics: list[dict[str, Any]] = []
    for (segment, risk_bucket), stats in sorted(by_bucket.items()):
        row_count = int(stats["row_count"])
        avg_churn = _safe_div(stats["sum_churn_probability"], stats["row_count"]) or 0.0
        avg_accept = _safe_div(stats["sum_acceptance_probability"], stats["row_count"]) or 0.0
        avg_margin = _safe_div(stats["sum_margin"], stats["row_count"]) or 0.0
        avg_discount = _safe_div(stats["sum_discount_level"], stats["row_count"]) or 0.0
        bucket_roi = _safe_div(stats["sum_incremental_margin"], stats["sum_expected_offer_cost"])

        bucket_metrics.append(
            {
                "segment": segment,
                "risk_bucket": risk_bucket,
                "row_count": row_count,
                "avg_churn_probability": avg_churn,
                "avg_acceptance_probability": avg_accept,
                "avg_margin": avg_margin,
                "avg_discount_level": avg_discount,
                "expected_retained_margin": stats["sum_expected_retained_margin"],
                "expected_offer_cost": stats["sum_expected_offer_cost"],
                "incremental_margin": stats["sum_incremental_margin"],
                "roi": bucket_roi,
            }
        )

        overall["row_count"] += stats["row_count"]
        overall["sum_expected_retained_margin"] += stats["sum_expected_retained_margin"]
        overall["sum_expected_offer_cost"] += stats["sum_expected_offer_cost"]
        overall["sum_incremental_margin"] += stats["sum_incremental_margin"]

    overall_roi = _safe_div(overall["sum_incremental_margin"], overall["sum_expected_offer_cost"])
    metrics_payload = {
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "input_path": config.input_path,
        "simulated_rows": simulated_rows,
        "skipped_rows": skipped_rows,
        "overall": {
            "row_count": int(overall["row_count"]),
            "expected_retained_margin": overall["sum_expected_retained_margin"],
            "expected_offer_cost": overall["sum_expected_offer_cost"],
            "incremental_margin": overall["sum_incremental_margin"],
            "roi": overall_roi,
        },
        "by_segment_risk_bucket": bucket_metrics,
    }
    _write_bytes(
        config.metrics_output_path,
        json.dumps(metrics_payload, indent=2, sort_keys=True).encode("utf-8"),
        resolved_s3_client,
    )

    report_lines = [
        "# Offer Simulation Report",
        "",
        f"Input path: `{config.input_path}`",
        f"Simulated rows: `{simulated_rows}`",
        f"Skipped rows: `{skipped_rows}`",
        "",
        "## Assumptions",
        "",
        f"- Default acceptance probability: `{config.default_acceptance_probability:.4f}`",
        f"- Retention given acceptance: `{config.retention_given_acceptance:.4f}`",
        f"- Discount interpreted as fraction of margin: `{config.discount_is_fraction}`",
        "",
        "## Overall Metrics",
        "",
        f"- Expected retained margin: `{overall['sum_expected_retained_margin']:.6f}`",
        f"- Expected offer cost: `{overall['sum_expected_offer_cost']:.6f}`",
        f"- Incremental margin: `{overall['sum_incremental_margin']:.6f}`",
        f"- ROI: `{overall_roi if overall_roi is not None else 'n/a'}`",
        "",
        "## Segment and Risk Bucket Metrics",
        "",
        "| Segment | Risk Bucket | Rows | Expected Retained Margin | Incremental Margin | ROI |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for metric in bucket_metrics:
        roi_text = "n/a" if metric["roi"] is None else f"{metric['roi']:.6f}"
        report_lines.append(
            f"| {metric['segment']} | {metric['risk_bucket']} | {metric['row_count']} | "
            f"{metric['expected_retained_margin']:.6f} | {metric['incremental_margin']:.6f} | {roi_text} |"
        )
    _write_bytes(
        config.report_path,
        "\n".join(report_lines).encode("utf-8"),
        resolved_s3_client,
    )

    return SimulationSummary(
        input_path=config.input_path,
        report_path=config.report_path,
        metrics_output_path=config.metrics_output_path,
        total_rows=len(records),
        simulated_rows=simulated_rows,
        skipped_rows=skipped_rows,
        overall_roi=overall_roi,
        overall_incremental_margin=overall["sum_incremental_margin"],
    )


def _render_summary(summary: SimulationSummary) -> str:
    return "\n".join(
        [
            "Offer simulation complete:",
            f"- input={summary.input_path}",
            f"- total_rows={summary.total_rows}",
            f"- simulated_rows={summary.simulated_rows}",
            f"- skipped_rows={summary.skipped_rows}",
            f"- overall_incremental_margin={summary.overall_incremental_margin:.6f}",
            f"- overall_roi={summary.overall_roi if summary.overall_roi is not None else 'n/a'}",
            f"- metrics={summary.metrics_output_path}",
            f"- report={summary.report_path}",
        ]
    )


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    config = _load_config(args.config)
    summary = run_offer_simulation(config=config)
    print(_render_summary(summary))


if __name__ == "__main__":
    main()

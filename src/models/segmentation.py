"""Build customer segments and profiling artifacts."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from datetime import timezone
from pathlib import Path
from statistics import mean
from statistics import pstdev
from typing import Any
from typing import Iterable
from typing import Protocol

from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction import DictVectorizer


class S3ClientProtocol(Protocol):
    """Minimal S3 client protocol used by this module."""

    def get_object(self, **kwargs: Any) -> dict[str, Any]: ...

    def put_object(self, **kwargs: Any) -> dict[str, Any]: ...


@dataclass(frozen=True)
class SegmentationSummary:
    asof_date: str
    requested_segment_count: int
    effective_segment_count: int
    output_path: str
    profile_path: str
    report_path: str
    row_count: int
    churn_enriched: bool
    margin_proxy_feature: str | None


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build customer segmentation and profiling.")
    parser.add_argument(
        "--asof-date",
        required=True,
        help="As-of date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--segment-count",
        type=int,
        default=4,
        help="Requested number of segments.",
    )
    parser.add_argument(
        "--gold-root",
        default=os.environ.get("SPANISHGAS_GOLD_ROOT", "s3://spanishgas-data-g1/gold/"),
        help="Gold root location (local path or s3:// URI).",
    )
    parser.add_argument(
        "--report-path",
        default="artifacts/reports/segmentation_profile.md",
        help="Path for markdown segmentation report (local path or s3:// URI).",
    )
    parser.add_argument(
        "--top-driver-count",
        type=int,
        default=5,
        help="Number of top drivers to display per segment.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed used by clustering.",
    )
    return parser.parse_args(argv)


def _validate_iso_date(value: str) -> str:
    datetime.strptime(value, "%Y-%m-%d")
    return value


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
            "S3 segmentation requested but no S3 SDK is available. "
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


def _to_jsonl(rows: Iterable[dict[str, Any]]) -> bytes:
    lines = [json.dumps(row, sort_keys=True) for row in rows]
    content = "\n".join(lines)
    if content:
        content += "\n"
    return content.encode("utf-8")


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


def _numeric_feature_columns(rows: list[dict[str, Any]], candidate_cols: list[str]) -> list[str]:
    numeric_cols: list[str] = []
    for col in candidate_cols:
        values = [_to_float(row.get(col)) for row in rows]
        non_null = [value for value in values if value is not None]
        if non_null:
            numeric_cols.append(col)
    return numeric_cols


def _normalize_feature_row(
    row: dict[str, Any], feature_cols: list[str], numeric_cols: set[str]
) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for col in feature_cols:
        value = row.get(col)
        if col in numeric_cols:
            numeric = _to_float(value)
            normalized[col] = 0.0 if numeric is None else numeric
        else:
            normalized[col] = "__missing__" if value is None else str(value)
    return normalized


def _segment_id(raw_label: int) -> str:
    return f"S{raw_label:02d}"


def _find_margin_proxy_feature(rows: list[dict[str, Any]]) -> str | None:
    candidates = [
        "margin_proxy_eur",
        "expected_margin_impact_eur",
        "price_vs_benchmark_delta",
    ]
    for candidate in candidates:
        if any(_to_float(row.get(candidate)) is not None for row in rows):
            return candidate
    return None


def _safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(mean(values))


def _safe_zscore(segment_mean: float | None, global_mean: float | None, std: float | None) -> float:
    if segment_mean is None or global_mean is None or std is None or std == 0:
        return 0.0
    return float((segment_mean - global_mean) / std)


def _load_optional_churn_labels(
    gold_root: str,
    asof_date: str,
    s3_client: S3ClientProtocol | None,
) -> dict[tuple[str, str], int]:
    path = _join_location(
        gold_root,
        f"churn_training_dataset/cutoff_date={asof_date}/churn_training_dataset.jsonl",
    )
    try:
        rows = _read_jsonl(path, s3_client)
    except FileNotFoundError:
        return {}

    labels: dict[tuple[str, str], int] = {}
    for row in rows:
        customer_id = str(row.get("customer_id") or "").strip()
        row_asof = str(row.get("asof_date") or "").strip()
        label = row.get("churn_label")
        numeric = _to_float(label)
        if customer_id and row_asof and numeric is not None:
            labels[(customer_id, row_asof)] = 1 if numeric >= 0.5 else 0
    return labels


def build_segmentation(
    asof_date: str,
    segment_count: int,
    gold_root: str | Path,
    report_path: str | Path,
    top_driver_count: int = 5,
    random_seed: int = 42,
    s3_client: S3ClientProtocol | None = None,
) -> SegmentationSummary:
    """Build segments and profiling outputs for an as-of date."""

    asof = _validate_iso_date(asof_date)
    if segment_count <= 0:
        raise ValueError("segment_count must be > 0.")
    if top_driver_count <= 0:
        raise ValueError("top_driver_count must be > 0.")

    gold_root_str = str(gold_root)
    report_path_str = str(report_path)
    resolved_s3_client = _resolve_s3_client(
        paths=[gold_root_str, report_path_str],
        s3_client=s3_client,
    )

    features_path = _join_location(
        gold_root_str,
        f"customer_features_asof_date/asof_date={asof}/customer_features_asof_date.jsonl",
    )
    feature_rows = _read_jsonl(features_path, resolved_s3_client)
    if not feature_rows:
        raise ValueError("No feature rows available for segmentation.")

    candidate_cols = sorted(
        {
            key
            for row in feature_rows
            for key in row
            if key not in {"customer_id", "asof_date", "split", "churn_label"}
        }
    )
    if not candidate_cols:
        raise ValueError("No feature columns found for segmentation.")
    numeric_cols = set(_numeric_feature_columns(feature_rows, candidate_cols))

    vectorizer = DictVectorizer(sparse=False)
    x = vectorizer.fit_transform(
        [_normalize_feature_row(row, candidate_cols, numeric_cols) for row in feature_rows]
    )

    effective_segment_count = min(segment_count, len(feature_rows))
    if effective_segment_count == 1:
        raw_labels = [0] * len(feature_rows)
    else:
        clusterer = AgglomerativeClustering(
            n_clusters=effective_segment_count,
            linkage="ward",
        )
        raw_labels = clusterer.fit_predict(x).tolist()

    churn_map = _load_optional_churn_labels(
        gold_root=gold_root_str,
        asof_date=asof,
        s3_client=resolved_s3_client,
    )
    margin_proxy_feature = _find_margin_proxy_feature(feature_rows)

    assignments: list[dict[str, Any]] = []
    for row, raw_label in zip(feature_rows, raw_labels):
        customer_id = str(row.get("customer_id") or "").strip()
        row_asof = str(row.get("asof_date") or asof).strip() or asof
        churn_label = churn_map.get((customer_id, row_asof))
        margin_proxy = (
            _to_float(row.get(margin_proxy_feature)) if margin_proxy_feature is not None else None
        )

        assignment = {
            "customer_id": customer_id,
            "asof_date": row_asof,
            "segment_id": _segment_id(int(raw_label)),
            "churn_label": churn_label,
        }
        if margin_proxy_feature is not None:
            assignment["margin_proxy_feature"] = margin_proxy_feature
            assignment["margin_proxy_value"] = margin_proxy
        assignments.append(assignment)

    numeric_cols_for_drivers = sorted(numeric_cols)
    global_stats: dict[str, tuple[float | None, float | None]] = {}
    for col in numeric_cols_for_drivers:
        values = [_to_float(row.get(col)) for row in feature_rows]
        non_null = [value for value in values if value is not None]
        if non_null:
            global_stats[col] = (_safe_mean(non_null), float(pstdev(non_null)))
        else:
            global_stats[col] = (None, None)

    rows_by_segment: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row, assignment in zip(feature_rows, assignments):
        rows_by_segment[str(assignment["segment_id"])].append(row)

    assignment_by_segment: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for assignment in assignments:
        assignment_by_segment[str(assignment["segment_id"])].append(assignment)

    total_rows = len(feature_rows)
    segment_profiles: list[dict[str, Any]] = []
    for segment_id in sorted(rows_by_segment):
        segment_rows = rows_by_segment[segment_id]
        segment_assignments = assignment_by_segment[segment_id]
        size = len(segment_rows)
        share = size / total_rows if total_rows else 0.0

        churn_values = [
            int(item["churn_label"])
            for item in segment_assignments
            if item.get("churn_label") is not None
        ]
        churn_rate = (sum(churn_values) / len(churn_values)) if churn_values else None

        margin_values = []
        if margin_proxy_feature is not None:
            margin_values = [
                _to_float(row.get(margin_proxy_feature))
                for row in segment_rows
                if _to_float(row.get(margin_proxy_feature)) is not None
            ]
        margin_proxy_mean = _safe_mean([value for value in margin_values if value is not None])

        drivers: list[dict[str, Any]] = []
        for col in numeric_cols_for_drivers:
            segment_values = [
                _to_float(row.get(col))
                for row in segment_rows
                if _to_float(row.get(col)) is not None
            ]
            seg_mean = _safe_mean([v for v in segment_values if v is not None])
            global_mean, global_std = global_stats[col]
            z_score = _safe_zscore(seg_mean, global_mean, global_std)
            drivers.append(
                {
                    "feature": col,
                    "segment_mean": seg_mean,
                    "global_mean": global_mean,
                    "z_score": z_score,
                }
            )
        top_drivers = sorted(drivers, key=lambda item: abs(item["z_score"]), reverse=True)[
            :top_driver_count
        ]

        segment_profiles.append(
            {
                "segment_id": segment_id,
                "size": size,
                "population_share": share,
                "churn_rate": churn_rate,
                "margin_proxy_feature": margin_proxy_feature,
                "margin_proxy_mean": margin_proxy_mean,
                "top_drivers": top_drivers,
            }
        )

    output_dir = _join_location(gold_root_str, f"segments/asof_date={asof}")
    output_path = _join_location(output_dir, "segments.jsonl")
    profile_path = _join_location(output_dir, "segment_profiles.json")

    _write_bytes(output_path, _to_jsonl(assignments), resolved_s3_client)
    _write_bytes(
        profile_path,
        json.dumps(
            {
                "asof_date": asof,
                "requested_segment_count": segment_count,
                "effective_segment_count": len(segment_profiles),
                "segment_profiles": segment_profiles,
                "generated_at_utc": datetime.now(timezone.utc)
                .replace(microsecond=0)
                .isoformat(),
            },
            indent=2,
            sort_keys=True,
        ).encode("utf-8"),
        resolved_s3_client,
    )

    report_lines = [
        "# Segmentation Profile",
        "",
        f"As-of date: `{asof}`",
        f"Requested segments: `{segment_count}`",
        f"Effective segments: `{len(segment_profiles)}`",
        "",
        "## Segment Summary",
        "",
        "| Segment | Size | Share | Churn Rate | Margin Proxy |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for profile in segment_profiles:
        churn_text = (
            f"{profile['churn_rate']:.4f}" if profile["churn_rate"] is not None else "n/a"
        )
        if profile["margin_proxy_mean"] is None:
            margin_text = "n/a"
        else:
            margin_text = (
                f"{profile['margin_proxy_feature']}={profile['margin_proxy_mean']:.6f}"
            )
        report_lines.append(
            f"| {profile['segment_id']} | {profile['size']} | "
            f"{profile['population_share']:.4f} | {churn_text} | {margin_text} |"
        )

    report_lines.extend(["", "## Top Drivers by Segment", ""])
    for profile in segment_profiles:
        report_lines.append(f"### {profile['segment_id']}")
        if not profile["top_drivers"]:
            report_lines.append("- No numeric drivers available.")
            report_lines.append("")
            continue
        for driver in profile["top_drivers"]:
            seg_mean = driver["segment_mean"]
            global_mean = driver["global_mean"]
            report_lines.append(
                "- "
                f"`{driver['feature']}`: "
                f"segment_mean={seg_mean if seg_mean is not None else 'n/a'}, "
                f"global_mean={global_mean if global_mean is not None else 'n/a'}, "
                f"z_score={driver['z_score']:.4f}"
            )
        report_lines.append("")

    _write_bytes(
        report_path_str,
        "\n".join(report_lines).encode("utf-8"),
        resolved_s3_client,
    )

    return SegmentationSummary(
        asof_date=asof,
        requested_segment_count=segment_count,
        effective_segment_count=len(segment_profiles),
        output_path=output_path,
        profile_path=profile_path,
        report_path=report_path_str,
        row_count=len(assignments),
        churn_enriched=bool(churn_map),
        margin_proxy_feature=margin_proxy_feature,
    )


def _render_summary(summary: SegmentationSummary) -> str:
    return "\n".join(
        [
            "Segmentation complete:",
            f"- asof_date={summary.asof_date}",
            f"- requested_segments={summary.requested_segment_count}",
            f"- effective_segments={summary.effective_segment_count}",
            f"- rows={summary.row_count}",
            f"- churn_enriched={summary.churn_enriched}",
            f"- margin_proxy_feature={summary.margin_proxy_feature}",
            f"- output={summary.output_path}",
            f"- profile={summary.profile_path}",
            f"- report={summary.report_path}",
        ]
    )


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    summary = build_segmentation(
        asof_date=args.asof_date,
        segment_count=args.segment_count,
        gold_root=args.gold_root,
        report_path=args.report_path,
        top_driver_count=args.top_driver_count,
        random_seed=args.random_seed,
    )
    print(_render_summary(summary))


if __name__ == "__main__":
    main()

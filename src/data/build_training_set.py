"""Build churn training dataset from features and labels."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import Iterable
from typing import Protocol

import yaml


class S3ClientProtocol(Protocol):
    """Minimal S3 client protocol used by this module."""

    def get_object(self, **kwargs: Any) -> dict[str, Any]: ...

    def put_object(self, **kwargs: Any) -> dict[str, Any]: ...


@dataclass(frozen=True)
class SplitRules:
    train_end_date: date
    valid_end_date: date
    test_end_date: date


@dataclass(frozen=True)
class TrainingSetConfig:
    silver_root: str
    gold_root: str
    horizon_days: int
    cutoff_date: date
    label_run_date: str
    asof_dates: tuple[date, ...]
    split_rules: SplitRules
    feature_version: str


@dataclass(frozen=True)
class BuildSummary:
    output_path: str
    row_count: int
    split_counts: dict[str, int]
    skipped_no_label: int
    skipped_after_cutoff: int
    asof_dates_processed: int


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build churn_training_dataset from features and churn labels."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML config file.",
    )
    return parser.parse_args(argv)


def _parse_iso_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


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
    silver_root: str,
    gold_root: str,
    s3_client: S3ClientProtocol | None = None,
) -> S3ClientProtocol | None:
    if s3_client is not None:
        return s3_client

    if not (_is_s3_uri(silver_root) or _is_s3_uri(gold_root)):
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
            "S3 training-set build requested but no S3 SDK is available. "
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


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
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


def _to_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value

    text = str(value).strip()
    if not text:
        return None

    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            pass

    normalized = text.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized).date()
    except ValueError:
        return None


def _load_config(config_path: str) -> TrainingSetConfig:
    raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Config must be a YAML object.")

    paths = raw.get("paths") or {}
    labeling = raw.get("labeling") or {}
    features = raw.get("features") or {}
    splits = raw.get("splits") or {}

    silver_root = str(paths.get("silver_root", "")).strip()
    gold_root = str(paths.get("gold_root", "")).strip()
    if not silver_root or not gold_root:
        raise ValueError("Config paths.silver_root and paths.gold_root are required.")

    horizon_days = _to_int(labeling.get("horizon_days"))
    if horizon_days is None or horizon_days <= 0:
        raise ValueError("Config labeling.horizon_days must be a positive integer.")

    cutoff_raw = str(labeling.get("cutoff_date", "")).strip()
    if not cutoff_raw:
        raise ValueError("Config labeling.cutoff_date is required.")
    cutoff_date = _parse_iso_date(cutoff_raw)

    label_run_date = str(labeling.get("label_run_date", cutoff_raw)).strip()
    _parse_iso_date(label_run_date)

    asof_raw = features.get("asof_dates")
    if not isinstance(asof_raw, list) or not asof_raw:
        raise ValueError("Config features.asof_dates must be a non-empty list.")
    asof_dates = tuple(sorted({_parse_iso_date(str(item)) for item in asof_raw}))

    train_end = _parse_iso_date(str(splits.get("train_end_date", "")).strip())
    valid_end = _parse_iso_date(str(splits.get("valid_end_date", "")).strip())
    test_end = _parse_iso_date(str(splits.get("test_end_date", "")).strip())

    if not (train_end <= valid_end <= test_end):
        raise ValueError(
            "Temporal split rules must satisfy train_end_date <= "
            "valid_end_date <= test_end_date."
        )

    feature_version = str(features.get("feature_version", "v1")).strip() or "v1"

    return TrainingSetConfig(
        silver_root=silver_root,
        gold_root=gold_root,
        horizon_days=horizon_days,
        cutoff_date=cutoff_date,
        label_run_date=label_run_date,
        asof_dates=asof_dates,
        split_rules=SplitRules(
            train_end_date=train_end,
            valid_end_date=valid_end,
            test_end_date=test_end,
        ),
        feature_version=feature_version,
    )


def _assign_split(asof_date: date, split_rules: SplitRules) -> str | None:
    if asof_date <= split_rules.train_end_date:
        return "train"
    if asof_date <= split_rules.valid_end_date:
        return "valid"
    if asof_date <= split_rules.test_end_date:
        return "test"
    return None


def build_training_set(
    config: TrainingSetConfig,
    s3_client: S3ClientProtocol | None = None,
) -> BuildSummary:
    """Build churn_training_dataset from features and labels."""

    resolved_s3_client = _resolve_s3_client(
        silver_root=config.silver_root,
        gold_root=config.gold_root,
        s3_client=s3_client,
    )

    labels_path = _join_location(
        config.silver_root,
        f"churn_label/run_date={config.label_run_date}/churn_label.jsonl",
    )
    label_rows = _read_jsonl(labels_path, resolved_s3_client)
    labels_by_key: dict[tuple[str, str], int] = {}
    for row in label_rows:
        customer_id = str(row.get("customer_id") or "").strip()
        label_date = _to_date(row.get("label_date"))
        horizon_days = _to_int(row.get("horizon_days"))
        churned = _to_int(row.get("churned_within_horizon"))
        if not customer_id or label_date is None:
            continue
        if label_date > config.cutoff_date:
            continue
        if horizon_days != config.horizon_days:
            continue
        labels_by_key[(customer_id, label_date.isoformat())] = 1 if churned == 1 else 0

    training_rows: list[dict[str, Any]] = []
    skipped_no_label = 0
    skipped_after_cutoff = 0
    split_counts = {"train": 0, "valid": 0, "test": 0}
    asof_dates_processed = 0

    for asof_date in config.asof_dates:
        if asof_date > config.cutoff_date:
            skipped_after_cutoff += 1
            continue

        feature_path = _join_location(
            config.gold_root,
            "customer_features_asof_date/"
            f"asof_date={asof_date.isoformat()}/customer_features_asof_date.jsonl",
        )
        try:
            feature_rows = _read_jsonl(feature_path, resolved_s3_client)
        except FileNotFoundError:
            continue
        asof_dates_processed += 1

        for feature_row in feature_rows:
            customer_id = str(feature_row.get("customer_id") or "").strip()
            row_asof = _to_date(feature_row.get("asof_date")) or asof_date
            if not customer_id or row_asof > config.cutoff_date:
                skipped_after_cutoff += 1
                continue

            label_key = (customer_id, row_asof.isoformat())
            label_value = labels_by_key.get(label_key)
            if label_value is None:
                skipped_no_label += 1
                continue

            split = _assign_split(row_asof, config.split_rules)
            if split is None:
                continue

            output_row = dict(feature_row)
            output_row["asof_date"] = row_asof.isoformat()
            output_row["feature_version"] = (
                str(feature_row.get("feature_version") or config.feature_version).strip()
                or config.feature_version
            )
            output_row["label_horizon_days"] = config.horizon_days
            output_row["churn_label"] = label_value
            output_row["split"] = split
            training_rows.append(output_row)
            split_counts[split] += 1

    output_path = _join_location(
        config.gold_root,
        "churn_training_dataset/"
        f"cutoff_date={config.cutoff_date.isoformat()}/churn_training_dataset.jsonl",
    )
    _write_bytes(output_path, _to_jsonl(training_rows), resolved_s3_client)

    return BuildSummary(
        output_path=output_path,
        row_count=len(training_rows),
        split_counts=split_counts,
        skipped_no_label=skipped_no_label,
        skipped_after_cutoff=skipped_after_cutoff,
        asof_dates_processed=asof_dates_processed,
    )


def _render_summary(summary: BuildSummary, config: TrainingSetConfig) -> str:
    return "\n".join(
        [
            "Training dataset build complete:",
            f"- cutoff_date={config.cutoff_date.isoformat()}",
            f"- label_horizon_days={config.horizon_days}",
            f"- asof_dates_processed={summary.asof_dates_processed}",
            f"- total_rows={summary.row_count}",
            f"- train_rows={summary.split_counts['train']}",
            f"- valid_rows={summary.split_counts['valid']}",
            f"- test_rows={summary.split_counts['test']}",
            f"- skipped_no_label={summary.skipped_no_label}",
            f"- skipped_after_cutoff={summary.skipped_after_cutoff}",
            f"- output={summary.output_path}",
        ]
    )


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    config = _load_config(args.config)
    summary = build_training_set(config=config)
    print(_render_summary(summary, config))


if __name__ == "__main__":
    main()

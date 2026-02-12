"""Bronze-to-Silver transformations with quality reporting."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Iterable
from typing import List
from typing import Protocol

from src.data.contracts import get_source_contracts


ALLOWED_INTERACTION_CHANNELS = {
    "call",
    "email",
    "chat",
    "web",
    "app",
    "sms",
    "whatsapp",
    "store",
}


class S3ClientProtocol(Protocol):
    """Minimal S3 client protocol used by this module."""

    def get_object(self, **kwargs: Any) -> dict[str, Any]: ...

    def put_object(self, **kwargs: Any) -> dict[str, Any]: ...


@dataclass(frozen=True)
class QualityReportEntry:
    """Quality report entry for one transformed dataset."""

    run_date: str
    dataset_name: str
    table_name: str
    input_row_count: int
    output_row_count: int
    dropped_missing_customer_id: int
    negative_consumption_flagged: int
    unknown_channel_count: int
    output_path: str
    processed_at_utc: str


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transform Bronze datasets to Silver outputs."
    )
    parser.add_argument(
        "--run-date",
        required=True,
        help="Run date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--bronze-root",
        default=os.environ.get(
            "SPANISHGAS_BRONZE_ROOT", "s3://spanishgas-data-g1/bronze/"
        ),
        help="Bronze root location (local path or s3:// URI).",
    )
    parser.add_argument(
        "--silver-root",
        default=os.environ.get(
            "SPANISHGAS_SILVER_ROOT", "s3://spanishgas-data-g1/silver/"
        ),
        help="Silver root location (local path or s3:// URI).",
    )
    return parser.parse_args(argv)


def _validate_run_date(run_date: str) -> None:
    datetime.strptime(run_date, "%Y-%m-%d")


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
    bronze_root: str,
    silver_root: str,
    s3_client: S3ClientProtocol | None = None,
) -> S3ClientProtocol | None:
    if s3_client is not None:
        return s3_client

    if not (_is_s3_uri(bronze_root) or _is_s3_uri(silver_root)):
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
            "S3 Silver transformation requested but no S3 SDK is available. "
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


def _read_csv_rows(content: str) -> List[dict[str, Any]]:
    return list(csv.DictReader(io.StringIO(content)))


def _read_json_rows(content: str) -> List[dict[str, Any]]:
    content = content.strip()
    if not content:
        return []

    try:
        decoded = json.loads(content)
    except json.JSONDecodeError:
        rows: List[dict[str, Any]] = []
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
        return rows

    if isinstance(decoded, list):
        return [item for item in decoded if isinstance(item, dict)]

    if isinstance(decoded, dict):
        if isinstance(decoded.get("records"), list):
            return [item for item in decoded["records"] if isinstance(item, dict)]
        return [decoded]

    return []


def _read_rows(dataset_name: str, payload: bytes) -> List[dict[str, Any]]:
    suffix = Path(dataset_name).suffix.lower()
    content = payload.decode("utf-8")

    if suffix == ".csv":
        return _read_csv_rows(content)
    if suffix == ".json":
        return _read_json_rows(content)
    raise ValueError(f"Unsupported file type: {dataset_name}")


def _flatten_value(prefix: str, value: Any, output: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, nested_value in value.items():
            nested_prefix = f"{prefix}_{key}" if prefix else str(key)
            _flatten_value(nested_prefix, nested_value, output)
        return

    if isinstance(value, list):
        output[prefix] = json.dumps(value, sort_keys=True)
        return

    output[prefix] = value


def _flatten_record(row: dict[str, Any]) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in row.items():
        _flatten_value(str(key), value, flattened)
    return flattened


def _parse_date(value: str) -> str | None:
    value = value.strip()
    if not value:
        return None

    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(value, fmt).date().isoformat()
        except ValueError:
            pass

    normalized = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized).date().isoformat()
    except ValueError:
        return value


def _parse_timestamp(value: str) -> str | None:
    value = value.strip()
    if not value:
        return None

    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(value, fmt).isoformat()
        except ValueError:
            pass

    normalized = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized).isoformat()
    except ValueError:
        return value


def _coerce_value(value: Any, dtype: str) -> Any:
    if value is None:
        return None

    if isinstance(value, str):
        text = value.strip()
        if text == "":
            return None
    else:
        text = str(value)

    if dtype == "int":
        try:
            return int(float(text))
        except ValueError:
            return None

    if dtype == "float":
        try:
            return float(text)
        except ValueError:
            return None

    if dtype == "date":
        return _parse_date(text)

    if dtype == "timestamp":
        return _parse_timestamp(text)

    return text


def _cast_row(dataset_name: str, row: dict[str, Any]) -> dict[str, Any]:
    expected_types = {
        column.name: column.dtype
        for column in get_source_contracts()[dataset_name].columns
    }
    casted: dict[str, Any] = {
        column_name: _coerce_value(row.get(column_name), dtype)
        for column_name, dtype in expected_types.items()
    }
    for key, value in row.items():
        if key not in casted:
            casted[key] = value
    return casted


def _to_jsonl(rows: Iterable[dict[str, Any]]) -> bytes:
    lines = [json.dumps(row, sort_keys=True) for row in rows]
    content = "\n".join(lines)
    if content:
        content += "\n"
    return content.encode("utf-8")


def _dataset_id(dataset_name: str) -> str:
    return Path(dataset_name).stem


def _transform_dataset(
    dataset_name: str, input_rows: List[dict[str, Any]]
) -> tuple[List[dict[str, Any]], int, int, int]:
    output_rows: List[dict[str, Any]] = []
    dropped_missing_customer_id = 0
    negative_consumption_flagged = 0
    unknown_channel_count = 0

    for raw_row in input_rows:
        working_row = (
            _flatten_record(raw_row)
            if dataset_name == "customer_interactions.json"
            else dict(raw_row)
        )
        row = _cast_row(dataset_name, working_row)

        if "customer_id" in row:
            customer_id = row.get("customer_id")
            if customer_id is None or str(customer_id).strip() == "":
                dropped_missing_customer_id += 1
                continue
            row["customer_id"] = str(customer_id).strip()

        if dataset_name == "consumption_hourly_2024.csv":
            consumption = row.get("consumption_kwh")
            is_negative = isinstance(consumption, (int, float)) and consumption < 0
            row["negative_consumption_flag"] = 1 if is_negative else 0
            if is_negative:
                negative_consumption_flagged += 1

        if dataset_name == "customer_interactions.json":
            channel = str(row.get("channel") or "").strip().lower()
            is_unknown = bool(channel) and channel not in ALLOWED_INTERACTION_CHANNELS
            row["unknown_channel_flag"] = 1 if is_unknown else 0
            if is_unknown:
                unknown_channel_count += 1

        output_rows.append(row)

    return (
        output_rows,
        dropped_missing_customer_id,
        negative_consumption_flagged,
        unknown_channel_count,
    )


def run_silver_transforms(
    run_date: str,
    bronze_root: str | Path,
    silver_root: str | Path,
    s3_client: S3ClientProtocol | None = None,
) -> list[QualityReportEntry]:
    """Execute Silver transforms for all source datasets."""

    _validate_run_date(run_date)
    bronze_root_str = str(bronze_root)
    silver_root_str = str(silver_root)
    resolved_s3_client = _resolve_s3_client(
        bronze_root=bronze_root_str,
        silver_root=silver_root_str,
        s3_client=s3_client,
    )

    processed_at_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    reports: list[QualityReportEntry] = []

    for dataset_name in sorted(get_source_contracts()):
        table_name = _dataset_id(dataset_name)
        bronze_path = _join_location(
            bronze_root_str,
            f"{table_name}/run_date={run_date}/{dataset_name}",
        )
        payload = _read_bytes(bronze_path, resolved_s3_client)
        raw_rows = _read_rows(dataset_name, payload)

        (
            silver_rows,
            dropped_missing_customer_id,
            negative_consumption_flagged,
            unknown_channel_count,
        ) = _transform_dataset(dataset_name, raw_rows)

        silver_output_path = _join_location(
            silver_root_str,
            f"{table_name}/run_date={run_date}/{table_name}.jsonl",
        )
        _write_bytes(
            silver_output_path,
            _to_jsonl(silver_rows),
            resolved_s3_client,
        )

        reports.append(
            QualityReportEntry(
                run_date=run_date,
                dataset_name=dataset_name,
                table_name=table_name,
                input_row_count=len(raw_rows),
                output_row_count=len(silver_rows),
                dropped_missing_customer_id=dropped_missing_customer_id,
                negative_consumption_flagged=negative_consumption_flagged,
                unknown_channel_count=unknown_channel_count,
                output_path=silver_output_path,
                processed_at_utc=processed_at_utc,
            )
        )

    quality_report = {
        "run_date": run_date,
        "generated_at_utc": processed_at_utc,
        "datasets": [asdict(entry) for entry in reports],
    }
    quality_report_path = _join_location(
        silver_root_str,
        f"_meta/quality_report_run_date={run_date}.json",
    )
    _write_bytes(
        quality_report_path,
        json.dumps(quality_report, indent=2, sort_keys=True).encode("utf-8"),
        resolved_s3_client,
    )

    return reports


def _render_summary(entries: list[QualityReportEntry]) -> str:
    lines = ["Silver transformation complete:"]
    for entry in entries:
        lines.append(
            "- "
            f"{entry.dataset_name}: "
            f"in={entry.input_row_count}, "
            f"out={entry.output_row_count}, "
            f"dropped_missing_customer_id={entry.dropped_missing_customer_id}, "
            f"negative_consumption_flagged={entry.negative_consumption_flagged}, "
            f"unknown_channel_count={entry.unknown_channel_count}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    reports = run_silver_transforms(
        run_date=args.run_date,
        bronze_root=args.bronze_root,
        silver_root=args.silver_root,
    )
    print(_render_summary(reports))


if __name__ == "__main__":
    main()

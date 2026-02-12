"""Raw-to-Bronze ingestion for source datasets (local filesystem and S3)."""

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


class S3ClientProtocol(Protocol):
    """Minimal S3 client protocol used by this module."""

    def get_object(self, **kwargs: Any) -> dict[str, Any]: ...

    def put_object(self, **kwargs: Any) -> dict[str, Any]: ...

    def head_object(self, **kwargs: Any) -> dict[str, Any]: ...


@dataclass(frozen=True)
class IngestionRecord:
    """Metadata captured for one ingested dataset."""

    run_date: str
    dataset_name: str
    source_path: str
    output_path: str
    row_count: int
    schema_inferred: Dict[str, str]
    ingested_at_utc: str


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest raw files into Bronze layer.")
    parser.add_argument(
        "--run-date",
        required=True,
        help="Run date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--raw-root",
        default=os.environ.get("SPANISHGAS_RAW_ROOT", "s3://spanishgas-data-g1/raw/"),
        help="Root location containing source files (local path or s3:// URI).",
    )
    parser.add_argument(
        "--bronze-root",
        default=os.environ.get(
            "SPANISHGAS_BRONZE_ROOT", "s3://spanishgas-data-g1/bronze/"
        ),
        help="Root location where Bronze outputs are written (local path or s3:// URI).",
    )
    return parser.parse_args(argv)


def _validate_run_date(run_date: str) -> None:
    datetime.strptime(run_date, "%Y-%m-%d")


def _infer_scalar_type(value: Any) -> str:
    if value is None:
        return "null"

    if isinstance(value, bool):
        return "bool"

    if isinstance(value, int):
        return "int"

    if isinstance(value, float):
        return "float"

    if isinstance(value, (dict, list)):
        return "json"

    text = str(value).strip()
    if text == "":
        return "null"

    lowered = text.lower()
    if lowered in {"true", "false"}:
        return "bool"

    try:
        int(text)
        return "int"
    except ValueError:
        pass

    try:
        float(text)
        return "float"
    except ValueError:
        pass

    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            datetime.strptime(text, fmt)
            return "timestamp" if "H" in fmt else "date"
        except ValueError:
            pass

    if text.endswith("Z"):
        try:
            datetime.fromisoformat(text.replace("Z", "+00:00"))
            return "timestamp"
        except ValueError:
            pass

    return "string"


def _merge_inferred_types(existing: str | None, new_type: str) -> str:
    if existing is None or existing == "null":
        return new_type

    if new_type == "null":
        return existing

    if existing == new_type:
        return existing

    numeric_types = {existing, new_type}
    if numeric_types == {"int", "float"}:
        return "float"

    temporal_types = {existing, new_type}
    if temporal_types == {"date", "timestamp"}:
        return "timestamp"

    return "string"


def _infer_schema(rows: Iterable[dict[str, Any]]) -> Dict[str, str]:
    inferred: Dict[str, str] = {}
    for row in rows:
        for column, value in row.items():
            value_type = _infer_scalar_type(value)
            inferred[column] = _merge_inferred_types(inferred.get(column), value_type)
    return dict(sorted(inferred.items()))


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
    raise ValueError(f"Unsupported file type for ingestion: {dataset_name}")


def _dataset_id(dataset_name: str) -> str:
    return Path(dataset_name).stem


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
    raw_root: str,
    bronze_root: str,
    s3_client: S3ClientProtocol | None = None,
) -> S3ClientProtocol | None:
    if s3_client is not None:
        return s3_client

    if not (_is_s3_uri(raw_root) or _is_s3_uri(bronze_root)):
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
            "S3 ingestion requested but no S3 SDK is available. "
            "Install boto3 or use local paths."
        ) from exc


def _read_source_bytes(source_path: str, s3_client: S3ClientProtocol | None) -> bytes:
    if _is_s3_uri(source_path):
        if s3_client is None:
            raise RuntimeError("S3 path used without an S3 client.")
        bucket, key = _parse_s3_uri(source_path)
        try:
            response = s3_client.get_object(Bucket=bucket, Key=key)
        except Exception as exc:
            raise FileNotFoundError(f"Missing raw source file: {source_path}") from exc
        return response["Body"].read()

    path = Path(source_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing raw source file: {source_path}")
    return path.read_bytes()


def _write_output_bytes(
    output_path: str,
    payload: bytes,
    s3_client: S3ClientProtocol | None,
) -> None:
    if _is_s3_uri(output_path):
        if s3_client is None:
            raise RuntimeError("S3 path used without an S3 client.")
        bucket, key = _parse_s3_uri(output_path)
        s3_client.put_object(Bucket=bucket, Key=key, Body=payload)
        return

    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_bytes(payload)


def _s3_object_exists(
    s3_client: S3ClientProtocol,
    bucket: str,
    key: str,
) -> bool:
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False


def run_ingestion(
    run_date: str,
    raw_root: str | Path,
    bronze_root: str | Path,
    s3_client: S3ClientProtocol | None = None,
) -> list[IngestionRecord]:
    """Ingest all source datasets into Bronze partitioned paths.

    Supports local filesystem paths and S3 URIs for both input and output roots.
    """

    _validate_run_date(run_date)
    raw_root_str = str(raw_root)
    bronze_root_str = str(bronze_root)
    resolved_s3_client = _resolve_s3_client(
        raw_root=raw_root_str,
        bronze_root=bronze_root_str,
        s3_client=s3_client,
    )
    contracts = get_source_contracts()
    ingested_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    records: list[IngestionRecord] = []
    for dataset_name in sorted(contracts):
        source_path = _join_location(raw_root_str, dataset_name)
        output_relative_path = (
            f"{_dataset_id(dataset_name)}/run_date={run_date}/{dataset_name}"
        )
        output_path = _join_location(bronze_root_str, output_relative_path)

        payload = _read_source_bytes(source_path, resolved_s3_client)
        _write_output_bytes(output_path, payload, resolved_s3_client)

        rows = _read_rows(dataset_name, payload)
        schema_inferred = _infer_schema(rows)

        records.append(
            IngestionRecord(
                run_date=run_date,
                dataset_name=dataset_name,
                source_path=source_path,
                output_path=output_path,
                row_count=len(rows),
                schema_inferred=schema_inferred,
                ingested_at_utc=ingested_at,
            )
        )

    log_path = _join_location(bronze_root_str, "_meta/ingestion_log.jsonl")
    _append_ingestion_log(records, log_path, resolved_s3_client)
    return records


def _append_ingestion_log(
    records: list[IngestionRecord],
    log_path: str,
    s3_client: S3ClientProtocol | None,
) -> None:
    new_content = "".join(json.dumps(asdict(record), sort_keys=True) + "\n" for record in records)
    payload = new_content.encode("utf-8")

    if _is_s3_uri(log_path):
        if s3_client is None:
            raise RuntimeError("S3 path used without an S3 client.")
        bucket, key = _parse_s3_uri(log_path)
        existing = b""
        if _s3_object_exists(s3_client, bucket, key):
            existing = s3_client.get_object(Bucket=bucket, Key=key)["Body"].read()
            if existing and not existing.endswith(b"\n"):
                existing += b"\n"
        s3_client.put_object(Bucket=bucket, Key=key, Body=existing + payload)
        return

    target_log_path = Path(log_path)
    target_log_path.parent.mkdir(parents=True, exist_ok=True)
    with target_log_path.open("a", encoding="utf-8") as handle:
        handle.write(new_content)


def _render_summary(records: list[IngestionRecord]) -> str:
    lines = ["Ingestion complete:"]
    for record in records:
        lines.append(
            f"- {record.dataset_name}: rows={record.row_count}, output={record.output_path}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    records = run_ingestion(
        run_date=args.run_date,
        raw_root=args.raw_root,
        bronze_root=args.bronze_root,
    )
    print(_render_summary(records))


if __name__ == "__main__":
    main()

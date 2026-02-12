"""Build leakage-safe customer features for a given as-of date."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from datetime import date
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from pathlib import Path
from statistics import pstdev
from typing import Any
from typing import Dict
from typing import Iterable
from typing import Protocol


FEATURE_VERSION = "v1"


class S3ClientProtocol(Protocol):
    """Minimal S3 client protocol used by this module."""

    def get_object(self, **kwargs: Any) -> dict[str, Any]: ...

    def put_object(self, **kwargs: Any) -> dict[str, Any]: ...


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build customer_features_asof_date from Silver datasets."
    )
    parser.add_argument(
        "--asof-date",
        required=True,
        help="As-of date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--silver-root",
        default=os.environ.get(
            "SPANISHGAS_SILVER_ROOT", "s3://spanishgas-data-g1/silver/"
        ),
        help="Silver root location (local path or s3:// URI).",
    )
    parser.add_argument(
        "--gold-root",
        default=os.environ.get("SPANISHGAS_GOLD_ROOT", "s3://spanishgas-data-g1/gold/"),
        help="Gold root location (local path or s3:// URI).",
    )
    parser.add_argument(
        "--silver-run-date",
        default=os.environ.get("SPANISHGAS_SILVER_RUN_DATE"),
        help="Run date partition to read from Silver. Defaults to --asof-date.",
    )
    return parser.parse_args(argv)


def _validate_iso_date(value: str) -> date:
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
            "S3 feature building requested but no S3 SDK is available. "
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


def _to_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value

    text = str(value).strip()
    if not text:
        return None

    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass

    normalized = text.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def _round_or_none(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    if math.isnan(value):
        return None
    return round(value, digits)


def _safe_pstdev(values: list[float]) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    return float(pstdev(values))


def _load_silver_table(
    silver_root: str,
    silver_run_date: str,
    table_name: str,
    s3_client: S3ClientProtocol | None,
) -> list[dict[str, Any]]:
    path = _join_location(
        silver_root, f"{table_name}/run_date={silver_run_date}/{table_name}.jsonl"
    )
    return _read_jsonl(path, s3_client)


def build_customer_features(
    asof_date: str,
    silver_root: str | Path,
    gold_root: str | Path,
    silver_run_date: str | None = None,
    s3_client: S3ClientProtocol | None = None,
) -> list[dict[str, Any]]:
    """Build leakage-safe customer features for the provided as-of date."""

    asof = _validate_iso_date(asof_date)
    silver_run = silver_run_date or asof_date
    _validate_iso_date(silver_run)

    silver_root_str = str(silver_root)
    gold_root_str = str(gold_root)
    resolved_s3_client = _resolve_s3_client(
        silver_root=silver_root_str,
        gold_root=gold_root_str,
        s3_client=s3_client,
    )

    customer_rows = _load_silver_table(
        silver_root=silver_root_str,
        silver_run_date=silver_run,
        table_name="customer_attributes",
        s3_client=resolved_s3_client,
    )
    contracts_rows = _load_silver_table(
        silver_root=silver_root_str,
        silver_run_date=silver_run,
        table_name="customer_contracts",
        s3_client=resolved_s3_client,
    )
    price_rows = _load_silver_table(
        silver_root=silver_root_str,
        silver_run_date=silver_run,
        table_name="price_history",
        s3_client=resolved_s3_client,
    )
    consumption_rows = _load_silver_table(
        silver_root=silver_root_str,
        silver_run_date=silver_run,
        table_name="consumption_hourly_2024",
        s3_client=resolved_s3_client,
    )
    interaction_rows = _load_silver_table(
        silver_root=silver_root_str,
        silver_run_date=silver_run,
        table_name="customer_interactions",
        s3_client=resolved_s3_client,
    )

    eligible_customers: dict[str, dict[str, Any]] = {}
    for row in customer_rows:
        customer_id = str(row.get("customer_id") or "").strip()
        signup_date = _to_date(row.get("signup_date"))
        if not customer_id or signup_date is None:
            continue
        if signup_date > asof:
            continue
        eligible_customers[customer_id] = row

    contracts_by_customer: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in contracts_rows:
        customer_id = str(row.get("customer_id") or "").strip()
        if customer_id not in eligible_customers:
            continue
        contract_start = _to_date(row.get("contract_start_date"))
        if contract_start is None or contract_start > asof:
            continue
        contracts_by_customer[customer_id].append(row)

    price_by_tariff: dict[str, list[float]] = defaultdict(list)
    global_price_deltas: list[float] = []
    for row in price_rows:
        price_date = _to_date(row.get("price_date"))
        if price_date is None or price_date > asof:
            continue
        price = _to_float(row.get("price_eur_per_kwh"))
        benchmark = _to_float(row.get("market_benchmark_eur_per_kwh"))
        if price is None or benchmark is None:
            continue
        delta = price - benchmark
        global_price_deltas.append(delta)
        tariff_type = str(row.get("tariff_type") or "").strip()
        if tariff_type:
            price_by_tariff[tariff_type].append(delta)

    global_price_delta = (
        sum(global_price_deltas) / len(global_price_deltas) if global_price_deltas else None
    )

    ninety_day_start = asof - timedelta(days=89)
    consumption_window: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in consumption_rows:
        customer_id = str(row.get("customer_id") or "").strip()
        if customer_id not in eligible_customers:
            continue

        timestamp = _to_datetime(row.get("timestamp_utc"))
        if timestamp is None:
            continue
        reading_date = timestamp.date()
        if reading_date < ninety_day_start or reading_date > asof:
            continue
        consumption_window[customer_id].append(row)

    interactions_window: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in interaction_rows:
        customer_id = str(row.get("customer_id") or "").strip()
        if customer_id not in eligible_customers:
            continue

        interaction_ts = _to_datetime(row.get("interaction_ts"))
        if interaction_ts is None:
            continue
        event_date = interaction_ts.date()
        if event_date < ninety_day_start or event_date > asof:
            continue
        interactions_window[customer_id].append(row)

    feature_rows: list[dict[str, Any]] = []
    for customer_id in sorted(eligible_customers):
        profile = eligible_customers[customer_id]
        signup_date = _to_date(profile.get("signup_date"))
        tenure_days = (asof - signup_date).days if signup_date is not None else None

        days_to_contract_end: int | None = None
        for contract in contracts_by_customer.get(customer_id, []):
            contract_end = _to_date(contract.get("contract_end_date"))
            if contract_end is None:
                continue
            if contract_end < asof:
                continue
            delta_days = (contract_end - asof).days
            if days_to_contract_end is None or delta_days < days_to_contract_end:
                days_to_contract_end = delta_days

        tariff_type = str(profile.get("tariff_type") or "").strip()
        tariff_deltas = price_by_tariff.get(tariff_type, [])
        if tariff_deltas:
            price_vs_benchmark = sum(tariff_deltas) / len(tariff_deltas)
        else:
            price_vs_benchmark = global_price_delta

        customer_consumption = consumption_window.get(customer_id, [])
        consumption_values = [
            value
            for value in (_to_float(row.get("consumption_kwh")) for row in customer_consumption)
            if value is not None
        ]
        consumption_volatility_90d = _safe_pstdev(consumption_values)
        negative_consumption_flag = (
            1
            if any(int(row.get("negative_consumption_flag") or 0) == 1 for row in customer_consumption)
            else 0
        )

        interaction_count_90d = len(interactions_window.get(customer_id, []))

        feature_rows.append(
            {
                "customer_id": customer_id,
                "asof_date": asof.isoformat(),
                "feature_version": FEATURE_VERSION,
                "tenure_days": tenure_days,
                "days_to_contract_end": days_to_contract_end,
                "price_vs_benchmark_delta": _round_or_none(price_vs_benchmark),
                "consumption_volatility_90d": _round_or_none(consumption_volatility_90d),
                "interaction_count_90d": interaction_count_90d,
                "negative_consumption_flag": negative_consumption_flag,
            }
        )

    output_path = _join_location(
        gold_root_str,
        f"customer_features_asof_date/asof_date={asof_date}/customer_features_asof_date.jsonl",
    )
    _write_bytes(output_path, _to_jsonl(feature_rows), resolved_s3_client)

    manifest_path = _join_location(gold_root_str, "_meta/feature_manifest.json")
    manifest = {
        "feature_set": "customer_features_asof_date",
        "feature_version": FEATURE_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "asof_date": asof_date,
        "silver_run_date": silver_run,
        "output_path": output_path,
        "row_count": len(feature_rows),
        "features": [
            {
                "name": "tenure_days",
                "dtype": "int",
                "source": "customer_attributes.signup_date",
            },
            {
                "name": "days_to_contract_end",
                "dtype": "int",
                "source": "customer_contracts.contract_end_date",
            },
            {
                "name": "price_vs_benchmark_delta",
                "dtype": "float",
                "source": "price_history.price_eur_per_kwh - market_benchmark_eur_per_kwh",
            },
            {
                "name": "consumption_volatility_90d",
                "dtype": "float",
                "source": "consumption_hourly_2024.consumption_kwh",
            },
            {
                "name": "interaction_count_90d",
                "dtype": "int",
                "source": "customer_interactions.interaction_ts",
            },
            {
                "name": "negative_consumption_flag",
                "dtype": "int",
                "source": "consumption_hourly_2024.negative_consumption_flag",
            },
        ],
        "filters": {
            "rule": "All time-based inputs are filtered to <= asof_date",
            "window_days_for_recency_features": 90,
        },
    }
    _write_bytes(
        manifest_path,
        json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8"),
        resolved_s3_client,
    )

    return feature_rows


def _render_summary(feature_rows: list[dict[str, Any]], asof_date: str, gold_root: str) -> str:
    output_path = _join_location(
        gold_root,
        f"customer_features_asof_date/asof_date={asof_date}/customer_features_asof_date.jsonl",
    )
    return "\n".join(
        [
            "Feature build complete:",
            f"- asof_date={asof_date}",
            f"- rows={len(feature_rows)}",
            f"- output={output_path}",
            f"- manifest={_join_location(gold_root, '_meta/feature_manifest.json')}",
        ]
    )


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    feature_rows = build_customer_features(
        asof_date=args.asof_date,
        silver_root=args.silver_root,
        gold_root=args.gold_root,
        silver_run_date=args.silver_run_date,
    )
    print(_render_summary(feature_rows, args.asof_date, args.gold_root))


if __name__ == "__main__":
    main()

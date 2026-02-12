import json
from pathlib import Path

from src.features.build_features import build_customer_features


class _FakeBody:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload


class FakeS3Client:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}

    def get_object(self, **kwargs: str) -> dict[str, _FakeBody]:
        key = (kwargs["Bucket"], kwargs["Key"])
        if key not in self.objects:
            raise FileNotFoundError(f"Missing object: s3://{key[0]}/{key[1]}")
        return {"Body": _FakeBody(self.objects[key])}

    def put_object(self, **kwargs: str | bytes) -> dict[str, object]:
        body = kwargs["Body"]
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.objects[(str(kwargs["Bucket"]), str(kwargs["Key"]))] = body
        return {}


def _write_silver_table(
    silver_root: Path, run_date: str, table_name: str, rows: list[dict]
) -> None:
    path = silver_root / table_name / f"run_date={run_date}" / f"{table_name}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(row, sort_keys=True) for row in rows]
    content = "\n".join(lines) + ("\n" if lines else "")
    path.write_text(content, encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_build_customer_features_filters_future_data_and_writes_manifest(
    tmp_path: Path,
) -> None:
    asof_date = "2026-01-31"
    silver_run_date = "2026-01-31"
    silver_root = tmp_path / "silver"
    gold_root = tmp_path / "gold"

    _write_silver_table(
        silver_root,
        silver_run_date,
        "customer_attributes",
        [
            {
                "customer_id": "C001",
                "province": "Madrid",
                "customer_type": "residential",
                "tariff_type": "fixed",
                "signup_date": "2024-01-10",
                "product_bundle": "dual",
            },
            {
                "customer_id": "C002",
                "province": "Madrid",
                "customer_type": "residential",
                "tariff_type": "fixed",
                "signup_date": "2026-02-15",
                "product_bundle": "dual",
            },
        ],
    )
    _write_silver_table(
        silver_root,
        silver_run_date,
        "customer_contracts",
        [
            {
                "contract_id": "CT001",
                "customer_id": "C001",
                "contract_start_date": "2025-12-01",
                "contract_end_date": "2026-03-01",
                "contract_status": "active",
                "contract_term_months": 12,
                "product_type": "dual",
            },
            {
                "contract_id": "CT002",
                "customer_id": "C001",
                "contract_start_date": "2026-02-02",
                "contract_end_date": "2027-02-02",
                "contract_status": "active",
                "contract_term_months": 12,
                "product_type": "dual",
            },
        ],
    )
    _write_silver_table(
        silver_root,
        silver_run_date,
        "price_history",
        [
            {
                "price_date": "2026-01-15",
                "product_type": "gas",
                "tariff_type": "fixed",
                "region_code": "MD",
                "price_eur_per_kwh": 0.20,
                "market_benchmark_eur_per_kwh": 0.15,
            },
            {
                "price_date": "2026-02-10",
                "product_type": "gas",
                "tariff_type": "fixed",
                "region_code": "MD",
                "price_eur_per_kwh": 0.50,
                "market_benchmark_eur_per_kwh": 0.10,
            },
        ],
    )
    _write_silver_table(
        silver_root,
        silver_run_date,
        "consumption_hourly_2024",
        [
            {
                "customer_id": "C001",
                "timestamp_utc": "2026-01-10T00:00:00+00:00",
                "commodity": "gas",
                "consumption_kwh": 10.0,
                "meter_id": "MTR01",
                "source_system": "ami",
                "negative_consumption_flag": 0,
            },
            {
                "customer_id": "C001",
                "timestamp_utc": "2026-01-20T00:00:00+00:00",
                "commodity": "gas",
                "consumption_kwh": 20.0,
                "meter_id": "MTR01",
                "source_system": "ami",
                "negative_consumption_flag": 0,
            },
            {
                "customer_id": "C001",
                "timestamp_utc": "2026-01-25T00:00:00+00:00",
                "commodity": "gas",
                "consumption_kwh": -5.0,
                "meter_id": "MTR01",
                "source_system": "ami",
                "negative_consumption_flag": 1,
            },
            {
                "customer_id": "C001",
                "timestamp_utc": "2026-02-05T00:00:00+00:00",
                "commodity": "gas",
                "consumption_kwh": 100.0,
                "meter_id": "MTR01",
                "source_system": "ami",
                "negative_consumption_flag": 0,
            },
        ],
    )
    _write_silver_table(
        silver_root,
        silver_run_date,
        "customer_interactions",
        [
            {
                "interaction_id": "I001",
                "customer_id": "C001",
                "interaction_ts": "2026-01-05T09:00:00+00:00",
                "channel": "call",
                "interaction_type": "billing",
                "sentiment_score": -0.2,
                "resolution_status": "resolved",
                "agent_id": "A12",
                "unknown_channel_flag": 0,
            },
            {
                "interaction_id": "I002",
                "customer_id": "C001",
                "interaction_ts": "2026-02-02T09:00:00+00:00",
                "channel": "call",
                "interaction_type": "complaint",
                "sentiment_score": -0.7,
                "resolution_status": "open",
                "agent_id": "A13",
                "unknown_channel_flag": 0,
            },
        ],
    )

    feature_rows = build_customer_features(
        asof_date=asof_date,
        silver_root=silver_root,
        gold_root=gold_root,
        silver_run_date=silver_run_date,
    )
    assert len(feature_rows) == 1
    assert feature_rows[0]["customer_id"] == "C001"
    assert feature_rows[0]["days_to_contract_end"] == 29
    assert feature_rows[0]["price_vs_benchmark_delta"] == 0.05
    assert feature_rows[0]["interaction_count_90d"] == 1
    assert feature_rows[0]["negative_consumption_flag"] == 1
    assert 10.0 <= feature_rows[0]["consumption_volatility_90d"] <= 11.0

    output_path = (
        gold_root
        / "customer_features_asof_date"
        / f"asof_date={asof_date}"
        / "customer_features_asof_date.jsonl"
    )
    assert output_path.exists()
    assert len(_read_jsonl(output_path)) == 1

    manifest_path = gold_root / "_meta" / "feature_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["asof_date"] == asof_date
    assert manifest["row_count"] == 1


def test_build_customer_features_supports_s3_paths() -> None:
    s3_client = FakeS3Client()
    asof_date = "2026-01-31"
    silver_run_date = "2026-01-31"
    silver_root = "s3://spanishgas-data-g1/silver/"
    gold_root = "s3://spanishgas-data-g1/gold/"

    source_tables = {
        "customer_attributes": [
            {
                "customer_id": "C001",
                "province": "Madrid",
                "customer_type": "residential",
                "tariff_type": "fixed",
                "signup_date": "2024-01-10",
                "product_bundle": "dual",
            }
        ],
        "customer_contracts": [
            {
                "contract_id": "CT001",
                "customer_id": "C001",
                "contract_start_date": "2025-12-01",
                "contract_end_date": "2026-03-01",
                "contract_status": "active",
                "contract_term_months": 12,
                "product_type": "dual",
            }
        ],
        "price_history": [
            {
                "price_date": "2026-01-15",
                "product_type": "gas",
                "tariff_type": "fixed",
                "region_code": "MD",
                "price_eur_per_kwh": 0.20,
                "market_benchmark_eur_per_kwh": 0.15,
            }
        ],
        "consumption_hourly_2024": [
            {
                "customer_id": "C001",
                "timestamp_utc": "2026-01-10T00:00:00+00:00",
                "commodity": "gas",
                "consumption_kwh": 10.0,
                "meter_id": "MTR01",
                "source_system": "ami",
                "negative_consumption_flag": 0,
            }
        ],
        "customer_interactions": [
            {
                "interaction_id": "I001",
                "customer_id": "C001",
                "interaction_ts": "2026-01-05T09:00:00+00:00",
                "channel": "call",
                "interaction_type": "billing",
                "sentiment_score": -0.2,
                "resolution_status": "resolved",
                "agent_id": "A12",
                "unknown_channel_flag": 0,
            }
        ],
    }
    for table_name, rows in source_tables.items():
        lines = [json.dumps(row, sort_keys=True) for row in rows]
        payload = ("\n".join(lines) + "\n").encode("utf-8")
        s3_client.put_object(
            Bucket="spanishgas-data-g1",
            Key=f"silver/{table_name}/run_date={silver_run_date}/{table_name}.jsonl",
            Body=payload,
        )

    feature_rows = build_customer_features(
        asof_date=asof_date,
        silver_root=silver_root,
        gold_root=gold_root,
        silver_run_date=silver_run_date,
        s3_client=s3_client,
    )
    assert len(feature_rows) == 1

    output_key = (
        "spanishgas-data-g1",
        "gold/customer_features_asof_date/asof_date=2026-01-31/customer_features_asof_date.jsonl",
    )
    manifest_key = ("spanishgas-data-g1", "gold/_meta/feature_manifest.json")
    assert output_key in s3_client.objects
    assert manifest_key in s3_client.objects

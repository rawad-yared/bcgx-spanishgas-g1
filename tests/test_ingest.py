import json
from pathlib import Path

from src.data.ingest import run_ingestion


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

    def head_object(self, **kwargs: str) -> dict[str, object]:
        key = (kwargs["Bucket"], kwargs["Key"])
        if key not in self.objects:
            raise FileNotFoundError(f"Missing object: s3://{key[0]}/{key[1]}")
        return {}


def _write_raw_fixtures(raw_root: Path) -> None:
    raw_root.mkdir(parents=True, exist_ok=True)

    (raw_root / "customer_attributes.csv").write_text(
        "customer_id,province,customer_type,tariff_type,signup_date,product_bundle\n"
        "C001,Madrid,residential,fixed,2024-01-10,dual\n",
        encoding="utf-8",
    )
    (raw_root / "customer_contracts.csv").write_text(
        "contract_id,customer_id,contract_start_date,contract_end_date,contract_status,contract_term_months,product_type\n"
        "CT001,C001,2024-01-10,2025-01-10,active,12,dual\n",
        encoding="utf-8",
    )
    (raw_root / "price_history.csv").write_text(
        "price_date,product_type,tariff_type,region_code,price_eur_per_kwh,market_benchmark_eur_per_kwh\n"
        "2024-02-01,gas,fixed,MD,0.121,0.118\n",
        encoding="utf-8",
    )
    (raw_root / "consumption_hourly_2024.csv").write_text(
        "customer_id,timestamp_utc,commodity,consumption_kwh,meter_id,source_system\n"
        "C001,2024-02-01T00:00:00Z,gas,1.8,MTR01,ami\n",
        encoding="utf-8",
    )
    (raw_root / "customer_interactions.json").write_text(
        json.dumps(
            [
                {
                    "interaction_id": "I001",
                    "customer_id": "C001",
                    "interaction_ts": "2024-02-01T09:15:00Z",
                    "channel": "call",
                    "interaction_type": "billing",
                    "sentiment_score": -0.2,
                    "resolution_status": "resolved",
                    "agent_id": "A12",
                }
            ]
        ),
        encoding="utf-8",
    )
    (raw_root / "costs_by_province_month.csv").write_text(
        "cost_month,province,commodity,variable_cost_eur_per_kwh,fixed_cost_eur_month,network_cost_eur_per_kwh\n"
        "2024-02-01,Madrid,gas,0.070,2.50,0.010\n",
        encoding="utf-8",
    )
    (raw_root / "churn_label.csv").write_text(
        "customer_id,label_date,horizon_days,churned_within_horizon,churn_effective_date\n"
        "C001,2024-02-01,90,0,\n",
        encoding="utf-8",
    )


def test_run_ingestion_writes_all_bronze_outputs_and_log(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    bronze_root = tmp_path / "bronze"
    run_date = "2026-02-09"
    _write_raw_fixtures(raw_root)

    records = run_ingestion(run_date=run_date, raw_root=raw_root, bronze_root=bronze_root)
    assert len(records) == 7

    for record in records:
        dataset_folder = Path(record.output_path).parent
        assert dataset_folder.name == f"run_date={run_date}"
        assert dataset_folder.exists()
        assert Path(record.output_path).exists()
        assert record.row_count >= 1
        assert record.schema_inferred

    log_path = bronze_root / "_meta" / "ingestion_log.jsonl"
    assert log_path.exists()

    lines = [line for line in log_path.read_text(encoding="utf-8").splitlines() if line]
    assert len(lines) == 7
    first_entry = json.loads(lines[0])
    assert set(first_entry) >= {
        "dataset_name",
        "ingested_at_utc",
        "output_path",
        "row_count",
        "run_date",
        "schema_inferred",
        "source_path",
    }


def test_run_ingestion_supports_s3_input_and_output() -> None:
    s3_client = FakeS3Client()
    raw_root = "s3://spanishgas-data-g1/raw/"
    bronze_root = "s3://spanishgas-data-g1/bronze/"
    run_date = "2026-02-09"

    source_payloads = {
        "customer_attributes.csv": (
            "customer_id,province,customer_type,tariff_type,signup_date,product_bundle\n"
            "C001,Madrid,residential,fixed,2024-01-10,dual\n"
        ),
        "customer_contracts.csv": (
            "contract_id,customer_id,contract_start_date,contract_end_date,contract_status,contract_term_months,product_type\n"
            "CT001,C001,2024-01-10,2025-01-10,active,12,dual\n"
        ),
        "price_history.csv": (
            "price_date,product_type,tariff_type,region_code,price_eur_per_kwh,market_benchmark_eur_per_kwh\n"
            "2024-02-01,gas,fixed,MD,0.121,0.118\n"
        ),
        "consumption_hourly_2024.csv": (
            "customer_id,timestamp_utc,commodity,consumption_kwh,meter_id,source_system\n"
            "C001,2024-02-01T00:00:00Z,gas,1.8,MTR01,ami\n"
        ),
        "customer_interactions.json": json.dumps(
            [
                {
                    "interaction_id": "I001",
                    "customer_id": "C001",
                    "interaction_ts": "2024-02-01T09:15:00Z",
                    "channel": "call",
                    "interaction_type": "billing",
                    "sentiment_score": -0.2,
                    "resolution_status": "resolved",
                    "agent_id": "A12",
                }
            ]
        ),
        "costs_by_province_month.csv": (
            "cost_month,province,commodity,variable_cost_eur_per_kwh,fixed_cost_eur_month,network_cost_eur_per_kwh\n"
            "2024-02-01,Madrid,gas,0.070,2.50,0.010\n"
        ),
        "churn_label.csv": (
            "customer_id,label_date,horizon_days,churned_within_horizon,churn_effective_date\n"
            "C001,2024-02-01,90,0,\n"
        ),
    }
    for dataset_name, payload in source_payloads.items():
        s3_client.put_object(
            Bucket="spanishgas-data-g1",
            Key=f"raw/{dataset_name}",
            Body=payload.encode("utf-8"),
        )

    records = run_ingestion(
        run_date=run_date,
        raw_root=raw_root,
        bronze_root=bronze_root,
        s3_client=s3_client,
    )
    assert len(records) == 7

    for record in records:
        assert record.source_path.startswith("s3://spanishgas-data-g1/raw/")
        assert record.output_path.startswith("s3://spanishgas-data-g1/bronze/")
        assert f"/run_date={run_date}/" in record.output_path
        assert record.row_count >= 1
        assert record.schema_inferred

    log_key = ("spanishgas-data-g1", "bronze/_meta/ingestion_log.jsonl")
    assert log_key in s3_client.objects
    log_lines = s3_client.objects[log_key].decode("utf-8").splitlines()
    assert len(log_lines) == 7

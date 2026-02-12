import json
from pathlib import Path

from src.data.silver import run_silver_transforms


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


def _write_bronze_fixtures(bronze_root: Path, run_date: str) -> None:
    def write(dataset_name: str, content: str) -> None:
        table_name = Path(dataset_name).stem
        target = bronze_root / table_name / f"run_date={run_date}" / dataset_name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    write(
        "customer_attributes.csv",
        "customer_id,province,customer_type,tariff_type,signup_date,product_bundle\n"
        "C001,Madrid,residential,fixed,2024-01-10,dual\n"
        ",Madrid,residential,variable,2024-01-11,electricity\n",
    )
    write(
        "customer_contracts.csv",
        "contract_id,customer_id,contract_start_date,contract_end_date,contract_status,contract_term_months,product_type\n"
        "CT001,C001,2024-01-10,2025-01-10,active,12,dual\n",
    )
    write(
        "price_history.csv",
        "price_date,product_type,tariff_type,region_code,price_eur_per_kwh,market_benchmark_eur_per_kwh\n"
        "2024-02-01,gas,fixed,MD,0.121,0.118\n",
    )
    write(
        "consumption_hourly_2024.csv",
        "customer_id,timestamp_utc,commodity,consumption_kwh,meter_id,source_system\n"
        "C001,2024-02-01T00:00:00Z,gas,-1.8,MTR01,ami\n"
        "C001,2024-02-01T01:00:00Z,gas,2.0,MTR01,ami\n",
    )
    write(
        "customer_interactions.json",
        json.dumps(
            [
                {
                    "interaction_id": "I001",
                    "customer_id": "C001",
                    "interaction_ts": "2024-02-01T09:15:00Z",
                    "channel": "fax",
                    "interaction_type": "billing",
                    "sentiment_score": -0.2,
                    "resolution_status": "resolved",
                    "agent_id": "A12",
                    "metadata": {"topic": "billing"},
                },
                {
                    "interaction_id": "I002",
                    "customer_id": "",
                    "interaction_ts": "2024-02-01T11:00:00Z",
                    "channel": "call",
                    "interaction_type": "complaint",
                    "sentiment_score": -0.6,
                    "resolution_status": "open",
                    "agent_id": "A13",
                },
            ]
        ),
    )
    write(
        "costs_by_province_month.csv",
        "cost_month,province,commodity,variable_cost_eur_per_kwh,fixed_cost_eur_month,network_cost_eur_per_kwh\n"
        "2024-02-01,Madrid,gas,0.070,2.50,0.010\n",
    )
    write(
        "churn_label.csv",
        "customer_id,label_date,horizon_days,churned_within_horizon,churn_effective_date\n"
        "C001,2024-02-01,90,0,\n",
    )


def _read_jsonl(path: Path) -> list[dict]:
    content = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in content if line.strip()]


def test_run_silver_transforms_local_outputs_and_quality_report(tmp_path: Path) -> None:
    run_date = "2026-02-09"
    bronze_root = tmp_path / "bronze"
    silver_root = tmp_path / "silver"
    _write_bronze_fixtures(bronze_root, run_date)

    reports = run_silver_transforms(
        run_date=run_date,
        bronze_root=bronze_root,
        silver_root=silver_root,
    )
    assert len(reports) == 7

    expected_tables = {
        "customer_attributes",
        "customer_contracts",
        "price_history",
        "consumption_hourly_2024",
        "customer_interactions",
        "costs_by_province_month",
        "churn_label",
    }
    for table_name in expected_tables:
        assert (
            silver_root
            / table_name
            / f"run_date={run_date}"
            / f"{table_name}.jsonl"
        ).exists()

    consumption_rows = _read_jsonl(
        silver_root
        / "consumption_hourly_2024"
        / f"run_date={run_date}"
        / "consumption_hourly_2024.jsonl"
    )
    assert len(consumption_rows) == 2
    assert any(
        row["consumption_kwh"] < 0 and row["negative_consumption_flag"] == 1
        for row in consumption_rows
    )

    interaction_rows = _read_jsonl(
        silver_root
        / "customer_interactions"
        / f"run_date={run_date}"
        / "customer_interactions.jsonl"
    )
    assert len(interaction_rows) == 1
    assert interaction_rows[0]["metadata_topic"] == "billing"
    assert interaction_rows[0]["unknown_channel_flag"] == 1

    quality_report_path = silver_root / "_meta" / f"quality_report_run_date={run_date}.json"
    assert quality_report_path.exists()
    report = json.loads(quality_report_path.read_text(encoding="utf-8"))
    report_by_dataset = {entry["dataset_name"]: entry for entry in report["datasets"]}
    assert report_by_dataset["customer_attributes.csv"]["dropped_missing_customer_id"] == 1
    assert (
        report_by_dataset["consumption_hourly_2024.csv"][
            "negative_consumption_flagged"
        ]
        == 1
    )
    assert report_by_dataset["customer_interactions.json"]["unknown_channel_count"] == 1


def test_run_silver_transforms_supports_s3_paths() -> None:
    s3_client = FakeS3Client()
    run_date = "2026-02-09"
    bronze_root = "s3://spanishgas-data-g1/bronze/"
    silver_root = "s3://spanishgas-data-g1/silver/"

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
            "C001,2024-02-01T00:00:00Z,gas,-1.8,MTR01,ami\n"
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
        table_name = Path(dataset_name).stem
        s3_client.put_object(
            Bucket="spanishgas-data-g1",
            Key=f"bronze/{table_name}/run_date={run_date}/{dataset_name}",
            Body=payload.encode("utf-8"),
        )

    reports = run_silver_transforms(
        run_date=run_date,
        bronze_root=bronze_root,
        silver_root=silver_root,
        s3_client=s3_client,
    )
    assert len(reports) == 7

    report_key = ("spanishgas-data-g1", f"silver/_meta/quality_report_run_date={run_date}.json")
    assert report_key in s3_client.objects
    table_key = (
        "spanishgas-data-g1",
        f"silver/customer_interactions/run_date={run_date}/customer_interactions.jsonl",
    )
    assert table_key in s3_client.objects

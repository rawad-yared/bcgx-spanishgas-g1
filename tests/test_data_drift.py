import json
from pathlib import Path

from src.monitoring.data_drift import run_data_drift_monitoring


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

    def list_objects_v2(self, **kwargs: str) -> dict[str, object]:
        bucket = kwargs["Bucket"]
        prefix = kwargs.get("Prefix", "")
        contents = [
            {"Key": key}
            for (obj_bucket, key), _payload in self.objects.items()
            if obj_bucket == bucket and key.startswith(prefix)
        ]
        return {"Contents": contents, "IsTruncated": False}


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(row, sort_keys=True) for row in rows]
    path.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")


def test_run_data_drift_monitoring_flags_missingness_and_psi_changes(
    tmp_path: Path,
) -> None:
    gold_root = tmp_path / "gold"
    artifacts_root = tmp_path / "artifacts"
    run_date = "2026-02-09"

    _write_jsonl(
        gold_root
        / "customer_features_asof_date"
        / "asof_date=2026-01-31"
        / "customer_features_asof_date.jsonl",
        [
            {
                "customer_id": "C001",
                "asof_date": "2026-01-31",
                "tenure_days": 120,
                "days_to_contract_end": 95,
                "price_vs_benchmark_delta": 0.01,
                "interaction_count_90d": 0,
                "negative_consumption_flag": 0,
            },
            {
                "customer_id": "C002",
                "asof_date": "2026-01-31",
                "tenure_days": 140,
                "days_to_contract_end": 100,
                "price_vs_benchmark_delta": 0.02,
                "interaction_count_90d": 1,
                "negative_consumption_flag": 0,
            },
            {
                "customer_id": "C003",
                "asof_date": "2026-01-31",
                "tenure_days": 150,
                "days_to_contract_end": 110,
                "price_vs_benchmark_delta": 0.02,
                "interaction_count_90d": 1,
                "negative_consumption_flag": 0,
            },
        ],
    )
    _write_jsonl(
        gold_root
        / "customer_features_asof_date"
        / "asof_date=2026-02-01"
        / "customer_features_asof_date.jsonl",
        [
            {
                "customer_id": "C001",
                "asof_date": "2026-02-01",
                "tenure_days": 121,
                "days_to_contract_end": 20,
                "price_vs_benchmark_delta": 0.21,
                "interaction_count_90d": 5,
                "negative_consumption_flag": 1,
            },
            {
                "customer_id": "C002",
                "asof_date": "2026-02-01",
                "tenure_days": 141,
                "days_to_contract_end": None,
                "price_vs_benchmark_delta": 0.24,
                "interaction_count_90d": 6,
                "negative_consumption_flag": 1,
            },
            {
                "customer_id": "C003",
                "asof_date": "2026-02-01",
                "tenure_days": 151,
                "days_to_contract_end": None,
                "price_vs_benchmark_delta": 0.23,
                "interaction_count_90d": 4,
                "negative_consumption_flag": 1,
            },
        ],
    )
    _write_jsonl(
        gold_root / "scoring" / f"run_date={run_date}" / "scores.jsonl",
        [
            {
                "customer_id": "C001",
                "run_date": run_date,
                "asof_date": "2026-02-01",
                "risk_score": 0.9,
            }
        ],
    )

    summary = run_data_drift_monitoring(
        run_date=run_date,
        gold_root=gold_root,
        artifacts_root=artifacts_root,
        psi_threshold=0.1,
        missingness_delta_threshold=0.2,
        bins=5,
    )

    assert summary.current_asof_date == "2026-02-01"
    assert summary.baseline_asof_date == "2026-01-31"
    assert summary.baseline_source == "previous_feature_snapshot"
    assert summary.threshold_exceeded is True
    assert Path(summary.metrics_json_path).exists()
    assert Path(summary.report_path).exists()

    payload = json.loads(Path(summary.metrics_json_path).read_text(encoding="utf-8"))
    flagged = [item for item in payload["feature_metrics"] if item["threshold_exceeded"]]
    assert flagged
    assert any(item["missingness_exceeds_threshold"] for item in flagged)
    assert any(item["psi_exceeds_threshold"] for item in flagged)


def test_run_data_drift_monitoring_falls_back_when_no_prior_snapshot(
    tmp_path: Path,
) -> None:
    gold_root = tmp_path / "gold"
    artifacts_root = tmp_path / "artifacts"
    run_date = "2026-02-09"
    asof_date = "2026-01-31"

    _write_jsonl(
        gold_root
        / "customer_features_asof_date"
        / f"asof_date={asof_date}"
        / "customer_features_asof_date.jsonl",
        [
            {
                "customer_id": "C001",
                "asof_date": asof_date,
                "tenure_days": 200,
                "days_to_contract_end": 45,
                "price_vs_benchmark_delta": 0.02,
                "interaction_count_90d": 1,
                "negative_consumption_flag": 0,
            }
        ],
    )
    _write_jsonl(
        gold_root / "scoring" / f"run_date={run_date}" / "scores.jsonl",
        [
            {
                "customer_id": "C001",
                "run_date": run_date,
                "asof_date": asof_date,
                "risk_score": 0.3,
            }
        ],
    )

    summary = run_data_drift_monitoring(
        run_date=run_date,
        gold_root=gold_root,
        artifacts_root=artifacts_root,
    )

    assert summary.current_asof_date == asof_date
    assert summary.baseline_asof_date == asof_date
    assert summary.baseline_source == "fallback_current_as_baseline"
    assert summary.threshold_exceeded is False


def test_run_data_drift_monitoring_supports_s3_paths() -> None:
    s3_client = FakeS3Client()
    run_date = "2026-02-09"

    baseline_rows = [
        {
            "customer_id": "C001",
            "asof_date": "2026-01-31",
            "tenure_days": 100,
            "days_to_contract_end": 90,
            "price_vs_benchmark_delta": 0.01,
            "interaction_count_90d": 0,
            "negative_consumption_flag": 0,
        }
    ]
    current_rows = [
        {
            "customer_id": "C001",
            "asof_date": "2026-02-01",
            "tenure_days": 101,
            "days_to_contract_end": 30,
            "price_vs_benchmark_delta": 0.20,
            "interaction_count_90d": 4,
            "negative_consumption_flag": 1,
        }
    ]

    s3_client.put_object(
        Bucket="spanishgas-data-g1",
        Key=(
            "gold/customer_features_asof_date/"
            "asof_date=2026-01-31/customer_features_asof_date.jsonl"
        ),
        Body=(json.dumps(baseline_rows[0], sort_keys=True) + "\n").encode("utf-8"),
    )
    s3_client.put_object(
        Bucket="spanishgas-data-g1",
        Key=(
            "gold/customer_features_asof_date/"
            "asof_date=2026-02-01/customer_features_asof_date.jsonl"
        ),
        Body=(json.dumps(current_rows[0], sort_keys=True) + "\n").encode("utf-8"),
    )
    s3_client.put_object(
        Bucket="spanishgas-data-g1",
        Key="gold/scoring/run_date=2026-02-09/scores.jsonl",
        Body=(
            json.dumps(
                {
                    "customer_id": "C001",
                    "run_date": run_date,
                    "asof_date": "2026-02-01",
                    "risk_score": 0.7,
                },
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8"),
    )

    summary = run_data_drift_monitoring(
        run_date=run_date,
        gold_root="s3://spanishgas-data-g1/gold/",
        artifacts_root="s3://spanishgas-data-g1/artifacts/",
        s3_client=s3_client,
    )

    assert summary.current_asof_date == "2026-02-01"
    assert summary.baseline_asof_date == "2026-01-31"
    assert (
        "spanishgas-data-g1",
        "artifacts/monitoring/data_drift_run_date=2026-02-09.json",
    ) in s3_client.objects
    assert (
        "spanishgas-data-g1",
        "artifacts/monitoring/data_drift_run_date=2026-02-09.md",
    ) in s3_client.objects

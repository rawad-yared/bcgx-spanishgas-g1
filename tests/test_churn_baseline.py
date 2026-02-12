import json
from pathlib import Path

from src.models.churn_baseline import _load_config
from src.models.churn_baseline import train_baseline


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


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(row, sort_keys=True) for row in rows]
    path.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")


def test_train_baseline_writes_artifacts_and_report_locally(tmp_path: Path) -> None:
    dataset_path = (
        tmp_path
        / "gold"
        / "churn_training_dataset"
        / "cutoff_date=2026-01-31"
        / "churn_training_dataset.jsonl"
    )
    _write_jsonl(
        dataset_path,
        [
            {
                "customer_id": "C001",
                "asof_date": "2025-11-30",
                "feature_version": "v1",
                "tenure_days": 100,
                "days_to_contract_end": 30,
                "price_vs_benchmark_delta": 0.05,
                "consumption_volatility_90d": 1.2,
                "interaction_count_90d": 2,
                "negative_consumption_flag": 0,
                "label_horizon_days": 90,
                "churn_label": 1,
                "split": "train",
            },
            {
                "customer_id": "C002",
                "asof_date": "2025-11-30",
                "feature_version": "v1",
                "tenure_days": 230,
                "days_to_contract_end": 100,
                "price_vs_benchmark_delta": -0.01,
                "consumption_volatility_90d": 0.4,
                "interaction_count_90d": 0,
                "negative_consumption_flag": 0,
                "label_horizon_days": 90,
                "churn_label": 0,
                "split": "train",
            },
            {
                "customer_id": "C003",
                "asof_date": "2025-12-31",
                "feature_version": "v1",
                "tenure_days": 150,
                "days_to_contract_end": 20,
                "price_vs_benchmark_delta": 0.03,
                "consumption_volatility_90d": 1.0,
                "interaction_count_90d": 1,
                "negative_consumption_flag": 1,
                "label_horizon_days": 90,
                "churn_label": 1,
                "split": "valid",
            },
            {
                "customer_id": "C004",
                "asof_date": "2026-01-31",
                "feature_version": "v1",
                "tenure_days": 300,
                "days_to_contract_end": 200,
                "price_vs_benchmark_delta": -0.02,
                "consumption_volatility_90d": 0.2,
                "interaction_count_90d": 0,
                "negative_consumption_flag": 0,
                "label_horizon_days": 90,
                "churn_label": 0,
                "split": "test",
            },
        ],
    )

    config_path = tmp_path / "model_baseline.yaml"
    config_path.write_text(
        "\n".join(
            [
                "paths:",
                f"  training_dataset_path: {dataset_path}",
                f"  model_output_dir: {tmp_path / 'artifacts' / 'models' / 'churn_baseline'}",
                f"  report_path: {tmp_path / 'artifacts' / 'reports' / 'churn_baseline.md'}",
                "model:",
                "  random_seed: 42",
                "  max_iter: 200",
                "  regularization_c: 1.0",
                "evaluation:",
                "  top_k_fraction: 0.5",
                "  calibration_bins: 5",
            ]
        ),
        encoding="utf-8",
    )

    config = _load_config(str(config_path))
    result = train_baseline(config)

    assert result.model_path.endswith("model.pkl")
    assert Path(result.model_path).exists()
    assert Path(result.metrics_path).exists()
    assert Path(result.report_path).exists()
    assert result.split_counts == {"train": 2, "valid": 1, "test": 1}
    assert "test" in result.metrics

    metrics_json = json.loads(Path(result.metrics_path).read_text(encoding="utf-8"))
    assert set(metrics_json["metrics"]["test"]) >= {
        "pr_auc",
        "recall_at_k",
        "precision_at_k",
        "brier_score",
        "expected_calibration_error",
    }
    report_text = Path(result.report_path).read_text(encoding="utf-8")
    assert "PR-AUC" in report_text
    assert "Recall@K" in report_text
    assert "Precision@K" in report_text
    assert "ECE" in report_text


def test_train_baseline_supports_s3_paths(tmp_path: Path) -> None:
    s3_client = FakeS3Client()
    rows = [
        {
            "customer_id": "C001",
            "asof_date": "2025-11-30",
            "feature_version": "v1",
            "tenure_days": 100,
            "days_to_contract_end": 30,
            "price_vs_benchmark_delta": 0.05,
            "interaction_count_90d": 2,
            "negative_consumption_flag": 0,
            "label_horizon_days": 90,
            "churn_label": 1,
            "split": "train",
        },
        {
            "customer_id": "C002",
            "asof_date": "2025-11-30",
            "feature_version": "v1",
            "tenure_days": 220,
            "days_to_contract_end": 120,
            "price_vs_benchmark_delta": -0.02,
            "interaction_count_90d": 0,
            "negative_consumption_flag": 0,
            "label_horizon_days": 90,
            "churn_label": 0,
            "split": "train",
        },
        {
            "customer_id": "C003",
            "asof_date": "2026-01-31",
            "feature_version": "v1",
            "tenure_days": 150,
            "days_to_contract_end": 15,
            "price_vs_benchmark_delta": 0.08,
            "interaction_count_90d": 4,
            "negative_consumption_flag": 1,
            "label_horizon_days": 90,
            "churn_label": 1,
            "split": "test",
        },
    ]
    payload = ("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n").encode(
        "utf-8"
    )
    s3_client.put_object(
        Bucket="spanishgas-data-g1",
        Key=(
            "gold/churn_training_dataset/"
            "cutoff_date=2026-01-31/churn_training_dataset.jsonl"
        ),
        Body=payload,
    )

    config_path = tmp_path / "model_baseline.s3.yaml"
    config_path.write_text(
        "\n".join(
            [
                "paths:",
                "  training_dataset_path: s3://spanishgas-data-g1/gold/churn_training_dataset/cutoff_date=2026-01-31/churn_training_dataset.jsonl",
                "  model_output_dir: s3://spanishgas-data-g1/artifacts/models/churn_baseline/",
                "  report_path: s3://spanishgas-data-g1/artifacts/reports/churn_baseline.md",
                "model:",
                "  random_seed: 42",
                "  max_iter: 200",
                "  regularization_c: 1.0",
                "evaluation:",
                "  top_k_fraction: 0.5",
                "  calibration_bins: 5",
            ]
        ),
        encoding="utf-8",
    )
    config = _load_config(str(config_path))
    result = train_baseline(config=config, s3_client=s3_client)

    assert result.primary_eval_split in {"train", "valid", "test"}
    assert (
        "spanishgas-data-g1",
        "artifacts/models/churn_baseline/model.pkl",
    ) in s3_client.objects
    assert (
        "spanishgas-data-g1",
        "artifacts/models/churn_baseline/metrics.json",
    ) in s3_client.objects
    assert (
        "spanishgas-data-g1",
        "artifacts/reports/churn_baseline.md",
    ) in s3_client.objects

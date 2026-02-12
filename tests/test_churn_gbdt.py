import json
from pathlib import Path

import pytest

from src.models.churn_gbdt import _load_config
from src.models.churn_gbdt import train_gbdt


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


def _sample_training_rows() -> list[dict]:
    return [
        {
            "customer_id": "C001",
            "asof_date": "2025-11-30",
            "feature_version": "v1",
            "tenure_days": 100,
            "days_to_contract_end": 20,
            "price_vs_benchmark_delta": 0.08,
            "interaction_count_90d": 4,
            "negative_consumption_flag": 1,
            "label_horizon_days": 90,
            "churn_label": 1,
            "split": "train",
        },
        {
            "customer_id": "C002",
            "asof_date": "2025-11-30",
            "feature_version": "v1",
            "tenure_days": 260,
            "days_to_contract_end": 180,
            "price_vs_benchmark_delta": -0.03,
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
            "tenure_days": 130,
            "days_to_contract_end": 35,
            "price_vs_benchmark_delta": 0.05,
            "interaction_count_90d": 2,
            "negative_consumption_flag": 1,
            "label_horizon_days": 90,
            "churn_label": 1,
            "split": "valid",
        },
        {
            "customer_id": "C004",
            "asof_date": "2025-12-31",
            "feature_version": "v1",
            "tenure_days": 240,
            "days_to_contract_end": 210,
            "price_vs_benchmark_delta": -0.02,
            "interaction_count_90d": 0,
            "negative_consumption_flag": 0,
            "label_horizon_days": 90,
            "churn_label": 0,
            "split": "valid",
        },
        {
            "customer_id": "C005",
            "asof_date": "2026-01-31",
            "feature_version": "v1",
            "tenure_days": 120,
            "days_to_contract_end": 18,
            "price_vs_benchmark_delta": 0.06,
            "interaction_count_90d": 3,
            "negative_consumption_flag": 1,
            "label_horizon_days": 90,
            "churn_label": 1,
            "split": "test",
        },
        {
            "customer_id": "C006",
            "asof_date": "2026-01-31",
            "feature_version": "v1",
            "tenure_days": 280,
            "days_to_contract_end": 240,
            "price_vs_benchmark_delta": -0.04,
            "interaction_count_90d": 0,
            "negative_consumption_flag": 0,
            "label_horizon_days": 90,
            "churn_label": 0,
            "split": "test",
        },
    ]


def _baseline_metrics_payload(pr_auc: float, recall: float) -> dict:
    return {
        "primary_eval_split": "test",
        "split_counts": {"train": 2, "valid": 2, "test": 2},
        "metrics": {
            "train": {
                "split": "train",
                "row_count": 2,
                "positive_rate": 0.5,
                "pr_auc": 0.5,
                "recall_at_k": 0.5,
                "precision_at_k": 0.5,
                "brier_score": 0.25,
                "expected_calibration_error": 0.2,
                "avg_predicted_probability": 0.5,
            },
            "valid": {
                "split": "valid",
                "row_count": 2,
                "positive_rate": 0.5,
                "pr_auc": 0.5,
                "recall_at_k": 0.5,
                "precision_at_k": 0.5,
                "brier_score": 0.25,
                "expected_calibration_error": 0.2,
                "avg_predicted_probability": 0.5,
            },
            "test": {
                "split": "test",
                "row_count": 2,
                "positive_rate": 0.5,
                "pr_auc": pr_auc,
                "recall_at_k": recall,
                "precision_at_k": 0.5,
                "brier_score": 0.3,
                "expected_calibration_error": 0.2,
                "avg_predicted_probability": 0.5,
            },
        },
    }


def test_train_gbdt_writes_artifacts_and_comparison_report(tmp_path: Path) -> None:
    dataset_path = (
        tmp_path
        / "gold"
        / "churn_training_dataset"
        / "cutoff_date=2026-01-31"
        / "churn_training_dataset.jsonl"
    )
    _write_jsonl(dataset_path, _sample_training_rows())

    baseline_metrics_path = tmp_path / "artifacts" / "models" / "churn_baseline" / "metrics.json"
    baseline_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_metrics_path.write_text(
        json.dumps(_baseline_metrics_payload(pr_auc=0.1, recall=0.0), indent=2),
        encoding="utf-8",
    )

    config_path = tmp_path / "model_gbdt.yaml"
    config_path.write_text(
        "\n".join(
            [
                "paths:",
                f"  training_dataset_path: {dataset_path}",
                f"  baseline_metrics_path: {baseline_metrics_path}",
                f"  model_output_dir: {tmp_path / 'artifacts' / 'models' / 'churn_gbdt'}",
                f"  comparison_report_path: {tmp_path / 'artifacts' / 'reports' / 'churn_gbdt_vs_baseline.md'}",
                "model:",
                "  random_seed: 42",
                "  n_estimators: 150",
                "  learning_rate: 0.05",
                "  max_depth: 2",
                "evaluation:",
                "  top_k_fraction: 0.5",
                "  calibration_bins: 5",
                "  max_brier_increase: 0.5",
                "  max_ece_increase: 0.5",
                "  top_feature_count: 5",
                "  max_shap_samples: 50",
            ]
        ),
        encoding="utf-8",
    )

    config = _load_config(str(config_path))
    result = train_gbdt(config)

    assert result.champion_pass is True
    assert Path(result.model_path).exists()
    assert Path(result.metrics_path).exists()
    assert Path(result.report_path).exists()

    metrics_payload = json.loads(Path(result.metrics_path).read_text(encoding="utf-8"))
    assert metrics_payload["comparison"]["champion_pass"] is True
    assert metrics_payload["feature_importance"]
    assert metrics_payload["shap_summary"]["method"] in {"shap", "permutation_proxy"}
    report_text = Path(result.report_path).read_text(encoding="utf-8")
    assert "Feature Importance" in report_text
    assert "SHAP Summary" in report_text


def test_train_gbdt_raises_when_champion_criteria_fail(tmp_path: Path) -> None:
    dataset_path = (
        tmp_path
        / "gold"
        / "churn_training_dataset"
        / "cutoff_date=2026-01-31"
        / "churn_training_dataset.jsonl"
    )
    _write_jsonl(dataset_path, _sample_training_rows())

    baseline_metrics_path = tmp_path / "baseline_metrics.json"
    baseline_metrics_path.write_text(
        json.dumps(_baseline_metrics_payload(pr_auc=1.0, recall=1.0), indent=2),
        encoding="utf-8",
    )

    config_path = tmp_path / "model_gbdt_fail.yaml"
    config_path.write_text(
        "\n".join(
            [
                "paths:",
                f"  training_dataset_path: {dataset_path}",
                f"  baseline_metrics_path: {baseline_metrics_path}",
                f"  model_output_dir: {tmp_path / 'artifacts' / 'models' / 'churn_gbdt'}",
                f"  comparison_report_path: {tmp_path / 'artifacts' / 'reports' / 'churn_gbdt_vs_baseline.md'}",
                "model:",
                "  random_seed: 42",
                "  n_estimators: 100",
                "  learning_rate: 0.05",
                "  max_depth: 2",
                "evaluation:",
                "  top_k_fraction: 0.5",
                "  calibration_bins: 5",
                "  max_brier_increase: 0.0",
                "  max_ece_increase: 0.0",
                "  top_feature_count: 5",
                "  max_shap_samples: 50",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="champion criteria"):
        train_gbdt(_load_config(str(config_path)))


def test_train_gbdt_supports_s3_paths(tmp_path: Path) -> None:
    s3_client = FakeS3Client()
    rows = _sample_training_rows()
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
    s3_client.put_object(
        Bucket="spanishgas-data-g1",
        Key="artifacts/models/churn_baseline/metrics.json",
        Body=json.dumps(_baseline_metrics_payload(pr_auc=0.1, recall=0.0)).encode("utf-8"),
    )

    config_path = tmp_path / "model_gbdt_s3.yaml"
    config_path.write_text(
        "\n".join(
            [
                "paths:",
                "  training_dataset_path: s3://spanishgas-data-g1/gold/churn_training_dataset/cutoff_date=2026-01-31/churn_training_dataset.jsonl",
                "  baseline_metrics_path: s3://spanishgas-data-g1/artifacts/models/churn_baseline/metrics.json",
                "  model_output_dir: s3://spanishgas-data-g1/artifacts/models/churn_gbdt/",
                "  comparison_report_path: s3://spanishgas-data-g1/artifacts/reports/churn_gbdt_vs_baseline.md",
                "model:",
                "  random_seed: 42",
                "  n_estimators: 100",
                "  learning_rate: 0.05",
                "  max_depth: 2",
                "evaluation:",
                "  top_k_fraction: 0.5",
                "  calibration_bins: 5",
                "  max_brier_increase: 0.5",
                "  max_ece_increase: 0.5",
                "  top_feature_count: 5",
                "  max_shap_samples: 50",
            ]
        ),
        encoding="utf-8",
    )

    result = train_gbdt(_load_config(str(config_path)), s3_client=s3_client)
    assert result.champion_pass is True
    assert (
        "spanishgas-data-g1",
        "artifacts/models/churn_gbdt/model.pkl",
    ) in s3_client.objects
    assert (
        "spanishgas-data-g1",
        "artifacts/models/churn_gbdt/metrics.json",
    ) in s3_client.objects
    assert (
        "spanishgas-data-g1",
        "artifacts/reports/churn_gbdt_vs_baseline.md",
    ) in s3_client.objects

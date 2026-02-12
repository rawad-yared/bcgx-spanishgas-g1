import json
from pathlib import Path

from src.monitoring.model_perf import MonitorConfig
from src.monitoring.model_perf import run_model_performance_monitoring


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


def _config(
    scoring_path: str,
    labels_path: str,
    report_path: str,
    min_pr_auc: float = 0.6,
    min_recall_at_k: float = 0.5,
    max_segment_calibration_drift: float = 0.15,
    min_evaluation_rows: int = 1,
) -> MonitorConfig:
    return MonitorConfig(
        scoring_path=scoring_path,
        labels_path=labels_path,
        report_path=report_path,
        score_customer_id_col="customer_id",
        score_value_col="risk_score",
        score_segment_col="segment",
        score_time_bucket_col="run_date",
        label_customer_id_col="customer_id",
        label_value_col="churned_within_horizon",
        label_time_bucket_col="label_date",
        label_horizon_days_col="horizon_days",
        label_time_bucket_for_scores="2026-01-31",
        required_horizon_days=90,
        top_k_fraction=0.5,
        calibration_bins=5,
        min_pr_auc=min_pr_auc,
        min_recall_at_k=min_recall_at_k,
        max_segment_calibration_drift=max_segment_calibration_drift,
        min_evaluation_rows=min_evaluation_rows,
    )


def test_model_perf_generates_report_and_metrics_json(tmp_path: Path) -> None:
    scoring_path = tmp_path / "scores.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    report_path = tmp_path / "artifacts" / "monitoring" / "model_perf.md"

    _write_jsonl(
        scoring_path,
        [
            {"customer_id": "C001", "run_date": "2026-02-08", "segment": "S01", "risk_score": 0.8},
            {"customer_id": "C002", "run_date": "2026-02-08", "segment": "S01", "risk_score": 0.7},
            {"customer_id": "C003", "run_date": "2026-02-08", "segment": "S02", "risk_score": 0.3},
            {"customer_id": "C004", "run_date": "2026-02-08", "segment": "S02", "risk_score": 0.2},
            {"customer_id": "C001", "run_date": "2026-02-09", "segment": "S01", "risk_score": 0.9},
            {"customer_id": "C002", "run_date": "2026-02-09", "segment": "S01", "risk_score": 0.1},
            {"customer_id": "C003", "run_date": "2026-02-09", "segment": "S02", "risk_score": 0.85},
            {"customer_id": "C004", "run_date": "2026-02-09", "segment": "S02", "risk_score": 0.05},
        ],
    )
    _write_jsonl(
        labels_path,
        [
            {"customer_id": "C001", "label_date": "2026-01-31", "horizon_days": 90, "churned_within_horizon": 1},
            {"customer_id": "C002", "label_date": "2026-01-31", "horizon_days": 90, "churned_within_horizon": 0},
            {"customer_id": "C003", "label_date": "2026-01-31", "horizon_days": 90, "churned_within_horizon": 1},
            {"customer_id": "C004", "label_date": "2026-01-31", "horizon_days": 90, "churned_within_horizon": 0},
        ],
    )

    summary = run_model_performance_monitoring(
        _config(
            scoring_path=str(scoring_path),
            labels_path=str(labels_path),
            report_path=str(report_path),
            max_segment_calibration_drift=0.0001,
        )
    )

    assert summary.evaluated_rows == 8
    assert summary.current_bucket == "2026-02-09"
    assert summary.baseline_bucket == "2026-02-08"
    assert summary.retraining_trigger is True
    assert Path(summary.report_path).exists()
    assert Path(summary.metrics_json_path).exists()

    metrics_payload = json.loads(Path(summary.metrics_json_path).read_text(encoding="utf-8"))
    assert "overall_by_time_bucket" in metrics_payload
    assert "segment_calibration_drift" in metrics_payload
    assert metrics_payload["retraining_trigger"]["value"] is True

    report_text = Path(summary.report_path).read_text(encoding="utf-8")
    assert "PR-AUC" in report_text
    assert "Recall@K" in report_text
    assert "Calibration Drift vs Baseline" in report_text


def test_model_perf_no_trigger_under_lenient_thresholds(tmp_path: Path) -> None:
    scoring_path = tmp_path / "scores_one_bucket.jsonl"
    labels_path = tmp_path / "labels_one_bucket.jsonl"
    report_path = tmp_path / "artifacts" / "monitoring" / "model_perf.md"

    _write_jsonl(
        scoring_path,
        [
            {"customer_id": "C001", "run_date": "2026-02-09", "segment": "S01", "risk_score": 0.9},
            {"customer_id": "C002", "run_date": "2026-02-09", "segment": "S01", "risk_score": 0.1},
            {"customer_id": "C003", "run_date": "2026-02-09", "segment": "S02", "risk_score": 0.8},
            {"customer_id": "C004", "run_date": "2026-02-09", "segment": "S02", "risk_score": 0.2},
        ],
    )
    _write_jsonl(
        labels_path,
        [
            {"customer_id": "C001", "label_date": "2026-01-31", "horizon_days": 90, "churned_within_horizon": 1},
            {"customer_id": "C002", "label_date": "2026-01-31", "horizon_days": 90, "churned_within_horizon": 0},
            {"customer_id": "C003", "label_date": "2026-01-31", "horizon_days": 90, "churned_within_horizon": 1},
            {"customer_id": "C004", "label_date": "2026-01-31", "horizon_days": 90, "churned_within_horizon": 0},
        ],
    )

    summary = run_model_performance_monitoring(
        _config(
            scoring_path=str(scoring_path),
            labels_path=str(labels_path),
            report_path=str(report_path),
            min_pr_auc=0.2,
            min_recall_at_k=0.2,
            max_segment_calibration_drift=1.0,
            min_evaluation_rows=1,
        )
    )
    assert summary.retraining_trigger is False
    assert not summary.retraining_reasons


def test_model_perf_supports_s3_paths() -> None:
    s3_client = FakeS3Client()
    scoring_rows = [
        {"customer_id": "C001", "run_date": "2026-02-09", "segment": "S01", "risk_score": 0.9},
        {"customer_id": "C002", "run_date": "2026-02-09", "segment": "S01", "risk_score": 0.1},
    ]
    label_rows = [
        {"customer_id": "C001", "label_date": "2026-01-31", "horizon_days": 90, "churned_within_horizon": 1},
        {"customer_id": "C002", "label_date": "2026-01-31", "horizon_days": 90, "churned_within_horizon": 0},
    ]
    s3_client.put_object(
        Bucket="spanishgas-data-g1",
        Key="monitoring/scores.jsonl",
        Body=(json.dumps(scoring_rows[0], sort_keys=True) + "\n" + json.dumps(scoring_rows[1], sort_keys=True) + "\n").encode("utf-8"),
    )
    s3_client.put_object(
        Bucket="spanishgas-data-g1",
        Key="monitoring/labels.jsonl",
        Body=(json.dumps(label_rows[0], sort_keys=True) + "\n" + json.dumps(label_rows[1], sort_keys=True) + "\n").encode("utf-8"),
    )

    summary = run_model_performance_monitoring(
        _config(
            scoring_path="s3://spanishgas-data-g1/monitoring/scores.jsonl",
            labels_path="s3://spanishgas-data-g1/monitoring/labels.jsonl",
            report_path="s3://spanishgas-data-g1/artifacts/monitoring/model_perf.md",
            min_pr_auc=0.2,
            min_recall_at_k=0.2,
            max_segment_calibration_drift=1.0,
            min_evaluation_rows=1,
        ),
        s3_client=s3_client,
    )

    assert summary.evaluated_rows == 2
    assert (
        "spanishgas-data-g1",
        "artifacts/monitoring/model_perf.md",
    ) in s3_client.objects
    assert (
        "spanishgas-data-g1",
        "artifacts/monitoring/model_perf_metrics.json",
    ) in s3_client.objects

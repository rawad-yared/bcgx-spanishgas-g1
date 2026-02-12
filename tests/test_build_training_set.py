import json
from datetime import date
from pathlib import Path

from src.data.build_training_set import SplitRules
from src.data.build_training_set import TrainingSetConfig
from src.data.build_training_set import _load_config
from src.data.build_training_set import build_training_set


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


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_build_training_set_local_join_and_temporal_split(tmp_path: Path) -> None:
    silver_root = tmp_path / "silver"
    gold_root = tmp_path / "gold"

    _write_jsonl(
        silver_root / "churn_label" / "run_date=2026-01-31" / "churn_label.jsonl",
        [
            {
                "customer_id": "C001",
                "label_date": "2025-11-30",
                "horizon_days": 90,
                "churned_within_horizon": 1,
                "churn_effective_date": "2026-01-15",
            },
            {
                "customer_id": "C001",
                "label_date": "2025-12-31",
                "horizon_days": 90,
                "churned_within_horizon": 0,
                "churn_effective_date": None,
            },
            {
                "customer_id": "C001",
                "label_date": "2026-01-31",
                "horizon_days": 90,
                "churned_within_horizon": 1,
                "churn_effective_date": "2026-02-15",
            },
            {
                "customer_id": "C001",
                "label_date": "2026-01-31",
                "horizon_days": 60,
                "churned_within_horizon": 0,
                "churn_effective_date": None,
            },
        ],
    )

    _write_jsonl(
        gold_root
        / "customer_features_asof_date"
        / "asof_date=2025-11-30"
        / "customer_features_asof_date.jsonl",
        [
            {
                "customer_id": "C001",
                "asof_date": "2025-11-30",
                "feature_version": "v1",
                "tenure_days": 100,
                "interaction_count_90d": 2,
            }
        ],
    )
    _write_jsonl(
        gold_root
        / "customer_features_asof_date"
        / "asof_date=2025-12-31"
        / "customer_features_asof_date.jsonl",
        [
            {
                "customer_id": "C001",
                "asof_date": "2025-12-31",
                "feature_version": "v1",
                "tenure_days": 130,
                "interaction_count_90d": 1,
            }
        ],
    )
    _write_jsonl(
        gold_root
        / "customer_features_asof_date"
        / "asof_date=2026-01-31"
        / "customer_features_asof_date.jsonl",
        [
            {
                "customer_id": "C001",
                "asof_date": "2026-01-31",
                "feature_version": "v1",
                "tenure_days": 160,
                "interaction_count_90d": 0,
            },
            {
                "customer_id": "C999",
                "asof_date": "2026-01-31",
                "feature_version": "v1",
                "tenure_days": 80,
                "interaction_count_90d": 3,
            },
        ],
    )

    config_path = tmp_path / "training_set.yaml"
    config_path.write_text(
        "\n".join(
            [
                "paths:",
                f"  silver_root: {silver_root}",
                f"  gold_root: {gold_root}",
                "labeling:",
                "  horizon_days: 90",
                "  cutoff_date: 2026-01-31",
                "  label_run_date: 2026-01-31",
                "features:",
                "  feature_version: v1",
                "  asof_dates:",
                "    - 2025-11-30",
                "    - 2025-12-31",
                "    - 2026-01-31",
                "    - 2026-02-28",
                "splits:",
                "  train_end_date: 2025-11-30",
                "  valid_end_date: 2025-12-31",
                "  test_end_date: 2026-01-31",
            ]
        ),
        encoding="utf-8",
    )

    config = _load_config(str(config_path))
    summary = build_training_set(config)

    assert summary.row_count == 3
    assert summary.split_counts == {"train": 1, "valid": 1, "test": 1}
    assert summary.skipped_no_label == 1
    assert summary.skipped_after_cutoff == 1
    assert summary.asof_dates_processed == 3

    output_path = (
        gold_root
        / "churn_training_dataset"
        / "cutoff_date=2026-01-31"
        / "churn_training_dataset.jsonl"
    )
    rows = _read_jsonl(output_path)
    assert len(rows) == 3
    for row in rows:
        assert row["label_horizon_days"] == 90
        assert row["split"] in {"train", "valid", "test"}


def test_build_training_set_supports_s3_paths() -> None:
    s3_client = FakeS3Client()
    config = TrainingSetConfig(
        silver_root="s3://spanishgas-data-g1/silver/",
        gold_root="s3://spanishgas-data-g1/gold/",
        horizon_days=90,
        cutoff_date=date(2026, 1, 31),
        label_run_date="2026-01-31",
        asof_dates=(date(2026, 1, 31),),
        split_rules=SplitRules(
            train_end_date=date(2025, 11, 30),
            valid_end_date=date(2025, 12, 31),
            test_end_date=date(2026, 1, 31),
        ),
        feature_version="v1",
    )

    label_payload = json.dumps(
        {
            "customer_id": "C001",
            "label_date": "2026-01-31",
            "horizon_days": 90,
            "churned_within_horizon": 1,
            "churn_effective_date": "2026-02-10",
        },
        sort_keys=True,
    )
    feature_payload = json.dumps(
        {
            "customer_id": "C001",
            "asof_date": "2026-01-31",
            "feature_version": "v1",
            "tenure_days": 100,
        },
        sort_keys=True,
    )

    s3_client.put_object(
        Bucket="spanishgas-data-g1",
        Key="silver/churn_label/run_date=2026-01-31/churn_label.jsonl",
        Body=(label_payload + "\n").encode("utf-8"),
    )
    s3_client.put_object(
        Bucket="spanishgas-data-g1",
        Key=(
            "gold/customer_features_asof_date/"
            "asof_date=2026-01-31/customer_features_asof_date.jsonl"
        ),
        Body=(feature_payload + "\n").encode("utf-8"),
    )

    summary = build_training_set(config=config, s3_client=s3_client)
    assert summary.row_count == 1
    assert summary.split_counts["test"] == 1

    output_key = (
        "spanishgas-data-g1",
        "gold/churn_training_dataset/cutoff_date=2026-01-31/churn_training_dataset.jsonl",
    )
    assert output_key in s3_client.objects

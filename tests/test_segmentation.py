import json
from pathlib import Path

from src.models.segmentation import build_segmentation


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


def _feature_rows() -> list[dict]:
    return [
        {
            "customer_id": "C001",
            "asof_date": "2026-01-31",
            "feature_version": "v1",
            "tenure_days": 120,
            "days_to_contract_end": 25,
            "price_vs_benchmark_delta": 0.09,
            "consumption_volatility_90d": 1.4,
            "interaction_count_90d": 5,
            "negative_consumption_flag": 1,
        },
        {
            "customer_id": "C002",
            "asof_date": "2026-01-31",
            "feature_version": "v1",
            "tenure_days": 110,
            "days_to_contract_end": 20,
            "price_vs_benchmark_delta": 0.07,
            "consumption_volatility_90d": 1.1,
            "interaction_count_90d": 4,
            "negative_consumption_flag": 1,
        },
        {
            "customer_id": "C003",
            "asof_date": "2026-01-31",
            "feature_version": "v1",
            "tenure_days": 240,
            "days_to_contract_end": 180,
            "price_vs_benchmark_delta": -0.03,
            "consumption_volatility_90d": 0.3,
            "interaction_count_90d": 0,
            "negative_consumption_flag": 0,
        },
        {
            "customer_id": "C004",
            "asof_date": "2026-01-31",
            "feature_version": "v1",
            "tenure_days": 260,
            "days_to_contract_end": 220,
            "price_vs_benchmark_delta": -0.04,
            "consumption_volatility_90d": 0.2,
            "interaction_count_90d": 0,
            "negative_consumption_flag": 0,
        },
        {
            "customer_id": "C005",
            "asof_date": "2026-01-31",
            "feature_version": "v1",
            "tenure_days": 180,
            "days_to_contract_end": 90,
            "price_vs_benchmark_delta": 0.02,
            "consumption_volatility_90d": 0.8,
            "interaction_count_90d": 2,
            "negative_consumption_flag": 0,
        },
        {
            "customer_id": "C006",
            "asof_date": "2026-01-31",
            "feature_version": "v1",
            "tenure_days": 190,
            "days_to_contract_end": 95,
            "price_vs_benchmark_delta": 0.01,
            "consumption_volatility_90d": 0.9,
            "interaction_count_90d": 1,
            "negative_consumption_flag": 0,
        },
    ]


def _training_rows() -> list[dict]:
    return [
        {
            "customer_id": "C001",
            "asof_date": "2026-01-31",
            "churn_label": 1,
            "split": "test",
        },
        {
            "customer_id": "C002",
            "asof_date": "2026-01-31",
            "churn_label": 1,
            "split": "test",
        },
        {
            "customer_id": "C003",
            "asof_date": "2026-01-31",
            "churn_label": 0,
            "split": "test",
        },
        {
            "customer_id": "C004",
            "asof_date": "2026-01-31",
            "churn_label": 0,
            "split": "test",
        },
        {
            "customer_id": "C005",
            "asof_date": "2026-01-31",
            "churn_label": 1,
            "split": "test",
        },
        {
            "customer_id": "C006",
            "asof_date": "2026-01-31",
            "churn_label": 0,
            "split": "test",
        },
    ]


def test_build_segmentation_local_outputs_profiles_and_report(tmp_path: Path) -> None:
    gold_root = tmp_path / "gold"
    report_path = tmp_path / "artifacts" / "reports" / "segmentation_profile.md"
    asof_date = "2026-01-31"

    _write_jsonl(
        gold_root
        / "customer_features_asof_date"
        / f"asof_date={asof_date}"
        / "customer_features_asof_date.jsonl",
        _feature_rows(),
    )
    _write_jsonl(
        gold_root
        / "churn_training_dataset"
        / f"cutoff_date={asof_date}"
        / "churn_training_dataset.jsonl",
        _training_rows(),
    )

    summary = build_segmentation(
        asof_date=asof_date,
        segment_count=3,
        gold_root=gold_root,
        report_path=report_path,
        top_driver_count=3,
        random_seed=42,
    )

    assert summary.requested_segment_count == 3
    assert summary.effective_segment_count == 3
    assert summary.churn_enriched is True
    assert summary.margin_proxy_feature == "price_vs_benchmark_delta"

    assignments_path = (
        gold_root / "segments" / f"asof_date={asof_date}" / "segments.jsonl"
    )
    profiles_path = (
        gold_root / "segments" / f"asof_date={asof_date}" / "segment_profiles.json"
    )
    assert assignments_path.exists()
    assert profiles_path.exists()
    assert report_path.exists()

    assignments = _read_jsonl(assignments_path)
    assert len(assignments) == 6
    assert len({row["segment_id"] for row in assignments}) == 3

    profiles = json.loads(profiles_path.read_text(encoding="utf-8"))["segment_profiles"]
    assert len(profiles) == 3
    assert all(profile["top_drivers"] for profile in profiles)
    assert all(
        set(profile["top_drivers"][0]) >= {"feature", "segment_mean", "global_mean", "z_score"}
        for profile in profiles
    )
    assert any(profile["churn_rate"] is not None for profile in profiles)

    report_text = report_path.read_text(encoding="utf-8")
    assert "Top Drivers by Segment" in report_text
    assert "Segment Summary" in report_text


def test_build_segmentation_supports_s3_paths() -> None:
    s3_client = FakeS3Client()
    asof_date = "2026-01-31"
    gold_root = "s3://spanishgas-data-g1/gold/"
    report_path = "s3://spanishgas-data-g1/artifacts/reports/segmentation_profile.md"

    feature_payload = (
        "\n".join(json.dumps(row, sort_keys=True) for row in _feature_rows()) + "\n"
    ).encode("utf-8")
    training_payload = (
        "\n".join(json.dumps(row, sort_keys=True) for row in _training_rows()) + "\n"
    ).encode("utf-8")
    s3_client.put_object(
        Bucket="spanishgas-data-g1",
        Key=(
            "gold/customer_features_asof_date/"
            "asof_date=2026-01-31/customer_features_asof_date.jsonl"
        ),
        Body=feature_payload,
    )
    s3_client.put_object(
        Bucket="spanishgas-data-g1",
        Key=(
            "gold/churn_training_dataset/"
            "cutoff_date=2026-01-31/churn_training_dataset.jsonl"
        ),
        Body=training_payload,
    )

    summary = build_segmentation(
        asof_date=asof_date,
        segment_count=3,
        gold_root=gold_root,
        report_path=report_path,
        top_driver_count=3,
        random_seed=42,
        s3_client=s3_client,
    )
    assert summary.effective_segment_count == 3

    assert (
        "spanishgas-data-g1",
        "gold/segments/asof_date=2026-01-31/segments.jsonl",
    ) in s3_client.objects
    assert (
        "spanishgas-data-g1",
        "gold/segments/asof_date=2026-01-31/segment_profiles.json",
    ) in s3_client.objects
    assert (
        "spanishgas-data-g1",
        "artifacts/reports/segmentation_profile.md",
    ) in s3_client.objects

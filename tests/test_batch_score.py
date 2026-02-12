import json
from pathlib import Path

from src.serving.batch_score import run_batch_scoring


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


def test_run_batch_scoring_local_writes_scoring_and_recommendations(tmp_path: Path) -> None:
    run_date = "2026-02-09"
    gold_root = tmp_path / "gold"
    artifacts_root = tmp_path / "artifacts"

    _write_jsonl(
        gold_root
        / "customer_features_asof_date"
        / "asof_date=2025-12-31"
        / "customer_features_asof_date.jsonl",
        [
            {
                "customer_id": "OLD1",
                "asof_date": "2025-12-31",
                "price_vs_benchmark_delta": 0.03,
                "interaction_count_90d": 1,
                "negative_consumption_flag": 0,
                "days_to_contract_end": 70,
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
                "price_vs_benchmark_delta": 0.08,
                "interaction_count_90d": 3,
                "negative_consumption_flag": 1,
                "days_to_contract_end": 20,
            },
            {
                "customer_id": "C002",
                "asof_date": "2026-01-31",
                "price_vs_benchmark_delta": 0.0,
                "interaction_count_90d": 0,
                "negative_consumption_flag": 0,
                "days_to_contract_end": 180,
            },
        ],
    )
    _write_jsonl(
        gold_root / "segments" / "asof_date=2026-01-31" / "segments.jsonl",
        [
            {"customer_id": "C001", "segment_id": "high_value"},
            {"customer_id": "C002", "segment_id": "stable_low_risk"},
        ],
    )

    summary = run_batch_scoring(
        run_date=run_date,
        gold_root=gold_root,
        artifacts_root=artifacts_root,
    )

    assert summary.asof_date == "2026-01-31"
    assert summary.scoring_source == "heuristic"
    assert summary.row_count == 2
    assert summary.scoring_latency_ms >= 0
    assert summary.offer_rows >= 1
    assert summary.no_offer_rows >= 0

    scoring_path = gold_root / "scoring" / f"run_date={run_date}" / "scores.jsonl"
    recommendations_path = (
        gold_root / "recommendations" / f"run_date={run_date}" / "recommendations.jsonl"
    )
    assert scoring_path.exists()
    assert recommendations_path.exists()

    scoring_rows = _read_jsonl(scoring_path)
    assert len(scoring_rows) == 2
    assert all("churn_probability" in row for row in scoring_rows)
    assert all("risk_score" in row for row in scoring_rows)

    recommendation_rows = _read_jsonl(recommendations_path)
    assert len(recommendation_rows) == 2
    assert all(row["action"] in {"offer_small", "offer_medium", "offer_large", "no_offer"} for row in recommendation_rows)
    assert all(row["reason_codes"] for row in recommendation_rows)


def test_run_batch_scoring_supports_s3_paths() -> None:
    run_date = "2026-02-09"
    asof_date = "2026-01-31"
    s3_client = FakeS3Client()

    feature_payload = (
        "\n".join(
            [
                json.dumps(
                    {
                        "customer_id": "C100",
                        "asof_date": asof_date,
                        "price_vs_benchmark_delta": 0.07,
                        "interaction_count_90d": 2,
                        "negative_consumption_flag": 1,
                        "days_to_contract_end": 25,
                    },
                    sort_keys=True,
                )
            ]
        )
        + "\n"
    ).encode("utf-8")
    segment_payload = (
        json.dumps(
            {"customer_id": "C100", "segment_id": "price_sensitive"},
            sort_keys=True,
        )
        + "\n"
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
        Key="gold/segments/asof_date=2026-01-31/segments.jsonl",
        Body=segment_payload,
    )

    summary = run_batch_scoring(
        run_date=run_date,
        asof_date=asof_date,
        gold_root="s3://spanishgas-data-g1/gold/",
        artifacts_root="s3://spanishgas-data-g1/artifacts/",
        s3_client=s3_client,
    )

    assert summary.row_count == 1
    assert (
        "spanishgas-data-g1",
        "gold/scoring/run_date=2026-02-09/scores.jsonl",
    ) in s3_client.objects
    assert (
        "spanishgas-data-g1",
        "gold/recommendations/run_date=2026-02-09/recommendations.jsonl",
    ) in s3_client.objects

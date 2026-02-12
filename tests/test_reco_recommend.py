import json
from pathlib import Path

from src.reco.recommend import build_recommendations


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


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_build_recommendations_local_generates_tier_timing_and_reason_codes(
    tmp_path: Path,
) -> None:
    run_date = "2026-02-09"
    gold_root = tmp_path / "gold"
    input_path = (
        gold_root
        / "recommendation_candidates"
        / f"run_date={run_date}"
        / "recommendation_candidates.csv"
    )
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_text(
        "customer_id,segment,risk_score,acceptance_probability,margin_eur,"
        "days_to_contract_end,shap_top_features,is_protected_customer\n"
        "C001,high_value,0.82,0.30,100,20,\"price_vs_benchmark_delta,interaction_count_90d\",false\n"
        "C002,stable_low_risk,0.30,0.25,80,75,,false\n"
        "C003,price_sensitive,0.70,0.25,0,40,,false\n"
        "C004,price_sensitive,0.90,0.35,120,25,,true\n"
        ",price_sensitive,0.95,0.35,130,15,,false\n",
        encoding="utf-8",
    )

    summary = build_recommendations(run_date=run_date, gold_root=gold_root)
    assert summary.total_rows == 5
    assert summary.written_rows == 4
    assert summary.skipped_rows == 1
    assert summary.offer_rows == 1
    assert summary.no_offer_rows == 3

    output_path = (
        gold_root / "recommendations" / f"run_date={run_date}" / "recommendations.jsonl"
    )
    assert output_path.exists()
    rows = _read_jsonl(output_path)
    by_customer = {row["customer_id"]: row for row in rows}

    offer_row = by_customer["C001"]
    assert offer_row["action"] == "offer_large"
    assert offer_row["discount_tier"] == "large"
    assert offer_row["timing_window"] == "immediate"
    assert offer_row["expected_margin_impact"] > 0
    assert "shap_price_vs_benchmark_delta" in offer_row["reason_codes"]
    assert "selected_discount_tier_large" in offer_row["reason_codes"]

    assert by_customer["C002"]["action"] == "no_offer"
    assert "below_offer_risk_threshold" in by_customer["C002"]["reason_codes"]

    assert by_customer["C003"]["action"] == "no_offer"
    assert "non_positive_margin" in by_customer["C003"]["reason_codes"]

    assert by_customer["C004"]["action"] == "no_offer"
    assert "protected_customer_guardrail" in by_customer["C004"]["reason_codes"]


def test_build_recommendations_uses_churn_probability_and_segment_id_fallbacks(
    tmp_path: Path,
) -> None:
    run_date = "2026-02-09"
    gold_root = tmp_path / "gold"
    input_path = (
        gold_root
        / "recommendation_candidates"
        / f"run_date={run_date}"
        / "recommendation_candidates.jsonl"
    )
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_text(
        json.dumps(
            {
                "customer_id": "C100",
                "churn_probability": 0.6,
                "segment_id": "segment_b",
                "margin": 50,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    summary = build_recommendations(run_date=run_date, gold_root=gold_root)
    assert summary.written_rows == 1

    output_path = (
        gold_root / "recommendations" / f"run_date={run_date}" / "recommendations.jsonl"
    )
    rows = _read_jsonl(output_path)
    assert rows[0]["customer_id"] == "C100"
    assert rows[0]["risk_score"] == 0.6
    assert rows[0]["segment"] == "segment_b"
    assert rows[0]["action"] == "offer_large"
    assert rows[0]["reason_codes"]


def test_build_recommendations_supports_s3_paths() -> None:
    s3_client = FakeS3Client()
    run_date = "2026-02-09"
    gold_root = "s3://spanishgas-data-g1/gold/"

    s3_client.put_object(
        Bucket="spanishgas-data-g1",
        Key=(
            "gold/recommendation_candidates/run_date=2026-02-09/"
            "recommendation_candidates.csv"
        ),
        Body=(
            "customer_id,segment,risk_score,acceptance_probability,margin_eur\n"
            "C900,price_sensitive,0.78,0.30,90\n"
        ).encode("utf-8"),
    )

    summary = build_recommendations(
        run_date=run_date,
        gold_root=gold_root,
        s3_client=s3_client,
    )
    assert summary.written_rows == 1
    assert (
        "spanishgas-data-g1",
        "gold/recommendations/run_date=2026-02-09/recommendations.jsonl",
    ) in s3_client.objects

    payload = s3_client.objects[
        (
            "spanishgas-data-g1",
            "gold/recommendations/run_date=2026-02-09/recommendations.jsonl",
        )
    ].decode("utf-8")
    rows = [json.loads(line) for line in payload.splitlines() if line.strip()]
    assert rows[0]["customer_id"] == "C900"
    assert rows[0]["action"].startswith("offer_")

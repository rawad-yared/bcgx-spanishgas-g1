import json
from pathlib import Path

import pytest

from src.reco.simulate import RiskBucket
from src.reco.simulate import SimulationConfig
from src.reco.simulate import run_offer_simulation


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


def test_run_offer_simulation_local_outputs_roi_by_segment_and_risk_bucket(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "recommendation_candidates.csv"
    input_path.write_text(
        "customer_id,segment,churn_probability,acceptance_probability,margin,discount_level\n"
        "C001,seg_a,0.80,0.50,100,0.10\n"
        "C002,seg_a,0.20,,80,0.05\n"
        "C003,seg_b,0.50,0.40,60,0.20\n",
        encoding="utf-8",
    )

    metrics_output = tmp_path / "offer_simulation_metrics.json"
    report_path = tmp_path / "offer_simulation.md"
    config = SimulationConfig(
        input_path=str(input_path),
        metrics_output_path=str(metrics_output),
        report_path=str(report_path),
        customer_id_col="customer_id",
        segment_col="segment",
        churn_probability_col="churn_probability",
        acceptance_probability_col="acceptance_probability",
        margin_col="margin",
        discount_level_col="discount_level",
        default_acceptance_probability=0.25,
        retention_given_acceptance=1.0,
        discount_is_fraction=True,
        risk_buckets=(
            RiskBucket(name="low", min_inclusive=0.0, max_inclusive=0.33),
            RiskBucket(name="medium", min_inclusive=0.33, max_inclusive=0.66),
            RiskBucket(name="high", min_inclusive=0.66, max_inclusive=1.0),
        ),
    )

    summary = run_offer_simulation(config=config)

    assert summary.total_rows == 3
    assert summary.simulated_rows == 3
    assert summary.skipped_rows == 0
    assert summary.overall_incremental_margin == pytest.approx(45.2)
    assert summary.overall_roi == pytest.approx(45.2 / 10.8)

    metrics = json.loads(metrics_output.read_text(encoding="utf-8"))
    bucket_metrics = {
        (row["segment"], row["risk_bucket"]): row
        for row in metrics["by_segment_risk_bucket"]
    }

    seg_a_high = bucket_metrics[("seg_a", "high")]
    assert seg_a_high["expected_retained_margin"] == pytest.approx(36.0)
    assert seg_a_high["incremental_margin"] == pytest.approx(35.0)
    assert seg_a_high["roi"] == pytest.approx(7.0)

    seg_a_low = bucket_metrics[("seg_a", "low")]
    assert seg_a_low["avg_acceptance_probability"] == pytest.approx(0.25)
    assert seg_a_low["expected_retained_margin"] == pytest.approx(3.8)
    assert seg_a_low["roi"] == pytest.approx(3.0)

    report_text = report_path.read_text(encoding="utf-8")
    assert "Offer Simulation Report" in report_text
    assert "| seg_a | high |" in report_text


def test_run_offer_simulation_supports_s3_input_and_outputs() -> None:
    s3_client = FakeS3Client()

    input_key = "gold/recommendation_candidates/run_date=2026-02-09/reco.csv"
    metrics_key = "artifacts/reports/offer_simulation_metrics.json"
    report_key = "artifacts/reports/offer_simulation.md"

    s3_client.put_object(
        Bucket="spanishgas-data-g1",
        Key=input_key,
        Body=(
            "customer_id,segment,churn_probability,acceptance_probability,margin,discount_level\n"
            "C100,seg_x,0.90,0.50,200,0.10\n"
        ).encode("utf-8"),
    )

    config = SimulationConfig(
        input_path=f"s3://spanishgas-data-g1/{input_key}",
        metrics_output_path=f"s3://spanishgas-data-g1/{metrics_key}",
        report_path=f"s3://spanishgas-data-g1/{report_key}",
        customer_id_col="customer_id",
        segment_col="segment",
        churn_probability_col="churn_probability",
        acceptance_probability_col="acceptance_probability",
        margin_col="margin",
        discount_level_col="discount_level",
        default_acceptance_probability=0.2,
        retention_given_acceptance=1.0,
        discount_is_fraction=True,
        risk_buckets=(
            RiskBucket(name="high", min_inclusive=0.66, max_inclusive=1.0),
        ),
    )

    summary = run_offer_simulation(config=config, s3_client=s3_client)

    assert summary.simulated_rows == 1
    assert ("spanishgas-data-g1", metrics_key) in s3_client.objects
    assert ("spanishgas-data-g1", report_key) in s3_client.objects

    metrics_payload = json.loads(
        s3_client.objects[("spanishgas-data-g1", metrics_key)].decode("utf-8")
    )
    assert metrics_payload["by_segment_risk_bucket"][0]["segment"] == "seg_x"
    assert metrics_payload["by_segment_risk_bucket"][0]["risk_bucket"] == "high"

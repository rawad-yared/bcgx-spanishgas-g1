import json
from pathlib import Path

from fastapi.testclient import TestClient

from src.serving.api.main import ScoringConfig
from src.serving.api.main import create_app


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(row, sort_keys=True) for row in rows]
    path.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")


def test_score_endpoint_returns_churn_segment_recommendation_and_reason_codes(
    tmp_path: Path,
) -> None:
    gold_root = tmp_path / "gold"
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
                "days_to_contract_end": 25,
                "price_vs_benchmark_delta": 0.08,
                "interaction_count_90d": 3,
                "negative_consumption_flag": 1,
            }
        ],
    )
    _write_jsonl(
        gold_root / "segments" / f"asof_date={asof_date}" / "segments.jsonl",
        [{"customer_id": "C001", "segment_id": "high_value"}],
    )

    app = create_app(
        config=ScoringConfig(
            gold_root=str(gold_root),
            model_path=str(tmp_path / "missing_model.pkl"),
            top_reason_count=3,
        )
    )
    client = TestClient(app)

    response = client.post(
        "/score",
        json={"customer_id": "C001", "asof_date": asof_date},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["customer_id"] == "C001"
    assert payload["asof_date"] == asof_date
    assert isinstance(payload["churn_score"], float)
    assert payload["segment"] == "high_value"
    assert payload["recommendation"]["action"] in {
        "offer_small",
        "offer_medium",
        "offer_large",
        "no_offer",
    }
    assert payload["recommendation"]["timing_window"]
    assert payload["reason_codes"]


def test_score_endpoint_validates_asof_date(tmp_path: Path) -> None:
    app = create_app(
        config=ScoringConfig(
            gold_root=str(tmp_path / "gold"),
            model_path=str(tmp_path / "missing_model.pkl"),
            top_reason_count=3,
        )
    )
    client = TestClient(app)

    response = client.post(
        "/score",
        json={"customer_id": "C001", "asof_date": "31-01-2026"},
    )
    assert response.status_code == 422
    assert "YYYY-MM-DD" in response.text


def test_score_endpoint_returns_404_for_missing_customer(tmp_path: Path) -> None:
    gold_root = tmp_path / "gold"
    asof_date = "2026-01-31"
    _write_jsonl(
        gold_root
        / "customer_features_asof_date"
        / f"asof_date={asof_date}"
        / "customer_features_asof_date.jsonl",
        [
            {
                "customer_id": "C999",
                "asof_date": asof_date,
                "days_to_contract_end": 200,
                "price_vs_benchmark_delta": 0.0,
                "interaction_count_90d": 0,
                "negative_consumption_flag": 0,
            }
        ],
    )

    app = create_app(
        config=ScoringConfig(
            gold_root=str(gold_root),
            model_path=str(tmp_path / "missing_model.pkl"),
            top_reason_count=3,
        )
    )
    client = TestClient(app)

    response = client.post(
        "/score",
        json={"customer_id": "C001", "asof_date": asof_date},
    )
    assert response.status_code == 404
    assert "not found" in response.text.lower()

import json
from pathlib import Path

from src.serving.ui.app import list_available_run_dates
from src.serving.ui.app import load_dashboard_rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(row, sort_keys=True) for row in rows]
    path.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")


def test_list_available_run_dates_returns_sorted_values(tmp_path: Path) -> None:
    gold_root = tmp_path / "gold"
    _write_jsonl(
        gold_root / "scoring" / "run_date=2026-02-09" / "scores.jsonl",
        [{"customer_id": "C001", "risk_score": 0.5}],
    )
    _write_jsonl(
        gold_root / "scoring" / "run_date=2026-02-07" / "scores.jsonl",
        [{"customer_id": "C001", "risk_score": 0.4}],
    )

    run_dates = list_available_run_dates(str(gold_root))
    assert run_dates == ["2026-02-07", "2026-02-09"]


def test_load_dashboard_rows_merges_scoring_and_recommendations(tmp_path: Path) -> None:
    gold_root = tmp_path / "gold"
    run_date = "2026-02-09"

    _write_jsonl(
        gold_root / "scoring" / f"run_date={run_date}" / "scores.jsonl",
        [
            {
                "customer_id": "C001",
                "asof_date": "2026-01-31",
                "churn_probability": 0.90,
                "segment": "S01",
                "reason_codes": ["high_risk_signal"],
            },
            {
                "customer_id": "C002",
                "asof_date": "2026-01-31",
                "churn_probability": 0.20,
                "segment": "S02",
                "reason_codes": ["low_risk_signal"],
            },
        ],
    )
    _write_jsonl(
        gold_root / "recommendations" / f"run_date={run_date}" / "recommendations.jsonl",
        [
            {
                "customer_id": "C001",
                "segment": "S01",
                "action": "offer_large",
                "timing_window": "immediate",
                "expected_margin_impact": 18.2,
                "reason_codes": ["selected_discount_tier_large"],
            },
            {
                "customer_id": "C002",
                "segment": "S02",
                "action": "no_offer",
                "timing_window": "60_90_days",
                "expected_margin_impact": 0.0,
                "reason_codes": ["below_offer_risk_threshold"],
            },
        ],
    )

    resolved_date, rows = load_dashboard_rows(str(gold_root))

    assert resolved_date == run_date
    assert len(rows) == 2
    assert rows[0]["customer_id"] == "C001"
    assert rows[0]["action"] == "offer_large"
    assert rows[0]["risk_tier"] == "high"
    assert rows[0]["reason_codes"] == ["selected_discount_tier_large"]
    assert rows[1]["customer_id"] == "C002"
    assert rows[1]["action"] == "no_offer"

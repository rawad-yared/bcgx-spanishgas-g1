import json
from pathlib import Path

from src.pipelines.run_pipeline import _load_config
from src.pipelines.run_pipeline import run_pipeline


def _write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [",".join(header)]
    for row in rows:
        lines.append(",".join(str(value) for value in row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _raw_fixture(raw_root: Path) -> None:
    customers = [
        ("C001", "fixed", "2026-02-15", 1),
        ("C002", "variable", "2026-10-15", 0),
        ("C003", "fixed", "2026-03-10", 1),
        ("C004", "variable", "2026-11-20", 0),
        ("C005", "fixed", "2026-02-05", 1),
        ("C006", "variable", "2026-09-30", 0),
    ]

    _write_csv(
        raw_root / "customer_attributes.csv",
        [
            "customer_id",
            "province",
            "customer_type",
            "tariff_type",
            "signup_date",
            "product_bundle",
        ],
        [
            [customer_id, "Madrid", "residential", tariff_type, "2024-01-10", "dual"]
            for customer_id, tariff_type, _, _ in customers
        ],
    )

    _write_csv(
        raw_root / "customer_contracts.csv",
        [
            "contract_id",
            "customer_id",
            "contract_start_date",
            "contract_end_date",
            "contract_status",
            "contract_term_months",
            "product_type",
        ],
        [
            [f"CT{idx+1:03d}", customer_id, "2025-01-01", contract_end, "active", 12, "gas"]
            for idx, (customer_id, _, contract_end, _) in enumerate(customers)
        ],
    )

    _write_csv(
        raw_root / "price_history.csv",
        [
            "price_date",
            "product_type",
            "tariff_type",
            "region_code",
            "price_eur_per_kwh",
            "market_benchmark_eur_per_kwh",
        ],
        [
            ["2025-10-01", "gas", "fixed", "MD", 0.220, 0.160],
            ["2025-10-01", "gas", "variable", "MD", 0.135, 0.165],
            ["2025-12-01", "gas", "fixed", "MD", 0.230, 0.165],
            ["2025-12-01", "gas", "variable", "MD", 0.140, 0.170],
            ["2026-01-15", "gas", "fixed", "MD", 0.240, 0.170],
            ["2026-01-15", "gas", "variable", "MD", 0.145, 0.175],
        ],
    )

    consumption_rows: list[list[object]] = []
    timestamps = [
        "2025-10-12T00:00:00Z",
        "2025-11-18T00:00:00Z",
        "2025-12-16T00:00:00Z",
        "2026-01-09T00:00:00Z",
    ]
    for idx, (customer_id, _tariff, _contract_end, churned) in enumerate(customers, start=1):
        if churned == 1:
            values = [9.8, -2.6, 11.1, 8.9]
        else:
            values = [5.2, 5.1, 5.0, 5.2]
        for ts, value in zip(timestamps, values):
            consumption_rows.append(
                [customer_id, ts, "gas", value, f"MTR{idx:02d}", "ami"]
            )

    _write_csv(
        raw_root / "consumption_hourly_2024.csv",
        [
            "customer_id",
            "timestamp_utc",
            "commodity",
            "consumption_kwh",
            "meter_id",
            "source_system",
        ],
        consumption_rows,
    )

    interactions: list[dict[str, object]] = []
    counter = 1
    for customer_id, _tariff, _contract_end, churned in customers:
        interaction_count = 3 if churned == 1 else 1
        for i in range(interaction_count):
            interactions.append(
                {
                    "interaction_id": f"I{counter:03d}",
                    "customer_id": customer_id,
                    "interaction_ts": f"2025-12-{10+i:02d}T09:00:00Z",
                    "channel": "call" if churned == 1 else "email",
                    "interaction_type": "complaint" if churned == 1 else "service",
                    "sentiment_score": -0.6 if churned == 1 else 0.2,
                    "resolution_status": "open" if churned == 1 else "resolved",
                    "agent_id": f"A{counter:02d}",
                }
            )
            counter += 1
    _write_json(raw_root / "customer_interactions.json", interactions)

    _write_csv(
        raw_root / "costs_by_province_month.csv",
        [
            "cost_month",
            "province",
            "commodity",
            "variable_cost_eur_per_kwh",
            "fixed_cost_eur_month",
            "network_cost_eur_per_kwh",
        ],
        [["2026-01-01", "Madrid", "gas", 0.071, 2.50, 0.010]],
    )

    labels: list[list[object]] = []
    for label_date in ("2025-11-30", "2025-12-31", "2026-01-31"):
        for customer_id, _tariff, _contract_end, churned in customers:
            labels.append(
                [
                    customer_id,
                    label_date,
                    90,
                    churned,
                    "2026-04-20" if churned == 1 else "",
                ]
            )
    _write_csv(
        raw_root / "churn_label.csv",
        [
            "customer_id",
            "label_date",
            "horizon_days",
            "churned_within_horizon",
            "churn_effective_date",
        ],
        labels,
    )


def _pipeline_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "pipeline.yaml"
    config_path.write_text(
        "\n".join(
            [
                "run:",
                "  run_date: 2026-02-09",
                "paths:",
                f"  raw_root: {tmp_path / 'raw'}",
                f"  bronze_root: {tmp_path / 'bronze'}",
                f"  silver_root: {tmp_path / 'silver'}",
                f"  gold_root: {tmp_path / 'gold'}",
                f"  artifacts_root: {tmp_path / 'artifacts'}",
                "steps:",
                "  ingestion: true",
                "  silver: true",
                "  features: true",
                "  training_set: true",
                "  train_baseline: true",
                "  train_gbdt: false",
                "  segmentation: true",
                "  score: true",
                "  recommend: true",
                "training:",
                "  horizon_days: 90",
                "  cutoff_date: 2026-01-31",
                "  label_run_date: 2026-02-09",
                "  feature_version: v1",
                "  asof_dates:",
                "    - 2025-11-30",
                "    - 2025-12-31",
                "    - 2026-01-31",
                "  split_rules:",
                "    train_end_date: 2025-11-30",
                "    valid_end_date: 2025-12-31",
                "    test_end_date: 2026-01-31",
                "baseline_model:",
                "  random_seed: 42",
                "  max_iter: 500",
                "  regularization_c: 1.0",
                "  top_k_fraction: 0.2",
                "  calibration_bins: 5",
                "segmentation:",
                "  asof_date: 2026-01-31",
                "  segment_count: 3",
                "  top_driver_count: 4",
                "  random_seed: 42",
                f"  report_path: {tmp_path / 'artifacts' / 'reports' / 'segmentation_profile.md'}",
                "scoring:",
                "  asof_date: 2026-01-31",
                "  top_reason_count: 3",
                "  default_margin_eur: 85.0",
                "  min_margin_eur: 5.0",
                "  margin_price_delta_weight: 220.0",
                "  margin_interaction_weight: 2.0",
                "  margin_negative_flag_penalty: 6.0",
                "  default_acceptance_probability: 0.25",
            ]
        ),
        encoding="utf-8",
    )
    return config_path


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_run_pipeline_full_chain_and_rerun_idempotent(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    _raw_fixture(raw_root)
    config_path = _pipeline_config(tmp_path)

    summary_first = run_pipeline(_load_config(str(config_path)))
    step_names = [step.name for step in summary_first.steps_executed]
    assert step_names == [
        "ingestion",
        "silver",
        "gold_features",
        "gold_training_set",
        "train_baseline",
        "segmentation",
        "score",
        "recommend",
    ]

    recommendations_path = (
        tmp_path
        / "gold"
        / "recommendations"
        / "run_date=2026-02-09"
        / "recommendations.jsonl"
    )
    assert recommendations_path.exists()
    recommendations_first = _read_jsonl(recommendations_path)
    assert len(recommendations_first) == 6
    assert any(row["action"].startswith("offer_") for row in recommendations_first)
    assert all("reason_codes" in row and row["reason_codes"] for row in recommendations_first)

    model_path = tmp_path / "artifacts" / "models" / "churn_baseline" / "model.pkl"
    assert model_path.exists()

    summary_second = run_pipeline(_load_config(str(config_path)))
    assert [step.name for step in summary_second.steps_executed] == step_names
    recommendations_second = _read_jsonl(recommendations_path)
    assert len(recommendations_second) == len(recommendations_first)

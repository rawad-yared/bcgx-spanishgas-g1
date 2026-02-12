from src.data.contracts import get_gold_contracts
from src.data.contracts import get_source_contracts
from src.data.contracts import load_contracts
from src.data.contracts import render_contracts


EXPECTED_SOURCE_DATASETS = {
    "customer_attributes.csv",
    "customer_contracts.csv",
    "price_history.csv",
    "consumption_hourly_2024.csv",
    "customer_interactions.json",
    "costs_by_province_month.csv",
    "churn_label.csv",
}

EXPECTED_GOLD_DATASETS = {
    "customer_snapshot_daily",
    "customer_snapshot_monthly",
    "customer_features_asof_date",
    "churn_training_dataset",
    "recommendation_candidates",
}


def test_source_contracts_cover_all_brd_datasets() -> None:
    assert set(get_source_contracts()) == EXPECTED_SOURCE_DATASETS


def test_gold_contracts_cover_architecture_tables() -> None:
    gold_contracts = get_gold_contracts()
    assert EXPECTED_GOLD_DATASETS.issubset(set(gold_contracts))
    assert len(gold_contracts) >= 3


def test_all_contracts_have_required_fields() -> None:
    for contract in load_contracts().values():
        assert contract.dataset_name
        assert contract.columns
        assert contract.primary_keys
        assert contract.time_columns
        assert contract.grain


def test_render_contracts_contains_source_and_gold() -> None:
    rendered = render_contracts()
    assert "SOURCE | customer_attributes.csv" in rendered
    assert "GOLD | customer_features_asof_date" in rendered

from src.reco.schema import POLICY_GUARDRAILS
from src.reco.schema import REQUIRED_FIELDS
from src.reco.schema import validate_recommendation


def _valid_payload() -> dict:
    return {
        "customer_id": "C001",
        "risk_score": 0.82,
        "segment": "high_risk_value",
        "action": "offer_small",
        "timing_window": "30_60_days",
        "expected_margin_impact": 12.5,
        "reason_codes": ["high_churn_risk", "contract_expiry_soon"],
    }


def test_required_fields_match_ticket_contract() -> None:
    assert set(REQUIRED_FIELDS) == {
        "customer_id",
        "risk_score",
        "segment",
        "action",
        "timing_window",
        "expected_margin_impact",
        "reason_codes",
    }


def test_policy_guardrails_include_negative_margin_rule() -> None:
    guardrail_ids = {guardrail["id"] for guardrail in POLICY_GUARDRAILS}
    assert "no_negative_margin_offer" in guardrail_ids
    assert "protected_customer_guardrail" in guardrail_ids
    assert "reason_codes_required" in guardrail_ids


def test_validate_recommendation_accepts_valid_payload() -> None:
    result = validate_recommendation(_valid_payload())
    assert result.is_valid is True
    assert result.errors == ()


def test_validate_recommendation_rejects_missing_required_field() -> None:
    payload = _valid_payload()
    payload.pop("segment")
    result = validate_recommendation(payload)
    assert result.is_valid is False
    assert "Missing required field: segment" in result.errors


def test_validate_recommendation_applies_negative_margin_guardrail() -> None:
    payload = _valid_payload()
    payload["expected_margin_impact"] = -1.0
    payload["action"] = "offer_large"
    result = validate_recommendation(payload)
    assert result.is_valid is False
    assert any("no offer allowed" in error for error in result.errors)


def test_validate_recommendation_applies_protected_customer_guardrail() -> None:
    payload = _valid_payload()
    payload["is_protected_customer"] = True
    result = validate_recommendation(payload)
    assert result.is_valid is False
    assert any("protected/regulated customers" in error for error in result.errors)


def test_validate_recommendation_requires_reason_codes() -> None:
    payload = _valid_payload()
    payload["reason_codes"] = []
    result = validate_recommendation(payload)
    assert result.is_valid is False
    assert "reason_codes must be a non-empty list." in result.errors

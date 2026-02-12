"""Recommendation output schema and policy guardrails."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from typing import Mapping


REQUIRED_FIELDS: tuple[str, ...] = (
    "customer_id",
    "risk_score",
    "segment",
    "action",
    "timing_window",
    "expected_margin_impact",
    "reason_codes",
)


NO_OFFER_ACTIONS: tuple[str, ...] = ("no_offer", "no_action", "holdout")


POLICY_GUARDRAILS: tuple[dict[str, str], ...] = (
    {
        "id": "no_negative_margin_offer",
        "description": "No offer if expected_margin_impact is negative.",
    },
    {
        "id": "protected_customer_guardrail",
        "description": "No price offer for protected or regulated customers.",
    },
    {
        "id": "reason_codes_required",
        "description": "Every recommendation must include explainable reason_codes.",
    },
)


@dataclass(frozen=True)
class ValidationResult:
    is_valid: bool
    errors: tuple[str, ...]


def _is_no_offer_action(action: str) -> bool:
    return action.strip().lower() in NO_OFFER_ACTIONS


def validate_recommendation(payload: Mapping[str, Any]) -> ValidationResult:
    """Validate recommendation payload against output contract and policy guardrails."""

    errors: list[str] = []

    for field_name in REQUIRED_FIELDS:
        if field_name not in payload:
            errors.append(f"Missing required field: {field_name}")

    if errors:
        return ValidationResult(is_valid=False, errors=tuple(errors))

    customer_id = payload.get("customer_id")
    if not isinstance(customer_id, str) or not customer_id.strip():
        errors.append("customer_id must be a non-empty string.")

    risk_score = payload.get("risk_score")
    if not isinstance(risk_score, (int, float)):
        errors.append("risk_score must be numeric.")
    else:
        risk_score_value = float(risk_score)
        if risk_score_value < 0 or risk_score_value > 1:
            errors.append("risk_score must be between 0 and 1.")

    segment = payload.get("segment")
    if not isinstance(segment, str) or not segment.strip():
        errors.append("segment must be a non-empty string.")

    action = payload.get("action")
    if not isinstance(action, str) or not action.strip():
        errors.append("action must be a non-empty string.")

    timing_window = payload.get("timing_window")
    if not isinstance(timing_window, str) or not timing_window.strip():
        errors.append("timing_window must be a non-empty string.")

    expected_margin_impact = payload.get("expected_margin_impact")
    if not isinstance(expected_margin_impact, (int, float)):
        errors.append("expected_margin_impact must be numeric.")
    else:
        margin_value = float(expected_margin_impact)
        if isinstance(action, str) and margin_value < 0 and not _is_no_offer_action(action):
            errors.append(
                "Policy violation: no offer allowed when expected_margin_impact is negative."
            )

    reason_codes = payload.get("reason_codes")
    if not isinstance(reason_codes, list) or not reason_codes:
        errors.append("reason_codes must be a non-empty list.")
    else:
        for code in reason_codes:
            if not isinstance(code, str) or not code.strip():
                errors.append("reason_codes must contain only non-empty strings.")
                break

    protected_customer = bool(payload.get("is_protected_customer")) or bool(
        payload.get("is_regulated_customer")
    )
    if protected_customer and isinstance(action, str) and not _is_no_offer_action(action):
        errors.append(
            "Policy violation: protected/regulated customers must not receive an offer action."
        )

    return ValidationResult(is_valid=not errors, errors=tuple(errors))

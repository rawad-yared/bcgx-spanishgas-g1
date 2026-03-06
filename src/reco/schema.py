"""Phase 1F: Recommendation schema with policy guardrails."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Recommendation:
    """A single retention recommendation record.

    Guardrails:
      - No negative-margin offers: if expected_margin_impact < 0, action must be "no_offer"
      - Explainability required: reason_codes must be non-empty
    """

    customer_id: str
    risk_score: float
    segment: str
    action: str
    timing_window: str
    expected_margin_impact: float
    reason_codes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.reason_codes:
            raise ValueError("reason_codes must be non-empty for every recommendation")
        if self.expected_margin_impact < 0 and self.action != "no_offer":
            raise ValueError(
                f"Negative margin impact ({self.expected_margin_impact}) "
                "requires action='no_offer'"
            )
        if not 0 <= self.risk_score <= 1:
            raise ValueError(f"risk_score must be in [0, 1], got {self.risk_score}")

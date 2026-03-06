# Recommendation Policy Specification

This document defines the output contract and policy guardrails for recommendation
generation, aligned with:

- `/Users/rawadyared/bcgx-spanishgas-g1/docs/BRD.md` (Section 6.3 Recommendation & Intervention)
- `/Users/rawadyared/bcgx-spanishgas-g1/docs/ARCHITECTURE.md` (Batch scoring outputs)

## Output contract

Each recommendation record must include:

- `customer_id` (string): customer identifier
- `risk_score` (float): churn risk score in `[0, 1]` using conformal prediction if possible from MAPIE library
- `segment` (string): interpretable segment label
- `action` (string): recommended action (for example `offer_small`, `offer_medium`, `offer_large`, `no_offer`)
- `timing_window` (string): intervention window (for example `immediate`, `30_60_days`, `60_90_days`)
- `expected_margin_impact` (float): expected incremental margin impact from the action
- `reason_codes` (list[string]): explainability reason codes for auditability

## Guardrails

The following policy guardrails are mandatory:

1. No negative-margin offers
- If `expected_margin_impact < 0`, action must be a no-offer action.

2. Explainability required
- `reason_codes` must be present and non-empty for every recommendation.

## Notes

- This policy specification is intentionally strict and deterministic to support
  traceability, explainability, and auditability.
- The schema implementation for these constraints is defined in:
  `/Users/rawadyared/bcgx-spanishgas-g1/src/reco/schema.py`.

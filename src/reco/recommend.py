"""Deterministic recommendation engine v1 (rules + optimization)."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Iterable
from typing import Protocol

from src.reco.schema import validate_recommendation


class S3ClientProtocol(Protocol):
    """Minimal S3 client protocol used by this module."""

    def get_object(self, **kwargs: Any) -> dict[str, Any]: ...

    def put_object(self, **kwargs: Any) -> dict[str, Any]: ...


@dataclass(frozen=True)
class TierSpec:
    name: str
    discount_fraction: float
    acceptance_uplift: float
    min_risk_score: float


@dataclass(frozen=True)
class RecommendSummary:
    run_date: str
    input_path: str
    output_path: str
    total_rows: int
    written_rows: int
    skipped_rows: int
    offer_rows: int
    no_offer_rows: int


DEFAULT_BASE_ACCEPTANCE = 0.25
DEFAULT_MIN_OFFER_RISK = 0.35
DEFAULT_TIERS: tuple[TierSpec, ...] = (
    TierSpec(name="small", discount_fraction=0.05, acceptance_uplift=0.00, min_risk_score=0.35),
    TierSpec(
        name="medium",
        discount_fraction=0.10,
        acceptance_uplift=0.05,
        min_risk_score=0.45,
    ),
    TierSpec(name="large", discount_fraction=0.15, acceptance_uplift=0.10, min_risk_score=0.60),
)
NO_OFFER_ACTION = "no_offer"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate deterministic recommendation actions for a run date."
    )
    parser.add_argument(
        "--run-date",
        required=True,
        help="Run date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--gold-root",
        default=os.environ.get("SPANISHGAS_GOLD_ROOT", "data/gold/"),
        help="Gold root location (local path or s3:// URI).",
    )
    parser.add_argument(
        "--input-path",
        default=None,
        help="Optional explicit input path for recommendation candidates (csv/jsonl).",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Optional explicit output path for recommendations (jsonl).",
    )
    return parser.parse_args(argv)


def _validate_iso_date(value: str) -> str:
    year, month, day = value.split("-")
    if len(year) != 4 or len(month) != 2 or len(day) != 2:
        raise ValueError(f"Invalid run-date format: {value}")
    return value


def _is_s3_uri(path: str) -> bool:
    return path.startswith("s3://")


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    if not _is_s3_uri(uri):
        raise ValueError(f"Not an S3 URI: {uri}")
    bucket_and_key = uri[len("s3://") :]
    bucket, _, key = bucket_and_key.partition("/")
    if not bucket:
        raise ValueError(f"Invalid S3 URI, missing bucket: {uri}")
    return bucket, key.lstrip("/")


def _join_location(root: str, relative_path: str) -> str:
    if _is_s3_uri(root):
        return f"{root.rstrip('/')}/{relative_path.lstrip('/')}"
    return str(Path(root) / relative_path)


def _resolve_s3_client(
    paths: Iterable[str], s3_client: S3ClientProtocol | None = None
) -> S3ClientProtocol | None:
    if s3_client is not None:
        return s3_client
    if not any(_is_s3_uri(path) for path in paths):
        return None

    try:
        import boto3

        return boto3.client("s3")
    except ImportError:
        pass

    try:
        from botocore.session import Session

        return Session().create_client("s3")
    except Exception as exc:
        raise RuntimeError(
            "S3 recommendation I/O requested but no S3 SDK is available. "
            "Install boto3 or use local paths."
        ) from exc


def _read_bytes(path: str, s3_client: S3ClientProtocol | None) -> bytes:
    if _is_s3_uri(path):
        if s3_client is None:
            raise RuntimeError("S3 path used without an S3 client.")
        bucket, key = _parse_s3_uri(path)
        try:
            response = s3_client.get_object(Bucket=bucket, Key=key)
        except Exception as exc:
            raise FileNotFoundError(f"Missing input file: {path}") from exc
        return response["Body"].read()

    local_path = Path(path)
    if not local_path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    return local_path.read_bytes()


def _write_bytes(path: str, payload: bytes, s3_client: S3ClientProtocol | None) -> None:
    if _is_s3_uri(path):
        if s3_client is None:
            raise RuntimeError("S3 path used without an S3 client.")
        bucket, key = _parse_s3_uri(path)
        s3_client.put_object(Bucket=bucket, Key=key, Body=payload)
        return

    local_path = Path(path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_bytes(payload)


def _read_records(path: str, s3_client: S3ClientProtocol | None) -> list[dict[str, Any]]:
    payload = _read_bytes(path, s3_client).decode("utf-8")
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        return [dict(row) for row in csv.DictReader(io.StringIO(payload))]

    rows: list[dict[str, Any]] = []
    for line in payload.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        decoded = json.loads(stripped)
        if isinstance(decoded, dict):
            rows.append(decoded)
    return rows


def _to_jsonl(rows: Iterable[dict[str, Any]]) -> bytes:
    lines = [json.dumps(row, sort_keys=True) for row in rows]
    content = "\n".join(lines)
    if content:
        content += "\n"
    return content.encode("utf-8")


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return value != 0
    text = str(value).strip().lower()
    return text in {"1", "true", "t", "yes", "y"}


def _clip_probability(value: float) -> float:
    return min(1.0, max(0.0, value))


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return cleaned or "unknown"


def _normalize_segment(row: dict[str, Any]) -> str:
    value = row.get("segment")
    if value is None:
        value = row.get("segment_id")
    text = str(value or "").strip()
    return text or "unknown"


def _candidate_input_paths(gold_root: str, run_date: str) -> tuple[str, ...]:
    base = f"recommendation_candidates/run_date={run_date}"
    return (
        _join_location(gold_root, f"{base}/recommendation_candidates.csv"),
        _join_location(gold_root, f"{base}/recommendation_candidates.jsonl"),
    )


def _parse_reason_field(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    return [part.strip() for part in re.split(r"[;,|]", text) if part.strip()]


def _extract_model_reason_codes(row: dict[str, Any]) -> list[str]:
    raw_values: list[str] = []
    shap_fields = ("shap_reason_codes", "shap_top_features", "top_shap_features")
    generic_fields = ("reason_codes",)

    for field in shap_fields:
        for value in _parse_reason_field(row.get(field)):
            raw_values.append(f"shap_{_slug(value)}")

    for field in generic_fields:
        for value in _parse_reason_field(row.get(field)):
            raw_values.append(_slug(value))

    deduped: list[str] = []
    for value in raw_values:
        if value and value not in deduped:
            deduped.append(value)
    return deduped[:3]


def _segment_acceptance_adjustment(segment: str) -> float:
    lowered = segment.lower()
    adjustment = 0.0
    if "price" in lowered:
        adjustment += 0.08
    if "high_value" in lowered or "vip" in lowered:
        adjustment += 0.03
    if "stable" in lowered or "loyal" in lowered:
        adjustment -= 0.04
    if "low_risk" in lowered:
        adjustment -= 0.05
    return adjustment


def _compute_incremental_margin(
    risk_score: float,
    margin_eur: float,
    base_acceptance: float,
    segment_adjustment: float,
    tier: TierSpec,
) -> tuple[float, float]:
    acceptance_probability = _clip_probability(
        base_acceptance + segment_adjustment + tier.acceptance_uplift
    )
    discount_amount = margin_eur * tier.discount_fraction
    margin_after_discount = max(0.0, margin_eur - discount_amount)
    expected_retained_margin = risk_score * acceptance_probability * margin_after_discount
    expected_non_churn_discount_cost = (
        (1.0 - risk_score) * acceptance_probability * discount_amount
    )
    incremental_margin = expected_retained_margin - expected_non_churn_discount_cost
    return incremental_margin, acceptance_probability


def _timing_window(risk_score: float, days_to_contract_end: float | None) -> str:
    if days_to_contract_end is not None and days_to_contract_end >= 0:
        if days_to_contract_end <= 30:
            return "immediate"
        if days_to_contract_end <= 60:
            return "30_60_days"
        if days_to_contract_end <= 90:
            return "60_90_days"

    if risk_score >= 0.80:
        return "immediate"
    if risk_score >= 0.55:
        return "30_60_days"
    return "60_90_days"


def _build_reason_codes(
    model_reason_codes: list[str],
    risk_score: float,
    segment: str,
    margin_eur: float,
    action: str,
    selected_tier_name: str | None,
    protected_customer: bool,
    threshold_blocked: bool,
    non_positive_impact_blocked: bool,
) -> list[str]:
    reasons = list(model_reason_codes)

    if risk_score >= 0.75:
        reasons.append("high_churn_risk")
    elif risk_score >= 0.50:
        reasons.append("elevated_churn_risk")
    else:
        reasons.append("lower_churn_risk")

    reasons.append(f"segment_{_slug(segment)}")

    if margin_eur <= 0:
        reasons.append("non_positive_margin")
    else:
        reasons.append("positive_margin")

    if protected_customer:
        reasons.append("protected_customer_guardrail")
    if threshold_blocked:
        reasons.append("below_offer_risk_threshold")
    if non_positive_impact_blocked:
        reasons.append("non_positive_expected_impact")
    if action == NO_OFFER_ACTION:
        reasons.append("rule_based_no_offer")
    else:
        reasons.append(f"selected_discount_tier_{selected_tier_name or 'unknown'}")
        reasons.append("expected_margin_positive")

    deduped: list[str] = []
    for reason in reasons:
        normalized = _slug(reason)
        if normalized and normalized not in deduped:
            deduped.append(normalized)

    if not deduped:
        return ["rule_based_reasoning"]
    return deduped


def _recommend_record(row: dict[str, Any], run_date: str) -> dict[str, Any] | None:
    customer_id = str(row.get("customer_id") or "").strip()
    if not customer_id:
        return None

    risk_score = _to_float(row.get("risk_score"))
    if risk_score is None:
        risk_score = _to_float(row.get("churn_probability"))
    if risk_score is None:
        return None
    risk_score = _clip_probability(risk_score)

    segment = _normalize_segment(row)
    margin_value = _to_float(row.get("margin_eur"))
    if margin_value is None:
        margin_value = _to_float(row.get("margin"))
    margin_eur = 0.0 if margin_value is None else margin_value
    margin_eur = max(0.0, margin_eur)

    base_acceptance = _to_float(row.get("acceptance_probability"))
    if base_acceptance is None:
        base_acceptance = DEFAULT_BASE_ACCEPTANCE
    base_acceptance = _clip_probability(base_acceptance)

    protected_customer = _to_bool(row.get("is_protected_customer")) or _to_bool(
        row.get("is_regulated_customer")
    )
    days_to_contract_end = _to_float(row.get("days_to_contract_end"))

    model_reason_codes = _extract_model_reason_codes(row)
    timing_window = _timing_window(risk_score, days_to_contract_end)

    action = NO_OFFER_ACTION
    selected_tier_name: str | None = None
    selected_discount_fraction = 0.0
    expected_margin_impact = 0.0
    threshold_blocked = False
    non_positive_impact_blocked = False

    if not protected_customer and margin_eur > 0.0:
        if risk_score < DEFAULT_MIN_OFFER_RISK:
            threshold_blocked = True
        else:
            segment_adjustment = _segment_acceptance_adjustment(segment)
            candidates: list[tuple[float, TierSpec]] = []
            for tier in DEFAULT_TIERS:
                if risk_score < tier.min_risk_score:
                    continue
                impact, _acceptance = _compute_incremental_margin(
                    risk_score=risk_score,
                    margin_eur=margin_eur,
                    base_acceptance=base_acceptance,
                    segment_adjustment=segment_adjustment,
                    tier=tier,
                )
                candidates.append((impact, tier))

            if candidates:
                best_impact, best_tier = max(
                    candidates, key=lambda item: (item[0], -item[1].discount_fraction)
                )
                if best_impact > 0:
                    action = f"offer_{best_tier.name}"
                    selected_tier_name = best_tier.name
                    selected_discount_fraction = best_tier.discount_fraction
                    expected_margin_impact = best_impact
                else:
                    non_positive_impact_blocked = True
            else:
                threshold_blocked = True

    reason_codes = _build_reason_codes(
        model_reason_codes=model_reason_codes,
        risk_score=risk_score,
        segment=segment,
        margin_eur=margin_eur,
        action=action,
        selected_tier_name=selected_tier_name,
        protected_customer=protected_customer,
        threshold_blocked=threshold_blocked,
        non_positive_impact_blocked=non_positive_impact_blocked,
    )

    recommendation = {
        "customer_id": customer_id,
        "run_date": run_date,
        "risk_score": round(risk_score, 6),
        "segment": segment,
        "action": action,
        "timing_window": timing_window,
        "expected_margin_impact": round(expected_margin_impact, 6),
        "reason_codes": reason_codes,
        "discount_tier": selected_tier_name or "none",
        "discount_fraction": round(selected_discount_fraction, 4),
        "margin_eur": round(margin_eur, 6),
        "is_protected_customer": protected_customer,
        "decision_policy_version": "reco_v1_rules_optimization",
    }

    validation = validate_recommendation(recommendation)
    if not validation.is_valid:
        raise ValueError(
            f"Generated invalid recommendation for customer_id={customer_id}: {validation.errors}"
        )
    return recommendation


def build_recommendations(
    run_date: str,
    gold_root: str | Path = "data/gold/",
    input_path: str | None = None,
    output_path: str | None = None,
    s3_client: S3ClientProtocol | None = None,
) -> RecommendSummary:
    """Build recommendation actions for the requested run date."""

    run_date = _validate_iso_date(str(run_date))
    gold_root_str = str(gold_root)

    if output_path is None:
        output_path = _join_location(
            gold_root_str,
            f"recommendations/run_date={run_date}/recommendations.jsonl",
        )

    input_candidates = (input_path,) if input_path else _candidate_input_paths(gold_root_str, run_date)
    resolved_s3_client = _resolve_s3_client(
        paths=[*input_candidates, output_path],
        s3_client=s3_client,
    )

    selected_input_path: str | None = None
    records: list[dict[str, Any]] = []
    last_error: Exception | None = None
    for candidate in input_candidates:
        if candidate is None:
            continue
        try:
            records = _read_records(candidate, resolved_s3_client)
            selected_input_path = candidate
            break
        except FileNotFoundError as exc:
            last_error = exc
            continue

    if selected_input_path is None:
        if last_error is not None:
            raise last_error
        raise FileNotFoundError("No input recommendation candidate file found.")

    recommendations: list[dict[str, Any]] = []
    skipped_rows = 0
    offer_rows = 0
    no_offer_rows = 0
    for row in records:
        recommendation = _recommend_record(row=row, run_date=run_date)
        if recommendation is None:
            skipped_rows += 1
            continue
        recommendations.append(recommendation)
        if recommendation["action"] == NO_OFFER_ACTION:
            no_offer_rows += 1
        else:
            offer_rows += 1

    _write_bytes(output_path, _to_jsonl(recommendations), resolved_s3_client)

    return RecommendSummary(
        run_date=run_date,
        input_path=selected_input_path,
        output_path=output_path,
        total_rows=len(records),
        written_rows=len(recommendations),
        skipped_rows=skipped_rows,
        offer_rows=offer_rows,
        no_offer_rows=no_offer_rows,
    )


def _render_summary(summary: RecommendSummary) -> str:
    return "\n".join(
        [
            "Recommendation generation complete:",
            f"- run_date={summary.run_date}",
            f"- input={summary.input_path}",
            f"- output={summary.output_path}",
            f"- total_rows={summary.total_rows}",
            f"- written_rows={summary.written_rows}",
            f"- skipped_rows={summary.skipped_rows}",
            f"- offer_rows={summary.offer_rows}",
            f"- no_offer_rows={summary.no_offer_rows}",
        ]
    )


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    summary = build_recommendations(
        run_date=args.run_date,
        gold_root=args.gold_root,
        input_path=args.input_path,
        output_path=args.output_path,
    )
    print(_render_summary(summary))


if __name__ == "__main__":
    main()

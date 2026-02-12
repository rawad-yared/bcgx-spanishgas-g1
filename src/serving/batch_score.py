"""Batch scoring job for churn probabilities and recommendation outputs."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from dataclasses import dataclass
from datetime import date
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import Iterable
from typing import Protocol

from src.reco.recommend import build_recommendations


class S3ClientProtocol(Protocol):
    """Minimal S3 client protocol used by this module."""

    def get_object(self, **kwargs: Any) -> dict[str, Any]: ...

    def put_object(self, **kwargs: Any) -> dict[str, Any]: ...


@dataclass(frozen=True)
class BatchScoreSummary:
    run_date: str
    asof_date: str
    scoring_source: str
    row_count: int
    scoring_output_path: str
    candidates_output_path: str
    recommendations_output_path: str
    offer_rows: int
    no_offer_rows: int
    skipped_rows: int
    scoring_latency_ms: float


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run batch scoring and write scoring + recommendation outputs."
    )
    parser.add_argument(
        "--run-date",
        required=True,
        help="Run date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--asof-date",
        default=None,
        help="Optional as-of date in YYYY-MM-DD. If omitted, latest available <= run-date is used for local roots.",
    )
    parser.add_argument(
        "--gold-root",
        default=os.environ.get("SPANISHGAS_GOLD_ROOT", "data/gold/"),
        help="Gold root location (local path or s3:// URI).",
    )
    parser.add_argument(
        "--artifacts-root",
        default=os.environ.get("SPANISHGAS_ARTIFACTS_ROOT", "artifacts/"),
        help="Artifacts root containing model outputs (local path or s3:// URI).",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Optional explicit model path. Defaults to <artifacts_root>/models/churn_baseline/model.pkl.",
    )
    parser.add_argument(
        "--top-reason-count",
        type=int,
        default=3,
        help="Top reason count to keep for scoring reason codes.",
    )
    parser.add_argument(
        "--scoring-output-path",
        default=None,
        help="Optional explicit output path for scoring table JSONL.",
    )
    parser.add_argument(
        "--candidates-output-path",
        default=None,
        help="Optional explicit output path for recommendation_candidates JSONL.",
    )
    parser.add_argument(
        "--recommendations-output-path",
        default=None,
        help="Optional explicit output path for recommendations JSONL.",
    )
    return parser.parse_args(argv)


def _parse_iso_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


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
            "S3 batch scoring requested but no S3 SDK is available. "
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


def _read_jsonl(path: str, s3_client: S3ClientProtocol | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in _read_bytes(path, s3_client).decode("utf-8").splitlines():
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


def _clip_probability(value: float) -> float:
    return min(1.0, max(0.0, value))


def _slug(value: str) -> str:
    lowered = value.lower()
    normalized = []
    for char in lowered:
        if char.isalnum():
            normalized.append(char)
        else:
            normalized.append("_")
    result = "".join(normalized).strip("_")
    while "__" in result:
        result = result.replace("__", "_")
    return result or "unknown"


def _feature_path(gold_root: str, asof_date: str) -> str:
    return _join_location(
        gold_root,
        f"customer_features_asof_date/asof_date={asof_date}/customer_features_asof_date.jsonl",
    )


def _resolve_asof_date(
    run_date: str,
    requested_asof: str | None,
    gold_root: str,
) -> str:
    if requested_asof:
        _parse_iso_date(requested_asof)
        return requested_asof

    _parse_iso_date(run_date)
    default_candidate = run_date
    if _is_s3_uri(gold_root):
        return default_candidate

    default_path = Path(_feature_path(gold_root, default_candidate))
    if default_path.exists():
        return default_candidate

    root = Path(gold_root) / "customer_features_asof_date"
    if not root.exists():
        return default_candidate

    run_dt = _parse_iso_date(run_date)
    available: list[date] = []
    for child in root.iterdir():
        if not child.is_dir() or not child.name.startswith("asof_date="):
            continue
        value = child.name.split("=", 1)[1]
        try:
            parsed = _parse_iso_date(value)
        except ValueError:
            continue
        if parsed <= run_dt:
            available.append(parsed)
    if not available:
        return default_candidate
    return max(available).isoformat()


def _load_optional_segment_map(
    gold_root: str,
    asof_date: str,
    s3_client: S3ClientProtocol | None,
) -> dict[str, str]:
    segment_path = _join_location(gold_root, f"segments/asof_date={asof_date}/segments.jsonl")
    try:
        rows = _read_jsonl(segment_path, s3_client)
    except FileNotFoundError:
        return {}

    segment_map: dict[str, str] = {}
    for row in rows:
        customer_id = str(row.get("customer_id") or "").strip()
        segment = str(row.get("segment_id") or "unknown").strip() or "unknown"
        if customer_id:
            segment_map[customer_id] = segment
    return segment_map


def _load_optional_model(
    model_path: str,
    s3_client: S3ClientProtocol | None,
) -> dict[str, Any] | None:
    try:
        payload = _read_bytes(model_path, s3_client)
    except FileNotFoundError:
        return None
    decoded = pickle.loads(payload)
    if not isinstance(decoded, dict):
        return None
    if "model" not in decoded or "vectorizer" not in decoded:
        return None
    return decoded


def _normalize_for_model(
    row: dict[str, Any],
    feature_columns: list[str],
    numeric_columns: set[str],
) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for column in feature_columns:
        value = row.get(column)
        if column in numeric_columns:
            numeric = _to_float(value)
            normalized[column] = 0.0 if numeric is None else numeric
        else:
            normalized[column] = "__missing__" if value is None else str(value)
    return normalized


def _top_reason_features(
    matrix_row: Any,
    model: Any,
    feature_names: list[str],
    top_reason_count: int,
) -> list[str]:
    if top_reason_count <= 0:
        return []
    if not hasattr(model, "coef_"):
        return []

    coefficients = getattr(model, "coef_", None)
    if coefficients is None or len(coefficients) == 0:
        return []
    coef_vector = coefficients[0]
    if len(coef_vector) != len(feature_names):
        return []

    contributions: list[tuple[float, str]] = []
    for idx, feature_name in enumerate(feature_names):
        contribution = abs(float(matrix_row[idx]) * float(coef_vector[idx]))
        if contribution <= 0:
            continue
        contributions.append((contribution, feature_name))
    contributions.sort(reverse=True, key=lambda item: item[0])
    return [_slug(item[1]) for item in contributions[:top_reason_count]]


def _heuristic_score(row: dict[str, Any]) -> tuple[float, list[str]]:
    price_delta = _to_float(row.get("price_vs_benchmark_delta")) or 0.0
    interaction_count = _to_float(row.get("interaction_count_90d")) or 0.0
    negative_flag = _to_float(row.get("negative_consumption_flag")) or 0.0
    days_to_contract_end = _to_float(row.get("days_to_contract_end"))
    if days_to_contract_end is None:
        days_to_contract_end = 120.0
    days_to_contract_end = max(0.0, days_to_contract_end)

    score = 0.15
    score += max(0.0, price_delta) * 2.2
    score += min(1.0, interaction_count / 10.0) * 0.35
    score += min(1.0, negative_flag) * 0.15
    score += (1.0 - min(days_to_contract_end, 180.0) / 180.0) * 0.20
    score = _clip_probability(score)

    reasons: list[str] = []
    if price_delta > 0.02:
        reasons.append("high_price_delta")
    if interaction_count >= 2:
        reasons.append("frequent_interactions")
    if negative_flag >= 1:
        reasons.append("negative_consumption_flag")
    if days_to_contract_end <= 60:
        reasons.append("contract_end_soon")
    if not reasons:
        reasons.append("baseline_risk_factors")
    return score, reasons


def _estimated_margin_eur(row: dict[str, Any]) -> float:
    price_delta = _to_float(row.get("price_vs_benchmark_delta")) or 0.0
    interaction_count = _to_float(row.get("interaction_count_90d")) or 0.0
    negative_flag = _to_float(row.get("negative_consumption_flag")) or 0.0
    margin = 85.0
    margin -= max(0.0, price_delta) * 220.0
    margin -= interaction_count * 2.0
    margin -= negative_flag * 6.0
    return max(5.0, margin)


def _segment_acceptance_probability(segment: str) -> float:
    probability = 0.25
    lowered = segment.lower()
    if "price" in lowered:
        probability += 0.06
    if "high_value" in lowered:
        probability += 0.03
    if "stable" in lowered:
        probability -= 0.04
    return _clip_probability(probability)


def run_batch_scoring(
    run_date: str,
    asof_date: str | None = None,
    gold_root: str | Path = "data/gold/",
    artifacts_root: str | Path = "artifacts/",
    model_path: str | None = None,
    top_reason_count: int = 3,
    scoring_output_path: str | None = None,
    candidates_output_path: str | None = None,
    recommendations_output_path: str | None = None,
    s3_client: S3ClientProtocol | None = None,
) -> BatchScoreSummary:
    """Run batch scoring and produce scoring + recommendations outputs."""

    run_date = _parse_iso_date(str(run_date)).isoformat()
    resolved_asof = _resolve_asof_date(
        run_date=run_date,
        requested_asof=asof_date,
        gold_root=str(gold_root),
    )
    _parse_iso_date(resolved_asof)

    gold_root_str = str(gold_root)
    artifacts_root_str = str(artifacts_root)
    resolved_model_path = model_path or _join_location(
        artifacts_root_str, "models/churn_baseline/model.pkl"
    )
    scoring_output = scoring_output_path or _join_location(
        gold_root_str, f"scoring/run_date={run_date}/scores.jsonl"
    )
    candidates_output = candidates_output_path or _join_location(
        gold_root_str,
        f"recommendation_candidates/run_date={run_date}/recommendation_candidates.jsonl",
    )
    recommendations_output = recommendations_output_path or _join_location(
        gold_root_str, f"recommendations/run_date={run_date}/recommendations.jsonl"
    )

    feature_input_path = _feature_path(gold_root_str, resolved_asof)
    resolved_s3_client = _resolve_s3_client(
        paths=[
            feature_input_path,
            resolved_model_path,
            scoring_output,
            candidates_output,
            recommendations_output,
            _join_location(gold_root_str, f"segments/asof_date={resolved_asof}/segments.jsonl"),
        ],
        s3_client=s3_client,
    )

    feature_rows = _read_jsonl(feature_input_path, resolved_s3_client)
    segment_map = _load_optional_segment_map(
        gold_root=gold_root_str,
        asof_date=resolved_asof,
        s3_client=resolved_s3_client,
    )
    model_payload = _load_optional_model(resolved_model_path, resolved_s3_client)
    use_model = model_payload is not None

    model = None
    vectorizer = None
    feature_columns: list[str] = []
    numeric_columns: set[str] = set()
    feature_names: list[str] = []
    if use_model:
        model = model_payload["model"]
        vectorizer = model_payload["vectorizer"]
        feature_columns = list(model_payload.get("feature_columns", []))
        numeric_columns = set(model_payload.get("numeric_columns", []))
        if hasattr(vectorizer, "get_feature_names_out"):
            feature_names = list(vectorizer.get_feature_names_out())

    scoring_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []

    started = time.perf_counter()
    for row in sorted(feature_rows, key=lambda item: str(item.get("customer_id") or "")):
        customer_id = str(row.get("customer_id") or "").strip()
        if not customer_id:
            continue

        if use_model and model is not None and vectorizer is not None and feature_columns:
            normalized = _normalize_for_model(row, feature_columns, numeric_columns)
            matrix = vectorizer.transform([normalized])
            risk_score = _clip_probability(float(model.predict_proba(matrix)[0][1]))
            reason_codes = _top_reason_features(
                matrix_row=matrix[0],
                model=model,
                feature_names=feature_names,
                top_reason_count=top_reason_count,
            )
            if not reason_codes:
                reason_codes = ["model_based_risk_score"]
            scoring_source = "baseline_model"
        else:
            risk_score, reason_codes = _heuristic_score(row)
            scoring_source = "heuristic"

        segment = segment_map.get(customer_id, "unknown")
        margin_eur = _estimated_margin_eur(row)
        acceptance_probability = _segment_acceptance_probability(segment)

        scoring_rows.append(
            {
                "customer_id": customer_id,
                "run_date": run_date,
                "asof_date": resolved_asof,
                "churn_probability": round(risk_score, 6),
                "risk_score": round(risk_score, 6),
                "segment": segment,
                "reason_codes": reason_codes,
                "scoring_source": scoring_source,
            }
        )

        candidate_rows.append(
            {
                "customer_id": customer_id,
                "run_date": run_date,
                "risk_score": round(risk_score, 6),
                "segment": segment,
                "margin_eur": round(margin_eur, 6),
                "acceptance_probability": round(acceptance_probability, 6),
                "days_to_contract_end": row.get("days_to_contract_end"),
                "shap_top_features": reason_codes,
            }
        )
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    _write_bytes(scoring_output, _to_jsonl(scoring_rows), resolved_s3_client)
    _write_bytes(candidates_output, _to_jsonl(candidate_rows), resolved_s3_client)

    recommendation_summary = build_recommendations(
        run_date=run_date,
        gold_root=gold_root_str,
        input_path=candidates_output,
        output_path=recommendations_output,
        s3_client=resolved_s3_client,
    )

    return BatchScoreSummary(
        run_date=run_date,
        asof_date=resolved_asof,
        scoring_source="baseline_model" if use_model else "heuristic",
        row_count=len(scoring_rows),
        scoring_output_path=scoring_output,
        candidates_output_path=candidates_output,
        recommendations_output_path=recommendations_output,
        offer_rows=recommendation_summary.offer_rows,
        no_offer_rows=recommendation_summary.no_offer_rows,
        skipped_rows=recommendation_summary.skipped_rows,
        scoring_latency_ms=elapsed_ms,
    )


def _render_summary(summary: BatchScoreSummary) -> str:
    return "\n".join(
        [
            "Batch scoring complete:",
            f"- run_date={summary.run_date}",
            f"- asof_date={summary.asof_date}",
            f"- scoring_source={summary.scoring_source}",
            f"- scored_rows={summary.row_count}",
            f"- scoring_latency_ms={summary.scoring_latency_ms:.3f}",
            f"- scoring_output={summary.scoring_output_path}",
            f"- recommendation_candidates_output={summary.candidates_output_path}",
            f"- recommendations_output={summary.recommendations_output_path}",
            f"- offer_rows={summary.offer_rows}",
            f"- no_offer_rows={summary.no_offer_rows}",
            f"- skipped_rows={summary.skipped_rows}",
        ]
    )


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    summary = run_batch_scoring(
        run_date=args.run_date,
        asof_date=args.asof_date,
        gold_root=args.gold_root,
        artifacts_root=args.artifacts_root,
        model_path=args.model_path,
        top_reason_count=args.top_reason_count,
        scoring_output_path=args.scoring_output_path,
        candidates_output_path=args.candidates_output_path,
        recommendations_output_path=args.recommendations_output_path,
    )
    print(_render_summary(summary))


if __name__ == "__main__":
    main()

"""FastAPI real-time scoring endpoint."""

from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import Iterable
from typing import Protocol

from fastapi import FastAPI
from fastapi import HTTPException
from pydantic import BaseModel
from pydantic import Field

from src.reco.recommend import _recommend_record


class S3ClientProtocol(Protocol):
    """Minimal S3 client protocol used by this module."""

    def get_object(self, **kwargs: Any) -> dict[str, Any]: ...


@dataclass(frozen=True)
class ScoringConfig:
    gold_root: str
    model_path: str
    top_reason_count: int


class ScoreRequest(BaseModel):
    customer_id: str = Field(..., min_length=1)
    asof_date: str


class RecommendationPayload(BaseModel):
    action: str
    timing_window: str
    expected_margin_impact: float


class ScoreResponse(BaseModel):
    customer_id: str
    asof_date: str
    churn_score: float
    segment: str
    recommendation: RecommendationPayload
    reason_codes: list[str]


def _parse_iso_date(value: str) -> str:
    return datetime.strptime(value, "%Y-%m-%d").date().isoformat()


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
            "S3 API scoring requested but no S3 SDK is available. "
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
    compact = "".join(normalized).strip("_")
    while "__" in compact:
        compact = compact.replace("__", "_")
    return compact or "unknown"


def _normalize_for_model(
    row: dict[str, Any], feature_columns: list[str], numeric_columns: set[str]
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
    return [f"shap_{_slug(item[1])}" for item in contributions[:top_reason_count]]


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


def _read_customer_feature(
    gold_root: str,
    customer_id: str,
    asof_date: str,
    s3_client: S3ClientProtocol | None,
) -> dict[str, Any] | None:
    feature_path = _join_location(
        gold_root,
        f"customer_features_asof_date/asof_date={asof_date}/customer_features_asof_date.jsonl",
    )
    rows = _read_jsonl(feature_path, s3_client)
    for row in rows:
        if str(row.get("customer_id") or "").strip() == customer_id:
            return row
    return None


def _load_optional_segment(
    gold_root: str,
    customer_id: str,
    asof_date: str,
    s3_client: S3ClientProtocol | None,
) -> str:
    segment_path = _join_location(gold_root, f"segments/asof_date={asof_date}/segments.jsonl")
    try:
        rows = _read_jsonl(segment_path, s3_client)
    except FileNotFoundError:
        return "unknown"
    for row in rows:
        if str(row.get("customer_id") or "").strip() == customer_id:
            return str(row.get("segment_id") or "unknown").strip() or "unknown"
    return "unknown"


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


def _default_config() -> ScoringConfig:
    gold_root = os.environ.get("SPANISHGAS_GOLD_ROOT", "data/gold/")
    model_path = os.environ.get(
        "SPANISHGAS_MODEL_PATH", "artifacts/models/churn_baseline/model.pkl"
    )
    top_reason_count = int(os.environ.get("SPANISHGAS_TOP_REASON_COUNT", "3"))
    return ScoringConfig(
        gold_root=gold_root,
        model_path=model_path,
        top_reason_count=top_reason_count,
    )


def create_app(
    config: ScoringConfig | None = None,
    s3_client: S3ClientProtocol | None = None,
) -> FastAPI:
    scoring_config = config or _default_config()
    app = FastAPI(title="SpanishGas Real-Time Scoring API", version="0.1.0")

    @app.post("/score", response_model=ScoreResponse)
    def score(payload: ScoreRequest) -> ScoreResponse:
        customer_id = payload.customer_id.strip()
        if not customer_id:
            raise HTTPException(status_code=422, detail="customer_id must be non-empty.")

        try:
            asof_date = _parse_iso_date(payload.asof_date)
        except ValueError:
            raise HTTPException(status_code=422, detail="asof_date must be YYYY-MM-DD.")

        resolved_s3_client = _resolve_s3_client(
            paths=[scoring_config.gold_root, scoring_config.model_path],
            s3_client=s3_client,
        )

        try:
            feature_row = _read_customer_feature(
                gold_root=scoring_config.gold_root,
                customer_id=customer_id,
                asof_date=asof_date,
                s3_client=resolved_s3_client,
            )
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=404,
                detail=(
                    "Feature table not found for asof_date="
                    f"{asof_date}."
                ),
            ) from exc

        if feature_row is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Customer '{customer_id}' not found in feature table "
                    f"for asof_date={asof_date}."
                ),
            )

        segment = _load_optional_segment(
            gold_root=scoring_config.gold_root,
            customer_id=customer_id,
            asof_date=asof_date,
            s3_client=resolved_s3_client,
        )

        model_payload = _load_optional_model(scoring_config.model_path, resolved_s3_client)

        if model_payload is None:
            risk_score, reason_features = _heuristic_score(feature_row)
        else:
            model = model_payload["model"]
            vectorizer = model_payload["vectorizer"]
            feature_columns = list(model_payload.get("feature_columns", []))
            numeric_columns = set(model_payload.get("numeric_columns", []))

            if feature_columns:
                normalized = _normalize_for_model(feature_row, feature_columns, numeric_columns)
                matrix = vectorizer.transform([normalized])
                risk_score = _clip_probability(float(model.predict_proba(matrix)[0][1]))
                feature_names = list(vectorizer.get_feature_names_out())
                reason_features = _top_reason_features(
                    matrix_row=matrix[0],
                    model=model,
                    feature_names=feature_names,
                    top_reason_count=scoring_config.top_reason_count,
                )
                if not reason_features:
                    reason_features = ["model_based_risk_score"]
            else:
                risk_score, reason_features = _heuristic_score(feature_row)

        candidate_row = {
            "customer_id": customer_id,
            "risk_score": round(risk_score, 6),
            "segment": segment,
            "margin_eur": round(_estimated_margin_eur(feature_row), 6),
            "acceptance_probability": round(_segment_acceptance_probability(segment), 6),
            "days_to_contract_end": feature_row.get("days_to_contract_end"),
            "shap_top_features": reason_features,
        }

        try:
            recommendation = _recommend_record(candidate_row, run_date=asof_date)
        except Exception as exc:
            raise HTTPException(status_code=500, detail="Recommendation generation failed.") from exc

        if recommendation is None:
            raise HTTPException(status_code=500, detail="Recommendation generation failed.")

        return ScoreResponse(
            customer_id=customer_id,
            asof_date=asof_date,
            churn_score=float(recommendation["risk_score"]),
            segment=str(recommendation["segment"]),
            recommendation=RecommendationPayload(
                action=str(recommendation["action"]),
                timing_window=str(recommendation["timing_window"]),
                expected_margin_impact=float(recommendation["expected_margin_impact"]),
            ),
            reason_codes=[str(code) for code in recommendation["reason_codes"]],
        )

    return app


app = create_app()

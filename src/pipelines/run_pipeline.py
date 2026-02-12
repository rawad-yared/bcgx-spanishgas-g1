"""Orchestrated local pipeline runner for ingestion -> recommend."""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from datetime import date
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import Iterable
from typing import Mapping
from typing import Protocol

import yaml

from src.data.build_training_set import SplitRules
from src.data.build_training_set import TrainingSetConfig
from src.data.build_training_set import build_training_set
from src.data.ingest import run_ingestion
from src.data.silver import run_silver_transforms
from src.features.build_features import build_customer_features
from src.models.churn_baseline import BaselineConfig
from src.models.churn_baseline import train_baseline
from src.models.churn_gbdt import GBDTConfig
from src.models.churn_gbdt import train_gbdt
from src.models.segmentation import build_segmentation
from src.reco.recommend import build_recommendations


class S3ClientProtocol(Protocol):
    """Minimal S3 client protocol used by this module."""

    def get_object(self, **kwargs: Any) -> dict[str, Any]: ...

    def put_object(self, **kwargs: Any) -> dict[str, Any]: ...


@dataclass(frozen=True)
class PipelineSteps:
    ingestion: bool
    silver: bool
    features: bool
    training_set: bool
    train_baseline: bool
    train_gbdt: bool
    segmentation: bool
    score: bool
    recommend: bool


@dataclass(frozen=True)
class PipelinePaths:
    raw_root: str
    bronze_root: str
    silver_root: str
    gold_root: str
    artifacts_root: str


@dataclass(frozen=True)
class PipelineTraining:
    horizon_days: int
    cutoff_date: date
    label_run_date: str
    asof_dates: tuple[date, ...]
    split_rules: SplitRules
    feature_version: str


@dataclass(frozen=True)
class PipelineBaselineModel:
    random_seed: int
    max_iter: int
    regularization_c: float
    top_k_fraction: float
    calibration_bins: int


@dataclass(frozen=True)
class PipelineGBDTModel:
    random_seed: int
    n_estimators: int
    learning_rate: float
    max_depth: int
    top_k_fraction: float
    calibration_bins: int
    max_brier_increase: float
    max_ece_increase: float
    top_feature_count: int
    max_shap_samples: int


@dataclass(frozen=True)
class PipelineSegmentation:
    asof_date: date
    segment_count: int
    top_driver_count: int
    random_seed: int
    report_path: str


@dataclass(frozen=True)
class PipelineScoring:
    asof_date: date
    output_path: str | None
    top_reason_count: int
    default_margin_eur: float
    min_margin_eur: float
    margin_price_delta_weight: float
    margin_interaction_weight: float
    margin_negative_flag_penalty: float
    default_acceptance_probability: float


@dataclass(frozen=True)
class PipelineRecommend:
    input_path: str | None
    output_path: str | None


@dataclass(frozen=True)
class PipelineConfig:
    run_date: str
    paths: PipelinePaths
    steps: PipelineSteps
    training: PipelineTraining
    baseline_model: PipelineBaselineModel
    gbdt_model: PipelineGBDTModel
    segmentation: PipelineSegmentation
    scoring: PipelineScoring
    recommend: PipelineRecommend


@dataclass(frozen=True)
class StepResult:
    name: str
    detail: str


@dataclass(frozen=True)
class PipelineSummary:
    run_date: str
    steps_executed: tuple[StepResult, ...]


@dataclass(frozen=True)
class ScoreSummary:
    asof_date: str
    model_path: str
    output_path: str
    row_count: int


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run end-to-end orchestrated pipeline.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to pipeline config YAML.",
    )
    return parser.parse_args(argv)


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
            "S3 pipeline I/O requested but no S3 SDK is available. "
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


def _parse_iso_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


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
    value = value.lower()
    normalized = []
    for char in value:
        if char.isalnum():
            normalized.append(char)
        else:
            normalized.append("_")
    compact = "".join(normalized).strip("_")
    while "__" in compact:
        compact = compact.replace("__", "_")
    return compact or "unknown"


def _as_bool(raw: Any, default: bool) -> bool:
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return raw != 0
    return str(raw).strip().lower() in {"1", "true", "t", "yes", "y"}


def _as_mapping(value: Any, section: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return value
    raise ValueError(f"Config section `{section}` must be a mapping.")


def _asof_dates_from_config(training_map: Mapping[str, Any], fallback: date) -> tuple[date, ...]:
    raw = training_map.get("asof_dates")
    if raw is None:
        return (fallback,)
    if not isinstance(raw, list):
        raise ValueError("training.asof_dates must be a list of YYYY-MM-DD dates.")
    parsed = sorted({_parse_iso_date(str(item)) for item in raw})
    if not parsed:
        raise ValueError("training.asof_dates must not be empty.")
    return tuple(parsed)


def _load_config(config_path: str) -> PipelineConfig:
    raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("Pipeline config must be a YAML object.")

    run = _as_mapping(raw.get("run"), "run")
    paths_raw = _as_mapping(raw.get("paths"), "paths")
    steps_raw = _as_mapping(raw.get("steps"), "steps")
    training_raw = _as_mapping(raw.get("training"), "training")
    baseline_raw = _as_mapping(raw.get("baseline_model"), "baseline_model")
    gbdt_raw = _as_mapping(raw.get("gbdt_model"), "gbdt_model")
    segmentation_raw = _as_mapping(raw.get("segmentation"), "segmentation")
    scoring_raw = _as_mapping(raw.get("scoring"), "scoring")
    recommend_raw = _as_mapping(raw.get("recommend"), "recommend")

    run_date = str(run.get("run_date", "")).strip()
    if not run_date:
        raise ValueError("run.run_date is required.")
    _parse_iso_date(run_date)

    gold_root = str(paths_raw.get("gold_root", "data/gold/")).strip() or "data/gold/"
    artifacts_root = (
        str(paths_raw.get("artifacts_root", "artifacts/")).strip() or "artifacts/"
    )
    paths = PipelinePaths(
        raw_root=str(paths_raw.get("raw_root", "data/raw/")).strip() or "data/raw/",
        bronze_root=str(paths_raw.get("bronze_root", "data/bronze/")).strip()
        or "data/bronze/",
        silver_root=str(paths_raw.get("silver_root", "data/silver/")).strip()
        or "data/silver/",
        gold_root=gold_root,
        artifacts_root=artifacts_root,
    )

    steps = PipelineSteps(
        ingestion=_as_bool(steps_raw.get("ingestion"), True),
        silver=_as_bool(steps_raw.get("silver"), True),
        features=_as_bool(steps_raw.get("features"), True),
        training_set=_as_bool(steps_raw.get("training_set"), True),
        train_baseline=_as_bool(steps_raw.get("train_baseline"), True),
        train_gbdt=_as_bool(steps_raw.get("train_gbdt"), False),
        segmentation=_as_bool(steps_raw.get("segmentation"), True),
        score=_as_bool(steps_raw.get("score"), True),
        recommend=_as_bool(steps_raw.get("recommend"), True),
    )

    cutoff_date = _parse_iso_date(
        str(training_raw.get("cutoff_date", run.get("scoring_asof_date", run_date)))
    )
    asof_dates = _asof_dates_from_config(training_raw, fallback=cutoff_date)

    split_rules_raw = _as_mapping(training_raw.get("split_rules"), "training.split_rules")
    train_end = _parse_iso_date(
        str(split_rules_raw.get("train_end_date", asof_dates[0].isoformat()))
    )
    valid_end = _parse_iso_date(
        str(split_rules_raw.get("valid_end_date", asof_dates[min(1, len(asof_dates) - 1)].isoformat()))
    )
    test_end = _parse_iso_date(str(split_rules_raw.get("test_end_date", cutoff_date.isoformat())))
    if not (train_end <= valid_end <= test_end):
        raise ValueError("training.split_rules must satisfy train_end <= valid_end <= test_end.")

    training = PipelineTraining(
        horizon_days=int(training_raw.get("horizon_days", 90)),
        cutoff_date=cutoff_date,
        label_run_date=str(training_raw.get("label_run_date", run_date)),
        asof_dates=asof_dates,
        split_rules=SplitRules(
            train_end_date=train_end,
            valid_end_date=valid_end,
            test_end_date=test_end,
        ),
        feature_version=str(training_raw.get("feature_version", "v1")).strip() or "v1",
    )

    baseline_model = PipelineBaselineModel(
        random_seed=int(baseline_raw.get("random_seed", 42)),
        max_iter=int(baseline_raw.get("max_iter", 500)),
        regularization_c=float(baseline_raw.get("regularization_c", 1.0)),
        top_k_fraction=float(baseline_raw.get("top_k_fraction", 0.2)),
        calibration_bins=int(baseline_raw.get("calibration_bins", 10)),
    )

    gbdt_model = PipelineGBDTModel(
        random_seed=int(gbdt_raw.get("random_seed", 42)),
        n_estimators=int(gbdt_raw.get("n_estimators", 200)),
        learning_rate=float(gbdt_raw.get("learning_rate", 0.05)),
        max_depth=int(gbdt_raw.get("max_depth", 3)),
        top_k_fraction=float(gbdt_raw.get("top_k_fraction", 0.2)),
        calibration_bins=int(gbdt_raw.get("calibration_bins", 10)),
        max_brier_increase=float(gbdt_raw.get("max_brier_increase", 0.03)),
        max_ece_increase=float(gbdt_raw.get("max_ece_increase", 0.03)),
        top_feature_count=int(gbdt_raw.get("top_feature_count", 10)),
        max_shap_samples=int(gbdt_raw.get("max_shap_samples", 300)),
    )

    segmentation = PipelineSegmentation(
        asof_date=_parse_iso_date(
            str(segmentation_raw.get("asof_date", asof_dates[-1].isoformat()))
        ),
        segment_count=int(segmentation_raw.get("segment_count", 4)),
        top_driver_count=int(segmentation_raw.get("top_driver_count", 5)),
        random_seed=int(segmentation_raw.get("random_seed", 42)),
        report_path=str(
            segmentation_raw.get(
                "report_path",
                _join_location(artifacts_root, "reports/segmentation_profile.md"),
            )
        ),
    )

    scoring = PipelineScoring(
        asof_date=_parse_iso_date(
            str(scoring_raw.get("asof_date", segmentation.asof_date.isoformat()))
        ),
        output_path=(
            str(scoring_raw.get("output_path")).strip()
            if scoring_raw.get("output_path") is not None
            else None
        ),
        top_reason_count=int(scoring_raw.get("top_reason_count", 3)),
        default_margin_eur=float(scoring_raw.get("default_margin_eur", 85.0)),
        min_margin_eur=float(scoring_raw.get("min_margin_eur", 5.0)),
        margin_price_delta_weight=float(scoring_raw.get("margin_price_delta_weight", 200.0)),
        margin_interaction_weight=float(scoring_raw.get("margin_interaction_weight", 2.0)),
        margin_negative_flag_penalty=float(
            scoring_raw.get("margin_negative_flag_penalty", 6.0)
        ),
        default_acceptance_probability=float(
            scoring_raw.get("default_acceptance_probability", 0.25)
        ),
    )

    recommend = PipelineRecommend(
        input_path=(
            str(recommend_raw.get("input_path")).strip()
            if recommend_raw.get("input_path") is not None
            else None
        ),
        output_path=(
            str(recommend_raw.get("output_path")).strip()
            if recommend_raw.get("output_path") is not None
            else None
        ),
    )

    return PipelineConfig(
        run_date=run_date,
        paths=paths,
        steps=steps,
        training=training,
        baseline_model=baseline_model,
        gbdt_model=gbdt_model,
        segmentation=segmentation,
        scoring=scoring,
        recommend=recommend,
    )


def _model_feature_row(
    row: Mapping[str, Any], feature_columns: list[str], numeric_columns: set[str]
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for column in feature_columns:
        value = row.get(column)
        if column in numeric_columns:
            numeric = _to_float(value)
            output[column] = 0.0 if numeric is None else numeric
        else:
            output[column] = "__missing__" if value is None else str(value)
    return output


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
    if coefficients is None:
        return []
    if len(coefficients) == 0:
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
    return [item[1] for item in contributions[:top_reason_count]]


def _load_optional_segment_map(
    path: str, s3_client: S3ClientProtocol | None
) -> dict[str, str]:
    try:
        rows = _read_jsonl(path, s3_client)
    except FileNotFoundError:
        return {}

    segment_map: dict[str, str] = {}
    for row in rows:
        customer_id = str(row.get("customer_id") or "").strip()
        segment_id = str(row.get("segment_id") or "unknown").strip() or "unknown"
        if customer_id:
            segment_map[customer_id] = segment_id
    return segment_map


def score_recommendation_candidates(
    run_date: str,
    gold_root: str,
    artifacts_root: str,
    scoring: PipelineScoring,
    s3_client: S3ClientProtocol | None = None,
) -> ScoreSummary:
    """Score customers and materialize recommendation_candidates table."""

    run_asof = scoring.asof_date.isoformat()
    model_path = _join_location(artifacts_root, "models/churn_baseline/model.pkl")
    feature_path = _join_location(
        gold_root,
        f"customer_features_asof_date/asof_date={run_asof}/customer_features_asof_date.jsonl",
    )
    segment_path = _join_location(gold_root, f"segments/asof_date={run_asof}/segments.jsonl")
    output_path = scoring.output_path or _join_location(
        gold_root,
        f"recommendation_candidates/run_date={run_date}/recommendation_candidates.jsonl",
    )

    resolved_s3_client = _resolve_s3_client(
        paths=[model_path, feature_path, segment_path, output_path],
        s3_client=s3_client,
    )

    model_payload = pickle.loads(_read_bytes(model_path, resolved_s3_client))
    model = model_payload["model"]
    vectorizer = model_payload["vectorizer"]
    feature_columns = list(model_payload["feature_columns"])
    numeric_columns = set(model_payload["numeric_columns"])
    feature_names = vectorizer.get_feature_names_out().tolist()

    feature_rows = _read_jsonl(feature_path, resolved_s3_client)
    segment_map = _load_optional_segment_map(segment_path, resolved_s3_client)

    candidates: list[dict[str, Any]] = []
    for row in sorted(feature_rows, key=lambda item: str(item.get("customer_id") or "")):
        customer_id = str(row.get("customer_id") or "").strip()
        if not customer_id:
            continue

        normalized = _model_feature_row(row, feature_columns, numeric_columns)
        matrix = vectorizer.transform([normalized])
        risk_score = float(model.predict_proba(matrix)[0][1])

        price_delta = _to_float(row.get("price_vs_benchmark_delta")) or 0.0
        interaction_count = _to_float(row.get("interaction_count_90d")) or 0.0
        negative_flag = _to_float(row.get("negative_consumption_flag")) or 0.0
        margin_eur = (
            scoring.default_margin_eur
            - max(0.0, price_delta) * scoring.margin_price_delta_weight
            - interaction_count * scoring.margin_interaction_weight
            - negative_flag * scoring.margin_negative_flag_penalty
        )
        margin_eur = max(scoring.min_margin_eur, margin_eur)

        reason_features = _top_reason_features(
            matrix_row=matrix[0],
            model=model,
            feature_names=feature_names,
            top_reason_count=scoring.top_reason_count,
        )
        reason_features = [_slug(name) for name in reason_features]

        segment = segment_map.get(customer_id, "unknown")
        acceptance_probability = scoring.default_acceptance_probability
        lowered_segment = segment.lower()
        if "price" in lowered_segment:
            acceptance_probability += 0.06
        if "high_value" in lowered_segment:
            acceptance_probability += 0.03
        if "stable" in lowered_segment:
            acceptance_probability -= 0.04

        candidates.append(
            {
                "customer_id": customer_id,
                "run_date": run_date,
                "risk_score": round(risk_score, 6),
                "segment": segment,
                "margin_eur": round(margin_eur, 6),
                "acceptance_probability": round(
                    _clip_probability(acceptance_probability), 6
                ),
                "days_to_contract_end": row.get("days_to_contract_end"),
                "shap_top_features": reason_features,
            }
        )

    _write_bytes(output_path, _to_jsonl(candidates), resolved_s3_client)

    return ScoreSummary(
        asof_date=run_asof,
        model_path=model_path,
        output_path=output_path,
        row_count=len(candidates),
    )


def run_pipeline(
    config: PipelineConfig,
    s3_client: S3ClientProtocol | None = None,
) -> PipelineSummary:
    """Run the configured end-to-end pipeline steps."""

    steps_executed: list[StepResult] = []

    candidate_s3_client = _resolve_s3_client(
        paths=[
            config.paths.raw_root,
            config.paths.bronze_root,
            config.paths.silver_root,
            config.paths.gold_root,
            config.paths.artifacts_root,
            config.segmentation.report_path,
            config.scoring.output_path or "",
            config.recommend.input_path or "",
            config.recommend.output_path or "",
        ],
        s3_client=s3_client,
    )

    if config.steps.ingestion:
        records = run_ingestion(
            run_date=config.run_date,
            raw_root=config.paths.raw_root,
            bronze_root=config.paths.bronze_root,
            s3_client=candidate_s3_client,
        )
        steps_executed.append(
            StepResult(
                name="ingestion",
                detail=f"datasets={len(records)}",
            )
        )

    if config.steps.silver:
        reports = run_silver_transforms(
            run_date=config.run_date,
            bronze_root=config.paths.bronze_root,
            silver_root=config.paths.silver_root,
            s3_client=candidate_s3_client,
        )
        steps_executed.append(
            StepResult(
                name="silver",
                detail=f"tables={len(reports)}",
            )
        )

    if config.steps.features:
        total_feature_rows = 0
        for asof in config.training.asof_dates:
            feature_rows = build_customer_features(
                asof_date=asof.isoformat(),
                silver_root=config.paths.silver_root,
                gold_root=config.paths.gold_root,
                silver_run_date=config.run_date,
                s3_client=candidate_s3_client,
            )
            total_feature_rows += len(feature_rows)
        steps_executed.append(
            StepResult(
                name="gold_features",
                detail=(
                    f"asof_dates={len(config.training.asof_dates)}, "
                    f"rows={total_feature_rows}"
                ),
            )
        )

    if config.steps.training_set:
        summary = build_training_set(
            config=TrainingSetConfig(
                silver_root=config.paths.silver_root,
                gold_root=config.paths.gold_root,
                horizon_days=config.training.horizon_days,
                cutoff_date=config.training.cutoff_date,
                label_run_date=config.training.label_run_date,
                asof_dates=config.training.asof_dates,
                split_rules=config.training.split_rules,
                feature_version=config.training.feature_version,
            ),
            s3_client=candidate_s3_client,
        )
        steps_executed.append(
            StepResult(
                name="gold_training_set",
                detail=(
                    f"rows={summary.row_count}, "
                    f"train={summary.split_counts['train']}, "
                    f"valid={summary.split_counts['valid']}, "
                    f"test={summary.split_counts['test']}"
                ),
            )
        )

    training_dataset_path = _join_location(
        config.paths.gold_root,
        (
            "churn_training_dataset/"
            f"cutoff_date={config.training.cutoff_date.isoformat()}/"
            "churn_training_dataset.jsonl"
        ),
    )
    baseline_model_output_dir = _join_location(
        config.paths.artifacts_root, "models/churn_baseline/"
    )
    baseline_report_path = _join_location(
        config.paths.artifacts_root, "reports/churn_baseline.md"
    )

    if config.steps.train_baseline:
        baseline_result = train_baseline(
            config=BaselineConfig(
                training_dataset_path=training_dataset_path,
                model_output_dir=baseline_model_output_dir,
                report_path=baseline_report_path,
                random_seed=config.baseline_model.random_seed,
                max_iter=config.baseline_model.max_iter,
                regularization_c=config.baseline_model.regularization_c,
                top_k_fraction=config.baseline_model.top_k_fraction,
                calibration_bins=config.baseline_model.calibration_bins,
            ),
            s3_client=candidate_s3_client,
        )
        primary = baseline_result.metrics[baseline_result.primary_eval_split]
        steps_executed.append(
            StepResult(
                name="train_baseline",
                detail=(
                    f"split={baseline_result.primary_eval_split}, "
                    f"pr_auc={primary.pr_auc:.4f}"
                ),
            )
        )

    if config.steps.train_gbdt:
        gbdt_result = train_gbdt(
            config=GBDTConfig(
                training_dataset_path=training_dataset_path,
                baseline_metrics_path=_join_location(
                    baseline_model_output_dir, "metrics.json"
                ),
                model_output_dir=_join_location(
                    config.paths.artifacts_root, "models/churn_gbdt/"
                ),
                comparison_report_path=_join_location(
                    config.paths.artifacts_root,
                    "reports/churn_gbdt_vs_baseline.md",
                ),
                random_seed=config.gbdt_model.random_seed,
                n_estimators=config.gbdt_model.n_estimators,
                learning_rate=config.gbdt_model.learning_rate,
                max_depth=config.gbdt_model.max_depth,
                top_k_fraction=config.gbdt_model.top_k_fraction,
                calibration_bins=config.gbdt_model.calibration_bins,
                max_brier_increase=config.gbdt_model.max_brier_increase,
                max_ece_increase=config.gbdt_model.max_ece_increase,
                top_feature_count=config.gbdt_model.top_feature_count,
                max_shap_samples=config.gbdt_model.max_shap_samples,
            ),
            s3_client=candidate_s3_client,
        )
        steps_executed.append(
            StepResult(
                name="train_gbdt",
                detail=(
                    f"split={gbdt_result.primary_eval_split}, "
                    f"champion_pass={gbdt_result.champion_pass}"
                ),
            )
        )

    if config.steps.segmentation:
        segment_summary = build_segmentation(
            asof_date=config.segmentation.asof_date.isoformat(),
            segment_count=config.segmentation.segment_count,
            gold_root=config.paths.gold_root,
            report_path=config.segmentation.report_path,
            top_driver_count=config.segmentation.top_driver_count,
            random_seed=config.segmentation.random_seed,
            s3_client=candidate_s3_client,
        )
        steps_executed.append(
            StepResult(
                name="segmentation",
                detail=(
                    f"effective_segments={segment_summary.effective_segment_count}, "
                    f"rows={segment_summary.row_count}"
                ),
            )
        )

    if config.steps.score:
        score_summary = score_recommendation_candidates(
            run_date=config.run_date,
            gold_root=config.paths.gold_root,
            artifacts_root=config.paths.artifacts_root,
            scoring=config.scoring,
            s3_client=candidate_s3_client,
        )
        steps_executed.append(
            StepResult(
                name="score",
                detail=f"rows={score_summary.row_count}, output={score_summary.output_path}",
            )
        )

    if config.steps.recommend:
        recommendation_summary = build_recommendations(
            run_date=config.run_date,
            gold_root=config.paths.gold_root,
            input_path=config.recommend.input_path,
            output_path=config.recommend.output_path,
            s3_client=candidate_s3_client,
        )
        steps_executed.append(
            StepResult(
                name="recommend",
                detail=(
                    f"written_rows={recommendation_summary.written_rows}, "
                    f"offers={recommendation_summary.offer_rows}, "
                    f"no_offer={recommendation_summary.no_offer_rows}"
                ),
            )
        )

    return PipelineSummary(
        run_date=config.run_date,
        steps_executed=tuple(steps_executed),
    )


def _render_summary(summary: PipelineSummary) -> str:
    lines = ["Pipeline run complete:", f"- run_date={summary.run_date}"]
    for step in summary.steps_executed:
        lines.append(f"- {step.name}: {step.detail}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    config = _load_config(args.config)
    summary = run_pipeline(config=config)
    print(_render_summary(summary))


if __name__ == "__main__":
    main()

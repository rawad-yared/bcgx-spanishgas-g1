"""Centralised project settings loaded from environment / .env file."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

_ENV_LOADED = False


def _ensure_env() -> None:
    global _ENV_LOADED
    if not _ENV_LOADED:
        # Walk up to repo root looking for .env
        env_path = Path(__file__).resolve().parent.parent / ".env"
        if env_path.exists():
            load_dotenv(env_path)
        _ENV_LOADED = True


def _get(key: str, default: str = "") -> str:
    _ensure_env()
    return os.environ.get(key, default)


@dataclass(frozen=True)
class AWSSettings:
    region: str = field(default_factory=lambda: _get("AWS_REGION", "eu-west-1"))
    s3_bucket: str = field(default_factory=lambda: _get("S3_BUCKET", "spanishgas-data-g1"))
    s3_prefix_raw: str = field(default_factory=lambda: _get("S3_PREFIX_RAW", "raw/"))
    s3_prefix_bronze: str = field(default_factory=lambda: _get("S3_PREFIX_BRONZE", "bronze/"))
    s3_prefix_silver: str = field(default_factory=lambda: _get("S3_PREFIX_SILVER", "silver/"))
    s3_prefix_gold: str = field(default_factory=lambda: _get("S3_PREFIX_GOLD", "gold/"))
    s3_prefix_models: str = field(default_factory=lambda: _get("S3_PREFIX_MODELS", "models/"))
    s3_prefix_scored: str = field(default_factory=lambda: _get("S3_PREFIX_SCORED", "scored/"))
    dynamodb_manifest_table: str = field(
        default_factory=lambda: _get("DYNAMODB_MANIFEST_TABLE", "spanishgas-pipeline-manifest")
    )
    sagemaker_role_arn: str = field(default_factory=lambda: _get("SAGEMAKER_ROLE_ARN", ""))
    sagemaker_model_package_group: str = field(
        default_factory=lambda: _get("SAGEMAKER_MODEL_PACKAGE_GROUP", "spanishgas-churn")
    )
    sagemaker_processing_instance: str = field(
        default_factory=lambda: _get("SAGEMAKER_PROCESSING_INSTANCE", "ml.m5.xlarge")
    )
    sagemaker_training_instance: str = field(
        default_factory=lambda: _get("SAGEMAKER_TRAINING_INSTANCE", "ml.m5.xlarge")
    )
    step_functions_arn: str = field(default_factory=lambda: _get("STEP_FUNCTIONS_ARN", ""))
    sns_topic_arn: str = field(default_factory=lambda: _get("SNS_TOPIC_ARN", ""))
    cloudwatch_namespace: str = field(
        default_factory=lambda: _get("CLOUDWATCH_NAMESPACE", "SpanishGas/MLOps")
    )


@dataclass(frozen=True)
class ModelSettings:
    promotion_pr_auc_threshold: float = field(
        default_factory=lambda: float(_get("PROMOTION_PR_AUC_THRESHOLD", "0.70"))
    )
    target_recall: float = 0.70
    risk_tier_thresholds: tuple[float, ...] = (0.40, 0.60, 0.80)


@dataclass(frozen=True)
class Settings:
    aws: AWSSettings = field(default_factory=AWSSettings)
    model: ModelSettings = field(default_factory=ModelSettings)
    data_dir: Path = field(
        default_factory=lambda: Path(_get("DATA_DIR", "data"))
    )
    log_level: str = field(default_factory=lambda: _get("LOG_LEVEL", "INFO"))


def get_settings() -> Settings:
    """Return a Settings instance populated from env vars / .env file."""
    return Settings()

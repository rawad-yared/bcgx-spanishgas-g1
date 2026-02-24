"""Test that Settings loads correctly from environment."""

from pathlib import Path

from configs.settings import AWSSettings, ModelSettings, Settings, get_settings


def test_settings_defaults():
    settings = get_settings()
    assert isinstance(settings, Settings)
    assert isinstance(settings.aws, AWSSettings)
    assert isinstance(settings.model, ModelSettings)
    assert isinstance(settings.data_dir, Path)


def test_aws_defaults():
    aws = AWSSettings()
    assert aws.region == "eu-west-1"
    # s3_bucket comes from .env (spanishgas-data-dev) or code default (spanishgas-data-g1)
    assert aws.s3_bucket in ("spanishgas-data-dev", "spanishgas-data-g1")
    assert aws.s3_prefix_raw == "raw/"
    assert aws.s3_prefix_bronze == "bronze/"
    assert aws.s3_prefix_silver == "silver/"
    assert aws.s3_prefix_gold == "gold/"
    assert aws.s3_prefix_models == "models/"
    assert aws.s3_prefix_scored == "scored/"
    # manifest table comes from .env or code default
    assert aws.dynamodb_manifest_table in (
        "spanishgas-dev-pipeline-manifest", "spanishgas-pipeline-manifest"
    )


def test_model_defaults():
    model = ModelSettings()
    assert model.promotion_pr_auc_threshold == 0.70
    assert model.target_recall == 0.70
    assert model.risk_tier_thresholds == (0.40, 0.60, 0.80)


def test_env_override(monkeypatch):
    monkeypatch.setenv("S3_BUCKET", "my-custom-bucket")
    monkeypatch.setenv("PROMOTION_PR_AUC_THRESHOLD", "0.85")
    monkeypatch.setenv("DATA_DIR", "/tmp/test-data")
    settings = Settings()
    assert settings.aws.s3_bucket == "my-custom-bucket"
    assert settings.model.promotion_pr_auc_threshold == 0.85
    assert settings.data_dir == Path("/tmp/test-data")


def test_settings_frozen():
    settings = get_settings()
    try:
        settings.log_level = "DEBUG"
        assert False, "Should have raised FrozenInstanceError"
    except AttributeError:
        pass

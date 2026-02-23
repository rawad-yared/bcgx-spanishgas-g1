# CLAUDE.md - SpanishGas AWS MLOps Pipeline

## Project Overview
Churn prediction system for 20,099 Spanish energy customers. Converting Jupyter notebook logic (01: medallion ETL, 02: modeling) into a production AWS MLOps pipeline.

## Branch
`feature/aws-mlops-pipeline` (from `main`)

## Current State (session 4)
- **All 8 phases complete** (0-7)
- **93 tests passing** across 16 test files, 1 skipped (xgboost conditional)
- **Ruff lint clean**
- **Committed:** `be178d2` — 106 files, 26,884 insertions

## Key Architecture
- **S3 layout:** Single bucket, prefix-based (`raw/`, `bronze/`, `silver/`, `gold/`, `models/`, `scored/`)
- **Orchestration:** Lambda (S3 trigger) -> Step Functions -> SageMaker Processing/Training
- **Idempotency:** DynamoDB manifest table (PK: file_key)
- **Model:** XGBoost (E5 experiment), threshold optimized for precision at recall >= 0.70
- **Drift:** KS test (scipy), p_threshold=0.01
- **Promotion gate:** PR-AUC >= 0.70

## Module Map
```
configs/                   - Settings, feature_tiers.yaml, column_registry.yaml
src/data/                  - ingest.py (bronze), silver.py, build_training_set.py
src/features/              - build_features.py (7 feature tiers, gold master)
src/models/                - churn_model.py, preprocessing.py, scorer.py, artifacts.py, registry.py
src/reco/                  - schema.py, engine.py
src/pipelines/             - lambda_handler.py, manifest.py, s3_io.py, run.py
src/pipelines/steps/       - bronze, silver, gold, train, evaluate, score, drift steps
src/monitoring/            - drift.py, data_quality.py, alerts.py, reference_store.py
src/serving/ui/            - app.py, data_loader.py, pages/{model_performance, drift_monitor, customer_risk, pipeline_status}
infra/terraform/           - main.tf, backend.tf, variables.tf, outputs.tf + 8 modules
.github/workflows/         - ci.yml, deploy.yml, retrain.yml
Dockerfile.lambda          - Lambda container image
Dockerfile.processing      - SageMaker Processing container image
```

## Terraform Modules
```
infra/terraform/modules/
  s3/              - Single bucket, versioning, encryption, lifecycle rules
  dynamodb/        - Manifest table, PAY_PER_REQUEST
  iam/             - Lambda, Step Functions, SageMaker roles
  lambda/          - Pipeline trigger function, S3 notification
  step_functions/  - State machine + ASL definition (asl/pipeline.asl.json)
  sagemaker/       - Model Package Group
  monitoring/      - SNS topic, CloudWatch alarms (Lambda errors, SFN failures, drift)
  ecr/             - Lambda + Processing container repos
```

## Commands
```bash
make install    # pip install -e ".[dev]"
make lint       # ruff check src/ tests/
make test       # pytest tests/ -v
make test-cov   # pytest --cov
```

## Test Files (16 files, 93 tests)
```
tests/test_settings.py             - 5 tests (configs)
tests/test_ingest.py               - 6 tests (bronze)
tests/test_silver.py               - 8 tests (silver transforms)
tests/test_build_features.py       - 9 tests (gold features)
tests/test_build_training_set.py   - 5 tests (model matrix)
tests/test_models.py               - 7 tests (preprocessing, model defs, scoring)
tests/test_reco.py                 - 7 tests (recommendations)
tests/test_lambda_handler.py       - 3 tests (moto: DynamoDB + SFN mocks)
tests/test_manifest.py             - 5 tests (moto: DynamoDB mocks)
tests/test_s3_io.py                - 5 tests (moto: S3 parquet/json/csv round-trips)
tests/test_artifacts.py            - 2 tests (moto: save/load sklearn pipeline to S3)
tests/test_drift.py                - 10 tests (KS drift detection)
tests/test_data_quality.py         - 7 tests (null rates, duplicates, schema)
tests/test_alerts.py               - 5 tests (moto: SNS + CloudWatch)
tests/test_streamlit_data_loader.py - 8 tests (local file loading)
tests/test_pipeline_e2e.py         - 1 test (bronze smoke test)
```

## Known Issues
- None currently

## Full Context
See `CONTEXT.MD` for complete dump with all files changed, decisions, blockers, and ordered next steps.

# CLAUDE.md - SpanishGas AWS MLOps Pipeline

## Project Overview
Churn prediction system for 20,099 Spanish energy customers. Converting Jupyter notebook logic (01: medallion ETL, 02: modeling) into a production AWS MLOps pipeline.

## Branch
`feature/aws-mlops-pipeline` (from `main`)

## Current State (session 6)
- **All 8 phases complete** (0-7) + **Streamlit auto-deployment + UI enhancements + GitHub OIDC**
- **Infrastructure deployed to AWS** — 25/25 Terraform resources, Docker images in ECR
- **121 tests passing** across 18 test files, 1 skipped (xgboost conditional), 1 pre-existing failure (test_aws_defaults bucket name mismatch)
- **Ruff lint clean**
- **Latest commit:** `f3262ca` — README revamp
- **Prior commit:** `67df636` — Streamlit auto-deploy, enhanced UI, GitHub OIDC, expanded tests
- **Waiting on:** SageMaker training job quota (`ml.m5.xlarge`) approval before pipeline can run

## Key Architecture
- **S3 layout:** Single bucket, prefix-based (`raw/`, `bronze/`, `silver/`, `gold/`, `models/`, `scored/`)
- **Orchestration:** Lambda (S3 trigger) -> Step Functions -> SageMaker Processing/Training
- **Idempotency:** DynamoDB manifest table (PK: file_key)
- **Model:** XGBoost (E5 experiment), threshold optimized for precision at recall >= 0.70
- **Drift:** KS test (scipy), p_threshold=0.01
- **Promotion gate:** PR-AUC >= 0.70
- **Dashboard:** Streamlit on ECS Fargate behind ALB, auto-deployed via GitHub Actions

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
src/serving/ui/            - app.py, data_loader.py, pages/{overview, model_performance, drift_monitor, customer_risk, recommendations, pipeline_status}
infra/terraform/           - main.tf, backend.tf, variables.tf, outputs.tf + 11 modules
.github/workflows/         - ci.yml, deploy.yml, retrain.yml
Dockerfile.lambda          - Lambda container image
Dockerfile.processing      - SageMaker Processing container image
Dockerfile.streamlit       - Streamlit dashboard container image
```

## Terraform Modules (11)
```
infra/terraform/modules/
  s3/              - Single bucket, versioning, encryption, lifecycle rules
  dynamodb/        - Manifest table, PAY_PER_REQUEST
  iam/             - Lambda, SFN, SageMaker, ECS execution + task roles
  lambda/          - Pipeline trigger function, S3 notification
  step_functions/  - State machine + ASL definition (asl/pipeline.asl.json)
  sagemaker/       - Model Package Group
  monitoring/      - SNS topic, CloudWatch alarms (Lambda errors, SFN failures, drift)
  ecr/             - Lambda + Processing + Streamlit container repos
  networking/      - Default VPC, ALB + ECS security groups
  ecs/             - Fargate cluster, task def, service, ALB, target group, CW logs
  github_oidc/     - OIDC identity provider + scoped deploy role for GitHub Actions
```

## Commands
```bash
make install              # pip install -e ".[dev]"
make lint                 # ruff check src/ tests/
make test                 # pytest tests/ -v
make test-cov             # pytest --cov
make streamlit            # streamlit run (local)
make docker-build-streamlit  # build Streamlit container
make docker-run-streamlit    # run Streamlit container locally
```

## Test Files (18 files, 122 tests)
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
tests/test_streamlit_pages.py      - 8 tests (page render mocks for 4 pages)
tests/test_pipeline_e2e.py         - 22 tests (bronze, silver, gold, score, full pipeline E2E)
tests/test_imports.py              - 1 test (package import smoke test)
```

## AWS Deployment Info
- **Account:** 559307249592 | **Region:** eu-west-1 | **IAM User:** powerco-mlflow-local
- **TF State:** S3 `spanishgas-terraform-state` + DynamoDB `spanishgas-terraform-locks`
- **Data bucket:** `spanishgas-data-dev` | **ECR:** `spanishgas-dev-lambda`, `spanishgas-dev-processing`, `spanishgas-dev-streamlit`
- **Lambda:** `spanishgas-dev-pipeline-trigger` | **SFN:** `spanishgas-dev-pipeline`
- **ECS:** `spanishgas-dev-cluster` / `spanishgas-dev-streamlit` service | **ALB:** `spanishgas-dev-streamlit-alb`
- **OIDC:** `spanishgas-dev-github-deploy-role` (set as `AWS_DEPLOY_ROLE_ARN` GitHub secret)
- **Alert email:** `rawad.yared@student.ie.edu`
- **Docker build note:** Must use `--provenance=false` for Lambda images (V2 manifest required)

## Known Issues
- SageMaker `ml.m5.xlarge for training job usage` quota pending approval (processing quota approved)
- `test_aws_defaults` fails due to `.env` having `spanishgas-data-dev` vs test expecting `spanishgas-data-g1` (pre-existing)

## Full Context
See `CONTEXT.MD` for complete dump with all files changed, decisions, blockers, and ordered next steps.

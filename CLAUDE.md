# CLAUDE.md - SpanishGas AWS MLOps Pipeline

## Project Overview
Churn prediction system for 20,099 Spanish energy customers. Converting Jupyter notebook logic (01: medallion ETL, 02: modeling) into a production AWS MLOps pipeline.

## Branch
`feature/aws-mlops-pipeline` (from `main`)

## Current State (session 10)
- **Full pipeline E2E working on AWS** — all 8 SFN steps complete successfully (~23 min total)
- **Streamlit dashboard live** — ALB URL serving pages (rebuilt for amd64 + `pip install -e .` fix)
- **All 3 Docker images in ECR** — Lambda, Processing, Streamlit (all linux/amd64)
- **129 tests passing** across 18 test files, 1 skipped (xgboost conditional), 1 pre-existing failure (test_aws_defaults bucket name mismatch)
- **Ruff lint clean** (except pre-existing drift_step.py import order)
- **Latest commit:** `635139b` — sessions 8-10 changes uncommitted
- **feature_tiers.yaml fully rewritten** — now matches actual `build_features.py` output (41 features in E5_full, was ~10-15 matching)
- **Recommendations pipeline fixed** — `score_step.py` now calls `generate_recommendations()` and writes `scored/recommendations.parquet`
- **eval.json fixed** — now saves actual features used (`X.columns`) instead of YAML list
- **3 missing compound features added** — `sales_channel_x_renewal_bucket`, `has_interaction_x_renewal_bucket`, `competition_x_intent` now created in `build_compound_features()`
- **Needs: rebuild Docker images + re-trigger pipeline** to validate PR-AUC improvement

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

## Test Files (18 files, 129 tests)
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
tests/test_streamlit_data_loader.py - 10 tests (local file loading + recommendations)
tests/test_streamlit_pages.py      - 12 tests (page render mocks for 6 pages)
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
- **Docker build note:** Must use `--platform linux/amd64 --provenance=false` for all images (Apple Silicon builds arm64 by default; Lambda/SageMaker need x86_64)

## Known Issues
- `test_aws_defaults` fails due to `.env` having `spanishgas-data-dev` vs test expecting `spanishgas-data-g1` (pre-existing)
- `drift_step.py` has pre-existing ruff I001 import order lint warning (non-blocking)

## Session 8 Fixes Applied (uncommitted)
1. **Instance type `ml.m5.xlarge` → `ml.m5.large`** — updated `configs/settings.py`, `.env`, `.env.example`, `infra/terraform/variables.tf`, `infra/terraform/modules/step_functions/variables.tf` (dev.tfvars already had `ml.m5.large`)
2. **Docker images rebuilt for linux/amd64** — Lambda + Processing images were arm64 (Apple Silicon), causing `Exec format error` on SageMaker x86_64 instances. Rebuilt with `--platform linux/amd64 --provenance=false` and pushed to ECR
3. **SFN IAM policy** — added `sagemaker:AddTags` (required by `.sync` managed tags) and `StepFunctionsGetEventsForSageMakerTrainingJobsRule` EventBridge rule
4. **All 6 step files env var support** — `--bucket` now defaults to `S3_BUCKET` env var, `--region` to `AWS_REGION`, `--sns-topic-arn` to `SNS_TOPIC_ARN` (matching ASL Environment config). Files: `bronze_step.py`, `silver_step.py`, `gold_step.py`, `train_step.py`, `evaluate_step.py`, `score_step.py`, `drift_step.py`
5. **Lambda function code updated** — `aws lambda update-function-code` to pull new amd64 image digest

## Session 9 Fixes Applied (uncommitted)
1. **Consumption file: CSV→parquet converted locally, uploaded to S3** — `raw/consumption_hourly_2024.parquet` (1.8 GB)
2. **BronzeETL OOM fix** — rewrote `bronze_step.py` with chunked processing via `read_parquet_batches()` (downloads to temp file, iterates 2M-row batches via pyarrow). Runs on `ml.m5.large` (8 GB).
3. **TrainModel 0-features fix** — `train_step.py` had 3 bugs: experiment `"E5"` → `"E5_full"`, key `"tiers"` → `"features"`, top-level `"tiers"` → `"feature_tiers"`
4. **DriftCheck NoSuchKey fix** — `drift_step.py` now handles missing reference file on first run (saves baseline, skips comparison)
5. **Streamlit exec format error** — rebuilt Streamlit image for linux/amd64
6. **Streamlit ModuleNotFoundError** — added `pip install -e .` to `Dockerfile.streamlit`
7. **Added `processing_instance_large` TF variable** — for BronzeETL xlarge override (later reverted to ml.m5.large after chunked fix)
8. **New `read_parquet_batches()` in `s3_io.py`** — streams parquet from S3 via temp file + pyarrow batch iterator

## Session 10 Fixes Applied (uncommitted)
1. **feature_tiers.yaml complete rewrite** — all tier feature names now match actual `build_features.py` output columns. E5_full: 41 features (was ~10-15 matching). Key renames: `avg_elec_consumption_kwh` → `avg_monthly_elec_kwh`, `avg_gas_consumption_kwh` → `avg_monthly_gas_m3`, `provincial_elec_cost_eur_kwh` → `province_avg_elec_cost_2024`, etc. Removed 15 non-existent features, added 16+ missing features from gold master.
2. **eval.json now saves actual features** — `train_step.py` saves `X.columns.tolist()` as `"features"`, plus `"requested_features"` and `"missing_features"` for debugging.
3. **Recommendations pipeline integrated** — `score_step.py` now calls `generate_recommendations()` from `src/reco/engine.py` and writes `scored/recommendations.parquet` to S3. Fixes broken Streamlit Recommendations page.
4. **3 missing compound features added** — `build_compound_features()` in `build_features.py` now creates `sales_channel_x_renewal_bucket`, `has_interaction_x_renewal_bucket`, `competition_x_intent` (were in `build_training_set.py` CATEGORICAL_DEFAULT_COLS but never created).
5. **6 new Streamlit tests** — `TestOverviewPage` (2 tests), `TestRecommendationsPage` (2 tests), `TestLoadRecommendations` (2 tests). Total: 129 tests (127 pass, 1 pre-existing failure, 1 skipped).
6. **Added E8_no_sentiment experiment** — ablation experiment without sentiment features (production fallback if NLP unavailable).

## Session 10 Column Audit Notes
- `customer_intent`, `sentiment_label`, `sentiment_neg/pos/neu`, `interaction_summary`, `date` — these columns exist in the **real S3 data** because the notebook (01_data_layers_and_gold.ipynb) pre-computes them via HuggingFace NLP sentiment analysis and regex intent classification on `interaction_summary` text. They flow through `build_bronze_customer()` → silver → gold correctly.
- `column_registry.yaml` only describes raw file schemas, not the enriched columns — this is expected.
- `build_features.py` uses `if col in df.columns` guards throughout — correct defensive coding for environments where NLP hasn't been run.

## Session 9 Blockers (RESOLVED in session 10)
1. ~~**PR-AUC = 0.289 vs notebook ~0.745**~~ — FIXED: feature_tiers.yaml rewritten to match actual gold master columns.
2. ~~**Recommendations page error**~~ — FIXED: `score_step.py` now generates `scored/recommendations.parquet`.
3. ~~**eval.json misleading**~~ — FIXED: saves actual `X.columns` instead of YAML list.

## SageMaker Quotas (updated session 8)
- `ml.m5.large for training job usage`: 15 instances available
- `ml.m5.large for processing job usage`: 4 instances available
- `ml.m5.xlarge for processing job usage`: 1 instance available
- `ml.m5.xlarge for training job usage`: still pending (no longer needed — switched to ml.m5.large)

## IAM User Policy
- `powerco-mlflow-local` now has **AdministratorAccess** only (10 granular policies replaced in session 7 to fit the 10-policy limit and add EC2/ECS/ELB permissions)

## Streamlit Dashboard
- **ALB DNS:** `spanishgas-dev-streamlit-alb-1221532574.eu-west-1.elb.amazonaws.com`
- ECS Fargate service: `spanishgas-dev-streamlit` in `spanishgas-dev-cluster`

## Full Context
See `CONTEXT.MD` for complete dump with all files changed, decisions, blockers, and ordered next steps.

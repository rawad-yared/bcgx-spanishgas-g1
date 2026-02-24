# CLAUDE.md - SpanishGas AWS MLOps Pipeline

## Project Overview
Churn prediction system for 20,099 Spanish energy customers. Converting Jupyter notebook logic (01: medallion ETL, 02: modeling) into a production AWS MLOps pipeline.

## Branch
`feature/aws-mlops-pipeline` (from `main`)

## Current State (session 11)
- **Full pipeline E2E working on AWS** — all 8 SFN steps complete successfully (~23 min total)
- **Streamlit dashboard live** — ALB URL serving all 8 pages (rebuilt for amd64 + `pip install -e .` fix)
- **All 3 Docker images in ECR** — Lambda, Processing, Streamlit (all linux/amd64, rebuilt session 11)
- **129 tests passing** across 18 test files, 1 skipped (xgboost conditional), 1 pre-existing failure (test_aws_defaults bucket name mismatch)
- **Ruff lint clean** — all I001 import order issues fixed (s3_io.py + drift_step.py)
- **Latest commit:** `ea185bc` — pushed to `origin/feature/aws-mlops-pipeline`, CI passing
- **Drift step fix deployed** — `_json_default` numpy serializer in `s3_io.write_json`, pipeline runs clean
- **Customer Lookup fix deployed** — `reason_codes` numpy array converted to list before truthiness check
- **README updated** — all 7 ASCII diagrams replaced with Mermaid, added feature tiers diagram, 8 pages documented
- **PR-AUC = ~0.4** — improved from 0.289 but still below notebook's ~0.7 (see Next Steps for root cause + fix plan)

## Key Architecture
- **S3 layout:** Single bucket, prefix-based (`raw/`, `bronze/`, `silver/`, `gold/`, `models/`, `scored/`)
- **Orchestration:** Lambda (S3 trigger) -> Step Functions -> SageMaker Processing/Training
- **Idempotency:** DynamoDB manifest table (PK: file_key)
- **Model:** XGBoost (E5_full experiment), threshold optimized for precision at recall >= 0.70
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
src/serving/ui/            - app.py, data_loader.py, pages/{overview, data_explorer, model_performance, drift_monitor, customer_risk, customer_lookup, recommendations, pipeline_status}
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
- **PR-AUC ~0.4 vs notebook ~0.7** — root cause is missing features in production `build_features.py` (see Next Steps)

## Session 11 Fixes Applied (committed + pushed)
1. **Drift step JSON serialization fix** — `s3_io.py` now has `_json_default()` handler for `numpy.bool_`, `numpy.integer`, `numpy.floating`, `numpy.ndarray`. `write_json()` passes `default=_json_default` to `json.dumps()`. Previously crashed with `TypeError: Object of type bool is not JSON serializable`.
2. **Customer Lookup "Why This Recommendation" fix** — `customer_lookup.py` line 126: `reason_codes` from parquet comes back as numpy array. `if numpy_array:` raises `ValueError: truth value of array is ambiguous`. Fixed by converting with `.tolist()` before truthiness check.
3. **Ruff I001 import order fixes** — `s3_io.py`: `boto3` moved before `numpy` (alphabetical). `drift_step.py`: removed blank line between `pandas` and `botocore` (same import group).
4. **README Mermaid diagrams** — all 7 ASCII box-and-dash diagrams replaced with Mermaid. Added feature tiers diagram. Updated to 8 dashboard pages, 129+ tests, E5_full/E8_no_sentiment experiments, `--platform linux/amd64` in Docker docs.
5. **Docker images rebuilt + pushed** — Processing (drift fix) and Streamlit (customer lookup fix) images rebuilt for linux/amd64, pushed to ECR, ECS redeployment triggered.
6. **Pipeline re-triggered and SUCCEEDED** — `driftfix-20260224-141631` execution completed all 8 steps (~23 min). Drift step now runs clean.

## Session 10 Fixes Applied (committed in session 11)
1. **feature_tiers.yaml complete rewrite** — all tier feature names now match actual `build_features.py` output columns. E5_full: 41 features (was ~10-15 matching).
2. **eval.json now saves actual features** — `train_step.py` saves `X.columns.tolist()` as `"features"`, plus `"requested_features"` and `"missing_features"` for debugging.
3. **Recommendations pipeline integrated** — `score_step.py` now calls `generate_recommendations()` from `src/reco/engine.py` and writes `scored/recommendations.parquet` to S3.
4. **3 missing compound features added** — `build_compound_features()` in `build_features.py` now creates `sales_channel_x_renewal_bucket`, `has_interaction_x_renewal_bucket`, `competition_x_intent`.
5. **6 new Streamlit tests** — Total: 129 tests (127 pass, 1 pre-existing failure, 1 skipped).
6. **Added E8_no_sentiment experiment** — ablation experiment without sentiment features.

## Session 10 Column Audit Notes
- `customer_intent`, `sentiment_label`, `sentiment_neg/pos/neu`, `interaction_summary`, `date` — these columns exist in the **real S3 data** because the notebook (01_data_layers_and_gold.ipynb) pre-computes them via HuggingFace NLP sentiment analysis and regex intent classification on `interaction_summary` text. They flow through `build_bronze_customer()` → silver → gold correctly.
- `column_registry.yaml` only describes raw file schemas, not the enriched columns — this is expected.
- `build_features.py` uses `if col in df.columns` guards throughout — correct defensive coding for environments where NLP hasn't been run.

## PR-AUC Gap: Root Cause Analysis (session 11)
Production pipeline PR-AUC = ~0.4, notebook = ~0.7. Root cause is **feature parity** — the production `build_features.py` creates 41 features, but the notebook uses ~56 features.

### Fix Steps to Close the PR-AUC Gap
1. **`src/features/build_features.py`** — Add ~15 missing feature computations to match the notebook:
   - **MP Risk features (~9):** `std_monthly_elec_kwh`, `std_monthly_gas_m3`, `std_margin`, `min_monthly_margin`, `max_negative_margin`, `elec_price_volatility_12m`, `is_price_increase`, `rolling_margin_trend`, `province_elec_cost_trend`
   - **Behavioral features (~6):** `interaction_within_3m_of_renewal`, `complaint_near_renewal`, `is_interaction_within_30d`, `recent_complaint_flag`, `is_complaint_intent`, `intent_severity_score`
   - **Compound features (~3):** `dual_fuel_x_renewal`, `dual_fuel_x_competition`, `complaint_x_negative_sentiment`
2. **`configs/feature_tiers.yaml`** — Add the new feature names to the corresponding tier lists so `train_step.py` selects them for training
3. **`src/models/churn_model.py`** (optional) — Align threshold selection with notebook approach (notebook uses full training set for threshold optimization, pipeline uses smaller validation subset)

### Secondary Issue: Threshold Selection
The notebook picks the decision threshold on the **full training set** (~16K customers), while production picks it on a **validation subset** (~4K), leading to a different operating point.

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
- **8 pages:** Overview, Data Explorer, Model Performance, Drift Monitor, Customer Risk, Customer Lookup, Recommendations, Pipeline Status

## Full Context
See `CONTEXT.MD` for complete dump with all files changed, decisions, blockers, and ordered next steps.

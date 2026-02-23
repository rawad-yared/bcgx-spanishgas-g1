# Context Dump

## Session Info (updated 2026-02-23 session 6)
- **Timestamp:** 2026-02-23
- **Branch:** `feature/aws-mlops-pipeline`
- **Latest commit:** `f3262ca` ("Revamp README with full architecture diagrams, dataset docs, and setup guide")
- **Prior commit:** `67df636` ("Add Streamlit auto-deployment, enhanced UI, GitHub OIDC, and expanded tests")
- **All phases committed + infrastructure deployed to AWS + Streamlit auto-deploy stack added.** Clean working tree (except CLAUDE.md + CONTEXT.MD).

---

## Objective Status

Implementing a 8-phase plan to convert SpanishGas Jupyter notebooks into a production AWS MLOps system.

| Phase | Status | Notes |
|-------|--------|-------|
| 0: Foundation | DONE | configs, deps, tooling, .env.example |
| 1A: Data Ingestion | DONE | src/data/ingest.py + tests |
| 1B: Silver Transforms | DONE | src/data/silver.py + tests |
| 1C: Feature Engineering | DONE | src/features/build_features.py + tests |
| 1D: Training Set Builder | DONE | src/data/build_training_set.py + tests |
| 1E: Models | DONE | src/models/{preprocessing,churn_model,scorer}.py + tests |
| 1F: Recommendations | DONE | src/reco/{schema,engine}.py + tests |
| 1G: E2E Integration Test | DONE | tests/test_pipeline_e2e.py (bronze smoke test) |
| 2: Terraform IaC | DONE | 32 files under infra/terraform/ (8 modules) |
| 3: Lambda + Step Functions + Docker | DONE | All step files, run.py, Dockerfiles written |
| 4: SageMaker Training + Registry | DONE | train_step.py, evaluate_step.py, registry.py, artifacts.py all written |
| 5: Monitoring + Drift | DONE | drift.py, data_quality.py, alerts.py, reference_store.py, drift_step.py all written |
| 6: Streamlit Dashboard | DONE | app.py + 6/6 pages + data_loader.py (overview + recommendations added session 6) |
| 7: CI/CD + Ops | DONE | ci.yml, deploy.yml, retrain.yml (3 GitHub Actions workflows) |
| 8: Streamlit Auto-Deploy | DONE (session 6) | Dockerfile.streamlit, ECS Fargate, ALB, networking, ECR repo, deploy.yml updated |
| 9: GitHub OIDC | DONE (session 6) | OIDC provider + scoped deploy role Terraform module |
| 10: UI Enhancements | DONE (session 6) | Overview page, Recommendations page, env var config for S3/DynamoDB |
| 11: Test Expansion | DONE (session 6) | E2E tests 1->22, page render tests (8), pytest.mark.slow registered |
| 12: README Revamp | DONE (session 6) | 701-line README with architecture diagrams, dataset docs, setup guide |

---

## Files Inspected (this session)

- `src/pipelines/lambda_handler.py` (read at session start)
- `src/pipelines/manifest.py` (read at session start)
- `src/pipelines/s3_io.py` (read at session start)
- `src/pipelines/steps/__init__.py` (read at session start)
- `src/data/ingest.py` (read at session start)
- All files listed in `find src -name "*.py"` and `find tests -name "*.py"`
- `git status --short` output
- Test results from `python -m pytest tests/`

---

## Files Changed (and purpose)

### Session 6 — Streamlit Auto-Deploy + UI + OIDC + Tests + README:
- `Dockerfile.streamlit` - NEW: Streamlit container image (python:3.12-slim, healthcheck, env vars)
- `Makefile` - MODIFIED: Added docker-build-streamlit, docker-run-streamlit targets
- `pyproject.toml` - MODIFIED: Added pytest.mark.slow marker registration
- `src/serving/ui/app.py` - MODIFIED: Added Overview + Recommendations pages to navigation
- `src/serving/ui/data_loader.py` - MODIFIED: Added env var config (DATA_SOURCE, S3_BUCKET, AWS_REGION, DYNAMODB_TABLE), load_recommendations()
- `src/serving/ui/pages/overview.py` - NEW: Executive KPI dashboard page
- `src/serving/ui/pages/recommendations.py` - NEW: Retention recommendations page with filters + charts
- `tests/test_pipeline_e2e.py` - MODIFIED: Expanded from 1 to 22 tests (silver, gold, score, full pipeline)
- `tests/test_streamlit_pages.py` - NEW: 8 page render tests for 4 existing pages
- `infra/terraform/environments/dev.tfvars` - MODIFIED: alert_email set to rawad.yared@student.ie.edu
- `infra/terraform/main.tf` - MODIFIED: Added networking, ecs, github_oidc module blocks
- `infra/terraform/variables.tf` - MODIFIED: Added streamlit_cpu/memory/desired_count, github_repo
- `infra/terraform/outputs.tf` - MODIFIED: Added ecr_streamlit_repo_url, streamlit_alb_dns_name, ecs outputs, github_deploy_role_arn
- `infra/terraform/modules/ecr/main.tf` - MODIFIED: Added streamlit ECR repo + lifecycle
- `infra/terraform/modules/ecr/outputs.tf` - MODIFIED: Added streamlit_repo_url, streamlit_repo_arn
- `infra/terraform/modules/iam/main.tf` - MODIFIED: Added ecs_execution + ecs_task roles
- `infra/terraform/modules/iam/outputs.tf` - MODIFIED: Added ecs_execution_role_arn, ecs_task_role_arn
- `infra/terraform/modules/networking/` - NEW: Default VPC, ALB + ECS security groups (3 files)
- `infra/terraform/modules/ecs/` - NEW: Fargate cluster, task def, service, ALB, target group, CW logs (3 files)
- `infra/terraform/modules/github_oidc/` - NEW: OIDC provider + scoped deploy role (3 files)
- `.github/workflows/deploy.yml` - MODIFIED: Added Streamlit Docker build/push + ECS force redeploy
- `README.md` - REWRITTEN: 701-line comprehensive README with architecture diagrams, datasets, setup guide

### Phase 0 (all DONE in prior session):
- `.env.example` - NEW: All environment variables for AWS config
- `configs/__init__.py` - NEW: Package init
- `configs/settings.py` - NEW: Settings dataclass with dotenv loading
- `configs/feature_tiers.yaml` - NEW: 6 feature tier lists + 9 experiment defs (E0-E8)
- `configs/column_registry.yaml` - NEW: Raw-to-standardized column mappings, dtypes, structural fills
- `pyproject.toml` - MODIFIED: Added all runtime + dev dependencies, ruff config
- `.gitignore` - MODIFIED: Added .env, data layers, parquet, terraform state
- `Makefile` - MODIFIED: Added install, lint, test, docker, terraform, streamlit targets
- `.pre-commit-config.yaml` - MODIFIED: Added ruff-pre-commit hooks

### Phase 1 (all DONE in prior session):
- `src/data/ingest.py` - NEW: Raw loading, bronze customer & customer-month tables
- `src/data/silver.py` - NEW: Price imputation, segmentation, channels, margins
- `src/features/build_features.py` - NEW: All 7 feature tiers, gold master builder
- `src/data/build_training_set.py` - NEW: Model matrix builder, structural fills, stratified split
- `src/models/preprocessing.py` - NEW: ColumnTransformer pipeline
- `src/models/churn_model.py` - NEW: Model definitions, threshold optimization, evaluation
- `src/models/scorer.py` - NEW: Batch scoring, risk tier assignment
- `src/reco/schema.py` - NEW: Recommendation dataclass with guardrails
- `src/reco/engine.py` - NEW: Recommendation generation from risk tiers
- `src/monitoring/__init__.py` - NEW: Package init

### Phase 3 (DONE — session 3):
- `src/pipelines/lambda_handler.py` - S3 PutObject handler, DynamoDB check, Step Functions start
- `src/pipelines/manifest.py` - ManifestStore class with DynamoDB conditional writes (bug FIXED in session 3)
- `src/pipelines/s3_io.py` - read/write parquet/csv/json via boto3+pyarrow
- `src/pipelines/steps/__init__.py` - Package init
- `src/pipelines/steps/bronze_step.py` - NEW (session 3): SageMaker Processing entry, reads raw S3, builds bronze
- `src/pipelines/steps/silver_step.py` - NEW (session 3): Reads bronze, runs silver transforms
- `src/pipelines/steps/gold_step.py` - NEW (session 3): Reads silver, builds gold master
- `src/pipelines/steps/score_step.py` - NEW (session 3): Loads champion model, scores customers, assigns risk tiers
- `src/pipelines/run.py` - NEW (session 3): Local pipeline runner (filesystem I/O)
- `Dockerfile.lambda` - NEW (session 3): Lambda container image
- `Dockerfile.processing` - NEW (session 3): SageMaker Processing container image

### Phase 4 (DONE — session 3):
- `src/pipelines/steps/train_step.py` - NEW: Loads gold, trains XGBoost, saves artifacts to S3
- `src/pipelines/steps/evaluate_step.py` - NEW: Evaluates model, compares PR-AUC vs threshold, outputs promote decision
- `src/models/artifacts.py` - NEW: Save/load sklearn pipelines + metadata via joblib+S3
- `src/models/registry.py` - NEW: SageMaker Model Registry wrapper (register, approve, reject, get champion)

### Phase 5 (DONE — session 3):
- `src/monitoring/drift.py` - NEW: KS-test feature + prediction drift detection
- `src/monitoring/data_quality.py` - NEW: Null rates, duplicate keys, schema, numeric ranges
- `src/monitoring/alerts.py` - NEW: SNS publish + CloudWatch metrics
- `src/monitoring/reference_store.py` - NEW: Save/load reference distributions as JSON to S3
- `src/pipelines/steps/drift_step.py` - NEW: Drift detection Processing Job entry point

### Phase 6 (DONE — session 3+4):
- `src/serving/ui/app.py` - NEW: Main Streamlit entry with sidebar navigation
- `src/serving/ui/data_loader.py` - NEW: Cached parquet/JSON loading (local or S3); updated session 4 with load_pipeline_runs
- `src/serving/ui/pages/__init__.py` - NEW: Package init
- `src/serving/ui/pages/model_performance.py` - NEW: PR-AUC, confusion matrix, metrics
- `src/serving/ui/pages/drift_monitor.py` - NEW: Feature drift table + KS bar chart
- `src/serving/ui/pages/customer_risk.py` - NEW: Risk tier pie chart, filterable customer table
- `src/serving/ui/pages/pipeline_status.py` - NEW (session 4): Pipeline run history, status metrics, filterable table

### Phase 7 (DONE — session 4):
- `.github/workflows/ci.yml` - NEW: Lint + test on push/PR, coverage upload
- `.github/workflows/deploy.yml` - NEW: Terraform apply + ECR build/push + Lambda update
- `.github/workflows/retrain.yml` - NEW: Manual/weekly Step Functions trigger with status monitoring

### Phase 2 (DONE — session 4):
- `infra/terraform/main.tf` - NEW: Provider, module orchestration
- `infra/terraform/backend.tf` - NEW: S3 backend for remote state
- `infra/terraform/variables.tf` - NEW: project_name, environment, region, instance types
- `infra/terraform/outputs.tf` - NEW: Bucket, Lambda ARN, SFN ARN, ECR URIs, SNS
- `infra/terraform/environments/{dev,staging,prod}.tfvars` - NEW: Per-environment configs
- `infra/terraform/modules/s3/` - NEW: Single bucket, versioning, encryption, lifecycle
- `infra/terraform/modules/dynamodb/` - NEW: Manifest table, PAY_PER_REQUEST
- `infra/terraform/modules/iam/` - NEW: Lambda, SFN, SageMaker roles + policies
- `infra/terraform/modules/lambda/` - NEW: ECR-based function, S3 notification trigger
- `infra/terraform/modules/step_functions/` - NEW: State machine + ASL definition
- `infra/terraform/modules/step_functions/asl/pipeline.asl.json` - NEW: Full pipeline workflow
- `infra/terraform/modules/sagemaker/` - NEW: Model Package Group
- `infra/terraform/modules/monitoring/` - NEW: SNS topic, CloudWatch alarms
- `infra/terraform/modules/ecr/` - NEW: Lambda + Processing repos with lifecycle

### Tests (ALL DONE — session 4, 93 passing + 1 skipped):
- `tests/test_settings.py` - 5 tests
- `tests/test_ingest.py` - 6 tests
- `tests/test_silver.py` - 8 tests
- `tests/test_build_features.py` - 9 tests
- `tests/test_build_training_set.py` - 5 tests
- `tests/test_models.py` - 7 tests (xgboost test uses pytest.importorskip)
- `tests/test_reco.py` - 7 tests
- `tests/test_lambda_handler.py` - 3 tests (moto: DynamoDB + SFN mocks)
- `tests/test_manifest.py` - 5 tests (moto: DynamoDB mocks)
- `tests/test_s3_io.py` - 5 tests (moto: S3 parquet/json/csv round-trips)
- `tests/test_artifacts.py` - 2 tests (moto: save/load sklearn pipeline to S3)
- `tests/test_drift.py` - 10 tests (KS drift detection, numpy.bool_ comparison via ==)
- `tests/test_data_quality.py` - 7 tests (null rates, duplicates, schema, layer thresholds)
- `tests/test_alerts.py` - 5 tests (moto: SNS + CloudWatch)
- `tests/test_streamlit_data_loader.py` - 8 tests (local file loading, monkeypatched st.cache_data)
- `tests/test_pipeline_e2e.py` - 1 test (bronze smoke test, @pytest.mark.slow)

---

## Decisions Made and Rationale

| Decision | Rationale |
|----------|-----------|
| Terraform over CDK | Universal HCL, easier ops handoff, mature AWS provider |
| SageMaker Processing for ETL | Same container runtime as training, unified ecosystem |
| Single S3 bucket, prefix-based | Simpler management with per-prefix lifecycle |
| KS test for drift detection | Simple, statistically grounded, well-understood |
| Streamlit for dashboard | Python-native, team familiarity |
| GitHub Actions for CI/CD | Tight repo integration |
| pydantic-free Settings (dataclass+dotenv) | Simpler dependency, sufficient for config needs |
| Structural fills: 9999 for null days_ago, "no_interaction" for null categoricals | Preserves signal for tree models without imputation bias |
| Margin excluded from training features | EDA showed not churn-predictive; kept only for expected_monthly_loss |
| Gas conversion: 1 m3 = 11 kWh | Standard Spanish gas conversion factor |
| Risk tiers: Low<40%, Medium 40-60%, High 60-80%, Critical>80% | Per recommendation policy doc |
| Terraform modular structure (8 modules) | Separation of concerns, reusable per environment |
| ASL uses SageMaker Processing for all steps (not Training API) | Uniform container runtime, simpler IAM |
| ECR lifecycle: keep last 10 images | Prevent unbounded image accumulation |
| S3 lifecycle: bronze→STANDARD_IA at 90d, scored expires at 365d | Cost optimization for historical data |
| Deploy workflow uses OIDC role assumption | Avoids long-lived credentials in GitHub secrets |
| Retrain workflow: weekly Monday 06:00 UTC + manual trigger | Regular retraining cadence with on-demand capability |
| Moto tests use `with mock_aws():` context managers per test | Avoids state leaking between tests (no shared fixtures) |
| Drift test assertions use `==` not `is` for booleans | scipy returns numpy.bool_ which fails identity check |
| .gitignore uses `/data/` (leading slash) | Only ignores top-level data dir, not src/data/ |
| Docker `--provenance=false` for Lambda images | Lambda rejects OCI image manifests from Docker Desktop buildx; V2 Schema 2 required |
| SFN role needs EventBridge permissions | `.sync` integrations create managed EventBridge rules to monitor SageMaker job completion |
| Terraform state in separate S3 bucket | `spanishgas-terraform-state` created manually before `terraform init` (chicken-and-egg) |
| AdministratorAccess for IAM user during deploy | Simplest for dev account; should scope down for prod |

---

## Blockers / Open Questions

### FIXED: `manifest.py` duplicate ExpressionAttributeValues
Fixed in session 3 — merged into single dict with all three keys (`:s`, `:t`, `:r`).

### FIXED: xgboost test
Fixed in session 3 — split into two tests: `test_returns_models` (checks logistic_regression + random_forest only) and `test_xgboost_included_when_installed` (uses `pytest.importorskip("xgboost")`).

### FIXED: .gitignore `data/` pattern too broad (session 4)
Pattern `data/` was ignoring `src/data/` as well. Changed to `/data/` to only match root-level data directory.

### FIXED: numpy.bool_ identity comparison in drift tests (session 4)
`assert result["drifted"] is True` fails because scipy returns `numpy.bool_`, not Python `bool`. Changed to `== True`.

### FIXED: SFN IAM missing EventBridge permissions (session 5)
`terraform apply` failed creating Step Functions state machine — SFN role not authorized to create managed-rule. Added `events:PutRule/PutTargets/DescribeRule/DeleteRule/RemoveTargets` scoped to `StepFunctionsGetEventsForSageMakerProcessingJobsRule`.

### FIXED: Lambda OCI image manifest rejected (session 5)
Docker Desktop buildx creates OCI manifests; Lambda requires Docker V2 Schema 2. Rebuilt with `docker build --provenance=false`.

### FIXED: IAM user missing DynamoDB access (session 5)
`terraform plan` failed — `powerco-mlflow-local` user lacked DynamoDB permissions for lock table. Resolved by attaching `AdministratorAccess`.

### PENDING: SageMaker training job quota
`ml.m5.xlarge for training job usage` requested (increase to 1), awaiting approval. Processing quota already approved.

---

## Exact Next Steps (ordered)

All code work complete. Waiting on SageMaker training job quota approval.

1. **Wait for SageMaker training job quota** — `ml.m5.xlarge for training job usage` (requested, pending approval)
2. **Run `terraform apply`** — deploy session 6 infra changes (ECS, networking, OIDC, Streamlit ECR, alert_email)
3. **Build & push Streamlit Docker image** — `docker build --provenance=false -f Dockerfile.streamlit ...` + push to ECR
4. **Set GitHub secret** — `terraform output github_deploy_role_arn` -> set as `AWS_DEPLOY_ROLE_ARN` in repo secrets
5. **Upload raw data** — `aws s3 cp <data>.csv s3://spanishgas-data-dev/raw/` (triggers full pipeline automatically)
6. **Open PR** — `feature/aws-mlops-pipeline` -> `main`

---

## AWS Deployment (session 5)

### Account & Region
- **Account ID:** 559307249592
- **IAM User:** `powerco-mlflow-local` (AdministratorAccess)
- **Region:** eu-west-1

### Terraform State Backend (created manually in console)
- S3 bucket: `spanishgas-terraform-state` (versioning + encryption)
- DynamoDB table: `spanishgas-terraform-locks` (PK: LockID)

### Resources Created (25/25 via terraform apply)
- S3 bucket: `spanishgas-data-dev`
- DynamoDB table: `spanishgas-dev-pipeline-manifest`
- IAM roles: `spanishgas-dev-lambda-role`, `spanishgas-dev-sfn-role`, `spanishgas-dev-sagemaker-role`
- Lambda: `spanishgas-dev-pipeline-trigger` (ECR image, S3 trigger on `raw/` prefix)
- Step Functions: `spanishgas-dev-pipeline` (ASL with .sync SageMaker integrations)
- ECR repos: `spanishgas-dev-lambda`, `spanishgas-dev-processing`
- SageMaker Model Package Group: `spanishgas-dev-churn`
- SNS topic: `spanishgas-dev-alerts`
- CloudWatch alarms: lambda-errors, sfn-failures, drift-detected

### Docker Images Pushed
- `559307249592.dkr.ecr.eu-west-1.amazonaws.com/spanishgas-dev-lambda:latest`
- `559307249592.dkr.ecr.eu-west-1.amazonaws.com/spanishgas-dev-processing:latest`
- Both rebuilt with `--provenance=false` for Docker V2 manifest (Lambda requirement)

### SageMaker Quotas
- `ml.m5.xlarge for processing job usage`: approved (1)
- `ml.m5.xlarge for training job usage`: pending approval (requested 1)

### Files Changed (session 5)
- `.env` - MODIFIED: S3_BUCKET → spanishgas-data-dev, DYNAMODB_MANIFEST_TABLE → spanishgas-dev-pipeline-manifest, real ARNs from terraform output
- `infra/terraform/modules/iam/main.tf` - MODIFIED: Added EventBridge permissions (events:PutRule, PutTargets, DescribeRule, DeleteRule, RemoveTargets) for SFN .sync integrations
- `infra/terraform/.terraform.lock.hcl` - NEW: Provider lock file (AWS provider v5.100.0)

---

## Commands Run + Key Outputs (session 5)

```
terraform init                                        # Backend connected, 8 modules initialized, AWS provider v5.100.0
terraform plan -var-file=environments/dev.tfvars      # 25 to add (first run failed: DynamoDB access denied)
# Fixed: Added AdministratorAccess to IAM user
terraform apply -var-file=environments/dev.tfvars     # 23/25 created, SFN failed (EventBridge permission)
# Fixed: Added events:* permissions to SFN role for managed rules
terraform apply -var-file=environments/dev.tfvars     # 24/25, Lambda failed (OCI manifest unsupported)
# Fixed: Rebuilt with --provenance=false
terraform apply -var-file=environments/dev.tfvars     # 25/25 Apply complete!
docker build --provenance=false -f Dockerfile.lambda  # Built Lambda image
docker build --provenance=false -f Dockerfile.processing  # Built Processing image
docker push <ecr>/spanishgas-dev-lambda:latest        # Pushed
docker push <ecr>/spanishgas-dev-processing:latest    # Pushed
git commit                                            # 00ffbff — IAM fix + TF lock file
```

---

## Commands Run + Key Outputs (session 4)

```
ruff check src/ tests/                # 33 errors -> all auto-fixed with --fix
ruff check src/ tests/                # All checks passed!
python -m pytest tests/ -v --tb=short # 93 passed, 1 skipped (xgboost conditional)
git add <106 files>
git commit                            # be178d2 — 106 files changed, 26,884 insertions
git status                            # clean working tree
```

---

## Tests Run + Results (session 4)

```
93 passed, 1 skipped
SKIPPED: tests/test_models.py::TestModelDefinitions::test_xgboost_included_when_installed
  (xgboost not installed in venv — pytest.importorskip)
```

---

## Pending TODOs

- [x] Fix manifest.py duplicate ExpressionAttributeValues bug (DONE session 3)
- [x] Fix/skip xgboost test (DONE session 3 — pytest.importorskip)
- [x] Write 6 pipeline step files (DONE session 3)
- [x] Write src/pipelines/run.py local runner (DONE session 3)
- [x] Write src/models/registry.py + artifacts.py (DONE session 3)
- [x] Write 4 monitoring modules (DONE session 3)
- [x] Write src/pipelines/steps/drift_step.py (DONE session 3)
- [x] Write Dockerfile.lambda + Dockerfile.processing (DONE session 3)
- [x] Write 5/6 Streamlit UI files (DONE session 3 — missing pipeline_status.py)
- [x] Write src/serving/ui/pages/pipeline_status.py (DONE session 4)
- [x] Write 3 GitHub Actions workflows (DONE session 4 — ci, deploy, retrain)
- [x] Write 32 Terraform files across 8 modules (DONE session 4)
- [x] Write 9 missing test files (DONE session 4 — lambda, manifest, s3_io, artifacts, drift, data_quality, alerts, streamlit, e2e)
- [x] Run full lint + test suite (DONE session 4 — 93 passed, ruff clean)
- [x] Git commit all phases (DONE session 4 — be178d2)
- [x] Deploy infrastructure to AWS via Terraform (DONE session 5 — 25/25 resources)
- [x] Build & push Docker images to ECR (DONE session 5 — both Lambda + Processing)
- [x] Update .env with real ARNs from terraform output (DONE session 5)
- [x] Fix SFN IAM policy for EventBridge managed rules (DONE session 5 — 00ffbff)
- [x] Set alert_email in dev.tfvars (DONE session 6 — rawad.yared@student.ie.edu)
- [x] Set up GitHub OIDC for CI/CD (DONE session 6 — github_oidc Terraform module)
- [x] Expand E2E tests (DONE session 6 — 1 to 22 tests)
- [x] Register pytest.mark.slow (DONE session 6)
- [x] Add Streamlit auto-deployment (DONE session 6 — Dockerfile, ECS, ALB, networking, ECR, deploy.yml)
- [x] Add Overview + Recommendations pages (DONE session 6)
- [x] Add Streamlit page render tests (DONE session 6 — 8 tests)
- [x] Add env var config for Streamlit (DONE session 6 — DATA_SOURCE, S3_BUCKET, AWS_REGION, DYNAMODB_MANIFEST_TABLE)
- [x] Revamp README (DONE session 6 — 701 lines, full architecture diagrams)
- [ ] Wait for SageMaker training job quota approval
- [ ] Upload raw data to s3://spanishgas-data-dev/raw/
- [ ] Open PR: feature/aws-mlops-pipeline -> main

---

## Assumptions

- Python 3.12 target runtime
- AWS region: eu-west-1 (default)
- XGBoost E5 experiment is the champion model (full feature set excluding interaction strings)
- Threshold optimization: maximize precision at recall >= 70%
- Risk tiers: Low<40%, Medium 40-60%, High 60-80%, Critical>80%
- S3 layout: single bucket with raw/bronze/silver/gold/models/scored prefixes
- DynamoDB manifest table PK: file_key
- Step Functions ASL: BronzeETL -> SilverETL -> GoldETL -> TrainOrScore -> ... -> DriftCheck -> UpdateManifest
- Promotion gate: PR-AUC >= 0.70
- Spanish public holidays: {101, 106, 501, 815, 1012, 1101, 1206, 1208, 1225}
- Gas kWh conversion: 1 m3 = 11 kWh
- Customer segments: Residential (is_industrial=0), SME (is_industrial=1, power<=10kW), Corporate (is_industrial=1, power>10kW)

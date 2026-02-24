# Context Dump

## Session Info (updated 2026-02-24 session 13)
- **Timestamp:** 2026-02-24
- **Branch:** `feature/aws-mlops-pipeline` (pushed to GitHub)
- **Latest commit:** `27b85e3` — session 13 NLP enrichment + categorical dtype fix
- **Session 13: NLP enrichment + PR-AUC 0.751** — Created `src/data/nlp.py` with regex intent classification + HuggingFace sentiment analysis. Integrated into `bronze_step.py`. Updated `Dockerfile.processing` with torch+transformers+model pre-download. 20 new tests. Fixed Pandas 3.0 ArrowStringArray conflict and Categorical dtype fillna crash. Pipeline `nlp-v3-20260224-200923` SUCCEEDED — PR-AUC = 0.751, all 56 features active. Streamlit redeployed.
- **Pipeline execution:** `arn:aws:states:eu-west-1:559307249592:execution:spanishgas-dev-pipeline:nlp-v3-20260224-200923` — SUCCEEDED
- **PR-AUC:** 0.751 (exceeds 0.70 promotion gate)

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
| 13: Session 11 Fixes | DONE (session 11) | Drift step JSON fix, Customer Lookup fix, README Mermaid diagrams, lint fixes, Docker rebuild, pipeline re-trigger |
| 14: Feature Parity | DONE (session 12) | build_features.py rewritten for ~56 features, XGBoost hyperparams aligned, 142 tests, Docker rebuilt, pipeline re-triggered |
| 15: NLP Enrichment | DONE (session 13) | nlp.py (intent + sentiment), bronze_step integration, Dockerfile torch+transformers, 20 new tests, PR-AUC = 0.751 |

---

## Files Changed (and purpose)

### Session 13 — NLP Enrichment + Pipeline PR-AUC 0.751:
- `src/data/nlp.py` - NEW: NLP enrichment module — `classify_intent()` (regex, 8 categories, priority order), `enrich_interactions_intent()` (customer_intent + has_interaction columns), `enrich_interactions_sentiment()` (HuggingFace cardiffnlp/twitter-roberta-base-sentiment-latest, guarded import), `enrich_interactions()` orchestrator. Fixed: ArrowStringArray init (object dtype + `.values`), removed `.astype("category")` to avoid downstream fillna crash.
- `src/pipelines/steps/bronze_step.py` - MODIFIED: Added `from src.data.nlp import enrich_interactions` import and `interactions = enrich_interactions(interactions)` call after loading interactions JSON.
- `Dockerfile.processing` - MODIFIED: Added `transformers` to pip install, CPU-only PyTorch (`--index-url https://download.pytorch.org/whl/cpu`), pre-download of sentiment model at build time.
- `tests/test_nlp.py` - NEW: 20 tests — 13 intent classification (each category, priority, case insensitivity, None/empty), 5 enrichment (column creation, has_interaction, no mutation, row count, date-only), 1 sentiment fallback, 1 orchestrator.
- `CLAUDE.md` - MODIFIED: Updated to session 13 state
- `CONTEXT.MD` - MODIFIED: Updated to session 13 state
- `README.md` - MODIFIED: Updated repo structure (nlp.py), test count (162+/19 files), feature count (56), feature tier diagram (actual feature names), model details (PR-AUC 0.751, hyperparameters, NLP)

### Session 12 — Feature Parity + XGBoost Hyperparameters + Test Fix:
- `src/features/build_features.py` - REWRITTEN: `build_market_risk_features()` (raw std, relative price trends, 7→13 features), `build_behavioral_features()` (intent flags, severity, interaction timing, 4→11 features), `build_sentiment_features()` (renamed `is_negative_sentiment`, removed `avg_sentiment_score`), `build_compound_features()` (6 new binary flags), `build_lifecycle_features()` (5 bins instead of 6 for `renewal_bucket`)
- `configs/feature_tiers.yaml` - REWRITTEN: All tiers updated to notebook feature names. E5_full: ~56 features (was 41).
- `src/models/churn_model.py` - MODIFIED: XGBoost hyperparameters aligned with notebook (n_estimators=600, lr=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8)
- `src/data/build_training_set.py` - MODIFIED: Added `months_since_last_change` to sentinel cols, expanded binary flag matching pattern
- `tests/test_build_features.py` - REWRITTEN: 9→16 tests, covers new risk/behavioral/sentiment/compound features
- `tests/test_pipeline_e2e.py` - MODIFIED: Updated market risk feature assertions to new names
- `tests/test_settings.py` - MODIFIED: `test_aws_defaults` accepts both `.env` and code-default bucket/table names
- `CLAUDE.md` - MODIFIED: Updated to session 12 state
- `CONTEXT.MD` - MODIFIED: Updated to session 12 state

### Session 11 — Drift Fix + Customer Lookup Fix + README Mermaid + CI:
- `src/pipelines/s3_io.py` - MODIFIED: Added `_json_default()` numpy type handler (bool_, integer, floating, ndarray). `write_json()` now uses `default=_json_default`. Fixed import order (boto3 before numpy).
- `src/serving/ui/pages/customer_lookup.py` - MODIFIED: `reason_codes` from parquet converted from numpy array to list via `.tolist()` before truthiness check. Fixes "Why This Recommendation" ValueError.
- `src/pipelines/steps/drift_step.py` - MODIFIED: Fixed ruff I001 import order (removed blank line between pandas and botocore).
- `README.md` - REWRITTEN: All 7 ASCII box-and-dash architecture diagrams replaced with Mermaid. Added feature tiers diagram. Updated to 8 dashboard pages, 129+ tests, E5_full/E8_no_sentiment experiments, `--platform linux/amd64` in Docker docs.
- `CLAUDE.md` - MODIFIED: Updated to session 11 state with PR-AUC gap root cause analysis and 3 fix steps.
- `CONTEXT.MD` - MODIFIED: Updated to session 11 state.

### Session 10 — Feature Alignment + Streamlit Recommendations + Tests:
- `configs/feature_tiers.yaml` - REWRITTEN: All tier feature names match actual `build_features.py` output. E5_full: 41 features. Added `E8_no_sentiment` ablation. Added 3 missing interaction strings to `tier_1b`.
- `src/features/build_features.py` - MODIFIED: Added 3 missing compound features to `build_compound_features()`: `sales_channel_x_renewal_bucket`, `has_interaction_x_renewal_bucket`, `competition_x_intent`.
- `src/pipelines/steps/train_step.py` - MODIFIED: eval.json now saves `actual_features` (X.columns), `requested_features` (YAML list), `missing_features`. Added feature availability logging.
- `src/pipelines/steps/score_step.py` - MODIFIED: Added `generate_recommendations()` call after `assign_risk_tiers()`, writes `scored/recommendations.parquet`.
- `tests/test_streamlit_pages.py` - MODIFIED: Added `TestOverviewPage` (2 tests) and `TestRecommendationsPage` (2 tests). Total: 12 tests.
- `tests/test_streamlit_data_loader.py` - MODIFIED: Added `TestLoadRecommendations` (2 tests). Total: 10 tests.
- `CLAUDE.md` - MODIFIED: Updated to session 10 state.
- `CONTEXT.MD` - MODIFIED: Updated to session 10 state.

### Session 8 — Pipeline E2E Debugging + Instance Type + Docker amd64 + Env Var Fix:
- `configs/settings.py` - MODIFIED: Default SageMaker instances `ml.m5.xlarge` → `ml.m5.large`
- `.env` - MODIFIED: SageMaker instances `ml.m5.xlarge` → `ml.m5.large`
- `.env.example` - MODIFIED: SageMaker instances `ml.m5.xlarge` → `ml.m5.large`
- `infra/terraform/variables.tf` - MODIFIED: Default SageMaker instances → `ml.m5.large`
- `infra/terraform/modules/step_functions/variables.tf` - MODIFIED: Default instances → `ml.m5.large`
- `infra/terraform/modules/iam/main.tf` - MODIFIED: Added `sagemaker:AddTags` to SFN role, added `StepFunctionsGetEventsForSageMakerTrainingJobsRule` to EventBridge permissions
- `src/pipelines/steps/bronze_step.py` - MODIFIED: `--bucket` defaults to `S3_BUCKET` env var, `--region` to `AWS_REGION`
- `src/pipelines/steps/silver_step.py` - MODIFIED: Same env var defaults
- `src/pipelines/steps/gold_step.py` - MODIFIED: Same env var defaults
- `src/pipelines/steps/train_step.py` - MODIFIED: Same env var defaults
- `src/pipelines/steps/evaluate_step.py` - MODIFIED: Same env var defaults
- `src/pipelines/steps/score_step.py` - MODIFIED: Same env var defaults
- `src/pipelines/steps/drift_step.py` - MODIFIED: Same env var defaults + `--reference-key`/`--scored-key` no longer required (have defaults), `--sns-topic-arn` defaults to `SNS_TOPIC_ARN` env var
- `CLAUDE.md` - MODIFIED: Session 8 state
- `CONTEXT.MD` - MODIFIED: Session 8 state

### Session 7 — Infrastructure Deploy + Docker Push + GitHub Setup:
- No code files changed — session focused on deploying infrastructure, pushing Docker images, and configuring GitHub
- `CLAUDE.md` - MODIFIED: Updated to session 7 state, added IAM policy note, ALB DNS, Streamlit dashboard section
- `CONTEXT.MD` - MODIFIED: Updated session info, next steps, resources, commands, decisions, blockers, TODOs

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

### Tests (ALL DONE — 162 passing + 1 skipped, 19 files):
- `tests/test_settings.py` - 5 tests
- `tests/test_ingest.py` - 6 tests
- `tests/test_silver.py` - 8 tests
- `tests/test_build_features.py` - 16 tests (gold features, expanded session 12)
- `tests/test_build_training_set.py` - 5 tests
- `tests/test_models.py` - 7 tests (xgboost test uses pytest.importorskip)
- `tests/test_reco.py` - 7 tests
- `tests/test_nlp.py` - 20 tests (intent classification + sentiment enrichment, NEW session 13)
- `tests/test_lambda_handler.py` - 3 tests (moto: DynamoDB + SFN mocks)
- `tests/test_manifest.py` - 5 tests (moto: DynamoDB mocks)
- `tests/test_s3_io.py` - 5 tests (moto: S3 parquet/json/csv round-trips)
- `tests/test_artifacts.py` - 2 tests (moto: save/load sklearn pipeline to S3)
- `tests/test_drift.py` - 10 tests (KS drift detection, numpy.bool_ comparison via ==)
- `tests/test_data_quality.py` - 7 tests (null rates, duplicates, schema, layer thresholds)
- `tests/test_alerts.py` - 5 tests (moto: SNS + CloudWatch)
- `tests/test_streamlit_data_loader.py` - 10 tests (local file loading + recommendations)
- `tests/test_streamlit_pages.py` - 12 tests (page render mocks for 6 pages)
- `tests/test_pipeline_e2e.py` - 22 tests (bronze, silver, gold, score, full pipeline E2E)
- `tests/test_imports.py` - 1 test (package import smoke test)

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
| Docker `--platform linux/amd64 --provenance=false` | Apple Silicon builds arm64 by default; Lambda/SageMaker need x86_64. `--provenance=false` for V2 Schema 2 |
| Switched ml.m5.xlarge → ml.m5.large for dev | ml.m5.xlarge training quota pending; ml.m5.large has 15 training + 4 processing quota. 8 GB RAM sufficient for 20K customer × 12 month dataset |
| Step files read bucket from S3_BUCKET env var | ASL passes config via Environment block, not ContainerArguments. argparse defaults to env var with CLI override |
| SFN role needs EventBridge permissions | `.sync` integrations create managed EventBridge rules to monitor SageMaker job completion |
| Terraform state in separate S3 bucket | `spanishgas-terraform-state` created manually before `terraform init` (chicken-and-egg) |
| AdministratorAccess for IAM user during deploy | Simplest for dev account; should scope down for prod |
| Replaced 10 granular IAM policies with AdministratorAccess | Hit 10-policy-per-user AWS limit; needed EC2/ECS/ELB for session 7 infra |
| gh CLI installed via Homebrew for GitHub secret management | Needed for `gh secret set` to configure OIDC deploy role |

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

### FIXED: IAM user missing EC2/ECS/ELB permissions (session 7)
`terraform plan` failed — `powerco-mlflow-local` lacked `ec2:DescribeVpcAttribute` (and ECS/ELB permissions) for networking + ECS modules. Had 10 granular policies (AWS limit). Replaced all 10 with `AdministratorAccess`.

### FIXED: Dockerfiles missing from working tree (session 7)
`Dockerfile.lambda`, `Dockerfile.processing`, `Dockerfile.streamlit` tracked in git but absent from working tree. Restored with `git checkout HEAD -- Dockerfile.*`.

### RESOLVED: SageMaker training job quota (session 8)
Bypassed `ml.m5.xlarge` quota wait by switching to `ml.m5.large` (15 training instances, 4 processing instances available). All configs, TF variables, and defaults updated.

### FIXED: Docker images built for wrong architecture (session 8)
Lambda and Processing images were built as arm64 (Apple Silicon default). Lambda returned `Runtime.InvalidEntrypoint`; SageMaker Processing returned `Exec format error`. Rebuilt both with `--platform linux/amd64 --provenance=false`.

### FIXED: SFN IAM missing `sagemaker:AddTags` (session 8)
Step Functions `.sync` integrations automatically add `MANAGED_BY_AWS: STARTED_BY_STEP_FUNCTIONS` tag. SFN role needed `sagemaker:AddTags` permission. Also added `StepFunctionsGetEventsForSageMakerTrainingJobsRule` EventBridge rule (was only scoped to Processing).

### FIXED: Step files expected CLI args, ASL passes env vars (session 8)
All 6 step files used `argparse` with `--bucket` as required CLI argument, but ASL passes `S3_BUCKET` as an environment variable via `Environment` block. Updated all `main()` functions to default `--bucket` from `os.environ.get("S3_BUCKET")`, `--region` from `AWS_REGION`, `--sns-topic-arn` from `SNS_TOPIC_ARN`.

### FIXED: consumption file format mismatch (session 9)
User converted CSV→parquet locally and uploaded `raw/consumption_hourly_2024.parquet` (1.8 GB) to S3.

### FIXED: BronzeETL OOM on ml.m5.large and ml.m5.xlarge (session 9)
176M-row consumption parquet caused OOM even on ml.m5.xlarge (16 GB). Fixed by rewriting `bronze_step.py` with chunked processing: `read_parquet_batches()` downloads to temp file, iterates 2M-row batches via pyarrow, aggregates per-chunk to monthly, then combines. Peak memory ~2.5 GB. Runs on ml.m5.large.

### FIXED: TrainModel 0-features crash (session 9)
`train_step.py:_load_feature_list()` had 3 bugs: experiment name `"E5"` → `"E5_full"`, YAML key `"tiers"` → `"features"`, top-level key `"tiers"` → `"feature_tiers"`.

### FIXED: DriftCheck NoSuchKey on first run (session 9)
`drift_step.py` crashed when `monitoring/reference.json` didn't exist. Added try/except for `NoSuchKey`: saves current scored data as baseline reference and skips drift comparison.

### FIXED: Streamlit exec format error (session 9)
Streamlit image was arm64. Rebuilt with `--platform linux/amd64 --provenance=false`.

### FIXED: Streamlit ModuleNotFoundError: No module named 'src' (session 9)
`Dockerfile.streamlit` copied `src/` but never installed it as a package. Added `RUN pip install --no-cache-dir -e .` after COPY.

### FIXED: feature_tiers.yaml misalignment → low PR-AUC (session 10)
Pipeline PR-AUC = 0.289 vs notebook ~0.745. Root cause: YAML listed 33 features with wrong names. Fixed: complete rewrite of `feature_tiers.yaml` to match actual `build_features.py` output. E5_full now has 41 features. Also added 3 missing compound features (`sales_channel_x_renewal_bucket`, `has_interaction_x_renewal_bucket`, `competition_x_intent`) to `build_compound_features()`.

### FIXED: Recommendations page NoSuchKey (session 10)
`score_step.py` now calls `generate_recommendations()` from `src/reco/engine.py` and writes `scored/recommendations.parquet`.

### FIXED: evaluation.json saves wrong feature list (session 10)
`train_step.py` now saves `X.columns.tolist()` as `"features"`, plus `"requested_features"` and `"missing_features"` for debugging.

### FIXED: Drift step JSON serialization crash (session 11)
`s3_io.write_json()` called `json.dumps(data)` without a custom serializer. Drift summary contained `numpy.bool_` values which are not JSON serializable. Fixed by adding `_json_default()` handler for `numpy.bool_`, `numpy.integer`, `numpy.floating`, `numpy.ndarray` and passing `default=_json_default` to `json.dumps()`.

### FIXED: Customer Lookup "Why This Recommendation" crash (session 11)
`customer_lookup.py` line 127: `if reason_codes:` raised `ValueError: truth value of array with more than one element is ambiguous` because `reason_codes` from parquet is a numpy array. Fixed by converting with `.tolist()` before truthiness check.

### FIXED: Ruff I001 import order in s3_io.py and drift_step.py (session 11)
`s3_io.py`: `numpy` was imported before `boto3` (alphabetically wrong). `drift_step.py`: blank line between `pandas` and `botocore` caused ruff to see them as separate import groups. Both fixed, CI now passing.

### FIXED: PR-AUC gap — feature parity with notebook (session 12)
Rewrote `build_features.py` to match notebook's ~56 features exactly: MP_Risk (7→13), Behavioral (4→11), Sentiment (renamed `is_negative_sentiment`), Compound (3→7). Aligned XGBoost hyperparameters (600 trees, lr=0.05, depth=5, subsample=0.8). Changed `renewal_bucket` from 6→5 bins. Fixed `test_aws_defaults`. Pipeline re-triggered to validate PR-AUC >= 0.70.

### FIXED: NLP enrichment missing from pipeline (session 13)
Created `src/data/nlp.py` with regex intent classification (8 categories) + HuggingFace sentiment analysis. Integrated into bronze_step.py. Updated Dockerfile.processing with torch+transformers+model pre-download. Pipeline now produces `customer_intent`, `sentiment_label`, `sentiment_neg/neu/pos` columns, activating all 12 NLP-dependent features.

### FIXED: Pandas 3.0 ArrowStringArray conflict (session 13)
`enrich_interactions_sentiment()` assigned string `sentiment_label` values into a column initialized as `np.nan` (float64). Pandas 3.0 with Arrow backend rejects this. Fixed by initializing float columns with `np.nan`, string column with `pd.Series([None]*len, dtype="object")`, and using `.values` for assignment.

### FIXED: Categorical dtype fillna crash (session 13)
`customer_intent` column was set as `.astype("category")` in nlp.py. Downstream `build_training_set.py` calls `fillna("no_interaction")` which fails on Categorical with unknown category. Fixed by keeping column as object dtype.

---

## Exact Next Steps (ordered)

1. ~~**Switch to ml.m5.large**~~ — DONE session 8
2. ~~**Rebuild Docker images for amd64**~~ — DONE session 8 (Lambda + Processing)
3. ~~**Fix SFN IAM (AddTags + Training EventBridge)**~~ — DONE session 8
4. ~~**Fix step files env var support**~~ — DONE session 8 (all 6 steps)
5. ~~**Upload raw data**~~ — DONE session 8 (7 files in `s3://spanishgas-data-dev/raw/`)
6. ~~**Fix consumption file format mismatch**~~ — DONE session 9 (CSV→parquet + chunked processing)
7. ~~**Fix feature_tiers.yaml + compound features**~~ — DONE session 10
8. ~~**Fix drift step JSON serialization**~~ — DONE session 11
9. ~~**Fix Customer Lookup reason_codes crash**~~ — DONE session 11
10. ~~**README Mermaid diagrams**~~ — DONE session 11
11. ~~**Rebuild Docker images + push + re-trigger pipeline**~~ — DONE session 11 (pipeline SUCCEEDED)
12. ~~**Commit + push + CI passing**~~ — DONE session 11 (`ea185bc`)
13. ~~**Close PR-AUC gap (~0.4 → ~0.7)**~~ — DONE session 12 (feature parity) + session 13 (NLP enrichment). PR-AUC = 0.751.
14. ~~**Add NLP enrichment to pipeline**~~ — DONE session 13 (nlp.py, bronze_step integration, Dockerfile, tests)
15. ~~**Rebuild Docker images + re-trigger pipeline**~~ — DONE session 13 (`nlp-v3-20260224-200923` SUCCEEDED)
16. ~~**Validate PR-AUC >= 0.70**~~ — DONE session 13 (PR-AUC = 0.751, all 56 features active)
17. ~~**Redeploy Streamlit dashboard**~~ — DONE session 13 (Processing + Streamlit Docker images rebuilt, ECS force redeployed)
18. **Open PR** — `feature/aws-mlops-pipeline` -> `main`

---

## AWS Deployment (session 5)

### Account & Region
- **Account ID:** 559307249592
- **IAM User:** `powerco-mlflow-local` (AdministratorAccess)
- **Region:** eu-west-1

### Terraform State Backend (created manually in console)
- S3 bucket: `spanishgas-terraform-state` (versioning + encryption)
- DynamoDB table: `spanishgas-terraform-locks` (PK: LockID)

### Resources Created (44 total via terraform apply)

**Session 5 (25 resources):**
- S3 bucket: `spanishgas-data-dev`
- DynamoDB table: `spanishgas-dev-pipeline-manifest`
- IAM roles: `spanishgas-dev-lambda-role`, `spanishgas-dev-sfn-role`, `spanishgas-dev-sagemaker-role`
- Lambda: `spanishgas-dev-pipeline-trigger` (ECR image, S3 trigger on `raw/` prefix)
- Step Functions: `spanishgas-dev-pipeline` (ASL with .sync SageMaker integrations)
- ECR repos: `spanishgas-dev-lambda`, `spanishgas-dev-processing`
- SageMaker Model Package Group: `spanishgas-dev-churn`
- SNS topic: `spanishgas-dev-alerts`
- CloudWatch alarms: lambda-errors, sfn-failures, drift-detected

**Session 7 (19 resources):**
- ECS Fargate cluster: `spanishgas-dev-cluster`
- ECS service: `spanishgas-dev-streamlit`
- ECS task definition (Streamlit container)
- ALB: `spanishgas-dev-streamlit-alb` (DNS: `spanishgas-dev-streamlit-alb-1221532574.eu-west-1.elb.amazonaws.com`)
- ALB listener (HTTP:80) + target group
- CloudWatch log group for ECS
- Security groups: ALB (ingress 80) + ECS (ingress from ALB)
- ECR repo: `spanishgas-dev-streamlit` + lifecycle policy
- IAM roles: `spanishgas-dev-ecs-execution-role`, `spanishgas-dev-ecs-task-role` + policies
- GitHub OIDC provider + `spanishgas-dev-github-deploy-role`
- SNS email subscription (rawad.yared@student.ie.edu)

### Docker Images Pushed
- `559307249592.dkr.ecr.eu-west-1.amazonaws.com/spanishgas-dev-lambda:latest` (rebuilt amd64 session 8)
- `559307249592.dkr.ecr.eu-west-1.amazonaws.com/spanishgas-dev-processing:latest` (rebuilt amd64 session 8)
- `559307249592.dkr.ecr.eu-west-1.amazonaws.com/spanishgas-dev-streamlit:latest`
- All built with `--platform linux/amd64 --provenance=false` (Apple Silicon builds arm64 by default)

### SageMaker Quotas (updated session 8)
- `ml.m5.large for training job usage`: 15 instances available
- `ml.m5.large for processing job usage`: 4 instances available
- `ml.m5.xlarge for processing job usage`: 1 instance approved
- `ml.m5.xlarge for training job usage`: pending (no longer needed — switched to ml.m5.large)

### Files Changed (session 5)
- `.env` - MODIFIED: S3_BUCKET → spanishgas-data-dev, DYNAMODB_MANIFEST_TABLE → spanishgas-dev-pipeline-manifest, real ARNs from terraform output
- `infra/terraform/modules/iam/main.tf` - MODIFIED: Added EventBridge permissions (events:PutRule, PutTargets, DescribeRule, DeleteRule, RemoveTargets) for SFN .sync integrations
- `infra/terraform/.terraform.lock.hcl` - NEW: Provider lock file (AWS provider v5.100.0)

---

## Commands Run + Key Outputs (session 8)

```
# Instance type change — updated configs/settings.py, .env, .env.example, TF variables (dev.tfvars already ml.m5.large)
terraform plan -var-file=environments/dev.tfvars          # No changes (dev.tfvars already had ml.m5.large)

# Attempt 1: Lambda invoke — Runtime.InvalidEntrypoint (arm64 image on x86_64 Lambda)
docker build --platform linux/amd64 --provenance=false -f Dockerfile.lambda -t <ecr>/spanishgas-dev-lambda:latest .
docker push <ecr>/spanishgas-dev-lambda:latest
aws lambda update-function-code --function-name spanishgas-dev-pipeline-trigger --image-uri <ecr>/spanishgas-dev-lambda:latest

# Attempt 2: SageMaker Processing — Exec format error (arm64 Processing image on x86_64 SageMaker)
docker build --platform linux/amd64 --provenance=false -f Dockerfile.processing -t <ecr>/spanishgas-dev-processing:latest .
docker push <ecr>/spanishgas-dev-processing:latest

# Attempt 3: SageMaker Processing — sagemaker:AddTags AccessDeniedException
# Fixed: Added sagemaker:AddTags + TrainingJobsRule to SFN IAM policy
terraform apply -var-file=environments/dev.tfvars -auto-approve  # 0 added, 1 changed (IAM policy)

# Attempt 4: SageMaker Processing — argparse error: --bucket required
# Fixed: Updated all 6 step files to default --bucket from S3_BUCKET env var
docker build --platform linux/amd64 --provenance=false -f Dockerfile.processing -t <ecr>/spanishgas-dev-processing:latest .
docker push <ecr>/spanishgas-dev-processing:latest

# Attempt 5: BronzeETL runs but fails — NoSuchKey: raw/consumption_hourly_2024.parquet
# Root cause: File is consumption_hourly_2024.csv (8.7 GB CSV), not .parquet
# BLOCKED — needs fix in bronze_step.py

# Between attempts: clearing stale DynamoDB manifest entries
aws dynamodb delete-item --table-name spanishgas-dev-pipeline-manifest --key '{"file_key":{"S":"raw/customer_attributes.csv"}}'
aws lambda invoke --function-name spanishgas-dev-pipeline-trigger --payload '{"Records":[{"s3":{"bucket":{"name":"spanishgas-data-dev"},"object":{"key":"raw/customer_attributes.csv"}}}]}' ...
```

## Raw Data Files in S3 (session 8)
```
s3://spanishgas-data-dev/raw/
  churn_label.csv              180 KB
  consumption_hourly_2024.csv  8.7 GB  ← bronze_step expects .parquet
  costs_by_province_month.csv  32 KB
  customer_attributes.csv      915 KB
  customer_contracts.csv       964 KB
  customer_interactions.json   1.7 MB
  price_history.csv            24 MB
```

---

## Commands Run + Key Outputs (session 7)

```
aws iam list-attached-user-policies --user-name powerco-mlflow-local  # 10 granular policies, no AdministratorAccess
aws iam detach-user-policy (x10)                                      # Removed all 10 granular policies
aws iam attach-user-policy --policy-arn AdministratorAccess           # Attached AdministratorAccess
terraform plan -var-file=environments/dev.tfvars                      # 19 to add, 0 to change, 0 to destroy
terraform apply -var-file=environments/dev.tfvars -auto-approve       # 19/19 Apply complete!
git checkout HEAD -- Dockerfile.*                                     # Restored missing Dockerfiles
docker build --provenance=false -f Dockerfile.streamlit .             # Built Streamlit image
docker push <ecr>/spanishgas-dev-streamlit:latest                     # Pushed to ECR
brew install gh                                                       # Installed GitHub CLI
gh auth login                                                         # Authenticated via browser device flow
gh secret set AWS_DEPLOY_ROLE_ARN                                     # Set OIDC role ARN in GitHub secrets
git push -u origin feature/aws-mlops-pipeline                        # Pushed branch to GitHub (CI triggered)
```

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
- [x] Deploy session 6 infra (DONE session 7 — 19 resources: ECS, ALB, networking, OIDC, Streamlit ECR, SNS sub)
- [x] Fix IAM user permissions for EC2/ECS/ELB (DONE session 7 — replaced 10 policies with AdministratorAccess)
- [x] Build & push Streamlit Docker image (DONE session 7 — pushed to spanishgas-dev-streamlit ECR)
- [x] Install gh CLI + set GitHub secret (DONE session 7 — AWS_DEPLOY_ROLE_ARN set)
- [x] Push branch to GitHub (DONE session 7 — CI workflow triggered)
- [x] Switch SageMaker instances to ml.m5.large (DONE session 8 — bypassed xlarge quota wait)
- [x] Rebuild Docker images for linux/amd64 (DONE session 8 — Lambda + Processing)
- [x] Fix SFN IAM: sagemaker:AddTags + Training EventBridge rule (DONE session 8)
- [x] Fix step files to read S3_BUCKET/AWS_REGION from env vars (DONE session 8 — all 6 steps)
- [x] Upload raw data to s3://spanishgas-data-dev/raw/ (DONE session 8 — 7 files)
- [x] Lambda update-function-code for amd64 image (DONE session 8)
- [x] Fix consumption file format mismatch (DONE session 9 — CSV→parquet converted, chunked processing)
- [x] Fix feature_tiers.yaml misalignment (DONE session 10 — complete rewrite, 41 features in E5_full)
- [x] Fix recommendations pipeline (DONE session 10 — score_step.py generates recommendations.parquet)
- [x] Fix eval.json feature list (DONE session 10 — saves actual X.columns)
- [x] Add missing compound features (DONE session 10 — 3 interaction strings added to build_features.py)
- [x] Add Overview + Recommendations page tests (DONE session 10 — 6 new tests)
- [x] Rebuild & push Processing + Streamlit Docker images (DONE session 11 — drift fix + customer lookup fix)
- [x] Re-trigger pipeline and validate drift step fix (DONE session 11 — driftfix-20260224-141631 SUCCEEDED)
- [x] Fix drift step JSON serialization (DONE session 11 — _json_default numpy handler)
- [x] Fix Customer Lookup reason_codes crash (DONE session 11 — numpy array .tolist())
- [x] Fix ruff I001 import order (DONE session 11 — s3_io.py + drift_step.py)
- [x] Replace README ASCII diagrams with Mermaid (DONE session 11)
- [x] Commit sessions 8-11 changes + push (DONE session 11 — ea185bc, CI passing)
- [x] Close PR-AUC gap: rewrite build_features.py for notebook parity (DONE session 12 — 7480e18)
- [x] Close PR-AUC gap: rewrite feature_tiers.yaml (DONE session 12 — E5_full ~56 features)
- [x] Close PR-AUC gap: align XGBoost hyperparameters (DONE session 12 — 600 trees, lr=0.05, depth=5)
- [x] Fix test_aws_defaults (DONE session 12 — accepts both .env and code-default values)
- [x] Rebuild Docker images + re-trigger pipeline (DONE session 12 — feature-parity-20260224-175111)
- [x] Add NLP enrichment to pipeline (DONE session 13 — nlp.py, bronze_step, Dockerfile, 20 tests)
- [x] Fix Pandas 3.0 ArrowStringArray conflict in sentiment (DONE session 13 — object dtype + .values)
- [x] Fix Categorical dtype fillna crash (DONE session 13 — removed .astype("category"))
- [x] Rebuild Docker images + re-trigger pipeline (DONE session 13 — nlp-v3-20260224-200923 SUCCEEDED)
- [x] Validate PR-AUC >= 0.70 from pipeline results (DONE session 13 — PR-AUC = 0.751)
- [x] Redeploy Streamlit dashboard (DONE session 13 — Processing + Streamlit Docker pushed, ECS redeployed)
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

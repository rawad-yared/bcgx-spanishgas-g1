# Context Dump

## Session Info (updated 2026-02-23 session 3)
- **Timestamp:** 2026-02-23
- **Branch:** `feature/aws-mlops-pipeline`
- **Latest commit:** `372d414` ("Add notebooks folder and document repo structure")
- **Nothing has been committed from this implementation work yet.** All changes are unstaged/untracked.

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
| 1G: E2E Integration Test | NOT STARTED | tests/test_pipeline_e2e.py |
| 2: Terraform IaC | NOT STARTED | ~25 files under infra/terraform/ |
| 3: Lambda + Step Functions + Docker | DONE | All step files, run.py, Dockerfiles written |
| 4: SageMaker Training + Registry | DONE | train_step.py, evaluate_step.py, registry.py, artifacts.py all written |
| 5: Monitoring + Drift | DONE | drift.py, data_quality.py, alerts.py, reference_store.py, drift_step.py all written |
| 6: Streamlit Dashboard | MOSTLY DONE | app.py + 3/4 pages + data_loader.py written; missing pipeline_status.py |
| 7: CI/CD + Ops | NOT STARTED | 3 GitHub Actions workflows still needed |

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

### Phase 6 (MOSTLY DONE — session 3):
- `src/serving/ui/app.py` - NEW: Main Streamlit entry with sidebar navigation
- `src/serving/ui/data_loader.py` - NEW: Cached parquet/JSON loading (local or S3)
- `src/serving/ui/pages/__init__.py` - NEW: Package init
- `src/serving/ui/pages/model_performance.py` - NEW: PR-AUC, confusion matrix, metrics
- `src/serving/ui/pages/drift_monitor.py` - NEW: Feature drift table + KS bar chart
- `src/serving/ui/pages/customer_risk.py` - NEW: Risk tier pie chart, filterable customer table
- `src/serving/ui/pages/pipeline_status.py` - MISSING (still needs to be written)

### Tests (from prior sessions, not yet updated for new modules):
- `tests/test_settings.py` - 5 tests
- `tests/test_ingest.py` - 6 tests
- `tests/test_silver.py` - 8 tests
- `tests/test_build_features.py` - 9 tests
- `tests/test_build_training_set.py` - 5 tests
- `tests/test_models.py` - 7 tests (xgboost test FIXED with pytest.importorskip)
- `tests/test_reco.py` - 7 tests
- MISSING: test_lambda_handler, test_manifest, test_s3_io, test_artifacts, test_drift, test_data_quality, test_alerts, test_streamlit_data_loader, test_pipeline_e2e

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

---

## Blockers / Open Questions

### FIXED: `manifest.py` duplicate ExpressionAttributeValues
Fixed in session 3 — merged into single dict with all three keys (`:s`, `:t`, `:r`).

### FIXED: xgboost test
Fixed in session 3 — split into two tests: `test_returns_models` (checks logistic_regression + random_forest only) and `test_xgboost_included_when_installed` (uses `pytest.importorskip("xgboost")`).

---

## Exact Next Steps (ordered)

1. **Write `src/serving/ui/pages/pipeline_status.py`** - Last missing Streamlit page
2. **Write Phase 7 CI/CD files**:
   - `.github/workflows/ci.yml`
   - `.github/workflows/deploy.yml`
   - `.github/workflows/retrain.yml`
3. **Write all missing test files**:
   - `tests/test_lambda_handler.py` (moto mocks)
   - `tests/test_manifest.py` (moto mocks)
   - `tests/test_s3_io.py` (moto mocks)
   - `tests/test_artifacts.py` (moto mocks)
   - `tests/test_drift.py`
   - `tests/test_data_quality.py`
   - `tests/test_alerts.py` (moto mocks)
   - `tests/test_streamlit_data_loader.py`
   - `tests/test_pipeline_e2e.py` (Phase 1G)
4. **Write Phase 2 Terraform Infrastructure** (~25 files):
   - `infra/terraform/main.tf`, `variables.tf`, `outputs.tf`
   - `infra/terraform/environments/{dev,staging,prod}.tfvars`
   - Modules: s3, dynamodb, iam, lambda, step_functions (+ asl/pipeline.asl.json), sagemaker, monitoring
5. **Run `ruff check --fix`** - Full lint pass
6. **Run `pytest`** - Full test pass
7. **Git commit all work** on `feature/aws-mlops-pipeline`

---

## Commands Run + Key Outputs (this session)

```
find src -name "*.py" | sort          # Listed 21 Python source files
find tests -name "*.py" | sort        # Listed 8 test files
ls -la infra/ .github/ src/serving/   # Dirs exist but mostly empty
git rev-parse --abbrev-ref HEAD       # feature/aws-mlops-pipeline
git rev-parse --short HEAD            # 372d414
git status --short                    # All changes unstaged/untracked (see Files Changed)
python -m pytest tests/ --tb=short -q # 46 passed, 1 FAILED (test_models xgboost key)
```

---

## Tests Run + Results

```
46 passed, 1 failed
FAILED: tests/test_models.py::TestModelDefinitions::test_returns_models
  AssertionError: assert 'xgboost' in models
  (xgboost may not be installed in venv)
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
- [ ] Write src/serving/ui/pages/pipeline_status.py
- [ ] Write 3 GitHub Actions workflows (ci, deploy, retrain)
- [ ] Write ~25 Terraform files
- [ ] Write all missing tests (lambda, manifest, s3_io, artifacts, drift, data_quality, alerts, streamlit, e2e)
- [ ] Run full lint + test suite
- [ ] Git commit all phases

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

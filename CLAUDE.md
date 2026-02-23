# CLAUDE.md - SpanishGas AWS MLOps Pipeline

## Project Overview
Churn prediction system for 20,099 Spanish energy customers. Converting Jupyter notebook logic (01: medallion ETL, 02: modeling) into a production AWS MLOps pipeline.

## Branch
`feature/aws-mlops-pipeline` (from `main`)

## Current State (session 3)
- **Phases 0 + 1 complete** (configs, all core Python modules, 47 tests passing)
- **Phases 3 + 4 + 5 complete** (pipeline steps, model artifacts/registry, monitoring/drift)
- **Phase 6 mostly complete** (Streamlit dashboard — missing pipeline_status.py page)
- **Phases 2 + 7 not started** (Terraform IaC, GitHub Actions CI/CD)
- **Tests not yet written** for new Phases 3-6 code
- **Nothing committed yet** — all changes unstaged

## Next Step Handoff
1. Write `src/serving/ui/pages/pipeline_status.py` (last missing Streamlit page)
2. Write 3 GitHub Actions workflows (ci.yml, deploy.yml, retrain.yml)
3. Write all missing test files (9 test files)
4. Write Phase 2 Terraform infrastructure (~25 files)
5. Run `ruff check --fix` + `pytest` for full lint/test pass
6. Git commit all work

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
src/serving/ui/            - app.py, data_loader.py, pages/{model_performance, drift_monitor, customer_risk}
infra/terraform/           - (not yet: all IaC)
Dockerfile.lambda          - Lambda container image
Dockerfile.processing      - SageMaker Processing container image
```

## Commands
```bash
make install    # pip install -e ".[dev]"
make lint       # ruff check src/ tests/
make test       # pytest tests/ -v
make test-cov   # pytest --cov
```

## Known Issues
- None currently (manifest.py bug and xgboost test both fixed in session 3)

## Full Context
See `CONTEXT.MD` for complete dump with all files changed, decisions, blockers, and ordered next steps.

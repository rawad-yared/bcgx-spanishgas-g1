# SpanishGas

## How to run locally

1. Create and activate a virtual environment:
   - `python3 -m venv .venv`
   - `source .venv/bin/activate`
2. Install project dependencies:
   - `python -m pip install -e .`
   - `python -m pip install pytest pre-commit`
3. Run checks:
   - `make lint`
   - `make test`

## Repository structure

```text
.
├── configs/
├── docs/
│   ├── ARCHITECTURE.md
│   ├── BRD.md
│   └── TICKETS.md
├── notebooks/
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── pipelines/
│   ├── reco/
│   └── serving/
├── tests/
├── Makefile
└── pyproject.toml
```

- `notebooks/`: Jupyter notebooks for exploration, prototyping, and analysis notes.

## Python scripts and responsibilities

Current scripts:

- `src/__init__.py`: marks `src` as the root Python package.
- `src/data/__init__.py`: package marker for data ingestion/transformation modules.
- `src/features/__init__.py`: package marker for feature engineering modules.
- `src/models/__init__.py`: package marker for model training/evaluation modules.
- `src/pipelines/__init__.py`: package marker for orchestration entrypoints.
- `src/reco/__init__.py`: package marker for recommendation logic modules.
- `src/serving/__init__.py`: package marker for batch/API serving modules.
- `tests/test_imports.py`: smoke test that validates package imports.

Planned scripts (from `docs/TICKETS.md`):

- `src/data/contracts.py`: schema contracts for source and derived datasets.
- `src/data/ingest.py`: raw CSV/JSON ingestion into Bronze outputs with metadata logs.
- `src/data/silver.py`: Silver cleaning/conformance transforms and data quality reports.
- `src/features/build_features.py`: leakage-safe as-of feature table generation.
- `src/data/build_training_set.py`: churn training set assembly with temporal splits.
- `src/models/churn_baseline.py`: logistic regression baseline training and reporting.
- `src/models/churn_gbdt.py`: gradient boosted churn model training and comparison.
- `src/models/segmentation.py`: clustering-based segmentation and profiling outputs.
- `src/reco/schema.py`: recommendation output schema definition and validation helpers.
- `src/reco/simulate.py`: profit-aware offer simulation.
- `src/reco/recommend.py`: recommendation engine using risk, segment, and margin rules.
- `src/pipelines/run_pipeline.py`: end-to-end local pipeline orchestrator.
- `src/serving/batch_score.py`: batch scoring and recommendation generation job.
- `src/serving/api/main.py`: FastAPI real-time scoring endpoint.

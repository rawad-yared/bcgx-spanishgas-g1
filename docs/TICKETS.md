SpanishGas – Implementation Tickets (Codex Backlog)

How to Use This File
	•	One ticket per Codex session.
	•	Codex must work on a feature branch, never main.
	•	Each ticket includes: Context → Task → Deliverables → Acceptance Criteria → Verify.
	•	No ticket is “done” unless acceptance criteria are met and verification commands run.

⸻

EPIC 0 — Repo & Standards

T-000: Repository scaffold + standards

Context: Create a consistent structure for pipelines, models, and deployment artifacts.
Task: Add a standard repo layout and baseline tooling configs.
Deliverables:
	•	Folders:
	•	docs/, src/, src/data/, src/features/, src/models/, src/reco/, src/pipelines/, src/serving/, tests/, configs/
	•	Tooling:
	•	pyproject.toml or requirements.txt
	•	.pre-commit-config.yaml
	•	.gitignore
	•	Makefile (or taskfile.yml) with common commands
	•	Baseline README section: “How to run locally”
Acceptance Criteria:
	•	make lint and make test exist (even if minimal).
	•	Project imports work (python -c "import src").
Verify:
	•	python -m pytest
	•	python -m compileall src

⸻

EPIC 1 — Data Layer (Bronze/Silver/Gold)

T-010: Data contracts + schema definitions

Context: Codex needs “table contracts” to build deterministic pipelines.
Task: Define schema contracts for all source datasets and key derived tables.
Deliverables:
	•	src/data/contracts.py (or contracts/ folder) with:
	•	Dataset name
	•	Expected columns + types
	•	Primary keys
	•	Time columns + grain
	•	docs/DATA_DICTIONARY.md (initial version)
Acceptance Criteria:
	•	Contracts cover all provided datasets and at least 3 gold outputs.
Verify:
	•	Run a script that loads contracts and prints them without error.

⸻

T-011: Raw ingestion loader (local filesystem)

Context: We need repeatable ingestion into Bronze.
Task: Build a loader that reads raw CSV/JSON and writes “Bronze” datasets to data/bronze/ with run metadata.
Deliverables:
	•	src/data/ingest.py
	•	data/bronze/<dataset_name>/run_date=YYYY-MM-DD/ output paths
	•	data/bronze/_meta/ingestion_log.parquet (or .jsonl)
Acceptance Criteria:
	•	Ingestion runs end-to-end for all datasets.
	•	Row counts and schema inferred are logged.
Verify:
	•	python -m src.data.ingest --run-date 2026-02-09

⸻

T-012: Silver transformations (clean + conform)

Context: Silver is clean, typed, consistent tables.
Task: Build Silver transforms for all datasets using the BRD data quality rules.
Deliverables:
	•	src/data/silver.py with transforms per dataset
	•	Output to data/silver/<table_name>/
	•	Quality report per run (missing IDs, negatives, unknown channels, etc.)
Acceptance Criteria:
	•	Rows with missing customer_id are dropped (log count).
	•	Negative consumption is flagged (not silently removed).
	•	JSON interactions are flattened into a table.
Verify:
	•	python -m src.data.silver --run-date 2026-02-09

⸻

T-013: Gold feature table builder (point-in-time “as-of”)

Context: Gold tables must be leakage-safe.
Task: Create customer_features_asof_date generation for a given as-of date.
Deliverables:
	•	src/features/build_features.py
	•	Output: data/gold/customer_features_asof_date/asof_date=YYYY-MM-DD/
	•	Feature metadata file: data/gold/_meta/feature_manifest.json
Acceptance Criteria:
	•	Feature builder supports an --asof-date argument.
	•	No feature uses data after the as-of date (enforced by filtering).
Verify:
	•	python -m src.features.build_features --asof-date 2026-01-31

⸻

T-014: Training dataset builder (churn label join)

Context: Training dataset must be reproducible and time-safe.
Task: Build churn_training_dataset with explicit label window/horizon config.
Deliverables:
	•	src/data/build_training_set.py
	•	configs/training_set.yaml (horizon, cutoff, split rules)
	•	Output: data/gold/churn_training_dataset/
Acceptance Criteria:
	•	Temporal split implemented (train/valid/test by date).
	•	Clear logs for row counts per split.
Verify:
	•	python -m src.data.build_training_set --config configs/training_set.yaml

⸻

EPIC 2 — Modeling (Churn + Segmentation)

T-020: Baseline churn model (logistic regression)

Context: Establish a strong baseline with interpretability.
Task: Train and evaluate a baseline churn model.
Deliverables:
	•	src/models/churn_baseline.py
	•	Saved model artifact in artifacts/models/churn_baseline/
	•	Evaluation report: artifacts/reports/churn_baseline.md
Acceptance Criteria:
	•	Outputs PR-AUC, recall@K, precision@K, calibration stats.
	•	Reproducible run using config.
Verify:
	•	python -m src.models.churn_baseline --config configs/model_baseline.yaml

⸻

T-021: Gradient boosted churn model (champion candidate)

Context: Improve performance vs baseline.
Task: Train XGBoost/LightGBM-style model + compare to baseline.
Deliverables:
	•	src/models/churn_gbdt.py
	•	Model artifact + metrics JSON
	•	Comparison report baseline vs gbdt
Acceptance Criteria:
	•	Must outperform baseline on PR-AUC or recall@K without severe calibration loss.
	•	Includes feature importance and SHAP summary.
Verify:
	•	python -m src.models.churn_gbdt --config configs/model_gbdt.yaml

⸻

T-022: Segmentation model + profiling

Context: Segmentation must be interpretable and usable.
Task: Build clustering pipeline and segment profiling output.
Deliverables:
	•	src/models/segmentation.py
	•	data/gold/segments/asof_date=.../
	•	artifacts/reports/segmentation_profile.md
Acceptance Criteria:
	•	Segment count is configurable.
	•	Outputs top drivers per segment (feature means/z-scores).
	•	Includes churn rate and margin proxies per segment (if available).
Verify:
	•	python -m src.models.segmentation --asof-date 2026-01-31

⸻

EPIC 3 — Recommendation System (Timing + Offer Policy)

T-030: Recommendation schema + policy spec

Context: The recommender needs a strict output contract.
Task: Define recommendation outputs and policy constraints.
Deliverables:
	•	docs/RECOMMENDATION_POLICY.md
	•	src/reco/schema.py defining required fields
Acceptance Criteria:
	•	Output includes: customer_id, risk_score, segment, action, timing_window, expected_margin_impact, reason_codes.
	•	Policy includes guardrails (no offer if negative margin, etc.).
Verify:
	•	Schema import works and unit tests validate required fields.

⸻

T-031: Offer simulator (profit-aware)

Context: Need a business KPI evaluator for recommendations.
Task: Build a simulator to estimate expected profit impact of offering discounts.
Deliverables:
	•	src/reco/simulate.py
	•	artifacts/reports/offer_simulation.md
Acceptance Criteria:
	•	Takes churn probability, acceptance probability (assumed or learned), margin, discount level.
	•	Outputs expected retained margin and ROI by segment/risk bucket.
Verify:
	•	python -m src.reco.simulate --config configs/reco_sim.yaml

⸻

T-032: Recommendation engine v1 (rules + optimization)

Context: MVP recommender should be deterministic and auditable.
Task: Produce recommended actions for a scoring run date.
Deliverables:
	•	src/reco/recommend.py
	•	Output: data/gold/recommendations/run_date=YYYY-MM-DD/
Acceptance Criteria:
	•	Uses churn score + segment + margin to select:
	•	who gets an offer
	•	discount tier (small/med/large)
	•	timing bucket (e.g., 30–60 days)
	•	Adds reason codes from top SHAP features or rule reasons.
Verify:
	•	python -m src.reco.recommend --run-date 2026-02-09

⸻

EPIC 4 — Pipelines & Orchestration

T-040: Orchestrated pipeline (local)

Context: Must run end-to-end reliably.
Task: Build a single “pipeline runner” that executes ingestion → silver → gold → train → score → recommend.
Deliverables:
	•	src/pipelines/run_pipeline.py
	•	configs/pipeline.yaml (enable/disable steps)
Acceptance Criteria:
	•	One command can produce a full run with artifacts and outputs.
	•	Steps are idempotent (re-run safe).
Verify:
	•	python -m src.pipelines.run_pipeline --config configs/pipeline.yaml

⸻

EPIC 5 — Serving (Batch + Real-Time)

T-050: Batch scoring job

Context: Produce daily/weekly churn scores and recommendations.
Task: Implement batch scoring that reads gold features and writes outputs.
Deliverables:
	•	src/serving/batch_score.py
	•	Output tables in data/gold/scoring/ and data/gold/recommendations/
Acceptance Criteria:
	•	Supports run-date parameter.
	•	Logs row counts and scoring latency.
Verify:
	•	python -m src.serving.batch_score --run-date 2026-02-09

⸻

T-051: Real-time scoring API (FastAPI)

Context: On-demand scoring endpoint.
Task: Build FastAPI service to score a customer and return recommendation payload.
Deliverables:
	•	src/serving/api/main.py
	•	Endpoint: POST /score
	•	Request: { "customer_id": "...", "asof_date": "YYYY-MM-DD" }
Acceptance Criteria:
	•	Returns churn score + segment + recommendation + reason codes.
	•	Includes basic input validation and error handling.
Verify:
	•	uvicorn src.serving.api.main:app --reload
	•	Send sample request returns 200.

⸻

T-052: Streamlit UI (optional but recommended)

Context: Demonstration layer for stakeholders.
Task: Build Streamlit dashboard to view risk list + segments + recommended actions.
Deliverables:
	•	src/serving/ui/app.py
Acceptance Criteria:
	•	Loads latest scoring output
	•	Displays top at-risk customers and recommendation details
Verify:
	•	streamlit run src/serving/ui/app.py

⸻

EPIC 6 — Monitoring & Governance

T-060: Data drift monitoring

Context: Detect changes in feature distributions and missingness.
Task: Implement drift checks between current run vs baseline.
Deliverables:
	•	src/monitoring/data_drift.py
	•	Drift report artifact in artifacts/monitoring/
Acceptance Criteria:
	•	Computes at least: missingness deltas + PSI for key features.
	•	Flags threshold exceedance.
Verify:
	•	python -m src.monitoring.data_drift --run-date 2026-02-09

⸻

T-061: Model performance monitoring (delayed labels)

Context: Monitor performance once labels arrive.
Task: Compute performance by time bucket and segment.
Deliverables:
	•	src/monitoring/model_perf.py
	•	Report: artifacts/monitoring/model_perf.md
Acceptance Criteria:
	•	Reports PR-AUC, recall@K, calibration drift by segment.
	•	Produces retraining trigger signal (boolean + reason).
Verify:
	•	python -m src.monitoring.model_perf --config configs/monitoring.yaml

⸻

EPIC 7 — Packaging, CI/CD, Reproducibility

T-070: Dockerization

Context: Reproducible environments + deploy readiness.
Task: Add Dockerfiles for training and API service.
Deliverables:
	•	Dockerfile (API)
	•	Dockerfile.train (training)
	•	docker-compose.yml (optional)
Acceptance Criteria:
	•	docker build succeeds for both images.
Verify:
	•	docker build -t spanishgas-api .
	•	docker build -f Dockerfile.train -t spanishgas-train .

⸻

T-071: CI pipeline (lint + tests)

Context: Quality gates.
Task: Add CI config (GitHub Actions) that runs lint + tests on PR.
Deliverables:
	•	.github/workflows/ci.yml
Acceptance Criteria:
	•	PR triggers run and pass.
	•	Artifacts uploaded for reports (optional).
Verify:
	•	CI runs on push/PR successfully.

⸻

Notes / Sequencing Recommendation (Do Not Skip)

Suggested order for fastest end-to-end MVP:
	1.	T-000 → T-010 → T-011 → T-012 → T-013 → T-014
	2.	T-020 → T-021 → T-022
	3.	T-030 → T-031 → T-032
	4.	T-040 → T-050 → T-051 (→ T-052 optional)
	5.	T-060 → T-061
	6.	T-070 → T-071

SpanishGas – Technical Architecture Document

1. Architecture Overview

SpanishGas is designed as a modular, production-oriented analytics platform that supports:
	•	Batch and real-time churn scoring
	•	Periodic customer segmentation
	•	Prescriptive price-intervention recommendations
	•	Full MLOps lifecycle (training, deployment, monitoring, retraining)

The architecture follows a layered data + ML system design, separating concerns across:
	•	Data ingestion and storage
	•	Feature engineering
	•	Model training and versioning
	•	Model serving
	•	Monitoring and governance

This ensures scalability, reproducibility, and auditability while remaining feasible within an academic context.

⸻

2. High-Level Architecture Components

Core layers
	1.	Ingestion Layer
	2.	Storage Layer (Bronze / Silver / Gold)
	3.	Feature Engineering Layer
	4.	Model Training & Registry
	5.	Model Serving (Batch + Real-Time)
	6.	Monitoring & Retraining
	7.	CI/CD & Environment Management

⸻

3. Ingestion Layer

Purpose

Load raw source data into the platform in an immutable, traceable manner.

Data Sources
	•	CSV and JSON files provided by SpanishGas:
	•	Customer attributes
	•	Contracts
	•	Price history
	•	Hourly consumption
	•	Customer interactions
	•	Costs by province/month
	•	Churn labels

Design Principles
	•	Raw data is never modified
	•	Each ingestion run is versioned
	•	Schema validation is enforced at ingestion

Output
	•	Raw files stored in object storage as Bronze data
	•	Metadata logged (row counts, schema, ingestion timestamp)

⸻

4. Storage Layer (Bronze / Silver / Gold)

4.1 Bronze Layer (Raw)

Purpose
	•	Preserve original data for traceability and reprocessing

Characteristics
	•	Immutable
	•	Schema-on-read
	•	Minimal transformations

⸻

4.2 Silver Layer (Cleaned & Conformed)

Purpose
	•	Clean, normalize, and standardize data across sources

Key Transformations
	•	Type casting and timestamp normalization
	•	Deduplication
	•	Handling missing values according to BRD rules
	•	Explicit flagging of anomalies (e.g., negative consumption)
	•	Normalization of nested JSON (customer interactions)

Outputs
	•	One table per domain entity (customers, contracts, interactions, consumption, pricing)

⸻

4.3 Gold Layer (Analytics-Ready)

Purpose
	•	Provide point-in-time–correct, business-ready datasets

Core Gold Tables
	•	customer_snapshot_daily
	•	customer_snapshot_monthly
	•	customer_features_asof_date
	•	churn_training_dataset
	•	recommendation_candidates

Design Rules
	•	All features are computed as of a reference date
	•	No future information is allowed (leakage prevention)
	•	Time grain is explicit and documented

⸻

5. Feature Engineering Layer

Feature Categories
	•	Contract lifecycle (days to contract end, tenure)
	•	Pricing competitiveness (vs OMIE, vs cost)
	•	Consumption aggregates and volatility
	•	Customer interaction recency, frequency, sentiment
	•	Product bundle indicators
	•	Segment proxies (SME, residential, second residence)

Design Principles
	•	Deterministic, idempotent feature pipelines
	•	Reusable feature functions
	•	Unit tests for feature logic
	•	Explicit feature metadata (description, grain, owner)

Output
	•	Feature matrices consumed by training and inference pipelines

⸻

6. Model Training & Registry

6.1 Training Pipelines

Churn Model
	•	Supervised binary classification
	•	Time-aware splits (temporal validation)
	•	Emphasis on recall at top risk percentiles

Segmentation Model
	•	Unsupervised clustering
	•	Interpretable segments with clear business meaning

Recommendation Logic
	•	Policy-based MVP (rules + constraints)
	•	Optional model-based scoring for expected value

Training Orchestration
	•	Orchestrated workflows (e.g., Prefect or equivalent)
	•	Reproducible runs with fixed seeds and data versions

⸻

6.2 Model Registry

Purpose
	•	Version control and governance of models

Capabilities
	•	Model versioning
	•	Stage transitions (Staging → Production)
	•	Linked metrics, parameters, and artifacts
	•	Model cards for documentation

⸻

7. Model Serving Layer

7.1 Batch Scoring

Use Case
	•	Daily or weekly generation of churn scores and recommendations

Outputs
	•	customer_id
	•	Churn probability
	•	Segment
	•	Recommended action
	•	Timing window
	•	Expected margin impact
	•	Reason codes

⸻

7.2 Real-Time API

Use Case
	•	On-demand scoring during customer interactions

Interface
	•	REST API
	•	JSON request/response
	•	Stateless inference

Latency Target
	•	Sub-second response (best-effort)

⸻

7.3 User Interface (Optional but Recommended)
	•	Lightweight Streamlit application
	•	Displays:
	•	At-risk customers
	•	Segment profiles
	•	Recommended actions with explanations
	•	Aggregate business KPIs

⸻

8. Monitoring & Observability

8.1 Data Monitoring
	•	Schema changes
	•	Missing values
	•	Feature distribution drift (PSI / KS)

8.2 Model Monitoring
	•	Prediction distribution drift
	•	Performance decay (delayed labels)
	•	Calibration drift

8.3 Business Monitoring
	•	Retention uplift
	•	Incremental retained margin
	•	Cost of interventions

⸻

9. Retraining Strategy

Retraining Triggers
	•	Scheduled retraining (e.g., monthly)
	•	Data drift beyond thresholds
	•	Performance degradation
	•	Major pricing or market regime changes

Governance
	•	New models evaluated before promotion
	•	Rollback capability to previous versions

⸻

10. CI/CD & Environment Management

Code Management
	•	Git-based workflow
	•	Feature branches and pull requests
	•	Mandatory reviews

CI
	•	Linting
	•	Unit tests
	•	Pipeline validation

CD
	•	Containerized builds (Docker)
	•	Automated deployment of pipelines and APIs

Environment Separation
	•	Development
	•	Staging
	•	Production

⸻

11. Architecture Decision Records (ADR)

All major design choices must be recorded as ADRs, including:
	•	Cloud provider selection
	•	Tooling choices (MLflow, orchestration, storage)
	•	Batch vs real-time tradeoffs

ADRs ensure transparency and prevent undocumented changes.

⸻

12. Non-Goals
	•	Full-scale CRM automation
	•	Real-time market trading integration
	•	Manual or notebook-only pipelines
	•	Undocumented model experimentation

⸻

13. Open Dependencies

The following must be resolved to finalize the architecture:
	•	Definition of SME vs corporate customers
	•	Handling of negative or missing consumption
	•	Identification of regulated customers
	•	Cost data clarifications


SpanishGas – Business Requirements Document (BRD)

1. Purpose & Background

SpanishGas is an end-to-end Decision Intelligence platform for a gas and electricity utility, designed to proactively reduce customer churn while protecting profitability.

Unlike traditional churn projects, this solution combines:
	•	Predictive analytics (who is at risk),
	•	Customer understanding (why they are at risk),
	•	Prescriptive actions (what to do and when).

The platform must support real operational decisions, not just offline analysis, and must be designed as a production-grade analytics system aligned with consulting best practices (BCG X–level delivery).

⸻

2. Business Problem Statement

SpanishGas faces customer churn driven by:
	•	Contract expirations,
	•	Price competitiveness vs market benchmarks,
	•	Customer service interactions,
	•	Consumption and pricing volatility.

Current retention actions are either:
	•	Reactive (too late),
	•	Untargeted (over-discounting),
	•	Or margin-destructive (saving unprofitable customers).

The core business challenge is to intervene at the right time, for the right customer, with the right price action, maximizing net retained margin, not just retention rate.

⸻

3. Business Objectives

Primary Objective:
	•	Maximize incremental retained margin (€) through data-driven churn prevention.

Secondary Objectives:
	•	Reduce churn rate in high-value customer segments.
	•	Avoid unnecessary discounts for customers unlikely to churn.
	•	Enable consistent, explainable, and auditable retention decisions.
	•	Provide actionable outputs usable by pricing and customer care teams.

⸻

4. Stakeholders & Users

Business Stakeholders
	•	Commercial / Pricing Teams – define pricing constraints and offers.
	•	Customer Care / Operations – execute interventions.
	•	Leadership / Management – evaluate financial impact and ROI.

Technical Stakeholders
	•	Data Science & Analytics – model development and validation.
	•	IT / Platform Teams – deployment, monitoring, and scalability.

⸻

5. Users & Decision Points

User	Decision
Pricing Manager	Which customers should receive discounts and how much
Customer Care Agent	Whether to intervene and how to explain the offer
Commercial Leadership	Whether retention actions improve profitability

The platform must support decision-making, not replace it.

⸻

6. Functional Requirements

6.1 Churn Prediction Module

Purpose
Estimate the probability that a customer will churn within a defined future window.

Inputs
	•	Customer attributes
	•	Contract details
	•	Consumption patterns
	•	Price history and benchmarks
	•	Customer interactions

Outputs
	•	Churn probability score
	•	Risk tier (e.g., Low / Medium / High)
	•	Primary churn drivers (explainability)

Frequency
	•	Batch: daily or weekly
	•	Real-time: on-demand scoring

Key Metrics
	•	PR-AUC
	•	Recall@K
	•	Calibration
	•	Stability over time

⸻

6.2 Customer Segmentation Module

Purpose
Group customers into interpretable segments based on behavior and value.

Segmentation Dimensions
	•	Consumption level and volatility
	•	Price sensitivity
	•	Contract type
	•	Product bundle (electricity, gas, dual)
	•	Interaction behavior

Outputs
	•	Segment ID
	•	Segment profile and description
	•	Segment-level churn and margin characteristics

Frequency
	•	Batch refresh (monthly or quarterly)

⸻

6.3 Recommendation & Intervention Module

Purpose
Recommend who to target, when to intervene, and what price action to take.

Inputs
	•	Churn probability
	•	Customer segment
	•	Contract end date
	•	Margin and cost data
	•	Pricing and regulatory constraints

Outputs
	•	Intervention recommendation (Yes/No)
	•	Recommended timing window
	•	Suggested price/discount action
	•	Expected churn reduction
	•	Expected margin impact
	•	Reason codes (explainability)

Constraints
	•	Must respect regulatory or protected customer rules (e.g., social tariffs).
	•	Must avoid negative-margin interventions.

Evaluation Metrics
	•	Incremental retained margin (€)
	•	Cost per retained customer
	•	Offer acceptance rate
	•	ROI of interventions

⸻

7. Non-Functional Requirements

7.1 Scalability & Reliability
	•	Support batch and real-time use cases.
	•	Reproducible pipelines with versioned data and models.

7.2 Explainability & Auditability
	•	All model outputs must include explainability.
	•	Recommendations must be traceable and justifiable.

7.3 Data Privacy & Compliance
	•	GDPR-compliant handling of personal data.
	•	Clear data lineage and access control.

7.4 Observability
	•	Monitoring for data drift, model decay, and business KPI degradation.
	•	Alerting for retraining or investigation triggers.

⸻

8. Data Requirements

Source Datasets
	•	customer_attributes.csv
	•	customer_contracts.csv
	•	price_history.csv
	•	consumption_hourly_2024.csv
	•	customer_interactions.json
	•	costs_by_province_month.csv
	•	churn_label.csv

Data Quality Rules
	•	Rows with missing customer_id must be dropped.
	•	Negative consumption values must be explicitly flagged and handled.
	•	Missing pricing or tier data must follow defined imputation logic.
	•	All time-based features must be point-in-time correct (no leakage).

⸻

9. Success Metrics

Business Metrics
	•	Incremental retained margin (€)
	•	Retention uplift vs baseline
	•	ROI of retention actions

Model Metrics
	•	PR-AUC
	•	Recall@K
	•	Calibration
	•	Segment-level performance parity

⸻

10. Out of Scope
	•	CRM campaign execution (email/SMS delivery).
	•	Manual, non-reproducible analytics workflows.
	•	Unvalidated causal claims without experimentation.
	•	Market trading or operational pricing execution.

⸻

11. Open Questions & Clarifications Needed
	•	Definition and identification of SME vs corporate customers.
	•	Interpretation of negative electricity and gas consumption values.
	•	Treatment of missing consumption data (true zero vs missing).
	•	Gas volume to kWh conversion factor to use.
	•	Identification and handling of regulated/protected customers.
	•	Missing definitions in costs_by_province_month.csv.

These must be resolved before final model training.

⸻

Status
	•	Version: v1.0


# SpanishGas — Business Requirements Document (BRD)

**Version:** 1.0
**Date:** 2026-03-06
**Status:** Delivered
**Prepared for:** BCG X / Spanish Energy Utility
**Branch:** `feature/aws-mlops-pipeline`

---

## Table of Contents

- [1. Executive Summary](#1-executive-summary)
- [2. Business Context](#2-business-context)
  - [2.1 Problem Statement](#21-problem-statement)
  - [2.2 Business Objectives](#22-business-objectives)
  - [2.3 Success Criteria](#23-success-criteria)
  - [2.4 Stakeholders](#24-stakeholders)
- [3. Scope](#3-scope)
  - [3.1 In Scope](#31-in-scope)
  - [3.2 Out of Scope](#32-out-of-scope)
- [4. Data Requirements](#4-data-requirements)
  - [4.1 Source Datasets](#41-source-datasets)
  - [4.2 Data Quality Requirements](#42-data-quality-requirements)
  - [4.3 Data Retention & Privacy](#43-data-retention--privacy)
- [5. Functional Requirements](#5-functional-requirements)
  - [5.1 Data Ingestion & ETL](#51-data-ingestion--etl)
  - [5.2 Feature Engineering](#52-feature-engineering)
  - [5.3 Model Training & Evaluation](#53-model-training--evaluation)
  - [5.4 Customer Scoring & Risk Classification](#54-customer-scoring--risk-classification)
  - [5.5 Retention Recommendations](#55-retention-recommendations)
  - [5.6 Drift Detection & Monitoring](#56-drift-detection--monitoring)
  - [5.7 Dashboard & Reporting](#57-dashboard--reporting)
  - [5.8 Pipeline Automation](#58-pipeline-automation)
- [6. Non-Functional Requirements](#6-non-functional-requirements)
  - [6.1 Performance](#61-performance)
  - [6.2 Reliability & Availability](#62-reliability--availability)
  - [6.3 Scalability](#63-scalability)
  - [6.4 Security](#64-security)
  - [6.5 Maintainability](#65-maintainability)
  - [6.6 Testability](#66-testability)
- [7. Business Rules & Constraints](#7-business-rules--constraints)
  - [7.1 Model Promotion Gate](#71-model-promotion-gate)
  - [7.2 Retention Offer Guardrails](#72-retention-offer-guardrails)
  - [7.3 Risk Tier Definitions](#73-risk-tier-definitions)
  - [7.4 Threshold Selection Policy](#74-threshold-selection-policy)
- [8. Delivered Solution Summary](#8-delivered-solution-summary)
  - [8.1 Model Performance](#81-model-performance)
  - [8.2 Customer Risk Distribution](#82-customer-risk-distribution)
  - [8.3 System Components](#83-system-components)
  - [8.4 Dashboard](#84-dashboard)
- [9. Acceptance Criteria](#9-acceptance-criteria)
- [10. Dependencies & Assumptions](#10-dependencies--assumptions)
- [11. Risks & Mitigations](#11-risks--mitigations)
- [12. Glossary](#12-glossary)

---

## 1. Executive Summary

A Spanish gas and electricity utility is experiencing customer churn that erodes revenue and increases customer acquisition costs. This document defines the business requirements for an end-to-end MLOps platform that predicts customer churn risk, classifies customers into actionable risk tiers, generates retention recommendations with financial guardrails, and delivers insights through an executive dashboard.

The delivered system processes 7 datasets covering 20,099 customers, engineers 56 predictive features, trains a calibrated XGBoost model (PR-AUC = 0.757, ROC-AUC = 0.932), and automates the full pipeline from data upload to dashboard refresh — running weekly or on-demand with drift monitoring and email alerts.

---

## 2. Business Context

### 2.1 Problem Statement

The utility faces significant revenue risk from customer churn in the Spanish deregulated energy market:
- Customers can switch providers with minimal friction
- High competition in certain provinces drives price-sensitive behavior
- Current retention efforts are reactive (responding to cancellation requests) rather than proactive
- No systematic way to identify at-risk customers before they initiate cancellation
- Retention offers are applied uniformly without regard to customer profitability

### 2.2 Business Objectives

| # | Objective | Measure |
|---|-----------|---------|
| O1 | **Predict churn risk** for all customers with high accuracy | PR-AUC >= 0.70, ROC-AUC > 0.90 |
| O2 | **Classify customers** into actionable risk tiers | 4-tier system with clear action mapping |
| O3 | **Generate retention recommendations** with financial guardrails | Margin-aware offer sizing; no offers for unprofitable customers |
| O4 | **Automate the ML pipeline** end-to-end | Triggered by data upload; retrains weekly |
| O5 | **Monitor model health** continuously | Drift detection with automated alerts |
| O6 | **Provide executive visibility** via dashboard | 8-page dashboard with KPIs, risk views, and individual customer lookup |
| O7 | **Enable reproducibility** and auditability | IaC, CI/CD, versioned artifacts, test coverage |

### 2.3 Success Criteria

| Criterion | Target | Achieved |
|-----------|--------|----------|
| PR-AUC on held-out test set | >= 0.70 | 0.757 |
| ROC-AUC on held-out test set | > 0.90 | 0.932 |
| Recall at operating threshold | >= 0.70 | >= 0.70 (by design) |
| Pipeline automation | End-to-end on data upload | S3 trigger -> Lambda -> Step Functions |
| Retraining cadence | Weekly or on-demand | Monday 06:00 UTC (GitHub Actions) |
| Drift detection | Automated with alerting | KS test + SNS email |
| Dashboard availability | Always-on, web-accessible | ECS Fargate behind ALB |
| Test coverage | Comprehensive | 165 tests, 19 files |
| Infrastructure as Code | 100% | 11 Terraform modules |

### 2.4 Stakeholders

| Role | Interest |
|------|----------|
| Retention Team | Risk-scored customer lists, recommended actions, timing windows |
| Commercial Leadership | Executive KPIs (at-risk %, monthly loss, tier distribution) |
| Data Science Team | Model performance metrics, feature importance, experiment tracking |
| Operations | Pipeline health, drift alerts, run history |
| IT / Infrastructure | Terraform modules, IAM roles, cost management |

---

## 3. Scope

### 3.1 In Scope

- Automated data ingestion from 7 source datasets
- Medallion ETL pipeline (raw -> bronze -> silver -> gold)
- NLP enrichment of customer interactions (intent classification + sentiment analysis)
- Feature engineering (56 features across 7 tiers)
- XGBoost model training with Platt calibration
- Threshold optimization for precision at recall >= 0.70
- Batch scoring of all 20,099 customers
- 4-tier risk classification (Low / Medium / High / Critical)
- Margin-aware retention recommendations
- KS-test drift detection with SNS email alerts
- 8-page Streamlit dashboard on ECS Fargate
- CI/CD via GitHub Actions (lint, test, deploy, retrain)
- Terraform IaC for all AWS infrastructure (11 modules)

### 3.2 Out of Scope

- Real-time (single-customer) scoring API
- A/B testing of retention offers
- Integration with CRM or billing systems
- Customer self-service portal
- Multi-region deployment
- Cost optimization (reserved instances, Savings Plans)
- GDPR consent management workflows
- Historical model versioning beyond latest champion

---

## 4. Data Requirements

### 4.1 Source Datasets

| # | Dataset | Format | Grain | Key Columns | Volume |
|---|---------|--------|-------|-------------|--------|
| 1 | churn_label | CSV | 1/customer | `customer_id`, `churned_within_horizon`, `churn_effective_date`, `label_date` | 20,099 rows |
| 2 | customer_attributes | CSV | 1/customer | `customer_id`, `province`, `customer_type`, `tariff_type`, `signup_date`, `product_bundle` | 20,099 rows |
| 3 | customer_contracts | CSV | 1/contract | `customer_id`, `contract_start_date`, `contract_end_date`, `contract_status`, `contract_term_months`, `product_type` | 20,099 rows |
| 4 | consumption_hourly_2024 | Parquet | 1/hour/customer | `customer_id`, `timestamp`, `elec_kwh`, `gas_m3` | ~17M rows |
| 5 | price_history | CSV | 1/date/product/tariff | `date`, `product_type`, `tariff_type`, `price_per_kwh`, `market_price` | Variable |
| 6 | costs_by_province_month | CSV | 1/month/province | `province`, `month`, `variable_cost`, `fixed_cost`, `network_cost` | Variable |
| 7 | customer_interactions | JSON | 1/interaction | `customer_id`, `date`, `channel`, `interaction_type`, `interaction_summary`, `resolution_status` | Variable |

### 4.2 Data Quality Requirements

| Pipeline Layer | Max Null Rate | Duplicate Policy |
|----------------|--------------|-----------------|
| Raw | 80% | Accepted (logged) |
| Bronze | 20% | No duplicate customer_ids |
| Silver | 5% | No duplicates |
| Gold | 2% | No duplicates; validated 1-row-per-customer |

Missing values are handled through:
- **Price imputation:** 3-level hierarchical fallback (customer -> segment x month -> national month)
- **Structural fills:** Sentinel values with business meaning (9999 = "never interacted", 0 = "no flag")

### 4.3 Data Retention & Privacy

- All data stored in S3 with server-side encryption (AES-256)
- S3 bucket versioning enabled for audit trail
- Lifecycle rules for cost management
- Customer IDs are opaque identifiers (no PII in feature set)

---

## 5. Functional Requirements

### 5.1 Data Ingestion & ETL

| Req # | Requirement | Status |
|-------|-------------|--------|
| FR-1.1 | System shall ingest 7 raw datasets in CSV, JSON, and Parquet formats | Delivered |
| FR-1.2 | System shall implement a medallion architecture (bronze -> silver -> gold) | Delivered |
| FR-1.3 | Bronze layer shall merge customer-level tables into a single customer view | Delivered |
| FR-1.4 | Bronze layer shall aggregate hourly consumption to monthly totals with tariff tier splits | Delivered |
| FR-1.5 | Bronze layer shall handle ~17M hourly consumption rows within SageMaker memory limits | Delivered (2M-row chunked processing) |
| FR-1.6 | Silver layer shall derive customer segments (Residential / SME / Corporate) | Delivered |
| FR-1.7 | Silver layer shall impute missing prices using hierarchical fallback | Delivered (3-level) |
| FR-1.8 | Silver layer shall compute P&L margins per customer per month (electricity + gas) | Delivered |
| FR-1.9 | Gold layer shall produce a single model-ready table (1 row/customer, 56 features) | Delivered |

### 5.2 Feature Engineering

| Req # | Requirement | Status |
|-------|-------------|--------|
| FR-2.1 | System shall engineer features across 7 distinct tiers | Delivered (1A, MP Core, MP Risk, 2A, 2B, 3, 1B) |
| FR-2.2 | Lifecycle features shall capture contract timing, tenure, channel, and segment | Delivered (15 features) |
| FR-2.3 | Market features shall capture consumption volumes, revenue mix, and price trends | Delivered (8 core + 13 risk = 21 features) |
| FR-2.4 | Behavioral features shall capture customer interactions, intent, and timing | Delivered (11 features) |
| FR-2.5 | Sentiment features shall be derived from NLP analysis of interaction text | Delivered (3 features via HuggingFace) |
| FR-2.6 | Compound features shall capture cross-tier interaction effects | Delivered (6 features) |
| FR-2.7 | System shall support multiple experiment configurations for ablation studies | Delivered (9 experiments: E0-E8) |
| FR-2.8 | System shall gracefully degrade if NLP dependencies are unavailable | Delivered (E8_no_sentiment fallback) |

### 5.3 Model Training & Evaluation

| Req # | Requirement | Status |
|-------|-------------|--------|
| FR-3.1 | System shall train XGBoost as the champion model | Delivered (n_estimators=600, depth=3, lr=0.02) |
| FR-3.2 | System shall handle class imbalance via dynamic `scale_pos_weight` | Delivered |
| FR-3.3 | System shall apply Platt scaling (sigmoid calibration) for well-calibrated probabilities | Delivered (CalibratedClassifierCV) |
| FR-3.4 | System shall optimize decision threshold for precision at recall >= 0.70 | Delivered (sub-train held-out approach) |
| FR-3.5 | System shall enforce a promotion gate of PR-AUC >= 0.70 | Delivered |
| FR-3.6 | System shall save model artifacts (pipeline.pkl, metadata, evaluation) to S3 | Delivered |
| FR-3.7 | System shall register models in SageMaker Model Package Group | Delivered |
| FR-3.8 | System shall support sklearn >= 1.6 and earlier versions | Delivered (FrozenEstimator compat) |

### 5.4 Customer Scoring & Risk Classification

| Req # | Requirement | Status |
|-------|-------------|--------|
| FR-4.1 | System shall batch-score all 20,099 customers per pipeline run | Delivered |
| FR-4.2 | System shall produce calibrated churn probabilities in [0, 1] | Delivered (Platt scaling) |
| FR-4.3 | System shall assign customers to 4 risk tiers based on calibrated probability | Delivered |
| FR-4.4 | System shall compute expected monthly loss per customer | Delivered (churn_proba x avg_monthly_margin) |
| FR-4.5 | System shall output scored_customers.parquet with all scoring columns | Delivered |

### 5.5 Retention Recommendations

| Req # | Requirement | Status |
|-------|-------------|--------|
| FR-5.1 | System shall map risk tiers to retention actions with timing windows | Delivered |
| FR-5.2 | System shall block offers for negative-margin customers | Delivered (hard guardrail) |
| FR-5.3 | Every recommendation shall include non-empty reason codes | Delivered (enforced by dataclass) |
| FR-5.4 | Risk scores shall be validated to [0, 1] range | Delivered |
| FR-5.5 | System shall output recommendations.parquet per pipeline run | Delivered |

### 5.6 Drift Detection & Monitoring

| Req # | Requirement | Status |
|-------|-------------|--------|
| FR-6.1 | System shall detect feature distribution drift using KS test | Delivered (p < 0.01 threshold) |
| FR-6.2 | System shall detect prediction distribution drift | Delivered (KS test on churn_proba) |
| FR-6.3 | System shall maintain a reference distribution store in S3 | Delivered |
| FR-6.4 | System shall send SNS email alerts when drift is detected | Delivered |
| FR-6.5 | System shall publish CloudWatch metrics (DriftDetected, FeaturesDrifted) | Delivered |
| FR-6.6 | System shall perform data quality checks at each pipeline layer | Delivered (null rates, duplicates, schema) |

### 5.7 Dashboard & Reporting

| Req # | Requirement | Status |
|-------|-------------|--------|
| FR-7.1 | System shall provide an always-on web dashboard | Delivered (ECS Fargate + ALB) |
| FR-7.2 | Dashboard shall include executive KPI overview (at-risk %, monthly loss) | Delivered |
| FR-7.3 | Dashboard shall allow browsing data across all pipeline layers | Delivered (Data Explorer page) |
| FR-7.4 | Dashboard shall display model performance metrics and confusion matrix | Delivered |
| FR-7.5 | Dashboard shall visualize feature drift results | Delivered (Drift Monitor page) |
| FR-7.6 | Dashboard shall show risk tier distribution with expected loss | Delivered (Customer Risk page) |
| FR-7.7 | Dashboard shall support individual customer lookup with risk profile | Delivered (Customer Lookup page) |
| FR-7.8 | Dashboard shall display retention recommendations by tier | Delivered (Recommendations page) |
| FR-7.9 | Dashboard shall show pipeline run history and status | Delivered (Pipeline Status page) |

### 5.8 Pipeline Automation

| Req # | Requirement | Status |
|-------|-------------|--------|
| FR-8.1 | Pipeline shall trigger automatically on S3 data upload to raw/ prefix | Delivered (Lambda S3 event) |
| FR-8.2 | Pipeline shall be idempotent (no duplicate runs for same file) | Delivered (DynamoDB manifest) |
| FR-8.3 | Pipeline shall support train mode (full retrain + evaluate + score) and score-only mode | Delivered (SFN Choice state) |
| FR-8.4 | Pipeline shall retrain weekly on schedule | Delivered (GitHub Actions cron: Monday 06:00 UTC) |
| FR-8.5 | Pipeline shall support manual trigger | Delivered (GitHub Actions manual dispatch) |
| FR-8.6 | Pipeline shall complete within 60 minutes | Delivered (~28 minutes) |

---

## 6. Non-Functional Requirements

### 6.1 Performance

| Req # | Requirement | Target | Achieved |
|-------|-------------|--------|----------|
| NFR-1.1 | Full pipeline runtime | < 60 min | ~28 min |
| NFR-1.2 | Bronze consumption processing (17M rows) | < 10 min | ~4 min |
| NFR-1.3 | Model training (16K samples) | < 10 min | ~3 min |
| NFR-1.4 | Batch scoring (20K customers) | < 5 min | ~2 min |
| NFR-1.5 | Dashboard page load time | < 5 sec | < 3 sec (cached) |

### 6.2 Reliability & Availability

| Req # | Requirement | Status |
|-------|-------------|--------|
| NFR-2.1 | Pipeline idempotency (no duplicate runs) | Delivered (DynamoDB conditional writes) |
| NFR-2.2 | Dashboard availability (ECS service auto-restart) | Delivered (desired_count=1, health checks) |
| NFR-2.3 | Graceful degradation if NLP unavailable | Delivered (E8_no_sentiment fallback) |
| NFR-2.4 | Alerting on pipeline failures | Delivered (CloudWatch Alarms + SNS) |

### 6.3 Scalability

| Req # | Requirement | Status |
|-------|-------------|--------|
| NFR-3.1 | Handle 20K+ customers per run | Delivered |
| NFR-3.2 | Handle 17M+ hourly consumption rows | Delivered (chunked processing) |
| NFR-3.3 | Support vertical scaling (larger SageMaker instances) | Delivered (configurable instance type) |

### 6.4 Security

| Req # | Requirement | Status |
|-------|-------------|--------|
| NFR-4.1 | S3 data encryption at rest (AES-256) | Delivered |
| NFR-4.2 | S3 bucket versioning | Delivered |
| NFR-4.3 | IAM least-privilege roles (5 roles) | Delivered |
| NFR-4.4 | GitHub OIDC authentication (no long-lived credentials) | Delivered |
| NFR-4.5 | Security groups for network isolation | Delivered (ALB + ECS groups) |

### 6.5 Maintainability

| Req # | Requirement | Status |
|-------|-------------|--------|
| NFR-5.1 | Infrastructure as Code (100%) | Delivered (11 Terraform modules) |
| NFR-5.2 | CI/CD automation | Delivered (3 GitHub Actions workflows) |
| NFR-5.3 | Lint-clean codebase | Delivered (Ruff) |
| NFR-5.4 | Documented architecture and BRD | Delivered |
| NFR-5.5 | Configurable via YAML and environment variables | Delivered |

### 6.6 Testability

| Req # | Requirement | Status |
|-------|-------------|--------|
| NFR-6.1 | Unit tests for all modules | Delivered (165 tests, 19 files) |
| NFR-6.2 | AWS mocking for integration tests | Delivered (moto) |
| NFR-6.3 | End-to-end pipeline test | Delivered (test_pipeline_e2e.py, 22 tests) |
| NFR-6.4 | CI gate (tests must pass before deploy) | Delivered |

---

## 7. Business Rules & Constraints

### 7.1 Model Promotion Gate

A trained model is promoted to champion status **only if PR-AUC >= 0.70** on the held-out test set. Models below this threshold are registered with "Rejected" status. This prevents deployment of underperforming models.

**Current performance:** PR-AUC = 0.757 (exceeds gate by 8.1%)

### 7.2 Retention Offer Guardrails

| Rule | Implementation | Rationale |
|------|---------------|-----------|
| **No negative-margin offers** | If `avg_monthly_margin < 0`, action forced to `no_offer` | Retention discount would increase losses |
| **Reason codes required** | Every recommendation must include >= 1 reason code | Transparency and auditability |
| **Risk score validation** | Churn probability must be in [0, 1] | Prevents erroneous recommendations |
| **Tier-appropriate offers** | Offer size scales with risk tier (none -> small -> medium -> large) | Resource allocation proportional to risk |

### 7.3 Risk Tier Definitions

| Tier | Churn Probability | Action | Timing | Current Count |
|------|-------------------|--------|--------|---------------|
| Low | < 40% | No offer | 60-90 days | 18,158 (90.3%) |
| Medium | 40% - 60% | Small offer | 30-60 days | 673 (3.3%) |
| High | 60% - 80% | Medium offer | Immediate | 527 (2.6%) |
| Critical | > 80% | Large offer | Immediate | 741 (3.7%) |

The tier boundaries (0.40, 0.60, 0.80) are applied to **calibrated probabilities** (Platt scaling) so that a 70% predicted probability corresponds approximately to a 70% actual churn rate.

### 7.4 Threshold Selection Policy

The decision threshold is selected to **maximize precision** subject to **recall >= 0.70**. This means:
- At least 70% of actual churners are identified (recall floor)
- Among those flagged, as many as possible are true churners (precision maximized)
- If no threshold achieves recall >= 0.70, the system falls back to the threshold with best F1 score

**Current threshold:** ~0.691

---

## 8. Delivered Solution Summary

### 8.1 Model Performance

| Metric | Value |
|--------|-------|
| PR-AUC | 0.757 |
| ROC-AUC | 0.932 |
| Decision Threshold | ~0.691 |
| Algorithm | XGBoost + Platt Scaling (sigmoid calibration) |
| Features | 56 (E5_full experiment) |
| Hyperparameters | n_estimators=600, max_depth=3, lr=0.02, subsample=0.8, colsample_bytree=0.7 |

### 8.2 Customer Risk Distribution

| Tier | Count | Percentage |
|------|-------|-----------|
| Low (<40%) | 18,158 | 90.3% |
| Medium (40-60%) | 673 | 3.3% |
| High (60-80%) | 527 | 2.6% |
| Critical (>80%) | 741 | 3.7% |
| **At-Risk (Medium+)** | **1,941** | **9.7%** |

### 8.3 System Components

| Component | Technology | Status |
|-----------|-----------|--------|
| Data Pipeline | SageMaker Processing (7 steps) | Operational |
| Orchestration | Lambda + Step Functions | Operational |
| Model Training | XGBoost + CalibratedClassifierCV | Operational |
| Scoring | Batch (20K customers/run) | Operational |
| Recommendations | Rule-based with guardrails | Operational |
| Drift Detection | KS test + SNS alerts | Operational |
| Dashboard | Streamlit on ECS Fargate | Operational |
| Infrastructure | Terraform (11 modules) | Operational |
| CI/CD | GitHub Actions (3 workflows) | Operational |
| Tests | 165 tests, 19 files | All passing |

### 8.4 Dashboard

**URL:** `spanishgas-dev-streamlit-alb-1221532574.eu-west-1.elb.amazonaws.com`

8 pages providing executive visibility into churn risk, model performance, drift status, individual customer lookup, retention recommendations, and pipeline health.

---

## 9. Acceptance Criteria

| # | Criterion | Met? |
|---|-----------|------|
| AC-1 | PR-AUC >= 0.70 on held-out test set | Yes (0.757) |
| AC-2 | Pipeline completes end-to-end without manual intervention | Yes (~28 min) |
| AC-3 | Pipeline triggers automatically on data upload to S3 | Yes (Lambda S3 event) |
| AC-4 | Pipeline is idempotent (no duplicate runs) | Yes (DynamoDB manifest) |
| AC-5 | Drift detection runs after every scoring step | Yes (KS test) |
| AC-6 | Email alerts sent when drift is detected | Yes (SNS) |
| AC-7 | Dashboard is accessible via web URL | Yes (ALB) |
| AC-8 | Dashboard shows all 8 required pages | Yes |
| AC-9 | Negative-margin customers receive no retention offer | Yes (hard guardrail) |
| AC-10 | All recommendations include reason codes | Yes (enforced by schema) |
| AC-11 | Infrastructure is fully managed via Terraform | Yes (11 modules) |
| AC-12 | CI/CD pipeline runs tests before deployment | Yes (ci.yml gates deploy.yml) |
| AC-13 | All tests pass (zero failures) | Yes (165/165) |
| AC-14 | Lint checks pass | Yes (Ruff clean) |
| AC-15 | Weekly retraining is automated | Yes (GitHub Actions cron) |

---

## 10. Dependencies & Assumptions

### Dependencies

| Dependency | Description |
|------------|-------------|
| AWS Account | Account 559307249592, eu-west-1 region |
| SageMaker Quotas | ml.m5.xlarge: 4 processing, 15 training instances |
| GitHub Repository | rawad-yared/bcgx-spanishgas-g1 |
| HuggingFace Model | cardiffnlp/twitter-roberta-base-sentiment-latest (pre-downloaded at build time) |
| Python 3.12 | Runtime for all containers |

### Assumptions

| # | Assumption |
|---|-----------|
| A1 | Raw data files follow the schema defined in `column_registry.yaml` |
| A2 | Churn labels are computed externally and provided in `churn_label.csv` |
| A3 | Customer IDs are consistent across all 7 source datasets |
| A4 | Hourly consumption data covers calendar year 2024 |
| A5 | Province names are consistent between customer attributes and cost tables |
| A6 | The churn horizon (days) is defined externally and embedded in the label dataset |
| A7 | Spanish PVPC tariff tier rules apply for electricity consumption bucketing |
| A8 | Gas-to-kWh conversion factor is 11 kWh/m3 |

---

## 11. Risks & Mitigations

| Risk | Severity | Likelihood | Mitigation |
|------|----------|-----------|------------|
| **Model degradation over time** | High | Medium | Weekly retraining + KS drift detection + SNS alerts |
| **Concept drift (P(churn\|X) changes)** | High | Medium | Currently not detected; recommended: track P(churn\|tier) over time |
| **Data quality issues in raw files** | Medium | Medium | Data quality checks at bronze/silver/gold layers with threshold alerts |
| **NLP model unavailable** | Low | Low | Graceful fallback to E8_no_sentiment experiment (53 features) |
| **SageMaker quota limits** | Medium | Low | Quotas verified (4 processing, 15 training instances available) |
| **Cost overrun** | Medium | Low | PAY_PER_REQUEST DynamoDB; SageMaker jobs run only during pipeline; ECS desired_count=1 |
| **Single point of failure (1 ECS task)** | Medium | Low | ECS auto-restart on failure; ALB health checks |
| **Calibration drift** | Medium | Medium | Platt scaling re-fitted on each retrain; probabilities recalibrated weekly |
| **Threshold staleness** | Medium | Medium | Threshold re-optimized on each retrain; consider quarterly business review |

---

## 12. Glossary

| Term | Definition |
|------|-----------|
| **Churn** | A customer discontinuing their energy contract within a defined horizon period |
| **PR-AUC** | Area Under the Precision-Recall Curve; measures model ability to identify churners among predicted positives |
| **ROC-AUC** | Area Under the Receiver Operating Characteristic Curve; measures overall discriminative ability |
| **Platt Scaling** | Sigmoid calibration that transforms raw model scores into well-calibrated probabilities |
| **KS Test** | Kolmogorov-Smirnov test; non-parametric test for distribution differences between two samples |
| **Medallion Architecture** | Data lakehouse pattern with progressive quality layers: bronze (raw), silver (cleaned), gold (curated) |
| **Risk Tier** | Customer classification based on calibrated churn probability (Low/Medium/High/Critical) |
| **Structural Fill** | Replacing missing values with semantically meaningful defaults (e.g., 9999 = "never interacted") |
| **Champion Model** | The current production model that passes the promotion gate (PR-AUC >= 0.70) |
| **Feature Tier** | A logical grouping of related features (e.g., Lifecycle, Behavioral, Sentiment) |
| **Promotion Gate** | Minimum performance threshold a model must exceed to be deployed to production |
| **Expected Monthly Loss** | `churn_probability x avg_monthly_margin` — the revenue at risk for each customer |
| **Guardrail** | A hard business constraint applied to recommendations (e.g., no offers for negative-margin customers) |
| **Reason Code** | A machine-generated explanation for why a recommendation was made |
| **OIDC** | OpenID Connect — passwordless authentication protocol used for GitHub -> AWS access |
| **IaC** | Infrastructure as Code — managing infrastructure through declarative configuration files (Terraform) |
| **SFN** | AWS Step Functions — serverless workflow orchestration service |

# SpanishGas — Customer Churn Prediction MLOps Platform

An end-to-end MLOps system that predicts customer churn for 20,099 Spanish energy customers, generates retention recommendations, and serves results through an auto-deployed Streamlit dashboard. Built on AWS with Terraform IaC, GitHub Actions CI/CD, and a medallion data architecture (bronze/silver/gold).

---

## Table of Contents

- [Project Overview](#project-overview)
- [Datasets](#datasets)
- [High-Level Architecture](#high-level-architecture)
- [Data Pipeline Architecture](#data-pipeline-architecture)
- [ML Training & Evaluation Architecture](#ml-training--evaluation-architecture)
- [Monitoring & Drift Detection Architecture](#monitoring--drift-detection-architecture)
- [Serving & Dashboard Architecture](#serving--dashboard-architecture)
- [CI/CD & Deployment Architecture](#cicd--deployment-architecture)
- [Infrastructure Architecture](#infrastructure-architecture)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Local Development Setup](#local-development-setup)
  - [Running the Pipeline Locally](#running-the-pipeline-locally)
  - [Running the Dashboard Locally](#running-the-dashboard-locally)
  - [Running Tests](#running-tests)
- [AWS Deployment](#aws-deployment)
  - [Terraform Setup](#terraform-setup)
  - [Docker Images](#docker-images)
  - [Deploying Infrastructure](#deploying-infrastructure)
  - [Triggering the Pipeline](#triggering-the-pipeline)
  - [GitHub Actions CI/CD](#github-actions-cicd)
- [Feature Engineering](#feature-engineering)
- [Model Details](#model-details)
- [Recommendation Engine](#recommendation-engine)

---

## Project Overview

SpanishGas is a decision intelligence platform for a Spanish gas and electricity utility. The system:

1. **Ingests** 7 raw datasets through a medallion ETL pipeline (raw -> bronze -> silver -> gold)
2. **Trains** an XGBoost churn model with 7 feature tiers across 9 experiment configurations
3. **Scores** all customers with churn probability, risk tier assignment, and expected monthly loss
4. **Detects drift** using Kolmogorov-Smirnov tests on features and predictions
5. **Generates recommendations** mapping risk tiers to retention actions with policy guardrails
6. **Serves** results through a 6-page Streamlit dashboard auto-deployed on AWS ECS Fargate
7. **Orchestrates** everything via AWS Step Functions, triggered automatically on S3 data upload

---

## Datasets

The system consumes **7 raw datasets** that describe customers, their contracts, consumption patterns, pricing, costs, interactions, and churn outcomes:

| # | Dataset | Format | Grain | Description |
|---|---------|--------|-------|-------------|
| 1 | **churn_label** | CSV | 1 row/customer | Binary churn outcome within a configurable horizon (days). Includes `churned_within_horizon`, `churn_effective_date`, and `label_date`. |
| 2 | **customer_attributes** | CSV | 1 row/customer | Static customer demographics: `province`, `customer_type`, `tariff_type`, `signup_date`, `product_bundle`. |
| 3 | **customer_contracts** | CSV | 1 row/contract | Contract lifecycle: `contract_start_date`, `contract_end_date`, `contract_status`, `contract_term_months`, `product_type`. |
| 4 | **consumption_hourly_2024** | CSV | 1 row/hour/customer | Hourly electricity (kWh) and gas (m3) meter readings. ~17M rows for 20K customers across 2024. Gas converted to kWh at 11 kWh/m3. |
| 5 | **price_history** | CSV | 1 row/date/product/tariff | Daily electricity and gas prices per tariff type and region, plus market benchmark prices for delta calculation. |
| 6 | **costs_by_province_month** | CSV | 1 row/month/province | Monthly variable costs, fixed costs, and network costs per province and commodity. |
| 7 | **customer_interactions** | JSON | 1 row/interaction | Customer service interactions: `channel`, `interaction_type`, `sentiment_score`, `resolution_status`. Includes complaints, billing issues, and cancellation intents. |

---

## High-Level Architecture

```
                                    SPANISHGAS MLOps PLATFORM
 ┌──────────────────────────────────────────────────────────────────────────────────┐
 │                                                                                  │
 │   ┌─────────┐    ┌──────────┐    ┌──────────────────────────────────────────┐    │
 │   │  Raw    │───>│  Lambda  │───>│          Step Functions Pipeline         │    │
 │   │  CSV/   │ S3 │ Trigger  │    │                                          │    │
 │   │  JSON   │ put│          │    │  Bronze ─> Silver ─> Gold ─> Train/Score │    │
 │   └─────────┘    └──────────┘    │          ─> Evaluate ─> Drift            │    │
 │                                  └──────────────┬───────────────────────────┘    │
 │                                                 │                                │
 │                    ┌────────────────────────────┼────────────────────┐           │
 │                    │              S3 Data Lake   │                    │           │
 │                    │  raw/ bronze/ silver/ gold/ │ models/ scored/    │           │
 │                    └────────────────────────────┼────────────────────┘           │
 │                                                 │                                │
 │   ┌──────────────┐    ┌──────────────┐    ┌────┴─────────┐    ┌─────────────┐   │
 │   │  CloudWatch  │    │  SNS Alerts  │    │  DynamoDB    │    │  SageMaker  │   │
 │   │  Alarms      │    │  Email       │    │  Manifest    │    │  Registry   │   │
 │   └──────────────┘    └──────────────┘    └──────────────┘    └─────────────┘   │
 │                                                                                  │
 │   ┌──────────────────────────────────────────────────────────────────────────┐   │
 │   │  Streamlit Dashboard (ECS Fargate + ALB)                                │   │
 │   │  Overview | Model Performance | Drift | Customer Risk | Recos | Status  │   │
 │   └──────────────────────────────────────────────────────────────────────────┘   │
 │                                                                                  │
 │   ┌──────────────────────────────────────────────────────────────────────────┐   │
 │   │  GitHub Actions CI/CD                                                    │   │
 │   │  CI (lint+test) ─> Terraform Apply ─> Docker Build/Push ─> ECS Deploy   │   │
 │   └──────────────────────────────────────────────────────────────────────────┘   │
 │                                                                                  │
 └──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Data Pipeline Architecture

The ETL pipeline follows a **medallion architecture** (bronze/silver/gold) implemented as SageMaker Processing Jobs orchestrated by Step Functions.

```
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                        DATA PIPELINE (Medallion ETL)                        │
 │                                                                             │
 │  RAW LAYER                 BRONZE LAYER              SILVER LAYER           │
 │  ─────────                 ────────────              ────────────           │
 │  7 source files            2 merged tables           Enriched tables        │
 │                                                                             │
 │  churn_label.csv ─────┐                                                    │
 │  customer_attributes ──┤   bronze_customer           silver_customer        │
 │  customer_contracts ───┼──> (1 row/customer)  ──────> + price imputation    │
 │  customer_interactions ┘   merge on customer_id      + segments (Res/      │
 │                                                        SME/Corp)            │
 │  consumption_hourly ───┐                              + channel flags       │
 │  price_history ────────┼──> bronze_customer_month ──> silver_customer_month │
 │  costs_by_province ────┘   (1 row/cust/month)        + gross margins       │
 │                            + tariff tier splits       + price deltas        │
 │                            + gas kWh conversion       + cost allocations    │
 │                                                                             │
 │                                                                             │
 │  GOLD LAYER                                                                 │
 │  ──────────                                                                 │
 │  gold_master (1 row/customer) ─ 40+ features across 7 tiers:               │
 │                                                                             │
 │  ┌─────────────────┬───────────────┬──────────────────┬──────────────────┐  │
 │  │ Tier 1A:        │ Tier MP Core: │ Tier MP Risk:    │ Tier 2A:         │  │
 │  │ Lifecycle       │ Market        │ Volatility       │ Behavioral       │  │
 │  │ ─────────       │ ──────        │ ──────────       │ ──────────       │  │
 │  │ months_to_      │ avg_elec_     │ elec_consump_    │ interaction_     │  │
 │  │  renewal        │  consumption  │  volatility      │  count           │  │
 │  │ tenure_months   │ avg_gas_      │ gas_consump_     │ days_since_last  │  │
 │  │ renewal_bucket  │  consumption  │  volatility      │ has_complaint    │  │
 │  │ contract_term   │ avg_elec_     │ elec_price_      │ complaint_count  │  │
 │  │ has_active_     │  price        │  trend           │ has_billing_     │  │
 │  │  contract       │ avg_gas_price │ gas_price_trend  │  issue           │  │
 │  │                 │ price_vs_     │ margin_stability │ intent_to_cancel │  │
 │  │                 │  benchmark    │ total_margin_avg │ severity_score   │  │
 │  │                 │ is_dual_fuel  │                  │                  │  │
 │  ├─────────────────┼───────────────┼──────────────────┼──────────────────┤  │
 │  │ Tier 2B:        │ Tier 3:       │ Tier 1B:         │                  │  │
 │  │ Sentiment       │ Compound      │ Interaction Str  │                  │  │
 │  │ ─────────       │ ────────      │ ───────────────  │                  │  │
 │  │ sentiment_label │ renewal_x_    │ lifecycle_string │                  │  │
 │  │ has_negative_   │  complaint    │ market_string    │                  │  │
 │  │  sentiment      │ high_risk_x_  │ behavioral_      │                  │  │
 │  │ avg_sentiment_  │  neg_sentim   │  string          │                  │  │
 │  │  score          │ tenure_x_     │                  │                  │  │
 │  │                 │  interaction  │                  │                  │  │
 │  └─────────────────┴───────────────┴──────────────────┴──────────────────┘  │
 └─────────────────────────────────────────────────────────────────────────────┘
```

---

## ML Training & Evaluation Architecture

```
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                     ML TRAINING & EVALUATION PIPELINE                       │
 │                                                                             │
 │  Gold Master                                                                │
 │      │                                                                      │
 │      ▼                                                                      │
 │  ┌──────────────────┐    ┌────────────────────────────────┐                │
 │  │ Build Training   │    │ Preprocessing Pipeline         │                │
 │  │ Set              │    │ (ColumnTransformer)            │                │
 │  │ ──────────────   │    │                                │                │
 │  │ structural fills │───>│ numeric: StandardScaler        │                │
 │  │ stratified split │    │ categorical: OneHotEncoder     │                │
 │  │ (80/20)          │    │ passthrough: binary flags      │                │
 │  └──────────────────┘    └──────────────┬─────────────────┘                │
 │                                         │                                   │
 │                                         ▼                                   │
 │                          ┌──────────────────────────┐                      │
 │                          │ Model Training            │                      │
 │                          │ ──────────────            │                      │
 │                          │ - Logistic Regression     │                      │
 │                          │ - Random Forest           │                      │
 │                          │ - XGBoost (champion)      │                      │
 │                          └──────────────┬───────────┘                      │
 │                                         │                                   │
 │                                         ▼                                   │
 │                          ┌──────────────────────────┐                      │
 │                          │ Threshold Optimization    │                      │
 │                          │ ──────────────────────    │                      │
 │                          │ Maximize precision at     │                      │
 │                          │ recall >= 0.70            │                      │
 │                          └──────────────┬───────────┘                      │
 │                                         │                                   │
 │                                         ▼                                   │
 │                          ┌──────────────────────────┐                      │
 │                          │ Evaluation & Promotion    │                      │
 │                          │ ──────────────────────    │                      │
 │                          │ PR-AUC >= 0.70 gate       │                      │
 │                          │ ROC-AUC, precision,       │                      │
 │                          │ recall, confusion matrix   │                      │
 │                          └──────────────┬───────────┘                      │
 │                                         │                                   │
 │                              ┌──────────┴──────────┐                       │
 │                              ▼                     ▼                        │
 │                     ┌──────────────┐     ┌──────────────┐                  │
 │                     │ S3: models/  │     │ SageMaker    │                  │
 │                     │ pipeline.pkl │     │ Model        │                  │
 │                     │ metadata.json│     │ Registry     │                  │
 │                     │ evaluation/  │     │ (approve/    │                  │
 │                     └──────────────┘     │  reject)     │                  │
 │                                          └──────────────┘                  │
 └─────────────────────────────────────────────────────────────────────────────┘
```

---

## Monitoring & Drift Detection Architecture

```
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                    MONITORING & DRIFT DETECTION                             │
 │                                                                             │
 │  ┌─────────────────────┐     ┌──────────────────────────────┐              │
 │  │ Reference Store     │     │ Current Scoring Run          │              │
 │  │ (S3 JSON)           │     │ (gold_master + scored)       │              │
 │  │ feature_distributions│    └──────────────┬───────────────┘              │
 │  │ prediction_histogram │                   │                               │
 │  └──────────┬──────────┘                    │                               │
 │             │                               │                               │
 │             ▼                               ▼                               │
 │  ┌──────────────────────────────────────────────────┐                      │
 │  │ Drift Detection (KS Test)                        │                      │
 │  │ ─────────────────────────                        │                      │
 │  │ For each numeric feature:                        │                      │
 │  │   KS statistic + p-value vs reference            │                      │
 │  │   Drift if p < 0.01                              │                      │
 │  │                                                  │                      │
 │  │ Prediction drift:                                │                      │
 │  │   KS test on churn_proba distribution            │                      │
 │  └──────────────────┬───────────────────────────────┘                      │
 │                     │                                                       │
 │          ┌──────────┴──────────┐                                           │
 │          ▼                     ▼                                            │
 │  ┌──────────────┐     ┌──────────────────────────────┐                    │
 │  │ S3: drift    │     │ Alerting                      │                    │
 │  │ results JSON │     │ ────────                      │                    │
 │  └──────────────┘     │ SNS email if drift detected   │                    │
 │                       │ CloudWatch metric: drift_score │                    │
 │                       │ CloudWatch alarm threshold     │                    │
 │                       └──────────────────────────────┘                    │
 │                                                                             │
 │  ┌──────────────────────────────────────────────────────────────────┐      │
 │  │ Data Quality Checks (every pipeline run)                         │      │
 │  │ ────────────────────                                             │      │
 │  │ - Null rate thresholds per layer (bronze: 20%, silver: 5%,       │      │
 │  │   gold: 2%)                                                      │      │
 │  │ - Duplicate key detection                                        │      │
 │  │ - Schema validation (expected columns + dtypes)                  │      │
 │  │ - Numeric range checks                                           │      │
 │  └──────────────────────────────────────────────────────────────────┘      │
 └─────────────────────────────────────────────────────────────────────────────┘
```

---

## Serving & Dashboard Architecture

```
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                     STREAMLIT DASHBOARD (ECS Fargate)                       │
 │                                                                             │
 │  Internet ──> ALB (:80) ──> ECS Task (:8501) ──> Streamlit App             │
 │                                                                             │
 │  ┌───────────────────────────────────────────────────────────────────────┐  │
 │  │                                                                       │  │
 │  │  ┌──────────┐ ┌──────────────┐ ┌───────────┐ ┌──────────────────┐   │  │
 │  │  │ Overview │ │ Model Perf   │ │ Drift     │ │ Customer Risk    │   │  │
 │  │  │ ──────── │ │ ──────────   │ │ Monitor   │ │ ─────────────    │   │  │
 │  │  │ KPI cards│ │ PR-AUC       │ │ ───────── │ │ Risk tier pie    │   │  │
 │  │  │ At-risk %│ │ ROC-AUC      │ │ KS stats  │ │ Critical/High   │   │  │
 │  │  │ Monthly  │ │ Confusion    │ │ Feature   │ │ Expected loss    │   │  │
 │  │  │ loss     │ │ matrix       │ │ drift tbl │ │ Filterable       │   │  │
 │  │  │ Pipeline │ │ Precision    │ │ Bar chart │ │ customer table   │   │  │
 │  │  │ runs     │ │ Recall       │ │ Pred drift│ │                  │   │  │
 │  │  └──────────┘ └──────────────┘ └───────────┘ └──────────────────┘   │  │
 │  │                                                                       │  │
 │  │  ┌──────────────────┐  ┌──────────────────────────────────────────┐  │  │
 │  │  │ Recommendations  │  │ Pipeline Status                          │  │  │
 │  │  │ ───────────────  │  │ ───────────────                          │  │  │
 │  │  │ Action breakdown │  │ Total/completed/failed/in-progress runs  │  │  │
 │  │  │ By risk tier     │  │ Filterable run history table             │  │  │
 │  │  │ Retention offers │  │ Status from DynamoDB manifest            │  │  │
 │  │  │ Timing windows   │  │                                          │  │  │
 │  │  └──────────────────┘  └──────────────────────────────────────────┘  │  │
 │  └───────────────────────────────────────────────────────────────────────┘  │
 │                                                                             │
 │  Data Sources:                                                              │
 │  - S3: scored/scored_customers.parquet, models/evaluation.json,             │
 │         monitoring/drift_results.json, scored/recommendations.parquet       │
 │  - DynamoDB: pipeline manifest table (run history)                          │
 │  - Configured via env vars: DATA_SOURCE, S3_BUCKET, AWS_REGION             │
 └─────────────────────────────────────────────────────────────────────────────┘
```

---

## CI/CD & Deployment Architecture

```
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                        CI/CD (GitHub Actions)                               │
 │                                                                             │
 │  push to feature/**  ──────>  CI Workflow                                   │
 │  pull_request to main ────>  ┌─────────────────────────┐                   │
 │                              │ 1. ruff check (lint)     │                   │
 │                              │ 2. pytest (130+ tests)   │                   │
 │                              │ 3. coverage upload       │                   │
 │                              └─────────────────────────┘                   │
 │                                                                             │
 │  push to main  ───────────>  Deploy Workflow                                │
 │                              ┌─────────────────────────┐                   │
 │                              │ 1. CI (lint + test)      │                   │
 │                              │ 2. AWS OIDC login        │                   │
 │                              │ 3. terraform init/plan/  │                   │
 │                              │    apply                 │                   │
 │                              │ 4. Docker build + push:  │                   │
 │                              │    - Lambda image        │                   │
 │                              │    - Processing image    │                   │
 │                              │    - Streamlit image     │                   │
 │                              │ 5. Update Lambda code    │                   │
 │                              │ 6. Force ECS redeploy    │                   │
 │                              └─────────────────────────┘                   │
 │                                                                             │
 │  schedule (Mon 06:00 UTC)    Retrain Workflow                               │
 │  or manual dispatch  ──────> ┌─────────────────────────┐                   │
 │                              │ 1. AWS OIDC login        │                   │
 │                              │ 2. Start Step Functions   │                   │
 │                              │    execution              │                   │
 │                              │ 3. Poll for completion    │                   │
 │                              │    (60 min timeout)       │                   │
 │                              └─────────────────────────┘                   │
 │                                                                             │
 │  Auth: GitHub OIDC ──> IAM Identity Provider ──> Deploy Role (scoped)      │
 └─────────────────────────────────────────────────────────────────────────────┘
```

---

## Infrastructure Architecture

```
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                TERRAFORM INFRASTRUCTURE (11 modules)                        │
 │                                                                             │
 │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────────┐       │
 │  │ S3       │  │ DynamoDB │  │ ECR      │  │ IAM                  │       │
 │  │ ──       │  │ ──────── │  │ ───      │  │ ───                  │       │
 │  │ Data     │  │ Manifest │  │ Lambda   │  │ Lambda role          │       │
 │  │ bucket   │  │ table    │  │ Process  │  │ SFN role             │       │
 │  │ Lifecycle│  │ PAY_PER_ │  │ Streamlit│  │ SageMaker role       │       │
 │  │ rules    │  │ REQUEST  │  │ repos    │  │ ECS execution role   │       │
 │  │ Encrypt  │  │          │  │          │  │ ECS task role         │       │
 │  └──────────┘  └──────────┘  └──────────┘  └──────────────────────┘       │
 │                                                                             │
 │  ┌──────────┐  ┌──────────────┐  ┌──────────┐  ┌──────────────────┐       │
 │  │ Lambda   │  │ Step         │  │ SageMaker│  │ Monitoring       │       │
 │  │ ──────   │  │ Functions    │  │ ──────── │  │ ──────────       │       │
 │  │ Pipeline │  │ ──────────── │  │ Model    │  │ SNS topic        │       │
 │  │ trigger  │  │ 7-step ASL   │  │ Package  │  │ CW alarms:       │       │
 │  │ S3 event │  │ workflow     │  │ Group    │  │  lambda-errors   │       │
 │  │ notify   │  │ .sync calls  │  │          │  │  sfn-failures    │       │
 │  └──────────┘  └──────────────┘  └──────────┘  │  drift-detected  │       │
 │                                                  └──────────────────┘       │
 │  ┌──────────────┐  ┌──────────────────┐  ┌────────────────────────┐       │
 │  │ Networking   │  │ ECS              │  │ GitHub OIDC            │       │
 │  │ ──────────   │  │ ───              │  │ ───────────            │       │
 │  │ Default VPC  │  │ Fargate cluster  │  │ Identity provider      │       │
 │  │ ALB sec grp  │  │ Task definition  │  │ Deploy role            │       │
 │  │ ECS sec grp  │  │ Service          │  │ Scoped permissions     │       │
 │  │              │  │ ALB + listener   │  │                        │       │
 │  │              │  │ Target group     │  │                        │       │
 │  │              │  │ CW log group     │  │                        │       │
 │  └──────────────┘  └──────────────────┘  └────────────────────────┘       │
 │                                                                             │
 │  State Backend: S3 (spanishgas-terraform-state) + DynamoDB lock table      │
 └─────────────────────────────────────────────────────────────────────────────┘
```

---

## Repository Structure

```
.
├── configs/
│   ├── settings.py                  # Settings dataclass with dotenv loading
│   ├── feature_tiers.yaml           # 7 feature tiers + 9 experiment definitions
│   └── column_registry.yaml         # Raw dataset schemas, dtypes, structural fills
├── src/
│   ├── data/
│   │   ├── ingest.py                # Raw loading, bronze customer & customer-month
│   │   ├── silver.py                # Price imputation, segmentation, margins
│   │   └── build_training_set.py    # Model matrix, structural fills, stratified split
│   ├── features/
│   │   └── build_features.py        # 7 feature tiers, gold master builder
│   ├── models/
│   │   ├── preprocessing.py         # ColumnTransformer pipeline (scale + encode)
│   │   ├── churn_model.py           # Model defs, threshold optimization, evaluation
│   │   ├── scorer.py                # Batch scoring, risk tier assignment
│   │   ├── artifacts.py             # Save/load sklearn pipelines + metadata to S3
│   │   └── registry.py              # SageMaker Model Registry wrapper
│   ├── reco/
│   │   ├── schema.py                # Recommendation dataclass with guardrails
│   │   └── engine.py                # Risk tier -> retention action mapping
│   ├── pipelines/
│   │   ├── lambda_handler.py        # S3 trigger -> DynamoDB check -> SFN start
│   │   ├── manifest.py              # DynamoDB manifest with conditional writes
│   │   ├── s3_io.py                 # Read/write parquet/CSV/JSON via boto3
│   │   ├── run.py                   # Local pipeline runner (filesystem I/O)
│   │   └── steps/
│   │       ├── bronze_step.py       # SageMaker Processing: raw -> bronze
│   │       ├── silver_step.py       # SageMaker Processing: bronze -> silver
│   │       ├── gold_step.py         # SageMaker Processing: silver -> gold
│   │       ├── train_step.py        # SageMaker Processing: train XGBoost
│   │       ├── evaluate_step.py     # SageMaker Processing: evaluate + promote gate
│   │       ├── score_step.py        # SageMaker Processing: batch scoring
│   │       └── drift_step.py        # SageMaker Processing: KS drift detection
│   ├── monitoring/
│   │   ├── drift.py                 # KS-test feature + prediction drift
│   │   ├── data_quality.py          # Null rates, duplicates, schema, ranges
│   │   ├── alerts.py                # SNS publish + CloudWatch metrics
│   │   └── reference_store.py       # Save/load reference distributions (S3 JSON)
│   └── serving/ui/
│       ├── app.py                   # Streamlit main entry, sidebar navigation
│       ├── data_loader.py           # Cached loading (local / S3 / DynamoDB)
│       └── pages/
│           ├── overview.py          # Executive KPI dashboard
│           ├── model_performance.py # PR-AUC, confusion matrix, metrics
│           ├── drift_monitor.py     # Feature drift table + KS bar chart
│           ├── customer_risk.py     # Risk tier distribution, customer table
│           ├── recommendations.py   # Retention actions, filterable table
│           └── pipeline_status.py   # Run history from DynamoDB manifest
├── infra/terraform/
│   ├── main.tf                      # Provider, 11 module blocks
│   ├── backend.tf                   # S3 remote state backend
│   ├── variables.tf                 # All configurable variables
│   ├── outputs.tf                   # Key ARNs, URLs, DNS names
│   ├── environments/
│   │   ├── dev.tfvars
│   │   ├── staging.tfvars
│   │   └── prod.tfvars
│   └── modules/
│       ├── s3/                      # Data bucket with lifecycle
│       ├── dynamodb/                # Manifest table
│       ├── ecr/                     # 3 container repos (lambda, processing, streamlit)
│       ├── iam/                     # 5 IAM roles (lambda, sfn, sagemaker, ecs x2)
│       ├── lambda/                  # Pipeline trigger + S3 notification
│       ├── step_functions/          # State machine + ASL definition
│       ├── sagemaker/               # Model Package Group
│       ├── monitoring/              # SNS + CloudWatch alarms
│       ├── networking/              # Default VPC, ALB + ECS security groups
│       ├── ecs/                     # Fargate cluster, ALB, task def, service
│       └── github_oidc/             # OIDC provider + deploy role
├── .github/workflows/
│   ├── ci.yml                       # Lint + test on push/PR
│   ├── deploy.yml                   # Terraform + Docker + ECS on push to main
│   └── retrain.yml                  # Weekly/manual Step Functions trigger
├── tests/                           # 17 test files, 130+ tests
├── notebooks/                       # Source Jupyter notebooks (01: ETL, 02: modeling)
├── Dockerfile.lambda                # Lambda container image
├── Dockerfile.processing            # SageMaker Processing container image
├── Dockerfile.streamlit             # Streamlit dashboard container image
├── Makefile                         # install, lint, test, docker, terraform targets
├── pyproject.toml                   # Dependencies, ruff config, pytest config
└── .env.example                     # All environment variables template
```

---

## Getting Started

### Prerequisites

- **Python 3.10+** (3.12 recommended)
- **Docker** (for container builds)
- **AWS CLI v2** (for deployment)
- **Terraform >= 1.5** (for infrastructure)
- **Git**

### Local Development Setup

```bash
# 1. Clone the repository
git clone https://github.com/rawad-yared/bcgx-spanishgas-g1.git
cd bcgx-spanishgas-g1

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies (runtime + dev)
make install
# or manually:
pip install -e ".[dev]"

# 4. Copy environment template
cp .env.example .env
# Edit .env with your AWS credentials and settings

# 5. Verify everything works
make lint    # ruff check — should pass clean
make test    # pytest — should pass 130+ tests
```

### Running the Pipeline Locally

The local pipeline runner uses filesystem I/O instead of S3/SageMaker:

```bash
# Place your 7 raw data files in data/raw/
mkdir -p data/raw
cp /path/to/churn_label.csv data/raw/
cp /path/to/customer_attributes.csv data/raw/
cp /path/to/customer_contracts.csv data/raw/
cp /path/to/consumption_hourly_2024.csv data/raw/
cp /path/to/price_history.csv data/raw/
cp /path/to/costs_by_province_month.csv data/raw/
cp /path/to/customer_interactions.json data/raw/

# Run the pipeline (creates bronze/ silver/ gold/ scored/ directories)
python -m src.pipelines.run
```

### Running the Dashboard Locally

```bash
# Option A: Direct
make streamlit
# or:
streamlit run src/serving/ui/app.py --server.headless=true

# Option B: Docker
make docker-build-streamlit
make docker-run-streamlit
# Then open http://localhost:8501
```

The dashboard works in local mode by default — it reads from `data/scored/`, `data/models/`, and `data/monitoring/` directories. Set `DATA_SOURCE=s3` to read from AWS.

### Running Tests

```bash
# All tests
make test

# With coverage report
make test-cov

# Skip slow E2E tests
pytest tests/ -m "not slow"

# Run a specific test file
pytest tests/test_drift.py -v
```

---

## AWS Deployment

### Terraform Setup

```bash
# 1. Create the state backend (one-time, via AWS Console or CLI)
aws s3 mb s3://YOUR-PROJECT-terraform-state --region eu-west-1
aws s3api put-bucket-versioning \
  --bucket YOUR-PROJECT-terraform-state \
  --versioning-configuration Status=Enabled
aws dynamodb create-table \
  --table-name YOUR-PROJECT-terraform-locks \
  --attribute-definitions AttributeName=LockID,AttributeType=S \
  --key-schema AttributeName=LockID,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST \
  --region eu-west-1

# 2. Update infra/terraform/backend.tf with your bucket/table names

# 3. Initialize and deploy
cd infra/terraform
terraform init
terraform plan -var-file=environments/dev.tfvars
terraform apply -var-file=environments/dev.tfvars
```

### Docker Images

```bash
# Login to ECR
aws ecr get-login-password --region eu-west-1 | \
  docker login --username AWS --password-stdin YOUR_ACCOUNT_ID.dkr.ecr.eu-west-1.amazonaws.com

# Build and push all 3 images (use --provenance=false for Lambda compatibility)
ECR=YOUR_ACCOUNT_ID.dkr.ecr.eu-west-1.amazonaws.com

docker build --provenance=false -f Dockerfile.lambda -t $ECR/YOUR-PROJECT-dev-lambda:latest .
docker push $ECR/YOUR-PROJECT-dev-lambda:latest

docker build --provenance=false -f Dockerfile.processing -t $ECR/YOUR-PROJECT-dev-processing:latest .
docker push $ECR/YOUR-PROJECT-dev-processing:latest

docker build --provenance=false -f Dockerfile.streamlit -t $ECR/YOUR-PROJECT-dev-streamlit:latest .
docker push $ECR/YOUR-PROJECT-dev-streamlit:latest
```

### Deploying Infrastructure

After `terraform apply`, note the key outputs:

```bash
terraform output
# s3_bucket_name           = "spanishgas-data-dev"
# lambda_function_arn      = "arn:aws:lambda:..."
# state_machine_arn        = "arn:aws:states:..."
# streamlit_alb_dns_name   = "spanishgas-dev-streamlit-alb-XXXX.eu-west-1.elb.amazonaws.com"
# github_deploy_role_arn   = "arn:aws:iam::...:role/spanishgas-dev-github-deploy-role"
```

The Streamlit dashboard will be accessible at the `streamlit_alb_dns_name` URL.

### Triggering the Pipeline

```bash
# Upload raw data to S3 — this auto-triggers the full pipeline
aws s3 cp data/raw/ s3://YOUR-BUCKET/raw/ --recursive

# Or manually start via Step Functions
aws stepfunctions start-execution \
  --state-machine-arn YOUR_SFN_ARN \
  --input '{"mode": "train", "run_id": "manual-001", "bucket": "YOUR-BUCKET", "file_key": "raw/churn_label.csv"}'
```

### GitHub Actions CI/CD

To enable automated deployments:

1. Run `terraform output github_deploy_role_arn` to get the OIDC deploy role ARN
2. Go to your GitHub repo **Settings > Secrets and Variables > Actions**
3. Add secret: `AWS_DEPLOY_ROLE_ARN` = the role ARN from step 1
4. Pushes to `main` will now auto-deploy (Terraform + Docker + ECS)

---

## Feature Engineering

The gold master contains **40+ features** organized into 7 tiers, tested across 9 experiment configurations:

| Experiment | Tiers Included | Description |
|------------|---------------|-------------|
| E0 | 1A | Lifecycle features only (baseline) |
| E1 | 1A + MP Core | Add market/pricing features |
| E2 | 1A + MP Core + MP Risk | Add volatility/risk metrics |
| E3 | + 2A Behavioral | Add interaction features |
| E4 | + 2B Sentiment | Add sentiment analysis |
| **E5** | **+ 3 Compound** | **Full set (champion)** |
| E6 | + 1B Strings | Full + interaction strings (linear models) |
| E7 | MP Core + MP Risk | Market features only (ablation) |
| E8 | 2A + 2B | Behavioral only (ablation) |

---

## Model Details

| Property | Value |
|----------|-------|
| Algorithm | XGBoost (E5 experiment) |
| Target | `churned_within_horizon` (binary) |
| Threshold | Optimized for precision at recall >= 0.70 |
| Promotion gate | PR-AUC >= 0.70 |
| Risk tiers | Low (<40%), Medium (40-60%), High (60-80%), Critical (>80%) |
| Retraining | Weekly (Monday 06:00 UTC) or manual trigger |

---

## Recommendation Engine

The recommendation engine maps risk tiers to retention actions with policy guardrails:

| Risk Tier | Action | Timing Window |
|-----------|--------|---------------|
| Low (<40%) | No offer | 60-90 days |
| Medium (40-60%) | Small offer | 30-60 days |
| High (60-80%) | Medium offer | Immediate |
| Critical (>80%) | Large offer | Immediate |

**Guardrails:**
- No negative-margin offers: if `expected_margin_impact < 0`, action forced to `no_offer`
- Explainability required: every recommendation must include non-empty `reason_codes`
- Risk score validated to [0, 1] range

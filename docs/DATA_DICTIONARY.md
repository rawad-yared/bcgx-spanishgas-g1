# SpanishGas Data Dictionary (Initial)

This initial data dictionary defines contract-level expectations for source datasets
and key gold outputs, aligned to:

- `/Users/rawadyared/bcgx-spanishgas-g1/docs/BRD.md` (Section 8, Data Requirements)
- `/Users/rawadyared/bcgx-spanishgas-g1/docs/ARCHITECTURE.md` (Section 4.3, Core Gold Tables)

## Source datasets (BRD)

### `customer_attributes.csv`
- Layer: Source (raw)
- Path: `s3://spanishgas-data-g1/raw/customer_attributes.csv`
- Primary key: `customer_id`
- Time column(s): `signup_date`
- Grain: One row per customer
- Columns:
  - `customer_id` (string)
  - `province` (string)
  - `customer_type` (string)
  - `tariff_type` (string)
  - `signup_date` (date)
  - `product_bundle` (string)

### `customer_contracts.csv`
- Layer: Source (raw)
- Path: `s3://spanishgas-data-g1/raw/customer_contracts.csv`
- Primary key: `contract_id`
- Time column(s): `contract_start_date`, `contract_end_date`
- Grain: One row per contract
- Columns:
  - `contract_id` (string)
  - `customer_id` (string)
  - `contract_start_date` (date)
  - `contract_end_date` (date)
  - `contract_status` (string)
  - `contract_term_months` (int)
  - `product_type` (string)

### `price_history.csv`
- Layer: Source (raw)
- Path: `s3://spanishgas-data-g1/raw/price_history.csv`
- Primary key: `price_date`, `product_type`, `tariff_type`, `region_code`
- Time column(s): `price_date`
- Grain: Daily row per product/tariff/region
- Columns:
  - `price_date` (date)
  - `product_type` (string)
  - `tariff_type` (string)
  - `region_code` (string)
  - `price_eur_per_kwh` (float)
  - `market_benchmark_eur_per_kwh` (float)

### `consumption_hourly_2024.csv`
- Layer: Source (raw)
- Path: `s3://spanishgas-data-g1/raw/consumption_hourly_2024.csv`
- Primary key: `customer_id`, `timestamp_utc`, `commodity`
- Time column(s): `timestamp_utc`
- Grain: Hourly row per customer and commodity
- Columns:
  - `customer_id` (string)
  - `timestamp_utc` (timestamp)
  - `commodity` (string)
  - `consumption_kwh` (float)
  - `meter_id` (string)
  - `source_system` (string)

### `customer_interactions.json`
- Layer: Source (raw)
- Path: `s3://spanishgas-data-g1/raw/customer_interactions.json`
- Primary key: `interaction_id`
- Time column(s): `interaction_ts`
- Grain: Event row per interaction
- Columns:
  - `interaction_id` (string)
  - `customer_id` (string)
  - `interaction_ts` (timestamp)
  - `channel` (string)
  - `interaction_type` (string)
  - `sentiment_score` (float)
  - `resolution_status` (string)
  - `agent_id` (string)

### `costs_by_province_month.csv`
- Layer: Source (raw)
- Path: `s3://spanishgas-data-g1/raw/costs_by_province_month.csv`
- Primary key: `cost_month`, `province`, `commodity`
- Time column(s): `cost_month`
- Grain: Monthly row per province and commodity
- Columns:
  - `cost_month` (date)
  - `province` (string)
  - `commodity` (string)
  - `variable_cost_eur_per_kwh` (float)
  - `fixed_cost_eur_month` (float)
  - `network_cost_eur_per_kwh` (float)

### `churn_label.csv`
- Layer: Source (raw)
- Path: `s3://spanishgas-data-g1/raw/churn_label.csv`
- Primary key: `customer_id`, `label_date`, `horizon_days`
- Time column(s): `label_date`, `churn_effective_date`
- Grain: One label row per customer, label date, and horizon
- Columns:
  - `customer_id` (string)
  - `label_date` (date)
  - `horizon_days` (int)
  - `churned_within_horizon` (int)
  - `churn_effective_date` (date)

## Gold outputs (Architecture)

### `customer_snapshot_daily`
- Layer: Gold
- Path: `data/gold/customer_snapshot_daily/`
- Primary key: `customer_id`, `snapshot_date`
- Time column(s): `snapshot_date`
- Grain: Daily snapshot per customer
- Columns:
  - `customer_id` (string)
  - `snapshot_date` (date)
  - `active_contract_count` (int)
  - `days_to_contract_end` (int)
  - `latest_price_eur_per_kwh` (float)
  - `consumption_30d_kwh` (float)
  - `interaction_count_30d` (int)

### `customer_snapshot_monthly`
- Layer: Gold
- Path: `data/gold/customer_snapshot_monthly/`
- Primary key: `customer_id`, `snapshot_month`
- Time column(s): `snapshot_month`
- Grain: Monthly snapshot per customer
- Columns:
  - `customer_id` (string)
  - `snapshot_month` (date)
  - `monthly_consumption_kwh` (float)
  - `avg_price_eur_per_kwh` (float)
  - `monthly_interaction_count` (int)
  - `active_contract_count` (int)

### `customer_features_asof_date`
- Layer: Gold
- Path: `data/gold/customer_features_asof_date/asof_date=YYYY-MM-DD/`
- Primary key: `customer_id`, `asof_date`, `feature_version`
- Time column(s): `asof_date`
- Grain: As-of snapshot per customer
- Columns:
  - `customer_id` (string)
  - `asof_date` (date)
  - `feature_version` (string)
  - `tenure_days` (int)
  - `days_to_contract_end` (int)
  - `price_vs_benchmark_delta` (float)
  - `consumption_volatility_90d` (float)
  - `interaction_count_90d` (int)
  - `negative_consumption_flag` (int)

### `churn_training_dataset`
- Layer: Gold
- Path: `data/gold/churn_training_dataset/`
- Primary key: `customer_id`, `asof_date`, `label_horizon_days`
- Time column(s): `asof_date`
- Grain: One training row per customer and as-of date
- Columns:
  - `customer_id` (string)
  - `asof_date` (date)
  - `label_horizon_days` (int)
  - `churn_label` (int)
  - `split` (string)
  - `feature_version` (string)

### `recommendation_candidates`
- Layer: Gold
- Path: `data/gold/recommendation_candidates/`
- Primary key: `customer_id`, `run_date`
- Time column(s): `run_date`
- Grain: One recommendation row per customer and run date
- Columns:
  - `customer_id` (string)
  - `run_date` (date)
  - `risk_score` (float)
  - `segment_id` (string)
  - `recommended_action` (string)
  - `timing_window` (string)
  - `expected_margin_impact_eur` (float)
  - `reason_codes` (array<string>)

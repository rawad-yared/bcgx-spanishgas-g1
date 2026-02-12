# Offer Simulation Report

Input path: `data/gold/recommendation_candidates/run_date=2026-02-09/recommendation_candidates.csv`
Simulated rows: `5`
Skipped rows: `1`

## Assumptions

- Default acceptance probability: `0.2500`
- Retention given acceptance: `1.0000`
- Discount interpreted as fraction of margin: `True`

## Overall Metrics

- Expected retained margin: `62.710802`
- Expected offer cost: `9.909700`
- Incremental margin: `59.495400`
- ROI: `6.003753897696196`

## Segment and Risk Bucket Metrics

| Segment | Risk Bucket | Rows | Expected Retained Margin | Incremental Margin | ROI |
| --- | --- | ---: | ---: | ---: | ---: |
| high_value | high | 2 | 54.856700 | 53.302500 | 7.423747 |
| price_sensitive | low | 1 | 1.923750 | 1.462500 | 2.600000 |
| price_sensitive | medium | 1 | 5.385600 | 4.356000 | 2.200000 |
| stable_low_risk | low | 1 | 0.544752 | 0.374400 | 2.000000 |
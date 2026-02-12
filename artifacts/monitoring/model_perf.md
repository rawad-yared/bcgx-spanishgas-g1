# Model Performance Monitoring

- Scoring path: `data/pipeline/gold/recommendation_candidates/run_date=2026-02-09/recommendation_candidates.jsonl`
- Labels path: `data/pipeline/silver/churn_label/run_date=2026-02-09/churn_label.jsonl`
- Evaluated rows: `6`
- Current bucket: `2026-02-09`
- Baseline bucket: `2026-02-09`

## Overall Performance By Time Bucket

| Time Bucket | Rows | Positive Rate | PR-AUC | Recall@K | Precision@K | ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-02-09 | 6 | 0.5000 | 1.0000 | 1.0000 | 1.0000 | 0.0315 |

## Segment Metrics (Current Bucket: 2026-02-09)

| Segment | Rows | PR-AUC | Recall@K | ECE | Calibration Drift vs Baseline |
| --- | ---: | ---: | ---: | ---: | ---: |
| S00 | 3 | 1.0000 | 0.6667 | 0.0000 | 0.0000 |
| S01 | 2 | 0.0000 | 0.0000 | 0.0930 | 0.0000 |
| S02 | 1 | 0.0000 | 0.0000 | 0.0029 | 0.0000 |

## Retraining Trigger

- Trigger: `False`
- Reason: `none`
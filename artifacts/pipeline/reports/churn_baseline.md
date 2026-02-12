# Churn Baseline Evaluation

Primary evaluation split: `test`

## Metrics

| Split | Rows | PR-AUC | Recall@K | Precision@K | Brier | ECE | Positive Rate | Avg Pred |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 6 | 1.0000 | 0.6667 | 1.0000 | 0.0000 | 0.0001 | 0.5000 | 0.5000 |
| valid | 6 | 1.0000 | 0.6667 | 1.0000 | 0.0000 | 0.0015 | 0.5000 | 0.5015 |
| test | 6 | 1.0000 | 0.6667 | 1.0000 | 0.0035 | 0.0315 | 0.5000 | 0.5315 |

## Artifacts

- Model: `artifacts/pipeline/models/churn_baseline/model.pkl`
- Metrics JSON: `artifacts/pipeline/models/churn_baseline/metrics.json`

## Reproducibility

- Random seed: `42`
- Max iterations: `500`
- Regularization C: `1.0`
- Top-K fraction: `0.2`
- Calibration bins: `5`
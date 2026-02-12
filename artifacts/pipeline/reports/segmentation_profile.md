# Segmentation Profile

As-of date: `2026-01-31`
Requested segments: `3`
Effective segments: `3`

## Segment Summary

| Segment | Size | Share | Churn Rate | Margin Proxy |
| --- | ---: | ---: | ---: | --- |
| S00 | 3 | 0.5000 | 1.0000 | price_vs_benchmark_delta=0.065000 |
| S01 | 2 | 0.3333 | 0.0000 | price_vs_benchmark_delta=-0.030000 |
| S02 | 1 | 0.1667 | 0.0000 | price_vs_benchmark_delta=-0.030000 |

## Top Drivers by Segment

### S00
- `interaction_count_90d`: segment_mean=2.0, global_mean=1.5, z_score=1.0000
- `negative_consumption_flag`: segment_mean=1.0, global_mean=0.5, z_score=1.0000
- `price_vs_benchmark_delta`: segment_mean=0.065, global_mean=0.0175, z_score=1.0000
- `consumption_volatility_90d`: segment_mean=5.9890316666666665, global_mean=3.0425195, z_score=0.9965

### S01
- `interaction_count_90d`: segment_mean=1.0, global_mean=1.5, z_score=-1.0000
- `negative_consumption_flag`: segment_mean=0.0, global_mean=0.5, z_score=-1.0000
- `price_vs_benchmark_delta`: segment_mean=-0.03, global_mean=0.0175, z_score=-1.0000
- `consumption_volatility_90d`: segment_mean=0.103186, global_mean=3.0425195, z_score=-0.9940

### S02
- `days_to_contract_end`: segment_mean=293.0, global_mean=141.66666666666666, z_score=1.2239
- `consumption_volatility_90d`: segment_mean=0.08165, global_mean=3.0425195, z_score=-1.0013
- `interaction_count_90d`: segment_mean=1.0, global_mean=1.5, z_score=-1.0000
- `negative_consumption_flag`: segment_mean=0.0, global_mean=0.5, z_score=-1.0000

# Evaluation Report — Suicide Risk Detection

**Timestamp:** 2026-07-16T22:33:24.422516
**Agent:** gpt-4o-mini-suicide-detector
**Model:** openai/gpt-4o-mini
**Task:** suicide_risk_classification
**Dataset:** rsd_new_200_small_2.csv (n=200)
**Duration:** 7.8724 s
**Agent Mode:** multi-class-chance
**Base Class:** Unsupportive
**Max Host Iterations:** 10
**Workers:** 50 | **Retries:** 3 | **Seed:** 42
**Cache Hit Rate:** 0.0% (0/200)

## Label Distribution (ground truth)

| Label | Count | % |
|---|---|---|
| Unsupportive | 100 | 50.0% |
| Ideation | 25 | 12.5% |
| Behavior | 25 | 12.5% |
| Indicator | 25 | 12.5% |
| Attempt | 25 | 12.5% |

## Aggregate Metrics

- **accuracy**: 0.535000
- **precision**: 0.480813
- **recall**: 0.430000
- **f1_score**: 0.393136
- **avg_ttft**: 1.105754
- **avg_tps**: 262.106955
- **avg_generation_time**: 0.378335
- **avg_total_time**: 1.484089
- **total_prompt_tokens**: 71769
- **total_completion_tokens**: 8670
- **total_tokens**: 80439
- **avg_prompt_tokens**: 358.845000
- **avg_completion_tokens**: 43.350000
- **total_cost_usd**: 0.015967

## Error Report

- **error_cases**: 0
- **unique_error_keys**: 0

```json
{}
```

## Per-Class Metrics

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Unsupportive | 0.9467 | 0.7100 | 0.8114 | 100 |
| Ideation | 0.2561 | 0.8400 | 0.3925 | 25 |
| Behavior | 0.3571 | 0.2000 | 0.2564 | 25 |
| Indicator | 0.2727 | 0.2400 | 0.2553 | 25 |
| Attempt | 0.5714 | 0.1600 | 0.2500 | 25 |

## Confusion Matrix

*Rows = Actual, Columns = Predicted*

| Actual \ Predicted | Unsupportive | Ideation | Behavior | Indicator | Attempt |
|---|---|---|---|---|---|
| **Unsupportive** | 71 | 18 | 0 | 11 | 0 |
| **Ideation** | 1 | 21 | 0 | 3 | 0 |
| **Behavior** | 0 | 16 | 5 | 1 | 3 |
| **Indicator** | 3 | 16 | 0 | 6 | 0 |
| **Attempt** | 0 | 11 | 9 | 1 | 4 |


# Evaluation Report — Suicide Risk Detection

**Timestamp:** 2026-07-16T22:32:21.573885
**Agent:** gpt-4o-mini-suicide-detector
**Model:** openai/gpt-4o-mini
**Task:** suicide_risk_classification
**Dataset:** rsd_new_200_small_2.csv (n=200)
**Duration:** 9.9049 s
**Agent Mode:** single-class
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

- **accuracy**: 0.675000
- **precision**: 0.496308
- **recall**: 0.504000
- **f1_score**: 0.457468
- **avg_ttft**: 1.191469
- **avg_tps**: 74.261567
- **avg_generation_time**: 0.058281
- **avg_total_time**: 1.249750
- **total_prompt_tokens**: 101169
- **total_completion_tokens**: 542
- **total_tokens**: 101711
- **avg_prompt_tokens**: 505.845000
- **avg_completion_tokens**: 2.710000
- **total_cost_usd**: 0.015501

## Error Report

- **error_cases**: 0
- **unique_error_keys**: 0

```json
{}
```

## Per-Class Metrics

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Unsupportive | 0.8889 | 0.9600 | 0.9231 | 100 |
| Ideation | 0.3571 | 0.8000 | 0.4938 | 25 |
| Behavior | 0.2500 | 0.0400 | 0.0690 | 25 |
| Indicator | 0.3333 | 0.1200 | 0.1765 | 25 |
| Attempt | 0.6522 | 0.6000 | 0.6250 | 25 |

## Confusion Matrix

*Rows = Actual, Columns = Predicted*

| Actual \ Predicted | Unsupportive | Ideation | Behavior | Indicator | Attempt |
|---|---|---|---|---|---|
| **Unsupportive** | 96 | 1 | 0 | 3 | 0 |
| **Ideation** | 3 | 20 | 0 | 2 | 0 |
| **Behavior** | 0 | 15 | 1 | 1 | 8 |
| **Indicator** | 9 | 13 | 0 | 3 | 0 |
| **Attempt** | 0 | 7 | 3 | 0 | 15 |


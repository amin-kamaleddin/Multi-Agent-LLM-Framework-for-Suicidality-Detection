# Evaluation Report — Suicide Risk Detection

**Timestamp:** 2026-07-16T22:34:33.448717
**Agent:** gpt-4o-mini-suicide-detector
**Model:** openai/gpt-4o-mini
**Task:** suicide_risk_classification
**Dataset:** rsd_new_200_small_2.csv (n=200)
**Duration:** 20.0960 s
**Agent Mode:** per-class-score
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

- **accuracy**: 0.610000
- **precision**: 0.478667
- **recall**: 0.496000
- **f1_score**: 0.452586
- **avg_ttft**: 0.000000
- **avg_tps**: 8.134129
- **avg_generation_time**: 4.413920
- **avg_total_time**: 4.413920
- **total_prompt_tokens**: 309245
- **total_completion_tokens**: 6980
- **total_tokens**: 316225
- **avg_prompt_tokens**: 1546.225000
- **avg_completion_tokens**: 34.900000
- **total_cost_usd**: 0.050575

## Error Report

- **error_cases**: 0
- **unique_error_keys**: 0

```json
{}
```

## Per-Class Metrics

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Unsupportive | 0.9195 | 0.8000 | 0.8556 | 100 |
| Ideation | 0.3256 | 0.5600 | 0.4118 | 25 |
| Behavior | 0.2857 | 0.0800 | 0.1250 | 25 |
| Indicator | 0.4667 | 0.2800 | 0.3500 | 25 |
| Attempt | 0.3958 | 0.7600 | 0.5205 | 25 |

## Confusion Matrix

*Rows = Actual, Columns = Predicted*

| Actual \ Predicted | Unsupportive | Ideation | Behavior | Indicator | Attempt |
|---|---|---|---|---|---|
| **Unsupportive** | 80 | 13 | 1 | 6 | 0 |
| **Ideation** | 2 | 14 | 1 | 1 | 7 |
| **Behavior** | 0 | 2 | 2 | 1 | 20 |
| **Indicator** | 5 | 11 | 0 | 7 | 2 |
| **Attempt** | 0 | 3 | 3 | 0 | 19 |


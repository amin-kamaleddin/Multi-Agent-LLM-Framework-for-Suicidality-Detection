# Evaluation Report — Suicide Risk Detection

**Timestamp:** 2026-07-26T12:35:08.019573
**Agent:** gpt-4o-mini-suicide-detector
**Model:** openai/gpt-5.6-luna
**Task:** suicide_risk_classification
**Dataset:** suicidality_full_dataset_posneg.csv (n=29725)
**Duration:** 799.4902 s
**Agent Mode:** single-class
**Base Class:** Unsupportive
**Max Host Iterations:** 10
**Workers:** 50 | **Retries:** 3 | **Seed:** 42
**Cache Hit Rate:** 0.7% (217/29725)

## Label Distribution (ground truth)

| Label | Count | % |
|---|---|---|
| Unsupportive | 15329 | 51.6% |
| Ideation | 7022 | 23.6% |
| Behavior | 2039 | 6.9% |
| Indicator | 4550 | 15.3% |
| Attempt | 785 | 2.6% |

## Aggregate Metrics

- **accuracy**: 0.753978
- **precision**: 0.588614
- **recall**: 0.572670
- **f1_score**: 0.538461
- **avg_ttft**: 1.322996
- **avg_tps**: 1332.387176
- **avg_generation_time**: 0.028675
- **avg_total_time**: 1.351671
- **total_prompt_tokens**: 14779547
- **total_completion_tokens**: 845847
- **total_tokens**: 15625394
- **avg_prompt_tokens**: 500.865765
- **avg_completion_tokens**: 28.665006
- **total_cost_usd**: 19.854629

## Error Report

- **error_cases**: 0
- **unique_error_keys**: 0

```json
{}
```

## Per-Class Metrics

*Accuracy is one-vs-rest: (TP + TN) / N for each class treated as C-vs-not-C.*

| Class | Precision | Recall | F1 | Accuracy | Support |
|---|---|---|---|---|---|
| Unsupportive | 0.8379 | 0.9872 | 0.9065 | 0.8949 | 15329 |
| Ideation | 0.6788 | 0.8341 | 0.7485 | 0.8676 | 7022 |
| Behavior | 0.4407 | 0.3134 | 0.3663 | 0.9256 | 2039 |
| Indicator | 0.4904 | 0.0560 | 0.1006 | 0.8466 | 4550 |
| Attempt | 0.4953 | 0.6726 | 0.5705 | 0.9733 | 785 |

## Confusion Matrix

*Rows = Actual, Columns = Predicted*

| Actual \ Predicted | Unsupportive | Ideation | Behavior | Indicator | Attempt |
|---|---|---|---|---|---|
| **Unsupportive** | 15133 | 108 | 6 | 77 | 5 |
| **Ideation** | 267 | 5857 | 450 | 177 | 271 |
| **Behavior** | 17 | 1183 | 639 | 10 | 190 |
| **Indicator** | 2642 | 1388 | 193 | 255 | 72 |
| **Attempt** | 1 | 93 | 162 | 1 | 528 |

## Binary Metrics (Positive vs Negative)

*Negative = `Unsupportive`; Positive = any of `Ideation`, `Behavior`, `Indicator`, `Attempt`.*

- **accuracy_positive**: 0.894937
- **precision_positive**: 0.910563
- **recall_positive**: 0.891947
- **f1_positive**: 0.893316
- **counts**: TP=11469 | TN=15133 | FP=196 | FN=2927

*precision/recall/f1 above are macro-averaged over the Positive and Negative classes.*

## Binary Confusion Matrix

*Rows = Actual, Columns = Predicted*

| Actual \ Predicted | Positive | Negative |
|---|---|---|
| **Positive** | 11469 | 2927 |
| **Negative** | 196 | 15133 |


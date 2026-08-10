# Evaluation Report — Suicide Risk Detection

**Timestamp:** 2026-08-02T19:22:04.524165
**Agent:** gpt-4o-mini-suicide-detector
**Model:** openai/gpt-4o-mini
**Task:** suicide_risk_classification
**Dataset:** suicidality_full_dataset_posneg.csv (n=28792)
**Duration:** 1704.6287 s
**Agent Mode:** base-vs-rest-binary
**Base Class:** Unsupportive
**Max Host Iterations:** 10
**Workers:** 50 | **Retries:** 3 | **Seed:** 42
**Cache Hit Rate:** 0.7% (191/28792)

## Label Distribution (ground truth)

| Label | Count | % |
|---|---|---|
| Unsupportive | 14396 | 50.0% |
| Ideation | 7022 | 24.4% |
| Behavior | 2039 | 7.1% |
| Indicator | 4550 | 15.8% |
| Attempt | 785 | 2.7% |

## Aggregate Metrics

- **accuracy**: 0.290671
- **precision**: 0.309419
- **recall**: 0.364282
- **f1_score**: 0.231240
- **avg_ttft**: 0.000000
- **avg_tps**: 3.107419
- **avg_generation_time**: 2.976048
- **avg_total_time**: 2.976048
- **total_prompt_tokens**: 18215088
- **total_completion_tokens**: 250283
- **total_tokens**: 18465371
- **avg_prompt_tokens**: 636.868921
- **avg_completion_tokens**: 8.750848
- **total_cost_usd**: 2.882433

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
| Unsupportive | 0.8246 | 0.0885 | 0.1598 | 0.5348 | 14396 |
| Ideation | 0.4285 | 0.8275 | 0.5646 | 0.6887 | 7022 |
| Behavior | 0.0086 | 0.0137 | 0.0106 | 0.8179 | 2039 |
| Indicator | 0.0904 | 0.1477 | 0.1122 | 0.6306 | 4550 |
| Attempt | 0.1950 | 0.7439 | 0.3090 | 0.9093 | 785 |

## Confusion Matrix

*Rows = Actual, Columns = Predicted*

| Actual \ Predicted | Unsupportive | Ideation | Behavior | Indicator | Attempt |
|---|---|---|---|---|---|
| **Unsupportive** | 1274 | 3474 | 3023 | 6618 | 7 |
| **Ideation** | 59 | 5811 | 44 | 92 | 1016 |
| **Behavior** | 1 | 827 | 28 | 34 | 1149 |
| **Indicator** | 209 | 3280 | 150 | 672 | 239 |
| **Attempt** | 2 | 170 | 15 | 14 | 584 |

## Positive-Class Differentiation Metrics

*Computed only over cases where BOTH the true label and the predicted label are one of the 4 positive classes (`Ideation`, `Behavior`, `Indicator`, `Attempt`). Cases where either side is `Unsupportive` (or an unrecognized prediction) are excluded from this subset entirely — this isolates how well the agent tells the risk levels apart from each other, independent of the Unsupportive-vs-rest screening decision covered in the Binary Metrics section below.*

- **cases considered**: 14125 of 28792 total (14667 excluded because the true or predicted label was `Unsupportive` or unrecognized)

| Class | Precision | Recall | F1 | Support (within this subset) |
|---|---|---|---|---|
| Ideation | 0.5760 | 0.8346 | 0.6816 | 6963 |
| Behavior | 0.1181 | 0.0137 | 0.0246 | 2038 |
| Indicator | 0.8276 | 0.1548 | 0.2608 | 4341 |
| Attempt | 0.1954 | 0.7458 | 0.3097 | 783 |

- **accuracy_4class**: 0.502301
- **precision_4class_macro**: 0.429302
- **recall_4class_macro**: 0.437236
- **f1_4class_macro**: 0.319192

*precision/recall/f1 macros above are the unweighted mean across the 4 positive classes. accuracy_4class is the plain 4-label classification accuracy over the included cases only.*

### Positive-Class Confusion Matrix (4x4 subset)

*Rows = Actual, Columns = Predicted — restricted to the 4 positive classes*

| Actual \ Predicted | Ideation | Behavior | Indicator | Attempt |
|---|---|---|---|---|
| **Ideation** | 5811 | 44 | 92 | 1016 |
| **Behavior** | 827 | 28 | 34 | 1149 |
| **Indicator** | 3280 | 150 | 672 | 239 |
| **Attempt** | 170 | 15 | 14 | 584 |

## Binary Metrics (Positive vs Negative)

*Negative = `Unsupportive`; Positive = any of `Ideation`, `Behavior`, `Indicator`, `Attempt`.*

- **accuracy_positive**: 0.534836
- **precision_positive**: 0.671501
- **recall_positive**: 0.534836
- **f1_positive**: 0.419112
- **counts**: TP=14125 | TN=1274 | FP=13122 | FN=271

*precision/recall/f1 above are macro-averaged over the Positive and Negative classes.*

## Binary Confusion Matrix

*Rows = Actual, Columns = Predicted*

| Actual \ Predicted | Positive | Negative |
|---|---|---|
| **Positive** | 14125 | 271 |
| **Negative** | 13122 | 1274 |


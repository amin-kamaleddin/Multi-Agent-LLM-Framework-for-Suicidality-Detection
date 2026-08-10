# Evaluation Report — Suicide Risk Detection

**Timestamp:** 2026-08-03T16:25:54.547437
**Agent:** gpt-4o-mini-suicide-detector
**Model:** openai/gpt-5.6-luna
**Task:** suicide_risk_classification
**Dataset:** suicidality_full_dataset_posneg.csv (n=28792)
**Duration:** 391.2735 s
**Agent Mode:** base-vs-rest-binary
**Base Class:** Unsupportive
**Max Host Iterations:** 10
**Workers:** 50 | **Retries:** 3 | **Seed:** 42
**Cache Hit Rate:** 94.2% (27126/28792)

## Label Distribution (ground truth)

| Label | Count | % |
|---|---|---|
| Unsupportive | 14396 | 50.0% |
| Ideation | 7022 | 24.4% |
| Behavior | 2039 | 7.1% |
| Indicator | 4550 | 15.8% |
| Attempt | 785 | 2.7% |

## Aggregate Metrics

- **accuracy**: 0.485760
- **precision**: 0.465264
- **recall**: 0.496751
- **f1_score**: 0.435558
- **avg_ttft**: 0.000000
- **avg_tps**: 25.244879
- **avg_generation_time**: 10.243892
- **avg_total_time**: 10.243892
- **total_prompt_tokens**: 915940
- **total_completion_tokens**: 485062
- **total_tokens**: 1401002
- **avg_prompt_tokens**: 549.783914
- **avg_completion_tokens**: 291.153661
- **total_cost_usd**: 0.382631

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
| Unsupportive | 0.8983 | 0.3367 | 0.4898 | 0.6493 | 14396 |
| Ideation | 0.6728 | 0.8388 | 0.7467 | 0.8612 | 7022 |
| Behavior | 0.1097 | 0.1236 | 0.1162 | 0.8669 | 2039 |
| Indicator | 0.2238 | 0.5490 | 0.3180 | 0.6279 | 4550 |
| Attempt | 0.4218 | 0.6357 | 0.5071 | 0.9663 | 785 |

## Confusion Matrix

*Rows = Actual, Columns = Predicted*

| Actual \ Predicted | Unsupportive | Ideation | Behavior | Indicator | Attempt |
|---|---|---|---|---|---|
| **Unsupportive** | 4847 | 66 | 1598 | 7871 | 14 |
| **Ideation** | 36 | 5890 | 177 | 623 | 296 |
| **Behavior** | 1 | 1379 | 252 | 136 | 271 |
| **Indicator** | 512 | 1276 | 161 | 2498 | 103 |
| **Attempt** | 0 | 144 | 110 | 32 | 499 |

## Positive-Class Differentiation Metrics

*Computed only over cases where BOTH the true label and the predicted label are one of the 4 positive classes (`Ideation`, `Behavior`, `Indicator`, `Attempt`). Cases where either side is `Unsupportive` (or an unrecognized prediction) are excluded from this subset entirely — this isolates how well the agent tells the risk levels apart from each other, independent of the Unsupportive-vs-rest screening decision covered in the Binary Metrics section below.*

- **cases considered**: 13847 of 28792 total (14945 excluded because the true or predicted label was `Unsupportive` or unrecognized)

| Class | Precision | Recall | F1 | Support (within this subset) |
|---|---|---|---|---|
| Ideation | 0.6779 | 0.8431 | 0.7515 | 6986 |
| Behavior | 0.3600 | 0.1237 | 0.1841 | 2038 |
| Indicator | 0.7595 | 0.6186 | 0.6819 | 4038 |
| Attempt | 0.4269 | 0.6357 | 0.5107 | 785 |

- **accuracy_4class**: 0.659999
- **precision_4class_macro**: 0.556058
- **recall_4class_macro**: 0.555264
- **f1_4class_macro**: 0.532050

*precision/recall/f1 macros above are the unweighted mean across the 4 positive classes. accuracy_4class is the plain 4-label classification accuracy over the included cases only.*

### Positive-Class Confusion Matrix (4x4 subset)

*Rows = Actual, Columns = Predicted — restricted to the 4 positive classes*

| Actual \ Predicted | Ideation | Behavior | Indicator | Attempt |
|---|---|---|---|---|
| **Ideation** | 5890 | 177 | 623 | 296 |
| **Behavior** | 1379 | 252 | 136 | 271 |
| **Indicator** | 1276 | 161 | 2498 | 103 |
| **Attempt** | 144 | 110 | 32 | 499 |

## Binary Metrics (Positive vs Negative)

*Negative = `Unsupportive`; Positive = any of `Ideation`, `Behavior`, `Indicator`, `Attempt`.*

- **accuracy_positive**: 0.649278
- **precision_positive**: 0.745056
- **recall_positive**: 0.649278
- **f1_positive**: 0.611297
- **counts**: TP=13847 | TN=4847 | FP=9549 | FN=549

*precision/recall/f1 above are macro-averaged over the Positive and Negative classes.*

## Binary Confusion Matrix

*Rows = Actual, Columns = Predicted*

| Actual \ Predicted | Positive | Negative |
|---|---|---|
| **Positive** | 13847 | 549 |
| **Negative** | 9549 | 4847 |


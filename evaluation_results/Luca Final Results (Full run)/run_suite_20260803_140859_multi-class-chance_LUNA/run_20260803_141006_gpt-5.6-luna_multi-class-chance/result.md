# Evaluation Report — Suicide Risk Detection

**Timestamp:** 2026-08-03T14:34:25.005801
**Agent:** gpt-4o-mini-suicide-detector
**Model:** openai/gpt-5.6-luna
**Task:** suicide_risk_classification
**Dataset:** suicidality_full_dataset_posneg.csv (n=28792)
**Duration:** 1458.0525 s
**Agent Mode:** multi-class-chance
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

- **accuracy**: 0.744026
- **precision**: 0.593849
- **recall**: 0.580812
- **f1_score**: 0.557267
- **avg_ttft**: 2.501504
- **avg_tps**: 5286.939238
- **avg_generation_time**: 0.039729
- **avg_total_time**: 2.541233
- **total_prompt_tokens**: 10050085
- **total_completion_tokens**: 2647818
- **total_tokens**: 12697903
- **avg_prompt_tokens**: 351.389287
- **avg_completion_tokens**: 92.577812
- **total_cost_usd**: 2.593699

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
| Unsupportive | 0.8689 | 0.9548 | 0.9098 | 0.9054 | 14396 |
| Ideation | 0.6328 | 0.8509 | 0.7258 | 0.8432 | 7022 |
| Behavior | 0.4556 | 0.3727 | 0.4100 | 0.9240 | 2039 |
| Indicator | 0.4777 | 0.0989 | 0.1639 | 0.8405 | 4550 |
| Attempt | 0.5342 | 0.6268 | 0.5768 | 0.9749 | 785 |

## Confusion Matrix

*Rows = Actual, Columns = Predicted*

| Actual \ Predicted | Unsupportive | Ideation | Behavior | Indicator | Attempt |
|---|---|---|---|---|---|
| **Unsupportive** | 13745 | 340 | 25 | 281 | 5 |
| **Ideation** | 130 | 5975 | 480 | 192 | 245 |
| **Behavior** | 8 | 1142 | 760 | 18 | 111 |
| **Indicator** | 1936 | 1890 | 206 | 450 | 68 |
| **Attempt** | 0 | 95 | 197 | 1 | 492 |

## Positive-Class Differentiation Metrics

*Computed only over cases where BOTH the true label and the predicted label are one of the 4 positive classes (`Ideation`, `Behavior`, `Indicator`, `Attempt`). Cases where either side is `Unsupportive` (or an unrecognized prediction) are excluded from this subset entirely — this isolates how well the agent tells the risk levels apart from each other, independent of the Unsupportive-vs-rest screening decision covered in the Binary Metrics section below.*

- **cases considered**: 12322 of 28792 total (16470 excluded because the true or predicted label was `Unsupportive` or unrecognized)

| Class | Precision | Recall | F1 | Support (within this subset) |
|---|---|---|---|---|
| Ideation | 0.6564 | 0.8669 | 0.7472 | 6892 |
| Behavior | 0.4626 | 0.3742 | 0.4137 | 2031 |
| Indicator | 0.6808 | 0.1721 | 0.2748 | 2614 |
| Attempt | 0.5371 | 0.6268 | 0.5785 | 785 |

- **accuracy_4class**: 0.623032
- **precision_4class_macro**: 0.584231
- **recall_4class_macro**: 0.510012
- **f1_4class_macro**: 0.503541

*precision/recall/f1 macros above are the unweighted mean across the 4 positive classes. accuracy_4class is the plain 4-label classification accuracy over the included cases only.*

### Positive-Class Confusion Matrix (4x4 subset)

*Rows = Actual, Columns = Predicted — restricted to the 4 positive classes*

| Actual \ Predicted | Ideation | Behavior | Indicator | Attempt |
|---|---|---|---|---|
| **Ideation** | 5975 | 480 | 192 | 245 |
| **Behavior** | 1142 | 760 | 18 | 111 |
| **Indicator** | 1890 | 206 | 450 | 68 |
| **Attempt** | 95 | 197 | 1 | 492 |

## Binary Metrics (Positive vs Negative)

*Negative = `Unsupportive`; Positive = any of `Ideation`, `Behavior`, `Indicator`, `Attempt`.*

- **accuracy_positive**: 0.905356
- **precision_positive**: 0.909355
- **recall_positive**: 0.905356
- **f1_positive**: 0.905124
- **counts**: TP=12322 | TN=13745 | FP=651 | FN=2074

*precision/recall/f1 above are macro-averaged over the Positive and Negative classes.*

## Binary Confusion Matrix

*Rows = Actual, Columns = Predicted*

| Actual \ Predicted | Positive | Negative |
|---|---|---|
| **Positive** | 12322 | 2074 |
| **Negative** | 651 | 13745 |


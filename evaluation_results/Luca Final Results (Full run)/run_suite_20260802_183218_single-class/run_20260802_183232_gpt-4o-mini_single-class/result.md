# Evaluation Report — Suicide Risk Detection

**Timestamp:** 2026-08-02T18:32:44.749325
**Agent:** gpt-4o-mini-suicide-detector
**Model:** openai/gpt-4o-mini
**Task:** suicide_risk_classification
**Dataset:** suicidality_full_dataset_posneg.csv (n=28792)
**Duration:** 11.6791 s
**Agent Mode:** single-class
**Base Class:** Unsupportive
**Max Host Iterations:** 10
**Workers:** 50 | **Retries:** 3 | **Seed:** 42
**Cache Hit Rate:** 98.1% (28254/28792)

## Label Distribution (ground truth)

| Label | Count | % |
|---|---|---|
| Unsupportive | 14396 | 50.0% |
| Ideation | 7022 | 24.4% |
| Behavior | 2039 | 7.1% |
| Indicator | 4550 | 15.8% |
| Attempt | 785 | 2.7% |

## Aggregate Metrics

- **accuracy**: 0.725931
- **precision**: 0.508922
- **recall**: 0.529740
- **f1_score**: 0.465627
- **avg_ttft**: 0.862372
- **avg_tps**: 102.108867
- **avg_generation_time**: 0.050424
- **avg_total_time**: 0.912795
- **total_prompt_tokens**: 239873
- **total_completion_tokens**: 1872
- **total_tokens**: 241745
- **avg_prompt_tokens**: 445.860595
- **avg_completion_tokens**: 3.479554
- **total_cost_usd**: 0.037104

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
| Unsupportive | 0.8722 | 0.9480 | 0.9086 | 0.9046 | 14396 |
| Ideation | 0.6220 | 0.8630 | 0.7230 | 0.8387 | 7022 |
| Behavior | 0.2787 | 0.0667 | 0.1076 | 0.9217 | 2039 |
| Indicator | 0.4866 | 0.1200 | 0.1925 | 0.8409 | 4550 |
| Attempt | 0.2850 | 0.6510 | 0.3964 | 0.9460 | 785 |

## Confusion Matrix

*Rows = Actual, Columns = Predicted*

| Actual \ Predicted | Unsupportive | Ideation | Behavior | Indicator | Attempt |
|---|---|---|---|---|---|
| **Unsupportive** | 13648 | 416 | 13 | 311 | 8 |
| **Ideation** | 148 | 6060 | 130 | 235 | 449 |
| **Behavior** | 14 | 1160 | 136 | 23 | 706 |
| **Indicator** | 1836 | 1921 | 128 | 546 | 119 |
| **Attempt** | 1 | 185 | 81 | 7 | 511 |

## Positive-Class Differentiation Metrics

*Computed only over cases where BOTH the true label and the predicted label are one of the 4 positive classes (`Ideation`, `Behavior`, `Indicator`, `Attempt`). Cases where either side is `Unsupportive` (or an unrecognized prediction) are excluded from this subset entirely — this isolates how well the agent tells the risk levels apart from each other, independent of the Unsupportive-vs-rest screening decision covered in the Binary Metrics section below.*

- **cases considered**: 12397 of 28792 total (16395 excluded because the true or predicted label was `Unsupportive` or unrecognized)

| Class | Precision | Recall | F1 | Support (within this subset) |
|---|---|---|---|---|
| Ideation | 0.6498 | 0.8816 | 0.7481 | 6874 |
| Behavior | 0.2863 | 0.0672 | 0.1088 | 2025 |
| Indicator | 0.6732 | 0.2012 | 0.3098 | 2714 |
| Attempt | 0.2863 | 0.6518 | 0.3978 | 784 |

- **accuracy_4class**: 0.585061
- **precision_4class_macro**: 0.473907
- **recall_4class_macro**: 0.450427
- **f1_4class_macro**: 0.391139

*precision/recall/f1 macros above are the unweighted mean across the 4 positive classes. accuracy_4class is the plain 4-label classification accuracy over the included cases only.*

### Positive-Class Confusion Matrix (4x4 subset)

*Rows = Actual, Columns = Predicted — restricted to the 4 positive classes*

| Actual \ Predicted | Ideation | Behavior | Indicator | Attempt |
|---|---|---|---|---|
| **Ideation** | 6060 | 130 | 235 | 449 |
| **Behavior** | 1160 | 136 | 23 | 706 |
| **Indicator** | 1921 | 128 | 546 | 119 |
| **Attempt** | 185 | 81 | 7 | 511 |

## Binary Metrics (Positive vs Negative)

*Negative = `Unsupportive`; Positive = any of `Ideation`, `Behavior`, `Indicator`, `Attempt`.*

- **accuracy_positive**: 0.904592
- **precision_positive**: 0.907670
- **recall_positive**: 0.904592
- **f1_positive**: 0.904411
- **counts**: TP=12397 | TN=13648 | FP=748 | FN=1999

*precision/recall/f1 above are macro-averaged over the Positive and Negative classes.*

## Binary Confusion Matrix

*Rows = Actual, Columns = Predicted*

| Actual \ Predicted | Positive | Negative |
|---|---|---|
| **Positive** | 12397 | 1999 |
| **Negative** | 748 | 13648 |


# Run Metrics

- Workflow: framework_2_parallel_with_evidence
- Service tier: standard
- Stop-after token threshold: None
- Current invocation tokens: 118473800
- Total samples: 28792
- Failed records: 0
- Failure rate: 0.000000
- LLM calls: 172808
- JSON/schema parsing errors: 56
- Cumulative case execution duration seconds: 142303.897463
- Real execution duration seconds: 9614.207126
- Total latency seconds: 349115.269818
- Average latency seconds per request: 2.020249
- Average input tokens per request: 613.503443
- Average output tokens per request: 72.077086
- Average cached tokens per request: 0.013801
- Total cost USD: 18.871221
- Average TTFT seconds when exposed: n/a
- Average provider completion TPS when exposed: n/a
- Average end-to-end completion TPS: 30.762110
- Suicidal-intent grouping accuracy: 0.887121
- Suicidal-intent grouping positive-class F1: 0.874575
- Suicidal-intent TP: 11331
- Suicidal-intent TN: 14211
- Suicidal-intent FP: 185
- Suicidal-intent FN: 3065

## Suicidal-Intent Positive-Class Grouping Metrics

- Positive: Indicator, Ideation, Behavior, Attempt
- Negative: Unsupportive
- TP/TN/FP/FN use this fixed positive orientation; failed predictions remain in the Error column.

| Accuracy | F1 | Precision | Recall | Positive Support | Negative Support | TP | FP | TN | FN |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.887121 | 0.874575 | 0.983935 | 0.787094 | 14396 | 14396 | 11331 | 185 | 14211 | 3065 |

## Positive-Label Subclassification Metrics

- Scope: records whose ground-truth and predicted labels are both Indicator, Ideation, Behavior, or Attempt.
- Precision, recall, and F1 are macro averages across the four positive labels.
- Correct: included records on the diagonal of the 4x4 matrix.
- Support: all records represented in the 4x4 matrix.

Formulas for each label L:
- TP(L) = actual L and predicted L.
- FP(L) = actual another positive label and predicted L.
- FN(L) = actual L and predicted another positive label.
- Precision(L) = TP(L) / (TP(L) + FP(L)).
- Recall(L) = TP(L) / (TP(L) + FN(L)).
- F1(L) = 2 * Precision(L) * Recall(L) / (Precision(L) + Recall(L)).
- Overall accuracy = correct / support.
- Macro metric = sum of the four per-label metric values / 4.

| Accuracy | F1 Score | Precision | Recall | Support | Correct | Averaging |
|---:|---:|---:|---:|---:|---:|---|
| 0.639838 | 0.531429 | 0.573401 | 0.567605 | 11331 | 7250 | macro |

### Positive-Label 4x4 Confusion Matrix

| Actual \ Predicted | Indicator | Ideation | Behavior | Attempt |
|---|---|---|---|---|
| Indicator | 278 | 1120 | 352 | 64 |
| Ideation | 201 | 4989 | 1260 | 264 |
| Behavior | 15 | 406 | 1483 | 119 |
| Attempt | 3 | 45 | 232 | 500 |

### Per-Label Calculations

| Label | TP | FP | FN | Support | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|---:|---:|---:|
| Indicator | 278 | 219 | 1536 | 1814 | 0.559356 | 0.153252 | 0.240588 |
| Ideation | 4989 | 1571 | 1725 | 6714 | 0.760518 | 0.743074 | 0.751695 |
| Behavior | 1483 | 1844 | 540 | 2023 | 0.445747 | 0.733070 | 0.554393 |
| Attempt | 500 | 447 | 280 | 780 | 0.527983 | 0.641026 | 0.579039 |

# Run Metrics

- Workflow: framework_4_hierarchical_with_reviewer
- Service tier: standard
- Stop-after token threshold: None
- Current invocation tokens: 50724599
- Total samples: 28792
- Failed records: 1
- Failure rate: 0.000035
- LLM calls: 72977
- JSON/schema parsing errors: 23
- Cumulative case execution duration seconds: 136095.335867
- Real execution duration seconds: 17151.440877
- Total latency seconds: 136009.151157
- Average latency seconds per request: 1.863726
- Average input tokens per request: 599.248750
- Average output tokens per request: 95.827768
- Average cached tokens per request: 0.000000
- Total cost USD: 8.635986
- Average TTFT seconds when exposed: n/a
- Average provider completion TPS when exposed: n/a
- Average end-to-end completion TPS: 52.528707
- Suicidal-intent grouping accuracy: 0.897923
- Suicidal-intent grouping positive-class F1: 0.888543
- Suicidal-intent TP: 11711
- Suicidal-intent TN: 14142
- Suicidal-intent FP: 254
- Suicidal-intent FN: 2684

## Suicidal-Intent Positive-Class Grouping Metrics

- Positive: Indicator, Ideation, Behavior, Attempt
- Negative: Unsupportive
- TP/TN/FP/FN use this fixed positive orientation; failed predictions remain in the Error column.

| Accuracy | F1 | Precision | Recall | Positive Support | Negative Support | TP | FP | TN | FN |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.897923 | 0.888543 | 0.978771 | 0.813546 | 14396 | 14396 | 11711 | 254 | 14142 | 2684 |

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
| 0.646913 | 0.529308 | 0.588546 | 0.533948 | 11711 | 7576 | macro |

### Positive-Label 4x4 Confusion Matrix

| Actual \ Predicted | Indicator | Ideation | Behavior | Attempt |
|---|---|---|---|---|
| Indicator | 408 | 1403 | 219 | 63 |
| Ideation | 210 | 5735 | 609 | 254 |
| Behavior | 20 | 967 | 932 | 106 |
| Attempt | 3 | 77 | 204 | 501 |

### Per-Label Calculations

| Label | TP | FP | FN | Support | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|---:|---:|---:|
| Indicator | 408 | 233 | 1685 | 2093 | 0.636505 | 0.194935 | 0.298464 |
| Ideation | 5735 | 2447 | 1073 | 6808 | 0.700929 | 0.842391 | 0.765177 |
| Behavior | 932 | 1032 | 1093 | 2025 | 0.474542 | 0.460247 | 0.467285 |
| Attempt | 501 | 423 | 284 | 785 | 0.542208 | 0.638217 | 0.586308 |

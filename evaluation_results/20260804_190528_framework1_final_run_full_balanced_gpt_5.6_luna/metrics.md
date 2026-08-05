# Run Metrics

- Workflow: framework_1_parallel_no_evidence
- Service tier: standard
- Stop-after token threshold: None
- Current invocation tokens: 60387014
- Total samples: 28792
- Failed records: 0
- Failure rate: 0.000000
- LLM calls: 172761
- JSON/schema parsing errors: 4
- Cumulative case execution duration seconds: 136610.571418
- Real execution duration seconds: 15094.755306
- Total latency seconds: 301548.276120
- Average latency seconds per request: 1.745465
- Average input tokens per request: 319.973779
- Average output tokens per request: 29.566997
- Average cached tokens per request: 0.000000
- Total cost USD: 8.650027
- Average TTFT seconds when exposed: n/a
- Average provider completion TPS when exposed: n/a
- Average end-to-end completion TPS: 16.768568
- Suicidal-intent grouping accuracy: 0.894589
- Suicidal-intent grouping positive-class F1: 0.884076
- Suicidal-intent TP: 11573
- Suicidal-intent TN: 14184
- Suicidal-intent FP: 212
- Suicidal-intent FN: 2823

## Suicidal-Intent Positive-Class Grouping Metrics

- Positive: Indicator, Ideation, Behavior, Attempt
- Negative: Unsupportive
- TP/TN/FP/FN use this fixed positive orientation; failed predictions remain in the Error column.

| Accuracy | F1 | Precision | Recall | Positive Support | Negative Support | TP | FP | TN | FN |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.894589 | 0.884076 | 0.982011 | 0.803904 | 14396 | 14396 | 11573 | 212 | 14184 | 2823 |

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
| 0.636827 | 0.499247 | 0.571628 | 0.520756 | 11573 | 7370 | macro |

### Positive-Label 4x4 Confusion Matrix

| Actual \ Predicted | Indicator | Ideation | Behavior | Attempt |
|---|---|---|---|---|
| Indicator | 224 | 1432 | 250 | 83 |
| Ideation | 135 | 5722 | 642 | 273 |
| Behavior | 8 | 1015 | 883 | 122 |
| Attempt | 2 | 72 | 169 | 541 |

### Per-Label Calculations

| Label | TP | FP | FN | Support | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|---:|---:|---:|
| Indicator | 224 | 145 | 1765 | 1989 | 0.607046 | 0.112619 | 0.189992 |
| Ideation | 5722 | 2519 | 1050 | 6772 | 0.694333 | 0.844950 | 0.762273 |
| Behavior | 883 | 1061 | 1145 | 2028 | 0.454218 | 0.435404 | 0.444612 |
| Attempt | 541 | 478 | 243 | 784 | 0.530913 | 0.690051 | 0.600111 |

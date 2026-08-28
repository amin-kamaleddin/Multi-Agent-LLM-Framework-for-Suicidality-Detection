# Run Metrics

- Workflow: framework_3_hierarchical_no_reviewer
- Service tier: standard
- Stop-after token threshold: None
- Current invocation tokens: 8717580
- Total samples: 4317
- Failed records: 0
- Failure rate: 0.000000
- LLM calls: 10702
- JSON/schema parsing errors: 2
- Cumulative case execution duration seconds: 20178.383414
- Real execution duration seconds: 5057.221787
- Total latency seconds: 20165.168051
- Average latency seconds per request: 1.884243
- Average input tokens per request: 734.851803
- Average output tokens per request: 79.723042
- Average cached tokens per request: 0.000000
- Total cost USD: 2.624225
- Average TTFT seconds when exposed: n/a
- Average provider completion TPS when exposed: n/a
- Average end-to-end completion TPS: 42.200450
- Suicidal-intent grouping accuracy: 0.893445
- Suicidal-intent grouping positive-class F1: 0.882533
- Suicidal-intent TP: 1728
- Suicidal-intent TN: 2129
- Suicidal-intent FP: 31
- Suicidal-intent FN: 429

## Suicidal-Intent Positive-Class Grouping Metrics

- Positive: Indicator, Ideation, Behavior, Attempt
- Negative: Unsupportive
- TP/TN/FP/FN use this fixed positive orientation; failed predictions remain in the Error column.

| Accuracy | F1 | Precision | Recall | Positive Support | Negative Support | TP | FP | TN | FN |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.893445 | 0.882533 | 0.982376 | 0.801113 | 2157 | 2160 | 1728 | 31 | 2129 | 429 |

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
| 0.660880 | 0.523582 | 0.617556 | 0.508453 | 1728 | 1142 | macro |

### Positive-Label 4x4 Confusion Matrix

| Actual \ Predicted | Indicator | Ideation | Behavior | Attempt |
|---|---|---|---|---|
| Indicator | 53 | 202 | 24 | 8 |
| Ideation | 23 | 894 | 82 | 21 |
| Behavior | 0 | 158 | 132 | 14 |
| Attempt | 1 | 19 | 34 | 63 |

### Per-Label Calculations

| Label | TP | FP | FN | Support | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|---:|---:|---:|
| Indicator | 53 | 24 | 234 | 287 | 0.688312 | 0.184669 | 0.291209 |
| Ideation | 894 | 379 | 126 | 1020 | 0.702278 | 0.876471 | 0.779765 |
| Behavior | 132 | 140 | 172 | 304 | 0.485294 | 0.434211 | 0.458333 |
| Attempt | 63 | 43 | 54 | 117 | 0.594340 | 0.538462 | 0.565022 |

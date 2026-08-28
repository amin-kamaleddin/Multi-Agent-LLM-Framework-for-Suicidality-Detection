# Run Metrics

- Workflow: framework_4_hierarchical_with_reviewer
- Service tier: standard
- Stop-after token threshold: None
- Current invocation tokens: 7504098
- Total samples: 4317
- Failed records: 0
- Failure rate: 0.000000
- LLM calls: 10870
- JSON/schema parsing errors: 1
- Cumulative case execution duration seconds: 21733.426793
- Real execution duration seconds: 5445.728885
- Total latency seconds: 21717.703257
- Average latency seconds per request: 1.997949
- Average input tokens per request: 595.720699
- Average output tokens per request: 94.628703
- Average cached tokens per request: 0.000000
- Total cost USD: 2.545682
- Average TTFT seconds when exposed: n/a
- Average provider completion TPS when exposed: n/a
- Average end-to-end completion TPS: 47.636983
- Suicidal-intent grouping accuracy: 0.897382
- Suicidal-intent grouping positive-class F1: 0.887649
- Suicidal-intent TP: 1750
- Suicidal-intent TN: 2124
- Suicidal-intent FP: 36
- Suicidal-intent FN: 407

## Suicidal-Intent Positive-Class Grouping Metrics

- Positive: Indicator, Ideation, Behavior, Attempt
- Negative: Unsupportive
- TP/TN/FP/FN use this fixed positive orientation; failed predictions remain in the Error column.

| Accuracy | F1 | Precision | Recall | Positive Support | Negative Support | TP | FP | TN | FN |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.897382 | 0.887649 | 0.979843 | 0.811312 | 2157 | 2160 | 1750 | 36 | 2124 | 407 |

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
| 0.657714 | 0.538283 | 0.619286 | 0.516946 | 1750 | 1151 | macro |

### Positive-Label 4x4 Confusion Matrix

| Actual \ Predicted | Indicator | Ideation | Behavior | Attempt |
|---|---|---|---|---|
| Indicator | 80 | 196 | 26 | 5 |
| Ideation | 27 | 881 | 92 | 21 |
| Behavior | 3 | 158 | 129 | 15 |
| Attempt | 1 | 19 | 36 | 61 |

### Per-Label Calculations

| Label | TP | FP | FN | Support | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|---:|---:|---:|
| Indicator | 80 | 31 | 227 | 307 | 0.720721 | 0.260586 | 0.382775 |
| Ideation | 881 | 373 | 140 | 1021 | 0.702552 | 0.862880 | 0.774505 |
| Behavior | 129 | 154 | 176 | 305 | 0.455830 | 0.422951 | 0.438776 |
| Attempt | 61 | 41 | 56 | 117 | 0.598039 | 0.521368 | 0.557078 |

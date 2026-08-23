# Run Metrics

- Workflow: framework_3_hierarchical_no_reviewer
- Service tier: standard
- Stop-after token threshold: None
- Current invocation tokens: 8804177
- Total samples: 4319
- Failed records: 0
- Failure rate: 0.000000
- LLM calls: 10804
- JSON/schema parsing errors: 3
- Cumulative case execution duration seconds: 17839.968986
- Real execution duration seconds: 2236.966769
- Total latency seconds: 17828.553973
- Average latency seconds per request: 1.650181
- Average input tokens per request: 736.078119
- Average output tokens per request: 78.821640
- Average cached tokens per request: 0.000000
- Total cost USD: 2.643377
- Average TTFT seconds when exposed: n/a
- Average provider completion TPS when exposed: n/a
- Average end-to-end completion TPS: 47.481899
- Suicidal-intent grouping accuracy: 0.897893
- Suicidal-intent grouping positive-class F1: 0.887986
- Suicidal-intent TP: 1748
- Suicidal-intent TN: 2130
- Suicidal-intent FP: 29
- Suicidal-intent FN: 412

## Suicidal-Intent Positive-Class Grouping Metrics

- Positive: Indicator, Ideation, Behavior, Attempt
- Negative: Unsupportive
- TP/TN/FP/FN use this fixed positive orientation; failed predictions remain in the Error column.

| Accuracy | F1 | Precision | Recall | Positive Support | Negative Support | TP | FP | TN | FN |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.897893 | 0.887986 | 0.983680 | 0.809259 | 2160 | 2159 | 1748 | 29 | 2130 | 412 |

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
| 0.639588 | 0.504981 | 0.578820 | 0.514711 | 1748 | 1118 | macro |

### Positive-Label 4x4 Confusion Matrix

| Actual \ Predicted | Indicator | Ideation | Behavior | Attempt |
|---|---|---|---|---|
| Indicator | 42 | 215 | 33 | 9 |
| Ideation | 20 | 859 | 112 | 36 |
| Behavior | 4 | 135 | 146 | 19 |
| Attempt | 0 | 14 | 33 | 71 |

### Per-Label Calculations

| Label | TP | FP | FN | Support | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|---:|---:|---:|
| Indicator | 42 | 24 | 257 | 299 | 0.636364 | 0.140468 | 0.230137 |
| Ideation | 859 | 364 | 168 | 1027 | 0.702371 | 0.836417 | 0.763556 |
| Behavior | 146 | 178 | 158 | 304 | 0.450617 | 0.480263 | 0.464968 |
| Attempt | 71 | 64 | 47 | 118 | 0.525926 | 0.601695 | 0.561265 |

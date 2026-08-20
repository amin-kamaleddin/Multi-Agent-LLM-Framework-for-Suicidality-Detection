# Run Metrics

- Workflow: framework_3_hierarchical_no_reviewer
- Service tier: standard
- Stop-after token threshold: None
- Current invocation tokens: 58660647
- Total samples: 28792
- Failed records: 0
- Failure rate: 0.000000
- LLM calls: 71800
- JSON/schema parsing errors: 14
- Cumulative case execution duration seconds: 118177.942039
- Real execution duration seconds: 14909.002945
- Total latency seconds: 118112.590851
- Average latency seconds per request: 1.645022
- Average input tokens per request: 737.219694
- Average output tokens per request: 79.780961
- Average cached tokens per request: 0.000000
- Total cost USD: 17.663059
- Average TTFT seconds when exposed: n/a
- Average provider completion TPS when exposed: n/a
- Average end-to-end completion TPS: 49.883443
- Suicidal-intent grouping accuracy: 0.895249
- Suicidal-intent grouping positive-class F1: 0.884709
- Suicidal-intent TP: 11572
- Suicidal-intent TN: 14204
- Suicidal-intent FP: 192
- Suicidal-intent FN: 2824

## Suicidal-Intent Positive-Class Grouping Metrics

- Positive: Indicator, Ideation, Behavior, Attempt
- Negative: Unsupportive
- TP/TN/FP/FN use this fixed positive orientation; failed predictions remain in the Error column.

| Accuracy | F1 | Precision | Recall | Positive Support | Negative Support | TP | FP | TN | FN |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.895249 | 0.884709 | 0.983679 | 0.803834 | 14396 | 14396 | 11572 | 192 | 14204 | 2824 |

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
| 0.641721 | 0.508747 | 0.573387 | 0.523382 | 11572 | 7426 | macro |

### Positive-Label 4x4 Confusion Matrix

| Actual \ Predicted | Indicator | Ideation | Behavior | Attempt |
|---|---|---|---|---|
| Indicator | 263 | 1439 | 216 | 69 |
| Ideation | 162 | 5722 | 624 | 270 |
| Behavior | 19 | 969 | 923 | 112 |
| Attempt | 2 | 69 | 195 | 518 |

### Per-Label Calculations

| Label | TP | FP | FN | Support | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|---:|---:|---:|
| Indicator | 263 | 183 | 1724 | 1987 | 0.589686 | 0.132360 | 0.216194 |
| Ideation | 5722 | 2477 | 1056 | 6778 | 0.697890 | 0.844202 | 0.764105 |
| Behavior | 923 | 1035 | 1100 | 2023 | 0.471399 | 0.456253 | 0.463703 |
| Attempt | 518 | 451 | 266 | 784 | 0.534572 | 0.660714 | 0.590987 |

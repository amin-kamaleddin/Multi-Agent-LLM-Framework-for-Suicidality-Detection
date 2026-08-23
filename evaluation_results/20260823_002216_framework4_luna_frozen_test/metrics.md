# Run Metrics

- Workflow: framework_4_hierarchical_with_reviewer
- Service tier: standard
- Stop-after token threshold: None
- Current invocation tokens: 7573830
- Total samples: 4319
- Failed records: 0
- Failure rate: 0.000000
- LLM calls: 10953
- JSON/schema parsing errors: 8
- Cumulative case execution duration seconds: 19187.345767
- Real execution duration seconds: 2404.616227
- Total latency seconds: 19172.268685
- Average latency seconds per request: 1.750413
- Average input tokens per request: 598.265498
- Average output tokens per request: 93.219027
- Average cached tokens per request: 0.000000
- Total cost USD: 2.551906
- Average TTFT seconds when exposed: n/a
- Average provider completion TPS when exposed: n/a
- Average end-to-end completion TPS: 53.001609
- Suicidal-intent grouping accuracy: 0.902061
- Suicidal-intent grouping positive-class F1: 0.893263
- Suicidal-intent TP: 1770
- Suicidal-intent TN: 2126
- Suicidal-intent FP: 33
- Suicidal-intent FN: 390

## Suicidal-Intent Positive-Class Grouping Metrics

- Positive: Indicator, Ideation, Behavior, Attempt
- Negative: Unsupportive
- TP/TN/FP/FN use this fixed positive orientation; failed predictions remain in the Error column.

| Accuracy | F1 | Precision | Recall | Positive Support | Negative Support | TP | FP | TN | FN |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.902061 | 0.893263 | 0.981697 | 0.819444 | 2160 | 2159 | 1770 | 33 | 2126 | 390 |

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
| 0.642373 | 0.523171 | 0.589838 | 0.523158 | 1770 | 1137 | macro |

### Positive-Label 4x4 Confusion Matrix

| Actual \ Predicted | Indicator | Ideation | Behavior | Attempt |
|---|---|---|---|---|
| Indicator | 70 | 207 | 33 | 9 |
| Ideation | 26 | 858 | 108 | 36 |
| Behavior | 4 | 140 | 141 | 20 |
| Attempt | 0 | 15 | 35 | 68 |

### Per-Label Calculations

| Label | TP | FP | FN | Support | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|---:|---:|---:|
| Indicator | 70 | 30 | 249 | 319 | 0.700000 | 0.219436 | 0.334129 |
| Ideation | 858 | 362 | 170 | 1028 | 0.703279 | 0.834630 | 0.763345 |
| Behavior | 141 | 176 | 164 | 305 | 0.444795 | 0.462295 | 0.453376 |
| Attempt | 68 | 65 | 50 | 118 | 0.511278 | 0.576271 | 0.541833 |

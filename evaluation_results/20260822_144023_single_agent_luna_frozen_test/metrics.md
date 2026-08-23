# Run Metrics

- Workflow: single_agent
- Service tier: standard
- Stop-after token threshold: None
- Current invocation tokens: 2288377
- Total samples: 4319
- Failed records: 0
- Failure rate: 0.000000
- LLM calls: 4319
- JSON/schema parsing errors: 0
- Cumulative case execution duration seconds: 5724.938436
- Real execution duration seconds: 719.370310
- Total latency seconds: 5724.756352
- Average latency seconds per request: 1.325482
- Average input tokens per request: 501.206066
- Average output tokens per request: 28.633480
- Average cached tokens per request: 0.000000
- Total cost USD: 0.582981
- Average TTFT seconds when exposed: n/a
- Average provider completion TPS when exposed: n/a
- Average end-to-end completion TPS: 17.550804
- Suicidal-intent grouping accuracy: 0.895346
- Suicidal-intent grouping positive-class F1: 0.884694
- Suicidal-intent TP: 1734
- Suicidal-intent TN: 2133
- Suicidal-intent FP: 26
- Suicidal-intent FN: 426

## Suicidal-Intent Positive-Class Grouping Metrics

- Positive: Indicator, Ideation, Behavior, Attempt
- Negative: Unsupportive
- TP/TN/FP/FN use this fixed positive orientation; failed predictions remain in the Error column.

| Accuracy | F1 | Precision | Recall | Positive Support | Negative Support | TP | FP | TN | FN |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.895346 | 0.884694 | 0.985227 | 0.802778 | 2160 | 2159 | 1734 | 26 | 2133 | 426 |

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
| 0.627451 | 0.469740 | 0.538878 | 0.490860 | 1734 | 1088 | macro |

### Positive-Label 4x4 Confusion Matrix

| Actual \ Predicted | Indicator | Ideation | Behavior | Attempt |
|---|---|---|---|---|
| Indicator | 34 | 208 | 31 | 12 |
| Ideation | 21 | 871 | 92 | 43 |
| Behavior | 3 | 159 | 107 | 35 |
| Attempt | 0 | 15 | 27 | 76 |

### Per-Label Calculations

| Label | TP | FP | FN | Support | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|---:|---:|---:|
| Indicator | 34 | 24 | 251 | 285 | 0.586207 | 0.119298 | 0.198251 |
| Ideation | 871 | 382 | 156 | 1027 | 0.695132 | 0.848101 | 0.764035 |
| Behavior | 107 | 150 | 197 | 304 | 0.416342 | 0.351974 | 0.381462 |
| Attempt | 76 | 90 | 42 | 118 | 0.457831 | 0.644068 | 0.535211 |

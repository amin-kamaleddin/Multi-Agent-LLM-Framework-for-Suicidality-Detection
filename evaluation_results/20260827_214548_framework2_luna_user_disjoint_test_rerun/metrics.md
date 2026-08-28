# Run Metrics

- Workflow: framework_2_parallel_with_evidence
- Service tier: standard
- Stop-after token threshold: None
- Current invocation tokens: 17659314
- Total samples: 4317
- Failed records: 0
- Failure rate: 0.000000
- LLM calls: 25906
- JSON/schema parsing errors: 4
- Cumulative case execution duration seconds: 18728.292586
- Real execution duration seconds: 4706.079690
- Total latency seconds: 46191.717323
- Average latency seconds per request: 1.783051
- Average input tokens per request: 610.279626
- Average output tokens per request: 71.389253
- Average cached tokens per request: 0.000000
- Total cost USD: 5.607305
- Average TTFT seconds when exposed: n/a
- Average provider completion TPS when exposed: n/a
- Average end-to-end completion TPS: 31.759404
- Suicidal-intent grouping accuracy: 0.885105
- Suicidal-intent grouping positive-class F1: 0.871569
- Suicidal-intent TP: 1683
- Suicidal-intent TN: 2138
- Suicidal-intent FP: 22
- Suicidal-intent FN: 474

## Suicidal-Intent Positive-Class Grouping Metrics

- Positive: Indicator, Ideation, Behavior, Attempt
- Negative: Unsupportive
- TP/TN/FP/FN use this fixed positive orientation; failed predictions remain in the Error column.

| Accuracy | F1 | Precision | Recall | Positive Support | Negative Support | TP | FP | TN | FN |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.885105 | 0.871569 | 0.987097 | 0.780250 | 2157 | 2160 | 1683 | 22 | 2138 | 474 |

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
| 0.657160 | 0.548843 | 0.617732 | 0.559521 | 1683 | 1106 | macro |

### Positive-Label 4x4 Confusion Matrix

| Actual \ Predicted | Indicator | Ideation | Behavior | Attempt |
|---|---|---|---|---|
| Indicator | 55 | 154 | 45 | 5 |
| Ideation | 24 | 769 | 186 | 26 |
| Behavior | 0 | 67 | 220 | 16 |
| Attempt | 1 | 12 | 41 | 62 |

### Per-Label Calculations

| Label | TP | FP | FN | Support | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|---:|---:|---:|
| Indicator | 55 | 25 | 204 | 259 | 0.687500 | 0.212355 | 0.324484 |
| Ideation | 769 | 233 | 236 | 1005 | 0.767465 | 0.765174 | 0.766318 |
| Behavior | 220 | 272 | 83 | 303 | 0.447154 | 0.726073 | 0.553459 |
| Attempt | 62 | 47 | 54 | 116 | 0.568807 | 0.534483 | 0.551111 |

# Run Metrics

- Workflow: framework_1_parallel_no_evidence
- Service tier: standard
- Stop-after token threshold: None
- Current invocation tokens: 8968081
- Total samples: 4317
- Failed records: 0
- Failure rate: 0.000000
- LLM calls: 25905
- JSON/schema parsing errors: 0
- Cumulative case execution duration seconds: 17384.976859
- Real execution duration seconds: 2184.640545
- Total latency seconds: 42245.952992
- Average latency seconds per request: 1.630803
- Average input tokens per request: 316.943679
- Average output tokens per request: 29.247443
- Average cached tokens per request: 0.000000
- Total cost USD: 2.560884
- Average TTFT seconds when exposed: n/a
- Average provider completion TPS when exposed: n/a
- Average end-to-end completion TPS: 16.553925
- Suicidal-intent grouping accuracy: 0.889275
- Suicidal-intent grouping positive-class F1: 0.877999
- Suicidal-intent TP: 1720
- Suicidal-intent TN: 2119
- Suicidal-intent FP: 41
- Suicidal-intent FN: 437

## Suicidal-Intent Positive-Class Grouping Metrics

- Positive: Indicator, Ideation, Behavior, Attempt
- Negative: Unsupportive
- TP/TN/FP/FN use this fixed positive orientation; failed predictions remain in the Error column.

| Accuracy | F1 | Precision | Recall | Positive Support | Negative Support | TP | FP | TN | FN |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.889275 | 0.877999 | 0.976718 | 0.797404 | 2157 | 2160 | 1720 | 41 | 2119 | 437 |

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
| 0.654651 | 0.504931 | 0.610827 | 0.498073 | 1720 | 1126 | macro |

### Positive-Label 4x4 Confusion Matrix

| Actual \ Predicted | Indicator | Ideation | Behavior | Attempt |
|---|---|---|---|---|
| Indicator | 44 | 201 | 28 | 9 |
| Ideation | 14 | 893 | 86 | 24 |
| Behavior | 1 | 159 | 125 | 19 |
| Attempt | 1 | 16 | 36 | 64 |

### Per-Label Calculations

| Label | TP | FP | FN | Support | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|---:|---:|---:|
| Indicator | 44 | 16 | 238 | 282 | 0.733333 | 0.156028 | 0.257310 |
| Ideation | 893 | 376 | 124 | 1017 | 0.703704 | 0.878073 | 0.781277 |
| Behavior | 125 | 150 | 179 | 304 | 0.454545 | 0.411184 | 0.431779 |
| Attempt | 64 | 52 | 53 | 117 | 0.551724 | 0.547009 | 0.549356 |

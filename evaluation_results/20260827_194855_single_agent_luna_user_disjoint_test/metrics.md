# Run Metrics

- Workflow: single_agent
- Service tier: standard
- Stop-after token threshold: None
- Current invocation tokens: 2275080
- Total samples: 4317
- Failed records: 0
- Failure rate: 0.000000
- LLM calls: 4317
- JSON/schema parsing errors: 0
- Cumulative case execution duration seconds: 6383.321366
- Real execution duration seconds: 801.738207
- Total latency seconds: 6383.132482
- Average latency seconds per request: 1.478604
- Average input tokens per request: 498.444290
- Average output tokens per request: 28.560574
- Average cached tokens per request: 0.000000
- Total cost USD: 0.580690
- Average TTFT seconds when exposed: n/a
- Average provider completion TPS when exposed: n/a
- Average end-to-end completion TPS: 15.997821
- Suicidal-intent grouping accuracy: 0.887885
- Suicidal-intent grouping positive-class F1: 0.875578
- Suicidal-intent TP: 1703
- Suicidal-intent TN: 2130
- Suicidal-intent FP: 30
- Suicidal-intent FN: 454

## Suicidal-Intent Positive-Class Grouping Metrics

- Positive: Indicator, Ideation, Behavior, Attempt
- Negative: Unsupportive
- TP/TN/FP/FN use this fixed positive orientation; failed predictions remain in the Error column.

| Accuracy | F1 | Precision | Recall | Positive Support | Negative Support | TP | FP | TN | FN |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.887885 | 0.875578 | 0.982689 | 0.789522 | 2157 | 2160 | 1703 | 30 | 2130 | 454 |

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
| 0.647093 | 0.481662 | 0.583065 | 0.478054 | 1703 | 1102 | macro |

### Positive-Label 4x4 Confusion Matrix

| Actual \ Predicted | Indicator | Ideation | Behavior | Attempt |
|---|---|---|---|---|
| Indicator | 46 | 194 | 23 | 8 |
| Ideation | 19 | 908 | 59 | 25 |
| Behavior | 0 | 194 | 80 | 30 |
| Attempt | 1 | 23 | 25 | 68 |

### Per-Label Calculations

| Label | TP | FP | FN | Support | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|---:|---:|---:|
| Indicator | 46 | 20 | 225 | 271 | 0.696970 | 0.169742 | 0.272997 |
| Ideation | 908 | 411 | 103 | 1011 | 0.688400 | 0.898121 | 0.779399 |
| Behavior | 80 | 107 | 224 | 304 | 0.427807 | 0.263158 | 0.325866 |
| Attempt | 68 | 63 | 49 | 117 | 0.519084 | 0.581197 | 0.548387 |

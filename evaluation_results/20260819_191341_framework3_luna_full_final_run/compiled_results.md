# Compiled Evaluation Results

- Run directory: `logs/runs/20260819_191341_framework3_luna_full_final_run`
- Workflow: `framework_3_hierarchical_no_reviewer`
- Model: `openai/gpt-5.6-luna`
- Requested model: `gpt-5.6-luna`
- Mode: `bool`
- Service tier: `standard`
- Stop-after token threshold: `None`
- Current invocation tokens: `58660647`
- Evaluation status: `completed`
- Stop reason: `None`

## Run Summary

| Metric | Value |
|---|---:|
| Processed records | 28792 |
| Total records | 28792 |
| Failed records | 0 |
| Failure rate | 0.000000 |
| LLM requests | 71800 |
| Failed LLM attempts | 14 |
| JSON/schema parsing errors | 14 |

## Execution Attempts

| Attempt | Start Date (UTC) | End Date (UTC) |
|---:|---|---|
| 1 | 2026-08-19T23:13:41.048755+00:00 | 2026-08-20T03:22:11.159470+00:00 |

## Final-Label Confusion Matrix

| Actual \ Predicted | Indicator | Ideation | Behavior | Attempt | Unsupportive | Error |
|---|---|---|---|---|---|---|
| Indicator | 263 | 1439 | 216 | 69 | 2563 | 0 |
| Ideation | 162 | 5722 | 624 | 270 | 244 | 0 |
| Behavior | 19 | 969 | 923 | 112 | 16 | 0 |
| Attempt | 2 | 69 | 195 | 518 | 1 | 0 |
| Unsupportive | 60 | 115 | 13 | 4 | 14204 | 0 |

## Positive-Label Subclassification Metrics

- Scope: records whose ground-truth and predicted labels are both Indicator, Ideation, Behavior, or Attempt.
- Precision, recall, and F1 are macro averages across the four positive labels.
- Correct: included records on the diagonal of the 4x4 matrix.
- Support: all records represented in the 4x4 matrix.

Formulas for each label `L`:
- `TP(L) = actual L and predicted L`.
- `FP(L) = actual another positive label and predicted L`.
- `FN(L) = actual L and predicted another positive label`.
- `Precision(L) = TP(L) / (TP(L) + FP(L))`.
- `Recall(L) = TP(L) / (TP(L) + FN(L))`.
- `F1(L) = 2 * Precision(L) * Recall(L) / (Precision(L) + Recall(L))`.
- `Overall accuracy = correct / support`.
- `Macro metric = sum of the four per-label metric values / 4`.

| Accuracy | F1 Score | Precision | Recall | Support | Correct | Averaging |
|---|---|---|---|---|---|---|
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
|---|---|---|---|---|---|---|---|
| Indicator | 263 | 183 | 1724 | 1987 | 0.589686 | 0.132360 | 0.216194 |
| Ideation | 5722 | 2477 | 1056 | 6778 | 0.697890 | 0.844202 | 0.764105 |
| Behavior | 923 | 1035 | 1100 | 2023 | 0.471399 | 0.456253 | 0.463703 |
| Attempt | 518 | 451 | 266 | 784 | 0.534572 | 0.660714 | 0.590987 |

## Suicidal-Intent Positive-Class Grouping Metrics

- Positive: Indicator, Ideation, Behavior, Attempt
- Negative: Unsupportive
- TP/TN/FP/FN below always treat the four suicidality labels as Positive.

| Outcome | Count |
|---|---:|
| TP | 11572 |
| TN | 14204 |
| FP | 192 |
| FN | 2824 |

| Accuracy | F1 Score | Precision | Recall | Positive Support | Negative Support | TP | FP | TN | FN |
|---|---|---|---|---|---|---|---|---|---|
| 0.895249 | 0.884709 | 0.983679 | 0.803834 | 14396 | 14396 | 11572 | 192 | 14204 | 2824 |

### Suicidal-Intent Positive-Class Grouping Confusion Matrix

| Actual \ Predicted | false | true | Error |
|---|---|---|---|
| false | 14204 | 192 | 0 |
| true | 2824 | 11572 | 0 |

## Label Distributions

### Ground Truth

| Label | Count |
|---|---:|
| Attempt | 785 |
| Behavior | 2039 |
| Ideation | 7022 |
| Indicator | 4550 |
| Unsupportive | 14396 |

### Predictions

| Predicted Label | Count |
|---|---:|
| Attempt | 973 |
| Behavior | 1971 |
| Ideation | 8314 |
| Indicator | 506 |
| Unsupportive | 17028 |

## Runtime And Tokens

| Metric | Value |
|---|---:|
| Cumulative case execution duration seconds | 118177.942039 |
| Real execution duration seconds | 14909.002945 |
| Total latency seconds | 118112.590851 |
| Average latency seconds per request | 1.645022 |
| Min latency seconds | 0.751899 |
| Max latency seconds | 54.977277 |
| Total input tokens | 52932374 |
| Total output tokens | 5728273 |
| Total tokens | 58660647 |
| Total cached input tokens | 0 |
| Total cache-write tokens | 0 |
| Average input tokens per request | 737.219694 |
| Average output tokens per request | 79.780961 |
| Average total tokens per request | 817.000655 |
| Total cost USD | 17.663059 |
| Average cost USD per request | 0.000246 |
| Observed TTFT calls | 0 |
| Average TTFT seconds | n/a |
| Average provider generation seconds | n/a |
| Average provider latency seconds | n/a |
| Average moderation latency seconds | n/a |
| Average provider completion TPS | n/a |
| Average end-to-end completion TPS | 49.883443 |

## LLM Errors And Retries

| Error Type | Count |
|---|---:|
| JSONDecodeError | 12 |
| ValidationError | 2 |

### Recovered Error Attempts

| Sample ID | Role | Attempt | Error Type |
|---|---|---:|---|
| 2779 | behavior_attempt_agent | 1 | JSONDecodeError |
| 3941 | evaluator_agent | 1 | ValidationError |
| 5265 | ideation_action_agent | 1 | JSONDecodeError |
| 5388 | ideation_action_agent | 1 | JSONDecodeError |
| 6265 | evaluator_agent | 1 | ValidationError |
| 6294 | ideation_action_agent | 1 | JSONDecodeError |
| 8765 | perspective_agent | 1 | JSONDecodeError |
| 8914 | perspective_agent | 1 | JSONDecodeError |
| 9939 | perspective_agent | 1 | JSONDecodeError |
| 10017 | ideation_action_agent | 1 | JSONDecodeError |
| 11051 | ideation_action_agent | 1 | JSONDecodeError |
| 12868 | perspective_agent | 1 | JSONDecodeError |
| 13398 | perspective_agent | 1 | JSONDecodeError |
| 14051 | behavior_attempt_agent | 1 | JSONDecodeError |

## Trace Files

| Artifact | Purpose |
|---|---|
| `predictions.csv` | Per-record source fields, shared outputs, correctness, errors, and optional workflow-supplied agent outputs. |
| `suicidal_intent_metrics.csv` | One positive-oriented binary summary with TP, FP, TN, and FN. |
| `positive_subclassification_metrics.csv` | Aggregate accuracy and macro precision, recall, and F1 across the four positive labels. |
| `metrics.json` | Complete metrics payload. |
| `results.json` | Complete per-record outputs and metrics payload. |
| `misclassified_ids.txt` | IDs with an incorrect final-label evaluation. |
| `suicidal_intent_misclassified_ids.txt` | IDs with an incorrect binary evaluation. |
| `suicidal_intent_metrics.svg` | Accuracy, precision, recall, and F1 for the suicidal-intent grouping. |
| `runtime_metrics.svg` | Cost, token, latency, call, failure, and parse-error visualization. |
| `llm_calls.jsonl` | Every LLM request attempt and retry. |
| `traces.jsonl` | Workflow-supplied execution events. |
| `trace_summary.json` | Runtime, token, and failure summary. |

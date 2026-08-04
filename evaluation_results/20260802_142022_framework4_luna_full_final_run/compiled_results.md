# Compiled Evaluation Results

- Run directory: `logs/runs/20260802_142022_framework4_luna_full_final_run`
- Workflow: `framework_4_hierarchical_with_reviewer`
- Model: `openai/gpt-5.6-luna`
- Requested model: `gpt-5.6-luna`
- Mode: `bool`
- Service tier: `standard`
- Stop-after token threshold: `None`
- Current invocation tokens: `50724599`
- Evaluation status: `completed`
- Stop reason: `None`

## Run Summary

| Metric | Value |
|---|---:|
| Processed records | 28792 |
| Total records | 28792 |
| Failed records | 1 |
| Failure rate | 0.000035 |
| LLM requests | 72977 |
| Failed LLM attempts | 24 |
| JSON/schema parsing errors | 23 |

## Execution Attempts

| Attempt | Start Date (UTC) | End Date (UTC) |
|---:|---|---|
| 1 | 2026-08-02T18:20:22.096766+00:00 | 2026-08-02T23:06:14.672455+00:00 |

## Final-Label Confusion Matrix

| Actual \ Predicted | Indicator | Ideation | Behavior | Attempt | Unsupportive | Error |
|---|---|---|---|---|---|---|
| Indicator | 408 | 1403 | 219 | 63 | 2457 | 0 |
| Ideation | 210 | 5735 | 609 | 254 | 214 | 0 |
| Behavior | 20 | 967 | 932 | 106 | 13 | 1 |
| Attempt | 3 | 77 | 204 | 501 | 0 | 0 |
| Unsupportive | 123 | 114 | 13 | 4 | 14142 | 0 |

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
| 0.646913 | 0.529308 | 0.588546 | 0.533948 | 11711 | 7576 | macro |

### Positive-Label 4x4 Confusion Matrix

| Actual \ Predicted | Indicator | Ideation | Behavior | Attempt |
|---|---|---|---|---|
| Indicator | 408 | 1403 | 219 | 63 |
| Ideation | 210 | 5735 | 609 | 254 |
| Behavior | 20 | 967 | 932 | 106 |
| Attempt | 3 | 77 | 204 | 501 |

### Per-Label Calculations

| Label | TP | FP | FN | Support | Precision | Recall | F1 Score |
|---|---|---|---|---|---|---|---|
| Indicator | 408 | 233 | 1685 | 2093 | 0.636505 | 0.194935 | 0.298464 |
| Ideation | 5735 | 2447 | 1073 | 6808 | 0.700929 | 0.842391 | 0.765177 |
| Behavior | 932 | 1032 | 1093 | 2025 | 0.474542 | 0.460247 | 0.467285 |
| Attempt | 501 | 423 | 284 | 785 | 0.542208 | 0.638217 | 0.586308 |

## Suicidal-Intent Positive-Class Grouping Metrics

- Positive: Indicator, Ideation, Behavior, Attempt
- Negative: Unsupportive
- TP/TN/FP/FN below always treat the four suicidality labels as Positive.

| Outcome | Count |
|---|---:|
| TP | 11711 |
| TN | 14142 |
| FP | 254 |
| FN | 2684 |

| Accuracy | F1 Score | Precision | Recall | Positive Support | Negative Support | TP | FP | TN | FN |
|---|---|---|---|---|---|---|---|---|---|
| 0.897923 | 0.888543 | 0.978771 | 0.813546 | 14396 | 14396 | 11711 | 254 | 14142 | 2684 |

### Suicidal-Intent Positive-Class Grouping Confusion Matrix

| Actual \ Predicted | false | true | Error |
|---|---|---|---|
| false | 14142 | 254 | 0 |
| true | 2684 | 11711 | 1 |

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
| Attempt | 928 |
| Behavior | 1977 |
| Error | 1 |
| Ideation | 8296 |
| Indicator | 764 |
| Unsupportive | 16826 |

## Runtime And Tokens

| Metric | Value |
|---|---:|
| Cumulative case execution duration seconds | 136095.335867 |
| Real execution duration seconds | 17151.440877 |
| Total latency seconds | 136009.151157 |
| Average latency seconds per request | 1.863726 |
| Min latency seconds | 0.743851 |
| Max latency seconds | 48.145500 |
| Total input tokens | 43731376 |
| Total output tokens | 6993223 |
| Total tokens | 50724599 |
| Total cached input tokens | 0 |
| Total cache-write tokens | 0 |
| Average input tokens per request | 599.248750 |
| Average output tokens per request | 95.827768 |
| Average total tokens per request | 695.076517 |
| Total cost USD | 8.635986 |
| Average cost USD per request | 0.000118 |
| Observed TTFT calls | 0 |
| Average TTFT seconds | n/a |
| Average provider generation seconds | n/a |
| Average provider latency seconds | n/a |
| Average moderation latency seconds | n/a |
| Average provider completion TPS | n/a |
| Average end-to-end completion TPS | 52.528707 |

## LLM Errors And Retries

| Error Type | Count |
|---|---:|
| JSONDecodeError | 23 |
| ValueError | 1 |

### Recovered Error Attempts

| Sample ID | Role | Attempt | Error Type |
|---|---|---:|---|
| 82 | perspective_agent | 1 | JSONDecodeError |
| 92 | perspective_agent | 1 | ValueError |
| 1900 | perspective_agent | 1 | JSONDecodeError |
| 2779 | behavior_attempt_agent | 1 | JSONDecodeError |
| 3594 | ideation_action_agent | 1 | JSONDecodeError |
| 3757 | perspective_agent | 1 | JSONDecodeError |
| 4614 | perspective_agent | 1 | JSONDecodeError |
| 5439 | secondary_review_agent | 1 | JSONDecodeError |
| 5475 | secondary_review_agent | 1 | JSONDecodeError |
| 5687 | ideation_action_agent | 1 | JSONDecodeError |
| 6261 | ideation_action_agent | 1 | JSONDecodeError |
| 8162 | ideation_action_agent | 1 | JSONDecodeError |
| 8163 | ideation_action_agent | 1 | JSONDecodeError |
| 8165 | ideation_action_agent | 1 | JSONDecodeError |
| 8165 | ideation_action_agent | 2 | JSONDecodeError |
| 8837 | ideation_action_agent | 1 | JSONDecodeError |
| 9365 | judge_agent | 1 | JSONDecodeError |
| 10884 | ideation_action_agent | 1 | JSONDecodeError |
| 12714 | perspective_agent | 1 | JSONDecodeError |
| 12714 | perspective_agent | 2 | JSONDecodeError |
| 12714 | perspective_agent | 3 | JSONDecodeError |
| 12714 | perspective_agent | 4 | JSONDecodeError |
| 13127 | behavior_attempt_agent | 1 | JSONDecodeError |
| 13421 | ideation_action_agent | 1 | JSONDecodeError |

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

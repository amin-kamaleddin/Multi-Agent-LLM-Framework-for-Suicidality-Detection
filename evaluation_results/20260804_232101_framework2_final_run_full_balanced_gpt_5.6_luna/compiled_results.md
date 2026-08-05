# Compiled Evaluation Results

- Run directory: `logs\runs\20260804_232101_framework2_final_run_full_balanced_gpt_5.6_luna`
- Workflow: `framework_2_parallel_with_evidence`
- Model: `openai/gpt-5.6-luna`
- Requested model: `gpt-5.6-luna`
- Mode: `evidence`
- Service tier: `standard`
- Stop-after token threshold: `None`
- Current invocation tokens: `118473800`
- Evaluation status: `completed`
- Stop reason: `None`

## Run Summary

| Metric | Value |
|---|---:|
| Processed records | 28792 |
| Total records | 28792 |
| Failed records | 0 |
| Failure rate | 0.000000 |
| LLM requests | 172808 |
| Failed LLM attempts | 56 |
| JSON/schema parsing errors | 56 |

## Execution Attempts

| Attempt | Start Date (UTC) | End Date (UTC) |
|---:|---|---|
| 1 | 2026-08-05T03:21:01.872254+00:00 | 2026-08-05T06:02:02.761052+00:00 |

## Final-Label Confusion Matrix

| Actual \ Predicted | Indicator | Ideation | Behavior | Attempt | Unsupportive | Error |
|---|---|---|---|---|---|---|
| Indicator | 278 | 1120 | 352 | 64 | 2736 | 0 |
| Ideation | 201 | 4989 | 1260 | 264 | 308 | 0 |
| Behavior | 15 | 406 | 1483 | 119 | 16 | 0 |
| Attempt | 3 | 45 | 232 | 500 | 5 | 0 |
| Unsupportive | 44 | 93 | 44 | 4 | 14211 | 0 |

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
| 0.639838 | 0.531429 | 0.573401 | 0.567605 | 11331 | 7250 | macro |

### Positive-Label 4x4 Confusion Matrix

| Actual \ Predicted | Indicator | Ideation | Behavior | Attempt |
|---|---|---|---|---|
| Indicator | 278 | 1120 | 352 | 64 |
| Ideation | 201 | 4989 | 1260 | 264 |
| Behavior | 15 | 406 | 1483 | 119 |
| Attempt | 3 | 45 | 232 | 500 |

### Per-Label Calculations

| Label | TP | FP | FN | Support | Precision | Recall | F1 Score |
|---|---|---|---|---|---|---|---|
| Indicator | 278 | 219 | 1536 | 1814 | 0.559356 | 0.153252 | 0.240588 |
| Ideation | 4989 | 1571 | 1725 | 6714 | 0.760518 | 0.743074 | 0.751695 |
| Behavior | 1483 | 1844 | 540 | 2023 | 0.445747 | 0.733070 | 0.554393 |
| Attempt | 500 | 447 | 280 | 780 | 0.527983 | 0.641026 | 0.579039 |

## Suicidal-Intent Positive-Class Grouping Metrics

- Positive: Indicator, Ideation, Behavior, Attempt
- Negative: Unsupportive
- TP/TN/FP/FN below always treat the four suicidality labels as Positive.

| Outcome | Count |
|---|---:|
| TP | 11331 |
| TN | 14211 |
| FP | 185 |
| FN | 3065 |

| Accuracy | F1 Score | Precision | Recall | Positive Support | Negative Support | TP | FP | TN | FN |
|---|---|---|---|---|---|---|---|---|---|
| 0.887121 | 0.874575 | 0.983935 | 0.787094 | 14396 | 14396 | 11331 | 185 | 14211 | 3065 |

### Suicidal-Intent Positive-Class Grouping Confusion Matrix

| Actual \ Predicted | false | true | Error |
|---|---|---|---|
| false | 14211 | 185 | 0 |
| true | 3065 | 11331 | 0 |

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
| Attempt | 951 |
| Behavior | 3371 |
| Ideation | 6653 |
| Indicator | 541 |
| Unsupportive | 17276 |

## Runtime And Tokens

| Metric | Value |
|---|---:|
| Cumulative case execution duration seconds | 142303.897463 |
| Real execution duration seconds | 9614.207126 |
| Total latency seconds | 349115.269818 |
| Average latency seconds per request | 2.020249 |
| Min latency seconds | 0.584618 |
| Max latency seconds | 69.067501 |
| Total input tokens | 106018303 |
| Total output tokens | 12455497 |
| Total tokens | 118473800 |
| Total cached input tokens | 2385 |
| Total cache-write tokens | 30657062 |
| Average input tokens per request | 613.503443 |
| Average output tokens per request | 72.077086 |
| Average total tokens per request | 685.580529 |
| Total cost USD | 18.871221 |
| Average cost USD per request | 0.000109 |
| Observed TTFT calls | 0 |
| Average TTFT seconds | n/a |
| Average provider generation seconds | n/a |
| Average provider latency seconds | n/a |
| Average moderation latency seconds | n/a |
| Average provider completion TPS | n/a |
| Average end-to-end completion TPS | 30.762110 |

## LLM Errors And Retries

| Error Type | Count |
|---|---:|
| JSONDecodeError | 37 |
| ValidationError | 19 |

### Recovered Error Attempts

| Sample ID | Role | Attempt | Error Type |
|---|---|---:|---|
| 686 | Attempt_collector | 1 | JSONDecodeError |
| 744 | Ideation_collector | 1 | JSONDecodeError |
| 966 | Unsupportive_collector | 1 | JSONDecodeError |
| 1053 | Indicator_collector | 1 | JSONDecodeError |
| 1421 | Unsupportive_collector | 1 | ValidationError |
| 1713 | Indicator_collector | 1 | JSONDecodeError |
| 1931 | Ideation_collector | 1 | JSONDecodeError |
| 2338 | Indicator_collector | 1 | JSONDecodeError |
| 2506 | Indicator_collector | 1 | JSONDecodeError |
| 3232 | Unsupportive_collector | 1 | JSONDecodeError |
| 3271 | Indicator_collector | 1 | JSONDecodeError |
| 3309 | Indicator_collector | 1 | JSONDecodeError |
| 3437 | Indicator_collector | 1 | JSONDecodeError |
| 3553 | Ideation_collector | 1 | ValidationError |
| 3906 | Behavior_collector | 1 | JSONDecodeError |
| 3948 | Ideation_collector | 1 | JSONDecodeError |
| 4390 | Indicator_collector | 1 | JSONDecodeError |
| 4445 | Ideation_collector | 1 | ValidationError |
| 4499 | Indicator_collector | 1 | JSONDecodeError |
| 4736 | Ideation_collector | 1 | ValidationError |
| 5842 | Ideation_collector | 1 | ValidationError |
| 5905 | Attempt_collector | 1 | JSONDecodeError |
| 6829 | Ideation_collector | 1 | ValidationError |
| 7064 | Ideation_collector | 1 | ValidationError |
| 7296 | Unsupportive_collector | 1 | JSONDecodeError |
| 7614 | Ideation_collector | 1 | ValidationError |
| 7614 | Ideation_collector | 2 | ValidationError |
| 8281 | Behavior_collector | 1 | JSONDecodeError |
| 8396 | Ideation_collector | 1 | ValidationError |
| 8905 | Indicator_collector | 1 | JSONDecodeError |
| 9215 | Indicator_collector | 1 | JSONDecodeError |
| 10188 | Indicator_collector | 1 | JSONDecodeError |
| 10546 | Ideation_collector | 1 | JSONDecodeError |
| 10601 | Indicator_collector | 1 | JSONDecodeError |
| 10643 | Ideation_collector | 1 | JSONDecodeError |
| 10733 | Indicator_collector | 1 | JSONDecodeError |
| 11446 | Attempt_collector | 1 | JSONDecodeError |
| 11469 | Indicator_collector | 1 | JSONDecodeError |
| 11578 | Ideation_collector | 1 | JSONDecodeError |
| 12132 | Indicator_collector | 1 | JSONDecodeError |
| 12184 | Ideation_collector | 1 | ValidationError |
| 12711 | Unsupportive_collector | 1 | JSONDecodeError |
| 13032 | Behavior_collector | 1 | JSONDecodeError |
| 13235 | Indicator_collector | 1 | JSONDecodeError |
| 13608 | Attempt_collector | 1 | JSONDecodeError |
| 13719 | Indicator_collector | 1 | JSONDecodeError |
| 14152 | Ideation_collector | 1 | ValidationError |
| 23433 | Unsupportive_collector | 1 | ValidationError |
| 23433 | Unsupportive_collector | 2 | ValidationError |
| 28427 | Unsupportive_collector | 1 | JSONDecodeError |
| 29226 | Unsupportive_collector | 1 | ValidationError |
| 29988 | Unsupportive_collector | 1 | ValidationError |
| 33368 | Unsupportive_collector | 1 | ValidationError |
| 33368 | Unsupportive_collector | 2 | ValidationError |
| 36238 | Unsupportive_collector | 1 | ValidationError |
| 36660 | Indicator_collector | 1 | JSONDecodeError |

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

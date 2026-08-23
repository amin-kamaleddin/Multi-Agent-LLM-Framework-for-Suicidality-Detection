# Compiled Evaluation Results

- Run directory: `logs/runs/20260823_002216_framework4_luna_frozen_test`
- Workflow: `framework_4_hierarchical_with_reviewer`
- Model: `openai/gpt-5.6-luna`
- Requested model: `gpt-5.6-luna`
- Mode: `bool`
- Service tier: `standard`
- Stop-after token threshold: `None`
- Current invocation tokens: `7573830`
- Evaluation status: `completed`
- Stop reason: `None`

## Run Summary

| Metric | Value |
|---|---:|
| Processed records | 4319 |
| Total records | 4319 |
| Failed records | 0 |
| Failure rate | 0.000000 |
| LLM requests | 10953 |
| Failed LLM attempts | 8 |
| JSON/schema parsing errors | 8 |

## Execution Attempts

| Attempt | Start Date (UTC) | End Date (UTC) |
|---:|---|---|
| 1 | 2026-08-23T04:22:16.304074+00:00 | 2026-08-23T05:02:21.231396+00:00 |

## Final-Label Confusion Matrix

| Actual \ Predicted | Indicator | Ideation | Behavior | Attempt | Unsupportive | Error |
|---|---|---|---|---|---|---|
| Indicator | 70 | 207 | 33 | 9 | 364 | 0 |
| Ideation | 26 | 858 | 108 | 36 | 25 | 0 |
| Behavior | 4 | 140 | 141 | 20 | 1 | 0 |
| Attempt | 0 | 15 | 35 | 68 | 0 | 0 |
| Unsupportive | 18 | 11 | 3 | 1 | 2126 | 0 |

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
|---|---|---|---|---|---|---|---|
| Indicator | 70 | 30 | 249 | 319 | 0.700000 | 0.219436 | 0.334129 |
| Ideation | 858 | 362 | 170 | 1028 | 0.703279 | 0.834630 | 0.763345 |
| Behavior | 141 | 176 | 164 | 305 | 0.444795 | 0.462295 | 0.453376 |
| Attempt | 68 | 65 | 50 | 118 | 0.511278 | 0.576271 | 0.541833 |

## Suicidal-Intent Positive-Class Grouping Metrics

- Positive: Indicator, Ideation, Behavior, Attempt
- Negative: Unsupportive
- TP/TN/FP/FN below always treat the four suicidality labels as Positive.

| Outcome | Count |
|---|---:|
| TP | 1770 |
| TN | 2126 |
| FP | 33 |
| FN | 390 |

| Accuracy | F1 Score | Precision | Recall | Positive Support | Negative Support | TP | FP | TN | FN |
|---|---|---|---|---|---|---|---|---|---|
| 0.902061 | 0.893263 | 0.981697 | 0.819444 | 2160 | 2159 | 1770 | 33 | 2126 | 390 |

### Suicidal-Intent Positive-Class Grouping Confusion Matrix

| Actual \ Predicted | false | true | Error |
|---|---|---|---|
| false | 2126 | 33 | 0 |
| true | 390 | 1770 | 0 |

## Prespecified Statistical Analysis

- Primary endpoint (prespecified): **Suicidal-intent sensitivity** — `TP / (TP + FN), where any of Indicator, Ideation, Behavior, or Attempt is positive.`
- Rationale: This safety-oriented endpoint measures missed suicidality signs and is invariant to deployment prevalence.
- CI method: two-sided 95% percentile user-cluster bootstrap (2000 resamples; 2721 users; seed 20260820).
- The primary endpoint is prevalence-invariant. PPV and NPV are secondary and depend on prevalence.

### Observed-Sample Estimates And 95% CIs

| Metric | Estimate | 95% CI |
|---|---:|---|
| Sensitivity (primary) | 0.819444 | 0.796138 to 0.842591 |
| Specificity | 0.984715 | 0.979535 to 0.989909 |
| PPV at observed sample prevalence | 0.981697 | 0.975100 to 0.987848 |
| NPV at observed sample prevalence | 0.844992 | 0.820697 to 0.868597 |
| Positive-phase macro F1 | 0.523171 | 0.491022 to 0.555113 |

### Prevalence-Adjusted Predictive Values

These are transport estimates assuming sensitivity and specificity transfer to a target population. They are not estimates of the balanced evaluation sample's prevalence.

| Target Positive Prevalence | PPV (95% CI) | NPV (95% CI) |
|---:|---|---|
| 1.00% | 0.351294 (0.286607 to 0.448405) | 0.998151 (0.997913 to 0.998385) |
| 5.00% | 0.738334 (0.676725 to 0.809006) | 0.990442 (0.989222 to 0.991644) |
| 10.00% | 0.856257 (0.815473 to 0.899418) | 0.980034 (0.977516 to 0.982521) |
| 20.00% | 0.930569 (0.908620 to 0.952651) | 0.956170 (0.950794 to 0.961513) |
| 50.00% | 0.981689 (0.975474 to 0.987727) | 0.845053 (0.828494 to 0.861987) |

## Label Distributions

### Ground Truth

| Label | Count |
|---|---:|
| Attempt | 118 |
| Behavior | 306 |
| Ideation | 1053 |
| Indicator | 683 |
| Unsupportive | 2159 |

### Predictions

| Predicted Label | Count |
|---|---:|
| Attempt | 134 |
| Behavior | 320 |
| Ideation | 1231 |
| Indicator | 118 |
| Unsupportive | 2516 |

## Runtime And Tokens

| Metric | Value |
|---|---:|
| Cumulative case execution duration seconds | 19187.345767 |
| Real execution duration seconds | 2404.616227 |
| Total latency seconds | 19172.268685 |
| Average latency seconds per request | 1.750413 |
| Min latency seconds | 0.933689 |
| Max latency seconds | 23.321273 |
| Total input tokens | 6552802 |
| Total output tokens | 1021028 |
| Total tokens | 7573830 |
| Total cached input tokens | 0 |
| Total cache-write tokens | 0 |
| Average input tokens per request | 598.265498 |
| Average output tokens per request | 93.219027 |
| Average total tokens per request | 691.484525 |
| Total cost USD | 2.551906 |
| Average cost USD per request | 0.000233 |
| Observed TTFT calls | 0 |
| Average TTFT seconds | n/a |
| Average provider generation seconds | n/a |
| Average provider latency seconds | n/a |
| Average moderation latency seconds | n/a |
| Average provider completion TPS | n/a |
| Average end-to-end completion TPS | 53.001609 |

## LLM Errors And Retries

| Error Type | Count |
|---|---:|
| JSONDecodeError | 8 |

### Recovered Error Attempts

| Sample ID | Role | Attempt | Error Type |
|---|---|---:|---|
| 306 | judge_agent | 1 | JSONDecodeError |
| 306 | judge_agent | 2 | JSONDecodeError |
| 2332 | perspective_agent | 1 | JSONDecodeError |
| 4118 | ideation_action_agent | 1 | JSONDecodeError |
| 10773 | ideation_action_agent | 1 | JSONDecodeError |
| 12514 | ideation_action_agent | 1 | JSONDecodeError |
| 12514 | ideation_action_agent | 2 | JSONDecodeError |
| 12514 | ideation_action_agent | 3 | JSONDecodeError |

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

# Compiled Evaluation Results

- Run directory: `logs/runs/20260828_003121_framework4_luna_user_disjoint_test`
- Workflow: `framework_4_hierarchical_with_reviewer`
- Model: `openai/gpt-5.6-luna`
- Requested model: `gpt-5.6-luna`
- Mode: `bool`
- Service tier: `standard`
- Stop-after token threshold: `None`
- Current invocation tokens: `7504098`
- Evaluation status: `completed`
- Stop reason: `None`

## Run Summary

| Metric | Value |
|---|---:|
| Processed records | 4317 |
| Total records | 4317 |
| Failed records | 0 |
| Failure rate | 0.000000 |
| LLM requests | 10870 |
| Failed LLM attempts | 5 |
| JSON/schema parsing errors | 1 |

## Execution Attempts

| Attempt | Start Date (UTC) | End Date (UTC) |
|---:|---|---|
| 1 | 2026-08-28T04:31:21.235667+00:00 | 2026-08-28T06:17:34.257077+00:00 |

## Final-Label Confusion Matrix

| Actual \ Predicted | Indicator | Ideation | Behavior | Attempt | Unsupportive | Error |
|---|---|---|---|---|---|---|
| Indicator | 80 | 196 | 26 | 5 | 374 | 0 |
| Ideation | 27 | 881 | 92 | 21 | 32 | 0 |
| Behavior | 3 | 158 | 129 | 15 | 1 | 0 |
| Attempt | 1 | 19 | 36 | 61 | 0 | 0 |
| Unsupportive | 9 | 24 | 3 | 0 | 2124 | 0 |

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
| 0.657714 | 0.538283 | 0.619286 | 0.516946 | 1750 | 1151 | macro |

### Positive-Label 4x4 Confusion Matrix

| Actual \ Predicted | Indicator | Ideation | Behavior | Attempt |
|---|---|---|---|---|
| Indicator | 80 | 196 | 26 | 5 |
| Ideation | 27 | 881 | 92 | 21 |
| Behavior | 3 | 158 | 129 | 15 |
| Attempt | 1 | 19 | 36 | 61 |

### Per-Label Calculations

| Label | TP | FP | FN | Support | Precision | Recall | F1 Score |
|---|---|---|---|---|---|---|---|
| Indicator | 80 | 31 | 227 | 307 | 0.720721 | 0.260586 | 0.382775 |
| Ideation | 881 | 373 | 140 | 1021 | 0.702552 | 0.862880 | 0.774505 |
| Behavior | 129 | 154 | 176 | 305 | 0.455830 | 0.422951 | 0.438776 |
| Attempt | 61 | 41 | 56 | 117 | 0.598039 | 0.521368 | 0.557078 |

## Suicidal-Intent Positive-Class Grouping Metrics

- Positive: Indicator, Ideation, Behavior, Attempt
- Negative: Unsupportive
- TP/TN/FP/FN below always treat the four suicidality labels as Positive.

| Outcome | Count |
|---|---:|
| TP | 1750 |
| TN | 2124 |
| FP | 36 |
| FN | 407 |

| Accuracy | F1 Score | Precision | Recall | Positive Support | Negative Support | TP | FP | TN | FN |
|---|---|---|---|---|---|---|---|---|---|
| 0.897382 | 0.887649 | 0.979843 | 0.811312 | 2157 | 2160 | 1750 | 36 | 2124 | 407 |

### Suicidal-Intent Positive-Class Grouping Confusion Matrix

| Actual \ Predicted | false | true | Error |
|---|---|---|---|
| false | 2124 | 36 | 0 |
| true | 407 | 1750 | 0 |

## Prespecified Statistical Analysis

- Primary endpoint (prespecified): **Suicidal-intent sensitivity** — `TP / (TP + FN), where any of Indicator, Ideation, Behavior, or Attempt is positive.`
- Rationale: This safety-oriented endpoint measures missed suicidality signs and is invariant to deployment prevalence.
- CI method: two-sided 95% percentile user-cluster bootstrap (2000 resamples; 2036 users; seed 20260820).
- The primary endpoint is prevalence-invariant. PPV and NPV are secondary and depend on prevalence.

### Observed-Sample Estimates And 95% CIs

| Metric | Estimate | 95% CI |
|---|---:|---|
| Sensitivity (primary) | 0.811312 | 0.765762 to 0.850302 |
| Specificity | 0.983333 | 0.977653 to 0.988516 |
| PPV at observed sample prevalence | 0.979843 | 0.970080 to 0.986604 |
| NPV at observed sample prevalence | 0.839194 | 0.789670 to 0.888187 |
| Positive-phase macro F1 | 0.538283 | 0.493614 to 0.581112 |

### Prevalence-Adjusted Predictive Values

These are transport estimates assuming sensitivity and specificity transfer to a target population. They are not estimates of the balanced evaluation sample's prevalence.

| Target Positive Prevalence | PPV (95% CI) | NPV (95% CI) |
|---:|---|---|
| 1.00% | 0.329626 (0.266369 to 0.418087) | 0.998066 (0.997595 to 0.998463) |
| 5.00% | 0.719262 (0.654201 to 0.789190) | 0.990002 (0.987596 to 0.992043) |
| 10.00% | 0.843963 (0.799756 to 0.887680) | 0.979124 (0.974170 to 0.983350) |
| 20.00% | 0.924068 (0.899863 to 0.946758) | 0.954224 (0.943700 to 0.963301) |
| 50.00% | 0.979871 (0.972933 to 0.986136) | 0.839006 (0.807340 to 0.867763) |

## Label Distributions

### Ground Truth

| Label | Count |
|---|---:|
| Attempt | 117 |
| Behavior | 306 |
| Ideation | 1053 |
| Indicator | 681 |
| Unsupportive | 2160 |

### Predictions

| Predicted Label | Count |
|---|---:|
| Attempt | 102 |
| Behavior | 286 |
| Ideation | 1278 |
| Indicator | 120 |
| Unsupportive | 2531 |

## Runtime And Tokens

| Metric | Value |
|---|---:|
| Cumulative case execution duration seconds | 21733.426793 |
| Real execution duration seconds | 5445.728885 |
| Total latency seconds | 21717.703257 |
| Average latency seconds per request | 1.997949 |
| Min latency seconds | 0.894438 |
| Max latency seconds | 62.842071 |
| Total input tokens | 6475484 |
| Total output tokens | 1028614 |
| Total tokens | 7504098 |
| Total cached input tokens | 0 |
| Total cache-write tokens | 0 |
| Average input tokens per request | 595.720699 |
| Average output tokens per request | 94.628703 |
| Average total tokens per request | 690.349402 |
| Total cost USD | 2.545682 |
| Average cost USD per request | 0.000234 |
| Observed TTFT calls | 0 |
| Average TTFT seconds | n/a |
| Average provider generation seconds | n/a |
| Average provider latency seconds | n/a |
| Average moderation latency seconds | n/a |
| Average provider completion TPS | n/a |
| Average end-to-end completion TPS | 47.636983 |

## LLM Errors And Retries

| Error Type | Count |
|---|---:|
| APIConnectionError | 4 |
| JSONDecodeError | 1 |

### Recovered Error Attempts

| Sample ID | Role | Attempt | Error Type |
|---|---|---:|---|
| 2099 | ideation_action_agent | 1 | JSONDecodeError |
| 16015 | secondary_review_agent | 1 | APIConnectionError |
| 16045 | judge_agent | 1 | APIConnectionError |
| 16051 | perspective_agent | 1 | APIConnectionError |
| 16052 | perspective_agent | 1 | APIConnectionError |

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

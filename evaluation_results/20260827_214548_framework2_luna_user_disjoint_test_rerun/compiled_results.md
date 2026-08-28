# Compiled Evaluation Results

- Run directory: `logs/runs/20260827_214548_framework2_luna_user_disjoint_test_rerun`
- Workflow: `framework_2_parallel_with_evidence`
- Model: `openai/gpt-5.6-luna`
- Requested model: `gpt-5.6-luna`
- Mode: `evidence`
- Service tier: `standard`
- Stop-after token threshold: `None`
- Current invocation tokens: `17659314`
- Evaluation status: `completed`
- Stop reason: `None`

## Run Summary

| Metric | Value |
|---|---:|
| Processed records | 4317 |
| Total records | 4317 |
| Failed records | 0 |
| Failure rate | 0.000000 |
| LLM requests | 25906 |
| Failed LLM attempts | 4 |
| JSON/schema parsing errors | 4 |

## Execution Attempts

| Attempt | Start Date (UTC) | End Date (UTC) |
|---:|---|---|
| 1 | 2026-08-28T01:45:48.815761+00:00 | 2026-08-28T03:04:15.740945+00:00 |

## Final-Label Confusion Matrix

| Actual \ Predicted | Indicator | Ideation | Behavior | Attempt | Unsupportive | Error |
|---|---|---|---|---|---|---|
| Indicator | 55 | 154 | 45 | 5 | 422 | 0 |
| Ideation | 24 | 769 | 186 | 26 | 48 | 0 |
| Behavior | 0 | 67 | 220 | 16 | 3 | 0 |
| Attempt | 1 | 12 | 41 | 62 | 1 | 0 |
| Unsupportive | 2 | 14 | 6 | 0 | 2138 | 0 |

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
|---|---|---|---|---|---|---|---|
| Indicator | 55 | 25 | 204 | 259 | 0.687500 | 0.212355 | 0.324484 |
| Ideation | 769 | 233 | 236 | 1005 | 0.767465 | 0.765174 | 0.766318 |
| Behavior | 220 | 272 | 83 | 303 | 0.447154 | 0.726073 | 0.553459 |
| Attempt | 62 | 47 | 54 | 116 | 0.568807 | 0.534483 | 0.551111 |

## Suicidal-Intent Positive-Class Grouping Metrics

- Positive: Indicator, Ideation, Behavior, Attempt
- Negative: Unsupportive
- TP/TN/FP/FN below always treat the four suicidality labels as Positive.

| Outcome | Count |
|---|---:|
| TP | 1683 |
| TN | 2138 |
| FP | 22 |
| FN | 474 |

| Accuracy | F1 Score | Precision | Recall | Positive Support | Negative Support | TP | FP | TN | FN |
|---|---|---|---|---|---|---|---|---|---|
| 0.885105 | 0.871569 | 0.987097 | 0.780250 | 2157 | 2160 | 1683 | 22 | 2138 | 474 |

### Suicidal-Intent Positive-Class Grouping Confusion Matrix

| Actual \ Predicted | false | true | Error |
|---|---|---|---|
| false | 2138 | 22 | 0 |
| true | 474 | 1683 | 0 |

## Prespecified Statistical Analysis

- Primary endpoint (prespecified): **Suicidal-intent sensitivity** — `TP / (TP + FN), where any of Indicator, Ideation, Behavior, or Attempt is positive.`
- Rationale: This safety-oriented endpoint measures missed suicidality signs and is invariant to deployment prevalence.
- CI method: two-sided 95% percentile user-cluster bootstrap (2000 resamples; 2036 users; seed 20260820).
- The primary endpoint is prevalence-invariant. PPV and NPV are secondary and depend on prevalence.

### Observed-Sample Estimates And 95% CIs

| Metric | Estimate | 95% CI |
|---|---:|---|
| Sensitivity (primary) | 0.780250 | 0.734279 to 0.820214 |
| Specificity | 0.989815 | 0.985251 to 0.993937 |
| PPV at observed sample prevalence | 0.987097 | 0.979531 to 0.992619 |
| NPV at observed sample prevalence | 0.818530 | 0.766744 to 0.870776 |
| Positive-phase macro F1 | 0.548843 | 0.509903 to 0.586918 |

### Prevalence-Adjusted Predictive Values

These are transport estimates assuming sensitivity and specificity transfer to a target population. They are not estimates of the balanced evaluation sample's prevalence.

| Target Positive Prevalence | PPV (95% CI) | NPV (95% CI) |
|---:|---|---|
| 1.00% | 0.436239 (0.349143 to 0.566128) | 0.997762 (0.997298 to 0.998171) |
| 5.00% | 0.801269 (0.736503 to 0.871776) | 0.988450 (0.986077 to 0.990545) |
| 10.00% | 0.894868 (0.855089 to 0.934867) | 0.975926 (0.971055 to 0.980246) |
| 20.00% | 0.950376 (0.929956 to 0.969965) | 0.947416 (0.937148 to 0.956625) |
| 50.00% | 0.987114 (0.981518 to 0.992318) | 0.818323 (0.788476 to 0.846476) |

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
| Attempt | 109 |
| Behavior | 498 |
| Ideation | 1016 |
| Indicator | 82 |
| Unsupportive | 2612 |

## Runtime And Tokens

| Metric | Value |
|---|---:|
| Cumulative case execution duration seconds | 18728.292586 |
| Real execution duration seconds | 4706.079690 |
| Total latency seconds | 46191.717323 |
| Average latency seconds per request | 1.783051 |
| Min latency seconds | 0.694208 |
| Max latency seconds | 61.871107 |
| Total input tokens | 15809904 |
| Total output tokens | 1849410 |
| Total tokens | 17659314 |
| Total cached input tokens | 0 |
| Total cache-write tokens | 4519855 |
| Average input tokens per request | 610.279626 |
| Average output tokens per request | 71.389253 |
| Average total tokens per request | 681.668880 |
| Total cost USD | 5.607305 |
| Average cost USD per request | 0.000216 |
| Observed TTFT calls | 0 |
| Average TTFT seconds | n/a |
| Average provider generation seconds | n/a |
| Average provider latency seconds | n/a |
| Average moderation latency seconds | n/a |
| Average provider completion TPS | n/a |
| Average end-to-end completion TPS | 31.759404 |

## LLM Errors And Retries

| Error Type | Count |
|---|---:|
| JSONDecodeError | 2 |
| ValidationError | 2 |

### Recovered Error Attempts

| Sample ID | Role | Attempt | Error Type |
|---|---|---:|---|
| 5427 | Indicator_collector | 1 | JSONDecodeError |
| 5978 | Indicator_collector | 1 | JSONDecodeError |
| 16573 | Unsupportive_collector | 1 | ValidationError |
| 29790 | Unsupportive_collector | 1 | ValidationError |

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

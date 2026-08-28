# Compiled Evaluation Results

- Run directory: `logs/runs/20260827_230650_framework3_luna_user_disjoint_test`
- Workflow: `framework_3_hierarchical_no_reviewer`
- Model: `openai/gpt-5.6-luna`
- Requested model: `gpt-5.6-luna`
- Mode: `bool`
- Service tier: `standard`
- Stop-after token threshold: `None`
- Current invocation tokens: `8717580`
- Evaluation status: `completed`
- Stop reason: `None`

## Run Summary

| Metric | Value |
|---|---:|
| Processed records | 4317 |
| Total records | 4317 |
| Failed records | 0 |
| Failure rate | 0.000000 |
| LLM requests | 10702 |
| Failed LLM attempts | 2 |
| JSON/schema parsing errors | 2 |

## Execution Attempts

| Attempt | Start Date (UTC) | End Date (UTC) |
|---:|---|---|
| 1 | 2026-08-28T03:06:50.938318+00:00 | 2026-08-28T04:31:08.334466+00:00 |

## Final-Label Confusion Matrix

| Actual \ Predicted | Indicator | Ideation | Behavior | Attempt | Unsupportive | Error |
|---|---|---|---|---|---|---|
| Indicator | 53 | 202 | 24 | 8 | 394 | 0 |
| Ideation | 23 | 894 | 82 | 21 | 33 | 0 |
| Behavior | 0 | 158 | 132 | 14 | 2 | 0 |
| Attempt | 1 | 19 | 34 | 63 | 0 | 0 |
| Unsupportive | 2 | 24 | 5 | 0 | 2129 | 0 |

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
| 0.660880 | 0.523582 | 0.617556 | 0.508453 | 1728 | 1142 | macro |

### Positive-Label 4x4 Confusion Matrix

| Actual \ Predicted | Indicator | Ideation | Behavior | Attempt |
|---|---|---|---|---|
| Indicator | 53 | 202 | 24 | 8 |
| Ideation | 23 | 894 | 82 | 21 |
| Behavior | 0 | 158 | 132 | 14 |
| Attempt | 1 | 19 | 34 | 63 |

### Per-Label Calculations

| Label | TP | FP | FN | Support | Precision | Recall | F1 Score |
|---|---|---|---|---|---|---|---|
| Indicator | 53 | 24 | 234 | 287 | 0.688312 | 0.184669 | 0.291209 |
| Ideation | 894 | 379 | 126 | 1020 | 0.702278 | 0.876471 | 0.779765 |
| Behavior | 132 | 140 | 172 | 304 | 0.485294 | 0.434211 | 0.458333 |
| Attempt | 63 | 43 | 54 | 117 | 0.594340 | 0.538462 | 0.565022 |

## Suicidal-Intent Positive-Class Grouping Metrics

- Positive: Indicator, Ideation, Behavior, Attempt
- Negative: Unsupportive
- TP/TN/FP/FN below always treat the four suicidality labels as Positive.

| Outcome | Count |
|---|---:|
| TP | 1728 |
| TN | 2129 |
| FP | 31 |
| FN | 429 |

| Accuracy | F1 Score | Precision | Recall | Positive Support | Negative Support | TP | FP | TN | FN |
|---|---|---|---|---|---|---|---|---|---|
| 0.893445 | 0.882533 | 0.982376 | 0.801113 | 2157 | 2160 | 1728 | 31 | 2129 | 429 |

### Suicidal-Intent Positive-Class Grouping Confusion Matrix

| Actual \ Predicted | false | true | Error |
|---|---|---|---|
| false | 2129 | 31 | 0 |
| true | 429 | 1728 | 0 |

## Prespecified Statistical Analysis

- Primary endpoint (prespecified): **Suicidal-intent sensitivity** — `TP / (TP + FN), where any of Indicator, Ideation, Behavior, or Attempt is positive.`
- Rationale: This safety-oriented endpoint measures missed suicidality signs and is invariant to deployment prevalence.
- CI method: two-sided 95% percentile user-cluster bootstrap (2000 resamples; 2036 users; seed 20260820).
- The primary endpoint is prevalence-invariant. PPV and NPV are secondary and depend on prevalence.

### Observed-Sample Estimates And 95% CIs

| Metric | Estimate | 95% CI |
|---|---:|---|
| Sensitivity (primary) | 0.801113 | 0.756029 to 0.840752 |
| Specificity | 0.985648 | 0.980327 to 0.990396 |
| PPV at observed sample prevalence | 0.982376 | 0.973271 to 0.988678 |
| NPV at observed sample prevalence | 0.832291 | 0.782782 to 0.881546 |
| Positive-phase macro F1 | 0.523582 | 0.478862 to 0.567244 |

### Prevalence-Adjusted Predictive Values

These are transport estimates assuming sensitivity and specificity transfer to a target population. They are not estimates of the balanced evaluation sample's prevalence.

| Target Positive Prevalence | PPV (95% CI) | NPV (95% CI) |
|---:|---|---|
| 1.00% | 0.360546 (0.290804 to 0.459009) | 0.997966 (0.997505 to 0.998371) |
| 5.00% | 0.746055 (0.681180 to 0.815530) | 0.989491 (0.987134 to 0.991568) |
| 10.00% | 0.861153 (0.818529 to 0.903223) | 0.978071 (0.973222 to 0.982364) |
| 20.00% | 0.933132 (0.910303 to 0.954544) | 0.951977 (0.941701 to 0.961174) |
| 50.00% | 0.982400 (0.975958 to 0.988235) | 0.832097 (0.801519 to 0.860898) |

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
| Attempt | 106 |
| Behavior | 277 |
| Ideation | 1297 |
| Indicator | 79 |
| Unsupportive | 2558 |

## Runtime And Tokens

| Metric | Value |
|---|---:|
| Cumulative case execution duration seconds | 20178.383414 |
| Real execution duration seconds | 5057.221787 |
| Total latency seconds | 20165.168051 |
| Average latency seconds per request | 1.884243 |
| Min latency seconds | 0.898848 |
| Max latency seconds | 14.683605 |
| Total input tokens | 7864384 |
| Total output tokens | 853196 |
| Total tokens | 8717580 |
| Total cached input tokens | 0 |
| Total cache-write tokens | 0 |
| Average input tokens per request | 734.851803 |
| Average output tokens per request | 79.723042 |
| Average total tokens per request | 814.574846 |
| Total cost USD | 2.624225 |
| Average cost USD per request | 0.000245 |
| Observed TTFT calls | 0 |
| Average TTFT seconds | n/a |
| Average provider generation seconds | n/a |
| Average provider latency seconds | n/a |
| Average moderation latency seconds | n/a |
| Average provider completion TPS | n/a |
| Average end-to-end completion TPS | 42.200450 |

## LLM Errors And Retries

| Error Type | Count |
|---|---:|
| JSONDecodeError | 2 |

### Recovered Error Attempts

| Sample ID | Role | Attempt | Error Type |
|---|---|---:|---|
| 2099 | ideation_action_agent | 1 | JSONDecodeError |
| 2099 | ideation_action_agent | 2 | JSONDecodeError |

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

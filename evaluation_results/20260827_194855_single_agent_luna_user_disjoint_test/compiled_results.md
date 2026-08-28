# Compiled Evaluation Results

- Run directory: `logs/runs/20260827_194855_single_agent_luna_user_disjoint_test`
- Workflow: `single_agent`
- Model: `openai/gpt-5.6-luna`
- Requested model: `gpt-5.6-luna`
- Mode: `single-class`
- Service tier: `standard`
- Stop-after token threshold: `None`
- Current invocation tokens: `2275080`
- Evaluation status: `completed`
- Stop reason: `None`

## Run Summary

| Metric | Value |
|---|---:|
| Processed records | 4317 |
| Total records | 4317 |
| Failed records | 0 |
| Failure rate | 0.000000 |
| LLM requests | 4317 |
| Failed LLM attempts | 0 |
| JSON/schema parsing errors | 0 |

## Execution Attempts

| Attempt | Start Date (UTC) | End Date (UTC) |
|---:|---|---|
| 1 | 2026-08-27T23:48:55.941397+00:00 | 2026-08-28T00:02:17.829187+00:00 |

## Final-Label Confusion Matrix

| Actual \ Predicted | Indicator | Ideation | Behavior | Attempt | Unsupportive | Error |
|---|---|---|---|---|---|---|
| Indicator | 46 | 194 | 23 | 8 | 410 | 0 |
| Ideation | 19 | 908 | 59 | 25 | 42 | 0 |
| Behavior | 0 | 194 | 80 | 30 | 2 | 0 |
| Attempt | 1 | 23 | 25 | 68 | 0 | 0 |
| Unsupportive | 6 | 22 | 2 | 0 | 2130 | 0 |

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
|---|---|---|---|---|---|---|---|
| Indicator | 46 | 20 | 225 | 271 | 0.696970 | 0.169742 | 0.272997 |
| Ideation | 908 | 411 | 103 | 1011 | 0.688400 | 0.898121 | 0.779399 |
| Behavior | 80 | 107 | 224 | 304 | 0.427807 | 0.263158 | 0.325866 |
| Attempt | 68 | 63 | 49 | 117 | 0.519084 | 0.581197 | 0.548387 |

## Suicidal-Intent Positive-Class Grouping Metrics

- Positive: Indicator, Ideation, Behavior, Attempt
- Negative: Unsupportive
- TP/TN/FP/FN below always treat the four suicidality labels as Positive.

| Outcome | Count |
|---|---:|
| TP | 1703 |
| TN | 2130 |
| FP | 30 |
| FN | 454 |

| Accuracy | F1 Score | Precision | Recall | Positive Support | Negative Support | TP | FP | TN | FN |
|---|---|---|---|---|---|---|---|---|---|
| 0.887885 | 0.875578 | 0.982689 | 0.789522 | 2157 | 2160 | 1703 | 30 | 2130 | 454 |

### Suicidal-Intent Positive-Class Grouping Confusion Matrix

| Actual \ Predicted | false | true | Error |
|---|---|---|---|
| false | 2130 | 30 | 0 |
| true | 454 | 1703 | 0 |

## Prespecified Statistical Analysis

- Primary endpoint (prespecified): **Suicidal-intent sensitivity** — `TP / (TP + FN), where any of Indicator, Ideation, Behavior, or Attempt is positive.`
- Rationale: This safety-oriented endpoint measures missed suicidality signs and is invariant to deployment prevalence.
- CI method: two-sided 95% percentile user-cluster bootstrap (2000 resamples; 2036 users; seed 20260820).
- The primary endpoint is prevalence-invariant. PPV and NPV are secondary and depend on prevalence.

### Observed-Sample Estimates And 95% CIs

| Metric | Estimate | 95% CI |
|---|---:|---|
| Sensitivity (primary) | 0.789522 | 0.743742 to 0.829345 |
| Specificity | 0.986111 | 0.981069 to 0.990797 |
| PPV at observed sample prevalence | 0.982689 | 0.973937 to 0.989163 |
| NPV at observed sample prevalence | 0.824303 | 0.773980 to 0.874695 |
| Positive-phase macro F1 | 0.481662 | 0.442408 to 0.523579 |

### Prevalence-Adjusted Predictive Values

These are transport estimates assuming sensitivity and specificity transfer to a target population. They are not estimates of the balanced evaluation sample's prevalence.

| Target Positive Prevalence | PPV (95% CI) | NPV (95% CI) |
|---:|---|---|
| 1.00% | 0.364756 (0.295154 to 0.464864) | 0.997849 (0.997384 to 0.998256) |
| 5.00% | 0.749491 (0.685723 to 0.819047) | 0.988891 (0.986516 to 0.990980) |
| 10.00% | 0.863317 (0.821627 to 0.905263) | 0.976834 (0.971953 to 0.981146) |
| 20.00% | 0.934260 (0.912003 to 0.955555) | 0.949343 (0.939032 to 0.958556) |
| 50.00% | 0.982713 (0.976446 to 0.988506) | 0.824102 (0.793835 to 0.852555) |

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
| Attempt | 131 |
| Behavior | 189 |
| Ideation | 1341 |
| Indicator | 72 |
| Unsupportive | 2584 |

## Runtime And Tokens

| Metric | Value |
|---|---:|
| Cumulative case execution duration seconds | 6383.321366 |
| Real execution duration seconds | 801.738207 |
| Total latency seconds | 6383.132482 |
| Average latency seconds per request | 1.478604 |
| Min latency seconds | 0.701493 |
| Max latency seconds | 8.118277 |
| Total input tokens | 2151784 |
| Total output tokens | 123296 |
| Total tokens | 2275080 |
| Total cached input tokens | 0 |
| Total cache-write tokens | 0 |
| Average input tokens per request | 498.444290 |
| Average output tokens per request | 28.560574 |
| Average total tokens per request | 527.004864 |
| Total cost USD | 0.580690 |
| Average cost USD per request | 0.000135 |
| Observed TTFT calls | 0 |
| Average TTFT seconds | n/a |
| Average provider generation seconds | n/a |
| Average provider latency seconds | n/a |
| Average moderation latency seconds | n/a |
| Average provider completion TPS | n/a |
| Average end-to-end completion TPS | 15.997821 |

## LLM Errors And Retries

| Error Type | Count |
|---|---:|
| None | 0 |

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

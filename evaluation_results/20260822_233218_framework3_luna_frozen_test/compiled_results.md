# Compiled Evaluation Results

- Run directory: `logs/runs/20260822_233218_framework3_luna_frozen_test`
- Workflow: `framework_3_hierarchical_no_reviewer`
- Model: `openai/gpt-5.6-luna`
- Requested model: `gpt-5.6-luna`
- Mode: `bool`
- Service tier: `standard`
- Stop-after token threshold: `None`
- Current invocation tokens: `8804177`
- Evaluation status: `completed`
- Stop reason: `None`

## Run Summary

| Metric | Value |
|---|---:|
| Processed records | 4319 |
| Total records | 4319 |
| Failed records | 0 |
| Failure rate | 0.000000 |
| LLM requests | 10804 |
| Failed LLM attempts | 3 |
| JSON/schema parsing errors | 3 |

## Execution Attempts

| Attempt | Start Date (UTC) | End Date (UTC) |
|---:|---|---|
| 1 | 2026-08-23T03:32:18.257966+00:00 | 2026-08-23T04:09:35.420173+00:00 |

## Final-Label Confusion Matrix

| Actual \ Predicted | Indicator | Ideation | Behavior | Attempt | Unsupportive | Error |
|---|---|---|---|---|---|---|
| Indicator | 42 | 215 | 33 | 9 | 384 | 0 |
| Ideation | 20 | 859 | 112 | 36 | 26 | 0 |
| Behavior | 4 | 135 | 146 | 19 | 2 | 0 |
| Attempt | 0 | 14 | 33 | 71 | 0 | 0 |
| Unsupportive | 10 | 16 | 2 | 1 | 2130 | 0 |

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
| 0.639588 | 0.504981 | 0.578820 | 0.514711 | 1748 | 1118 | macro |

### Positive-Label 4x4 Confusion Matrix

| Actual \ Predicted | Indicator | Ideation | Behavior | Attempt |
|---|---|---|---|---|
| Indicator | 42 | 215 | 33 | 9 |
| Ideation | 20 | 859 | 112 | 36 |
| Behavior | 4 | 135 | 146 | 19 |
| Attempt | 0 | 14 | 33 | 71 |

### Per-Label Calculations

| Label | TP | FP | FN | Support | Precision | Recall | F1 Score |
|---|---|---|---|---|---|---|---|
| Indicator | 42 | 24 | 257 | 299 | 0.636364 | 0.140468 | 0.230137 |
| Ideation | 859 | 364 | 168 | 1027 | 0.702371 | 0.836417 | 0.763556 |
| Behavior | 146 | 178 | 158 | 304 | 0.450617 | 0.480263 | 0.464968 |
| Attempt | 71 | 64 | 47 | 118 | 0.525926 | 0.601695 | 0.561265 |

## Suicidal-Intent Positive-Class Grouping Metrics

- Positive: Indicator, Ideation, Behavior, Attempt
- Negative: Unsupportive
- TP/TN/FP/FN below always treat the four suicidality labels as Positive.

| Outcome | Count |
|---|---:|
| TP | 1748 |
| TN | 2130 |
| FP | 29 |
| FN | 412 |

| Accuracy | F1 Score | Precision | Recall | Positive Support | Negative Support | TP | FP | TN | FN |
|---|---|---|---|---|---|---|---|---|---|
| 0.897893 | 0.887986 | 0.983680 | 0.809259 | 2160 | 2159 | 1748 | 29 | 2130 | 412 |

### Suicidal-Intent Positive-Class Grouping Confusion Matrix

| Actual \ Predicted | false | true | Error |
|---|---|---|---|
| false | 2130 | 29 | 0 |
| true | 412 | 1748 | 0 |

## Prespecified Statistical Analysis

- Primary endpoint (prespecified): **Suicidal-intent sensitivity** — `TP / (TP + FN), where any of Indicator, Ideation, Behavior, or Attempt is positive.`
- Rationale: This safety-oriented endpoint measures missed suicidality signs and is invariant to deployment prevalence.
- CI method: two-sided 95% percentile user-cluster bootstrap (2000 resamples; 2721 users; seed 20260820).
- The primary endpoint is prevalence-invariant. PPV and NPV are secondary and depend on prevalence.

### Observed-Sample Estimates And 95% CIs

| Metric | Estimate | 95% CI |
|---|---:|---|
| Sensitivity (primary) | 0.809259 | 0.784840 to 0.832114 |
| Specificity | 0.986568 | 0.981583 to 0.991167 |
| PPV at observed sample prevalence | 0.983680 | 0.977339 to 0.989457 |
| NPV at observed sample prevalence | 0.837923 | 0.813854 to 0.861634 |
| Positive-phase macro F1 | 0.504981 | 0.474542 to 0.536648 |

### Prevalence-Adjusted Predictive Values

These are transport estimates assuming sensitivity and specificity transfer to a target population. They are not estimates of the balanced evaluation sample's prevalence.

| Target Positive Prevalence | PPV (95% CI) | NPV (95% CI) |
|---:|---|---|
| 1.00% | 0.378328 (0.306162 to 0.478660) | 0.998051 (0.997803 to 0.998287) |
| 5.00% | 0.760246 (0.696896 to 0.827108) | 0.989927 (0.988656 to 0.991136) |
| 10.00% | 0.870032 (0.829172 to 0.909906) | 0.978970 (0.976350 to 0.981469) |
| 20.00% | 0.937741 (0.916116 to 0.957848) | 0.953894 (0.948316 to 0.959250) |
| 50.00% | 0.983673 (0.977621 to 0.989118) | 0.837986 (0.821016 to 0.854755) |

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
| Attempt | 136 |
| Behavior | 326 |
| Ideation | 1239 |
| Indicator | 76 |
| Unsupportive | 2542 |

## Runtime And Tokens

| Metric | Value |
|---|---:|
| Cumulative case execution duration seconds | 17839.968986 |
| Real execution duration seconds | 2236.966769 |
| Total latency seconds | 17828.553973 |
| Average latency seconds per request | 1.650181 |
| Min latency seconds | 0.882155 |
| Max latency seconds | 22.564605 |
| Total input tokens | 7952588 |
| Total output tokens | 851589 |
| Total tokens | 8804177 |
| Total cached input tokens | 0 |
| Total cache-write tokens | 0 |
| Average input tokens per request | 736.078119 |
| Average output tokens per request | 78.821640 |
| Average total tokens per request | 814.899759 |
| Total cost USD | 2.643377 |
| Average cost USD per request | 0.000245 |
| Observed TTFT calls | 0 |
| Average TTFT seconds | n/a |
| Average provider generation seconds | n/a |
| Average provider latency seconds | n/a |
| Average moderation latency seconds | n/a |
| Average provider completion TPS | n/a |
| Average end-to-end completion TPS | 47.481899 |

## LLM Errors And Retries

| Error Type | Count |
|---|---:|
| JSONDecodeError | 3 |

### Recovered Error Attempts

| Sample ID | Role | Attempt | Error Type |
|---|---|---:|---|
| 212 | behavior_attempt_agent | 1 | JSONDecodeError |
| 6294 | ideation_action_agent | 1 | JSONDecodeError |
| 11746 | perspective_agent | 1 | JSONDecodeError |

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

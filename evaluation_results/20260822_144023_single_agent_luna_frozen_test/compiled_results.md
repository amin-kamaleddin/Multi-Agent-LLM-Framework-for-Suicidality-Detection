# Compiled Evaluation Results

- Run directory: `logs/runs/20260822_144023_single_agent_luna_frozen_test`
- Workflow: `single_agent`
- Model: `openai/gpt-5.6-luna`
- Requested model: `gpt-5.6-luna`
- Mode: `single-class`
- Service tier: `standard`
- Stop-after token threshold: `None`
- Current invocation tokens: `2288377`
- Evaluation status: `completed`
- Stop reason: `None`

## Run Summary

| Metric | Value |
|---|---:|
| Processed records | 4319 |
| Total records | 4319 |
| Failed records | 0 |
| Failure rate | 0.000000 |
| LLM requests | 4319 |
| Failed LLM attempts | 0 |
| JSON/schema parsing errors | 0 |

## Execution Attempts

| Attempt | Start Date (UTC) | End Date (UTC) |
|---:|---|---|
| 1 | 2026-08-22T18:40:23.289908+00:00 | 2026-08-22T18:52:22.802404+00:00 |

## Final-Label Confusion Matrix

| Actual \ Predicted | Indicator | Ideation | Behavior | Attempt | Unsupportive | Error |
|---|---|---|---|---|---|---|
| Indicator | 34 | 208 | 31 | 12 | 398 | 0 |
| Ideation | 21 | 871 | 92 | 43 | 26 | 0 |
| Behavior | 3 | 159 | 107 | 35 | 2 | 0 |
| Attempt | 0 | 15 | 27 | 76 | 0 | 0 |
| Unsupportive | 11 | 14 | 0 | 1 | 2133 | 0 |

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
| 0.627451 | 0.469740 | 0.538878 | 0.490860 | 1734 | 1088 | macro |

### Positive-Label 4x4 Confusion Matrix

| Actual \ Predicted | Indicator | Ideation | Behavior | Attempt |
|---|---|---|---|---|
| Indicator | 34 | 208 | 31 | 12 |
| Ideation | 21 | 871 | 92 | 43 |
| Behavior | 3 | 159 | 107 | 35 |
| Attempt | 0 | 15 | 27 | 76 |

### Per-Label Calculations

| Label | TP | FP | FN | Support | Precision | Recall | F1 Score |
|---|---|---|---|---|---|---|---|
| Indicator | 34 | 24 | 251 | 285 | 0.586207 | 0.119298 | 0.198251 |
| Ideation | 871 | 382 | 156 | 1027 | 0.695132 | 0.848101 | 0.764035 |
| Behavior | 107 | 150 | 197 | 304 | 0.416342 | 0.351974 | 0.381462 |
| Attempt | 76 | 90 | 42 | 118 | 0.457831 | 0.644068 | 0.535211 |

## Suicidal-Intent Positive-Class Grouping Metrics

- Positive: Indicator, Ideation, Behavior, Attempt
- Negative: Unsupportive
- TP/TN/FP/FN below always treat the four suicidality labels as Positive.

| Outcome | Count |
|---|---:|
| TP | 1734 |
| TN | 2133 |
| FP | 26 |
| FN | 426 |

| Accuracy | F1 Score | Precision | Recall | Positive Support | Negative Support | TP | FP | TN | FN |
|---|---|---|---|---|---|---|---|---|---|
| 0.895346 | 0.884694 | 0.985227 | 0.802778 | 2160 | 2159 | 1734 | 26 | 2133 | 426 |

### Suicidal-Intent Positive-Class Grouping Confusion Matrix

| Actual \ Predicted | false | true | Error |
|---|---|---|---|
| false | 2133 | 26 | 0 |
| true | 426 | 1734 | 0 |

## Prespecified Statistical Analysis

- Primary endpoint (prespecified): **Suicidal-intent sensitivity** — `TP / (TP + FN), where any of Indicator, Ideation, Behavior, or Attempt is positive.`
- Rationale: This safety-oriented endpoint measures missed suicidality signs and is invariant to deployment prevalence.
- CI method: two-sided 95% percentile user-cluster bootstrap (2000 resamples; 2721 users; seed 20260820).
- The primary endpoint is prevalence-invariant. PPV and NPV are secondary and depend on prevalence.

### Observed-Sample Estimates And 95% CIs

| Metric | Estimate | 95% CI |
|---|---:|---|
| Sensitivity (primary) | 0.802778 | 0.777725 to 0.826255 |
| Specificity | 0.987957 | 0.982798 to 0.992288 |
| PPV at observed sample prevalence | 0.985227 | 0.978761 to 0.990767 |
| NPV at observed sample prevalence | 0.833529 | 0.808682 to 0.858186 |
| Positive-phase macro F1 | 0.469740 | 0.440904 to 0.498386 |

### Prevalence-Adjusted Predictive Values

These are transport estimates assuming sensitivity and specificity transfer to a target population. They are not estimates of the balanced evaluation sample's prevalence.

| Target Positive Prevalence | PPV (95% CI) | NPV (95% CI) |
|---:|---|---|
| 1.00% | 0.402396 (0.320830 to 0.513194) | 0.997988 (0.997734 to 0.998230) |
| 5.00% | 0.778197 (0.711098 to 0.845987) | 0.989603 (0.988307 to 0.990848) |
| 10.00% | 0.881049 (0.838612 to 0.920612) | 0.978301 (0.975631 to 0.980873) |
| 20.00% | 0.943392 (0.921207 to 0.963088) | 0.952466 (0.946790 to 0.957968) |
| 50.00% | 0.985221 (0.979065 to 0.990509) | 0.833593 (0.816457 to 0.850699) |

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
| Attempt | 167 |
| Behavior | 257 |
| Ideation | 1267 |
| Indicator | 69 |
| Unsupportive | 2559 |

## Runtime And Tokens

| Metric | Value |
|---|---:|
| Cumulative case execution duration seconds | 5724.938436 |
| Real execution duration seconds | 719.370310 |
| Total latency seconds | 5724.756352 |
| Average latency seconds per request | 1.325482 |
| Min latency seconds | 0.617076 |
| Max latency seconds | 6.920468 |
| Total input tokens | 2164709 |
| Total output tokens | 123668 |
| Total tokens | 2288377 |
| Total cached input tokens | 0 |
| Total cache-write tokens | 0 |
| Average input tokens per request | 501.206066 |
| Average output tokens per request | 28.633480 |
| Average total tokens per request | 529.839546 |
| Total cost USD | 0.582981 |
| Average cost USD per request | 0.000135 |
| Observed TTFT calls | 0 |
| Average TTFT seconds | n/a |
| Average provider generation seconds | n/a |
| Average provider latency seconds | n/a |
| Average moderation latency seconds | n/a |
| Average provider completion TPS | n/a |
| Average end-to-end completion TPS | 17.550804 |

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

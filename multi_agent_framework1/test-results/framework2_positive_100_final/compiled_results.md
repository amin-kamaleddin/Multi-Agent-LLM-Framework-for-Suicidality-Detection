# Compiled Evaluation Results

- Run directory: `logs\runs\20260717_120911_framework2_gpt4o_mini_positive_100_final`
- Workflow: `framework2`
- Model: `openai/gpt-4o-mini`
- Requested model: `gpt-4o-mini`
- Mode: `evidence`
- Evaluation status: `completed`
- Stop reason: `None`

## Run Summary

| Metric | Value |
|---|---:|
| Processed records | 100 |
| Total records | 100 |
| Correct predictions | 35 |
| Failed records | 0 |
| Failure rate | 0.000000 |
| LLM requests | 600 |
| Failed LLM attempts | 0 |
| JSON/schema parsing errors | 0 |

## Model Metrics

| Model | Accuracy | F1 Score | Precision | Recall |
|---|---|---|---|---|
| openai/gpt-4o-mini | 0.350000 | 0.226414 | 0.352727 | 0.280000 |

## Per-Label Metrics

| Label | Accuracy | F1 Score | Precision | Recall | Support | TP | FP | TN | FN |
|---|---|---|---|---|---|---|---|---|---|
| Indicator | 0.760000 | 0.200000 | 0.600000 | 0.120000 | 25 | 3 | 2 | 73 | 22 |
| Ideation | 0.670000 | 0.266667 | 0.300000 | 0.240000 | 25 | 6 | 14 | 61 | 19 |
| Behavior | 0.750000 | 0.137931 | 0.500000 | 0.080000 | 25 | 2 | 2 | 73 | 23 |
| Attempt | 0.570000 | 0.527473 | 0.363636 | 0.960000 | 25 | 24 | 42 | 33 | 1 |
| Unsupportive | 0.950000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 5 | 95 | 0 |

## Confusion Matrix

| Actual \ Predicted | Indicator | Ideation | Behavior | Attempt | Unsupportive | Error |
|---|---|---|---|---|---|---|
| Indicator | 3 | 12 | 0 | 6 | 4 | 0 |
| Ideation | 2 | 6 | 1 | 15 | 1 | 0 |
| Behavior | 0 | 2 | 2 | 21 | 0 | 0 |
| Attempt | 0 | 0 | 1 | 24 | 0 | 0 |
| Unsupportive | 0 | 0 | 0 | 0 | 0 | 0 |

## Suicidal-Intent Positive-Class Grouping Metrics

| Value | Accuracy | F1 Score | Precision | Recall | Support | TP | FP | TN | FN |
|---|---|---|---|---|---|---|---|---|---|
| false | 0.950000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 5 | 95 | 0 |
| true | 0.950000 | 0.974359 | 1.000000 | 0.950000 | 100 | 95 | 0 | 0 | 5 |

### Suicidal-Intent Positive-Class Grouping Confusion Matrix

| Actual \ Predicted | false | true | Error |
|---|---|---|---|
| false | 0 | 0 | 0 |
| true | 5 | 95 | 0 |

## Label Distributions

### Ground Truth

| Label | Count |
|---|---:|
| Attempt | 25 |
| Behavior | 25 |
| Ideation | 25 |
| Indicator | 25 |

### Predictions

| Predicted Label | Count |
|---|---:|
| Attempt | 66 |
| Behavior | 4 |
| Ideation | 20 |
| Indicator | 5 |
| Unsupportive | 5 |

## Runtime And Tokens

| Metric | Value |
|---|---:|
| Total latency seconds | 1131.738986 |
| Average latency seconds per request | 1.886232 |
| Min latency seconds | 0.389235 |
| Max latency seconds | 6.100801 |
| Total input tokens | 407009 |
| Total output tokens | 47592 |
| Total tokens | 454601 |
| Total cached input tokens | 10240 |
| Total cache-write tokens | 0 |
| Average input tokens per request | 678.348333 |
| Average output tokens per request | 79.320000 |
| Average total tokens per request | 757.668333 |
| Total cost USD | 0.088839 |
| Average cost USD per request | 0.000148 |
| Observed TTFT calls | 0 |
| Average TTFT seconds | n/a |
| Average provider generation seconds | n/a |
| Average provider latency seconds | n/a |
| Average moderation latency seconds | n/a |
| Average provider completion TPS | n/a |
| Average end-to-end completion TPS | 34.478943 |

## LLM Errors And Retries

| Error Type | Count |
|---|---:|
| None | 0 |

## Suicidal-Intent Grouping Metric Limitation

> Only one suicidal_intent ground-truth class is present; the confusion matrix is valid, but macro and missing-class metrics are not representative of two-class performance.

## Trace Files

| Artifact | Purpose |
|---|---|
| `predictions.csv` | Per-record source fields, shared outputs, correctness, errors, and optional workflow-supplied agent outputs. |
| `model_metrics.csv` | Aggregate model-level Accuracy, F1, Precision, Recall. |
| `label_metrics.csv` | Per-label one-vs-rest metrics. |
| `agent_metrics.csv` | Optional workflow-supplied per-agent metric summary; absent when agent tasks are not comparable to ground truth. |
| `suicidal_intent_metrics.csv` | Binary false/true per-class metrics. |
| `metrics.json` | Complete metrics payload. |
| `results.json` | Complete per-record outputs and metrics payload. |
| `misclassified_ids.txt` | IDs with an incorrect final-label evaluation. |
| `suicidal_intent_misclassified_ids.txt` | IDs with an incorrect binary evaluation. |
| `classification_metrics.svg` | Final-label and suicidal-intent positive-class grouping visualization. |
| `runtime_metrics.svg` | Cost, token, latency, call, failure, and parse-error visualization. |
| `llm_calls.jsonl` | Every LLM request attempt and retry. |
| `traces.jsonl` | Workflow-supplied execution events. |
| `trace_summary.json` | Runtime, token, and failure summary. |

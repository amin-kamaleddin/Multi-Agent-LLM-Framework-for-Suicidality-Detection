# Compiled Evaluation Results

- Run directory: `logs\runs\20260717_123053_framework_1_parallel_no_evidence_bool_gpt4o_mini_positive_100_direct_json_final`
- Workflow: `framework_1_parallel_no_evidence`
- Model: `openai/gpt-4o-mini`
- Requested model: `gpt-4o-mini`
- Mode: `bool`
- Evaluation status: `completed`
- Stop reason: `None`

## Run Summary

| Metric | Value |
|---|---:|
| Processed records | 100 |
| Total records | 100 |
| Correct predictions | 31 |
| Failed records | 0 |
| Failure rate | 0.000000 |
| LLM requests | 500 |
| Failed LLM attempts | 0 |
| JSON/schema parsing errors | 0 |

## Model Metrics

| Model | Accuracy | F1 Score | Precision | Recall |
|---|---|---|---|---|
| openai/gpt-4o-mini | 0.310000 | 0.187047 | 0.453634 | 0.248000 |

## Per-Label Metrics

| Label | Accuracy | F1 Score | Precision | Recall | Support | TP | FP | TN | FN |
|---|---|---|---|---|---|---|---|---|---|
| Indicator | 0.770000 | 0.258065 | 0.666667 | 0.160000 | 25 | 4 | 2 | 73 | 21 |
| Ideation | 0.720000 | 0.125000 | 0.285714 | 0.080000 | 25 | 2 | 5 | 70 | 23 |
| Behavior | 0.760000 | 0.076923 | 1.000000 | 0.040000 | 25 | 1 | 0 | 75 | 24 |
| Attempt | 0.470000 | 0.475248 | 0.315789 | 0.960000 | 25 | 24 | 52 | 23 | 1 |
| Unsupportive | 0.900000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 10 | 90 | 0 |

## Per-Agent Metrics

| Agent | Model | Task | Accuracy | F1 Score | Precision | Recall |
|---|---|---|---|---|---|---|
| Indicator Agent | openai/gpt-4o-mini | Indicator one-vs-rest | 0.780000 | 0.590774 | 0.749616 | 0.586667 |
| Ideation Agent | openai/gpt-4o-mini | Ideation one-vs-rest | 0.440000 | 0.440000 | 0.586667 | 0.586667 |
| Behavior Agent | openai/gpt-4o-mini | Behavior one-vs-rest | 0.600000 | 0.579655 | 0.615415 | 0.653333 |
| Attempt Agent | openai/gpt-4o-mini | Attempt one-vs-rest | 0.460000 | 0.459784 | 0.634105 | 0.626667 |
| Final Judge | openai/gpt-4o-mini | final label | 0.310000 | 0.187047 | 0.453634 | 0.248000 |
| Final Judge | openai/gpt-4o-mini | suicidal_intent | 0.900000 | 0.473684 | 0.500000 | 0.450000 |

## Confusion Matrix

| Actual \ Predicted | Indicator | Ideation | Behavior | Attempt | Unsupportive | Error |
|---|---|---|---|---|---|---|
| Indicator | 4 | 5 | 0 | 9 | 7 | 0 |
| Ideation | 1 | 2 | 0 | 20 | 2 | 0 |
| Behavior | 1 | 0 | 1 | 23 | 0 | 0 |
| Attempt | 0 | 0 | 0 | 24 | 1 | 0 |
| Unsupportive | 0 | 0 | 0 | 0 | 0 | 0 |

## Suicidal-Intent Positive-Class Grouping Metrics

| Value | Accuracy | F1 Score | Precision | Recall | Support | TP | FP | TN | FN |
|---|---|---|---|---|---|---|---|---|---|
| false | 0.900000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 10 | 90 | 0 |
| true | 0.900000 | 0.947368 | 1.000000 | 0.900000 | 100 | 90 | 0 | 0 | 10 |

### Suicidal-Intent Positive-Class Grouping Confusion Matrix

| Actual \ Predicted | false | true | Error |
|---|---|---|---|
| false | 0 | 0 | 0 |
| true | 10 | 90 | 0 |

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
| Attempt | 76 |
| Behavior | 1 |
| Ideation | 7 |
| Indicator | 6 |
| Unsupportive | 10 |

## Runtime And Tokens

| Metric | Value |
|---|---:|
| Total latency seconds | 539.409089 |
| Average latency seconds per request | 1.078818 |
| Min latency seconds | 0.393159 |
| Max latency seconds | 61.171855 |
| Total input tokens | 165726 |
| Total output tokens | 3746 |
| Total tokens | 169472 |
| Total cached input tokens | 0 |
| Total cache-write tokens | 0 |
| Average input tokens per request | 331.452000 |
| Average output tokens per request | 7.492000 |
| Average total tokens per request | 338.944000 |
| Total cost USD | 0.027106 |
| Average cost USD per request | 0.000054 |
| Observed TTFT calls | 0 |
| Average TTFT seconds | n/a |
| Average provider generation seconds | n/a |
| Average provider latency seconds | n/a |
| Average moderation latency seconds | n/a |
| Average provider completion TPS | n/a |
| Average end-to-end completion TPS | 8.706129 |

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

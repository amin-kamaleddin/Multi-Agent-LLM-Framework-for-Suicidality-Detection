# Compiled Evaluation Results

- Run directory: `logs\runs\20260717_123616_framework1_bool_gpt4o_mini_negative_100_direct_json_final`
- Workflow: `framework1`
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
| Correct predictions | 47 |
| Failed records | 0 |
| Failure rate | 0.000000 |
| LLM requests | 500 |
| Failed LLM attempts | 0 |
| JSON/schema parsing errors | 0 |

## Model Metrics

| Model | Accuracy | F1 Score | Precision | Recall |
|---|---|---|---|---|
| openai/gpt-4o-mini | 0.470000 | 0.127891 | 0.200000 | 0.094000 |

## Per-Label Metrics

| Label | Accuracy | F1 Score | Precision | Recall | Support | TP | FP | TN | FN |
|---|---|---|---|---|---|---|---|---|---|
| Indicator | 0.580000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 42 | 58 | 0 |
| Ideation | 0.890000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 11 | 89 | 0 |
| Behavior | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 0 | 100 | 0 |
| Attempt | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 0 | 100 | 0 |
| Unsupportive | 0.470000 | 0.639456 | 1.000000 | 0.470000 | 100 | 47 | 0 | 0 | 53 |

## Per-Agent Metrics

| Agent | Model | Task | Accuracy | F1 Score | Precision | Recall |
|---|---|---|---|---|---|---|
| Indicator Agent | openai/gpt-4o-mini | Indicator one-vs-rest | 0.570000 | 0.363057 | 0.500000 | 0.285000 |
| Ideation Agent | openai/gpt-4o-mini | Ideation one-vs-rest | 0.890000 | 0.470899 | 0.500000 | 0.445000 |
| Behavior Agent | openai/gpt-4o-mini | Behavior one-vs-rest | 1.000000 | 0.500000 | 0.500000 | 0.500000 |
| Attempt Agent | openai/gpt-4o-mini | Attempt one-vs-rest | 1.000000 | 0.500000 | 0.500000 | 0.500000 |
| Final Judge | openai/gpt-4o-mini | final label | 0.470000 | 0.127891 | 0.200000 | 0.094000 |
| Final Judge | openai/gpt-4o-mini | suicidal_intent | 0.470000 | 0.319728 | 0.500000 | 0.235000 |

## Confusion Matrix

| Actual \ Predicted | Indicator | Ideation | Behavior | Attempt | Unsupportive | Error |
|---|---|---|---|---|---|---|
| Indicator | 0 | 0 | 0 | 0 | 0 | 0 |
| Ideation | 0 | 0 | 0 | 0 | 0 | 0 |
| Behavior | 0 | 0 | 0 | 0 | 0 | 0 |
| Attempt | 0 | 0 | 0 | 0 | 0 | 0 |
| Unsupportive | 42 | 11 | 0 | 0 | 47 | 0 |

## Suicidal-Intent Positive-Class Grouping Metrics

| Value | Accuracy | F1 Score | Precision | Recall | Support | TP | FP | TN | FN |
|---|---|---|---|---|---|---|---|---|---|
| false | 0.470000 | 0.639456 | 1.000000 | 0.470000 | 100 | 47 | 0 | 0 | 53 |
| true | 0.470000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 53 | 47 | 0 |

### Suicidal-Intent Positive-Class Grouping Confusion Matrix

| Actual \ Predicted | false | true | Error |
|---|---|---|---|
| false | 47 | 53 | 0 |
| true | 0 | 0 | 0 |

## Label Distributions

### Ground Truth

| Label | Count |
|---|---:|
| Unsupportive | 100 |

### Predictions

| Predicted Label | Count |
|---|---:|
| Ideation | 11 |
| Indicator | 42 |
| Unsupportive | 47 |

## Runtime And Tokens

| Metric | Value |
|---|---:|
| Total latency seconds | 445.590358 |
| Average latency seconds per request | 0.891181 |
| Min latency seconds | 0.359585 |
| Max latency seconds | 6.052451 |
| Total input tokens | 122027 |
| Total output tokens | 3960 |
| Total tokens | 125987 |
| Total cached input tokens | 0 |
| Total cache-write tokens | 0 |
| Average input tokens per request | 244.054000 |
| Average output tokens per request | 7.920000 |
| Average total tokens per request | 251.974000 |
| Total cost USD | 0.020680 |
| Average cost USD per request | 0.000041 |
| Observed TTFT calls | 0 |
| Average TTFT seconds | n/a |
| Average provider generation seconds | n/a |
| Average provider latency seconds | n/a |
| Average moderation latency seconds | n/a |
| Average provider completion TPS | n/a |
| Average end-to-end completion TPS | 9.659739 |

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

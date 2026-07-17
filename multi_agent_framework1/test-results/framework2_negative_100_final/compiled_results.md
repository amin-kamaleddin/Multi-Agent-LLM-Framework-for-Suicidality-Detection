# Compiled Evaluation Results

- Run directory: `logs\runs\20260717_121658_framework2_gpt4o_mini_negative_100_final`
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
| Correct predictions | 77 |
| Failed records | 0 |
| Failure rate | 0.000000 |
| LLM requests | 603 |
| Failed LLM attempts | 3 |
| JSON/schema parsing errors | 3 |

## Model Metrics

| Model | Accuracy | F1 Score | Precision | Recall |
|---|---|---|---|---|
| openai/gpt-4o-mini | 0.770000 | 0.174011 | 0.200000 | 0.154000 |

## Per-Label Metrics

| Label | Accuracy | F1 Score | Precision | Recall | Support | TP | FP | TN | FN |
|---|---|---|---|---|---|---|---|---|---|
| Indicator | 0.970000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 3 | 97 | 0 |
| Ideation | 0.800000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 20 | 80 | 0 |
| Behavior | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 0 | 100 | 0 |
| Attempt | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 0 | 100 | 0 |
| Unsupportive | 0.770000 | 0.870056 | 1.000000 | 0.770000 | 100 | 77 | 0 | 0 | 23 |

## Confusion Matrix

| Actual \ Predicted | Indicator | Ideation | Behavior | Attempt | Unsupportive | Error |
|---|---|---|---|---|---|---|
| Indicator | 0 | 0 | 0 | 0 | 0 | 0 |
| Ideation | 0 | 0 | 0 | 0 | 0 | 0 |
| Behavior | 0 | 0 | 0 | 0 | 0 | 0 |
| Attempt | 0 | 0 | 0 | 0 | 0 | 0 |
| Unsupportive | 3 | 20 | 0 | 0 | 77 | 0 |

## Suicidal-Intent Positive-Class Grouping Metrics

| Value | Accuracy | F1 Score | Precision | Recall | Support | TP | FP | TN | FN |
|---|---|---|---|---|---|---|---|---|---|
| false | 0.770000 | 0.870056 | 1.000000 | 0.770000 | 100 | 77 | 0 | 0 | 23 |
| true | 0.770000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 23 | 77 | 0 |

### Suicidal-Intent Positive-Class Grouping Confusion Matrix

| Actual \ Predicted | false | true | Error |
|---|---|---|---|
| false | 77 | 23 | 0 |
| true | 0 | 0 | 0 |

## Label Distributions

### Ground Truth

| Label | Count |
|---|---:|
| Unsupportive | 100 |

### Predictions

| Predicted Label | Count |
|---|---:|
| Ideation | 20 |
| Indicator | 3 |
| Unsupportive | 77 |

## Runtime And Tokens

| Metric | Value |
|---|---:|
| Total latency seconds | 705.001160 |
| Average latency seconds per request | 1.169156 |
| Min latency seconds | 0.410544 |
| Max latency seconds | 6.014463 |
| Total input tokens | 318049 |
| Total output tokens | 17642 |
| Total tokens | 335691 |
| Total cached input tokens | 0 |
| Total cache-write tokens | 0 |
| Average input tokens per request | 527.444444 |
| Average output tokens per request | 29.257048 |
| Average total tokens per request | 556.701493 |
| Total cost USD | 0.058293 |
| Average cost USD per request | 0.000097 |
| Observed TTFT calls | 0 |
| Average TTFT seconds | n/a |
| Average provider generation seconds | n/a |
| Average provider latency seconds | n/a |
| Average moderation latency seconds | n/a |
| Average provider completion TPS | n/a |
| Average end-to-end completion TPS | 18.306485 |

## LLM Errors And Retries

| Error Type | Count |
|---|---:|
| ValidationError | 3 |

### Recovered Error Attempts

| Sample ID | Role | Attempt | Error Type |
|---|---|---:|---|
| 30774 | judge | 1 | ValidationError |
| 30774 | judge | 2 | ValidationError |
| 30774 | judge | 3 | ValidationError |

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

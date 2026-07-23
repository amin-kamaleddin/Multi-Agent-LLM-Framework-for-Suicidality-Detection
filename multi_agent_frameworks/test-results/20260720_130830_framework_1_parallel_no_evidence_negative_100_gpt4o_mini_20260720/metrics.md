# Run Metrics

- Workflow: framework_1_parallel_no_evidence
- Accuracy: 0.460000
- Macro F1: 0.126027
- Macro Precision: 0.200000
- Macro Recall: 0.092000
- Total samples: 100
- Correct: 46
- Failed records: 0
- Failure rate: 0.000000
- LLM calls: 500
- JSON/schema parsing errors: 0
- Total latency seconds: 470.448670
- Average latency seconds per request: 0.940897
- Average input tokens per request: 244.054000
- Average output tokens per request: 7.898000
- Average cached tokens per request: 0.000000
- Total cost USD: 0.020673
- Average TTFT seconds when exposed: n/a
- Average provider completion TPS when exposed: n/a
- Average end-to-end completion TPS: 9.101242
- Suicidal-intent grouping accuracy: 0.460000
- Suicidal-intent grouping positive-class F1: 0.000000
- Suicidal-intent TP: 0
- Suicidal-intent TN: 46
- Suicidal-intent FP: 54
- Suicidal-intent FN: 0

## Per-Agent Metrics

| Agent | Model | Task | Accuracy | F1 | Precision | Recall |
|---|---|---|---:|---:|---:|---:|
| Indicator Agent | openai/gpt-4o-mini | Indicator one-vs-rest | 0.570000 | 0.000000 | 0.000000 | 0.000000 |
| Ideation Agent | openai/gpt-4o-mini | Ideation one-vs-rest | 0.890000 | 0.000000 | 0.000000 | 0.000000 |
| Behavior Agent | openai/gpt-4o-mini | Behavior one-vs-rest | 1.000000 | 0.000000 | 0.000000 | 0.000000 |
| Attempt Agent | openai/gpt-4o-mini | Attempt one-vs-rest | 1.000000 | 0.000000 | 0.000000 | 0.000000 |
| Final Judge | openai/gpt-4o-mini | final label | 0.460000 | 0.126027 | 0.200000 | 0.092000 |
| Final Judge | openai/gpt-4o-mini | suicidal_intent | 0.460000 | 0.000000 | 0.000000 | 0.000000 |

## Per-Class Metrics

| Label | Accuracy | F1 | Precision | Recall | Support |
|---|---:|---:|---:|---:|---:|
| Indicator | 0.570000 | 0.000000 | 0.000000 | 0.000000 | 0 |
| Ideation | 0.890000 | 0.000000 | 0.000000 | 0.000000 | 0 |
| Behavior | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 0 |
| Attempt | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 0 |
| Unsupportive | 0.460000 | 0.630137 | 1.000000 | 0.460000 | 100 |

## Suicidal-Intent Positive-Class Grouping Metrics

- Positive: Indicator, Ideation, Behavior, Attempt
- Negative: Unsupportive
- TP/TN/FP/FN use this fixed positive orientation; failed predictions remain in the Error column.

| Accuracy | F1 | Precision | Recall | Positive Support | Negative Support | TP | FP | TN | FN |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.460000 | 0.000000 | 0.000000 | 0.000000 | 0 | 100 | 0 | 54 | 46 | 0 |

> Limitation: Only one suicidal_intent ground-truth class is present; the confusion matrix is valid, but metrics for the missing ground-truth class cannot be estimated.

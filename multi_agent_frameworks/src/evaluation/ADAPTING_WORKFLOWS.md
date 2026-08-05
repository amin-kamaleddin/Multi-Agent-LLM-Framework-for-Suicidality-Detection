# Adapting Another Agentic Workflow

The evaluator shares a case boundary, not a graph architecture. A workflow may use one agent, several parallel specialists, a router, a hierarchy, tool calls, or a variable number of agents. To receive the same final-label, `suicidal_intent`, runtime, token, cost, failure, and guardrail artifacts, expose one method:

```python
run_case(*, sample_id: str, text: str) -> case_result
```

The case result must expose six fields:

```text
sample_id, final_label, suicidal_intent, errors, traces, llm_calls
```

No specialist names or fixed call count are required. `agent_outputs` is optional.

## 1. Use the shared input and final-output contracts

The preferred evaluation CSV is the canonical RSD15K shape:

```text
ID,users,text,sentiment,time[,real]
```

`ID` must be unique. `sentiment` must be one of `Indicator`, `Ideation`, `Behavior`, `Attempt`, or `Unsupportive`. All source columns are copied into the sampled and prediction artifacts. Legacy fixtures with `source_row,text,sentiment[,real]` remain accepted, but full runs should use the original RSD15K identity and metadata fields.

Every successful workflow must produce this final JSON contract:

```json
{
  "label_name": "Ideation",
  "suicidal_intent": true
}
```

The boolean is deterministic dataset grouping, not a second model judgment:

```text
Indicator | Ideation | Behavior | Attempt -> true
Unsupportive                              -> false
```

In particular, `Indicator = true` means the record belongs to the four-label positive grouping; it does not necessarily mean the author personally expresses suicidal intent. `JudgeAnswer` validates the label, strict JSON boolean, extra fields, and label/boolean consistency.

## 2. Add a thin adapter around the workflow

The adapter translates framework-specific state into the shared result. This complete example wraps a graph whose `invoke` method returns a final judge object plus telemetry:

```python
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from domain.final_output import JudgeAnswer


@dataclass
class SharedCaseResult:
    sample_id: str
    final_label: str
    suicidal_intent: bool | None
    errors: list[dict[str, Any]] = field(default_factory=list)
    traces: list[dict[str, Any]] = field(default_factory=list)
    llm_calls: list[dict[str, Any]] = field(default_factory=list)
    agent_outputs: dict[str, Any] | None = None


class MyWorkflowAdapter:
    def __init__(self, graph: Any) -> None:
        self.graph = graph

    def run_case(self, *, sample_id: str, text: str) -> SharedCaseResult:
        try:
            state = self.graph.invoke({"sample_id": sample_id, "text": text})
        except Exception as exc:
            return SharedCaseResult(
                sample_id=sample_id,
                final_label="Error",
                suicidal_intent=None,
                errors=[
                    {
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }
                ],
            )
        case_errors = list(state.get("case_errors", []))
        if case_errors:
            return SharedCaseResult(
                sample_id=sample_id,
                final_label="Error",
                suicidal_intent=None,
                errors=case_errors,
                traces=list(state.get("traces", [])),
                llm_calls=list(state.get("llm_calls", [])),
                agent_outputs=state.get("agent_outputs"),
            )

        try:
            final = JudgeAnswer.model_validate(state["final_output"])
        except (KeyError, ValidationError) as exc:
            return SharedCaseResult(
                sample_id=sample_id,
                final_label="Error",
                suicidal_intent=None,
                errors=[
                    {
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }
                ],
                traces=list(state.get("traces", [])),
                llm_calls=list(state.get("llm_calls", [])),
                agent_outputs=state.get("agent_outputs"),
            )
        return SharedCaseResult(
            sample_id=sample_id,
            final_label=final.label_name,
            suicidal_intent=final.suicidal_intent,
            traces=list(state.get("traces", [])),
            llm_calls=list(state.get("llm_calls", [])),
            agent_outputs=state.get("agent_outputs"),
        )
```

Only unrecovered case failures belong in `errors`. Failed attempts that later recover belong in `llm_calls[*].error`, so retries affect runtime/failure-attempt metrics without turning the completed record into an `Error` prediction. The adapter must return the same `sample_id` it receives; the evaluator rejects mismatches.

If a final output cannot be produced, return `final_label="Error"`, `suicidal_intent=None`, and at least one error object. Do not use `Error` as a successful judge label.

## 3. Record one telemetry item per actual attempt

The evaluator derives architecture-independent runtime metrics by aggregating `llm_calls`. Append one item for every real model request, including failed attempts and retries. Roles are arbitrary strings and may vary by record.

```python
{
    "role": "triage_router",
    "model_name": "provider/model-name",
    "attempt": 1,
    "latency_s": 0.84,
    "prompt_tokens": 121,
    "completion_tokens": 16,
    "total_tokens": 137,
    "cost": 0.00042,
    "raw_output": "{...}",
    "parsed_output": {"route": "high_risk"},
    "error": None,
}
```

On a failed attempt, preserve any provider usage that is available and set `error`, for example:

```python
{
    "error": {
        "error_type": "ValidationError",
        "error_message": "label_name must be one of the five canonical labels"
    }
}
```

The shared evaluator calculates call count, failed-attempt count, latency, prompt/completion/total tokens, cost, throughput, and parsing failures from these records. It also measures cumulative per-case time in `runtime.agent_execution_duration_s` and concurrency-aware first-call-to-last-call time in `runtime.real_execution_duration_s`, so adapters do not need to report workflow duration themselves. Optional provider timing and cache-token fields are documented in [the full contract](EVALUATION_CONTRACT.md#runtime-metrics-and-guardrails). Missing provider metadata remains zero or null; the evaluator does not estimate it.

`traces` is a list of workflow-defined execution events. For example, `{"event": "route_selected", "route": "high_risk"}` is written unchanged with the sample ID to `traces.jsonl`.

## 4. Call the generic evaluator

For repeatable CLI use, register the adapter in `workflows/registry.py`, then select its exact registered name:

```powershell
uv run python -m cli list-workflows
uv run python -m cli evaluate --workflow framework_1_parallel_no_evidence --sample-path dataset/my_combined_sample.csv --artifact-base-dir logs/experiments
```

`--workflow` accepts a predetermined lowercase registry key. It is not a file path or class name. See the [shared CLI usage guide](../../../../docs/USAGE.md#shared-agent-workflow-evaluator-cli) for configuration files and registration instructions.

For direct API use, create a small runner in your framework, for example `scripts/run_my_framework.py`:

```python
from evaluation import (
    EvaluationConfig,
    load_evaluation_sample,
    run_evaluation,
)

from my_framework import build_graph
from my_framework_adapter import MyWorkflowAdapter


sample_path = "dataset/my_combined_sample.csv"
sample = load_evaluation_sample(sample_path)
workflow = MyWorkflowAdapter(build_graph())

config = EvaluationConfig(
    model_name="mixed-workflow",
    requested_model_name="mixed-workflow",
    sample_path=sample_path,
    dataset_audit_path=None,
    max_retries=2,
    max_completion_tokens=1024,
    temperature=0.0,
    total_records=len(sample),
    agent_configuration={
        "triage_router": {"model_name": "model-a", "temperature": 0.0},
        "risk_reviewer": {"model_name": "model-b", "temperature": 0.0},
        "final_decider": {"model_name": "model-c", "temperature": 0.0},
    },
)

result = run_evaluation(
    sample_df=sample,
    workflow=workflow,
    config=config,
    run_id="my-framework-v1",
)
print(result.run_dir)
```

Run it from the package directory:

```powershell
uv run python scripts/run_my_framework.py
```

`agent_configuration` is recorded metadata. It may contain any role names; the generic evaluator does not construct or route models. Your framework remains responsible for creating its clients with the recorded model and temperature settings. Likewise, `EvaluationConfig.max_retries` records the intended policy but `run_evaluation` does not execute retries for a generic workflow. Implement that policy inside the workflow/adapter and emit one `llm_calls` item for every attempt.

The evaluator now emits the same mandatory artifacts regardless of the internal workflow: predictions and nested results, a five-label confusion matrix, `suicidal_intent` binary metrics, misclassified IDs, runtime/token/cost/failure summaries, traces, reports, and visualizations. It does not calculate aggregate or per-label scores for the five-label task.

## 5. Preserve optional framework-specific outputs without evaluating them

The generic path does not invent specialist columns or internal-agent metrics. Only the final judge output is evaluated.

If `SharedCaseResult.agent_outputs` is populated, it is serialized into `predictions.csv` and `results.json` but is not interpreted. This works for dynamically named agents:

```python
agent_outputs={
    "triage_router": {"route": "high_risk"},
    "reviewer_7": {"confidence": 0.91},
}
```

For additional flat prediction columns, pass a `prediction_enricher(record, result) -> dict`. It cannot overwrite shared fields.

Framework 1 demonstrates the prediction-enrichment extension point: its `workflows/framework_1_parallel_no_evidence/evaluation.py` adapter builds the fixed five-specialist graph, calls the shared `run_evaluation`, and preserves real specialist answers without calculating specialist metrics. Its CLI flow is:

```text
python -m cli evaluate
  -> framework_1_parallel_no_evidence.evaluation.run_evaluation
  -> run_evaluation
  -> MultiAgentWorkflow.run_case (once per record)
```

Other frameworks can call `run_evaluation` directly or register a runner that calls it; they do not need to imitate the five-specialist graph.

## Integration checklist

- The CSV uses the canonical schema and unique `ID` values.
- `run_case` accepts keyword-only `sample_id` and `text`.
- The returned `sample_id` exactly matches the input.
- Success uses one of five labels and the matching strict boolean.
- Failure uses `Error`, null `suicidal_intent`, and an unrecovered error record.
- Every actual request/retry contributes one `llm_calls` item.
- Agent roles and call counts reflect the real workflow and may vary.
- Intermediate agent outputs are stored but never converted into evaluation metrics.
- A smoke run produces `metrics.json`, `predictions.csv`, `llm_calls.jsonl`, `compiled_results.md`, and the remaining required artifacts.

See [Evaluation Contracts, Metrics, and Guardrails](EVALUATION_CONTRACT.md) for the precise schemas, calculations, stop thresholds, and complete artifact inventory.

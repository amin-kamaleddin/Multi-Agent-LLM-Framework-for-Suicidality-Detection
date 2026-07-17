"""Shared evaluation runner plus framework1's specialist compatibility layer."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Callable, Protocol

import pandas as pd

from ..core.labels import FINAL_LABELS, VALID_LABELS, label_to_suicidal_intent
from ..data.dataset import DEFAULT_PROCESSED_DIR
from ..workflow.graph import InvokableModel, MultiAgentWorkflow
from .artifacts import ArtifactWriter
from .metrics import compute_binary_classification_metrics, compute_classification_metrics, summarize_runtime
from .reporting import compile_run_results


DEFAULT_EVALUATION_SAMPLE_PATH = DEFAULT_PROCESSED_DIR / "dataset1_balanced_500_seed42.csv"
DEFAULT_EVALUATION_AUDIT_PATH = DEFAULT_PROCESSED_DIR / "dataset1_audit.json"
SPECIALIST_ANSWER_COLUMNS = {
    label: f"{label.lower()}_answer"
    for label in VALID_LABELS
}
FRAMEWORK1_AGENT_ROLES = tuple(f"{label}_specialist" for label in VALID_LABELS) + ("judge",)


class EvaluationCaseResult(Protocol):
    """Architecture-neutral result required from one evaluated workflow case."""

    sample_id: str
    final_label: str
    suicidal_intent: bool | None
    errors: list[dict[str, Any]]
    traces: list[dict[str, Any]]
    llm_calls: list[dict[str, Any]]


class EvaluationWorkflow(Protocol):
    """Minimal workflow interface consumed by the shared evaluator."""

    def run_case(self, *, sample_id: str, text: str) -> EvaluationCaseResult:
        """Evaluate one sample and return the shared result fields."""


PredictionEnricher = Callable[[dict[str, Any], EvaluationCaseResult], dict[str, Any]]
AgentMetricsProvider = Callable[
    [list[dict[str, Any]], "EvaluationConfig", dict[str, Any], dict[str, Any]],
    dict[str, Any],
]


@dataclass(frozen=True)
class EvaluationConfig:
    """Configuration persisted with every evaluator run."""

    model_name: str
    requested_model_name: str
    sample_path: str
    dataset_audit_path: str | None
    max_retries: int
    max_completion_tokens: int
    temperature: float
    total_records: int
    limit: int | None = None
    mode: str = "bool"
    workflow_name: str = "framework1"
    workflow_config: dict[str, Any] | None = None
    agent_configuration: dict[str, dict[str, Any]] | None = None
    error_stop_failed_llm_attempts: int = 10
    error_stop_failed_records: int = 3
    error_stop_rate: float = 0.5
    error_stop_min_records: int = 10
    command: str = "evaluate"


@dataclass(frozen=True)
class EvaluationResult:
    """Summary returned to the CLI after artifacts are written."""

    run_dir: str
    status: str
    processed_records: int
    total_records: int
    stop_reason: str | None
    metrics: dict[str, Any]


def load_evaluation_sample(sample_path: str | Path, *, limit: int | None = None) -> pd.DataFrame:
    """Load and validate the frozen sampled dataset used for evaluation."""
    path = Path(sample_path)
    df = pd.read_csv(path, encoding="utf-8-sig")
    canonical_columns = ["ID", "users", "text", "sentiment", "time"]
    legacy_columns = ["source_row", "text", "sentiment"]
    allowed_columns = [
        canonical_columns,
        [*canonical_columns, "real"],
        legacy_columns,
        [*legacy_columns, "real"],
    ]
    if list(df.columns) not in allowed_columns:
        raise ValueError(
            f"Expected canonical columns {canonical_columns} or legacy columns {legacy_columns}, "
            "each with optional trailing 'real'; "
            f"got {list(df.columns)} from {path}"
        )

    invalid_labels = sorted(set(df["sentiment"]) - set(FINAL_LABELS))
    if invalid_labels:
        raise ValueError(f"Invalid sentiment labels in {path}: {invalid_labels}")

    id_column = _sample_id_column(df.columns)
    if df[id_column].duplicated().any():
        raise ValueError(f"{id_column} must be unique in {path}")

    if "real" in df.columns and not df["real"].isin([True, False]).all():
        raise ValueError(f"real must contain only boolean values in {path}")

    if limit is not None:
        df = df.head(limit)
    return df.reset_index(drop=True)


def run_evaluation(
    *,
    sample_df: pd.DataFrame,
    workflow: EvaluationWorkflow,
    config: EvaluationConfig,
    run_id: str | None = None,
    artifact_base_dir: str | Path = "logs/runs",
    dataset_audit: dict[str, Any] | None = None,
    prediction_enricher: PredictionEnricher | None = None,
    agent_metrics_provider: AgentMetricsProvider | None = None,
) -> EvaluationResult:
    """Evaluate any workflow that implements the shared per-case result contract."""
    if sample_df.empty:
        raise ValueError("sample_df must contain at least one evaluation record")
    writer = ArtifactWriter(base_dir=artifact_base_dir, run_id=run_id)

    run_config = asdict(config)
    run_config["agent_configuration"] = config.agent_configuration or {}
    writer.write_json("run_config.json", run_config)
    writer.write_json(
        "dataset_audit.json",
        (
            dataset_audit
            if dataset_audit is not None
            else _load_dataset_audit(config.dataset_audit_path, config.sample_path)
        ),
    )
    writer.write_dataframe("sampled_dataset.csv", sample_df)

    rows: list[dict[str, Any]] = []
    all_calls: list[dict[str, Any]] = []
    stop_reason: str | None = None

    id_column = _sample_id_column(sample_df.columns)
    for position, record in enumerate(sample_df.to_dict(orient="records"), start=1):
        expected_sample_id = str(record[id_column])
        result = workflow.run_case(sample_id=expected_sample_id, text=record["text"])
        if str(result.sample_id) != expected_sample_id:
            raise ValueError(
                "Workflow result sample_id does not match the evaluated record: "
                f"expected {expected_sample_id!r}, got {result.sample_id!r}"
            )

        for call in result.llm_calls:
            call_record = {"sample_id": result.sample_id, **call}
            all_calls.append(call_record)
            writer.append_jsonl("llm_calls.jsonl", call_record)
        for trace in result.traces:
            writer.append_jsonl("traces.jsonl", {"sample_id": result.sample_id, **trace})
        for error in result.errors:
            writer.append_jsonl("errors.jsonl", {"sample_id": result.sample_id, **error})

        rows.append(
            build_prediction_row(record=record, result=result, prediction_enricher=prediction_enricher)
        )

        stop_reason = _stop_reason(rows, all_calls, config)
        status = "stopped_early" if stop_reason else "running"
        metrics = _write_evaluation_artifacts(
            writer=writer,
            rows=rows,
            all_calls=all_calls,
            config=config,
            status=status,
            stop_reason=stop_reason,
            agent_metrics_provider=agent_metrics_provider,
        )
        print(
            json.dumps(
                {
                    "event": "record_complete",
                    "processed_records": position,
                    "total_records": len(sample_df),
                    "prediction": result.final_label,
                    "errors_this_record": len(result.errors),
                    "failed_llm_attempts_total": metrics["runtime"]["failed_llm_attempts"],
                    "stop_reason": stop_reason,
                },
                ensure_ascii=False,
            )
        )
        if stop_reason:
            break

    final_status = "stopped_early" if stop_reason else "completed"
    final_metrics = finalize_evaluation_artifacts(
        writer=writer,
        rows=rows,
        all_calls=all_calls,
        config=config,
        status=final_status,
        stop_reason=stop_reason,
        agent_metrics_provider=agent_metrics_provider,
    )

    return EvaluationResult(
        run_dir=str(writer.run_dir),
        status=final_status,
        processed_records=len(rows),
        total_records=len(sample_df),
        stop_reason=stop_reason,
        metrics=final_metrics,
    )


def run_workflow_evaluation(
    *,
    sample_df: pd.DataFrame,
    model: InvokableModel,
    config: EvaluationConfig,
    run_id: str | None = None,
    artifact_base_dir: str | Path = "logs/runs",
    agent_models: dict[str, InvokableModel] | None = None,
    dataset_audit: dict[str, Any] | None = None,
) -> EvaluationResult:
    """Compatibility wrapper for the four-specialist framework1 workflow."""
    agent_configuration = _framework1_agent_configuration(config)
    _validate_framework1_agent_overrides(config, agent_models, agent_configuration)
    model_names_by_role = {
        role: str(details.get("model_name") or config.model_name)
        for role, details in agent_configuration.items()
    }
    workflow = MultiAgentWorkflow(
        model=model,
        model_name=config.model_name,
        max_retries=config.max_retries,
        mode=config.mode,
        models_by_role=agent_models,
        model_names_by_role=model_names_by_role,
    )
    resolved_config = EvaluationConfig(
        **{
            **asdict(config),
            "agent_configuration": agent_configuration,
        }
    )
    return run_evaluation(
        sample_df=sample_df,
        workflow=workflow,
        config=resolved_config,
        run_id=run_id,
        artifact_base_dir=artifact_base_dir,
        dataset_audit=dataset_audit,
        prediction_enricher=_specialist_prediction_fields,
        agent_metrics_provider=_specialist_agent_metric_payload,
    )


def finalize_evaluation_artifacts(
    *,
    writer: ArtifactWriter,
    rows: list[dict[str, Any]],
    all_calls: list[dict[str, Any]],
    config: EvaluationConfig,
    status: str,
    stop_reason: str | None,
    agent_metrics_provider: AgentMetricsProvider | None = None,
) -> dict[str, Any]:
    """Write the final evaluator artifacts and compile the review reports."""
    metrics = _write_evaluation_artifacts(
        writer=writer,
        rows=rows,
        all_calls=all_calls,
        config=config,
        status=status,
        stop_reason=stop_reason,
        agent_metrics_provider=agent_metrics_provider,
    )
    metrics["compiled_reports"] = compile_run_results(writer.run_dir)
    metrics["artifact_status"] = writer.required_artifact_status()
    metrics["optional_artifact_status"] = writer.optional_artifact_status()
    writer.write_json("metrics.json", metrics)
    writer.write_json("results.json", _complete_results_payload(rows, metrics))
    return metrics


def build_prediction_row(
    *,
    record: dict[str, Any],
    result: EvaluationCaseResult,
    prediction_enricher: PredictionEnricher | None = None,
) -> dict[str, Any]:
    """Return mandatory result fields plus optional workflow-specific fields."""
    id_column = _sample_id_column(record)
    ground_truth_suicidal_intent = label_to_suicidal_intent(record["sentiment"])
    row = {
        **record,
        "sample_id": record[id_column],
        "ground_truth": record["sentiment"],
        "ground_truth_suicidal_intent": ground_truth_suicidal_intent,
        "prediction": result.final_label,
        "suicidal_intent": result.suicidal_intent,
        "correct": int(record["sentiment"] == result.final_label),
        "suicidal_intent_correct": int(ground_truth_suicidal_intent == result.suicidal_intent),
        "error": "; ".join(err.get("error_message", "") for err in result.errors),
    }
    agent_outputs = getattr(result, "agent_outputs", None)
    if agent_outputs is not None:
        row["agent_outputs"] = json.dumps(agent_outputs, ensure_ascii=False)
    if prediction_enricher is not None:
        enrichment = prediction_enricher(record, result)
        conflicting_fields = sorted(set(row) & set(enrichment))
        if conflicting_fields:
            raise ValueError(f"Prediction enrichment cannot overwrite shared fields: {conflicting_fields}")
        row.update(enrichment)
    return row


def _specialist_prediction_fields(
    _record: dict[str, Any],
    result: EvaluationCaseResult,
) -> dict[str, Any]:
    """Return framework1-only specialist columns for its compatibility wrapper."""
    specialist_answers = {
        label: getattr(result, "specialist_answers", {}).get(label)
        for label in VALID_LABELS
    }
    specialist_outputs = {}
    for label in VALID_LABELS:
        output = getattr(result, "specialist_outputs", {}).get(label)
        if output is None and specialist_answers[label] is not None:
            output = {"answer": specialist_answers[label]}
        specialist_outputs[label] = output

    row = {
        "specialist_answers": json.dumps(specialist_answers, ensure_ascii=False),
        "specialist_outputs": json.dumps(specialist_outputs, ensure_ascii=False),
    }
    for label, column in SPECIALIST_ANSWER_COLUMNS.items():
        row[column] = specialist_answers[label]
        output = specialist_outputs[label] if isinstance(specialist_outputs[label], dict) else {}
        row[f"{label.lower()}_reasoning"] = output.get("reasoning")
    return row


def _sample_id_column(columns: Any) -> str:
    """Prefer canonical RSD15K IDs while accepting legacy source-row fixtures."""
    if "ID" in columns:
        return "ID"
    if "source_row" in columns:
        return "source_row"
    raise ValueError("Evaluation data must include either canonical 'ID' or legacy 'source_row'.")


def _load_dataset_audit(dataset_audit_path: str | None, sample_path: str) -> dict[str, Any]:
    if dataset_audit_path:
        path = Path(dataset_audit_path)
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                audit = json.load(f)
            audit["evaluation_sample_path"] = sample_path
            return audit
    return {"evaluation_sample_path": sample_path, "audit_available": False}


def _write_evaluation_artifacts(
    *,
    writer: ArtifactWriter,
    rows: list[dict[str, Any]],
    all_calls: list[dict[str, Any]],
    config: EvaluationConfig,
    status: str,
    stop_reason: str | None,
    agent_metrics_provider: AgentMetricsProvider | None = None,
) -> dict[str, Any]:
    predictions = pd.DataFrame(rows)
    writer.write_dataframe("predictions.csv", predictions)
    judge_model_name = _configured_model_name(config, "judge")
    classification = compute_classification_metrics(
        predictions["ground_truth"].tolist() if not predictions.empty else [],
        predictions["prediction"].tolist() if not predictions.empty else [],
        model_name=judge_model_name,
    )
    suicidal_intent = compute_binary_classification_metrics(
        [row["ground_truth_suicidal_intent"] for row in rows],
        [row["suicidal_intent"] for row in rows],
        model_name=judge_model_name,
    )
    agent_metric_payload = (
        agent_metrics_provider(rows, config, classification, suicidal_intent)
        if agent_metrics_provider is not None
        else {"agent_metrics": []}
    )
    agent_metrics = list(agent_metric_payload.get("agent_metrics") or [])
    runtime = summarize_runtime(all_calls)
    metrics = {
        "workflow_name": config.workflow_name,
        "model_name": config.model_name,
        "requested_model_name": config.requested_model_name,
        "mode": config.mode,
        "evaluation_status": status,
        "stop_reason": stop_reason,
        "processed_records": len(rows),
        "total_records": config.total_records,
        "classification": classification,
        "suicidal_intent": suicidal_intent,
        "agent_metrics": agent_metrics,
        "runtime": runtime,
    }
    for key, value in agent_metric_payload.items():
        if key == "agent_metrics":
            continue
        if key in metrics:
            raise ValueError(f"Agent metric provider cannot overwrite shared metric field {key!r}")
        metrics[key] = value
    writer.write_dataframe("model_metrics.csv", pd.DataFrame([classification["model_summary"]]))
    writer.write_dataframe("label_metrics.csv", pd.DataFrame(classification["per_label"]))
    if agent_metrics:
        writer.write_dataframe("agent_metrics.csv", pd.DataFrame(agent_metrics))
    writer.write_dataframe("suicidal_intent_metrics.csv", pd.DataFrame(suicidal_intent["per_class"]))
    writer.write_json("metrics.json", metrics)
    writer.write_json("results.json", _complete_results_payload(rows, metrics))
    writer.write_text(
        "misclassified_ids.txt",
        "".join(f'{row["sample_id"]}\n' for row in rows if not row["correct"]),
    )
    writer.write_text(
        "suicidal_intent_misclassified_ids.txt",
        "".join(f'{row["sample_id"]}\n' for row in rows if not row["suicidal_intent_correct"]),
    )
    writer.write_metrics_md(metrics)
    writer.write_visualizations(metrics)
    writer.write_json(
        "trace_summary.json",
        {
            "workflow_name": config.workflow_name,
            "model_name": config.model_name,
            "requested_model_name": config.requested_model_name,
            "mode": config.mode,
            "evaluation_status": status,
            "processed_records": len(rows),
            "total_records": config.total_records,
            "stop_reason": stop_reason,
            "runtime": runtime,
            "json_parsing_errors": runtime["json_parsing_errors"],
            "failed_records": classification["failed_records"],
            "failure_rate": classification["failure_rate"],
            "llm_trace_file": "llm_calls.jsonl",
            "node_trace_file": "traces.jsonl",
        },
    )
    return metrics


def _compute_specialist_metrics(
    rows: list[dict[str, Any]],
    config: EvaluationConfig,
) -> dict[str, dict[str, Any]]:
    return {
        label: compute_binary_classification_metrics(
            [row["ground_truth"] == label for row in rows],
            [row[SPECIALIST_ANSWER_COLUMNS[label]] for row in rows],
            model_name=_configured_model_name(config, f"{label}_specialist"),
        )
        for label in VALID_LABELS
    }


def _specialist_agent_metric_payload(
    rows: list[dict[str, Any]],
    config: EvaluationConfig,
    classification: dict[str, Any],
    suicidal_intent: dict[str, Any],
) -> dict[str, Any]:
    """Compute the optional framework1 one-vs-rest agent metrics."""
    specialist_metrics = _compute_specialist_metrics(rows, config)
    return {
        "specialist_metrics": specialist_metrics,
        "agent_metrics": _agent_metric_rows(classification, suicidal_intent, specialist_metrics),
    }


def _agent_metric_rows(
    classification: dict[str, Any],
    suicidal_intent: dict[str, Any],
    specialist_metrics: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for label in VALID_LABELS:
        metric = specialist_metrics[label]
        rows.append(
            {
                "agent_name": f"{label} Agent",
                "model_name": metric["model_name"],
                "task": f"{label} one-vs-rest",
                "accuracy": metric["accuracy"],
                "f1_score": metric["f1_score"],
                "precision": metric["precision"],
                "recall": metric["recall"],
            }
        )
    rows.extend(
        [
            {
                "agent_name": "Final Judge",
                "model_name": classification["model_name"],
                "task": "final label",
                "accuracy": classification["accuracy"],
                "f1_score": classification["f1_score"],
                "precision": classification["precision"],
                "recall": classification["recall"],
            },
            {
                "agent_name": "Final Judge",
                "model_name": suicidal_intent["model_name"],
                "task": "suicidal_intent",
                "accuracy": suicidal_intent["accuracy"],
                "f1_score": suicidal_intent["f1_score"],
                "precision": suicidal_intent["precision"],
                "recall": suicidal_intent["recall"],
            },
        ]
    )
    return rows


def _complete_results_payload(rows: list[dict[str, Any]], metrics: dict[str, Any]) -> dict[str, Any]:
    results = []
    for row in rows:
        result = dict(row)
        for field in ("agent_outputs", "specialist_answers", "specialist_outputs"):
            if isinstance(result.get(field), str):
                result[field] = json.loads(result[field])
        result["ground_truth_output"] = {
            "label_name": result["ground_truth"],
            "suicidal_intent": result["ground_truth_suicidal_intent"],
        }
        result["final_output"] = {
            "label_name": result["prediction"],
            "suicidal_intent": result["suicidal_intent"],
        }
        results.append(result)
    return {"schema_version": "1.1", "results": results, "metrics": metrics}


def _framework1_agent_configuration(config: EvaluationConfig) -> dict[str, dict[str, Any]]:
    configuration = {
        f"{label}_specialist": {
            "agent_name": f"{label} Agent",
            "model_name": config.model_name,
            "temperature": config.temperature,
        }
        for label in VALID_LABELS
    }
    configuration["judge"] = {
        "agent_name": "Final Judge",
        "model_name": config.model_name,
        "temperature": config.temperature,
    }
    for role, details in (config.agent_configuration or {}).items():
        configuration[role] = {**configuration.get(role, {}), **details}
    return configuration


def _configured_model_name(config: EvaluationConfig, role: str) -> str:
    details = (config.agent_configuration or {}).get(role, {})
    return str(details.get("model_name") or config.model_name)


def _validate_framework1_agent_overrides(
    config: EvaluationConfig,
    agent_models: dict[str, InvokableModel] | None,
    agent_configuration: dict[str, dict[str, Any]],
) -> None:
    supplied_roles = set(agent_models or {})
    configured_roles = set(agent_configuration)
    unknown_roles = sorted((supplied_roles | configured_roles) - set(FRAMEWORK1_AGENT_ROLES))
    if unknown_roles:
        raise ValueError(
            f"Unknown framework1 role overrides: {unknown_roles}. "
            f"Expected roles: {list(FRAMEWORK1_AGENT_ROLES)}"
        )
    override_roles = {
        role
        for role, details in agent_configuration.items()
        if details.get("model_name", config.model_name) != config.model_name
        or float(details.get("temperature", config.temperature)) != config.temperature
    }
    missing_models = sorted(override_roles - supplied_roles)
    if missing_models:
        raise ValueError(
            "Framework1 per-agent model/temperature overrides require matching model objects for roles: "
            f"{missing_models}"
        )


def _stop_reason(
    rows: list[dict[str, Any]],
    all_calls: list[dict[str, Any]],
    config: EvaluationConfig,
) -> str | None:
    processed_records = len(rows)
    failed_records = sum(1 for row in rows if row["prediction"] not in FINAL_LABELS)
    failed_llm_attempts = sum(1 for call in all_calls if call.get("error"))

    if failed_llm_attempts >= config.error_stop_failed_llm_attempts:
        return (
            f"failed_llm_attempts={failed_llm_attempts} reached "
            f"threshold={config.error_stop_failed_llm_attempts}"
        )
    if failed_records >= config.error_stop_failed_records:
        return f"failed_records={failed_records} reached threshold={config.error_stop_failed_records}"
    if processed_records >= config.error_stop_min_records:
        failure_rate = failed_records / processed_records if processed_records else 0.0
        if failure_rate >= config.error_stop_rate:
            return (
                f"failure_rate={failure_rate:.6f} reached threshold={config.error_stop_rate:.6f} "
                f"after processed_records={processed_records}"
            )
    return None

"""Evaluation runner for the full four-specialist-plus-judge workflow."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import pandas as pd

from ..core.labels import VALID_LABELS
from ..data.dataset import DEFAULT_PROCESSED_DIR
from ..workflow.graph import InvokableModel, MultiAgentWorkflow
from .artifacts import ArtifactWriter
from .metrics import compute_classification_metrics, summarize_runtime


DEFAULT_EVALUATION_SAMPLE_PATH = DEFAULT_PROCESSED_DIR / "dataset1_balanced_500_seed42.csv"
DEFAULT_EVALUATION_AUDIT_PATH = DEFAULT_PROCESSED_DIR / "dataset1_audit.json"
SPECIALIST_ANSWER_COLUMNS = {
    label: f"{label.lower()}_answer"
    for label in VALID_LABELS
}


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
    expected_columns = ["source_row", "text", "sentiment"]
    if list(df.columns) != expected_columns:
        raise ValueError(f"Expected columns {expected_columns}, got {list(df.columns)} from {path}")

    invalid_labels = sorted(set(df["sentiment"]) - set(VALID_LABELS))
    if invalid_labels:
        raise ValueError(f"Invalid sentiment labels in {path}: {invalid_labels}")

    if df["source_row"].duplicated().any():
        raise ValueError(f"source_row must be unique in {path}")

    if limit is not None:
        df = df.head(limit)
    return df.reset_index(drop=True)


def run_workflow_evaluation(
    *,
    sample_df: pd.DataFrame,
    model: InvokableModel,
    config: EvaluationConfig,
    run_id: str | None = None,
    artifact_base_dir: str | Path = "logs/runs",
) -> EvaluationResult:
    """Run the workflow over a prepared sample and continuously write artifacts."""
    writer = ArtifactWriter(base_dir=artifact_base_dir, run_id=run_id)
    workflow = MultiAgentWorkflow(
        model=model,
        model_name=config.model_name,
        max_retries=config.max_retries,
    )

    writer.write_json("run_config.json", asdict(config))
    writer.write_json("dataset_audit.json", _load_dataset_audit(config.dataset_audit_path, config.sample_path))
    writer.write_dataframe("sampled_dataset.csv", sample_df)

    rows: list[dict[str, Any]] = []
    all_calls: list[dict[str, Any]] = []
    stop_reason: str | None = None

    for position, record in enumerate(sample_df.to_dict(orient="records"), start=1):
        result = workflow.run_case(sample_id=str(record["source_row"]), text=record["text"])

        for call in result.llm_calls:
            call_record = {"sample_id": result.sample_id, **call}
            all_calls.append(call_record)
            writer.append_jsonl("llm_calls.jsonl", call_record)
        for trace in result.traces:
            writer.append_jsonl("traces.jsonl", {"sample_id": result.sample_id, **trace})
        for error in result.errors:
            writer.append_jsonl("errors.jsonl", {"sample_id": result.sample_id, **error})

        rows.append(
            build_prediction_row(record=record, result=result)
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
    final_metrics = _write_evaluation_artifacts(
        writer=writer,
        rows=rows,
        all_calls=all_calls,
        config=config,
        status=final_status,
        stop_reason=stop_reason,
    )
    final_metrics["artifact_status"] = writer.required_artifact_status()
    writer.write_json("metrics.json", final_metrics)

    return EvaluationResult(
        run_dir=str(writer.run_dir),
        status=final_status,
        processed_records=len(rows),
        total_records=len(sample_df),
        stop_reason=stop_reason,
        metrics=final_metrics,
    )


def build_prediction_row(*, record: dict[str, Any], result: Any) -> dict[str, Any]:
    """Return the stable per-sample result row written to predictions.csv."""
    specialist_answers = {
        label: result.specialist_answers.get(label)
        for label in VALID_LABELS
    }
    row = {
        "source_row": record["source_row"],
        "ground_truth": record["sentiment"],
        "prediction": result.final_label,
        "correct": int(record["sentiment"] == result.final_label),
        "error": "; ".join(err.get("error_message", "") for err in result.errors),
        "specialist_answers": json.dumps(specialist_answers, ensure_ascii=False),
    }
    for label, column in SPECIALIST_ANSWER_COLUMNS.items():
        row[column] = specialist_answers[label]
    row["text"] = record["text"]
    return row


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
) -> dict[str, Any]:
    predictions = pd.DataFrame(rows)
    writer.write_dataframe("predictions.csv", predictions)
    classification = compute_classification_metrics(
        predictions["ground_truth"].tolist() if not predictions.empty else [],
        predictions["prediction"].tolist() if not predictions.empty else [],
        model_name=config.model_name,
    )
    runtime = summarize_runtime(all_calls)
    metrics = {
        "model_name": config.model_name,
        "requested_model_name": config.requested_model_name,
        "evaluation_status": status,
        "stop_reason": stop_reason,
        "processed_records": len(rows),
        "total_records": config.total_records,
        "classification": classification,
        "runtime": runtime,
    }
    writer.write_dataframe("model_metrics.csv", pd.DataFrame([classification["model_summary"]]))
    writer.write_dataframe("label_metrics.csv", pd.DataFrame(classification["per_label"]))
    writer.write_json("metrics.json", metrics)
    writer.write_metrics_md(metrics)
    writer.write_json(
        "trace_summary.json",
        {
            "model_name": config.model_name,
            "requested_model_name": config.requested_model_name,
            "evaluation_status": status,
            "processed_records": len(rows),
            "total_records": config.total_records,
            "stop_reason": stop_reason,
            "runtime": runtime,
            "failed_records": classification["failed_records"],
            "failure_rate": classification["failure_rate"],
            "llm_trace_file": "llm_calls.jsonl",
            "node_trace_file": "traces.jsonl",
        },
    )
    return metrics


def _stop_reason(
    rows: list[dict[str, Any]],
    all_calls: list[dict[str, Any]],
    config: EvaluationConfig,
) -> str | None:
    processed_records = len(rows)
    failed_records = sum(1 for row in rows if row["prediction"] not in VALID_LABELS)
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

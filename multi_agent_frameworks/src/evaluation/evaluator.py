"""Architecture-neutral dataset evaluation and artifact generation."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path
import time
from typing import Any, Callable

import pandas as pd

from data.dataset import DEFAULT_PROCESSED_DIR
from domain.labels import FINAL_LABELS, label_to_suicidal_intent
from llm.protocol import InvokableModel
from workflows.contracts import CaseResult as EvaluationCaseResult
from workflows.contracts import Workflow as EvaluationWorkflow
from .artifacts import ArtifactWriter
from .metrics import (
    compute_final_label_confusion_matrix,
    compute_positive_subclassification_metrics,
    compute_suicidal_intent_metrics,
    summarize_runtime,
)
from .reporting import compile_run_results
from .progress import (
    COMPLETED,
    ProgressStore,
    dataframe_sha256,
    json_sha256,
    load_manifest,
    utc_now,
    write_manifest,
)


DEFAULT_EVALUATION_SAMPLE_PATH = DEFAULT_PROCESSED_DIR / "dataset1_balanced_500_seed42.csv"
DEFAULT_EVALUATION_AUDIT_PATH = DEFAULT_PROCESSED_DIR / "dataset1_audit.json"
SPECIALIST_ANSWER_COLUMNS = {
    label: f"{label.lower()}_answer"
    for label in FINAL_LABELS
}
FRAMEWORK_1_PARALLEL_NO_EVIDENCE_AGENT_ROLES = tuple(f"{label}_specialist" for label in FINAL_LABELS) + ("judge",)
FRAMEWORK_2_PARALLEL_WITH_EVIDENCE_AGENT_ROLES = tuple(f"{label}_collector" for label in FINAL_LABELS) + ("judge",)


PredictionEnricher = Callable[[dict[str, Any], EvaluationCaseResult], dict[str, Any]]


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
    service_tier: str = "standard"
    workflow_name: str = "framework_1_parallel_no_evidence"
    workflow_config: dict[str, Any] | None = None
    agent_configuration: dict[str, dict[str, Any]] | None = None
    error_stop_failed_llm_attempts: int = 10
    error_stop_failed_records: int = 3
    error_stop_rate: float = 0.5
    error_stop_min_records: int = 10
    case_workers: int = 4
    max_in_flight_llm: int = 20
    checkpoint_every: int = 250
    stop_after_tokens: int | None = None
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


@dataclass(frozen=True)
class EvaluationSnapshotResult:
    """Summary for a read-only partial artifact snapshot."""

    source_run_dir: str
    snapshot_dir: str
    source_status: str
    processed_records: int
    total_records: int


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


def snapshot_evaluation_results(
    run_dir: str | Path,
) -> EvaluationSnapshotResult:
    """Export a consistent partial report without modifying an active run."""
    source_run_dir = Path(run_dir)
    if not source_run_dir.is_dir():
        raise ValueError(f"Run directory does not exist: {source_run_dir}")

    sample_path = source_run_dir / "sampled_dataset.csv"
    config_path = source_run_dir / "run_config.json"
    manifest_path = source_run_dir / "manifest.json"
    progress_path = source_run_dir / "progress.sqlite"
    for required_path in (sample_path, config_path, manifest_path, progress_path):
        if not required_path.is_file():
            raise ValueError(
                f"Cannot snapshot run; required file is missing: {required_path}"
            )

    sample_df = load_evaluation_sample(sample_path)
    id_column = _sample_id_column(sample_df.columns)
    expected_ids = [str(value) for value in sample_df[id_column].tolist()]
    config_payload = _read_json_object(config_path)
    config_fields = {field.name for field in fields(EvaluationConfig)}
    config_values = {
        key: value
        for key, value in config_payload.items()
        if key in config_fields
    }
    config_values["sample_path"] = str(sample_path)
    config_values["total_records"] = len(sample_df)
    config = EvaluationConfig(**config_values)
    source_manifest = load_manifest(manifest_path)

    with ProgressStore(progress_path, read_only=True) as progress:
        progress.connection.execute("BEGIN")
        try:
            rows, all_calls, traces, errors = _materialize_completed_results(
                sample_df=sample_df,
                id_column=id_column,
                progress=progress,
            )
            progress_summary = progress.summary()
            coverage = progress.coverage(
                expected_ids,
                output_ids=[str(row["sample_id"]) for row in rows],
            )
        finally:
            progress.connection.rollback()

    if not rows:
        raise ValueError("Cannot snapshot run because it has no completed records")

    writer = ArtifactWriter(base_dir=source_run_dir / "snapshots")
    source_prompt_snapshot = source_run_dir / "first_completion_input_prompts.json"
    if source_prompt_snapshot.is_file():
        writer.write_json(
            "first_completion_input_prompts.json",
            _read_json_object(source_prompt_snapshot),
        )
    writer.write_jsonl("llm_calls.jsonl", all_calls)
    writer.write_jsonl("traces.jsonl", traces)
    writer.write_jsonl("errors.jsonl", errors)
    metrics = finalize_evaluation_artifacts(
        writer=writer,
        rows=rows,
        all_calls=all_calls,
        config=config,
        status="snapshot",
        stop_reason=None,
        agent_execution_duration_s=progress_summary["agent_execution_duration_s"],
        real_execution_duration_s=progress_summary["real_execution_duration_s"],
        coverage=coverage,
    )
    writer.write_json(
        "snapshot_manifest.json",
        {
            "schema_version": "1.0",
            "created_at": utc_now(),
            "source_run_dir": str(source_run_dir.resolve()),
            "source_status": str(source_manifest.get("status") or "unknown"),
            "progress": progress_summary,
            "coverage": coverage,
            "compiled_reports": metrics["compiled_reports"],
        },
    )
    return EvaluationSnapshotResult(
        source_run_dir=str(source_run_dir),
        snapshot_dir=str(writer.run_dir),
        source_status=str(source_manifest.get("status") or "unknown"),
        processed_records=len(rows),
        total_records=len(sample_df),
    )


def run_evaluation(
    *,
    sample_df: pd.DataFrame,
    workflow: EvaluationWorkflow,
    config: EvaluationConfig,
    run_id: str | None = None,
    artifact_base_dir: str | Path = "logs/runs",
    dataset_audit: dict[str, Any] | None = None,
    prediction_enricher: PredictionEnricher | None = None,
    resume_run_dir: str | Path | None = None,
) -> EvaluationResult:
    """Evaluate cases concurrently with crash-safe, ID-keyed progress tracking."""
    if sample_df.empty:
        raise ValueError("sample_df must contain at least one evaluation record")
    if config.case_workers <= 0:
        raise ValueError("case_workers must be greater than zero")
    if config.max_in_flight_llm <= 0:
        raise ValueError("max_in_flight_llm must be greater than zero")
    if config.checkpoint_every <= 0:
        raise ValueError("checkpoint_every must be greater than zero")
    if config.stop_after_tokens is not None and config.stop_after_tokens <= 0:
        raise ValueError("stop_after_tokens must be greater than zero when provided")

    id_column = _sample_id_column(sample_df.columns)
    records = sample_df.to_dict(orient="records")
    expected_ids = [str(record[id_column]) for record in records]
    if len(expected_ids) != len(set(expected_ids)):
        raise ValueError(f"{id_column} must be unique in the evaluation dataset")

    writer = ArtifactWriter(
        base_dir=artifact_base_dir,
        run_id=run_id,
        resume_run_dir=resume_run_dir,
    )

    run_config = asdict(config)
    run_config["agent_configuration"] = config.agent_configuration or {}
    dataset_fingerprint = dataframe_sha256(sample_df)
    workflow_identity = _workflow_identity(config)
    workflow_fingerprint = json_sha256(workflow_identity)
    manifest_path = writer.run_dir / "manifest.json"
    progress_path = writer.run_dir / "progress.sqlite"
    resumed = resume_run_dir is not None
    attempt_start_date = utc_now()

    if resumed:
        if not progress_path.is_file():
            raise ValueError(
                f"Cannot resume: progress database is missing from {writer.run_dir}"
            )
        manifest = load_manifest(manifest_path)
        _validate_resume_manifest(
            manifest,
            dataset_fingerprint=dataset_fingerprint,
            workflow_fingerprint=workflow_fingerprint,
            expected_ids=expected_ids,
        )
        manifest["status"] = "running"
        manifest["resume_count"] = int(manifest.get("resume_count", 0)) + 1
        manifest["last_resumed_at"] = attempt_start_date
        manifest["execution"] = {
            "case_workers": config.case_workers,
            "max_in_flight_llm": config.max_in_flight_llm,
            "checkpoint_every": config.checkpoint_every,
            "stop_after_tokens": config.stop_after_tokens,
        }
    else:
        manifest = {
            "schema_version": "1.0",
            "run_id": writer.run_dir.name,
            "status": "running",
            "created_at": attempt_start_date,
            "resume_count": 0,
            "dataset": {
                "sha256": dataset_fingerprint,
                "id_column": id_column,
                "total_records": len(sample_df),
                "frozen_path": "sampled_dataset.csv",
            },
            "workflow": {
                "name": config.workflow_name,
                "config_sha256": workflow_fingerprint,
                "identity": workflow_identity,
            },
            "execution": {
                "case_workers": config.case_workers,
                "max_in_flight_llm": config.max_in_flight_llm,
                "checkpoint_every": config.checkpoint_every,
                "stop_after_tokens": config.stop_after_tokens,
            },
            "progress_store": "progress.sqlite",
        }
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
    execution_attempts = manifest.setdefault("execution_attempts", [])
    if not isinstance(execution_attempts, list):
        raise ValueError("Cannot resume: manifest execution_attempts must be a list")
    current_attempt = {
        "attempt_number": int(manifest["resume_count"]) + 1,
        "start_date": attempt_start_date,
        "end_date": None,
    }
    execution_attempts.append(current_attempt)
    write_manifest(manifest_path, manifest)

    stop_reason: str | None = None
    worker_error: BaseException | None = None
    completed_this_session = 0
    session_total_tokens = 0
    rows: list[dict[str, Any]] = []
    all_calls: list[dict[str, Any]] = []
    coverage: dict[str, Any] = {}

    progress: ProgressStore | None = None
    try:
        progress = ProgressStore(progress_path)
        progress.initialize(expected_ids)
        reset_records = progress.reset_running() if resumed else 0
        statuses = progress.statuses()
        pending_records = [
            record
            for record in records
            if statuses[str(record[id_column])] != COMPLETED
        ]
        if resumed and not progress.has_execution_sessions():
            progress.add_legacy_execution_session(
                _trace_execution_duration_s(writer.run_dir / "traces.jsonl")
            )
        execution_session_id = (
            progress.start_execution_session() if pending_records else None
        )
        manifest["reset_running_records"] = reset_records
        manifest["progress"] = progress.summary()
        manifest["token_stop"] = {
            "threshold": config.stop_after_tokens,
            "current_session_total_tokens": session_total_tokens,
        }
        write_manifest(manifest_path, manifest)

        in_flight: dict[
            Future[tuple[EvaluationCaseResult, float, float, float]],
            tuple[str, dict[str, Any]],
        ] = {}
        next_record = 0

        def submit_available(executor: ThreadPoolExecutor) -> None:
            nonlocal next_record
            while (
                stop_reason is None
                and worker_error is None
                and len(in_flight) < config.case_workers
                and next_record < len(pending_records)
            ):
                record = pending_records[next_record]
                next_record += 1
                sample_id = str(record[id_column])
                progress.claim(sample_id)
                future = executor.submit(
                    _run_case_timed,
                    workflow,
                    sample_id,
                    str(record["text"]),
                )
                in_flight[future] = (sample_id, record)

        with ThreadPoolExecutor(
            max_workers=config.case_workers,
            thread_name_prefix="evaluation-case",
        ) as executor:
            submit_available(executor)
            while in_flight:
                completed_futures, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                for future in completed_futures:
                    expected_sample_id, record = in_flight.pop(future)
                    try:
                        (
                            result,
                            duration_s,
                            case_started_at_s,
                            case_finished_at_s,
                        ) = future.result()
                        if str(result.sample_id) != expected_sample_id:
                            raise ValueError(
                                "Workflow result sample_id does not match the evaluated record: "
                                f"expected {expected_sample_id!r}, got {result.sample_id!r}"
                            )
                        _write_first_completion_input_prompts(
                            writer=writer,
                            result=result,
                            workflow_name=config.workflow_name,
                        )
                        payload = _case_payload(
                            record=record,
                            result=result,
                            prediction_enricher=prediction_enricher,
                        )
                        call_started_at_s, call_finished_at_s = (
                            _call_execution_bounds(
                                result.traces,
                                fallback_started_at_s=case_started_at_s,
                                fallback_finished_at_s=case_finished_at_s,
                            )
                        )
                        if execution_session_id is None:
                            raise RuntimeError(
                                "Execution session is missing for a scheduled case"
                            )
                        progress.complete(
                            expected_sample_id,
                            payload=payload,
                            agent_execution_duration_s=duration_s,
                            execution_session_id=execution_session_id,
                            call_started_at_s=call_started_at_s,
                            call_finished_at_s=call_finished_at_s,
                            failed_llm_attempts=sum(
                                1 for call in payload["llm_calls"] if call.get("error")
                            ),
                            failed_record=result.final_label not in FINAL_LABELS,
                        )
                        completed_this_session += 1
                        session_total_tokens += _llm_call_total_tokens(
                            payload["llm_calls"]
                        )
                    except BaseException as exc:
                        progress.release(expected_sample_id)
                        if worker_error is None:
                            worker_error = exc
                    summary = progress.summary()
                    manifest["token_stop"] = {
                        "threshold": config.stop_after_tokens,
                        "current_session_total_tokens": session_total_tokens,
                    }
                    if worker_error is None and stop_reason is None:
                        stop_reason = _stop_reason_from_summary(
                            summary,
                            config,
                            session_total_tokens=session_total_tokens,
                        )
                    print(
                        json.dumps(
                            {
                                "event": "record_complete",
                                "sample_id": expected_sample_id,
                                "processed_records": summary["completed_records"],
                                "total_records": len(sample_df),
                                "failed_llm_attempts_total": summary[
                                    "failed_llm_attempts"
                                ],
                                "session_total_tokens": session_total_tokens,
                                "stop_reason": stop_reason,
                            },
                            ensure_ascii=False,
                        )
                    )
                    if (
                        completed_this_session > 0
                        and completed_this_session % config.checkpoint_every == 0
                    ):
                        manifest["progress"] = summary
                        manifest["checkpointed_at"] = utc_now()
                        write_manifest(manifest_path, manifest)
                submit_available(executor)

        rows, all_calls, traces, errors = _materialize_completed_results(
            sample_df=sample_df,
            id_column=id_column,
            progress=progress,
        )
        writer.write_jsonl("llm_calls.jsonl", all_calls)
        writer.write_jsonl("traces.jsonl", traces)
        writer.write_jsonl("errors.jsonl", errors)
        coverage = progress.coverage(
            expected_ids,
            output_ids=[str(row["sample_id"]) for row in rows],
        )
        progress_summary = progress.summary()
    except BaseException as exc:
        attempt_end_date = utc_now()
        current_attempt["end_date"] = attempt_end_date
        manifest.update(
            {
                "status": "interrupted",
                "interrupted_at": attempt_end_date,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        if progress is not None:
            manifest["progress"] = progress.summary()
        write_manifest(manifest_path, manifest)
        raise
    finally:
        if progress is not None:
            progress.close()

    if worker_error is not None:
        attempt_end_date = utc_now()
        current_attempt["end_date"] = attempt_end_date
        manifest.update(
            {
                "status": "interrupted",
                "interrupted_at": attempt_end_date,
                "progress": progress_summary,
                "coverage": coverage,
                "error": f"{type(worker_error).__name__}: {worker_error}",
            }
        )
        write_manifest(manifest_path, manifest)
        raise worker_error

    if stop_reason:
        final_status = "stopped_early"
    elif coverage["complete"]:
        final_status = "completed"
    else:
        final_status = "incomplete"

    attempt_end_date = utc_now()
    current_attempt["end_date"] = attempt_end_date
    manifest.update(
        {
            "status": final_status,
            "finished_at": attempt_end_date,
            "progress": progress_summary,
            "coverage": coverage,
            "stop_reason": stop_reason,
            "token_stop": {
                "threshold": config.stop_after_tokens,
                "current_session_total_tokens": session_total_tokens,
            },
        }
    )
    write_manifest(manifest_path, manifest)

    try:
        final_metrics = finalize_evaluation_artifacts(
            writer=writer,
            rows=rows,
            all_calls=all_calls,
            config=config,
            status=final_status,
            stop_reason=stop_reason,
            agent_execution_duration_s=progress_summary["agent_execution_duration_s"],
            real_execution_duration_s=progress_summary["real_execution_duration_s"],
            coverage=coverage,
            session_total_tokens=session_total_tokens,
        )
    except BaseException as exc:
        attempt_end_date = utc_now()
        current_attempt["end_date"] = attempt_end_date
        manifest.update(
            {
                "status": "interrupted",
                "interrupted_at": attempt_end_date,
                "progress": progress_summary,
                "coverage": coverage,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        write_manifest(manifest_path, manifest)
        raise

    return EvaluationResult(
        run_dir=str(writer.run_dir),
        status=final_status,
        processed_records=len(rows),
        total_records=len(sample_df),
        stop_reason=stop_reason,
        metrics=final_metrics,
    )


def finalize_evaluation_artifacts(
    *,
    writer: ArtifactWriter,
    rows: list[dict[str, Any]],
    all_calls: list[dict[str, Any]],
    config: EvaluationConfig,
    status: str,
    stop_reason: str | None,
    agent_execution_duration_s: float,
    real_execution_duration_s: float,
    coverage: dict[str, Any],
    session_total_tokens: int | None = None,
) -> dict[str, Any]:
    """Write the final evaluator artifacts and compile the review reports."""
    metrics = _write_evaluation_artifacts(
        writer=writer,
        rows=rows,
        all_calls=all_calls,
        config=config,
        status=status,
        stop_reason=stop_reason,
        agent_execution_duration_s=agent_execution_duration_s,
        real_execution_duration_s=real_execution_duration_s,
        coverage=coverage,
        session_total_tokens=session_total_tokens,
    )
    metrics["compiled_reports"] = compile_run_results(writer.run_dir)
    metrics["artifact_status"] = writer.required_artifact_status()
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
    """Return framework_1_parallel_no_evidence-only specialist columns for its compatibility wrapper."""
    specialist_answers = {
        label: getattr(result, "specialist_answers", {}).get(label)
        for label in FINAL_LABELS
    }
    specialist_outputs = {}
    for label in FINAL_LABELS:
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
    return row


def _collector_prediction_fields(
    _record: dict[str, Any],
    result: EvaluationCaseResult,
) -> dict[str, Any]:
    """Persist validated evidence packets without treating collectors as classifiers."""
    collector_outputs = {
        label: getattr(result, "collector_outputs", {}).get(label)
        for label in FINAL_LABELS
    }
    return {"collector_outputs": json.dumps(collector_outputs, ensure_ascii=False)}


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


def _run_case_timed(
    workflow: EvaluationWorkflow,
    sample_id: str,
    text: str,
) -> tuple[EvaluationCaseResult, float, float, float]:
    started_at = time.perf_counter()
    result = workflow.run_case(sample_id=sample_id, text=text)
    finished_at = time.perf_counter()
    return result, finished_at - started_at, started_at, finished_at


def _call_execution_bounds(
    traces: list[dict[str, Any]],
    *,
    fallback_started_at_s: float,
    fallback_finished_at_s: float,
) -> tuple[float, float]:
    """Return the first agent-call start and last agent-call end for one case."""
    starts: list[float] = []
    finishes: list[float] = []
    for trace in traces:
        try:
            starts.append(float(trace["node_started_at_s"]))
            finishes.append(float(trace["node_finished_at_s"]))
        except (KeyError, TypeError, ValueError):
            continue
    if starts and finishes:
        return min(starts), max(finishes)
    return float(fallback_started_at_s), float(fallback_finished_at_s)


def _trace_execution_duration_s(path: Path) -> float:
    """Recover a legacy run's first-call-to-last-call span from its traces."""
    if not path.is_file():
        return 0.0
    starts: list[float] = []
    finishes: list[float] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            trace = json.loads(line)
            try:
                starts.append(float(trace["node_started_at_s"]))
                finishes.append(float(trace["node_finished_at_s"]))
            except (KeyError, TypeError, ValueError):
                continue
    if not starts or not finishes:
        return 0.0
    return max(0.0, max(finishes) - min(starts))


def _case_payload(
    *,
    record: dict[str, Any],
    result: EvaluationCaseResult,
    prediction_enricher: PredictionEnricher | None,
) -> dict[str, Any]:
    return {
        "row": build_prediction_row(
            record=record,
            result=result,
            prediction_enricher=prediction_enricher,
        ),
        "llm_calls": [
            {
                "sample_id": result.sample_id,
                **{
                    key: value
                    for key, value in call.items()
                    if key != "_input_messages"
                },
            }
            for call in result.llm_calls
        ],
        "traces": [
            {"sample_id": result.sample_id, **trace}
            for trace in result.traces
        ],
        "errors": [
            {"sample_id": result.sample_id, **error}
            for error in result.errors
        ],
    }


def _write_first_completion_input_prompts(
    *,
    writer: ArtifactWriter,
    result: EvaluationCaseResult,
    workflow_name: str,
) -> None:
    """Persist actual agent inputs once, without duplicating them per case."""
    path = writer.run_dir / "first_completion_input_prompts.json"
    if path.exists():
        return

    agents: list[dict[str, Any]] = []
    seen_inputs: set[tuple[str, str]] = set()
    for call in result.llm_calls:
        messages = call.get("_input_messages")
        role = call.get("role")
        if not isinstance(role, str) or not isinstance(messages, list):
            continue
        normalized_messages = [
            {
                "role": str(message.get("role", "")),
                "content": str(message.get("content", "")),
            }
            for message in messages
            if isinstance(message, dict)
        ]
        serialized_messages = json.dumps(
            normalized_messages,
            ensure_ascii=False,
            sort_keys=True,
        )
        input_identity = (role, serialized_messages)
        if not normalized_messages or input_identity in seen_inputs:
            continue
        seen_inputs.add(input_identity)
        agents.append(
            {
                "role": role,
                "model_name": call.get("model_name"),
                "messages": normalized_messages,
            }
        )

    writer.write_json(
        "first_completion_input_prompts.json",
        {
            "schema_version": "1.0",
            "workflow_name": workflow_name,
            "sample_id": str(result.sample_id),
            "final_output": {
                "label_name": result.final_label,
                "suicidal_intent": result.suicidal_intent,
            },
            "agents": agents,
        },
    )


def _materialize_completed_results(
    *,
    sample_df: pd.DataFrame,
    id_column: str,
    progress: ProgressStore,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Rebuild final artifacts in frozen dataset order by joining on unique ID."""
    payloads = progress.completed_payloads()
    rows: list[dict[str, Any]] = []
    all_calls: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for record in sample_df.to_dict(orient="records"):
        payload = payloads.get(str(record[id_column]))
        if payload is None:
            continue
        rows.append(payload["row"])
        all_calls.extend(payload["llm_calls"])
        traces.extend(payload["traces"])
        errors.extend(payload["errors"])
    return rows, all_calls, traces, errors


def _read_json_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _workflow_identity(config: EvaluationConfig) -> dict[str, Any]:
    """Return behavior-affecting settings that must match when resuming."""
    return {
        "workflow_name": config.workflow_name,
        "model_name": config.model_name,
        "requested_model_name": config.requested_model_name,
        "max_retries": config.max_retries,
        "max_completion_tokens": config.max_completion_tokens,
        "temperature": config.temperature,
        "mode": config.mode,
        "service_tier": config.service_tier,
        "workflow_config": config.workflow_config or {},
        "agent_configuration": config.agent_configuration or {},
    }


def _validate_resume_manifest(
    manifest: dict[str, Any],
    *,
    dataset_fingerprint: str,
    workflow_fingerprint: str,
    expected_ids: list[str],
) -> None:
    dataset = manifest.get("dataset") or {}
    workflow = manifest.get("workflow") or {}
    if dataset.get("sha256") != dataset_fingerprint:
        raise ValueError(
            "Cannot resume: sampled_dataset.csv does not match the run manifest"
        )
    if int(dataset.get("total_records", -1)) != len(expected_ids):
        raise ValueError(
            "Cannot resume: dataset record count does not match the run manifest"
        )
    if workflow.get("config_sha256") != workflow_fingerprint:
        raise ValueError(
            "Cannot resume: workflow/model configuration does not match the run manifest"
        )


def _stop_reason_from_summary(
    summary: dict[str, Any],
    config: EvaluationConfig,
    *,
    session_total_tokens: int,
) -> str | None:
    processed_records = int(summary["completed_records"])
    failed_records = int(summary["failed_records"])
    failed_llm_attempts = int(summary["failed_llm_attempts"])
    if (
        config.stop_after_tokens is not None
        and session_total_tokens >= config.stop_after_tokens
    ):
        return (
            f"session_total_tokens={session_total_tokens} reached "
            f"stop_after_tokens={config.stop_after_tokens}"
        )
    if failed_llm_attempts >= config.error_stop_failed_llm_attempts:
        return (
            f"failed_llm_attempts={failed_llm_attempts} reached "
            f"threshold={config.error_stop_failed_llm_attempts}"
        )
    if failed_records >= config.error_stop_failed_records:
        return (
            f"failed_records={failed_records} reached "
            f"threshold={config.error_stop_failed_records}"
        )
    if processed_records >= config.error_stop_min_records:
        failure_rate = failed_records / processed_records if processed_records else 0.0
        if failure_rate >= config.error_stop_rate:
            return (
                f"failure_rate={failure_rate:.6f} reached "
                f"threshold={config.error_stop_rate:.6f} "
                f"after processed_records={processed_records}"
            )
    return None


def _llm_call_total_tokens(calls: list[dict[str, Any]]) -> int:
    """Count tokens reported by LLM attempts in one completed case."""
    total = 0
    for call in calls:
        reported_total = call.get("total_tokens")
        if reported_total is None:
            reported_total = (call.get("prompt_tokens") or 0) + (
                call.get("completion_tokens") or 0
            )
        try:
            total += max(0, int(reported_total))
        except (TypeError, ValueError):
            continue
    return total


def _write_evaluation_artifacts(
    *,
    writer: ArtifactWriter,
    rows: list[dict[str, Any]],
    all_calls: list[dict[str, Any]],
    config: EvaluationConfig,
    status: str,
    stop_reason: str | None,
    agent_execution_duration_s: float,
    real_execution_duration_s: float,
    coverage: dict[str, Any],
    session_total_tokens: int | None,
) -> dict[str, Any]:
    predictions = pd.DataFrame(rows)
    writer.write_dataframe("predictions.csv", predictions)
    judge_model_name = _configured_model_name(config, "judge")
    classification = compute_final_label_confusion_matrix(
        predictions["ground_truth"].tolist() if not predictions.empty else [],
        predictions["prediction"].tolist() if not predictions.empty else [],
    )
    suicidal_intent = compute_suicidal_intent_metrics(
        [row["ground_truth"] for row in rows],
        [row["prediction"] for row in rows],
        model_name=judge_model_name,
    )
    positive_subclassification = compute_positive_subclassification_metrics(
        [row["ground_truth"] for row in rows],
        [row["prediction"] for row in rows],
    )
    runtime = summarize_runtime(
        all_calls,
        agent_execution_duration_s=agent_execution_duration_s,
        real_execution_duration_s=real_execution_duration_s,
    )
    metrics = {
        "workflow_name": config.workflow_name,
        "model_name": config.model_name,
        "requested_model_name": config.requested_model_name,
        "mode": config.mode,
        "service_tier": config.service_tier,
        "evaluation_status": status,
        "stop_reason": stop_reason,
        "processed_records": len(rows),
        "total_records": config.total_records,
        "coverage": coverage,
        "token_stop": {
            "threshold": config.stop_after_tokens,
            "current_session_total_tokens": session_total_tokens,
            "reached": bool(
                config.stop_after_tokens is not None
                and session_total_tokens is not None
                and session_total_tokens >= config.stop_after_tokens
            ),
        },
        "classification": classification,
        "suicidal_intent": suicidal_intent,
        "positive_subclassification": positive_subclassification,
        "runtime": runtime,
    }
    writer.write_dataframe("suicidal_intent_metrics.csv", pd.DataFrame([suicidal_intent["binary_summary"]]))
    writer.write_dataframe(
        "positive_subclassification_metrics.csv",
        pd.DataFrame([positive_subclassification["summary"]]),
    )
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
            "service_tier": config.service_tier,
            "evaluation_status": status,
            "processed_records": len(rows),
            "total_records": config.total_records,
            "stop_reason": stop_reason,
            "coverage": coverage,
            "token_stop": metrics["token_stop"],
            "runtime": runtime,
            "json_parsing_errors": runtime["json_parsing_errors"],
            "failed_records": classification["failed_records"],
            "failure_rate": classification["failure_rate"],
            "llm_trace_file": "llm_calls.jsonl",
            "node_trace_file": "traces.jsonl",
        },
    )
    return metrics


def _complete_results_payload(rows: list[dict[str, Any]], metrics: dict[str, Any]) -> dict[str, Any]:
    results = []
    for row in rows:
        result = dict(row)
        for field in (
            "agent_outputs",
            "specialist_answers",
            "specialist_outputs",
            "collector_outputs",
        ):
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


def _framework_1_parallel_no_evidence_agent_configuration(config: EvaluationConfig) -> dict[str, dict[str, Any]]:
    evidence_mode = config.mode == "evidence"
    labels = FINAL_LABELS
    role_suffix = "collector" if evidence_mode else "specialist"
    agent_suffix = " Evidence Collector" if evidence_mode else " Agent"
    configuration = {
        f"{label}_{role_suffix}": {
            "agent_name": f"{label}{agent_suffix}",
            "model_name": config.model_name,
            "temperature": config.temperature,
        }
        for label in labels
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


def _validate_framework_1_parallel_no_evidence_agent_overrides(
    config: EvaluationConfig,
    agent_models: dict[str, InvokableModel] | None,
    agent_configuration: dict[str, dict[str, Any]],
) -> None:
    supplied_roles = set(agent_models or {})
    configured_roles = set(agent_configuration)
    allowed_roles = FRAMEWORK_2_PARALLEL_WITH_EVIDENCE_AGENT_ROLES if config.mode == "evidence" else FRAMEWORK_1_PARALLEL_NO_EVIDENCE_AGENT_ROLES
    unknown_roles = sorted((supplied_roles | configured_roles) - set(allowed_roles))
    if unknown_roles:
        framework_name = (
            "framework_2_parallel_with_evidence"
            if config.mode == "evidence"
            else "framework_1_parallel_no_evidence"
        )
        raise ValueError(
            f"Unknown {framework_name} role overrides: {unknown_roles}. "
            f"Expected roles: {list(allowed_roles)}"
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
            "framework_1_parallel_no_evidence per-agent model/temperature overrides require matching model objects for roles: "
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

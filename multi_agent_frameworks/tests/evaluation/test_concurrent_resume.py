import json
from pathlib import Path
import sqlite3
import threading
import time
from types import SimpleNamespace

import pandas as pd

import cli
from evaluation.evaluator import EvaluationConfig, run_evaluation


class CountingWorkflow:
    def __init__(self, delay_s: float = 0.0):
        self.delay_s = delay_s
        self.counts: dict[str, int] = {}
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def run_case(self, *, sample_id: str, text: str):
        with self.lock:
            self.counts[sample_id] = self.counts.get(sample_id, 0) + 1
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            if self.delay_s:
                time.sleep(self.delay_s)
        finally:
            with self.lock:
                self.active -= 1
        return SimpleNamespace(
            sample_id=sample_id,
            final_label="Unsupportive",
            suicidal_intent=False,
            errors=[],
            traces=[{"event": "complete"}],
            llm_calls=[],
        )


class TokenCountingWorkflow(CountingWorkflow):
    def run_case(self, *, sample_id: str, text: str):
        result = super().run_case(sample_id=sample_id, text=text)
        result.llm_calls = [
            {
                "role": "token_test",
                "attempt": 1,
                "model_name": "test-model",
                "latency_s": 0.0,
                "prompt_tokens": 7,
                "completion_tokens": 3,
                "total_tokens": 10,
            }
        ]
        return result


def _sample(size: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ID": 1000 + index * 7,
                "users": index,
                "text": f"neutral {index}",
                "sentiment": "Unsupportive",
                "time": index,
            }
            for index in range(size)
        ]
    )


def _config(
    tmp_path: Path,
    size: int,
    *,
    case_workers: int = 4,
    stop_after_tokens: int | None = None,
) -> EvaluationConfig:
    return EvaluationConfig(
        model_name="test-model",
        requested_model_name="test-model",
        sample_path=str(tmp_path / "input.csv"),
        dataset_audit_path=None,
        max_retries=0,
        max_completion_tokens=32,
        temperature=0,
        total_records=size,
        workflow_name="test_counting_workflow",
        error_stop_failed_llm_attempts=1000,
        error_stop_failed_records=1000,
        error_stop_rate=1.0,
        error_stop_min_records=1000,
        case_workers=case_workers,
        max_in_flight_llm=20,
        checkpoint_every=3,
        stop_after_tokens=stop_after_tokens,
    )


def test_coordinator_schedules_every_unique_id_once_and_checks_coverage(tmp_path):
    sample = _sample(16)
    workflow = CountingWorkflow(delay_s=0.01)

    result = run_evaluation(
        sample_df=sample,
        workflow=workflow,
        config=_config(tmp_path, len(sample)),
        run_id="concurrent",
        artifact_base_dir=tmp_path / "runs",
    )

    run_dir = Path(result.run_dir)
    assert workflow.counts == {str(sample_id): 1 for sample_id in sample["ID"]}
    assert workflow.max_active == 4
    assert (
        0
        < result.metrics["runtime"]["real_execution_duration_s"]
        < result.metrics["runtime"]["agent_execution_duration_s"]
    )
    assert result.metrics["coverage"] == {
        "expected_records": 16,
        "output_records": 16,
        "unique_output_ids": 16,
        "missing_ids": [],
        "unexpected_ids": [],
        "duplicate_ids": [],
        "complete": True,
    }
    predictions = pd.read_csv(run_dir / "predictions.csv", encoding="utf-8-sig")
    assert predictions["ID"].tolist() == sample["ID"].tolist()
    results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    assert [row["sample_id"] for row in results["results"]] == sample["ID"].tolist()

    with sqlite3.connect(run_dir / "progress.sqlite") as connection:
        columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(cases)")
        }
    assert "sample_id" in columns
    assert "input_position" not in columns

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["dataset"]["id_column"] == "ID"
    assert manifest["coverage"]["complete"] is True
    assert manifest["progress"]["real_execution_duration_s"] == result.metrics["runtime"][
        "real_execution_duration_s"
    ]


def test_resume_schedules_only_pending_ids_and_restores_dataset_order(tmp_path):
    sample = _sample(5)
    config = _config(tmp_path, len(sample), case_workers=2)
    initial_workflow = CountingWorkflow()
    initial = run_evaluation(
        sample_df=sample,
        workflow=initial_workflow,
        config=config,
        run_id="resume",
        artifact_base_dir=tmp_path / "runs",
    )
    run_dir = Path(initial.run_dir)
    original_prompt_snapshot = (
        run_dir / "first_completion_input_prompts.json"
    ).read_text(encoding="utf-8")
    pending_id = str(sample.loc[2, "ID"])

    with sqlite3.connect(run_dir / "progress.sqlite") as connection:
        connection.execute(
            """
            UPDATE cases
            SET status='running',
                result_json=NULL,
                agent_execution_duration_s=0,
                failed_llm_attempts=0,
                failed_record=0
            WHERE sample_id=?
            """,
            (pending_id,),
        )
        connection.commit()

    resumed_workflow = CountingWorkflow()
    resumed = run_evaluation(
        sample_df=sample,
        workflow=resumed_workflow,
        config=config,
        resume_run_dir=run_dir,
    )

    assert resumed.status == "completed"
    assert resumed_workflow.counts == {pending_id: 1}
    assert resumed.metrics["coverage"]["complete"] is True
    assert (
        run_dir / "first_completion_input_prompts.json"
    ).read_text(encoding="utf-8") == original_prompt_snapshot
    predictions = pd.read_csv(run_dir / "predictions.csv", encoding="utf-8-sig")
    assert predictions["ID"].tolist() == sample["ID"].tolist()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["resume_count"] == 1
    assert manifest["reset_running_records"] == 1
    assert manifest["progress"]["completed_records"] == len(sample)
    assert manifest["progress"]["real_execution_duration_s"] >= initial.metrics["runtime"][
        "real_execution_duration_s"
    ]


def test_token_stop_counter_resets_for_each_resume_invocation(tmp_path):
    sample = _sample(5)
    config = _config(
        tmp_path,
        len(sample),
        case_workers=1,
        stop_after_tokens=15,
    )
    initial_workflow = TokenCountingWorkflow()
    initial = run_evaluation(
        sample_df=sample,
        workflow=initial_workflow,
        config=config,
        run_id="token-stop",
        artifact_base_dir=tmp_path / "runs",
    )

    assert initial.status == "stopped_early"
    assert initial.processed_records == 2
    assert initial.stop_reason == (
        "session_total_tokens=20 reached stop_after_tokens=15"
    )
    assert initial.metrics["runtime"]["tokens"]["total"] == 20
    assert initial.metrics["token_stop"] == {
        "threshold": 15,
        "current_session_total_tokens": 20,
        "reached": True,
    }

    resumed_workflow = TokenCountingWorkflow()
    resumed = run_evaluation(
        sample_df=sample,
        workflow=resumed_workflow,
        config=config,
        resume_run_dir=initial.run_dir,
    )

    assert resumed.status == "stopped_early"
    assert resumed.processed_records == 4
    assert resumed.stop_reason == (
        "session_total_tokens=20 reached stop_after_tokens=15"
    )
    assert resumed.metrics["runtime"]["tokens"]["total"] == 40
    assert resumed.metrics["token_stop"] == {
        "threshold": 15,
        "current_session_total_tokens": 20,
        "reached": True,
    }
    assert resumed_workflow.counts == {
        str(sample.loc[2, "ID"]): 1,
        str(sample.loc[3, "ID"]): 1,
    }


def test_snapshot_results_exports_completed_sqlite_rows_without_mutating_run(
    tmp_path,
    capsys,
):
    sample = _sample(5)
    result = run_evaluation(
        sample_df=sample,
        workflow=CountingWorkflow(),
        config=_config(tmp_path, len(sample)),
        run_id="snapshot-source",
        artifact_base_dir=tmp_path / "runs",
    )
    run_dir = Path(result.run_dir)
    pending_id = str(sample.loc[2, "ID"])
    original_predictions = (run_dir / "predictions.csv").read_bytes()
    capsys.readouterr()

    with sqlite3.connect(run_dir / "progress.sqlite") as connection:
        connection.execute(
            """
            UPDATE cases
            SET status='running', result_json=NULL
            WHERE sample_id=?
            """,
            (pending_id,),
        )
        connection.commit()

    assert (
        cli.main(
            [
                "snapshot-results",
                "--run-dir",
                str(run_dir),
            ]
        )
        == 0
    )
    command_result = json.loads(capsys.readouterr().out)
    snapshot_dir = Path(command_result["snapshot_dir"])
    snapshot_metrics = json.loads(
        (snapshot_dir / "metrics.json").read_text(encoding="utf-8")
    )
    snapshot_predictions = pd.read_csv(
        snapshot_dir / "predictions.csv",
        encoding="utf-8-sig",
    )
    snapshot_manifest = json.loads(
        (snapshot_dir / "snapshot_manifest.json").read_text(encoding="utf-8")
    )

    assert command_result["processed_records"] == 4
    assert snapshot_metrics["evaluation_status"] == "snapshot"
    assert snapshot_metrics["processed_records"] == 4
    assert snapshot_metrics["total_records"] == 5
    assert snapshot_metrics["coverage"]["missing_ids"] == [pending_id]
    assert snapshot_predictions["ID"].tolist() == [
        value for value in sample["ID"].tolist() if str(value) != pending_id
    ]
    assert snapshot_manifest["progress"]["completed_records"] == 4
    assert (
        snapshot_dir / "first_completion_input_prompts.json"
    ).read_text(encoding="utf-8") == (
        run_dir / "first_completion_input_prompts.json"
    ).read_text(encoding="utf-8")
    assert (snapshot_dir / "compiled_results.csv").exists()
    assert (snapshot_dir / "compiled_results.md").exists()
    assert (run_dir / "predictions.csv").read_bytes() == original_predictions

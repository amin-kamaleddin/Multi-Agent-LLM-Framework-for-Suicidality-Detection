import json
from pathlib import Path

import pandas as pd

from suicidal_multi_agent.core.labels import VALID_LABELS
from suicidal_multi_agent.evaluation.evaluator import EvaluationConfig, load_evaluation_sample, run_workflow_evaluation


class EvaluatorFakeModel:
    def invoke(self, messages):
        system = messages[0]["content"]
        if "final judge" in system.lower():
            return json.dumps({"label_name": "Indicator"})
        for label in VALID_LABELS:
            if f"Target label: {label}" in system:
                return json.dumps({"answer": label == "Indicator"})
        raise AssertionError("Unexpected evaluator prompt")


class EvaluatorFailingModel:
    def invoke(self, messages):
        raise RuntimeError("synthetic transport failure")


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"source_row": 10, "text": "indicator text", "sentiment": "Indicator"},
            {"source_row": 11, "text": "ideation text", "sentiment": "Ideation"},
            {"source_row": 12, "text": "behavior text", "sentiment": "Behavior"},
            {"source_row": 13, "text": "attempt text", "sentiment": "Attempt"},
        ]
    )


def _config(tmp_path: Path, *, model_name: str = "test-model", total_records: int = 4) -> EvaluationConfig:
    return EvaluationConfig(
        model_name=model_name,
        requested_model_name=model_name,
        sample_path=str(tmp_path / "sample.csv"),
        dataset_audit_path=None,
        max_retries=0,
        max_completion_tokens=32,
        temperature=0.0,
        total_records=total_records,
        error_stop_failed_llm_attempts=4,
        error_stop_failed_records=1,
        error_stop_rate=0.5,
        error_stop_min_records=2,
    )


def test_load_evaluation_sample_validates_frozen_columns(tmp_path):
    sample_path = tmp_path / "sample.csv"
    _sample_df().to_csv(sample_path, index=False, encoding="utf-8-sig")

    loaded = load_evaluation_sample(sample_path, limit=2)

    assert list(loaded.columns) == ["source_row", "text", "sentiment"]
    assert len(loaded) == 2
    assert loaded["source_row"].tolist() == [10, 11]


def test_evaluator_writes_complete_artifacts_for_fake_model(tmp_path):
    sample_df = _sample_df()
    config = _config(tmp_path)

    result = run_workflow_evaluation(
        sample_df=sample_df,
        model=EvaluatorFakeModel(),
        config=config,
        run_id="test-evaluator-complete",
        artifact_base_dir=tmp_path / "runs",
    )

    run_dir = Path(result.run_dir)
    assert result.status == "completed"
    assert result.processed_records == 4
    assert (run_dir / "model_metrics.csv").exists()
    assert (run_dir / "label_metrics.csv").exists()
    assert (run_dir / "trace_summary.json").exists()
    assert (run_dir / "llm_calls.jsonl").exists()
    assert (run_dir / "traces.jsonl").exists()

    model_metrics = pd.read_csv(run_dir / "model_metrics.csv", encoding="utf-8-sig")
    assert model_metrics.loc[0, "model_name"] == "test-model"
    assert model_metrics.loc[0, "accuracy"] == 0.25

    predictions = pd.read_csv(run_dir / "predictions.csv", encoding="utf-8-sig")
    assert {
        "specialist_answers",
        "indicator_answer",
        "ideation_answer",
        "behavior_answer",
        "attempt_answer",
    } <= set(predictions.columns)
    specialist_answers = json.loads(predictions.loc[0, "specialist_answers"])
    assert specialist_answers == {
        "Indicator": True,
        "Ideation": False,
        "Behavior": False,
        "Attempt": False,
    }
    assert predictions.loc[0, "indicator_answer"] == True
    assert predictions.loc[0, "ideation_answer"] == False


def test_evaluator_stops_early_on_abnormal_request_errors(tmp_path):
    sample_df = _sample_df()
    config = _config(tmp_path, model_name="failing-model")

    result = run_workflow_evaluation(
        sample_df=sample_df,
        model=EvaluatorFailingModel(),
        config=config,
        run_id="test-evaluator-stopped",
        artifact_base_dir=tmp_path / "runs",
    )

    run_dir = Path(result.run_dir)
    trace_summary = json.loads((run_dir / "trace_summary.json").read_text(encoding="utf-8"))
    assert result.status == "stopped_early"
    assert result.processed_records == 1
    assert "failed_llm_attempts=4" in result.stop_reason
    assert trace_summary["evaluation_status"] == "stopped_early"
    assert trace_summary["runtime"]["failed_llm_attempts"] == 4

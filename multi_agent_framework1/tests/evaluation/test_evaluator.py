import json
from pathlib import Path

import pandas as pd

from suicidal_multi_agent.core.labels import VALID_LABELS
from suicidal_multi_agent.evaluation.evaluator import EvaluationConfig, load_evaluation_sample, run_workflow_evaluation


class EvaluatorFakeModel:
    def invoke(self, messages):
        system = messages[0]["content"]
        if "final judge" in system.lower():
            return json.dumps({"label_name": "Indicator", "suicidal_intent": True})
        for label in VALID_LABELS:
            if f"Target label: {label}" in system:
                return json.dumps({"answer": label == "Indicator"})
        raise AssertionError("Unexpected evaluator prompt")


class EvaluatorReasonedFakeModel:
    def invoke(self, messages):
        system = messages[0]["content"]
        if "final judge" in system.lower():
            assert "Specialist outputs" in messages[1]["content"]
            assert '"reasoning"' in messages[1]["content"]
            assert '"evidence"' not in messages[1]["content"]
            return json.dumps({"label_name": "Attempt", "suicidal_intent": True})
        for label in VALID_LABELS:
            if f"Target label: {label}" in system:
                answer = label == "Attempt"
                return json.dumps(
                    {
                        "answer": answer,
                        "reasoning": f"{label} evaluator rationale",
                    }
                )
        raise AssertionError("Unexpected evaluator prompt")


class EvaluatorFailingModel:
    def invoke(self, messages):
        raise RuntimeError("synthetic transport failure")


class EvaluatorUnsupportiveFakeModel:
    def invoke(self, messages):
        system = messages[0]["content"]
        if "final judge" in system.lower():
            return json.dumps({"label_name": "Unsupportive", "suicidal_intent": False})
        for label in VALID_LABELS:
            if f"Target label: {label}" in system:
                return json.dumps({"answer": False})
        raise AssertionError("Unexpected evaluator prompt")


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"source_row": 10, "text": "indicator text", "sentiment": "Indicator"},
            {"source_row": 11, "text": "ideation text", "sentiment": "Ideation"},
            {"source_row": 12, "text": "behavior text", "sentiment": "Behavior"},
            {"source_row": 13, "text": "attempt text", "sentiment": "Attempt"},
        ]
    )


def _config(
    tmp_path: Path,
    *,
    model_name: str = "test-model",
    total_records: int = 4,
    mode: str = "bool",
) -> EvaluationConfig:
    return EvaluationConfig(
        model_name=model_name,
        requested_model_name=model_name,
        sample_path=str(tmp_path / "sample.csv"),
        dataset_audit_path=None,
        max_retries=0,
        max_completion_tokens=32,
        temperature=0.0,
        total_records=total_records,
        mode=mode,
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


def test_load_evaluation_sample_accepts_optional_real_provenance(tmp_path):
    sample_path = tmp_path / "sample-with-provenance.csv"
    sample = _sample_df().copy()
    sample["real"] = [True, True, False, False]
    sample.to_csv(sample_path, index=False, encoding="utf-8-sig")

    loaded = load_evaluation_sample(sample_path)

    assert list(loaded.columns) == ["source_row", "text", "sentiment", "real"]
    assert loaded["real"].tolist() == [True, True, False, False]


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
    assert (run_dir / "compiled_results.csv").exists()
    assert (run_dir / "compiled_results.md").exists()
    assert (run_dir / "agent_metrics.csv").exists()
    assert (run_dir / "suicidal_intent_metrics.csv").exists()
    assert (run_dir / "results.json").exists()
    assert (run_dir / "misclassified_ids.txt").read_text(encoding="utf-8") == "11\n12\n13\n"
    assert (run_dir / "suicidal_intent_misclassified_ids.txt").read_text(encoding="utf-8") == ""
    assert (run_dir / "classification_metrics.svg").exists()
    assert (run_dir / "runtime_metrics.svg").exists()
    assert result.metrics["compiled_reports"] == {
        "compiled_csv": str(run_dir / "compiled_results.csv"),
        "compiled_markdown": str(run_dir / "compiled_results.md"),
    }
    assert result.metrics["artifact_status"]["compiled_results.csv"] is True
    assert result.metrics["artifact_status"]["compiled_results.md"] is True
    assert all(result.metrics["artifact_status"].values())
    assert result.metrics["suicidal_intent"]["accuracy"] == 1.0
    assert result.metrics["suicidal_intent"]["has_both_ground_truth_classes"] is False
    assert len(result.metrics["agent_metrics"]) == 6

    model_metrics = pd.read_csv(run_dir / "model_metrics.csv", encoding="utf-8-sig")
    assert model_metrics.loc[0, "model_name"] == "test-model"
    assert model_metrics.loc[0, "accuracy"] == 0.25

    predictions = pd.read_csv(run_dir / "predictions.csv", encoding="utf-8-sig")
    assert {
        "specialist_answers",
        "specialist_outputs",
        "indicator_answer",
        "ideation_answer",
        "behavior_answer",
        "attempt_answer",
        "ground_truth_suicidal_intent",
        "suicidal_intent",
        "suicidal_intent_correct",
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
    specialist_outputs = json.loads(predictions.loc[0, "specialist_outputs"])
    assert specialist_outputs["Indicator"] == {"answer": True}
    assert predictions["ground_truth_suicidal_intent"].tolist() == [True, True, True, True]
    assert predictions["suicidal_intent"].tolist() == [True, True, True, True]

    complete_results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    assert complete_results["schema_version"] == "1.0"
    assert complete_results["results"][0]["final_output"] == {
        "label_name": "Indicator",
        "suicidal_intent": True,
    }
    run_config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    assert run_config["agent_configuration"]["Final Judge"] == {
        "role": "judge",
        "model_name": "test-model",
        "temperature": 0.0,
    }


def test_evaluator_writes_reasoned_specialist_outputs(tmp_path):
    sample_df = _sample_df()
    config = _config(tmp_path, model_name="reasoned-test-model", mode="reasoned")

    result = run_workflow_evaluation(
        sample_df=sample_df,
        model=EvaluatorReasonedFakeModel(),
        config=config,
        run_id="test-evaluator-reasoned",
        artifact_base_dir=tmp_path / "runs",
    )

    run_dir = Path(result.run_dir)
    assert result.status == "completed"

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["mode"] == "reasoned"
    assert metrics["compiled_reports"]["compiled_csv"].endswith("compiled_results.csv")

    predictions = pd.read_csv(run_dir / "predictions.csv", encoding="utf-8-sig")
    assert {
        "specialist_outputs",
        "attempt_reasoning",
    } <= set(predictions.columns)
    assert "attempt_evidence" not in predictions.columns
    specialist_outputs = json.loads(predictions.loc[0, "specialist_outputs"])
    assert specialist_outputs["Attempt"] == {
        "answer": True,
        "reasoning": "Attempt evaluator rationale",
    }
    assert predictions.loc[0, "attempt_reasoning"] == "Attempt evaluator rationale"


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


def test_evaluator_accepts_unsupportive_and_computes_negative_intent_metrics(tmp_path):
    sample_df = pd.DataFrame(
        [{"source_row": 20, "text": "neutral text", "sentiment": "Unsupportive"}]
    )
    config = _config(tmp_path, total_records=1)

    result = run_workflow_evaluation(
        sample_df=sample_df,
        model=EvaluatorUnsupportiveFakeModel(),
        config=config,
        run_id="test-evaluator-unsupportive",
        artifact_base_dir=tmp_path / "runs",
    )

    predictions = pd.read_csv(Path(result.run_dir) / "predictions.csv", encoding="utf-8-sig")
    assert predictions.loc[0, "prediction"] == "Unsupportive"
    assert bool(predictions.loc[0, "suicidal_intent"]) is False
    assert bool(predictions.loc[0, "ground_truth_suicidal_intent"]) is False
    assert result.metrics["classification"]["failed_records"] == 0
    assert result.metrics["suicidal_intent"]["accuracy"] == 1.0

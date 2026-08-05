import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from domain.labels import FINAL_LABELS
from evaluation.evaluator import (
    EvaluationConfig,
    load_evaluation_sample,
    run_evaluation,
)
from workflows.framework_1_parallel_no_evidence.evaluation import run_evaluation as run_framework_1_evaluation
from workflows.framework_2_parallel_with_evidence.evaluation import run_evaluation as run_framework_2_evaluation


class EvaluatorFakeModel:
    def invoke(self, messages):
        system = messages[0]["content"]
        if "final judge" in system.lower():
            return json.dumps({"label_name": "Indicator"})
        for label in FINAL_LABELS:
            if f"Target label: {label}" in system:
                return json.dumps({"answer": label == "Indicator"})
        raise AssertionError("Unexpected evaluator prompt")


class EvaluatorFailingModel:
    def invoke(self, messages):
        raise RuntimeError("synthetic transport failure")


class EvaluatorUnsupportiveFakeModel:
    def invoke(self, messages):
        system = messages[0]["content"]
        if "final judge" in system.lower():
            return json.dumps({"label_name": "Unsupportive"})
        for label in FINAL_LABELS:
            if f"Target label: {label}" in system:
                return json.dumps({"answer": label == "Unsupportive"})
        raise AssertionError("Unexpected evaluator prompt")


class LabelOnlyJudgeFakeModel:
    """Return only a label from the judge while supporting both parallel workflows."""

    def invoke(self, messages):
        system = messages[0]["content"]
        user = messages[1]["content"]
        expected = next(
            label for label in FINAL_LABELS if f"EXPECTED={label}" in user
        )
        if "final judge" in system.lower():
            return json.dumps({"label_name": expected})
        for label in FINAL_LABELS:
            if f"Evidence family: {label}" in system:
                return json.dumps({"evidence": []})
            if f"Target label: {label}" in system:
                return json.dumps({"answer": label == expected})
        raise AssertionError("Unexpected evaluator prompt")


class GenericWorkflow:
    """Variable-role workflow with no comparable agent-level outputs."""

    def run_case(self, *, sample_id, text):
        is_negative = "neutral" in text
        llm_calls = [
            {
                "role": "router",
                "attempt": 1,
                "model_name": "router-model",
                "latency_s": 0.01,
                "prompt_tokens": 3,
                "completion_tokens": 1,
                "total_tokens": 4,
                "cost": 0.001,
            }
        ]
        if is_negative:
            llm_calls.append(
                {
                    "role": "critic",
                    "attempt": 1,
                    "model_name": "critic-model",
                    "latency_s": 0.02,
                    "prompt_tokens": 5,
                    "completion_tokens": 2,
                    "total_tokens": 7,
                    "cost": 0.002,
                }
            )
        return SimpleNamespace(
            sample_id=sample_id,
            final_label="Unsupportive" if is_negative else "Indicator",
            suicidal_intent=not is_negative,
            errors=[],
            traces=[{"event": "route_complete", "role": "router"}],
            llm_calls=llm_calls,
        )


class GenericWorkflowWithAgentOutputs(GenericWorkflow):
    def run_case(self, *, sample_id, text):
        result = super().run_case(sample_id=sample_id, text=text)
        result.agent_outputs = {"router": {"decision": result.final_label}}
        return result


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
    agent_configuration=None,
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
        agent_configuration=agent_configuration,
        error_stop_failed_llm_attempts=4,
        error_stop_failed_records=1,
        error_stop_rate=0.5,
        error_stop_min_records=2,
        case_workers=1,
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


def test_load_evaluation_sample_accepts_canonical_rsd15k_fields(tmp_path):
    sample_path = tmp_path / "canonical.csv"
    sample = pd.DataFrame(
        [
            {"ID": 10, "users": 2, "text": "indicator", "sentiment": "Indicator", "time": 100, "real": True},
            {"ID": 20, "users": 3, "text": "neutral", "sentiment": "Unsupportive", "time": 200, "real": False},
        ]
    )
    sample.to_csv(sample_path, index=False, encoding="utf-8-sig")

    loaded = load_evaluation_sample(sample_path)

    assert list(loaded.columns) == ["ID", "users", "text", "sentiment", "time", "real"]
    assert loaded["ID"].tolist() == [10, 20]
    assert loaded["users"].tolist() == [2, 3]


def test_load_evaluation_sample_rejects_duplicate_canonical_ids(tmp_path):
    sample_path = tmp_path / "duplicate-ids.csv"
    sample = pd.DataFrame(
        [
            {"ID": 10, "users": 2, "text": "indicator", "sentiment": "Indicator", "time": 100},
            {"ID": 10, "users": 3, "text": "neutral", "sentiment": "Unsupportive", "time": 200},
        ]
    )
    sample.to_csv(sample_path, index=False, encoding="utf-8-sig")

    with pytest.raises(ValueError, match="ID must be unique"):
        load_evaluation_sample(sample_path)


def test_generic_evaluator_calculates_only_final_judge_metrics(tmp_path):
    sample = pd.DataFrame(
        [
            {"ID": 10, "users": 2, "text": "indicator", "sentiment": "Indicator", "time": 100},
            {"ID": 20, "users": 3, "text": "neutral", "sentiment": "Unsupportive", "time": 200},
        ]
    )
    config = _config(
        tmp_path,
        total_records=2,
        agent_configuration={
            "router": {"model_name": "router-model", "temperature": 0.0},
            "critic": {"model_name": "critic-model", "temperature": 0.0},
        },
    )

    result = run_evaluation(
        sample_df=sample,
        workflow=GenericWorkflow(),
        config=config,
        run_id="generic-no-agent-metrics",
        artifact_base_dir=tmp_path / "runs",
    )

    run_dir = Path(result.run_dir)
    predictions = pd.read_csv(run_dir / "predictions.csv", encoding="utf-8-sig")
    assert result.metrics["classification"]["confusion_matrix"]["Indicator"]["Indicator"] == 1
    assert result.metrics["classification"]["confusion_matrix"]["Unsupportive"]["Unsupportive"] == 1
    assert result.metrics["suicidal_intent"]["accuracy"] == 1.0
    assert result.metrics["positive_subclassification"]["support"] == 1
    assert result.metrics["positive_subclassification"]["accuracy"] == 1.0
    assert result.metrics["positive_subclassification"]["precision"] == 0.25
    assert result.metrics["positive_subclassification"]["f1_score"] == 0.25
    assert result.metrics["runtime"]["call_count"] == 3
    assert result.metrics["runtime"]["tokens"]["total"] == 15
    assert "agent_metrics" not in result.metrics
    assert "specialist_metrics" not in result.metrics
    assert "specialist_answers" not in predictions.columns
    assert "agent_outputs" not in predictions.columns
    assert predictions["ID"].tolist() == [10, 20]
    assert predictions["users"].tolist() == [2, 3]
    assert predictions["sample_id"].tolist() == [10, 20]
    assert not (run_dir / "agent_metrics.csv").exists()
    assert "optional_artifact_status" not in result.metrics
    assert all(result.metrics["artifact_status"].values())
    compiled_markdown = (run_dir / "compiled_results.md").read_text(encoding="utf-8")
    assert "## Per-Agent Metrics" not in compiled_markdown

    run_config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    assert set(run_config["agent_configuration"]) == {"router", "critic"}


def test_evaluator_tracks_only_workflow_execution_duration(monkeypatch, tmp_path):
    sample = pd.DataFrame(
        [
            {"ID": 10, "users": 2, "text": "indicator", "sentiment": "Indicator", "time": 100},
            {"ID": 20, "users": 3, "text": "neutral", "sentiment": "Unsupportive", "time": 200},
        ]
    )
    timer_values = iter([10.0, 11.25, 20.0, 22.5])
    monkeypatch.setattr(
        "evaluation.evaluator.time.perf_counter",
        lambda: next(timer_values),
    )

    result = run_evaluation(
        sample_df=sample,
        workflow=GenericWorkflow(),
        config=_config(tmp_path, total_records=2),
        run_id="agent-execution-duration",
        artifact_base_dir=tmp_path / "runs",
    )

    run_dir = Path(result.run_dir)
    assert result.metrics["runtime"]["agent_execution_duration_s"] == 3.75
    assert result.metrics["runtime"]["real_execution_duration_s"] == 12.5

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    trace_summary = json.loads((run_dir / "trace_summary.json").read_text(encoding="utf-8"))
    assert metrics["runtime"]["agent_execution_duration_s"] == 3.75
    assert metrics["runtime"]["real_execution_duration_s"] == 12.5
    assert trace_summary["runtime"]["agent_execution_duration_s"] == 3.75
    assert trace_summary["runtime"]["real_execution_duration_s"] == 12.5

    compiled_rows = pd.read_csv(run_dir / "compiled_results.csv", encoding="utf-8-sig")
    duration_row = compiled_rows[
        (compiled_rows["section"] == "runtime_metrics")
        & (compiled_rows["metric_name"] == "agent_execution_duration_s")
    ]
    assert duration_row["metric_value"].iloc[0] == 3.75
    real_duration_row = compiled_rows[
        (compiled_rows["section"] == "runtime_metrics")
        & (compiled_rows["metric_name"] == "real_execution_duration_s")
    ]
    assert real_duration_row["metric_value"].iloc[0] == 12.5
    assert "Cumulative case execution duration seconds: 3.750000" in (
        run_dir / "metrics.md"
    ).read_text(encoding="utf-8")
    assert "Real execution duration seconds: 12.500000" in (
        run_dir / "metrics.md"
    ).read_text(encoding="utf-8")
    assert "Cumulative case execution duration seconds | 3.750000" in (
        run_dir / "compiled_results.md"
    ).read_text(encoding="utf-8")
    assert "Real execution duration seconds | 12.500000" in (
        run_dir / "compiled_results.md"
    ).read_text(encoding="utf-8")
    runtime_svg = (run_dir / "runtime_metrics.svg").read_text(
        encoding="utf-8"
    )
    assert "Cumulative case execution (s)" in runtime_svg
    assert "Real execution (s)" in runtime_svg


def test_generic_evaluator_rejects_empty_samples_before_creating_artifacts(tmp_path):
    with pytest.raises(ValueError, match="at least one evaluation record"):
        run_evaluation(
            sample_df=pd.DataFrame(columns=["ID", "users", "text", "sentiment", "time"]),
            workflow=GenericWorkflow(),
            config=_config(tmp_path, total_records=0),
            run_id="empty-sample",
            artifact_base_dir=tmp_path / "runs",
        )

    assert not (tmp_path / "runs").exists()


def test_generic_evaluator_propagates_canonical_id_to_results_and_wrong_id_file(tmp_path):
    sample = pd.DataFrame(
        [{"ID": 99, "users": 7, "text": "indicator", "sentiment": "Ideation", "time": 300}]
    )
    result = run_evaluation(
        sample_df=sample,
        workflow=GenericWorkflow(),
        config=_config(tmp_path, total_records=1),
        run_id="generic-canonical-id",
        artifact_base_dir=tmp_path / "runs",
    )

    run_dir = Path(result.run_dir)
    assert (run_dir / "misclassified_ids.txt").read_text(encoding="utf-8") == "99\n"
    complete_results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    assert complete_results["results"][0]["ID"] == 99
    assert complete_results["results"][0]["sample_id"] == 99
    assert complete_results["results"][0]["users"] == 7
    assert complete_results["results"][0]["time"] == 300


def test_generic_evaluator_persists_dynamic_agent_outputs_without_metrics(tmp_path):
    sample = pd.DataFrame(
        [{"ID": 10, "users": 2, "text": "indicator", "sentiment": "Indicator", "time": 100}]
    )
    config = _config(tmp_path, total_records=1)

    result = run_evaluation(
        sample_df=sample,
        workflow=GenericWorkflowWithAgentOutputs(),
        config=config,
        run_id="generic-agent-metrics",
        artifact_base_dir=tmp_path / "runs",
    )

    run_dir = Path(result.run_dir)
    predictions = pd.read_csv(run_dir / "predictions.csv", encoding="utf-8-sig")
    assert json.loads(predictions.loc[0, "agent_outputs"]) == {
        "router": {"decision": "Indicator"}
    }
    assert "specialist_answers" not in predictions.columns
    assert "agent_metrics" not in result.metrics
    assert not (run_dir / "agent_metrics.csv").exists()


def test_evaluator_writes_complete_artifacts_for_fake_model(tmp_path):
    sample_df = _sample_df()
    config = _config(tmp_path)

    result = run_framework_1_evaluation(
        sample_df=sample_df,
        model=EvaluatorFakeModel(),
        config=config,
        run_id="test-evaluator-complete",
        artifact_base_dir=tmp_path / "runs",
    )

    run_dir = Path(result.run_dir)
    assert result.status == "completed"
    assert result.processed_records == 4
    assert not (run_dir / "model_metrics.csv").exists()
    assert not (run_dir / "label_metrics.csv").exists()
    assert (run_dir / "trace_summary.json").exists()
    prompt_snapshot = json.loads(
        (run_dir / "first_completion_input_prompts.json").read_text(
            encoding="utf-8"
        )
    )
    assert prompt_snapshot["workflow_name"] == "framework_1_parallel_no_evidence"
    assert prompt_snapshot["sample_id"] == "10"
    assert {agent["role"] for agent in prompt_snapshot["agents"]} == {
        "Indicator_specialist",
        "Ideation_specialist",
        "Behavior_specialist",
        "Attempt_specialist",
        "Unsupportive_specialist",
        "judge",
    }
    assert all(
        [message["role"] for message in agent["messages"]] == ["system", "user"]
        for agent in prompt_snapshot["agents"]
    )
    specialist_agents = [
        agent for agent in prompt_snapshot["agents"] if agent["role"] != "judge"
    ]
    judge_agent = next(
        agent for agent in prompt_snapshot["agents"] if agent["role"] == "judge"
    )
    assert all(
        "indicator text" in agent["messages"][1]["content"]
        for agent in specialist_agents
    )
    assert "Specialist answers" in judge_agent["messages"][1]["content"]
    assert (run_dir / "llm_calls.jsonl").exists()
    assert (run_dir / "traces.jsonl").exists()
    assert (run_dir / "compiled_results.csv").exists()
    assert (run_dir / "compiled_results.md").exists()
    assert not (run_dir / "agent_metrics.csv").exists()
    assert (run_dir / "suicidal_intent_metrics.csv").exists()
    assert (run_dir / "positive_subclassification_metrics.csv").exists()
    assert (run_dir / "results.json").exists()
    assert (run_dir / "misclassified_ids.txt").read_text(encoding="utf-8") == "11\n12\n13\n"
    assert (run_dir / "suicidal_intent_misclassified_ids.txt").read_text(encoding="utf-8") == ""
    assert not (run_dir / "classification_metrics.svg").exists()
    assert (run_dir / "suicidal_intent_metrics.svg").exists()
    assert (run_dir / "runtime_metrics.svg").exists()
    persisted_calls = [
        json.loads(line)
        for line in (run_dir / "llm_calls.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert all("_input_messages" not in call for call in persisted_calls)
    assert result.metrics["compiled_reports"] == {
        "compiled_csv": str(run_dir / "compiled_results.csv"),
        "compiled_markdown": str(run_dir / "compiled_results.md"),
    }
    assert result.metrics["artifact_status"]["compiled_results.csv"] is True
    assert result.metrics["artifact_status"]["compiled_results.md"] is True
    assert all(result.metrics["artifact_status"].values())
    assert result.metrics["suicidal_intent"]["accuracy"] == 1.0
    assert result.metrics["suicidal_intent"]["has_both_ground_truth_classes"] is False
    assert result.metrics["positive_subclassification"]["support"] == 4
    assert result.metrics["positive_subclassification"]["accuracy"] == 0.25
    assert "agent_metrics" not in result.metrics
    assert "specialist_metrics" not in result.metrics
    assert set(result.metrics["classification"]) == {
        "confusion_matrix",
        "total",
        "failed_records",
        "failure_rate",
    }
    assert set(result.metrics["classification"]["confusion_matrix"]) == set(FINAL_LABELS)
    assert "## Per-Class Metrics" not in (run_dir / "metrics.md").read_text(encoding="utf-8")

    predictions = pd.read_csv(run_dir / "predictions.csv", encoding="utf-8-sig")
    assert {
        "specialist_answers",
        "specialist_outputs",
        "indicator_answer",
        "ideation_answer",
        "behavior_answer",
        "attempt_answer",
        "unsupportive_answer",
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
        "Unsupportive": False,
    }
    assert predictions.loc[0, "indicator_answer"] == True
    assert predictions.loc[0, "ideation_answer"] == False
    specialist_outputs = json.loads(predictions.loc[0, "specialist_outputs"])
    assert specialist_outputs["Indicator"] == {"answer": True}
    assert predictions["ground_truth_suicidal_intent"].tolist() == [True, True, True, True]
    assert predictions["suicidal_intent"].tolist() == [True, True, True, True]

    complete_results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    assert complete_results["schema_version"] == "1.1"
    assert complete_results["results"][0]["final_output"] == {
        "label_name": "Indicator",
        "suicidal_intent": True,
    }
    run_config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    assert run_config["agent_configuration"]["judge"] == {
        "agent_name": "Final Judge",
        "model_name": "test-model",
        "temperature": 0.0,
    }
    assert run_config["agent_configuration"]["Unsupportive_specialist"] == {
        "agent_name": "Unsupportive Agent",
        "model_name": "test-model",
        "temperature": 0.0,
    }


def test_framework_1_parallel_no_evidence_wrapper_routes_unsupportive_role_override(tmp_path):
    sample_df = pd.DataFrame(
        [{"source_row": 10, "text": "indicator text", "sentiment": "Indicator"}]
    )
    config = _config(
        tmp_path,
        total_records=1,
        agent_configuration={
            "Unsupportive_specialist": {
                "agent_name": "Unsupportive Agent",
                "model_name": "unsupportive-override",
                "temperature": 0.0,
            }
        },
    )

    result = run_framework_1_evaluation(
        sample_df=sample_df,
        model=EvaluatorFakeModel(),
        agent_models={"Unsupportive_specialist": EvaluatorFakeModel()},
        config=config,
        run_id="test-valid-framework_1_parallel_no_evidence-override",
        artifact_base_dir=tmp_path / "runs",
    )

    calls = [
        json.loads(line)
        for line in (Path(result.run_dir) / "llm_calls.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    unsupportive_call = next(call for call in calls if call["role"] == "Unsupportive_specialist")
    assert unsupportive_call["model_name"] == "unsupportive-override"
    assert "specialist_metrics" not in result.metrics


def test_framework_1_parallel_no_evidence_wrapper_rejects_unknown_roles(tmp_path):
    config = _config(
        tmp_path,
        total_records=1,
        agent_configuration={"router": {"model_name": "router-model", "temperature": 0.0}},
    )

    with pytest.raises(ValueError, match="Unknown framework_1_parallel_no_evidence role overrides"):
        run_framework_1_evaluation(
            sample_df=_sample_df().head(1),
            model=EvaluatorFakeModel(),
            config=config,
            artifact_base_dir=tmp_path / "runs",
        )


def test_evaluator_stops_early_on_abnormal_request_errors(tmp_path):
    sample_df = _sample_df()
    config = _config(tmp_path, model_name="failing-model")

    result = run_framework_1_evaluation(
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
    assert result.metrics["coverage"]["complete"] is False
    assert result.metrics["coverage"]["output_records"] == 1
    assert "failed_llm_attempts=5" in result.stop_reason
    assert trace_summary["evaluation_status"] == "stopped_early"
    assert trace_summary["runtime"]["failed_llm_attempts"] == 5


def test_evaluator_accepts_unsupportive_and_computes_negative_intent_metrics(tmp_path):
    sample_df = pd.DataFrame(
        [{"source_row": 20, "text": "neutral text", "sentiment": "Unsupportive"}]
    )
    config = _config(tmp_path, total_records=1)

    result = run_framework_1_evaluation(
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
    assert result.metrics["classification"]["confusion_matrix"]["Unsupportive"]["Unsupportive"] == 1
    assert result.metrics["suicidal_intent"]["accuracy"] == 1.0
    assert result.metrics["suicidal_intent"]["positive_class_counts"] == {
        "tp": 0,
        "tn": 1,
        "fp": 0,
        "fn": 0,
    }
    binary_metrics = pd.read_csv(
        Path(result.run_dir) / "suicidal_intent_metrics.csv",
        encoding="utf-8-sig",
    )
    assert len(binary_metrics) == 1
    assert binary_metrics.loc[0, ["positive_support", "tp", "fn"]].tolist() == [0, 0, 0]
    assert binary_metrics.loc[0, ["negative_support", "tn", "fp"]].tolist() == [1, 1, 0]


@pytest.mark.parametrize(
    ("runner", "mode", "workflow_name"),
    [
        (run_framework_1_evaluation, "bool", "framework_1_parallel_no_evidence"),
        (run_framework_2_evaluation, "evidence", "framework_2_parallel_with_evidence"),
    ],
)
def test_parallel_reports_derive_suicidal_intent_from_label_only_judge(
    tmp_path,
    runner,
    mode,
    workflow_name,
):
    sample_df = pd.DataFrame(
        [
            {
                "source_row": index,
                "text": f"EXPECTED={label}",
                "sentiment": label,
            }
            for index, label in enumerate(FINAL_LABELS, start=1)
        ]
    )
    result = runner(
        sample_df=sample_df,
        model=LabelOnlyJudgeFakeModel(),
        config=_config(tmp_path, total_records=5, mode=mode),
        run_id=f"test-{workflow_name}-derived-intent",
        artifact_base_dir=tmp_path / "runs",
    )
    run_dir = Path(result.run_dir)

    judge_calls = [
        json.loads(line)
        for line in (run_dir / "llm_calls.jsonl").read_text(encoding="utf-8").splitlines()
        if json.loads(line)["role"] == "judge"
    ]
    assert len(judge_calls) == 5
    assert all(set(call["parsed"]) == {"label_name"} for call in judge_calls)

    predictions = pd.read_csv(run_dir / "predictions.csv", encoding="utf-8-sig")
    assert predictions["prediction"].tolist() == list(FINAL_LABELS)
    assert predictions["suicidal_intent"].tolist() == [True, True, True, True, False]
    assert all(
        json.loads(value)["judge"] == {"label_name": label}
        for value, label in zip(predictions["agent_outputs"], FINAL_LABELS)
    )

    results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    assert [item["final_output"]["suicidal_intent"] for item in results["results"]] == [
        True,
        True,
        True,
        True,
        False,
    ]
    assert result.metrics["suicidal_intent"]["positive_class_counts"] == {
        "tp": 4,
        "tn": 1,
        "fp": 0,
        "fn": 0,
    }

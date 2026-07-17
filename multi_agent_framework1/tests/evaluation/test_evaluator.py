import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from suicidal_multi_agent.core.labels import VALID_LABELS
from suicidal_multi_agent.evaluation.evaluator import (
    EvaluationConfig,
    load_evaluation_sample,
    run_evaluation,
    run_workflow_evaluation,
)


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


def test_generic_evaluator_keeps_agent_outputs_and_metrics_optional(tmp_path):
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
    assert result.metrics["classification"]["accuracy"] == 1.0
    assert result.metrics["suicidal_intent"]["accuracy"] == 1.0
    assert result.metrics["runtime"]["call_count"] == 3
    assert result.metrics["runtime"]["tokens"]["total"] == 15
    assert result.metrics["agent_metrics"] == []
    assert "specialist_metrics" not in result.metrics
    assert "specialist_answers" not in predictions.columns
    assert "agent_outputs" not in predictions.columns
    assert predictions["ID"].tolist() == [10, 20]
    assert predictions["users"].tolist() == [2, 3]
    assert predictions["sample_id"].tolist() == [10, 20]
    assert not (run_dir / "agent_metrics.csv").exists()
    assert result.metrics["optional_artifact_status"]["agent_metrics.csv"] is False
    assert all(result.metrics["artifact_status"].values())
    compiled_markdown = (run_dir / "compiled_results.md").read_text(encoding="utf-8")
    assert "## Per-Agent Metrics" not in compiled_markdown

    run_config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    assert set(run_config["agent_configuration"]) == {"router", "critic"}


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


def test_generic_evaluator_accepts_dynamic_agent_outputs_and_metrics(tmp_path):
    sample = pd.DataFrame(
        [{"ID": 10, "users": 2, "text": "indicator", "sentiment": "Indicator", "time": 100}]
    )
    config = _config(tmp_path, total_records=1)

    def agent_metrics(rows, config, classification, suicidal_intent):
        return {
            "router_metrics": {"observed_routes": len(rows)},
            "agent_metrics": [
                {
                    "agent_name": "Router",
                    "model_name": "router-model",
                    "task": "route selection",
                    "accuracy": 1.0,
                    "f1_score": 1.0,
                    "precision": 1.0,
                    "recall": 1.0,
                }
            ],
        }

    result = run_evaluation(
        sample_df=sample,
        workflow=GenericWorkflowWithAgentOutputs(),
        config=config,
        run_id="generic-agent-metrics",
        artifact_base_dir=tmp_path / "runs",
        agent_metrics_provider=agent_metrics,
    )

    run_dir = Path(result.run_dir)
    predictions = pd.read_csv(run_dir / "predictions.csv", encoding="utf-8-sig")
    assert json.loads(predictions.loc[0, "agent_outputs"]) == {
        "router": {"decision": "Indicator"}
    }
    assert "specialist_answers" not in predictions.columns
    assert result.metrics["router_metrics"] == {"observed_routes": 1}
    assert result.metrics["optional_artifact_status"]["agent_metrics.csv"] is True


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


def test_framework1_wrapper_routes_valid_role_overrides(tmp_path):
    sample_df = pd.DataFrame(
        [{"source_row": 10, "text": "indicator text", "sentiment": "Indicator"}]
    )
    config = _config(
        tmp_path,
        total_records=1,
        agent_configuration={
            "Indicator_specialist": {
                "agent_name": "Indicator Agent",
                "model_name": "indicator-override",
                "temperature": 0.0,
            }
        },
    )

    result = run_workflow_evaluation(
        sample_df=sample_df,
        model=EvaluatorFakeModel(),
        agent_models={"Indicator_specialist": EvaluatorFakeModel()},
        config=config,
        run_id="test-valid-framework1-override",
        artifact_base_dir=tmp_path / "runs",
    )

    assert result.metrics["specialist_metrics"]["Indicator"]["model_name"] == "indicator-override"


def test_framework1_wrapper_rejects_unknown_roles(tmp_path):
    config = _config(
        tmp_path,
        total_records=1,
        agent_configuration={"router": {"model_name": "router-model", "temperature": 0.0}},
    )

    with pytest.raises(ValueError, match="Unknown framework1 role overrides"):
        run_workflow_evaluation(
            sample_df=_sample_df().head(1),
            model=EvaluatorFakeModel(),
            config=config,
            artifact_base_dir=tmp_path / "runs",
        )


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

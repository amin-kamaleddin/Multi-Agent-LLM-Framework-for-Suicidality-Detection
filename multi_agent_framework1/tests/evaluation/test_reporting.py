import json
from pathlib import Path

import pandas as pd

from suicidal_multi_agent.core.labels import VALID_LABELS
from suicidal_multi_agent.evaluation.evaluator import EvaluationConfig, run_workflow_evaluation
from suicidal_multi_agent.evaluation.reporting import compile_run_results


class ReportingFakeModel:
    def invoke(self, messages):
        system = messages[0]["content"]
        if "final judge" in system.lower():
            return json.dumps({"label_name": "Indicator", "suicidal_intent": True})
        for label in VALID_LABELS:
            if f"Target label: {label}" in system:
                return json.dumps({"answer": label == "Indicator"})
        raise AssertionError("Unexpected prompt")


def test_compile_run_results_writes_csv_and_markdown_tables(tmp_path):
    sample_df = pd.DataFrame(
        [
            {"source_row": 1, "text": "indicator", "sentiment": "Indicator"},
            {"source_row": 2, "text": "ideation", "sentiment": "Ideation"},
        ]
    )
    config = EvaluationConfig(
        model_name="reporting-model",
        requested_model_name="reporting-model",
        sample_path=str(tmp_path / "sample.csv"),
        dataset_audit_path=None,
        max_retries=0,
        max_completion_tokens=32,
        temperature=0.0,
        total_records=2,
    )
    result = run_workflow_evaluation(
        sample_df=sample_df,
        model=ReportingFakeModel(),
        config=config,
        run_id="test-reporting",
        artifact_base_dir=tmp_path / "runs",
    )

    outputs = compile_run_results(result.run_dir)

    compiled_csv = Path(outputs["compiled_csv"])
    compiled_markdown = Path(outputs["compiled_markdown"])
    assert compiled_csv.exists()
    assert compiled_markdown.exists()

    rows = pd.read_csv(compiled_csv, encoding="utf-8-sig")
    assert {
        "model_metrics",
        "per_label_metrics",
        "agent_metrics",
        "confusion_matrix",
        "suicidal_intent_metrics",
        "suicidal_intent_confusion_matrix",
        "runtime_metrics",
    } <= set(rows["section"])
    runtime_rows = rows[rows["section"] == "runtime_metrics"].set_index("metric_name")
    assert "avg_e2e_completion_tps" in runtime_rows.index

    predictions = pd.read_csv(Path(result.run_dir) / "predictions.csv", encoding="utf-8-sig")
    assert "specialist_answers" in predictions.columns
    assert "indicator_answer" in predictions.columns

    markdown = compiled_markdown.read_text(encoding="utf-8")
    assert "## Model Metrics" in markdown
    assert "## Confusion Matrix" in markdown
    assert "## Per-Agent Metrics" in markdown
    assert "## Suicidal-Intent Positive-Class Grouping Metrics" in markdown
    assert "Average end-to-end completion TPS" in markdown
    assert "reporting-model" in markdown

import json

import pandas as pd
import pytest

from suicidal_multi_agent import cli


def test_custom_samples_do_not_inherit_default_dataset_audit():
    assert cli._evaluation_audit_path("custom-sample.csv", None) is None
    assert cli._evaluation_audit_path("custom-sample.csv", "custom-audit.json") == "custom-audit.json"
    assert cli._evaluation_audit_path(str(cli.DEFAULT_EVALUATION_SAMPLE_PATH), None) == str(
        cli.DEFAULT_EVALUATION_AUDIT_PATH
    )


def test_cli_rejects_nonpositive_limits_and_uses_safe_token_default(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    monkeypatch.delenv("MAX_COMPLETION_TOKENS", raising=False)
    parser = cli.build_parser()

    assert parser.parse_args(["run"]).max_completion_tokens == cli.DEFAULT_MAX_COMPLETION_TOKENS
    assert parser.parse_args(["evaluate"]).workflow == "framework1"
    assert parser.parse_args(["evaluate"]).max_completion_tokens is None
    assert (
        cli._workflow_setting(
            parser.parse_args(["evaluate"]),
            {},
            "max_completion_tokens",
            cli.DEFAULT_MAX_COMPLETION_TOKENS,
        )
        == cli.DEFAULT_MAX_COMPLETION_TOKENS
    )
    assert (
        cli._workflow_setting(
            parser.parse_args(["evaluate", "--mode", "bool"]),
            {"mode": "reasoned"},
            "mode",
            "unused",
        )
        == "bool"
    )
    with pytest.raises(SystemExit):
        parser.parse_args(["evaluate", "--limit", "0"])
    with pytest.raises(SystemExit):
        parser.parse_args(["evaluate", "--limit", "-1"])
    with pytest.raises(SystemExit):
        parser.parse_args(["evaluate", "--workflow", "missing-workflow"])


def test_list_workflows_reports_valid_cli_names(capsys):
    assert cli.main(["list-workflows"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert [workflow["name"] for workflow in payload["workflows"]] == [
        "framework1",
        "framework2",
        "workflow2_hierarchical",
        "workflow3_hierarchical_reviewer",
    ]


@pytest.mark.parametrize(
    "workflow_name",
    ["workflow2_hierarchical", "workflow3_hierarchical_reviewer"],
)
def test_hierarchical_workflow_fake_llm_smoke(tmp_path, workflow_name):
    sample_path = tmp_path / "sample.csv"
    pd.DataFrame(
        [
            {
                "ID": 17,
                "users": 4,
                "text": "A suicide-related discussion about another person.",
                "sentiment": "Indicator",
                "time": 123,
            }
        ]
    ).to_csv(sample_path, index=False, encoding="utf-8-sig")
    artifact_dir = tmp_path / "runs"

    assert (
        cli.main(
            [
                "evaluate",
                "--workflow",
                workflow_name,
                "--fake-llm",
                "--mode",
                "bool",
                "--sample-path",
                str(sample_path),
                "--artifact-base-dir",
                str(artifact_dir),
                "--run-id",
                "smoke",
            ]
        )
        == 0
    )
    run_dirs = list(artifact_dir.iterdir())
    assert len(run_dirs) == 1
    metrics = json.loads((run_dirs[0] / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["evaluation_status"] == "completed"
    assert metrics["processed_records"] == 1
    assert metrics["classification"]["failed_records"] == 0
    assert metrics["runtime"]["failed_llm_attempts"] == 0


def test_shared_cli_uses_registered_workflow_config_dataset_and_output_dir(tmp_path):
    sample_path = tmp_path / "sample.csv"
    pd.DataFrame(
        [
            {
                "ID": 17,
                "users": 4,
                "text": "test record",
                "sentiment": "Indicator",
                "time": 123,
            }
        ]
    ).to_csv(sample_path, index=False, encoding="utf-8-sig")
    workflow_config_path = tmp_path / "framework1.json"
    workflow_config = {
        "fake_llm": True,
        "mode": "reasoned",
        "max_completion_tokens": 64,
        "max_retries": 1,
        "agents": {
            "judge": {
                "model": "judge-fake",
                "temperature": 0.0,
            }
        },
    }
    workflow_config_path.write_text(json.dumps(workflow_config), encoding="utf-8")
    artifact_base_dir = tmp_path / "custom-runs"

    exit_code = cli.main(
        [
            "evaluate",
            "--workflow",
            "framework1",
            "--workflow-config",
            str(workflow_config_path),
            "--sample-path",
            str(sample_path),
            "--artifact-base-dir",
            str(artifact_base_dir),
            "--run-id",
            "shared-cli",
        ]
    )

    assert exit_code == 0
    run_dirs = list(artifact_base_dir.glob("*_shared-cli"))
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    run_config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert run_config["workflow_name"] == "framework1"
    assert run_config["workflow_config"] == workflow_config
    assert run_config["mode"] == "reasoned"
    assert run_config["max_completion_tokens"] == 64
    assert run_config["agent_configuration"]["judge"]["model_name"] == "judge-fake"
    assert metrics["workflow_name"] == "framework1"
    assert "- Workflow: `framework1`" in (run_dir / "compiled_results.md").read_text(
        encoding="utf-8"
    )
    assert (run_dir / "compiled_results.csv").exists()
    assert (run_dir / "compiled_results.md").exists()


def test_framework2_cli_smoke_persists_evidence_without_specialist_metrics(tmp_path):
    sample_path = tmp_path / "sample.csv"
    pd.DataFrame(
        [
            {
                "ID": 18,
                "users": 5,
                "text": "ordinary test record",
                "sentiment": "Unsupportive",
                "time": 124,
            }
        ]
    ).to_csv(sample_path, index=False, encoding="utf-8-sig")
    artifact_base_dir = tmp_path / "framework2-runs"

    exit_code = cli.main(
        [
            "evaluate",
            "--workflow",
            "framework2",
            "--fake-llm",
            "--sample-path",
            str(sample_path),
            "--artifact-base-dir",
            str(artifact_base_dir),
            "--run-id",
            "evidence-smoke",
        ]
    )

    assert exit_code == 0
    run_dir = next(artifact_base_dir.glob("*_evidence-smoke"))
    run_config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    predictions = pd.read_csv(run_dir / "predictions.csv", encoding="utf-8-sig")
    collector_outputs = json.loads(predictions.loc[0, "collector_outputs"])
    assert run_config["mode"] == "evidence"
    assert set(collector_outputs) == {
        "Indicator",
        "Ideation",
        "Behavior",
        "Attempt",
        "Unsupportive",
    }
    assert "specialist_answers" not in predictions.columns
    assert not (run_dir / "agent_metrics.csv").exists()

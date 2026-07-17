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
    assert parser.parse_args(["evaluate"]).max_completion_tokens == cli.DEFAULT_MAX_COMPLETION_TOKENS
    with pytest.raises(SystemExit):
        parser.parse_args(["evaluate", "--limit", "0"])
    with pytest.raises(SystemExit):
        parser.parse_args(["evaluate", "--limit", "-1"])

"""Command-line entrypoint for the naive multi-agent workflow."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

from .data.dataset import DEFAULT_DATASET_PATH, DEFAULT_PROCESSED_DIR, audit_dataset, prepare_dataset_files, sample_balanced
from .evaluation.artifacts import ArtifactWriter
from .evaluation.evaluator import (
    DEFAULT_EVALUATION_AUDIT_PATH,
    DEFAULT_EVALUATION_SAMPLE_PATH,
    EvaluationConfig,
    load_evaluation_sample,
    run_workflow_evaluation,
)
from .evaluation.metrics import compute_classification_metrics, summarize_runtime
from .evaluation.reporting import compile_run_results
from .workflow.models import (
    DeterministicFakeModel,
    build_real_model,
    openai_base_url,
    resolve_model_name_for_base_url,
)
from .workflow.prompts import WORKFLOW_MODE_NAMES
from .workflow.registry import WORKFLOW_REGISTRY, WorkflowRunRequest


DEFAULT_MAX_COMPLETION_TOKENS = 256


def _ensure_help_log(argv: list[str]) -> None:
    if "--help" in argv or "-h" in argv:
        log_dir = Path("logs/tests")
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "cli_help_smoke.log").write_text(
            "CLI help invoked. Capture stdout from the command alongside this smoke marker.\n",
            encoding="utf-8",
        )


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-path", default=str(DEFAULT_DATASET_PATH), help="Path to the active anonymized CSV.")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed.")
    parser.add_argument("--sample-size", type=int, default=500, help="Balanced sample size. Must divide by 4.")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be zero or greater")
    return parsed


def _load_workflow_config(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Workflow config must be a JSON object: {config_path}")
    return payload


def _workflow_setting(
    args: argparse.Namespace,
    workflow_config: dict[str, Any],
    name: str,
    default: Any,
) -> Any:
    cli_value = getattr(args, name)
    return cli_value if cli_value is not None else workflow_config.get(name, default)


def _evaluation_audit_path(sample_path: str, requested_audit_path: str | None) -> str | None:
    """Use the paired default audit only with the default evaluation sample."""
    if requested_audit_path is not None:
        return requested_audit_path
    if Path(sample_path).resolve() == DEFAULT_EVALUATION_SAMPLE_PATH.resolve():
        return str(DEFAULT_EVALUATION_AUDIT_PATH)
    return None


def _args_config(args: argparse.Namespace, command_name: str) -> dict[str, Any]:
    config = {key: value for key, value in vars(args).items() if not callable(value)}
    config["command"] = command_name
    return config


def _write_setup_artifacts(args: argparse.Namespace, command_name: str) -> tuple[ArtifactWriter, pd.DataFrame, pd.DataFrame]:
    writer = ArtifactWriter()
    writer.write_json("run_config.json", _args_config(args, command_name))
    _, dataset1, audit = audit_dataset(args.data_path)
    writer.write_json("dataset_audit.json", audit)
    sampled = sample_balanced(dataset1, sample_size=args.sample_size, seed=args.seed)
    writer.write_dataframe("sampled_dataset.csv", sampled)
    return writer, dataset1, sampled


def cmd_audit(args: argparse.Namespace) -> int:
    writer = ArtifactWriter()
    writer.write_json("run_config.json", _args_config(args, "audit"))
    _, _, audit = audit_dataset(args.data_path)
    writer.write_json("dataset_audit.json", audit)
    writer.write_dataframe("sampled_dataset.csv", pd.DataFrame())
    writer.write_dataframe("predictions.csv", pd.DataFrame())
    empty_metrics = {"classification": compute_classification_metrics([], []), "runtime": summarize_runtime([])}
    writer.write_json("metrics.json", empty_metrics)
    writer.write_metrics_md(empty_metrics)
    print(json.dumps({"run_dir": str(writer.run_dir), "dataset_audit": audit}, indent=2))
    return 0


def cmd_sample(args: argparse.Namespace) -> int:
    writer, _, sampled = _write_setup_artifacts(args, "sample")
    writer.write_dataframe("predictions.csv", pd.DataFrame())
    empty_metrics = {"classification": compute_classification_metrics([], []), "runtime": summarize_runtime([])}
    writer.write_json("metrics.json", empty_metrics)
    writer.write_metrics_md(empty_metrics)
    counts = sampled["sentiment"].value_counts().to_dict()
    print(json.dumps({"run_dir": str(writer.run_dir), "sample_rows": len(sampled), "label_counts": counts}, indent=2))
    return 0


def cmd_prepare_data(args: argparse.Namespace) -> int:
    payload = prepare_dataset_files(
        data_path=args.data_path,
        output_dir=args.output_dir,
        sample_size=args.sample_size,
        seed=args.seed,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    _, dataset1, audit = audit_dataset(args.data_path)
    sampled = sample_balanced(dataset1, sample_size=args.sample_size, seed=args.seed)
    requested_model_name = "deterministic-fake-llm" if args.fake_llm else args.model
    model_name = (
        "deterministic-fake-llm"
        if args.fake_llm
        else resolve_model_name_for_base_url(args.model, openai_base_url())
    )
    model = (
        DeterministicFakeModel()
        if args.fake_llm
        else build_real_model(args.model, args.temperature, args.max_completion_tokens)
    )
    config = EvaluationConfig(
        model_name=model_name,
        requested_model_name=requested_model_name,
        sample_path=args.data_path,
        dataset_audit_path=None,
        max_retries=args.max_retries,
        max_completion_tokens=args.max_completion_tokens,
        temperature=args.temperature,
        total_records=len(sampled),
        mode=args.mode,
        command="run",
    )
    result = run_workflow_evaluation(
        sample_df=sampled,
        model=model,
        config=config,
        dataset_audit={**audit, "evaluation_sample_path": args.data_path},
    )
    print(json.dumps(result.__dict__, indent=2))
    return 2 if result.status == "stopped_early" else 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    workflow_config = _load_workflow_config(args.workflow_config)
    configured_model = str(_workflow_setting(args, workflow_config, "model", "gpt-4o"))
    temperature = float(
        _workflow_setting(
            args,
            workflow_config,
            "temperature",
            float(os.environ.get("OPENAI_TEMPERATURE", "0") or 0),
        )
    )
    max_completion_tokens = int(
        _workflow_setting(
            args,
            workflow_config,
            "max_completion_tokens",
            int(
                os.environ.get("MAX_COMPLETION_TOKENS", str(DEFAULT_MAX_COMPLETION_TOKENS))
                or DEFAULT_MAX_COMPLETION_TOKENS
            ),
        )
    )
    max_retries = int(
        _workflow_setting(
            args,
            workflow_config,
            "max_retries",
            int(os.environ.get("MAX_RETRIES", "3") or 3),
        )
    )
    mode = str(_workflow_setting(args, workflow_config, "mode", "bool"))
    fake_llm = _workflow_setting(args, workflow_config, "fake_llm", False)
    if not isinstance(fake_llm, bool):
        raise ValueError("workflow-config 'fake_llm' must be a JSON boolean")
    if max_completion_tokens <= 0:
        raise ValueError("max_completion_tokens must be greater than zero")
    if max_retries < 0:
        raise ValueError("max_retries must be zero or greater")

    requested_model_name = "deterministic-fake-llm" if fake_llm else configured_model
    resolved_model_name = (
        "deterministic-fake-llm"
        if fake_llm
        else resolve_model_name_for_base_url(configured_model, openai_base_url())
    )
    sample_df = load_evaluation_sample(args.sample_path, limit=args.limit)
    config = EvaluationConfig(
        model_name=resolved_model_name,
        requested_model_name=requested_model_name,
        sample_path=args.sample_path,
        dataset_audit_path=_evaluation_audit_path(
            args.sample_path,
            args.dataset_audit_path,
        ),
        max_retries=max_retries,
        max_completion_tokens=max_completion_tokens,
        temperature=temperature,
        total_records=len(sample_df),
        limit=args.limit,
        mode=mode,
        workflow_name=args.workflow,
        workflow_config=workflow_config,
        error_stop_failed_llm_attempts=args.error_stop_failed_llm_attempts,
        error_stop_failed_records=args.error_stop_failed_records,
        error_stop_rate=args.error_stop_rate,
        error_stop_min_records=args.error_stop_min_records,
    )
    registration = WORKFLOW_REGISTRY.get(args.workflow)
    result = registration.runner(
        WorkflowRunRequest(
            sample_df=sample_df,
            config=config,
            workflow_config=workflow_config,
            fake_llm=fake_llm,
            run_id=args.run_id,
            artifact_base_dir=args.artifact_base_dir,
        )
    )
    print(json.dumps(result.__dict__, indent=2))
    return 2 if result.status == "stopped_early" else 0


def cmd_compile_results(args: argparse.Namespace) -> int:
    outputs = compile_run_results(args.run_dir)
    print(json.dumps(outputs, indent=2))
    return 0


def cmd_list_workflows(_args: argparse.Namespace) -> int:
    print(json.dumps({"workflows": WORKFLOW_REGISTRY.describe()}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Naive LangGraph multi-agent suicide-risk classifier.")
    subparsers = parser.add_subparsers(dest="command")

    audit = subparsers.add_parser("audit", help="Write dataset audit artifacts.")
    _add_common_args(audit)
    audit.set_defaults(func=cmd_audit)

    sample = subparsers.add_parser("sample", help="Write Dataset1 audit and balanced sample artifacts.")
    _add_common_args(sample)
    sample.set_defaults(func=cmd_sample)

    prepare = subparsers.add_parser("prepare-data", help="Write persistent cleaned Dataset1 and balanced sample CSVs.")
    _add_common_args(prepare)
    prepare.add_argument("--output-dir", default=str(DEFAULT_PROCESSED_DIR), help="Directory for processed dataset files.")
    prepare.set_defaults(func=cmd_prepare_data)

    run = subparsers.add_parser("run", help="Run the multi-agent workflow over a balanced sample.")
    _add_common_args(run)
    run.add_argument("--fake-llm", action="store_true", help="Use a deterministic fake model for artifact smoke tests.")
    run.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "openai/gpt-4o-mini"), help="Chat model name.")
    run.add_argument("--temperature", type=float, default=float(os.environ.get("OPENAI_TEMPERATURE", "0") or 0))
    run.add_argument(
        "--max-completion-tokens",
        type=int,
        default=int(
            os.environ.get("MAX_COMPLETION_TOKENS", str(DEFAULT_MAX_COMPLETION_TOKENS))
            or DEFAULT_MAX_COMPLETION_TOKENS
        ),
    )
    run.add_argument("--max-retries", type=int, default=int(os.environ.get("MAX_RETRIES", "3") or 3))
    run.add_argument("--mode", choices=WORKFLOW_MODE_NAMES, default="bool", help="Prompt/schema mode for the shared graph.")
    run.set_defaults(func=cmd_run)

    evaluate = subparsers.add_parser(
        "evaluate",
        help="Evaluate a registered workflow on a frozen sampled dataset.",
    )
    evaluate.add_argument(
        "--workflow",
        choices=WORKFLOW_REGISTRY.names(),
        default="framework1",
        help="Registered workflow name. Use list-workflows to inspect available names.",
    )
    evaluate.add_argument(
        "--workflow-config",
        default=None,
        help="Optional workflow-specific JSON configuration file.",
    )
    evaluate.add_argument("--sample-path", default=str(DEFAULT_EVALUATION_SAMPLE_PATH), help="Frozen sampled CSV to evaluate.")
    evaluate.add_argument(
        "--dataset-audit-path",
        default=None,
        help=(
            "Dataset audit JSON copied into run artifacts. The paired Dataset1 audit is used "
            "only with the default sample; custom samples default to audit unavailable."
        ),
    )
    evaluate.add_argument(
        "--artifact-base-dir",
        default="logs/runs",
        help="Base directory under which the timestamped run directory is created.",
    )
    evaluate.add_argument(
        "--fake-llm",
        action="store_true",
        default=None,
        help="Use a deterministic fake model; overrides workflow-config.",
    )
    evaluate.add_argument(
        "--model",
        default=None,
        help="Default chat model name; overrides workflow-config.",
    )
    evaluate.add_argument("--temperature", type=float, default=None, help="Default temperature; overrides workflow-config.")
    evaluate.add_argument(
        "--max-completion-tokens",
        type=_positive_int,
        default=None,
        help="Default maximum completion tokens; overrides workflow-config.",
    )
    evaluate.add_argument(
        "--max-retries",
        type=_nonnegative_int,
        default=None,
        help="Default workflow retry policy; overrides workflow-config.",
    )
    evaluate.add_argument(
        "--mode",
        default=None,
        help="Workflow-defined mode; overrides workflow-config.",
    )
    evaluate.add_argument(
        "--limit",
        type=_positive_int,
        default=None,
        help="Optional positive first-N record limit for smoke tests.",
    )
    evaluate.add_argument("--run-id", default=None, help="Optional run directory suffix.")
    evaluate.add_argument("--error-stop-failed-llm-attempts", type=int, default=int(os.environ.get("EVAL_ERROR_STOP_FAILED_LLM_ATTEMPTS", "10") or 10))
    evaluate.add_argument("--error-stop-failed-records", type=int, default=int(os.environ.get("EVAL_ERROR_STOP_FAILED_RECORDS", "3") or 3))
    evaluate.add_argument("--error-stop-rate", type=float, default=float(os.environ.get("EVAL_ERROR_STOP_RATE", "0.5") or 0.5))
    evaluate.add_argument("--error-stop-min-records", type=int, default=int(os.environ.get("EVAL_ERROR_STOP_MIN_RECORDS", "10") or 10))
    evaluate.set_defaults(func=cmd_evaluate)

    list_workflows = subparsers.add_parser(
        "list-workflows",
        help="List workflow names accepted by evaluate --workflow.",
    )
    list_workflows.set_defaults(func=cmd_list_workflows)

    compile_results = subparsers.add_parser("compile-results", help="Compile a run directory into CSV and Markdown summary tables.")
    compile_results.add_argument("--run-dir", required=True, help="Run directory containing evaluator artifacts.")
    compile_results.set_defaults(func=cmd_compile_results)
    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    argv = list(sys.argv[1:] if argv is None else argv)
    _ensure_help_log(argv)
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 0
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

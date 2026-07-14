"""Compile evaluator run artifacts into review-friendly result reports."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

import pandas as pd


def compile_run_results(run_dir: str | Path) -> dict[str, str]:
    """Write a long-form CSV and Markdown tables for one completed run."""
    path = Path(run_dir)
    metrics = _read_json(path / "metrics.json")
    trace_summary = _read_json(path / "trace_summary.json")
    predictions = pd.read_csv(path / "predictions.csv", encoding="utf-8-sig")
    model_metrics = pd.read_csv(path / "model_metrics.csv", encoding="utf-8-sig")
    label_metrics = pd.read_csv(path / "label_metrics.csv", encoding="utf-8-sig")
    llm_calls = _read_jsonl(path / "llm_calls.jsonl")

    compiled_rows = _compiled_rows(
        metrics=metrics,
        trace_summary=trace_summary,
        predictions=predictions,
        model_metrics=model_metrics,
        label_metrics=label_metrics,
        llm_calls=llm_calls,
    )
    compiled_csv = path / "compiled_results.csv"
    pd.DataFrame(compiled_rows).to_csv(compiled_csv, index=False, encoding="utf-8-sig")

    compiled_md = path / "compiled_results.md"
    compiled_md.write_text(
        _compiled_markdown(
            run_dir=path,
            metrics=metrics,
            trace_summary=trace_summary,
            predictions=predictions,
            model_metrics=model_metrics,
            label_metrics=label_metrics,
            llm_calls=llm_calls,
        ),
        encoding="utf-8",
    )
    return {"compiled_csv": str(compiled_csv), "compiled_markdown": str(compiled_md)}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _compiled_rows(
    *,
    metrics: dict[str, Any],
    trace_summary: dict[str, Any],
    predictions: pd.DataFrame,
    model_metrics: pd.DataFrame,
    label_metrics: pd.DataFrame,
    llm_calls: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    runtime = metrics["runtime"]
    classification = metrics["classification"]
    suicidal_intent = metrics.get("suicidal_intent") or {}
    provider_timing = runtime.get("provider_timing", {})

    for _, row in model_metrics.iterrows():
        rows.append(
            {
                "section": "model_metrics",
                "model_name": row["model_name"],
                "metric_name": "aggregate",
                "accuracy": row["accuracy"],
                "f1_score": row["f1_score"],
                "precision": row["precision"],
                "recall": row["recall"],
            }
        )

    for _, row in label_metrics.iterrows():
        payload = row.to_dict()
        payload["section"] = "per_label_metrics"
        rows.append(payload)

    for row in metrics.get("agent_metrics", []):
        rows.append({"section": "agent_metrics", **row})

    for row in suicidal_intent.get("per_class", []):
        rows.append({"section": "suicidal_intent_metrics", **row})

    for actual, predicted_counts in suicidal_intent.get("confusion_matrix", {}).items():
        for predicted, count in predicted_counts.items():
            rows.append(
                {
                    "section": "suicidal_intent_confusion_matrix",
                    "actual_label": actual,
                    "predicted_label": predicted,
                    "count": count,
                }
            )

    for actual, predicted_counts in classification["confusion_matrix"].items():
        for predicted, count in predicted_counts.items():
            rows.append(
                {
                    "section": "confusion_matrix",
                    "actual_label": actual,
                    "predicted_label": predicted,
                    "count": count,
                }
            )

    for predicted, count in predictions["prediction"].value_counts(dropna=False).sort_index().items():
        rows.append({"section": "prediction_distribution", "predicted_label": predicted, "count": count})

    for actual, count in predictions["ground_truth"].value_counts(dropna=False).sort_index().items():
        rows.append({"section": "ground_truth_distribution", "actual_label": actual, "count": count})

    runtime_metrics = {
        "processed_records": metrics["processed_records"],
        "total_records": metrics["total_records"],
        "failed_records": classification["failed_records"],
        "failure_rate": classification["failure_rate"],
        "llm_request_count": runtime["llm_request_count"],
        "failed_llm_attempts": runtime["failed_llm_attempts"],
        "json_parsing_errors": runtime.get("json_parsing_errors", 0),
        "total_latency_s": runtime["latency_s"]["total"],
        "avg_latency_s": runtime["latency_s"]["avg"],
        "min_latency_s": runtime["latency_s"]["min"],
        "max_latency_s": runtime["latency_s"]["max"],
        "prompt_tokens_total": runtime["tokens"]["prompt_total"],
        "completion_tokens_total": runtime["tokens"]["completion_total"],
        "tokens_total": runtime["tokens"]["total"],
        "cached_tokens_total": runtime["tokens"]["cached_total"],
        "cache_write_tokens_total": runtime["tokens"]["cache_write_total"],
        "avg_prompt_tokens_per_request": runtime["tokens"]["avg_prompt_per_request"],
        "avg_completion_tokens_per_request": runtime["tokens"]["avg_completion_per_request"],
        "avg_tokens_per_request": runtime["tokens"]["avg_total_per_request"],
        "avg_cached_tokens_per_request": runtime["tokens"]["avg_cached_per_request"],
        "cost_usd_total": runtime["cost_usd"]["total"],
        "cost_usd_avg_per_call": runtime["cost_usd"]["avg_per_call"],
        "observed_ttft_calls": _stat_value(provider_timing, "ttft_s", "count"),
        "avg_ttft_s": _stat_value(provider_timing, "ttft_s", "avg"),
        "min_ttft_s": _stat_value(provider_timing, "ttft_s", "min"),
        "max_ttft_s": _stat_value(provider_timing, "ttft_s", "max"),
        "observed_generation_time_calls": _stat_value(provider_timing, "generation_time_s", "count"),
        "avg_generation_time_s": _stat_value(provider_timing, "generation_time_s", "avg"),
        "avg_provider_latency_s": _stat_value(provider_timing, "provider_latency_s", "avg"),
        "avg_moderation_latency_s": _stat_value(provider_timing, "moderation_latency_s", "avg"),
        "observed_completion_tps_calls": _stat_value(provider_timing, "completion_tps", "count"),
        "avg_completion_tps": _stat_value(provider_timing, "completion_tps", "avg"),
        "avg_e2e_completion_tps": _stat_value(provider_timing, "e2e_completion_tps", "avg"),
    }
    for metric_name, value in runtime_metrics.items():
        rows.append({"section": "runtime_metrics", "metric_name": metric_name, "metric_value": value})

    error_counter = Counter(call.get("error", {}).get("error_type", "Unknown") for call in llm_calls if call.get("error"))
    for error_type, count in sorted(error_counter.items()):
        rows.append({"section": "llm_error_summary", "error_type": error_type, "count": count})

    if runtime["cost_usd"]["total"] == 0 and runtime["llm_request_count"] > 0:
        rows.append(
            {
                "section": "notes",
                "metric_name": "cost_usd",
                "note": (
                    "This run recorded zero provider cost because it was executed before "
                    "OpenRouter response_metadata cost extraction was patched."
                ),
            }
        )

    rows.append(
        {
            "section": "trace_summary",
            "model_name": trace_summary["model_name"],
            "requested_model_name": trace_summary.get("requested_model_name"),
            "mode": trace_summary.get("mode") or metrics.get("mode"),
            "evaluation_status": trace_summary["evaluation_status"],
            "processed_records": trace_summary["processed_records"],
            "total_records": trace_summary["total_records"],
            "stop_reason": trace_summary.get("stop_reason"),
        }
    )
    return rows


def _compiled_markdown(
    *,
    run_dir: Path,
    metrics: dict[str, Any],
    trace_summary: dict[str, Any],
    predictions: pd.DataFrame,
    model_metrics: pd.DataFrame,
    label_metrics: pd.DataFrame,
    llm_calls: list[dict[str, Any]],
) -> str:
    runtime = metrics["runtime"]
    classification = metrics["classification"]
    suicidal_intent = metrics.get("suicidal_intent") or {}
    provider_timing = runtime.get("provider_timing", {})
    failed_calls = [call for call in llm_calls if call.get("error")]
    error_summary = Counter(call["error"].get("error_type", "Unknown") for call in failed_calls)

    lines = [
        "# Compiled Evaluation Results",
        "",
        f"- Run directory: `{run_dir}`",
        f"- Model: `{metrics['model_name']}`",
        f"- Requested model: `{metrics.get('requested_model_name')}`",
        f"- Mode: `{metrics.get('mode', 'bool')}`",
        f"- Evaluation status: `{metrics['evaluation_status']}`",
        f"- Stop reason: `{metrics.get('stop_reason')}`",
        "",
        "## Run Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Processed records | {metrics['processed_records']} |",
        f"| Total records | {metrics['total_records']} |",
        f"| Correct predictions | {classification['correct']} |",
        f"| Failed records | {classification['failed_records']} |",
        f"| Failure rate | {classification['failure_rate']:.6f} |",
        f"| LLM requests | {runtime['llm_request_count']} |",
        f"| Failed LLM attempts | {runtime['failed_llm_attempts']} |",
        f"| JSON/schema parsing errors | {runtime.get('json_parsing_errors', 0)} |",
        "",
        "## Model Metrics",
        "",
        _markdown_table(
            model_metrics,
            ["model_name", "accuracy", "f1_score", "precision", "recall"],
            {"model_name": "Model", "f1_score": "F1 Score"},
        ),
        "",
        "## Per-Label Metrics",
        "",
        _markdown_table(
            label_metrics,
            ["label_name", "accuracy", "f1_score", "precision", "recall", "support", "tp", "fp", "tn", "fn"],
            {"label_name": "Label", "f1_score": "F1 Score", "tp": "TP", "fp": "FP", "tn": "TN", "fn": "FN"},
        ),
        "",
        "## Per-Agent Metrics",
        "",
        _markdown_table(
            pd.DataFrame(metrics.get("agent_metrics", [])),
            ["agent_name", "model_name", "task", "accuracy", "f1_score", "precision", "recall"],
            {"agent_name": "Agent", "model_name": "Model", "f1_score": "F1 Score"},
        ),
        "",
        "## Confusion Matrix",
        "",
        _confusion_matrix_table(classification["confusion_matrix"]),
        "",
        "## Suicidal-Intent Metrics",
        "",
        _markdown_table(
            pd.DataFrame(suicidal_intent.get("per_class", [])),
            ["class_name", "accuracy", "f1_score", "precision", "recall", "support", "tp", "fp", "tn", "fn"],
            {"class_name": "Value", "f1_score": "F1 Score", "tp": "TP", "fp": "FP", "tn": "TN", "fn": "FN"},
        ) if suicidal_intent else "Not available for this legacy run.",
        "",
        "### Suicidal-Intent Confusion Matrix",
        "",
        _confusion_matrix_table(suicidal_intent["confusion_matrix"])
        if suicidal_intent.get("confusion_matrix")
        else "Not available for this legacy run.",
        "",
        "## Label Distributions",
        "",
        "### Ground Truth",
        "",
        _count_table(predictions["ground_truth"].value_counts().sort_index(), "Label"),
        "",
        "### Predictions",
        "",
        _count_table(predictions["prediction"].value_counts().sort_index(), "Predicted Label"),
        "",
        "## Runtime And Tokens",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Total latency seconds | {runtime['latency_s']['total']:.6f} |",
        f"| Average latency seconds per request | {runtime['latency_s']['avg']:.6f} |",
        f"| Min latency seconds | {runtime['latency_s']['min']:.6f} |",
        f"| Max latency seconds | {runtime['latency_s']['max']:.6f} |",
        f"| Total input tokens | {runtime['tokens']['prompt_total']} |",
        f"| Total output tokens | {runtime['tokens']['completion_total']} |",
        f"| Total tokens | {runtime['tokens']['total']} |",
        f"| Total cached input tokens | {runtime['tokens']['cached_total']} |",
        f"| Total cache-write tokens | {runtime['tokens']['cache_write_total']} |",
        f"| Average input tokens per request | {runtime['tokens']['avg_prompt_per_request']:.6f} |",
        f"| Average output tokens per request | {runtime['tokens']['avg_completion_per_request']:.6f} |",
        f"| Average total tokens per request | {runtime['tokens']['avg_total_per_request']:.6f} |",
        f"| Total cost USD | {runtime['cost_usd']['total']:.6f} |",
        f"| Average cost USD per request | {runtime['cost_usd']['avg_per_call']:.6f} |",
        f"| Observed TTFT calls | {_format_optional(_stat_value(provider_timing, 'ttft_s', 'count'))} |",
        f"| Average TTFT seconds | {_format_optional(_stat_value(provider_timing, 'ttft_s', 'avg'))} |",
        f"| Average provider generation seconds | {_format_optional(_stat_value(provider_timing, 'generation_time_s', 'avg'))} |",
        f"| Average provider latency seconds | {_format_optional(_stat_value(provider_timing, 'provider_latency_s', 'avg'))} |",
        f"| Average moderation latency seconds | {_format_optional(_stat_value(provider_timing, 'moderation_latency_s', 'avg'))} |",
        f"| Average provider completion TPS | {_format_optional(_stat_value(provider_timing, 'completion_tps', 'avg'))} |",
        f"| Average end-to-end completion TPS | {_format_optional(_stat_value(provider_timing, 'e2e_completion_tps', 'avg'))} |",
        "",
        "## LLM Errors And Retries",
        "",
        "| Error Type | Count |",
        "|---|---:|",
    ]
    if error_summary:
        for error_type, count in sorted(error_summary.items()):
            lines.append(f"| {error_type} | {count} |")
    else:
        lines.append("| None | 0 |")

    limitation = suicidal_intent.get("metric_limitation")
    if limitation:
        lines.extend(["", "## Suicidal-Intent Metric Limitation", "", f"> {limitation}"])

    if failed_calls:
        lines.extend(
            [
                "",
                "### Recovered Error Attempts",
                "",
                "| Sample ID | Role | Attempt | Error Type |",
                "|---|---|---:|---|",
            ]
        )
        for call in failed_calls:
            error = call["error"]
            lines.append(
                f"| {call.get('sample_id')} | {call.get('role')} | {call.get('attempt')} | {error.get('error_type')} |"
            )

    if runtime["cost_usd"]["total"] == 0 and runtime["llm_request_count"] > 0:
        lines.extend(
            [
                "",
                "## Notes",
                "",
                "- Cost is shown exactly as recorded in this first run. It is `0.0` because this run happened before OpenRouter cost extraction from `response_metadata.token_usage` was patched.",
            ]
        )

    lines.extend(
        [
            "",
            "## Trace Files",
            "",
            "| Artifact | Purpose |",
            "|---|---|",
            "| `predictions.csv` | Per-record ground truth, prediction, correctness, errors, label-keyed specialist answers, full specialist outputs, and flat specialist answer/reasoning columns. |",
            "| `model_metrics.csv` | Aggregate model-level Accuracy, F1, Precision, Recall. |",
            "| `label_metrics.csv` | Per-label one-vs-rest metrics. |",
            "| `agent_metrics.csv` | Final-judge and specialist-agent metric summary. |",
            "| `suicidal_intent_metrics.csv` | Binary false/true per-class metrics. |",
            "| `metrics.json` | Complete metrics payload. |",
            "| `results.json` | Complete per-record outputs and metrics payload. |",
            "| `misclassified_ids.txt` | IDs with an incorrect final-label evaluation. |",
            "| `suicidal_intent_misclassified_ids.txt` | IDs with an incorrect binary evaluation. |",
            "| `classification_metrics.svg` | Final-label and suicidal-intent score visualization. |",
            "| `runtime_metrics.svg` | Cost, token, latency, call, failure, and parse-error visualization. |",
            "| `llm_calls.jsonl` | Every LLM request attempt and retry. |",
            "| `traces.jsonl` | Graph specialist/judge events. |",
            "| `trace_summary.json` | Runtime, token, and failure summary. |",
        ]
    )
    return "\n".join(lines) + "\n"


def _markdown_table(df: pd.DataFrame, columns: list[str], labels: dict[str, str]) -> str:
    header = [labels.get(column, column.replace("_", " ").title()) for column in columns]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for _, row in df.iterrows():
        cells = [_format_value(row[column]) for column in columns]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _confusion_matrix_table(matrix: dict[str, dict[str, int]]) -> str:
    predicted_labels = list(next(iter(matrix.values())).keys())
    lines = [
        "| Actual \\ Predicted | " + " | ".join(predicted_labels) + " |",
        "|" + "|".join(["---"] * (len(predicted_labels) + 1)) + "|",
    ]
    for actual, counts in matrix.items():
        lines.append("| " + actual + " | " + " | ".join(str(counts[label]) for label in predicted_labels) + " |")
    return "\n".join(lines)


def _count_table(counts: pd.Series, label_name: str) -> str:
    lines = [f"| {label_name} | Count |", "|---|---:|"]
    for label, count in counts.items():
        lines.append(f"| {label} | {int(count)} |")
    return "\n".join(lines)


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _stat_value(provider_timing: dict[str, Any], metric: str, field: str) -> Any:
    if not isinstance(provider_timing, dict):
        return None
    stats = provider_timing.get(metric) or {}
    if not isinstance(stats, dict):
        return None
    return stats.get(field)


def _format_optional(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)

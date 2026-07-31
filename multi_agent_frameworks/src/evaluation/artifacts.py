"""Local artifact writing utilities."""

from __future__ import annotations

import datetime as dt
import json
import uuid
from pathlib import Path
from typing import Any

import pandas as pd


RUN_ARTIFACTS = (
    "run_config.json",
    "manifest.json",
    "progress.sqlite",
    "dataset_audit.json",
    "sampled_dataset.csv",
    "first_completion_input_prompts.json",
    "llm_calls.jsonl",
    "traces.jsonl",
    "predictions.csv",
    "suicidal_intent_metrics.csv",
    "positive_subclassification_metrics.csv",
    "metrics.json",
    "results.json",
    "metrics.md",
    "misclassified_ids.txt",
    "suicidal_intent_misclassified_ids.txt",
    "suicidal_intent_metrics.svg",
    "runtime_metrics.svg",
    "trace_summary.json",
    "errors.jsonl",
    "compiled_results.csv",
    "compiled_results.md",
)
class ArtifactWriter:
    """Write all run artifacts into one local run directory."""

    def __init__(
        self,
        base_dir: str | Path = "logs/runs",
        run_id: str | None = None,
        resume_run_dir: str | Path | None = None,
    ):
        if resume_run_dir is not None:
            self.run_dir = Path(resume_run_dir)
            if not self.run_dir.is_dir():
                raise ValueError(f"Resume run directory does not exist: {self.run_dir}")
        else:
            stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            suffix = run_id or uuid.uuid4().hex[:8]
            self.run_dir = Path(base_dir) / f"{stamp}_{suffix}"
            self.run_dir.mkdir(parents=True, exist_ok=True)
        for name in ("llm_calls.jsonl", "traces.jsonl", "errors.jsonl"):
            (self.run_dir / name).touch()

    def write_json(self, name: str, payload: Any) -> Path:
        path = self.run_dir / name
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return path

    def append_jsonl(self, name: str, payload: dict[str, Any]) -> Path:
        path = self.run_dir / name
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return path

    def write_jsonl(self, name: str, payloads: list[dict[str, Any]]) -> Path:
        path = self.run_dir / name
        with path.open("w", encoding="utf-8") as f:
            for payload in payloads:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return path

    def write_dataframe(self, name: str, df: pd.DataFrame) -> Path:
        path = self.run_dir / name
        df.to_csv(path, index=False, encoding="utf-8-sig")
        return path

    def write_text(self, name: str, content: str) -> Path:
        path = self.run_dir / name
        path.write_text(content, encoding="utf-8")
        return path

    def write_visualizations(self, metrics: dict[str, Any]) -> tuple[Path, Path]:
        suicidal_intent_path = self.write_text(
            "suicidal_intent_metrics.svg",
            _suicidal_intent_metrics_svg(metrics),
        )
        runtime_path = self.write_text("runtime_metrics.svg", _runtime_metrics_svg(metrics))
        return suicidal_intent_path, runtime_path

    def write_metrics_md(self, metrics: dict[str, Any]) -> Path:
        path = self.run_dir / "metrics.md"
        suicidal_intent = metrics.get("suicidal_intent")
        positive_subclassification = metrics.get("positive_subclassification")
        lines = [
            "# Run Metrics",
            "",
            f"- Workflow: {metrics.get('workflow_name', 'framework_1_parallel_no_evidence')}",
            f"- Service tier: {metrics.get('service_tier', 'standard')}",
            f"- Stop-after token threshold: {(metrics.get('token_stop') or {}).get('threshold')}",
            f"- Current invocation tokens: {(metrics.get('token_stop') or {}).get('current_session_total_tokens')}",
            f"- Total samples: {metrics['classification']['total']}",
            f"- Failed records: {metrics['classification']['failed_records']}",
            f"- Failure rate: {metrics['classification']['failure_rate']:.6f}",
            f"- LLM calls: {metrics['runtime']['call_count']}",
            f"- JSON/schema parsing errors: {metrics['runtime'].get('json_parsing_errors', 0)}",
            f"- Cumulative case execution duration seconds: {metrics['runtime']['agent_execution_duration_s']:.6f}",
            f"- Real execution duration seconds: {metrics['runtime']['real_execution_duration_s']:.6f}",
            f"- Total latency seconds: {metrics['runtime']['latency_s']['total']:.6f}",
            f"- Average latency seconds per request: {metrics['runtime']['latency_s']['avg']:.6f}",
            f"- Average input tokens per request: {metrics['runtime']['tokens']['avg_prompt_per_request']:.6f}",
            f"- Average output tokens per request: {metrics['runtime']['tokens']['avg_completion_per_request']:.6f}",
            f"- Average cached tokens per request: {metrics['runtime']['tokens']['avg_cached_per_request']:.6f}",
            f"- Total cost USD: {metrics['runtime']['cost_usd']['total']:.6f}",
            f"- Average TTFT seconds when exposed: {_runtime_stat(metrics, 'ttft_s', 'avg')}",
            f"- Average provider completion TPS when exposed: {_runtime_stat(metrics, 'completion_tps', 'avg')}",
            f"- Average end-to-end completion TPS: {_runtime_stat(metrics, 'e2e_completion_tps', 'avg')}",
        ]
        if suicidal_intent:
            counts = suicidal_intent["positive_class_counts"]
            lines.extend(
                [
                    f"- Suicidal-intent grouping accuracy: {suicidal_intent['accuracy']:.6f}",
                    f"- Suicidal-intent grouping positive-class F1: {suicidal_intent['positive_f1']:.6f}",
                    f"- Suicidal-intent TP: {counts['tp']}",
                    f"- Suicidal-intent TN: {counts['tn']}",
                    f"- Suicidal-intent FP: {counts['fp']}",
                    f"- Suicidal-intent FN: {counts['fn']}",
                ]
            )
        if suicidal_intent:
            mapping = suicidal_intent.get("class_mapping", {})
            lines.extend(
                [
                    "",
                    "## Suicidal-Intent Positive-Class Grouping Metrics",
                    "",
                    f"- Positive: {', '.join(mapping.get('positive', []))}",
                    f"- Negative: {', '.join(mapping.get('negative', []))}",
                    "- TP/TN/FP/FN use this fixed positive orientation; failed predictions remain in the Error column.",
                    "",
                    "| Accuracy | F1 | Precision | Recall | Positive Support | Negative Support | TP | FP | TN | FN |",
                    "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
                ]
            )
            lines.append(
                "| {accuracy:.6f} | {f1_score:.6f} | {precision:.6f} | {recall:.6f} | "
                "{positive_support} | {negative_support} | {tp} | {fp} | {tn} | {fn} |".format(
                    **suicidal_intent["binary_summary"]
                )
            )
            limitation = suicidal_intent.get("metric_limitation")
            if limitation:
                lines.extend(["", f"> Limitation: {limitation}"])
        if positive_subclassification:
            summary = positive_subclassification["summary"]
            lines.extend(
                [
                    "",
                    "## Positive-Label Subclassification Metrics",
                    "",
                    "- Scope: records whose ground-truth and predicted labels are both Indicator, Ideation, Behavior, or Attempt.",
                    "- Precision, recall, and F1 are macro averages across the four positive labels.",
                    "- Correct: included records on the diagonal of the 4x4 matrix.",
                    "- Support: all records represented in the 4x4 matrix.",
                    "",
                    "Formulas for each label L:",
                    "- TP(L) = actual L and predicted L.",
                    "- FP(L) = actual another positive label and predicted L.",
                    "- FN(L) = actual L and predicted another positive label.",
                    "- Precision(L) = TP(L) / (TP(L) + FP(L)).",
                    "- Recall(L) = TP(L) / (TP(L) + FN(L)).",
                    "- F1(L) = 2 * Precision(L) * Recall(L) / (Precision(L) + Recall(L)).",
                    "- Overall accuracy = correct / support.",
                    "- Macro metric = sum of the four per-label metric values / 4.",
                    "",
                    "| Accuracy | F1 Score | Precision | Recall | Support | Correct | Averaging |",
                    "|---:|---:|---:|---:|---:|---:|---|",
                    "| {accuracy:.6f} | {f1_score:.6f} | {precision:.6f} | {recall:.6f} | "
                    "{support} | {correct} | {averaging} |".format(**summary),
                    "",
                    "### Positive-Label 4x4 Confusion Matrix",
                    "",
                    _confusion_matrix_markdown(
                        positive_subclassification["confusion_matrix"]
                    ),
                    "",
                    "### Per-Label Calculations",
                    "",
                    _positive_per_label_markdown(
                        positive_subclassification["per_label"]
                    ),
                ]
            )
            limitation = positive_subclassification.get("metric_limitation")
            if limitation:
                lines.extend(["", f"> Limitation: {limitation}"])
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def required_artifact_status(self) -> dict[str, bool]:
        return {name: (self.run_dir / name).exists() for name in RUN_ARTIFACTS}

def _runtime_stat(metrics: dict[str, Any], metric_name: str, field: str) -> str:
    provider_timing = metrics.get("runtime", {}).get("provider_timing", {})
    if not isinstance(provider_timing, dict):
        return "n/a"
    stats = provider_timing.get(metric_name) or {}
    if not isinstance(stats, dict):
        return "n/a"
    value = stats.get(field)
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _confusion_matrix_markdown(matrix: dict[str, dict[str, int]]) -> str:
    predicted_labels = list(next(iter(matrix.values())).keys())
    lines = [
        "| Actual \\ Predicted | " + " | ".join(predicted_labels) + " |",
        "|" + "|".join(["---"] * (len(predicted_labels) + 1)) + "|",
    ]
    for actual, counts in matrix.items():
        lines.append(
            "| "
            + actual
            + " | "
            + " | ".join(str(counts[label]) for label in predicted_labels)
            + " |"
        )
    return "\n".join(lines)


def _positive_per_label_markdown(
    per_label: dict[str, dict[str, float | int]],
) -> str:
    lines = [
        "| Label | TP | FP | FN | Support | Precision | Recall | F1 Score |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, values in per_label.items():
        lines.append(
            "| {label} | {tp} | {fp} | {fn} | {support} | "
            "{precision:.6f} | {recall:.6f} | {f1_score:.6f} |".format(
                label=label,
                **values,
            )
        )
    return "\n".join(lines)


def _suicidal_intent_metrics_svg(metrics: dict[str, Any]) -> str:
    suicidal_intent = metrics["suicidal_intent"]
    values = [
        ("Suicidal-intent accuracy", suicidal_intent["accuracy"]),
        ("Suicidal-intent precision", suicidal_intent["precision"]),
        ("Suicidal-intent recall", suicidal_intent["recall"]),
        ("Suicidal-intent F1", suicidal_intent["f1_score"]),
    ]
    width = 900
    height = 70 + len(values) * 42
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="24" y="34" font-family="sans-serif" font-size="22" font-weight="bold">Suicidal-intent metrics</text>',
    ]
    for index, (label, value) in enumerate(values):
        y = 62 + index * 42
        bar_width = max(0.0, min(1.0, float(value))) * 560
        elements.extend(
            [
                f'<text x="24" y="{y + 18}" font-family="sans-serif" font-size="14">{label}</text>',
                f'<rect x="250" y="{y}" width="560" height="24" rx="4" fill="#e5e7eb"/>',
                f'<rect x="250" y="{y}" width="{bar_width:.2f}" height="24" rx="4" fill="#16a34a"/>',
                f'<text x="820" y="{y + 18}" font-family="sans-serif" font-size="14">{float(value):.3f}</text>',
            ]
        )
    elements.append("</svg>")
    return "\n".join(elements) + "\n"


def _runtime_metrics_svg(metrics: dict[str, Any]) -> str:
    runtime = metrics["runtime"]
    values = [
        ("Cumulative case execution (s)", f'{runtime["agent_execution_duration_s"]:.3f}'),
        ("Real execution (s)", f'{runtime["real_execution_duration_s"]:.3f}'),
        ("LLM calls", runtime["call_count"]),
        ("Total latency (s)", f'{runtime["latency_s"]["total"]:.3f}'),
        ("Input tokens", runtime["tokens"]["prompt_total"]),
        ("Output tokens", runtime["tokens"]["completion_total"]),
        ("Total tokens", runtime["tokens"]["total"]),
        ("Average cost/call (USD)", f'{runtime["cost_usd"]["avg_per_call"]:.6f}'),
        ("JSON/schema errors", runtime["json_parsing_errors"]),
        ("Failed attempts", runtime["failed_llm_attempts"]),
    ]
    row_count = (len(values) + 3) // 4
    height = 74 + row_count * 98
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="900" height="{height}" viewBox="0 0 900 {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="24" y="34" font-family="sans-serif" font-size="22" font-weight="bold">Runtime and guardrail metrics</text>',
    ]
    for index, (label, value) in enumerate(values):
        column = index % 4
        row = index // 4
        x = 24 + column * 216
        y = 58 + row * 98
        elements.extend(
            [
                f'<rect x="{x}" y="{y}" width="196" height="78" rx="8" fill="#f8fafc" stroke="#cbd5e1"/>',
                f'<text x="{x + 12}" y="{y + 25}" font-family="sans-serif" font-size="13" fill="#475569">{label}</text>',
                f'<text x="{x + 12}" y="{y + 58}" font-family="sans-serif" font-size="22" font-weight="bold">{value}</text>',
            ]
        )
    elements.append("</svg>")
    return "\n".join(elements) + "\n"

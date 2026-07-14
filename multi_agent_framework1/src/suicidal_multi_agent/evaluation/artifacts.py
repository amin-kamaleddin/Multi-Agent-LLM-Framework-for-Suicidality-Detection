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
    "dataset_audit.json",
    "sampled_dataset.csv",
    "llm_calls.jsonl",
    "traces.jsonl",
    "predictions.csv",
    "model_metrics.csv",
    "label_metrics.csv",
    "agent_metrics.csv",
    "suicidal_intent_metrics.csv",
    "metrics.json",
    "results.json",
    "metrics.md",
    "misclassified_ids.txt",
    "suicidal_intent_misclassified_ids.txt",
    "classification_metrics.svg",
    "runtime_metrics.svg",
    "trace_summary.json",
    "errors.jsonl",
    "compiled_results.csv",
    "compiled_results.md",
)


class ArtifactWriter:
    """Write all run artifacts into one local run directory."""

    def __init__(self, base_dir: str | Path = "logs/runs", run_id: str | None = None):
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

    def write_dataframe(self, name: str, df: pd.DataFrame) -> Path:
        path = self.run_dir / name
        df.to_csv(path, index=False, encoding="utf-8-sig")
        return path

    def write_text(self, name: str, content: str) -> Path:
        path = self.run_dir / name
        path.write_text(content, encoding="utf-8")
        return path

    def write_visualizations(self, metrics: dict[str, Any]) -> tuple[Path, Path]:
        classification_path = self.write_text(
            "classification_metrics.svg",
            _classification_metrics_svg(metrics),
        )
        runtime_path = self.write_text("runtime_metrics.svg", _runtime_metrics_svg(metrics))
        return classification_path, runtime_path

    def write_metrics_md(self, metrics: dict[str, Any]) -> Path:
        path = self.run_dir / "metrics.md"
        suicidal_intent = metrics.get("suicidal_intent")
        agent_metrics = metrics.get("agent_metrics", [])
        lines = [
            "# Run Metrics",
            "",
            f"- Accuracy: {metrics['classification']['accuracy']:.6f}",
            f"- Macro F1: {metrics['classification']['f1_score']:.6f}",
            f"- Macro Precision: {metrics['classification']['precision']:.6f}",
            f"- Macro Recall: {metrics['classification']['recall']:.6f}",
            f"- Total samples: {metrics['classification']['total']}",
            f"- Correct: {metrics['classification']['correct']}",
            f"- Failed records: {metrics['classification']['failed_records']}",
            f"- Failure rate: {metrics['classification']['failure_rate']:.6f}",
            f"- LLM calls: {metrics['runtime']['call_count']}",
            f"- JSON/schema parsing errors: {metrics['runtime'].get('json_parsing_errors', 0)}",
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
            lines.extend(
                [
                    f"- Suicidal-intent accuracy: {suicidal_intent['accuracy']:.6f}",
                    f"- Suicidal-intent macro F1: {suicidal_intent['f1_score']:.6f}",
                    f"- Suicidal-intent positive-class F1: {suicidal_intent['positive_f1']:.6f}",
                ]
            )
        if agent_metrics:
            lines.extend(
                [
                    "",
                    "## Per-Agent Metrics",
                    "",
                    "| Agent | Model | Task | Accuracy | F1 | Precision | Recall |",
                    "|---|---|---|---:|---:|---:|---:|",
                ]
            )
            for row in agent_metrics:
                lines.append(
                    "| {agent_name} | {model_name} | {task} | {accuracy:.6f} | {f1_score:.6f} | {precision:.6f} | {recall:.6f} |".format(**row)
                )
        lines.extend(
            [
            "",
            "## Per-Class Metrics",
            "",
            "| Label | Accuracy | F1 | Precision | Recall | Support |",
            "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in metrics["classification"]["per_class"]:
            lines.append(
                "| {label_name} | {accuracy:.6f} | {f1_score:.6f} | {precision:.6f} | {recall:.6f} | {support} |".format(**row)
            )
        if suicidal_intent:
            lines.extend(
                [
                    "",
                    "## Suicidal-Intent Per-Class Metrics",
                    "",
                    "| Value | Accuracy | F1 | Precision | Recall | Support |",
                    "|---|---:|---:|---:|---:|---:|",
                ]
            )
            for row in suicidal_intent["per_class"]:
                lines.append(
                    "| {class_name} | {accuracy:.6f} | {f1_score:.6f} | {precision:.6f} | {recall:.6f} | {support} |".format(**row)
                )
            limitation = suicidal_intent.get("metric_limitation")
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


def _classification_metrics_svg(metrics: dict[str, Any]) -> str:
    classification = metrics["classification"]
    suicidal_intent = metrics["suicidal_intent"]
    values = [
        ("Final label accuracy", classification["accuracy"]),
        ("Final label macro precision", classification["precision"]),
        ("Final label macro recall", classification["recall"]),
        ("Final label macro F1", classification["f1_score"]),
        ("Intent accuracy", suicidal_intent["accuracy"]),
        ("Intent macro precision", suicidal_intent["precision"]),
        ("Intent macro recall", suicidal_intent["recall"]),
        ("Intent macro F1", suicidal_intent["f1_score"]),
    ]
    width = 900
    height = 70 + len(values) * 42
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="24" y="34" font-family="sans-serif" font-size="22" font-weight="bold">Classification metrics</text>',
    ]
    for index, (label, value) in enumerate(values):
        y = 62 + index * 42
        bar_width = max(0.0, min(1.0, float(value))) * 560
        color = "#2563eb" if index < 4 else "#16a34a"
        elements.extend(
            [
                f'<text x="24" y="{y + 18}" font-family="sans-serif" font-size="14">{label}</text>',
                f'<rect x="250" y="{y}" width="560" height="24" rx="4" fill="#e5e7eb"/>',
                f'<rect x="250" y="{y}" width="{bar_width:.2f}" height="24" rx="4" fill="{color}"/>',
                f'<text x="820" y="{y + 18}" font-family="sans-serif" font-size="14">{float(value):.3f}</text>',
            ]
        )
    elements.append("</svg>")
    return "\n".join(elements) + "\n"


def _runtime_metrics_svg(metrics: dict[str, Any]) -> str:
    runtime = metrics["runtime"]
    values = [
        ("LLM calls", runtime["call_count"]),
        ("Total latency (s)", f'{runtime["latency_s"]["total"]:.3f}'),
        ("Input tokens", runtime["tokens"]["prompt_total"]),
        ("Output tokens", runtime["tokens"]["completion_total"]),
        ("Total tokens", runtime["tokens"]["total"]),
        ("Average cost/call (USD)", f'{runtime["cost_usd"]["avg_per_call"]:.6f}'),
        ("JSON/schema errors", runtime["json_parsing_errors"]),
        ("Failed attempts", runtime["failed_llm_attempts"]),
    ]
    elements = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="270" viewBox="0 0 900 270">',
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

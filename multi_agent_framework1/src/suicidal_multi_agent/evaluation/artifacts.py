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
    "metrics.json",
    "metrics.md",
    "trace_summary.json",
    "errors.jsonl",
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

    def write_metrics_md(self, metrics: dict[str, Any]) -> Path:
        path = self.run_dir / "metrics.md"
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
            f"- Total latency seconds: {metrics['runtime']['latency_s']['total']:.6f}",
            f"- Average latency seconds per request: {metrics['runtime']['latency_s']['avg']:.6f}",
            f"- Average input tokens per request: {metrics['runtime']['tokens']['avg_prompt_per_request']:.6f}",
            f"- Average output tokens per request: {metrics['runtime']['tokens']['avg_completion_per_request']:.6f}",
            f"- Average cached tokens per request: {metrics['runtime']['tokens']['avg_cached_per_request']:.6f}",
            f"- Total cost USD: {metrics['runtime']['cost_usd']['total']:.6f}",
            f"- Average TTFT seconds when exposed: {_runtime_stat(metrics, 'ttft_s', 'avg')}",
            f"- Average provider completion TPS when exposed: {_runtime_stat(metrics, 'completion_tps', 'avg')}",
            f"- Average end-to-end completion TPS: {_runtime_stat(metrics, 'e2e_completion_tps', 'avg')}",
            "",
            "## Per-Class Metrics",
            "",
            "| Label | Accuracy | F1 | Precision | Recall | Support |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for row in metrics["classification"]["per_class"]:
            lines.append(
                "| {label_name} | {accuracy:.6f} | {f1_score:.6f} | {precision:.6f} | {recall:.6f} | {support} |".format(**row)
            )
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

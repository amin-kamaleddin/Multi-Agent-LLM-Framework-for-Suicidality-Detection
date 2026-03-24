import sys
import os
import argparse
import csv
import json
import datetime
import platform
import time
from typing import Any, Dict, List, Optional
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# Add src directories to path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent.parent
src_path = project_root / "src"
sys.path.append(str(src_path))

try:
    from EvalRing.dataset import CSVDataset, DataFrameDataset
    from EvalRing.agent import MockAgent
    from EvalRing.evaluator import ClassificationEvaluator
    from llm_agent import OpenAISuicideDetectionAgent
except ImportError as e:
    print(f"Error importing EvalRing: {e}")
    print(f"Current sys.path: {sys.path}")
    sys.exit(1)


def _build_confusion_matrix(y_true, y_pred, labels):
    matrix = {label: {pred_label: 0 for pred_label in labels} for label in labels}
    for t, p in zip(y_true, y_pred):
        if t in matrix and p in matrix[t]:
            matrix[t][p] += 1
    return matrix


def _per_class_metrics(y_true, y_pred, labels):
    rows = []
    for cls in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)
        support = sum(1 for t in y_true if t == cls)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        rows.append({"class": cls, "precision": precision, "recall": recall, "f1": f1, "support": support})
    return rows


def _write_confusion_md(f, matrix, labels):
    f.write("| Actual \\ Predicted |")
    for label in labels:
        f.write(f" {label} |")
    f.write("\n|---|")
    for _ in labels:
        f.write("---|")
    f.write("\n")
    for actual in labels:
        f.write(f"| **{actual}** |")
        for predicted in labels:
            f.write(f" {matrix[actual][predicted]} |")
        f.write("\n")


def _to_float(v):
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if s == "" or s.lower() in {"none", "nan", "n/a", "—"}:
        return None
    try:
        return float(s)
    except Exception:
        return None


def _to_int(v):
    fv = _to_float(v)
    return int(fv) if fv is not None else None


def _latest_meta_path(base_dir: Path) -> Optional[Path]:
    runs_root = base_dir / ".EvalRing"
    if not runs_root.exists():
        return None
    run_dirs = [p for p in runs_root.iterdir() if p.is_dir() and p.name.startswith("run_")]
    if not run_dirs:
        return None
    latest = sorted(run_dirs, key=lambda p: p.stat().st_mtime, reverse=True)[0]
    meta = latest / "Meta.json"
    return meta if meta.exists() else None


def _read_all_cases_csv(csv_path: Path) -> List[Dict[str, Any]]:
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = [dict(r) for r in reader]

    normalized = []
    for r in rows:
        raw_class_scores = r.get("class_scores")
        parsed_class_scores = None
        if raw_class_scores:
            try:
                parsed_class_scores = json.loads(raw_class_scores)
            except Exception:
                parsed_class_scores = None

        normalized.append({
            "sample_id": r.get("sample_id", ""),
            "ground_truth": r.get("ground_truth", ""),
            "prediction": r.get("prediction", ""),
            "correct": _to_int(r.get("correct")) or 0,
            "prediction_confidence": _to_float(r.get("prediction_confidence")),
            "class_scores": parsed_class_scores,
            "ttft": _to_float(r.get("ttft")),
            "tps": _to_float(r.get("tps")),
            "total_time": _to_float(r.get("total_time")),
            "generation_time": _to_float(r.get("generation_time")),
            "prompt_tokens": _to_int(r.get("prompt_tokens")),
            "completion_tokens": _to_int(r.get("completion_tokens")),
            "total_tokens": _to_int(r.get("total_tokens")),
            "error": (r.get("error") or "").strip() or None,
            "text": r.get("text", ""),
        })
    return normalized


def _build_agent(
    agent_mode: str = "single-class",
    base_class: str = "Indicator",
    host_model: Optional[str] = None,
    role_models_json: Optional[str] = None,
    max_host_iterations: int = 10,
):
    api_key = os.environ.get("OPEN_ROUTER_KEY") or os.environ.get("OPENAI_API_KEY")
    if api_key:
        key_source = "OPEN_ROUTER_KEY" if os.environ.get("OPEN_ROUTER_KEY") else "OPENAI_API_KEY"
        role_model_map = None
        if role_models_json:
            try:
                parsed = json.loads(role_models_json)
                if isinstance(parsed, dict):
                    role_model_map = {str(k): str(v) for k, v in parsed.items()}
            except Exception:
                print("WARNING: --role-models-json is not valid JSON object. Ignoring it.")

        agent = OpenAISuicideDetectionAgent(
            api_key=api_key,
            agent_mode=agent_mode,
            base_class=base_class,
            host_model_name=host_model,
            role_model_map=role_model_map,
            max_host_iterations=max_host_iterations,
        )
        print(
            f"{key_source} found. Using OpenAISuicideDetectionAgent "
            f"({getattr(agent, 'model_name', 'unknown-model')}, mode={agent_mode}, base={base_class})."
        )
        agent.initialize()
        return agent

    print("No OPEN_ROUTER_KEY / OPENAI_API_KEY found in environment. Falling back to MockAgent.")
    possible_outputs = ["Ideation", "Behavior", "Indicator", "Attempt"]
    agent = MockAgent(name="MockSuicideDetectionAgent", possible_outputs=possible_outputs)
    agent.initialize()
    return agent


def _write_reports(
    *,
    eval_dir: Path,
    result,
    agent,
    all_cases: List[Dict[str, Any]],
    data_path: Path,
    n_samples: int,
    max_workers: int,
    max_retries: int,
    seed: int,
    agent_mode: str,
    base_class: str,
    host_model: Optional[str],
    role_models_json: Optional[str],
    max_host_iterations: int,
    retry_meta: Optional[Dict[str, Any]] = None,
    retry_summary: Optional[Dict[str, Any]] = None,
):
    y_true_all = [c["ground_truth"] for c in all_cases]
    y_pred_all = [c["prediction"] for c in all_cases]
    failed_cases = [c for c in all_cases if int(c.get("correct", 0)) < 1]

    LABELS = ["Ideation", "Behavior", "Indicator", "Attempt"]
    observed_labels = sorted(set(y_true_all) | set(y_pred_all))
    all_labels = [label for label in LABELS if label in observed_labels] + [label for label in observed_labels if label not in LABELS]

    confusion = _build_confusion_matrix(y_true_all, y_pred_all, all_labels)
    per_class = _per_class_metrics(y_true_all, y_pred_all, all_labels)
    label_dist = Counter(y_true_all)

    correct = sum(1 for t, p in zip(y_true_all, y_pred_all) if t == p)
    total = len(y_true_all)
    accuracy = (correct / total) if total > 0 else 0.0

    if len(per_class) > 0:
        macro_precision = sum(r["precision"] for r in per_class) / len(per_class)
        macro_recall = sum(r["recall"] for r in per_class) / len(per_class)
        macro_f1 = sum(r["f1"] for r in per_class) / len(per_class)
    else:
        macro_precision = macro_recall = macro_f1 = 0.0

    ttfts = [c["ttft"] for c in all_cases if c.get("ttft") is not None]
    tpss = [c["tps"] for c in all_cases if c.get("tps") is not None]
    gen_times = [c["generation_time"] for c in all_cases if c.get("generation_time") is not None]
    total_times = [c["total_time"] for c in all_cases if c.get("total_time") is not None]
    prompt_toks = [c["prompt_tokens"] for c in all_cases if c.get("prompt_tokens") is not None]
    completion_toks = [c["completion_tokens"] for c in all_cases if c.get("completion_tokens") is not None]
    total_toks = [c["total_tokens"] for c in all_cases if c.get("total_tokens") is not None]

    def _mean(xs):
        return (sum(xs) / len(xs)) if xs else 0.0

    metrics_dict = {
        "accuracy": accuracy,
        "precision": macro_precision,
        "recall": macro_recall,
        "f1_score": macro_f1,
        "avg_ttft": _mean(ttfts),
        "avg_tps": _mean(tpss),
        "avg_generation_time": _mean(gen_times),
        "avg_total_time": _mean(total_times),
        "total_prompt_tokens": sum(prompt_toks) if prompt_toks else 0,
        "total_completion_tokens": sum(completion_toks) if completion_toks else 0,
        "total_tokens": sum(total_toks) if total_toks else 0,
        "avg_prompt_tokens": _mean(prompt_toks),
        "avg_completion_tokens": _mean(completion_toks),
    }
    cost_usd = (metrics_dict["total_prompt_tokens"] * 2.0 / 1_000_000) + (metrics_dict["total_completion_tokens"] * 8.0 / 1_000_000)
    metrics_dict["total_cost_usd"] = cost_usd

    result_md_path = eval_dir / "result.md"
    with open(result_md_path, "w", encoding="utf-8") as f:
        f.write("# Evaluation Report — Suicide Risk Detection\n\n")
        f.write(f"**Timestamp:** {datetime.datetime.now().isoformat()}\n")
        f.write(f"**Agent:** {result.agent_name}\n")
        f.write(f"**Model:** {getattr(agent, 'model_name', 'N/A')}\n")
        f.write(f"**Task:** {result.task_name}\n")
        f.write(f"**Dataset:** rsd_15k.csv (n={len(all_cases)})\n")
        f.write(f"**Duration:** {result.duration:.4f} s\n")
        f.write(f"**Agent Mode:** {agent_mode}\n")
        f.write(f"**Base Class:** {base_class}\n")
        if host_model:
            f.write(f"**Host Model:** {host_model}\n")
        f.write(f"**Max Host Iterations:** {max_host_iterations}\n")
        f.write(f"**Workers:** {max_workers} | **Retries:** {max_retries} | **Seed:** {seed}\n\n")

        if retry_summary:
            f.write("## Retry Update\n\n")
            f.write("- **retry_failed_mode**: true\n")
            f.write(f"- **retried_cases**: {retry_summary.get('retried_cases', 0)}\n")
            f.write(f"- **resolved_errors**: {retry_summary.get('resolved_errors', 0)}\n")
            f.write(f"- **remaining_error_cases**: {retry_summary.get('remaining_error_cases', 0)}\n\n")

        f.write("## Label Distribution (ground truth)\n\n")
        f.write("| Label | Count | % |\n|---|---|---|\n")
        total_n = len(y_true_all)
        for lbl in all_labels:
            cnt = label_dist.get(lbl, 0)
            pct = (cnt / total_n * 100) if total_n > 0 else 0.0
            f.write(f"| {lbl} | {cnt} | {pct:.1f}% |\n")
        f.write("\n")

        f.write("## Aggregate Metrics\n\n")
        priority_metrics = [
            "accuracy", "precision", "recall", "f1_score",
            "avg_ttft", "avg_tps", "avg_generation_time", "avg_total_time",
            "total_prompt_tokens", "total_completion_tokens", "total_tokens",
            "avg_prompt_tokens", "avg_completion_tokens", "total_cost_usd",
        ]
        for key in priority_metrics:
            if key in metrics_dict:
                val = metrics_dict[key]
                f.write(f"- **{key}**: {val:.6f}\n" if isinstance(val, float) else f"- **{key}**: {val}\n")
        f.write("\n")

        f.write("## Per-Class Metrics\n\n")
        f.write("| Class | Precision | Recall | F1 | Support |\n|---|---|---|---|---|\n")
        for row in per_class:
            f.write(f"| {row['class']} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} | {row['support']} |\n")
        f.write("\n")

        f.write("## Confusion Matrix\n\n")
        f.write("*Rows = Actual, Columns = Predicted*\n\n")
        _write_confusion_md(f, confusion, all_labels)
        f.write("\n")

        f.write("## Per-Sample Results\n\n")
        f.write("| Sample ID | Ground Truth | Prediction | Confidence | Correct | TTFT (s) | TPS | Tokens |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for c in all_cases:
            ttft_s = f"{c['ttft']:.3f}" if c['ttft'] is not None else "—"
            tps_s = f"{c['tps']:.1f}" if c['tps'] is not None else "—"
            tok_s = str(c['total_tokens']) if c['total_tokens'] is not None else "—"
            conf_s = f"{c['prediction_confidence']:.3f}" if c.get("prediction_confidence") is not None else "—"
            ok = "✓" if int(c.get("correct", 0)) else "✗"
            f.write(
                f"| {c['sample_id']} | {c['ground_truth']} | {c['prediction']} | "
                f"{conf_s} | {ok} | {ttft_s} | {tps_s} | {tok_s} |\n"
            )

    all_cases_path = eval_dir / "all_cases.txt"
    with open(all_cases_path, "w", encoding="utf-8") as f:
        f.write(f"All Cases Log — Agent: {result.agent_name} | Task: {result.task_name}\n")
        f.write("=" * 80 + "\n\n")
        for case in all_cases:
            status = "PASS" if int(case.get("correct", 0)) else "FAIL"
            f.write(f"Sample ID : {case['sample_id']} [{status}]\n")
            f.write(f"GT        : {case['ground_truth']}\n")
            f.write(f"Pred      : {case['prediction']}\n")
            if case['ttft'] is not None:
                f.write(f"Timing    : TTFT={case['ttft']:.4f}s | Gen={case['generation_time']:.4f}s | TPS={case['tps']:.2f} | Total={case['total_time']:.4f}s\n")
            if case['prompt_tokens'] is not None:
                f.write(f"Tokens    : Prompt={case['prompt_tokens']} | Completion={case['completion_tokens']} | Total={case['total_tokens']}\n")
            if case.get("prediction_confidence") is not None:
                f.write(f"Confidence: {case['prediction_confidence']:.6f}\n")
            if case.get("class_scores"):
                f.write(f"ClassDist : {json.dumps(case['class_scores'], ensure_ascii=False)}\n")
            if case['error']:
                f.write(f"Error     : {case['error']}\n")
            f.write(f"Text:\n{case['text']}\n")
            f.write("-" * 40 + "\n\n")

    csv_fields = [
        "sample_id", "ground_truth", "prediction", "correct",
        "prediction_confidence", "class_scores",
        "ttft", "tps", "total_time", "generation_time",
        "prompt_tokens", "completion_tokens", "total_tokens", "error", "text",
    ]
    csv_path = eval_dir / "all_cases.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for case in all_cases:
            writer.writerow({k: case.get(k, "") for k in csv_fields})

    failed_log_path = eval_dir / "failed_cases.txt"
    if failed_cases:
        with open(failed_log_path, "w", encoding="utf-8") as f:
            f.write(f"Failed Cases Log — Agent: {result.agent_name} | Task: {result.task_name}\n")
            f.write("=" * 80 + "\n\n")
            for case in failed_cases:
                f.write(f"Sample ID : {case['sample_id']}\n")
                f.write(f"GT        : {case['ground_truth']}\n")
                f.write(f"Pred      : {case['prediction']}\n")
                if case['error']:
                    f.write(f"Error     : {case['error']}\n")
                f.write(f"Text:\n{case['text']}\n")
                f.write("-" * 40 + "\n\n")
    elif failed_log_path.exists():
        failed_log_path.unlink()

    meta_json_path = eval_dir / "Meta.json"
    old_meta = retry_meta or {}
    retry_history = old_meta.get("retry_history", []) if isinstance(old_meta.get("retry_history", []), list) else []
    if retry_summary:
        retry_history.append({"timestamp": datetime.datetime.now().isoformat(), **retry_summary})

    meta_payload = {
        **old_meta,
        **(result.metadata if hasattr(result, "metadata") else {}),
        "run_config": {
            "n_samples": n_samples,
            "max_workers": max_workers,
            "max_retries": max_retries,
            "seed": seed,
            "timestamp": datetime.datetime.now().isoformat(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
        "model_config": {
            "model_name": getattr(agent, 'model_name', 'N/A'),
            "agent_mode": agent_mode,
            "base_class": base_class,
            "host_model": host_model,
            "role_models_json": role_models_json,
            "max_host_iterations": max_host_iterations,
            "temperature": getattr(agent, 'temperature', 'N/A'),
            "max_completion_tokens": getattr(agent, 'max_completion_tokens', 'N/A'),
            "system_prompt": getattr(agent, 'system_prompt', 'N/A'),
        },
        "dataset_config": {
            "source": str(data_path),
            "text_field": "text",
            "label_field": "sentiment",
            "id_field": "users",
            "label_distribution": dict(label_dist),
        },
        "aggregate_metrics": metrics_dict,
        "per_class_metrics": per_class,
        "confusion_matrix": confusion,
        "total_cost_usd_estimate": cost_usd,
        "retry_history": retry_history,
    }
    with open(meta_json_path, "w", encoding="utf-8") as f:
        json.dump(meta_payload, f, indent=4, default=str)

    print("  ✓ result.md")
    print("  ✓ all_cases.txt")
    print("  ✓ all_cases.csv")
    if failed_cases:
        print(f"  ✓ failed_cases.txt ({len(failed_cases)} cases)")
    else:
        print("  ✓ No failed cases.")
    print("  ✓ Meta.json")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Agent:    {result.agent_name}  (model: {getattr(agent, 'model_name', 'N/A')})")
    print(f"Mode:     {agent_mode}")
    print(f"Base:     {base_class}")
    if host_model:
        print(f"Host:     {host_model}")
    print(f"HostMax:  {max_host_iterations}")
    print(f"Samples:  {len(all_cases)}  |  Failed: {len(failed_cases)}")
    print(f"Accuracy: {metrics_dict.get('accuracy', 0):.4f}  |  F1 (macro): {metrics_dict.get('f1_score', 0):.4f}")
    print(f"Duration: {result.duration:.2f}s  |  Est. Cost: ${cost_usd:.6f}")
    print(f"Output:   {eval_dir}")
    print("=" * 60)


def _run_retry_failed(meta_path: Path):
    if not meta_path.exists():
        print(f"Meta.json not found: {meta_path}")
        return

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    run_dir = meta_path.parent
    all_cases_csv = run_dir / "all_cases.csv"
    if not all_cases_csv.exists():
        print(f"all_cases.csv not found: {all_cases_csv}")
        return

    run_cfg = meta.get("run_config", {})
    model_cfg = meta.get("model_config", {})
    ds_cfg = meta.get("dataset_config", {})

    n_samples = int(run_cfg.get("n_samples", 50))
    max_workers = int(run_cfg.get("max_workers", 5))
    max_retries = int(run_cfg.get("max_retries", 3))
    seed = int(run_cfg.get("seed", 42))
    agent_mode = str(model_cfg.get("agent_mode", "single-class"))
    base_class = str(model_cfg.get("base_class", "Indicator"))
    host_model = model_cfg.get("host_model")
    role_models_json = model_cfg.get("role_models_json")
    max_host_iterations = int(model_cfg.get("max_host_iterations", 10))

    data_source = Path(ds_cfg.get("source", current_file.parent / "data" / "rsd_15k.csv"))

    print(f"Retry mode enabled. Using config from: {meta_path}")
    print(f"Dataset source: {data_source}")

    all_cases = _read_all_cases_csv(all_cases_csv)
    retry_indices = [
        i for i, row in enumerate(all_cases)
        if str(row.get("prediction", "")).strip().lower() == "error" or bool(row.get("error"))
    ]

    if not retry_indices:
        print("No Error cases found in all_cases.csv. Nothing to retry.")
        return

    print(f"Found {len(retry_indices)} Error case(s). Retrying only those cases...")

    import pandas as pd
    retry_df = pd.DataFrame([
        {
            "retry_row_id": str(idx),
            "text": all_cases[idx]["text"],
            "ground_truth": all_cases[idx]["ground_truth"],
        }
        for idx in retry_indices
    ])

    dataset = DataFrameDataset(name="rsd_15k_retry_subset")
    dataset.load_data(
        source=retry_df,
        text_field="text",
        label_field="ground_truth",
        id_field="retry_row_id",
    )

    agent = _build_agent(
        agent_mode=agent_mode,
        base_class=base_class,
        host_model=host_model,
        role_models_json=role_models_json,
        max_host_iterations=max_host_iterations,
    )
    evaluator = ClassificationEvaluator()

    start_time = time.time()
    result = evaluator.evaluate(
        agent=agent,
        dataset=dataset,
        task_name=meta.get("task_name", "suicide_risk_classification"),
        version=meta.get("version", "v1.0"),
        max_workers=max_workers,
        max_retries=max_retries,
        seed=seed,
        retry_failed=True,
    )
    retry_duration = time.time() - start_time

    per_sample_metrics = getattr(result.metrics, "per_sample_metrics", [])
    resolved = 0
    still_error = 0
    for sample_res in per_sample_metrics:
        row_idx = int(sample_res["sample_id"])
        pred = sample_res.get("prediction", "Error")
        gt = sample_res.get("ground_truth", all_cases[row_idx]["ground_truth"])
        err = sample_res.get("error", None)

        all_cases[row_idx]["prediction"] = pred
        all_cases[row_idx]["ground_truth"] = gt
        all_cases[row_idx]["correct"] = 1 if pred == gt else 0
        all_cases[row_idx]["error"] = err
        all_cases[row_idx]["ttft"] = sample_res.get("ttft", None)
        all_cases[row_idx]["tps"] = sample_res.get("tps", None)
        all_cases[row_idx]["total_time"] = sample_res.get("total_time", None)
        all_cases[row_idx]["generation_time"] = sample_res.get("generation_time", None)
        all_cases[row_idx]["prompt_tokens"] = sample_res.get("prompt_tokens", None)
        all_cases[row_idx]["completion_tokens"] = sample_res.get("completion_tokens", None)
        all_cases[row_idx]["total_tokens"] = sample_res.get("total_tokens", None)
        all_cases[row_idx]["prediction_confidence"] = sample_res.get("prediction_confidence", None)
        all_cases[row_idx]["class_scores"] = sample_res.get("class_scores", None)

        if str(pred).strip().lower() == "error" or bool(err):
            still_error += 1
        else:
            resolved += 1

    result.duration = retry_duration
    retry_summary = {
        "retried_cases": len(retry_indices),
        "resolved_errors": resolved,
        "remaining_error_cases": still_error,
        "source_meta_path": str(meta_path),
    }

    _write_reports(
        eval_dir=run_dir,
        result=result,
        agent=agent,
        all_cases=all_cases,
        data_path=data_source,
        n_samples=n_samples,
        max_workers=max_workers,
        max_retries=max_retries,
        seed=seed,
        agent_mode=agent_mode,
        base_class=base_class,
        host_model=host_model,
        role_models_json=role_models_json,
        max_host_iterations=max_host_iterations,
        retry_meta=meta,
        retry_summary=retry_summary,
    )


def run_main_eval(
    n_samples: int = 50,
    max_workers: int = 5,
    max_retries: int = 3,
    seed: int = 42,
    agent_mode: str = "single-class",
    base_class: str = "Indicator",
    host_model: Optional[str] = None,
    role_models_json: Optional[str] = None,
    max_host_iterations: int = 10,
):
    print(f"Running RSD_15K evaluation — sample limit: {n_samples}")

    base_dir = current_file.parent
    data_path = base_dir / "data" / "rsd_15k.csv"

    if not data_path.exists():
        print(f"Dataset file not found at: {data_path}")
        return

    print(f"Loading dataset from: {data_path}")
    dataset = CSVDataset(name="rsd_15k_sample")
    dataset.load_data(
        source=data_path,
        text_field="text",
        label_field="sentiment",
        id_field="users",
    )

    print(f"Total samples loaded: {len(dataset._samples)}")
    dataset._samples = dataset._samples[:n_samples]
    print(f"Samples limited to: {len(dataset._samples)}")

    agent = _build_agent(
        agent_mode=agent_mode,
        base_class=base_class,
        host_model=host_model,
        role_models_json=role_models_json,
        max_host_iterations=max_host_iterations,
    )
    evaluator = ClassificationEvaluator()

    print(f"\nStarting evaluation — workers={max_workers}, retries={max_retries}, seed={seed} ...")
    result = evaluator.evaluate(
        agent=agent,
        dataset=dataset,
        task_name="suicide_risk_classification",
        version="v1.0",
        max_workers=max_workers,
        max_retries=max_retries,
        seed=seed,
        n_samples=n_samples,
        agent_mode=agent_mode,
        base_class=base_class,
    )

    print("\nEvaluation complete. Collecting results...")
    all_cases = []

    per_sample_metrics = getattr(result.metrics, "per_sample_metrics", [])

    for sample_res in per_sample_metrics:
        sid = sample_res["sample_id"]
        pred = sample_res["prediction"]
        gt = sample_res["ground_truth"]
        acc = sample_res["accuracy"]

        status_str = "PASS" if float(acc) == 1.0 else "FAIL"
        print(f"  {sid}: GT={gt} | Pred={pred} [{status_str}]")
        if pred == "Error":
            print(f"    Error: {sample_res.get('error', 'unknown')}")

        case_data = {
            "sample_id": sid,
            "text": sample_res.get("input_text", "Text not found"),
            "prediction": pred,
            "ground_truth": gt,
            "correct": 1 if float(acc) == 1.0 else 0,
            "ttft": sample_res.get("ttft", None),
            "tps": sample_res.get("tps", None),
            "total_time": sample_res.get("total_time", None),
            "generation_time": sample_res.get("generation_time", None),
            "prompt_tokens": sample_res.get("prompt_tokens", None),
            "completion_tokens": sample_res.get("completion_tokens", None),
            "total_tokens": sample_res.get("total_tokens", None),
            "prediction_confidence": sample_res.get("prediction_confidence", None),
            "class_scores": sample_res.get("class_scores", None),
            "error": sample_res.get("error", None),
        }
        all_cases.append(case_data)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_dir = current_file.parent / ".EvalRing" / f"run_{timestamp}"
    eval_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving reports to: {eval_dir}")

    _write_reports(
        eval_dir=eval_dir,
        result=result,
        agent=agent,
        all_cases=all_cases,
        data_path=data_path,
        n_samples=n_samples,
        max_workers=max_workers,
        max_retries=max_retries,
        seed=seed,
        agent_mode=agent_mode,
        base_class=base_class,
        host_model=host_model,
        role_models_json=role_models_json,
        max_host_iterations=max_host_iterations,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate suicide risk detection on RSD_15K")
    parser.add_argument("--n-samples", type=int, default=50, help="Number of samples to evaluate (default: 50)")
    parser.add_argument("--max-workers", type=int, default=5, help="Parallel workers (default: 5)")
    parser.add_argument("--max-retries", type=int, default=3, help="Retries per sample on failure (default: 3)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--agent-mode",
        type=str,
        default="single-class",
        choices=["single-class", "multi-class-chance", "base-vs-rest-binary", "multi-agent-host"],
        help="Agent output mode: single label or structured class probabilities.",
    )
    parser.add_argument(
        "--base-class",
        type=str,
        default="Indicator",
        choices=["Indicator", "Ideation", "Behavior", "Attempt"],
        help="Base class used by base-vs-rest-binary mode.",
    )
    parser.add_argument(
        "--host-model",
        type=str,
        default=None,
        help="Optional host model for multi-agent-host mode.",
    )
    parser.add_argument(
        "--role-models-json",
        type=str,
        default=None,
        help="Optional JSON object mapping role name to model, e.g. {\"internet_advisor\":\"openai/gpt-4o-mini\"}.",
    )
    parser.add_argument(
        "--max-host-iterations",
        type=int,
        default=10,
        help="Maximum host questioning rounds in multi-agent-host mode (hard-capped at 10 in agent).",
    )
    parser.add_argument("--retry-failed", action="store_true", help="Retry only Error cases from an existing run and update that run in place")
    parser.add_argument("--meta-path", type=str, default=None, help="Path to existing Meta.json for retry mode (defaults to latest run)")
    args = parser.parse_args()

    if args.retry_failed:
        chosen_meta = Path(args.meta_path) if args.meta_path else _latest_meta_path(current_file.parent)
        if chosen_meta is None:
            print("Could not find a Meta.json to retry from. Provide --meta-path.")
            sys.exit(1)
        _run_retry_failed(chosen_meta)
    else:
        run_main_eval(
            n_samples=args.n_samples,
            max_workers=args.max_workers,
            max_retries=args.max_retries,
            seed=args.seed,
            agent_mode=args.agent_mode,
            base_class=args.base_class,
            host_model=args.host_model,
            role_models_json=args.role_models_json,
            max_host_iterations=args.max_host_iterations,
        )

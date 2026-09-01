import sys
import os
import argparse
import csv
import json
import datetime
import platform
import time
import requests
import threading
import pandas as pd
from typing import Any, Dict, List, Optional
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# Add the single_agent_frameworks package root to path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
sys.path.append(str(project_root))

try:
    import json
    import csv
    from single_agent_frameworks.dataset import CSVDataset, DataFrameDataset
    from single_agent_frameworks.agent import MockAgent
    from single_agent_frameworks.evaluator import ClassificationEvaluator
    from single_agent_frameworks.utils.global_cache import _load_cached_cases as _load_cached_cases
    from llm_agent import OpenAISuicideDetectionAgent
except ImportError as e:
    print(f"Error importing single_agent_frameworks: {e}")
    print(f"Current sys.path: {sys.path}")
    sys.exit(1)


def _build_confusion_matrix(y_true, y_pred, labels):
    matrix = {label: {pred_label: 0 for pred_label in labels} for label in labels}
    for t, p in zip(y_true, y_pred):
        if t in matrix and p in matrix[t]:
            matrix[t][p] += 1
    return matrix


def _per_class_metrics(y_true, y_pred, labels):
    tps = {cls: 0 for cls in labels}
    fps = {cls: 0 for cls in labels}
    fns = {cls: 0 for cls in labels}
    support_counts = {cls: 0 for cls in labels}
    
    for t, p in zip(y_true, y_pred):
        if t in labels:
            support_counts[t] += 1
            if t == p:
                tps[t] += 1
            else:
                fns[t] += 1
        if p in labels and t != p:
            fps[p] += 1

    rows = []
    for cls in labels:
        tp = tps[cls]
        fp = fps[cls]
        fn = fns[cls]
        support = support_counts[cls]
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
    runs_root = base_dir / "_runs"
    if not runs_root.exists():
        return None
    run_dirs = [p for p in runs_root.iterdir() if p.is_dir() and p.name.startswith("run_")]
    if not run_dirs:
        return None
    latest = sorted(run_dirs, key=lambda p: p.stat().st_mtime, reverse=True)[0]
    meta = latest / "Meta.json"
    return meta if meta.exists() else None






def _init_error_csv(error_csv_path: Path) -> None:
    """Create error.csv with a fixed schema if it does not exist yet."""
    if error_csv_path.exists():
        return

    fields = [
        "timestamp",
        "phase",
        "sample_id",
        "error_type",
        "error_message",
        "is_rate_limit",
        "attempt",
        "rate_limit_attempt",
        "max_retries",
    ]
    with open(error_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()


def _make_error_event_writer(error_csv_path: Path, write_lock):
    """Return a callback that appends one row per thrown exception."""
    fields = [
        "timestamp",
        "phase",
        "sample_id",
        "error_type",
        "error_message",
        "is_rate_limit",
        "attempt",
        "rate_limit_attempt",
        "max_retries",
    ]

    def _write_error_event(event: Dict[str, Any]):
        row = {k: event.get(k, "") for k in fields}
        with write_lock:
            with open(error_csv_path, "a", encoding="utf-8", newline="") as ef:
                writer = csv.DictWriter(ef, fieldnames=fields)
                writer.writerow(row)

    return _write_error_event


def _build_agent(
    agent_mode: str = "single-class",
    base_class: str = "Unsupportive",
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
    possible_outputs = ["Unsupportive", "Ideation", "Behavior", "Indicator", "Attempt"]
    agent = MockAgent(name="MockSuicideDetectionAgent", possible_outputs=possible_outputs)
    agent.initialize()
    return agent


_PRICE_CACHE = None

def _get_model_pricing(model_name: str) -> tuple[float, float]:
    global _PRICE_CACHE
    if _PRICE_CACHE is None:
        try:
            url = "https://openrouter.ai/api/v1/models"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                _PRICE_CACHE = {}
                models = response.json().get("data", [])
                for m in models:
                    pid = m.get("id")
                    pp = float(m.get("pricing", {}).get("prompt", 0))
                    cp = float(m.get("pricing", {}).get("completion", 0))
                    _PRICE_CACHE[pid] = (pp, cp)
            else:
                _PRICE_CACHE = {}
        except Exception as e:
            print(f"Warning: Could not fetch pricing for {model_name} from OpenRouter: {e}")
            _PRICE_CACHE = {}

    return _PRICE_CACHE.get(model_name, (0.0, 0.0))


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
    incorrect_cases = [c for c in all_cases if int(c.get("correct", 0)) < 1]

    # error report: {"{error_msg + content}": times}
    error_counts: Dict[str, int] = {}
    for c in all_cases:
        err = str(c.get("error") or "").strip()
        if not err:
            continue
        content = str(c.get("text") or "")
        key = f"{err}\n{content}"
        error_counts[key] = error_counts.get(key, 0) + 1

    LABELS = ["Unsupportive","Ideation", "Behavior", "Indicator", "Attempt"]
    observed_labels = sorted(set(y_true_all) | set(y_pred_all))
    all_labels = [label for label in LABELS if label in observed_labels] + [label for label in observed_labels if label not in LABELS]

    confusion = _build_confusion_matrix(y_true_all, y_pred_all, all_labels)
    per_class = _per_class_metrics(y_true_all, y_pred_all, all_labels)
    label_dist = Counter(y_true_all)

    correct = sum(1 for t, p in zip(y_true_all, y_pred_all) if t == p)
    total = len(y_true_all)
    accuracy = (correct / total) if total > 0 else 0.0

    # ------------------------------------------------------------------
    # Binary (Positive vs Negative) framing.
    # Negative == the base_class (e.g. "Unsupportive"); Positive == every
    # other risk label (Ideation / Behavior / Indicator / Attempt).
    # A prediction that is not one of the 5 canonical labels (e.g. "Error")
    # is never counted as a correct hit — it is booked as a miss so this
    # binary "correct" stays consistent with the 5-class "correct".
    # ------------------------------------------------------------------
    negative_class = base_class
    tp_bin = tn_bin = fp_bin = fn_bin = 0
    for t, p in zip(y_true_all, y_pred_all):
        t_pos = (t != negative_class)
        if p not in LABELS:
            # unrecognized / errored prediction -> a miss for whichever side is true
            if t_pos:
                fn_bin += 1
            else:
                fp_bin += 1
            continue
        p_pos = (p != negative_class)
        if t_pos and p_pos:
            tp_bin += 1
        elif (not t_pos) and (not p_pos):
            tn_bin += 1
        elif (not t_pos) and p_pos:
            fp_bin += 1
        else:  # t_pos and not p_pos
            fn_bin += 1

    bin_total = tp_bin + tn_bin + fp_bin + fn_bin
    accuracy_positive = ((tp_bin + tn_bin) / bin_total) if bin_total > 0 else 0.0

    prec_pos = tp_bin / (tp_bin + fp_bin) if (tp_bin + fp_bin) > 0 else 0.0
    rec_pos = tp_bin / (tp_bin + fn_bin) if (tp_bin + fn_bin) > 0 else 0.0
    f1_pos = (2 * prec_pos * rec_pos / (prec_pos + rec_pos)) if (prec_pos + rec_pos) > 0 else 0.0

    prec_neg = tn_bin / (tn_bin + fn_bin) if (tn_bin + fn_bin) > 0 else 0.0
    rec_neg = tn_bin / (tn_bin + fp_bin) if (tn_bin + fp_bin) > 0 else 0.0
    f1_neg = (2 * prec_neg * rec_neg / (prec_neg + rec_neg)) if (prec_neg + rec_neg) > 0 else 0.0

    # Macro-average over the two classes, mirroring the 5-class aggregate formulas.
    precision_positive = (prec_pos + prec_neg) / 2
    recall_positive = (rec_pos + rec_neg) / 2
    f1_positive = (f1_pos + f1_neg) / 2

    # 2x2 confusion matrix (Rows = Actual, Columns = Predicted).
    binary_confusion = {
        "Positive": {"Positive": tp_bin, "Negative": fn_bin},
        "Negative": {"Positive": fp_bin, "Negative": tn_bin},
    }
    binary_counts = {"TP": tp_bin, "TN": tn_bin, "FP": fp_bin, "FN": fn_bin}

    # ------------------------------------------------------------------
    # Per-class one-vs-rest accuracy: for each class C, binarize into
    # C-vs-not-C and report (TP_C + TN_C) / N. Attached to each per_class row.
    # ------------------------------------------------------------------
    for row in per_class:
        cls = row["class"]
        tp_c = confusion[cls][cls]
        support_c = sum(confusion[cls][p] for p in all_labels)   # actual == cls
        pred_c = sum(confusion[a][cls] for a in all_labels)      # predicted == cls
        fn_c = support_c - tp_c
        fp_c = pred_c - tp_c
        tn_c = total - tp_c - fn_c - fp_c
        row["accuracy"] = ((tp_c + tn_c) / total) if total > 0 else 0.0

    # ------------------------------------------------------------------
    # Positive-class differentiation metrics: precision / recall / F1 for
    # each of the 4 positive classes, computed ONLY over cases where BOTH
    # the true label and the predicted label are one of the 4 positive
    # classes. Cases involving `negative_class` (or an unrecognized
    # prediction) on either side are excluded from this subset entirely --
    # this isolates how well the agent tells Ideation / Behavior /
    # Indicator / Attempt apart from each other, independent of the
    # Unsupportive-vs-rest screening decision covered separately in the
    # binary section.
    # ------------------------------------------------------------------
    positive_classes = [l for l in all_labels if l != negative_class]
    pos_pairs = [
        (t, p) for t, p in zip(y_true_all, y_pred_all)
        if t in positive_classes and p in positive_classes
    ]
    pos_y_true = [t for t, _ in pos_pairs]
    pos_y_pred = [p for _, p in pos_pairs]
    n_excluded_positive_only = total - len(pos_pairs)

    per_class_positive_only = _per_class_metrics(pos_y_true, pos_y_pred, positive_classes)
    positive_confusion = _build_confusion_matrix(pos_y_true, pos_y_pred, positive_classes)

    # Aggregate accuracy for this population: plain 4-label classification
    # accuracy over the included (both-sides-positive) cases only.
    accuracy_4class = (
        sum(1 for t, p in pos_pairs if t == p) / len(pos_pairs) if pos_pairs else 0.0
    )

    if len(per_class_positive_only) > 0:
        macro_precision_positive_only = sum(r["precision"] for r in per_class_positive_only) / len(per_class_positive_only)
        macro_recall_positive_only = sum(r["recall"] for r in per_class_positive_only) / len(per_class_positive_only)
        macro_f1_positive_only = sum(r["f1"] for r in per_class_positive_only) / len(per_class_positive_only)
    else:
        macro_precision_positive_only = macro_recall_positive_only = macro_f1_positive_only = 0.0

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
        # Binary (Positive vs Negative) aggregate metrics.
        "accuracy_positive": accuracy_positive,
        "precision_positive": precision_positive,
        "recall_positive": recall_positive,
        "f1_positive": f1_positive,
        # Positive-class differentiation metrics (4-class subset, see above).
        "accuracy_4class": accuracy_4class,
        "precision_4class_macro": macro_precision_positive_only,
        "recall_4class_macro": macro_recall_positive_only,
        "f1_4class_macro": macro_f1_positive_only,
    }

    # Fetch dynamic pricing based on the used model name
    model_name_for_pricing = getattr(agent, 'model_name', 'N/A')
    prompt_price_per_token, completion_price_per_token = _get_model_pricing(model_name_for_pricing)
    
    cost_usd = (metrics_dict["total_prompt_tokens"] * prompt_price_per_token) + (metrics_dict["total_completion_tokens"] * completion_price_per_token)
    metrics_dict["total_cost_usd"] = cost_usd

    result_md_path = eval_dir / "result.md"
    with open(result_md_path, "w", encoding="utf-8") as f:
        f.write("# Evaluation Report — Suicide Risk Detection\n\n")
        f.write(f"**Timestamp:** {datetime.datetime.now().isoformat()}\n")
        f.write(f"**Agent:** {result.agent_name}\n")
        f.write(f"**Model:** {getattr(agent, 'model_name', 'N/A')}\n")
        f.write(f"**Task:** {result.task_name}\n")
        f.write(f"**Dataset:** suicidality_full_dataset_posneg.csv (n={len(all_cases)})\n")
        f.write(f"**Duration:** {result.duration:.4f} s\n")
        f.write(f"**Agent Mode:** {agent_mode}\n")
        f.write(f"**Base Class:** {base_class}\n")
        if host_model:
            f.write(f"**Host Model:** {host_model}\n")
        f.write(f"**Max Host Iterations:** {max_host_iterations}\n")
        f.write(f"**Workers:** {max_workers} | **Retries:** {max_retries} | **Seed:** {seed}\n")
        
        cache_hits = sum(1 for c in all_cases if c.get("from_cache"))
        total_cases = len(all_cases)
        if total_cases > 0:
            cache_hit_rate = (cache_hits / total_cases) * 100
            f.write(f"**Cache Hit Rate:** {cache_hit_rate:.1f}% ({cache_hits}/{total_cases})\n")
        f.write("\n")

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

        f.write("## Error Report\n\n")
        f.write(f"- **error_cases**: {sum(error_counts.values())}\n")
        f.write(f"- **unique_error_keys**: {len(error_counts)}\n\n")
        f.write("```json\n")
        f.write(json.dumps(error_counts, ensure_ascii=False, indent=2))
        f.write("\n```\n\n")

        f.write("## Per-Class Metrics\n\n")
        f.write("*Accuracy is one-vs-rest: (TP + TN) / N for each class treated as C-vs-not-C.*\n\n")
        f.write("| Class | Precision | Recall | F1 | Accuracy | Support |\n|---|---|---|---|---|---|\n")
        for row in per_class:
            f.write(
                f"| {row['class']} | {row['precision']:.4f} | {row['recall']:.4f} | "
                f"{row['f1']:.4f} | {row.get('accuracy', 0.0):.4f} | {row['support']} |\n"
            )
        f.write("\n")

        f.write("## Confusion Matrix\n\n")
        f.write("*Rows = Actual, Columns = Predicted*\n\n")
        _write_confusion_md(f, confusion, all_labels)
        f.write("\n")

        # ---- Positive-class differentiation section ----
        f.write("## Positive-Class Differentiation Metrics\n\n")
        f.write(
            f"*Computed only over cases where BOTH the true label and the predicted label "
            f"are one of the 4 positive classes ({', '.join(f'`{l}`' for l in positive_classes)}). "
            f"Cases where either side is `{negative_class}` (or an unrecognized prediction) are "
            f"excluded from this subset entirely — this isolates how well the agent tells the "
            f"risk levels apart from each other, independent of the Unsupportive-vs-rest screening "
            f"decision covered in the Binary Metrics section below.*\n\n"
        )
        f.write(
            f"- **cases considered**: {len(pos_pairs)} of {total} total "
            f"({n_excluded_positive_only} excluded because the true or predicted label was "
            f"`{negative_class}` or unrecognized)\n\n"
        )
        f.write("| Class | Precision | Recall | F1 | Support (within this subset) |\n|---|---|---|---|---|\n")
        for row in per_class_positive_only:
            f.write(
                f"| {row['class']} | {row['precision']:.4f} | {row['recall']:.4f} | "
                f"{row['f1']:.4f} | {row['support']} |\n"
            )
        f.write("\n")
        f.write(f"- **accuracy_4class**: {accuracy_4class:.6f}\n")
        f.write(f"- **precision_4class_macro**: {macro_precision_positive_only:.6f}\n")
        f.write(f"- **recall_4class_macro**: {macro_recall_positive_only:.6f}\n")
        f.write(f"- **f1_4class_macro**: {macro_f1_positive_only:.6f}\n\n")
        f.write("*precision/recall/f1 macros above are the unweighted mean across the 4 positive classes. "
                "accuracy_4class is the plain 4-label classification accuracy over the included cases only.*\n\n")

        f.write("### Positive-Class Confusion Matrix (4x4 subset)\n\n")
        f.write("*Rows = Actual, Columns = Predicted — restricted to the 4 positive classes*\n\n")
        _write_confusion_md(f, positive_confusion, positive_classes)
        f.write("\n")

        # ---- Binary (Positive vs Negative) section ----
        f.write("## Binary Metrics (Positive vs Negative)\n\n")
        f.write(
            f"*Negative = `{negative_class}`; "
            f"Positive = any of {', '.join(f'`{l}`' for l in all_labels if l != negative_class)}.*\n\n"
        )
        f.write(f"- **accuracy_positive**: {accuracy_positive:.6f}\n")
        f.write(f"- **precision_positive**: {precision_positive:.6f}\n")
        f.write(f"- **recall_positive**: {recall_positive:.6f}\n")
        f.write(f"- **f1_positive**: {f1_positive:.6f}\n")
        f.write(
            f"- **counts**: TP={tp_bin} | TN={tn_bin} | FP={fp_bin} | FN={fn_bin}\n\n"
        )
        f.write("*precision/recall/f1 above are macro-averaged over the Positive and Negative classes.*\n\n")

        f.write("## Binary Confusion Matrix\n\n")
        f.write("*Rows = Actual, Columns = Predicted*\n\n")
        _write_confusion_md(f, binary_confusion, ["Positive", "Negative"])
        f.write("\n")

    all_cases_path = eval_dir / "all_cases.txt"
    with open(all_cases_path, "w", encoding="utf-8") as f:
        f.write(f"All Cases Log — Agent: {result.agent_name} | Task: {result.task_name}\n")
        f.write("=" * 80 + "\n\n")
        for case in all_cases:
            status = "PASS" if int(case.get("correct", 0)) else "FAIL"
            f.write(f"Sample ID : {case['sample_id']} [{status}]\n")
            f.write(f"GT        : {case['ground_truth']}\n")
            f.write(f"Pred      : {case['prediction']}\n")
            if case.get('from_cache'):
                f.write("Cache     : True\n")
            if case.get('total_time') is not None:
                f.write(f"Timing    : TTFT={case.get('ttft') or 0.0:.4f}s | Gen={case.get('generation_time') or 0.0:.4f}s | TPS={case.get('tps') or 0.0:.2f} | Total={case.get('total_time') or 0.0:.4f}s\n")
            if case.get('prompt_tokens') is not None:
                f.write(f"Tokens    : Prompt={case.get('prompt_tokens')} | Completion={case.get('completion_tokens')} | Total={case.get('total_tokens')}\n")
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

    incorrect_log_path = eval_dir / "incorrect_cases.txt"
    if incorrect_cases:
        with open(incorrect_log_path, "w", encoding="utf-8") as f:
            f.write(f"Incorrect Cases Log — Agent: {result.agent_name} | Task: {result.task_name}\n")
            f.write("=" * 80 + "\n\n")
            for case in incorrect_cases:
                f.write(f"Sample ID : {case['sample_id']}\n")
                f.write(f"GT        : {case['ground_truth']}\n")
                f.write(f"Pred      : {case['prediction']}\n")
                if case['error']:
                    f.write(f"Error     : {case['error']}\n")
                f.write(f"Text:\n{case['text']}\n")
                f.write("-" * 40 + "\n\n")
    elif incorrect_log_path.exists():
        incorrect_log_path.unlink()

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
            "id_field": "ID",
            "label_distribution": dict(label_dist),
        },
        "aggregate_metrics": metrics_dict,
        "error": error_counts,
        "per_class_metrics": per_class,
        "confusion_matrix": confusion,
        "per_class_metrics_4class_only": per_class_positive_only,
        "positive_class_confusion_matrix": positive_confusion,
        "binary_confusion_matrix": binary_confusion,
        "binary_counts": binary_counts,
        "total_cost_usd_estimate": cost_usd,
        "retry_history": retry_history,
    }
    with open(meta_json_path, "w", encoding="utf-8") as f:
        json.dump(meta_payload, f, indent=4, default=str)

    print("  ✓ result.md")
    print("  ✓ all_cases.txt")
    print("  ✓ all_cases.csv")
    print("  ✓ error.csv")
    if incorrect_cases:
        print(f"  ✓ incorrect_cases.txt ({len(incorrect_cases)} cases)")
    else:
        print("  ✓ No incorrect cases.")
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
    print(f"Samples:  {len(all_cases)}  |  Incorrect: {len(incorrect_cases)}")
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
    
    # Try to load cases using unified loader (tries JSONL > TXT > CSV)
    all_cases = None
    
    if not all_cases:
        print(f"No case data found in {run_dir}")
        return

    run_cfg = meta.get("run_config", {})
    model_cfg = meta.get("model_config", {})
    ds_cfg = meta.get("dataset_config", {})
    
    agent_mode = model_cfg.get("agent_mode", "single-class")
    base_class = model_cfg.get("base_class", "Unsupportive")
    host_model = model_cfg.get("host_model", None)
    role_models_json = model_cfg.get("role_models_json", None)
    max_host_iterations = model_cfg.get("max_host_iterations", 10)
    
    max_workers = run_cfg.get("max_workers", 5)
    max_retries = run_cfg.get("max_retries", 3)
    seed = run_cfg.get("seed", 42)
    n_samples = run_cfg.get("n_samples", 50)
    
    data_source = ds_cfg.get("source", "unknown")
    cache_mode = "both" # Just default for retry unless stored
    
    retry_indices = [
        i for i, row in enumerate(all_cases)
        if str(row.get("prediction", "")).strip().lower() == "error" or bool(row.get("error"))
    ]

    if not retry_indices:
        print("No Error cases found in all_cases.csv. Nothing to retry.")
        return

    print(f"Found {len(retry_indices)} Error case(s). Retrying only those cases...")

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
    evaluator = ClassificationEvaluator(output_dir=run_dir, cache_mode=cache_mode)

    write_lock = threading.Lock()
    error_csv_path = run_dir / "error.csv"
    _init_error_csv(error_csv_path)
    retry_error_cb = _make_error_event_writer(error_csv_path, write_lock)

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
        show_progress=True,
        exit_on_first_error=True,
        error_cb=lambda event: retry_error_cb({**event, "phase": "retry"}),
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
    base_class: str = "Unsupportive",
    host_model: Optional[str] = None,
    role_models_json: Optional[str] = None,
    max_host_iterations: int = 10,
    data_path: Optional[str] = None,
    resume: bool = False,
    out_dir: Optional[str] = None,
    cache: Optional[str] = None,
    cache_mode: str = "both",
    no_confirm: bool = False,
    info_only: bool = False,
):
    if not info_only:
        print("=" * 80)
        print("Running RSD_15K evaluation — sample limit: " + str(n_samples))

    agent = _build_agent(
        agent_mode=agent_mode,
        base_class=base_class,
        host_model=host_model,
        role_models_json=role_models_json,
        max_host_iterations=max_host_iterations,
    )

    resolved_data_path = Path(data_path) if data_path else (current_file.parent / "data" / "suicidality_full_dataset_posneg.csv")

    if not resolved_data_path.exists():
        print(f"Dataset file not found at: {resolved_data_path}")
        print("Provide --data-path to point at your local rsd_15k.csv.")
        return

    print(f"Loading dataset from: {resolved_data_path}")
    dataset = CSVDataset(name="rsd_15k_sample")
    dataset.load_data(
        source=resolved_data_path,
        text_field="text",
        label_field="sentiment",
        id_field="ID",
    )

    print(f"Total samples loaded: {len(dataset._samples)}")
    total_loaded = len(dataset._samples)
    if n_samples < 0:
        raise ValueError("--n-samples must be >= 0")

    # Deterministic sampling: take rows [0..n-1] in file order.
    dataset._samples = dataset._samples[:n_samples]
    dataset.assert_unique_ids(
        expected_count=len(dataset._samples),
        context=f"run_main_eval.after_sampling(head={n_samples}, loaded={total_loaded})",
    )

    import json

    if info_only:
        evaluator = ClassificationEvaluator(output_dir=None, cache_mode=cache_mode)
        if cache:
            evaluator.global_cache.base_runs_dir = Path(cache)
        else:
            evaluator.global_cache.base_runs_dir = current_file.parent / "_runs"
            
        cache_info = evaluator.get_cache_info(
            agent=agent, dataset=dataset, agent_mode=agent_mode, base_class=base_class, show_progress=True
        )
        print(json.dumps(cache_info))
        sys.exit(0)

    completed_cases = []
    eval_dir = None

    if not eval_dir:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_suffix = getattr(agent, 'model_name', 'unknown').split('/')[-1]
        dir_suffix = f"{model_suffix}_{agent_mode}"
        if out_dir:
            eval_dir = Path(out_dir) / f"run_{timestamp}_{dir_suffix}"
            eval_dir.mkdir(parents=True, exist_ok=True)
        else:
            eval_dir = current_file.parent / "_runs" / f"run_{timestamp}_{dir_suffix}"
            eval_dir.mkdir(parents=True, exist_ok=True)
    
    partial_path = eval_dir / "all_cases_partial.jsonl"

    write_lock = threading.Lock()
    f_out = open(partial_path, "a", encoding="utf-8")
    error_csv_path = eval_dir / "error.csv"
    _init_error_csv(error_csv_path)
    error_event_cb = _make_error_event_writer(error_csv_path, write_lock)

    def write_partial(sample_metric):
        with write_lock:
            f_out.write(json.dumps(sample_metric, ensure_ascii=False) + "\n")
            f_out.flush()

    print(f"Samples limited to: {len(dataset._samples)}")

    evaluator = ClassificationEvaluator(output_dir=eval_dir, cache_mode=cache_mode)
    if cache:
        evaluator.global_cache.base_runs_dir = Path(cache)
    else:
        evaluator.global_cache.base_runs_dir = current_file.parent / "_runs"
        
    if len(dataset._samples) > 0:
        if not no_confirm:
            cache_info = evaluator.get_cache_info(
                agent=agent, dataset=dataset, agent_mode=agent_mode, base_class=base_class
            )
            print("\n--- Confirm Run ---")
            print(f"Model: {getattr(agent, 'model_name', 'unknown')}")
            print(f"Mode: {agent_mode}")
            print(f"Base Class: {base_class}")
            print(f"Data Path: {resolved_data_path}")
            print(f"Total Samples: {cache_info['total']}")
            print(f"Cached: {cache_info['cached']}")
            print(f"Uncached (Evaluating): {cache_info['uncached']}")
            print(f"Workers: {max_workers}")
            val = input("Press Enter to continue, or 'q' to quit: ")
            if val.strip().lower() == 'q':
                print("Run aborted.")
                sys.exit(0)

        print(f"\nStarting evaluation — workers={max_workers}, retries={max_retries}, seed={seed} ...")
        result = evaluator.evaluate(
            agent=agent,
            dataset=dataset,
            task_name="suicide_risk_classification",
            version="v1.0",
            max_workers=max_workers,
            max_retries=max_retries,
            seed=seed,
            n_samples=len(dataset._samples),
            agent_mode=agent_mode,
            base_class=base_class,
            show_progress=True,
            exit_on_first_error=True,
            partial_cb=write_partial,
            error_cb=lambda event: error_event_cb({**event, "phase": "main"}),
        )
    else:
        print("\nAll samples cached. Evaluation completely bypassed.")
        from single_agent_frameworks.evaluator.base import EvaluationResult, EvaluationMetrics
        result = EvaluationResult(
            agent_name=getattr(agent, "name", "unknown"),
            dataset_name=dataset.name,
            metrics=EvaluationMetrics(metrics={}, per_sample_metrics=[], metadata={}),
            duration=0.0,
            timestamp=datetime.datetime.now(),
            task_name="suicide_risk_classification",
            version="v1.0"
        )

    print("\nEvaluation complete. Collecting results...")
    all_cases = []

    merged_metrics_raw = completed_cases + getattr(result.metrics, "per_sample_metrics", [])
    merged_metrics_dict = {}
    for sm in merged_metrics_raw:
        merged_metrics_dict[str(sm["sample_id"])] = sm
    merged_metrics = list(merged_metrics_dict.values())
    # Ensure deterministic ordering for analysis/debugging (completion order can be arbitrary).
    def _sid_sort_key(m: Dict[str, Any]):
        sid = str(m.get("sample_id", ""))
        try:
            return (0, int(sid))
        except Exception:
            return (1, sid)

    merged_metrics.sort(key=_sid_sort_key)

    result.metrics.per_sample_metrics = merged_metrics

    for sample_res in merged_metrics:
        sid = sample_res["sample_id"]
        pred = sample_res["prediction"]
        gt = sample_res["ground_truth"]
        
        # Handle both "accuracy" (from evaluator) and "correct" (from cache)
        acc = sample_res.get("accuracy")
        if acc is None:
            correct_val = sample_res.get("correct", 0)
            acc = float(correct_val)

        status_str = "PASS" if float(acc) == 1.0 else "FAIL"
        
        if pred == "Error":
            print(f"  {sid}: GT={gt} | Pred={pred} [{status_str}]")
            print(f"    Error: {sample_res.get('error', 'unknown')}")

        case_data = {
            "sample_id": sid,
            "text": sample_res.get("input_text", sample_res.get("text", "Text not found")),
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
            "from_cache": sample_res.get("from_cache", False),
        }
        all_cases.append(case_data)

    print(f"\nSaving reports to: {eval_dir}")

    _write_reports(
        eval_dir=eval_dir,
        result=result,
        agent=agent,
        all_cases=all_cases,
        data_path=resolved_data_path,
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

    f_out.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate suicide risk detection on RSD_15K")
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Optional path to rsd_15k.csv if not located at the default ./data/rsd_15k.csv.",
    )
    parser.add_argument("--n-samples", type=int, default=29727, help="Number of samples to evaluate (default: 50)")
    parser.add_argument("--max-workers", type=int, default=5, help="Parallel workers (default: 5)")
    parser.add_argument("--max-retries", type=int, default=3, help="Retries per sample on failure (default: 3)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--agent-mode",
        type=str,
        default="single-class",
        choices=["single-class", "multi-class-chance", "base-vs-rest-binary", "multi-agent-host", "per-class-score"],
        help="Agent output mode: single label or structured class probabilities.",
    )
    parser.add_argument(
        "--base-class",
        type=str,
        default="Unsupportive",
        choices=["Unsupportive","Indicator", "Ideation", "Behavior", "Attempt"],
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
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory to store run directory")
    parser.add_argument("--continue", dest="resume", action="store_true", help="Resume partial runs from latest model folder")
    parser.add_argument("--cache", type=str, default=None, help="Directory to load cache from previous runs of the same config")
    parser.add_argument("--cache-mode", type=str, default="both", choices=["runs_only", "cache_file", "both", "none"], help="Global cache operation mode")
    parser.add_argument("-nc", "--no-confirm", action="store_true", help="Bypass interactive run confirmation prompt")
    parser.add_argument("--info-only", action="store_true", help="Just print the cache info payload and exit")
    args = parser.parse_args()

    if args.retry_failed:
        chosen_meta = Path(args.meta_path) if args.meta_path else _latest_meta_path(current_file.parent)
        if chosen_meta is None:
            print("Could not find a Meta.json to retry from. Provide --meta-path.")
            sys.exit(1)
        _run_retry_failed(chosen_meta)
    else:
        run_main_eval(
            data_path=args.data_path,
            n_samples=args.n_samples,
            max_workers=args.max_workers,
            max_retries=args.max_retries,
            seed=args.seed,
            agent_mode=args.agent_mode,
            base_class=args.base_class,
            host_model=args.host_model,
            role_models_json=args.role_models_json,
            max_host_iterations=args.max_host_iterations,
            resume=args.resume,
            out_dir=args.out_dir,
            cache=args.cache,
            cache_mode=args.cache_mode,
            no_confirm=args.no_confirm,
            info_only=args.info_only,
        )

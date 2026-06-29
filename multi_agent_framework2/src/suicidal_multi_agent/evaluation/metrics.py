"""Metrics and runtime summaries for classification runs."""

from __future__ import annotations

from typing import Any

from ..core.labels import VALID_LABELS


def compute_classification_metrics(
    y_true: list[str],
    y_pred: list[str],
    *,
    model_name: str | None = None,
) -> dict[str, Any]:
    """Compute model-level and one-vs-rest per-label classification metrics."""
    labels = list(VALID_LABELS)
    matrix = {actual: {pred: 0 for pred in labels + ["Error"]} for actual in labels}
    correct = 0

    for actual, predicted in zip(y_true, y_pred):
        pred_key = predicted if predicted in labels else "Error"
        if actual in matrix:
            matrix[actual][pred_key] += 1
        if actual == predicted:
            correct += 1

    per_class = []
    for label in labels:
        tp = matrix[label][label]
        fp = sum(matrix[other][label] for other in labels if other != label)
        fn = sum(matrix[label][pred] for pred in labels + ["Error"] if pred != label)
        tn = sum(
            matrix[actual][pred]
            for actual in labels
            for pred in labels + ["Error"]
            if actual != label and pred != label
        )
        support = sum(matrix[label].values())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
        label_accuracy = (tp + tn) / len(y_true) if y_true else 0.0
        per_class.append(
            {
                "model_name": model_name,
                "label_name": label,
                "accuracy": label_accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "f1_score": f1,
                "support": support,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
            }
        )

    total = len(y_true)
    failed_records = sum(1 for pred in y_pred if pred not in labels)
    accuracy = correct / total if total else 0.0
    macro_precision = sum(row["precision"] for row in per_class) / len(per_class)
    macro_recall = sum(row["recall"] for row in per_class) / len(per_class)
    macro_f1 = sum(row["f1"] for row in per_class) / len(per_class)
    model_summary = {
        "model_name": model_name,
        "accuracy": accuracy,
        "f1_score": macro_f1,
        "precision": macro_precision,
        "recall": macro_recall,
    }
    return {
        "model_name": model_name,
        "accuracy": accuracy,
        "precision": macro_precision,
        "recall": macro_recall,
        "f1_score": macro_f1,
        "macro_f1": macro_f1,
        "model_summary": model_summary,
        "per_class": per_class,
        "per_label": per_class,
        "confusion_matrix": matrix,
        "total": total,
        "correct": correct,
        "failed_records": failed_records,
        "failure_rate": failed_records / total if total else 0.0,
    }


def summarize_runtime(llm_calls: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize local runtime, token, cost, and optional provider timing telemetry."""
    latencies = [float(call.get("latency_s", 0.0)) for call in llm_calls]
    prompt_tokens = [int(call.get("prompt_tokens") or 0) for call in llm_calls]
    completion_tokens = [int(call.get("completion_tokens") or 0) for call in llm_calls]
    total_tokens = [int(call.get("total_tokens") or (call.get("prompt_tokens") or 0) + (call.get("completion_tokens") or 0)) for call in llm_calls]
    cached_tokens = [_cached_tokens(call) for call in llm_calls]
    cache_write_tokens = [_cache_write_tokens(call) for call in llm_calls]
    costs = [float(call.get("cost") or call.get("cost_usd") or 0.0) for call in llm_calls]
    ttfts = _numeric_values(llm_calls, "ttft_s")
    provider_latencies = _numeric_values(llm_calls, "provider_latency_s")
    generation_times = _numeric_values(llm_calls, "generation_time_s")
    moderation_latencies = _numeric_values(llm_calls, "moderation_latency_s")
    completion_tps = _numeric_values(llm_calls, "completion_tps")
    e2e_completion_tps = [
        value if value is not None else _derived_e2e_completion_tps(call)
        for call in llm_calls
        for value in [_optional_float(call.get("e2e_completion_tps"))]
    ]
    e2e_completion_tps = [value for value in e2e_completion_tps if value is not None]
    failed_attempts = [call for call in llm_calls if call.get("error")]

    def avg(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    request_count = len(llm_calls)
    return {
        "call_count": request_count,
        "llm_request_count": request_count,
        "failed_llm_attempts": len(failed_attempts),
        "latency_s": {
            "min": min(latencies) if latencies else 0.0,
            "max": max(latencies) if latencies else 0.0,
            "avg": avg(latencies),
            "total": sum(latencies),
        },
        "tokens": {
            "prompt_total": sum(prompt_tokens),
            "completion_total": sum(completion_tokens),
            "total": sum(total_tokens),
            "cached_total": sum(cached_tokens),
            "cache_write_total": sum(cache_write_tokens),
            "avg_prompt_per_request": avg(prompt_tokens),
            "avg_completion_per_request": avg(completion_tokens),
            "avg_total_per_request": avg(total_tokens),
            "avg_cached_per_request": avg(cached_tokens),
            "avg_cache_write_per_request": avg(cache_write_tokens),
        },
        "cost_usd": {
            "total": sum(costs),
            "avg_per_call": avg(costs),
        },
        "provider_timing": {
            "ttft_s": _stats(ttfts),
            "generation_time_s": _stats(generation_times),
            "provider_latency_s": _stats(provider_latencies),
            "moderation_latency_s": _stats(moderation_latencies),
            "completion_tps": _stats(completion_tps),
            "e2e_completion_tps": _stats(e2e_completion_tps),
        },
    }


def _cached_tokens(call: dict[str, Any]) -> int:
    details = call.get("prompt_tokens_details") or {}
    return int(
        details.get("cached_tokens")
        or details.get("cache_read")
        or details.get("cache_read_tokens")
        or 0
    )


def _cache_write_tokens(call: dict[str, Any]) -> int:
    details = call.get("prompt_tokens_details") or {}
    return int(details.get("cache_write_tokens") or details.get("cache_write") or 0)


def _numeric_values(calls: list[dict[str, Any]], key: str) -> list[float]:
    values = []
    for call in calls:
        value = _optional_float(call.get(key))
        if value is not None:
            values.append(value)
    return values


def _optional_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _derived_e2e_completion_tps(call: dict[str, Any]) -> float | None:
    completion_tokens = _optional_float(call.get("completion_tokens"))
    latency_s = _optional_float(call.get("latency_s"))
    if not completion_tokens or not latency_s:
        return None
    return completion_tokens / latency_s


def _stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "max": None, "avg": None}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "avg": sum(values) / len(values),
    }

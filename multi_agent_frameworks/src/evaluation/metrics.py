"""Metrics and runtime summaries for classification runs."""

from __future__ import annotations

from collections import Counter
from typing import Any

from domain.labels import FINAL_LABELS, UNSUPPORTIVE_LABEL, VALID_LABELS, label_to_suicidal_intent


def compute_final_label_confusion_matrix(
    y_true: list[str],
    y_pred: list[str],
) -> dict[str, Any]:
    """Compute the final-label confusion matrix and failure guardrail counts."""
    labels = list(FINAL_LABELS)
    matrix = {actual: {pred: 0 for pred in labels + ["Error"]} for actual in labels}

    for actual, predicted in zip(y_true, y_pred):
        pred_key = predicted if predicted in labels else "Error"
        if actual in matrix:
            matrix[actual][pred_key] += 1

    total = len(y_true)
    failed_records = sum(1 for pred in y_pred if pred not in labels)
    return {
        "confusion_matrix": matrix,
        "total": total,
        "failed_records": failed_records,
        "failure_rate": failed_records / total if total else 0.0,
    }


def compute_binary_classification_metrics(
    y_true: list[bool],
    y_pred: list[bool | None],
    *,
    model_name: str | None = None,
) -> dict[str, Any]:
    """Compute one positive-oriented confusion matrix for a binary task."""
    labels = (False, True)
    keys = {False: "false", True: "true"}
    matrix = {keys[actual]: {keys[pred]: 0 for pred in labels} | {"Error": 0} for actual in labels}
    correct = 0

    for actual, predicted in zip(y_true, y_pred):
        actual_key = keys[actual]
        pred_key = keys[predicted] if isinstance(predicted, bool) else "Error"
        matrix[actual_key][pred_key] += 1
        if actual is predicted:
            correct += 1

    total = len(y_true)
    failed_records = sum(1 for value in y_pred if not isinstance(value, bool))
    accuracy = correct / total if total else 0.0
    positive_class_counts = {
        "tp": matrix["true"]["true"],
        "tn": matrix["false"]["false"],
        "fp": matrix["false"]["true"],
        "fn": matrix["true"]["false"],
    }
    tp = positive_class_counts["tp"]
    tn = positive_class_counts["tn"]
    fp = positive_class_counts["fp"]
    fn = positive_class_counts["fn"]
    positive_support = sum(matrix["true"].values())
    negative_support = sum(matrix["false"].values())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    binary_summary = {
        "model_name": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f1_score": f1,
        "support": positive_support,
        "positive_support": positive_support,
        "negative_support": negative_support,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }
    ground_truth_classes = sorted({value for value in y_true})
    has_both_ground_truth_classes = len(ground_truth_classes) == 2
    return {
        "model_name": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "positive_precision": precision,
        "positive_recall": recall,
        "positive_f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "positive_class_counts": positive_class_counts,
        "binary_summary": binary_summary,
        "confusion_matrix": matrix,
        "total": total,
        "correct": correct,
        "failed_records": failed_records,
        "failure_rate": failed_records / total if total else 0.0,
        "ground_truth_counts": {
            "false": sum(value is False for value in y_true),
            "true": sum(value is True for value in y_true),
        },
        "has_both_ground_truth_classes": has_both_ground_truth_classes,
        "metric_limitation": (
            None
            if has_both_ground_truth_classes or not y_true
            else "Only one suicidal_intent ground-truth class is present; the confusion matrix is valid, "
            "but metrics for the missing ground-truth class cannot be estimated."
        ),
    }


def compute_suicidal_intent_metrics(
    y_true_labels: list[str],
    y_pred_labels: list[str],
    *,
    model_name: str | None = None,
) -> dict[str, Any]:
    """Group final labels into one fixed positive-vs-negative classification task."""
    y_true = [label_to_suicidal_intent(label) for label in y_true_labels]
    y_pred = [
        label_to_suicidal_intent(label) if label in FINAL_LABELS else None
        for label in y_pred_labels
    ]
    metrics = compute_binary_classification_metrics(y_true, y_pred, model_name=model_name)
    metrics["class_mapping"] = {
        "positive": list(VALID_LABELS),
        "negative": [UNSUPPORTIVE_LABEL],
    }
    return metrics


def compute_positive_subclassification_metrics(
    y_true_labels: list[str],
    y_pred_labels: list[str],
) -> dict[str, Any]:
    """Evaluate phase classification where actual and predicted labels are positive."""
    labels = list(VALID_LABELS)
    matrix = {actual: {predicted: 0 for predicted in labels} for actual in labels}

    for actual, predicted in zip(y_true_labels, y_pred_labels):
        if actual not in matrix:
            continue
        if predicted in labels:
            matrix[actual][predicted] += 1

    support = sum(sum(row.values()) for row in matrix.values())
    correct = sum(matrix[label][label] for label in labels)
    per_label: dict[str, dict[str, float | int]] = {}
    for label in labels:
        tp = matrix[label][label]
        fp = sum(matrix[actual][label] for actual in labels if actual != label)
        fn = sum(
            matrix[label][predicted]
            for predicted in labels
            if predicted != label
        )
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        per_label[label] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "support": tp + fn,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        }

    label_count = len(labels)
    macro_precision = sum(value["precision"] for value in per_label.values()) / label_count
    macro_recall = sum(value["recall"] for value in per_label.values()) / label_count
    macro_f1 = sum(value["f1_score"] for value in per_label.values()) / label_count
    ground_truth_counts = {
        label: sum(matrix[label].values())
        for label in labels
    }
    missing_ground_truth_labels = [
        label for label, count in ground_truth_counts.items() if count == 0
    ]
    summary = {
        "accuracy": correct / support if support else 0.0,
        "precision": macro_precision,
        "recall": macro_recall,
        "f1_score": macro_f1,
        "support": support,
        "correct": correct,
        "averaging": "macro",
    }
    return {
        **summary,
        "summary": summary,
        "labels": labels,
        "confusion_matrix": matrix,
        "per_label": per_label,
        "ground_truth_counts": ground_truth_counts,
        "has_all_ground_truth_labels": not missing_ground_truth_labels,
        "metric_limitation": (
            "No records have both a positive ground-truth label and a positive predicted label."
            if not support
            else (
                "The following positive labels are absent from ground truth and contribute zero to the fixed four-label macro average: "
                + ", ".join(missing_ground_truth_labels)
                if missing_ground_truth_labels
                else None
            )
        ),
    }


def summarize_runtime(
    llm_calls: list[dict[str, Any]],
    *,
    agent_execution_duration_s: float = 0.0,
    real_execution_duration_s: float = 0.0,
) -> dict[str, Any]:
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
    errors_by_type = Counter(
        call.get("error", {}).get("error_type", "Unknown")
        for call in failed_attempts
    )
    json_parsing_errors = sum(
        count
        for error_type, count in errors_by_type.items()
        if error_type
        in {
            "JSONDecodeError",
            "ValidationError",
            "StructuredOutputValidationError",
            "MultipleStructuredOutputsError",
        }
    )

    def avg(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    request_count = len(llm_calls)
    return {
        "agent_execution_duration_s": float(agent_execution_duration_s),
        "real_execution_duration_s": float(real_execution_duration_s),
        "call_count": request_count,
        "llm_request_count": request_count,
        "failed_llm_attempts": len(failed_attempts),
        "json_parsing_errors": json_parsing_errors,
        "errors_by_type": dict(sorted(errors_by_type.items())),
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

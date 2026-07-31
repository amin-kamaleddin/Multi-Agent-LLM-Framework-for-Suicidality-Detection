import pytest

from evaluation.metrics import (
    compute_binary_classification_metrics,
    compute_final_label_confusion_matrix,
    compute_positive_subclassification_metrics,
    compute_suicidal_intent_metrics,
    summarize_runtime,
)


def test_final_label_confusion_matrix_fixture():
    metrics = compute_final_label_confusion_matrix(
        ["Indicator", "Ideation", "Behavior", "Attempt"],
        ["Indicator", "Behavior", "Behavior", "Error"],
    )

    assert metrics["total"] == 4
    assert metrics["failure_rate"] == 0.25
    assert metrics["failed_records"] == 1
    assert metrics["confusion_matrix"]["Attempt"]["Error"] == 1
    assert set(metrics) == {
        "confusion_matrix",
        "total",
        "failed_records",
        "failure_rate",
    }


def test_runtime_summary_fixture():
    summary = summarize_runtime(
        [
            {
                "latency_s": 1.0,
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "cost": 0.01,
                "prompt_tokens_details": {"cached_tokens": 4, "cache_write_tokens": 2},
                "ttft_s": 0.2,
                "generation_time_s": 0.5,
                "provider_latency_s": 0.7,
                "moderation_latency_s": 0.1,
                "completion_tps": 10.0,
            },
            {
                "latency_s": 3.0,
                "prompt_tokens": 20,
                "completion_tokens": 5,
                "total_tokens": 25,
                "cost": 0.02,
                "prompt_tokens_details": {"cached_tokens": 6, "cache_write_tokens": 3},
                "e2e_completion_tps": 2.5,
                "error": {"error_type": "ValidationError"},
            },
        ],
        agent_execution_duration_s=4.25,
        real_execution_duration_s=2.75,
    )

    assert summary["agent_execution_duration_s"] == 4.25
    assert summary["real_execution_duration_s"] == 2.75
    assert summary["call_count"] == 2
    assert summary["llm_request_count"] == 2
    assert summary["failed_llm_attempts"] == 1
    assert summary["json_parsing_errors"] == 1
    assert summary["errors_by_type"] == {"ValidationError": 1}
    assert summary["latency_s"]["min"] == 1.0
    assert summary["latency_s"]["max"] == 3.0
    assert summary["latency_s"]["avg"] == 2.0
    assert summary["tokens"]["total"] == 40
    assert summary["tokens"]["prompt_total"] == 30
    assert summary["tokens"]["completion_total"] == 10
    assert summary["tokens"]["cached_total"] == 10
    assert summary["tokens"]["cache_write_total"] == 5
    assert summary["tokens"]["avg_total_per_request"] == 20
    assert summary["tokens"]["avg_prompt_per_request"] == 15
    assert summary["tokens"]["avg_completion_per_request"] == 5
    assert summary["tokens"]["avg_cached_per_request"] == 5
    assert summary["cost_usd"]["total"] == 0.03
    assert summary["provider_timing"]["ttft_s"] == {"count": 1, "min": 0.2, "max": 0.2, "avg": 0.2}
    assert summary["provider_timing"]["generation_time_s"] == {"count": 1, "min": 0.5, "max": 0.5, "avg": 0.5}
    assert summary["provider_timing"]["provider_latency_s"] == {"count": 1, "min": 0.7, "max": 0.7, "avg": 0.7}
    assert summary["provider_timing"]["moderation_latency_s"] == {"count": 1, "min": 0.1, "max": 0.1, "avg": 0.1}
    assert summary["provider_timing"]["completion_tps"] == {"count": 1, "min": 10.0, "max": 10.0, "avg": 10.0}
    assert summary["provider_timing"]["e2e_completion_tps"] == {"count": 2, "min": 2.5, "max": 5.0, "avg": 3.75}


def test_runtime_summary_counts_langchain_structured_output_errors():
    summary = summarize_runtime(
        [
            {"error": {"error_type": "StructuredOutputValidationError"}},
            {"error": {"error_type": "MultipleStructuredOutputsError"}},
        ]
    )

    assert summary["failed_llm_attempts"] == 2
    assert summary["json_parsing_errors"] == 2


def test_binary_classification_metrics_include_confusion_matrix_and_limitation():
    metrics = compute_binary_classification_metrics(
        [False, False, True, True],
        [False, True, True, None],
        model_name="test-model",
    )

    assert metrics["accuracy"] == 0.5
    assert metrics["failed_records"] == 1
    assert metrics["confusion_matrix"] == {
        "false": {"false": 1, "true": 1, "Error": 0},
        "true": {"false": 0, "true": 1, "Error": 1},
    }
    assert metrics["positive_precision"] == 0.5
    assert metrics["positive_recall"] == 1.0
    assert metrics["positive_f1"] == 2 / 3
    assert metrics["positive_class_counts"] == {"tp": 1, "tn": 1, "fp": 1, "fn": 0}
    assert "per_class" not in metrics
    assert metrics["binary_summary"] == {
        "model_name": "test-model",
        "accuracy": 0.5,
        "precision": 0.5,
        "recall": 1.0,
        "f1": 2 / 3,
        "f1_score": 2 / 3,
        "support": 2,
        "positive_support": 2,
        "negative_support": 2,
        "tp": 1,
        "fp": 1,
        "tn": 1,
        "fn": 0,
    }
    assert metrics["has_both_ground_truth_classes"] is True
    assert metrics["metric_limitation"] is None

    positive_only = compute_binary_classification_metrics([True, True], [True, True])
    assert positive_only["has_both_ground_truth_classes"] is False
    assert positive_only["metric_limitation"]


def test_suicidal_intent_metrics_group_four_labels_as_positive_and_unsupportive_as_negative():
    metrics = compute_suicidal_intent_metrics(
        ["Indicator", "Ideation", "Unsupportive", "Unsupportive"],
        ["Behavior", "Unsupportive", "Attempt", "Unsupportive"],
    )

    assert metrics["class_mapping"] == {
        "positive": ["Indicator", "Ideation", "Behavior", "Attempt"],
        "negative": ["Unsupportive"],
    }
    assert metrics["confusion_matrix"] == {
        "false": {"false": 1, "true": 1, "Error": 0},
        "true": {"false": 1, "true": 1, "Error": 0},
    }
    assert metrics["positive_class_counts"] == {"tp": 1, "tn": 1, "fp": 1, "fn": 1}


def test_negative_only_dataset_cannot_have_suicidal_intent_true_positives():
    metrics = compute_suicidal_intent_metrics(
        ["Unsupportive", "Unsupportive", "Unsupportive"],
        ["Unsupportive", "Indicator", "Error"],
    )

    assert metrics["confusion_matrix"] == {
        "false": {"false": 1, "true": 1, "Error": 1},
        "true": {"false": 0, "true": 0, "Error": 0},
    }
    assert metrics["positive_class_counts"] == {"tp": 0, "tn": 1, "fp": 1, "fn": 0}
    assert metrics["binary_summary"]["positive_support"] == 0
    assert metrics["binary_summary"]["negative_support"] == 3


def test_positive_subclassification_metrics_use_four_positive_labels_only():
    metrics = compute_positive_subclassification_metrics(
        ["Indicator", "Ideation", "Behavior", "Attempt", "Attempt", "Unsupportive"],
        ["Indicator", "Behavior", "Behavior", "Attempt", "Error", "Indicator"],
    )

    assert metrics["labels"] == ["Indicator", "Ideation", "Behavior", "Attempt"]
    assert set(metrics) == {
        "accuracy", "precision", "recall", "f1_score", "support", "correct",
        "averaging", "summary", "labels", "confusion_matrix", "per_label",
        "ground_truth_counts", "has_all_ground_truth_labels", "metric_limitation",
    }
    assert metrics["support"] == 4
    assert metrics["correct"] == 3
    assert metrics["accuracy"] == 0.75
    assert metrics["precision"] == 0.625
    assert metrics["recall"] == 0.75
    assert metrics["f1_score"] == pytest.approx(2 / 3)
    assert metrics["averaging"] == "macro"
    assert metrics["confusion_matrix"] == {
        "Indicator": {"Indicator": 1, "Ideation": 0, "Behavior": 0, "Attempt": 0},
        "Ideation": {"Indicator": 0, "Ideation": 0, "Behavior": 1, "Attempt": 0},
        "Behavior": {"Indicator": 0, "Ideation": 0, "Behavior": 1, "Attempt": 0},
        "Attempt": {"Indicator": 0, "Ideation": 0, "Behavior": 0, "Attempt": 1},
    }
    assert metrics["per_label"] == {
        "Indicator": {
            "tp": 1, "fp": 0, "fn": 0, "support": 1,
            "precision": 1.0, "recall": 1.0, "f1_score": 1.0,
        },
        "Ideation": {
            "tp": 0, "fp": 0, "fn": 1, "support": 1,
            "precision": 0.0, "recall": 0.0, "f1_score": 0.0,
        },
        "Behavior": {
            "tp": 1, "fp": 1, "fn": 0, "support": 1,
            "precision": 0.5, "recall": 1.0,
            "f1_score": pytest.approx(2 / 3),
        },
        "Attempt": {
            "tp": 1, "fp": 0, "fn": 0, "support": 1,
            "precision": 1.0, "recall": 1.0, "f1_score": 1.0,
        },
    }
    assert metrics["ground_truth_counts"] == {
        "Indicator": 1,
        "Ideation": 1,
        "Behavior": 1,
        "Attempt": 1,
    }
    assert metrics["has_all_ground_truth_labels"] is True
    assert metrics["metric_limitation"] is None


def test_positive_subclassification_metrics_report_no_positive_ground_truth():
    metrics = compute_positive_subclassification_metrics(
        ["Unsupportive", "Unsupportive"],
        ["Indicator", "Unsupportive"],
    )

    assert metrics["support"] == 0
    assert metrics["accuracy"] == 0.0
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1_score"] == 0.0
    assert metrics["has_all_ground_truth_labels"] is False
    assert metrics["metric_limitation"]

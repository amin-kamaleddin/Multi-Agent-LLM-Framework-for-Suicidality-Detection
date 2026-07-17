from suicidal_multi_agent.evaluation.metrics import (
    compute_binary_classification_metrics,
    compute_classification_metrics,
    summarize_runtime,
)


def test_classification_metrics_fixture():
    metrics = compute_classification_metrics(
        ["Indicator", "Ideation", "Behavior", "Attempt"],
        ["Indicator", "Behavior", "Behavior", "Error"],
        model_name="test-model",
    )

    assert metrics["model_name"] == "test-model"
    assert metrics["total"] == 4
    assert metrics["correct"] == 2
    assert metrics["accuracy"] == 0.5
    assert metrics["failure_rate"] == 0.25
    assert metrics["failed_records"] == 1
    assert metrics["confusion_matrix"]["Attempt"]["Error"] == 1
    assert metrics["model_summary"] == {
        "model_name": "test-model",
        "accuracy": metrics["accuracy"],
        "f1_score": metrics["f1_score"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
    }

    by_label = {row["label_name"]: row for row in metrics["per_class"]}
    assert by_label["Indicator"]["model_name"] == "test-model"
    assert by_label["Indicator"]["accuracy"] == 1.0
    assert by_label["Indicator"]["f1"] == 1.0
    assert by_label["Indicator"]["f1_score"] == 1.0
    assert by_label["Ideation"]["recall"] == 0.0
    assert by_label["Behavior"]["precision"] == 0.5
    assert by_label["Attempt"]["fn"] == 1


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
        ]
    )

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
    assert metrics["positive_recall"] == 0.5
    assert metrics["positive_f1"] == 0.5
    assert metrics["has_both_ground_truth_classes"] is True
    assert metrics["metric_limitation"] is None

    positive_only = compute_binary_classification_metrics([True, True], [True, True])
    assert positive_only["has_both_ground_truth_classes"] is False
    assert positive_only["metric_limitation"]

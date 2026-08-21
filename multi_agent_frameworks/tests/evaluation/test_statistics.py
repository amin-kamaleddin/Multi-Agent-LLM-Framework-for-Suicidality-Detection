import pandas as pd
import pytest

from evaluation.statistics import analyze_predictions, paired_primary_endpoint_comparison


def _predictions(predictions: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": ["a", "b", "c", "d", "e", "f", "g", "h"],
            "users": ["u1", "u1", "u2", "u2", "u3", "u3", "u4", "u4"],
            "ground_truth": [
                "Ideation", "Unsupportive", "Behavior", "Unsupportive",
                "Attempt", "Unsupportive", "Indicator", "Unsupportive",
            ],
            "prediction": predictions,
        }
    )


def test_clustered_statistics_reports_observed_and_prevalence_adjusted_ppv_npv():
    analysis = analyze_predictions(
        _predictions(["Ideation", "Unsupportive", "Unsupportive", "Unsupportive", "Attempt", "Ideation", "Indicator", "Unsupportive"]),
        bootstrap_replicates=200,
        bootstrap_seed=7,
        target_prevalences=(0.10, 0.50),
    )

    assert analysis["prespecification"]["primary_endpoint"]["name"] == "suicidal_intent_sensitivity"
    assert analysis["bootstrap"]["unit"] == "users"
    assert analysis["bootstrap"]["cluster_count"] == 4
    assert analysis["point_estimates"]["sensitivity"] == pytest.approx(0.75)
    assert analysis["point_estimates"]["specificity"] == pytest.approx(0.75)
    assert analysis["point_estimates"]["ppv"] == pytest.approx(0.75)
    assert analysis["point_estimates"]["npv"] == pytest.approx(0.75)
    assert analysis["confidence_intervals_95"]["sensitivity"]["valid_replicates"] > 0
    assert analysis["prevalence_adjusted_predictive_values"][0]["target_prevalence"] == 0.10
    assert analysis["prevalence_adjusted_predictive_values"][0]["ppv"] < analysis["prevalence_adjusted_predictive_values"][1]["ppv"]
    assert analysis["prevalence_adjusted_predictive_values"][0]["npv"] > analysis["prevalence_adjusted_predictive_values"][1]["npv"]


def test_paired_cluster_bootstrap_reports_candidate_minus_reference_difference():
    reference = _predictions(["Unsupportive", "Unsupportive", "Unsupportive", "Unsupportive", "Attempt", "Unsupportive", "Indicator", "Unsupportive"])
    candidate = _predictions(["Ideation", "Unsupportive", "Behavior", "Unsupportive", "Attempt", "Unsupportive", "Indicator", "Unsupportive"])

    comparison = paired_primary_endpoint_comparison(
        reference,
        candidate,
        bootstrap_replicates=200,
        bootstrap_seed=7,
    )

    assert comparison["direction"] == "candidate_minus_reference"
    assert comparison["reference"]["value"] == pytest.approx(0.5)
    assert comparison["candidate"]["value"] == pytest.approx(1.0)
    assert comparison["difference"] == pytest.approx(0.5)
    assert comparison["difference_ci_95"]["valid_replicates"] > 0
    assert comparison["interpretation"] == "inconclusive"
    assert comparison["difference_ci_95"]["lower"] <= 0 <= comparison["difference_ci_95"]["upper"]


def test_clustered_statistics_rejects_missing_user_clusters():
    with pytest.raises(ValueError, match="users"):
        analyze_predictions(
            _predictions(["Ideation"] * 8).drop(columns="users"),
            bootstrap_replicates=100,
        )

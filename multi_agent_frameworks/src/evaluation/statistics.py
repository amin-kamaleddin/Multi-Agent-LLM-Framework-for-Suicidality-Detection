"""Prespecified clustered statistical analysis for completed evaluation runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from domain.labels import FINAL_LABELS, VALID_LABELS, label_to_suicidal_intent


PRIMARY_ENDPOINT: dict[str, str] = {
    "name": "suicidal_intent_sensitivity",
    "display_name": "Suicidal-intent sensitivity",
    "definition": "TP / (TP + FN), where any of Indicator, Ideation, Behavior, or Attempt is positive.",
    "rationale": "This safety-oriented endpoint measures missed suicidality signs and is invariant to deployment prevalence.",
}
DEFAULT_BOOTSTRAP_REPLICATES = 2_000
DEFAULT_BOOTSTRAP_SEED = 20_260_820
DEFAULT_TARGET_PREVALENCES = (0.01, 0.05, 0.10, 0.20, 0.50)
_CONFIDENCE_LEVEL = 0.95


@dataclass(frozen=True)
class _PreparedPredictions:
    sample_ids: np.ndarray
    cluster_codes: np.ndarray
    cluster_count: int
    y_true: np.ndarray
    y_pred: np.ndarray
    positive_actual: np.ndarray
    positive_predicted: np.ndarray
    phase_actual: np.ndarray
    phase_predicted: np.ndarray


def analyze_predictions(
    predictions: pd.DataFrame,
    *,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    target_prevalences: Iterable[float] = DEFAULT_TARGET_PREVALENCES,
) -> dict[str, Any]:
    """Return point estimates and percentile CIs from a user-cluster bootstrap."""
    prepared = _prepare_predictions(predictions)
    prevalences = _validate_prevalences(target_prevalences)
    _validate_bootstrap(bootstrap_replicates)

    point = _binary_point_estimates(prepared, np.ones(len(prepared.y_true), dtype=np.int64))
    point["phase_macro_f1"] = _phase_macro_f1(prepared, np.ones(len(prepared.y_true), dtype=np.int64))
    rng = np.random.default_rng(bootstrap_seed)
    samples = _bootstrap_metric_samples(prepared, bootstrap_replicates, rng)

    prevalence_adjusted: list[dict[str, Any]] = []
    for prevalence in prevalences:
        ppv_values = _prevalence_adjusted_ppv(samples["sensitivity"], samples["specificity"], prevalence)
        npv_values = _prevalence_adjusted_npv(samples["sensitivity"], samples["specificity"], prevalence)
        prevalence_adjusted.append(
            {
                "target_prevalence": prevalence,
                "ppv": _point_prevalence_ppv(point["sensitivity"], point["specificity"], prevalence),
                "ppv_ci_95": _percentile_ci(ppv_values),
                "npv": _point_prevalence_npv(point["sensitivity"], point["specificity"], prevalence),
                "npv_ci_95": _percentile_ci(npv_values),
            }
        )

    return {
        "schema_version": "1.0",
        "prespecification": {
            "primary_endpoint": PRIMARY_ENDPOINT,
            "secondary_endpoints": [
                "suicidal_intent_specificity",
                "observed_sample_ppv",
                "observed_sample_npv",
                "prevalence_adjusted_ppv_npv",
                "positive_phase_macro_f1",
            ],
            "analysis_plan": "Two-sided 95% percentile confidence intervals from resampling users with replacement; all rows for a selected user remain together.",
        },
        "bootstrap": {
            "unit": "users",
            "cluster_column": "users",
            "cluster_count": prepared.cluster_count,
            "replicates_requested": bootstrap_replicates,
            "seed": bootstrap_seed,
            "confidence_level": _CONFIDENCE_LEVEL,
        },
        "sample": {
            "records": len(prepared.y_true),
            "observed_positive_prevalence": float(prepared.positive_actual.mean()),
        },
        "point_estimates": point,
        "confidence_intervals_95": {
            metric: _percentile_ci(values)
            for metric, values in samples.items()
        },
        "prevalence_adjusted_predictive_values": prevalence_adjusted,
    }


def paired_primary_endpoint_comparison(
    reference_predictions: pd.DataFrame,
    candidate_predictions: pd.DataFrame,
    *,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Compare candidate minus reference sensitivity with paired user bootstrap CIs."""
    reference = _prepare_predictions(reference_predictions)
    candidate = _prepare_predictions(candidate_predictions)
    _validate_bootstrap(bootstrap_replicates)
    if not np.array_equal(reference.sample_ids, candidate.sample_ids):
        raise ValueError("Paired comparison requires exactly the same unique sample IDs in both runs")
    if not np.array_equal(reference.y_true, candidate.y_true):
        raise ValueError("Paired comparison requires identical ground-truth labels")
    if not np.array_equal(reference.cluster_codes, candidate.cluster_codes):
        raise ValueError("Paired comparison requires the same users for every sample ID")

    unit_weights = np.ones(len(reference.y_true), dtype=np.int64)
    reference_value = _binary_point_estimates(reference, unit_weights)["sensitivity"]
    candidate_value = _binary_point_estimates(candidate, unit_weights)["sensitivity"]
    rng = np.random.default_rng(bootstrap_seed)
    differences = np.empty(bootstrap_replicates, dtype=float)
    for index in range(bootstrap_replicates):
        weights = _cluster_weights(reference, rng)
        differences[index] = (
            _binary_point_estimates(candidate, weights)["sensitivity"]
            - _binary_point_estimates(reference, weights)["sensitivity"]
        )
    ci = _percentile_ci(differences)
    return {
        "schema_version": "1.0",
        "endpoint": PRIMARY_ENDPOINT,
        "direction": "candidate_minus_reference",
        "reference": {"value": reference_value},
        "candidate": {"value": candidate_value},
        "difference": candidate_value - reference_value,
        "difference_ci_95": ci,
        "bootstrap": {
            "unit": "users",
            "cluster_count": reference.cluster_count,
            "replicates_requested": bootstrap_replicates,
            "seed": bootstrap_seed,
            "confidence_level": _CONFIDENCE_LEVEL,
            "paired": True,
        },
        "interpretation": (
            "candidate higher" if ci["lower"] > 0 else "reference higher" if ci["upper"] < 0 else "inconclusive"
        ),
    }


def _prepare_predictions(predictions: pd.DataFrame) -> _PreparedPredictions:
    required = {"sample_id", "users", "ground_truth", "prediction"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"User-cluster statistical analysis requires columns: {missing}")
    frame = predictions.loc[:, ["sample_id", "users", "ground_truth", "prediction"]].copy()
    frame["sample_id"] = frame["sample_id"].astype(str)
    if frame["sample_id"].duplicated().any():
        raise ValueError("Statistical analysis requires unique sample_id values")
    if frame["users"].isna().any():
        raise ValueError("User-cluster statistical analysis requires non-null users values")
    if not set(frame["ground_truth"]).issubset(FINAL_LABELS):
        raise ValueError("Statistical analysis found invalid ground-truth labels")
    frame = frame.sort_values("sample_id", kind="stable").reset_index(drop=True)
    codes, unique_users = pd.factorize(frame["users"], sort=True)
    if len(unique_users) < 2:
        raise ValueError("User-cluster bootstrap requires at least two distinct users")
    phase_index = {label: index for index, label in enumerate(VALID_LABELS)}
    y_true = frame["ground_truth"].to_numpy(dtype=str)
    y_pred = frame["prediction"].fillna("Error").to_numpy(dtype=str)
    phase_actual = np.array([phase_index.get(label, -1) for label in y_true], dtype=int)
    phase_predicted = np.array([phase_index.get(label, -1) for label in y_pred], dtype=int)
    return _PreparedPredictions(
        sample_ids=frame["sample_id"].to_numpy(dtype=str),
        cluster_codes=codes.astype(int),
        cluster_count=len(unique_users),
        y_true=y_true,
        y_pred=y_pred,
        positive_actual=np.array([label_to_suicidal_intent(label) for label in y_true]),
        positive_predicted=np.array([label_to_suicidal_intent(label) if label in FINAL_LABELS else False for label in y_pred]),
        phase_actual=phase_actual,
        phase_predicted=phase_predicted,
    )


def _validate_bootstrap(replicates: int) -> None:
    if replicates < 100:
        raise ValueError("bootstrap_replicates must be at least 100")


def _validate_prevalences(prevalences: Iterable[float]) -> tuple[float, ...]:
    values = tuple(float(value) for value in prevalences)
    if not values or any(value <= 0 or value >= 1 for value in values):
        raise ValueError("target prevalences must be a non-empty sequence strictly between 0 and 1")
    return values


def _cluster_weights(prepared: _PreparedPredictions, rng: np.random.Generator) -> np.ndarray:
    sampled_clusters = rng.integers(0, prepared.cluster_count, size=prepared.cluster_count)
    multiplicity = np.bincount(sampled_clusters, minlength=prepared.cluster_count)
    return multiplicity[prepared.cluster_codes]


def _binary_point_estimates(prepared: _PreparedPredictions, weights: np.ndarray) -> dict[str, float]:
    actual = prepared.positive_actual
    predicted = prepared.positive_predicted
    tp = int(weights[actual & predicted].sum())
    fp = int(weights[~actual & predicted].sum())
    tn = int(weights[~actual & ~predicted].sum())
    fn = int(weights[actual & ~predicted].sum())
    return {
        "sensitivity": _ratio(tp, tp + fn),
        "specificity": _ratio(tn, tn + fp),
        "ppv": _ratio(tp, tp + fp),
        "npv": _ratio(tn, tn + fn),
    }


def _phase_macro_f1(prepared: _PreparedPredictions, weights: np.ndarray) -> float:
    included = (prepared.phase_actual >= 0) & (prepared.phase_predicted >= 0)
    if not included.any():
        return float("nan")
    encoded = prepared.phase_actual[included] * len(VALID_LABELS) + prepared.phase_predicted[included]
    matrix = np.bincount(encoded, weights=weights[included], minlength=len(VALID_LABELS) ** 2).reshape(len(VALID_LABELS), -1)
    f1_values = []
    for index in range(len(VALID_LABELS)):
        tp = matrix[index, index]
        fp = matrix[:, index].sum() - tp
        fn = matrix[index, :].sum() - tp
        precision = _ratio(tp, tp + fp)
        recall = _ratio(tp, tp + fn)
        f1_values.append(_ratio(2 * precision * recall, precision + recall))
    return float(np.mean(f1_values))


def _bootstrap_metric_samples(
    prepared: _PreparedPredictions,
    replicates: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    values = {name: np.empty(replicates, dtype=float) for name in ("sensitivity", "specificity", "ppv", "npv", "phase_macro_f1")}
    for index in range(replicates):
        weights = _cluster_weights(prepared, rng)
        binary = _binary_point_estimates(prepared, weights)
        for name, value in binary.items():
            values[name][index] = value
        values["phase_macro_f1"][index] = _phase_macro_f1(prepared, weights)
    return values


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def _percentile_ci(values: np.ndarray) -> dict[str, float | int]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return {"lower": float("nan"), "upper": float("nan"), "valid_replicates": 0}
    lower, upper = np.quantile(finite, [0.025, 0.975])
    return {"lower": float(lower), "upper": float(upper), "valid_replicates": int(len(finite))}


def _prevalence_adjusted_ppv(sensitivity: np.ndarray, specificity: np.ndarray, prevalence: float) -> np.ndarray:
    return np.divide(sensitivity * prevalence, sensitivity * prevalence + (1 - specificity) * (1 - prevalence), out=np.full_like(sensitivity, np.nan), where=(sensitivity * prevalence + (1 - specificity) * (1 - prevalence)) > 0)


def _prevalence_adjusted_npv(sensitivity: np.ndarray, specificity: np.ndarray, prevalence: float) -> np.ndarray:
    return np.divide(specificity * (1 - prevalence), (1 - sensitivity) * prevalence + specificity * (1 - prevalence), out=np.full_like(sensitivity, np.nan), where=((1 - sensitivity) * prevalence + specificity * (1 - prevalence)) > 0)


def _point_prevalence_ppv(sensitivity: float, specificity: float, prevalence: float) -> float:
    return _ratio(sensitivity * prevalence, sensitivity * prevalence + (1 - specificity) * (1 - prevalence))


def _point_prevalence_npv(sensitivity: float, specificity: float, prevalence: float) -> float:
    return _ratio(specificity * (1 - prevalence), (1 - sensitivity) * prevalence + specificity * (1 - prevalence))

"""Reproducible comparison of the repository's completed Luna workflow runs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

import pandas as pd

from domain.labels import VALID_LABELS
from .statistics import analyze_predictions, paired_primary_endpoint_comparison


POSITIVE_LABELS = frozenset(VALID_LABELS)
_CASE_HEADER = re.compile(r"^Sample ID\s*:\s*(?P<sample_id>.+?)\s+\[(?:PASS|FAIL)\]\s*$")
_CASE_GT = re.compile(r"^GT\s*:\s*(?P<label>\S+)\s*$")
_CASE_PRED = re.compile(r"^Pred\s*:\s*(?P<label>\S+)\s*$")


@dataclass(frozen=True)
class ComparisonSystem:
    name: str
    predictions: pd.DataFrame
    phase_bootstrap_available: bool = True
    phase_point_override: float | None = None
    source_note: str = "Exact per-record final-label predictions."


def parse_single_agent_all_cases(path: str | Path) -> pd.DataFrame:
    """Parse exact sample ID, ground truth, and prediction from a single-agent run log."""
    records: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        header = _CASE_HEADER.match(line)
        if header:
            if current is not None:
                _finish_single_agent_record(current, records, line_number)
            current = {"sample_id": header.group("sample_id").strip()}
            continue
        if current is None:
            continue
        ground_truth = _CASE_GT.match(line)
        if ground_truth:
            current["ground_truth"] = ground_truth.group("label")
            continue
        prediction = _CASE_PRED.match(line)
        if prediction:
            current["prediction"] = prediction.group("label")
    if current is not None:
        _finish_single_agent_record(current, records, line_number + 1)
    frame = pd.DataFrame(records)
    if frame.empty:
        raise ValueError(f"No single-agent cases found in {path}")
    if frame["sample_id"].duplicated().any():
        raise ValueError(f"Duplicate sample IDs found in single-agent cases: {path}")
    return frame


def reconstruct_binary_predictions(
    dataset: pd.DataFrame,
    misclassified_ids: set[str],
) -> pd.DataFrame:
    """Reconstruct exact binary decisions when only binary error IDs survived archival."""
    frame = (
        dataset.loc[:, ["sample_id", "users", "ground_truth"]].copy()
        if {"sample_id", "users", "ground_truth"}.issubset(dataset.columns)
        else _canonical_dataset(dataset)
    )
    frame["sample_id"] = frame["sample_id"].astype(str)
    unknown = sorted(misclassified_ids - set(frame["sample_id"]))
    if unknown:
        raise ValueError(f"Binary misclassification file contains unknown IDs: {unknown[:10]}")
    actual_positive = frame["ground_truth"].isin(POSITIVE_LABELS)
    is_error = frame["sample_id"].isin(misclassified_ids)
    predicted_positive = actual_positive ^ is_error
    frame["prediction"] = [
        actual if predicted else "Unsupportive"
        for actual, predicted in zip(frame["ground_truth"], predicted_positive)
    ]
    false_positive = ~actual_positive & predicted_positive
    frame.loc[false_positive, "prediction"] = "Indicator"
    return frame


def build_repository_comparison(
    repository_root: str | Path,
    *,
    bootstrap_replicates: int = 2_000,
    bootstrap_seed: int = 20_260_820,
) -> dict[str, Any]:
    """Build the analysis payload from the completed runs committed in this repository."""
    root = Path(repository_root)
    module_root = root / "multi_agent_frameworks"
    result_root = root / "evaluation_results"
    dataset = pd.read_csv(
        module_root / "dataset" / "suicidality_full_dataset_posneg_balanced.csv",
        encoding="utf-8-sig",
    )
    canonical_dataset = _canonical_dataset(dataset)

    f1_dir = result_root / "20260804_190528_framework1_final_run_full_balanced_gpt_5.6_luna"
    f2_dir = result_root / "20260804_232101_framework2_final_run_full_balanced_gpt_5.6_luna"
    f3_dir = result_root / "20260819_191341_framework3_luna_full_final_run"
    f4_dir = result_root / "20260802_142022_framework4_luna_full_final_run"
    single_cases = (
        result_root
        / "Luca Final Results (Full run)"
        / "run_suite_20260803_140810_single-class_LUNA"
        / "run_20260803_140821_gpt-5.6-luna_single-class"
        / "all_cases.txt"
    )

    systems = [
        ComparisonSystem(
            "Single agent",
            _merge_with_dataset(parse_single_agent_all_cases(single_cases), canonical_dataset, "Single agent"),
        ),
        ComparisonSystem(
            "Framework 1",
            _validate_canonical_predictions(
                pd.DataFrame(_read_json(f1_dir / "results.json")["results"]),
                canonical_dataset,
                "Framework 1",
            ),
        ),
        ComparisonSystem(
            "Framework 2",
            _validate_canonical_predictions(
                pd.DataFrame(_read_json(f2_dir / "results.json")["results"]),
                canonical_dataset,
                "Framework 2",
            ),
        ),
        _framework_3_system(f3_dir, canonical_dataset),
        ComparisonSystem(
            "Framework 4",
            _validate_canonical_predictions(
                pd.DataFrame(_read_json(f4_dir / "results.json")["results"]),
                canonical_dataset,
                "Framework 4",
            ),
        ),
    ]

    analyses: dict[str, dict[str, Any]] = {}
    for system in systems:
        analysis = analyze_predictions(
            system.predictions,
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=bootstrap_seed,
        )
        if not system.phase_bootstrap_available:
            analysis["point_estimates"]["phase_macro_f1"] = system.phase_point_override
            analysis["confidence_intervals_95"]["phase_macro_f1"] = None
        analyses[system.name] = analysis

    comparisons: list[dict[str, Any]] = []
    for reference_index, reference in enumerate(systems):
        for candidate in systems[reference_index + 1 :]:
            comparison = paired_primary_endpoint_comparison(
                reference.predictions,
                candidate.predictions,
                bootstrap_replicates=bootstrap_replicates,
                bootstrap_seed=bootstrap_seed,
            )
            comparison["reference_name"] = reference.name
            comparison["candidate_name"] = candidate.name
            comparisons.append(comparison)

    return {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_status": "retrospective; endpoint designated before this final comparative analysis, not prospectively preregistered",
        "primary_endpoint": {
            "name": "suicidal_intent_sensitivity",
            "definition": "TP / (TP + FN)",
            "positive": list(VALID_LABELS),
            "negative": ["Unsupportive"],
        },
        "bootstrap": {
            "unit": "users",
            "paired": True,
            "replicates": bootstrap_replicates,
            "seed": bootstrap_seed,
        },
        "systems": {
            system.name: {
                "analysis": analyses[system.name],
                "source_note": system.source_note,
                "phase_bootstrap_available": system.phase_bootstrap_available,
            }
            for system in systems
        },
        "paired_comparisons": comparisons,
        "main_report_omissions": [
            "Post-level stratified bootstrap",
            "Per-post McNemar significance tests",
            "Five-label aggregate accuracy and macro-F1 as primary metrics",
        ],
    }


def write_repository_comparison(
    repository_root: str | Path,
    output_dir: str | Path,
    *,
    bootstrap_replicates: int = 2_000,
    bootstrap_seed: int = 20_260_820,
) -> dict[str, str]:
    payload = build_repository_comparison(
        repository_root,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
    )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / "workflow_comparison.json"
    markdown_path = destination / "workflow_comparison.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(_comparison_markdown(payload), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(markdown_path)}


def _framework_3_system(run_dir: Path, dataset: pd.DataFrame) -> ComparisonSystem:
    predictions_path = run_dir / "predictions.csv"
    if predictions_path.is_file():
        return ComparisonSystem(
            "Framework 3",
            _validate_canonical_predictions(
                pd.read_csv(predictions_path, encoding="utf-8-sig"),
                dataset,
                "Framework 3",
            ),
        )
    error_ids = {
        line.strip()
        for line in (run_dir / "suicidal_intent_misclassified_ids.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    metrics = _read_json(run_dir / "metrics.json")
    phase_point = float(metrics["positive_subclassification"]["summary"]["f1_score"])
    return ComparisonSystem(
        "Framework 3",
        reconstruct_binary_predictions(dataset, error_ids),
        phase_bootstrap_available=False,
        phase_point_override=phase_point,
        source_note=(
            "Exact binary decisions reconstructed from the archived binary misclassification IDs; "
            "per-record phase predictions were not retained."
        ),
    )


def _canonical_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    required = {"ID", "users", "sentiment"}
    missing = sorted(required - set(dataset.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    frame = dataset.loc[:, ["ID", "users", "sentiment"]].rename(
        columns={"ID": "sample_id", "sentiment": "ground_truth"}
    )
    frame["sample_id"] = frame["sample_id"].astype(str)
    if frame["sample_id"].duplicated().any():
        raise ValueError("Dataset sample IDs must be unique")
    return frame


def _merge_with_dataset(predictions: pd.DataFrame, dataset: pd.DataFrame, name: str) -> pd.DataFrame:
    predictions = predictions.copy()
    predictions["sample_id"] = predictions["sample_id"].astype(str)
    merged = dataset.merge(
        predictions,
        on="sample_id",
        how="left",
        validate="one_to_one",
        suffixes=("_dataset", "_prediction"),
    )
    if merged["prediction"].isna().any() or len(merged) != len(dataset):
        raise ValueError(f"{name} predictions do not cover every dataset ID exactly once")
    if not (merged["ground_truth_dataset"] == merged["ground_truth_prediction"]).all():
        raise ValueError(f"{name} ground truth does not match the canonical dataset")
    return merged.rename(columns={"ground_truth_dataset": "ground_truth"}).loc[
        :, ["sample_id", "users", "ground_truth", "prediction"]
    ]


def _validate_canonical_predictions(
    predictions: pd.DataFrame,
    dataset: pd.DataFrame,
    name: str,
) -> pd.DataFrame:
    required = {"sample_id", "users", "ground_truth", "prediction"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"{name} predictions are missing columns: {missing}")
    frame = predictions.loc[:, list(required)].copy()
    frame["sample_id"] = frame["sample_id"].astype(str)
    if frame["sample_id"].duplicated().any():
        raise ValueError(f"{name} contains duplicate sample IDs")
    expected = dataset.sort_values("sample_id").reset_index(drop=True)
    actual = frame.sort_values("sample_id").reset_index(drop=True)
    if not expected["sample_id"].equals(actual["sample_id"]):
        raise ValueError(f"{name} sample IDs do not match the canonical dataset")
    if not expected["ground_truth"].equals(actual["ground_truth"]):
        raise ValueError(f"{name} ground truth does not match the canonical dataset")
    if not expected["users"].astype(str).equals(actual["users"].astype(str)):
        raise ValueError(f"{name} user assignments do not match the canonical dataset")
    return actual.loc[:, ["sample_id", "users", "ground_truth", "prediction"]]


def _finish_single_agent_record(
    current: dict[str, str],
    records: list[dict[str, str]],
    line_number: int,
) -> None:
    missing = {"sample_id", "ground_truth", "prediction"} - set(current)
    if missing:
        raise ValueError(f"Incomplete single-agent record before line {line_number}: missing {sorted(missing)}")
    records.append(current.copy())


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _comparison_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Workflow Statistical Comparison",
        "",
        "## Analysis Contract",
        "",
        "- Primary endpoint: **suicidal-intent sensitivity**, `TP / (TP + FN)`.",
        "- Positive: Indicator, Ideation, Behavior, or Attempt. Negative: Unsupportive.",
        f"- Uncertainty: paired user-cluster percentile bootstrap with {payload['bootstrap']['replicates']} resamples "
        f"(seed {payload['bootstrap']['seed']}).",
        "- Status: retrospective. The endpoint was designated before this final comparative analysis, but the model outputs already existed; this is not a prospective preregistration.",
        "- Five-label aggregate accuracy and macro-F1 are not primary endpoints. The conditional four-positive-label macro-F1 is secondary.",
        "- Per-post McNemar tests are omitted because posts from the same user are not independent.",
        "",
        "A 95% CI for a paired difference is built from the central 95% of the user-bootstrap difference estimates. "
        "If the complete interval is above zero, the candidate has higher sensitivity; if it crosses zero, the comparison is inconclusive.",
        "",
        "## Primary Endpoint",
        "",
        "| System | Sensitivity (95% CI) | False negatives | Specificity (95% CI) |",
        "|---|---:|---:|---:|",
    ]
    for name, system in payload["systems"].items():
        analysis = system["analysis"]
        point = analysis["point_estimates"]
        cis = analysis["confidence_intervals_95"]
        records = analysis["sample"]["records"]
        positive_count = round(records * analysis["sample"]["observed_positive_prevalence"])
        false_negatives = round(positive_count * (1 - point["sensitivity"]))
        lines.append(
            f"| {name} | {_percent(point['sensitivity'])} ({_percent(cis['sensitivity']['lower'])} to "
            f"{_percent(cis['sensitivity']['upper'])}) | {false_negatives} | "
            f"{_percent(point['specificity'])} ({_percent(cis['specificity']['lower'])} to "
            f"{_percent(cis['specificity']['upper'])}) |"
        )

    lines.extend(
        [
            "",
            "## Paired Sensitivity Comparisons",
            "",
            "Every difference is candidate minus reference and is calculated inside each shared user resample; it is not obtained by subtracting two separate CIs.",
            "",
            "| Reference | Candidate | Difference (percentage points) | 95% CI (percentage points) | Interpretation |",
            "|---|---|---:|---:|---|",
        ]
    )
    for comparison in payload["paired_comparisons"]:
        ci = comparison["difference_ci_95"]
        lines.append(
            f"| {comparison['reference_name']} | {comparison['candidate_name']} | "
            f"{_signed_pp(comparison['difference'])} | {_signed_pp(ci['lower'])} to {_signed_pp(ci['upper'])} | "
            f"{comparison['interpretation']} |"
        )

    lines.extend(
        [
            "",
            "## Secondary Endpoints",
            "",
            "PPV and NPV below describe the observed 50/50 benchmark and should not be interpreted as deployment predictive values when prevalence differs.",
            "Positive-phase macro-F1 is conditional: it includes only records for which both the actual and predicted labels are one of the four positive phases.",
            "",
            "| System | PPV (95% CI) | NPV (95% CI) | Conditional positive-phase macro-F1 (95% CI) |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, system in payload["systems"].items():
        analysis = system["analysis"]
        point = analysis["point_estimates"]
        cis = analysis["confidence_intervals_95"]
        phase_ci = cis["phase_macro_f1"]
        phase = (
            f"{_percent(point['phase_macro_f1'])} ({_percent(phase_ci['lower'])} to {_percent(phase_ci['upper'])})"
            if phase_ci is not None
            else f"{_percent(point['phase_macro_f1'])} (clustered CI unavailable)"
        )
        lines.append(
            f"| {name} | {_percent(point['ppv'])} ({_percent(cis['ppv']['lower'])} to {_percent(cis['ppv']['upper'])}) | "
            f"{_percent(point['npv'])} ({_percent(cis['npv']['lower'])} to {_percent(cis['npv']['upper'])}) | {phase} |"
        )

    lines.extend(["", "## Data Provenance And Limitations", ""])
    for name, system in payload["systems"].items():
        lines.append(f"- **{name}:** {system['source_note']}")
    if not payload["systems"]["Framework 3"]["phase_bootstrap_available"]:
        lines.append(
            "- Framework 3's binary analysis is exact because its archived binary error-ID list identifies every false positive and false negative. Its per-user phase predictions cannot be recovered from aggregate metrics."
        )
    lines.extend(
        [
            "- Previously quoted counts of 426 and 390 false negatives use a different positive-class definition or different result artifact. They are incompatible with the current contract, under which Indicator is positive.",
            "- Confidence intervals quantify sampling uncertainty within this benchmark; they do not establish performance on a different population, platform, or time period.",
            "",
        ]
    )
    return "\n".join(lines)


def _percent(value: float) -> str:
    return f"{100 * value:.2f}%"


def _signed_pp(value: float) -> str:
    return f"{100 * value:+.2f}"

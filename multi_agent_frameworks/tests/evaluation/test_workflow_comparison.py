from pathlib import Path

import pandas as pd
import pytest

from evaluation.workflow_comparison import (
    parse_evalring_all_cases,
    reconstruct_binary_predictions,
)


def test_parse_evalring_all_cases_extracts_contract_fields(tmp_path: Path):
    source = tmp_path / "all_cases.txt"
    source.write_text(
        """Header
Sample ID : abc [PASS]
GT        : Ideation
Pred      : Ideation
Text:
example
---
Sample ID : def [FAIL]
GT        : Unsupportive
Pred      : Indicator
""",
        encoding="utf-8",
    )

    result = parse_evalring_all_cases(source)

    assert result.to_dict("records") == [
        {"sample_id": "abc", "ground_truth": "Ideation", "prediction": "Ideation"},
        {"sample_id": "def", "ground_truth": "Unsupportive", "prediction": "Indicator"},
    ]


def test_parse_evalring_all_cases_rejects_incomplete_record(tmp_path: Path):
    source = tmp_path / "all_cases.txt"
    source.write_text("Sample ID : abc [PASS]\nGT : Ideation\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Incomplete EvalRing record"):
        parse_evalring_all_cases(source)


def test_reconstruct_binary_predictions_toggles_only_error_ids():
    dataset = pd.DataFrame(
        [
            {"ID": "1", "users": "a", "sentiment": "Ideation"},
            {"ID": "2", "users": "b", "sentiment": "Unsupportive"},
            {"ID": "3", "users": "c", "sentiment": "Attempt"},
        ]
    )

    result = reconstruct_binary_predictions(dataset, {"1", "2"})

    assert result["prediction"].tolist() == ["Unsupportive", "Indicator", "Attempt"]


def test_reconstruct_binary_predictions_rejects_unknown_ids():
    dataset = pd.DataFrame([{"ID": "1", "users": "a", "sentiment": "Ideation"}])

    with pytest.raises(ValueError, match="unknown IDs"):
        reconstruct_binary_predictions(dataset, {"missing"})

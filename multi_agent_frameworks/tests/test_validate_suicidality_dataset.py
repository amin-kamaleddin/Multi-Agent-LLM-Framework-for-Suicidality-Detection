import csv
from pathlib import Path

import pytest

from scripts.validate_suicidality_dataset import (
    DatasetFormatError,
    validate_dataset,
)


def _write_dataset(path: Path, rows: list[dict[str, str]]) -> Path:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["ID", "users", "text", "sentiment", "time"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_valid_dataset_accepts_case_insensitive_labels_and_exact_balance(
    tmp_path: Path,
) -> None:
    path = _write_dataset(
        tmp_path / "valid.csv",
        [
            {"ID": "1", "users": "u1", "text": "a", "sentiment": "indicator", "time": "1"},
            {"ID": "2", "users": "u2", "text": "b", "sentiment": "IDEATION", "time": "2"},
            {"ID": "3", "users": "u3", "text": "c", "sentiment": "Unsupportive", "time": "3"},
            {"ID": "4", "users": "u4", "text": "d", "sentiment": "unsupportive", "time": "4"},
        ],
    )

    report = validate_dataset(path)

    assert report["passed"] is True
    assert report["summary"]["row_violation_count"] == 0
    assert report["checks"]["sentiment_labels"]["counts"] == {
        "Indicator": 1,
        "Ideation": 1,
        "Behavior": 0,
        "Attempt": 0,
        "Unsupportive": 2,
    }
    assert report["checks"]["positive_negative_balance"]["passed"] is True


def test_reports_every_row_in_duplicate_groups_and_invalid_or_missing_values(
    tmp_path: Path,
) -> None:
    path = _write_dataset(
        tmp_path / "invalid.csv",
        [
            {"ID": "1", "users": "u1", "text": "same", "sentiment": "Indicator", "time": "1"},
            {"ID": "1", "users": "u2", "text": " same ", "sentiment": "unknown", "time": "2"},
            {"ID": "3", "users": "u3", "text": "", "sentiment": "Unsupportive", "time": "3"},
            {"ID": "", "users": "u4", "text": "different", "sentiment": "", "time": "4"},
        ],
    )

    report = validate_dataset(path)
    violations = {
        row["data_row_number"]: set(row["rules"])
        for row in report["row_violations"]
    }

    assert report["passed"] is False
    assert violations == {
        1: {"duplicate_id", "duplicate_text"},
        2: {"duplicate_id", "duplicate_text", "invalid_sentiment"},
        3: {"missing_text"},
        4: {"missing_id", "missing_sentiment"},
    }
    assert report["checks"]["unique_ids"]["duplicate_group_count"] == 1
    assert report["checks"]["unique_texts"]["duplicate_group_count"] == 1
    assert report["dataset_violations"][0]["rule"] == (
        "positive_negative_balance_not_exactly_50_50"
    )


def test_reports_global_imbalance_without_arbitrarily_blaming_rows(
    tmp_path: Path,
) -> None:
    path = _write_dataset(
        tmp_path / "imbalanced.csv",
        [
            {"ID": "1", "users": "u1", "text": "a", "sentiment": "Indicator", "time": "1"},
            {"ID": "2", "users": "u2", "text": "b", "sentiment": "Ideation", "time": "2"},
            {"ID": "3", "users": "u3", "text": "c", "sentiment": "Unsupportive", "time": "3"},
        ],
    )

    report = validate_dataset(path)

    assert report["row_violations"] == []
    assert report["checks"]["positive_negative_balance"] == {
        "passed": False,
        "positive_labels": ["Indicator", "Ideation", "Behavior", "Attempt"],
        "negative_labels": ["Unsupportive"],
        "positive_count": 2,
        "negative_count": 1,
        "unclassified_count": 0,
        "positive_proportion": 2 / 3,
        "negative_proportion": 1 / 3,
        "required_proportion_each": 0.5,
    }
    assert "no individual row" in report["dataset_violations"][0]["row_attribution"]


@pytest.mark.parametrize(
    ("filename", "content", "message"),
    [
        ("dataset.txt", "ID,text,sentiment\n1,a,Indicator\n", r"Expected a \.csv"),
        ("dataset.csv", "", "no CSV header"),
        ("dataset.csv", "ID,text\n1,a\n", "Missing required columns"),
        ("dataset.csv", "ID,text,sentiment\n", "no data rows"),
    ],
)
def test_rejects_invalid_dataset_format(
    tmp_path: Path,
    filename: str,
    content: str,
    message: str,
) -> None:
    path = tmp_path / filename
    path.write_text(content, encoding="utf-8")

    with pytest.raises(DatasetFormatError, match=message):
        validate_dataset(path)


def test_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(DatasetFormatError, match="does not exist"):
        validate_dataset(tmp_path / "missing.csv")

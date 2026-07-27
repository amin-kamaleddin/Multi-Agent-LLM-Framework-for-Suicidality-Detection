"""Validate completeness, labels, balance, and uniqueness of a CSV dataset."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import sys
from typing import Any, Sequence


ALLOWED_LABELS = (
    "Indicator",
    "Ideation",
    "Behavior",
    "Attempt",
    "Unsupportive",
)
POSITIVE_LABELS = frozenset(ALLOWED_LABELS[:-1])
NEGATIVE_LABEL = "Unsupportive"
REQUIRED_COLUMNS = ("ID", "text", "sentiment")
DEFAULT_DATASET_PATH = Path(
    "dataset/suicidality_full_dataset_posneg_balanced.csv"
)
_CANONICAL_LABELS = {label.casefold(): label for label in ALLOWED_LABELS}

try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(2**31 - 1)


class DatasetFormatError(ValueError):
    """Raised when the input cannot be interpreted as the required CSV format."""


def normalize_text(text: str) -> str:
    """Match the repository's existing text-deduplication normalization."""

    return text.replace("\r\n", "\n").strip()


def _load_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if path.suffix.lower() != ".csv":
        raise DatasetFormatError(f"Expected a .csv file: {path}")
    if not path.exists():
        raise DatasetFormatError(f"Dataset does not exist: {path}")
    if not path.is_file():
        raise DatasetFormatError(f"Dataset path is not a file: {path}")

    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise DatasetFormatError(f"Dataset has no CSV header: {path}")
            columns = [str(column).strip() for column in reader.fieldnames]
            if any(not column for column in columns):
                raise DatasetFormatError("CSV header contains an empty column name")
            duplicate_columns = sorted(
                column for column, count in Counter(columns).items() if count > 1
            )
            if duplicate_columns:
                raise DatasetFormatError(
                    f"CSV header contains duplicate columns: {duplicate_columns}"
                )
            missing_columns = sorted(set(REQUIRED_COLUMNS) - set(columns))
            if missing_columns:
                raise DatasetFormatError(
                    f"Missing required columns: {missing_columns}; found {columns}"
                )

            rows: list[dict[str, str]] = []
            for data_row_number, raw_row in enumerate(reader, start=1):
                if None in raw_row:
                    raise DatasetFormatError(
                        f"Data row {data_row_number} has more fields than the header"
                    )
                if any(value is None for value in raw_row.values()):
                    raise DatasetFormatError(
                        f"Data row {data_row_number} has fewer fields than the header"
                    )
                rows.append(
                    {
                        column: str(raw_row.get(original_column, ""))
                        for original_column, column in zip(reader.fieldnames, columns)
                    }
                )
    except UnicodeDecodeError as exc:
        raise DatasetFormatError(f"Dataset is not valid UTF-8: {path}") from exc
    except csv.Error as exc:
        raise DatasetFormatError(f"Invalid CSV content in {path}: {exc}") from exc
    except OSError as exc:
        raise DatasetFormatError(f"Could not read {path}: {exc}") from exc

    if not rows:
        raise DatasetFormatError(f"Dataset contains no data rows: {path}")
    return columns, rows


def validate_dataset(path: str | Path) -> dict[str, Any]:
    """Return a JSON-compatible validation report for one CSV dataset."""

    dataset_path = Path(path).expanduser().resolve()
    columns, rows = _load_rows(dataset_path)
    row_violations: dict[int, dict[str, Any]] = {}
    ids_to_rows: dict[str, list[int]] = defaultdict(list)
    texts_to_rows: dict[str, list[int]] = defaultdict(list)
    canonical_labels: list[str | None] = []

    def add_violation(
        data_row_number: int,
        rule: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        record = rows[data_row_number - 1]
        violation = row_violations.setdefault(
            data_row_number,
            {
                "data_row_number": data_row_number,
                "row_id": record["ID"].strip() or None,
                "rules": [],
                "details": {},
            },
        )
        violation["rules"].append(rule)
        if details:
            violation["details"][rule] = details

    for data_row_number, row in enumerate(rows, start=1):
        row_id = row["ID"].strip()
        text = normalize_text(row["text"])
        sentiment = row["sentiment"].strip()

        if row_id:
            ids_to_rows[row_id].append(data_row_number)
        else:
            add_violation(data_row_number, "missing_id")

        if text:
            texts_to_rows[text].append(data_row_number)
        else:
            add_violation(data_row_number, "missing_text")

        if not sentiment:
            canonical_labels.append(None)
            add_violation(data_row_number, "missing_sentiment")
        else:
            canonical_label = _CANONICAL_LABELS.get(sentiment.casefold())
            canonical_labels.append(canonical_label)
            if canonical_label is None:
                add_violation(
                    data_row_number,
                    "invalid_sentiment",
                    {
                        "actual": sentiment,
                        "allowed_case_insensitive": list(ALLOWED_LABELS),
                    },
                )

    duplicate_id_groups = {
        row_id: data_row_numbers
        for row_id, data_row_numbers in ids_to_rows.items()
        if len(data_row_numbers) > 1
    }
    for row_id, data_row_numbers in duplicate_id_groups.items():
        for data_row_number in data_row_numbers:
            add_violation(
                data_row_number,
                "duplicate_id",
                {"value": row_id, "matching_data_rows": data_row_numbers},
            )

    duplicate_text_groups = {
        text: data_row_numbers
        for text, data_row_numbers in texts_to_rows.items()
        if len(data_row_numbers) > 1
    }
    for data_row_numbers in duplicate_text_groups.values():
        matching_row_ids = [
            rows[data_row_number - 1]["ID"].strip() or None
            for data_row_number in data_row_numbers
        ]
        for data_row_number in data_row_numbers:
            add_violation(
                data_row_number,
                "duplicate_text",
                {
                    "matching_data_rows": data_row_numbers,
                    "matching_row_ids": matching_row_ids,
                },
            )

    label_counts = Counter(
        label for label in canonical_labels if label is not None
    )
    positive_count = sum(label_counts[label] for label in POSITIVE_LABELS)
    negative_count = label_counts[NEGATIVE_LABEL]
    unclassified_count = len(rows) - positive_count - negative_count
    balance_passed = (
        unclassified_count == 0
        and positive_count == negative_count
        and positive_count + negative_count == len(rows)
    )
    dataset_violations: list[dict[str, Any]] = []
    if not balance_passed:
        dataset_violations.append(
            {
                "rule": "positive_negative_balance_not_exactly_50_50",
                "positive_count": positive_count,
                "negative_count": negative_count,
                "unclassified_count": unclassified_count,
                "difference": positive_count - negative_count,
                "row_attribution": (
                    "This is a dataset-level constraint; no individual row can be "
                    "objectively identified as the excess row."
                ),
            }
        )

    missing_id_count = sum(
        "missing_id" in violation["rules"]
        for violation in row_violations.values()
    )
    missing_text_count = sum(
        "missing_text" in violation["rules"]
        for violation in row_violations.values()
    )
    missing_sentiment_count = sum(
        "missing_sentiment" in violation["rules"]
        for violation in row_violations.values()
    )
    invalid_sentiment_count = sum(
        "invalid_sentiment" in violation["rules"]
        for violation in row_violations.values()
    )
    ordered_row_violations = [
        row_violations[data_row_number]
        for data_row_number in sorted(row_violations)
    ]

    report = {
        "schema_version": "1.0",
        "dataset_path": str(dataset_path),
        "passed": not ordered_row_violations and not dataset_violations,
        "summary": {
            "total_rows": len(rows),
            "row_violation_count": len(ordered_row_violations),
            "dataset_violation_count": len(dataset_violations),
        },
        "checks": {
            "required_columns": {
                "passed": True,
                "required": list(REQUIRED_COLUMNS),
                "actual": columns,
            },
            "completeness": {
                "passed": not (
                    missing_id_count
                    or missing_text_count
                    or missing_sentiment_count
                ),
                "missing_id_rows": missing_id_count,
                "missing_text_rows": missing_text_count,
                "missing_sentiment_rows": missing_sentiment_count,
            },
            "unique_ids": {
                "passed": not duplicate_id_groups and missing_id_count == 0,
                "duplicate_group_count": len(duplicate_id_groups),
                "duplicate_row_count": sum(
                    len(data_row_numbers)
                    for data_row_numbers in duplicate_id_groups.values()
                ),
            },
            "unique_texts": {
                "passed": not duplicate_text_groups and missing_text_count == 0,
                "normalization": "CRLF to LF, then trim leading/trailing whitespace",
                "duplicate_group_count": len(duplicate_text_groups),
                "duplicate_row_count": sum(
                    len(data_row_numbers)
                    for data_row_numbers in duplicate_text_groups.values()
                ),
            },
            "sentiment_labels": {
                "passed": invalid_sentiment_count == 0
                and missing_sentiment_count == 0,
                "case_insensitive": True,
                "allowed": list(ALLOWED_LABELS),
                "counts": {
                    label: label_counts[label] for label in ALLOWED_LABELS
                },
                "invalid_rows": invalid_sentiment_count,
            },
            "positive_negative_balance": {
                "passed": balance_passed,
                "positive_labels": list(ALLOWED_LABELS[:-1]),
                "negative_labels": [NEGATIVE_LABEL],
                "positive_count": positive_count,
                "negative_count": negative_count,
                "unclassified_count": unclassified_count,
                "positive_proportion": positive_count / len(rows),
                "negative_proportion": negative_count / len(rows),
                "required_proportion_each": 0.5,
            },
        },
        "row_violations": ordered_row_violations,
        "dataset_violations": dataset_violations,
    }
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate required values, unique IDs/text, five case-insensitive "
            "sentiment labels, and exact 50/50 positive-negative balance."
        )
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        default=str(DEFAULT_DATASET_PATH),
        help=(
            "CSV path; defaults to "
            "dataset/suicidality_full_dataset_posneg_balanced.csv"
        ),
    )
    parser.add_argument(
        "--output",
        help="Optional path for the complete JSON report; parent directories are created.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = validate_dataset(args.dataset)
    except DatasetFormatError as exc:
        parser.error(str(exc))

    serialized = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(serialized, encoding="utf-8")
        print(
            json.dumps(
                {
                    "passed": report["passed"],
                    **report["summary"],
                    "report_path": str(output_path),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(serialized, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

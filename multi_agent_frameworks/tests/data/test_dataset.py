import json
from pathlib import Path

import pandas as pd

from data.dataset import (
    audit_dataset,
    duplicate_normalized_text_count,
    normalize_text,
    prepare_dataset_files,
    sample_balanced,
)


DATASET_PATH = Path(__file__).resolve().parents[2] / "dataset" / "anonymizing_suicidal_dataset.csv"
COHORT_DIR = DATASET_PATH.parent / "positive-negative-final-datasets"
PROCESSED_DIR = DATASET_PATH.parent / "processed"
CANONICAL_COLUMNS = ["ID", "users", "text", "sentiment", "time"]


def test_dataset_audit_reproduces_expected_counts():
    _, dataset1, audit = audit_dataset(DATASET_PATH)

    assert audit["raw_count"] == 14613
    assert audit["raw_label_counts"] == {
        "Indicator": 4615,
        "Ideation": 7133,
        "Behavior": 2056,
        "Attempt": 809,
    }
    assert audit["missing_text"] == 0
    assert audit["duplicate_normalized_texts"] == 217
    assert audit["dataset1_count"] == 9800
    assert len(dataset1) == 9800
    assert "users" in audit["normalized_columns"]


def test_balanced_sample_has_equal_label_counts():
    _, dataset1, _ = audit_dataset(DATASET_PATH)
    sampled = sample_balanced(dataset1, sample_size=500, seed=42)

    assert len(sampled) == 500
    assert sampled["sentiment"].value_counts().to_dict() == {
        "Indicator": 125,
        "Ideation": 125,
        "Behavior": 125,
        "Attempt": 125,
    }
    assert sampled["ID"].is_unique
    assert list(sampled.columns) == CANONICAL_COLUMNS
    assert duplicate_normalized_text_count(sampled) == 0


def test_seeded_cohorts_preserve_canonical_rsd15k_fields_and_balances():
    positive = pd.read_csv(COHORT_DIR / "dataset1_positive_balanced_100_seed42.csv", encoding="utf-8-sig")
    negative = pd.read_csv(
        COHORT_DIR / "negative_real_synthetic_balanced_100_seed42.csv",
        encoding="utf-8-sig",
    )

    assert list(positive.columns) == [*CANONICAL_COLUMNS, "real"]
    assert list(negative.columns) == [*CANONICAL_COLUMNS, "real"]
    assert positive["sentiment"].value_counts().to_dict() == {
        "Indicator": 25,
        "Ideation": 25,
        "Behavior": 25,
        "Attempt": 25,
    }
    assert positive["real"].value_counts().to_dict() == {True: 100}
    assert negative["sentiment"].value_counts().to_dict() == {"Unsupportive": 100}
    assert negative["real"].value_counts().to_dict() == {True: 50, False: 50}
    assert positive["ID"].is_unique
    assert negative["ID"].is_unique
    assert set(positive["ID"]).isdisjoint(set(negative["ID"]))
    assert positive[["users", "time"]].notna().all().all()
    assert negative[["users", "time"]].notna().all().all()

    raw = pd.read_csv(DATASET_PATH, encoding="utf-8-sig")
    expected_positive_metadata = raw.iloc[positive["ID"].tolist()][
        ["users", "text", "sentiment", "time"]
    ].reset_index(drop=True)
    actual_positive_metadata = positive[
        ["users", "text", "sentiment", "time"]
    ].reset_index(drop=True)
    actual_positive_metadata["text"] = actual_positive_metadata["text"].map(
        normalize_text
    )
    expected_positive_metadata["text"] = expected_positive_metadata["text"].map(
        normalize_text
    )
    pd.testing.assert_frame_equal(
        actual_positive_metadata,
        expected_positive_metadata,
        check_dtype=False,
    )


def test_committed_processed_fixtures_use_canonical_rsd15k_fields():
    cleaned = pd.read_csv(PROCESSED_DIR / "dataset1_cleaned.csv", encoding="utf-8-sig")
    sampled = pd.read_csv(
        PROCESSED_DIR / "dataset1_balanced_500_seed42.csv",
        encoding="utf-8-sig",
    )
    first_ten = pd.read_csv(
        PROCESSED_DIR / "dataset1_balanced_500_seed42_first10.csv",
        encoding="utf-8-sig",
    )
    with (PROCESSED_DIR / "dataset1_audit.json").open(encoding="utf-8") as f:
        audit = json.load(f)

    assert list(cleaned.columns) == CANONICAL_COLUMNS
    assert list(sampled.columns) == CANONICAL_COLUMNS
    assert list(first_ten.columns) == CANONICAL_COLUMNS
    assert sampled["ID"].is_unique
    assert sampled["sentiment"].value_counts().to_dict() == {
        "Indicator": 125,
        "Ideation": 125,
        "Behavior": 125,
        "Attempt": 125,
    }
    pd.testing.assert_frame_equal(first_ten, sampled.head(10).reset_index(drop=True))
    assert audit["normalized_columns"] == [*CANONICAL_COLUMNS, "_normalized_text"]
    assert "source_row" not in audit["normalized_columns"]


def test_prepare_dataset_files_writes_cleaned_and_sample(tmp_path):
    payload = prepare_dataset_files(data_path=DATASET_PATH, output_dir=tmp_path, sample_size=500, seed=42)

    assert payload["dataset1_count"] == 9800
    assert payload["sample_size"] == 500
    assert payload["sample_label_counts"] == {
        "Indicator": 125,
        "Ideation": 125,
        "Behavior": 125,
        "Attempt": 125,
    }
    assert payload["sample_duplicate_normalized_texts"] == 0
    assert (tmp_path / "dataset1_cleaned.csv").exists()
    assert (tmp_path / "dataset1_balanced_500_seed42.csv").exists()
    assert (tmp_path / "dataset1_audit.json").exists()

    cleaned = pd.read_csv(tmp_path / "dataset1_cleaned.csv", encoding="utf-8-sig")
    sampled = pd.read_csv(tmp_path / "dataset1_balanced_500_seed42.csv", encoding="utf-8-sig")
    assert list(cleaned.columns) == CANONICAL_COLUMNS
    assert list(sampled.columns) == CANONICAL_COLUMNS
    assert cleaned["ID"].min() == 0
    assert cleaned["ID"].is_unique
    assert (tmp_path / "dataset1_cleaned.csv").read_bytes().startswith(b"\xef\xbb\xbf")
    assert (tmp_path / "dataset1_balanced_500_seed42.csv").read_bytes().startswith(b"\xef\xbb\xbf")

    row = cleaned[cleaned["ID"].eq(14468)].iloc[0]
    assert row["text"].startswith("I’m sitting here with my Glock in my hand")
    assert "?€?" not in row["text"]

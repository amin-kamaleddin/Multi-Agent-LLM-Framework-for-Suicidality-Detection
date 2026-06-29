from pathlib import Path

from suicidal_multi_agent.data.dataset import (
    audit_dataset,
    duplicate_normalized_text_count,
    prepare_dataset_files,
    sample_balanced,
)


DATASET_PATH = Path(__file__).resolve().parents[2] / "dataset" / "anonymizing_suicidal_dataset.csv"


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
    assert sampled["source_row"].is_unique
    assert list(sampled.columns) == ["source_row", "text", "sentiment"]
    assert duplicate_normalized_text_count(sampled) == 0


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

    import pandas as pd

    cleaned = pd.read_csv(tmp_path / "dataset1_cleaned.csv", encoding="utf-8-sig")
    sampled = pd.read_csv(tmp_path / "dataset1_balanced_500_seed42.csv", encoding="utf-8-sig")
    assert list(cleaned.columns) == ["source_row", "text", "sentiment"]
    assert list(sampled.columns) == ["source_row", "text", "sentiment"]
    assert cleaned["source_row"].min() == 2
    assert cleaned["source_row"].is_unique
    assert (tmp_path / "dataset1_cleaned.csv").read_bytes().startswith(b"\xef\xbb\xbf")
    assert (tmp_path / "dataset1_balanced_500_seed42.csv").read_bytes().startswith(b"\xef\xbb\xbf")

    row = cleaned[cleaned["source_row"].eq(14470)].iloc[0]
    assert row["text"].startswith("I’m sitting here with my Glock in my hand")
    assert "?€?" not in row["text"]

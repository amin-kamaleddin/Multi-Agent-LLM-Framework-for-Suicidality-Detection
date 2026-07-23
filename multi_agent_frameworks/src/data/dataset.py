"""Dataset loading, auditing, Dataset1 reproduction, and sampling."""

from __future__ import annotations

from collections import Counter
import datetime as dt
import json
from pathlib import Path
from typing import Any
import uuid

import pandas as pd

from domain.labels import VALID_LABELS


DEFAULT_DATASET_PATH = Path("dataset/anonymizing_suicidal_dataset.csv")
DEFAULT_PROCESSED_DIR = Path("dataset/processed")


def normalize_text(text: Any) -> str:
    """Match the legacy dedupe normalization: CRLF to LF plus strip."""
    return str(text if text is not None else "").replace("\r\n", "\n").strip()


def load_raw_dataset(path: str | Path = DEFAULT_DATASET_PATH) -> pd.DataFrame:
    """Load the active CSV and normalize BOM-prefixed headers."""
    csv_path = Path(path)
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df.columns = [str(col).lstrip("\ufeff").strip() for col in df.columns]
    required = {"users", "text", "sentiment", "time"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()
    if "ID" not in df.columns:
        # Match the repository's canonical RSD15K contract: a stable, zero-based
        # per-row identifier distinct from the non-unique author identifier.
        df.insert(0, "ID", range(len(df)))
    if df["ID"].duplicated().any():
        raise ValueError(f"ID must be unique in {csv_path}")
    df["_normalized_text"] = df["text"].map(normalize_text)
    return df


def reproduce_dataset1(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the legacy 9,800-row Dataset1 behavior."""
    deduped = raw_df.drop_duplicates("_normalized_text", keep="first").copy()
    user_counts = Counter(deduped["users"].astype(str))
    dataset1 = deduped[deduped["users"].astype(str).map(user_counts) < 50].copy()
    return dataset1.reset_index(drop=True)


def label_counts(df: pd.DataFrame) -> dict[str, int]:
    counts = df["sentiment"].value_counts().to_dict()
    return {label: int(counts.get(label, 0)) for label in VALID_LABELS}


def audit_dataset(path: str | Path = DEFAULT_DATASET_PATH) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Return raw data, Dataset1, and an audit payload."""
    raw_df = load_raw_dataset(path)
    dataset1 = reproduce_dataset1(raw_df)
    normalized_texts = raw_df["_normalized_text"].tolist()

    audit = {
        "source_path": str(Path(path)),
        "normalized_columns": list(raw_df.columns),
        "bom_header_normalized": "users" in raw_df.columns,
        "raw_count": int(len(raw_df)),
        "raw_label_counts": label_counts(raw_df),
        "missing_text": int((raw_df["_normalized_text"] == "").sum()),
        "duplicate_normalized_texts": int(len(normalized_texts) - len(set(normalized_texts))),
        "dataset1_policy": (
            "Exact normalized text dedupe, then remove users with 50 or more posts "
            "after deduplication. This reproduces the old 9,800-row Dataset1 behavior."
        ),
        "dataset1_count": int(len(dataset1)),
        "dataset1_label_counts": label_counts(dataset1),
    }
    return raw_df, dataset1, audit


def sample_balanced(
    dataset1: pd.DataFrame,
    *,
    sample_size: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """Sample the same number of rows per label from Dataset1."""
    if sample_size % len(VALID_LABELS) != 0:
        raise ValueError(f"sample_size must be divisible by {len(VALID_LABELS)}")
    per_label = sample_size // len(VALID_LABELS)

    sampled_groups = []
    for label in VALID_LABELS:
        group = dataset1[dataset1["sentiment"] == label]
        if len(group) < per_label:
            raise ValueError(f"Not enough rows for label {label}: need {per_label}, found {len(group)}")
        sampled_groups.append(group.sample(n=per_label, random_state=seed))

    sampled = pd.concat(sampled_groups, ignore_index=True)
    sampled = sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return public_dataset_columns(sampled)


def public_dataset_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Preserve the original RSD15K fields needed by every framework."""
    columns = ["ID", "users", "text", "sentiment", "time"]
    missing = set(columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing canonical RSD15K columns: {sorted(missing)}")
    return df[columns].copy()


def duplicate_normalized_text_count(df: pd.DataFrame) -> int:
    """Count duplicate normalized text occurrences in a dataframe."""
    normalized = df["text"].map(normalize_text)
    return int(len(normalized) - len(set(normalized)))


def _timestamped_fallback_path(path: Path) -> Path:
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return path.with_name(f"{path.stem}_{stamp}{path.suffix}")


def _write_csv_utf8_sig(df: pd.DataFrame, path: Path) -> Path:
    """Write a CSV with UTF-8 BOM, falling back if the target file is locked."""
    temp_path = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.tmp")
    df.to_csv(temp_path, index=False, encoding="utf-8-sig")
    try:
        temp_path.replace(path)
        return path
    except PermissionError:
        fallback = _timestamped_fallback_path(path)
        temp_path.replace(fallback)
        return fallback


def _write_json_utf8(payload: dict[str, Any], path: Path) -> Path:
    """Write JSON, falling back if the target file is locked."""
    temp_path = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.tmp")
    with temp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    try:
        temp_path.replace(path)
        return path
    except PermissionError:
        fallback = _timestamped_fallback_path(path)
        temp_path.replace(fallback)
        return fallback


def prepare_dataset_files(
    *,
    data_path: str | Path = DEFAULT_DATASET_PATH,
    output_dir: str | Path = DEFAULT_PROCESSED_DIR,
    sample_size: int = 500,
    seed: int = 42,
) -> dict[str, Any]:
    """Write canonical cleaned Dataset1 and balanced sample CSV files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    _, dataset1, audit = audit_dataset(data_path)
    cleaned_public = public_dataset_columns(dataset1)
    sampled = sample_balanced(dataset1, sample_size=sample_size, seed=seed)

    cleaned_path = output_path / "dataset1_cleaned.csv"
    sample_path = output_path / f"dataset1_balanced_{sample_size}_seed{seed}.csv"
    audit_path = output_path / "dataset1_audit.json"

    actual_cleaned_path = _write_csv_utf8_sig(cleaned_public, cleaned_path)
    actual_sample_path = _write_csv_utf8_sig(sampled, sample_path)

    sample_label_counts = label_counts(sampled)
    sample_duplicate_texts = duplicate_normalized_text_count(sampled)
    payload = {
        **audit,
        "cleaned_dataset_path": str(actual_cleaned_path),
        "sampled_dataset_path": str(actual_sample_path),
        "sample_size": int(len(sampled)),
        "sample_seed": int(seed),
        "sample_label_counts": sample_label_counts,
        "sample_duplicate_normalized_texts": sample_duplicate_texts,
        "sample_balanced": len(set(sample_label_counts.values())) == 1,
    }

    actual_audit_path = _write_json_utf8(payload, audit_path)
    payload["audit_path"] = str(actual_audit_path)
    _write_json_utf8(payload, actual_audit_path)

    return payload

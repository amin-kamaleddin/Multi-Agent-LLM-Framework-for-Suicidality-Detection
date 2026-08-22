from pathlib import Path
import hashlib
import json
import platform

import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = Path(
    "data/raw/suicidality_full_dataset_posneg_balanced.csv"
)

OUTPUT_DIR = Path("data/splits")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Fixed seed used for ALL dataset splitting.
# Do not change this after creating the canonical splits.
RANDOM_STATE = 42

# Canonical five-class task
ALL_LABELS = [
    "Unsupportive",
    "Indicator",
    "Ideation",
    "Behavior",
    "Attempt",
]

# Used only to create derived positive-only evaluation subsets.
# The model will still be trained on ALL five classes.
POSITIVE_LABELS = [
    "Indicator",
    "Ideation",
    "Behavior",
    "Attempt",
]

# Expected properties of the current canonical dataset.
# These checks help ensure the wrong dataset is not accidentally used.
EXPECTED_TOTAL_ROWS = 28792

EXPECTED_LABEL_COUNTS = {
    "Unsupportive": 14396,
    "Indicator": 4550,
    "Ideation": 7022,
    "Behavior": 2039,
    "Attempt": 785,
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def sha256_file(path: Path) -> str:
    """
    Compute SHA-256 hash of a file.

    This lets us document exactly which raw dataset and split files
    were used in the experiment.
    """
    sha256 = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def get_label_counts(dataframe: pd.DataFrame) -> dict:
    """
    Return label counts in a fixed canonical order.
    """
    counts = (
        dataframe["sentiment"]
        .value_counts()
        .reindex(ALL_LABELS, fill_value=0)
    )

    return {
        label: int(counts[label])
        for label in ALL_LABELS
    }


def print_split_report(name: str, dataframe: pd.DataFrame) -> None:
    """
    Print row counts and class proportions for a split.
    """
    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print(f"Total rows: {len(dataframe):,}")

    counts = (
        dataframe["sentiment"]
        .value_counts()
        .reindex(ALL_LABELS, fill_value=0)
    )

    proportions = (
        dataframe["sentiment"]
        .value_counts(normalize=True)
        .reindex(ALL_LABELS, fill_value=0)
        .round(4)
    )

    print("\nCounts:")
    print(counts)

    print("\nProportions:")
    print(proportions)


# ============================================================
# 1. LOAD RAW DATASET
# ============================================================

print("Loading dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"\nRaw rows loaded: {len(df):,}")

print("\nRaw label distribution:")
print(
    df["sentiment"]
    .value_counts()
    .reindex(ALL_LABELS, fill_value=0)
)


# ============================================================
# 2. BASIC DATA VALIDATION
# ============================================================

required_columns = {
    "ID",
    "text",
    "sentiment",
}

missing_columns = required_columns - set(df.columns)

assert not missing_columns, (
    f"Required columns are missing: {missing_columns}"
)

assert len(df) == EXPECTED_TOTAL_ROWS, (
    f"Expected {EXPECTED_TOTAL_ROWS:,} rows, "
    f"but found {len(df):,}."
)

assert df["ID"].notna().all(), (
    "Missing values found in ID column."
)

assert df["ID"].is_unique, (
    "Duplicate IDs found."
)

assert df["text"].notna().all(), (
    "Missing text values found."
)

assert df["sentiment"].notna().all(), (
    "Missing sentiment labels found."
)

unexpected_labels = (
    set(df["sentiment"].unique())
    - set(ALL_LABELS)
)

assert not unexpected_labels, (
    f"Unexpected labels found: {unexpected_labels}"
)

actual_label_counts = get_label_counts(df)

assert actual_label_counts == EXPECTED_LABEL_COUNTS, (
    "\nDataset label counts do not match the expected "
    "canonical dataset.\n"
    f"Expected: {EXPECTED_LABEL_COUNTS}\n"
    f"Actual:   {actual_label_counts}"
)

print("\nDataset validation passed.")


# ============================================================
# 3. STABLE ORDERING FOR REPRODUCIBILITY
# ============================================================

# train_test_split is deterministic for a fixed seed GIVEN
# the same input ordering.
#
# Sorting by unique ID ensures that accidental changes to CSV row
# order do not change the resulting train/validation/test assignment.

df = (
    df
    .sort_values("ID", kind="stable")
    .reset_index(drop=True)
)

print("\nDataset sorted by ID for reproducibility.")


# ============================================================
# 4. CREATE STRATIFIED 70 / 15 / 15 SPLIT
# ============================================================

# ------------------------------------------------------------
# First split:
#
# 70% -> training
# 30% -> temporary validation/test pool
# ------------------------------------------------------------

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=RANDOM_STATE,
    stratify=df["sentiment"],
)


# ------------------------------------------------------------
# Second split:
#
# Split the remaining 30% exactly in half:
#
# 15% -> validation
# 15% -> test
# ------------------------------------------------------------

validation_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=temp_df["sentiment"],
)


# ============================================================
# 5. VERIFY SPLIT INTEGRITY
# ============================================================

train_ids = set(train_df["ID"])
validation_ids = set(validation_df["ID"])
test_ids = set(test_df["ID"])


# No row is allowed to appear in more than one split.

assert train_ids.isdisjoint(validation_ids), (
    "ERROR: Train and validation IDs overlap."
)

assert train_ids.isdisjoint(test_ids), (
    "ERROR: Train and test IDs overlap."
)

assert validation_ids.isdisjoint(test_ids), (
    "ERROR: Validation and test IDs overlap."
)


# Verify that every original row appears exactly once.

assert (
    len(train_df)
    + len(validation_df)
    + len(test_df)
    == len(df)
), (
    "ERROR: Split sizes do not add up to the full dataset."
)

all_split_ids = (
    train_ids
    | validation_ids
    | test_ids
)

assert len(all_split_ids) == len(df), (
    "ERROR: Some IDs are missing or duplicated across splits."
)


print("\nSplit integrity checks passed.")


# ============================================================
# 6. SORT EACH FINISHED SPLIT BY ID
# ============================================================

# This does NOT change which split a record belongs to.
# It merely gives each saved CSV a deterministic row order.

train_df = (
    train_df
    .sort_values("ID", kind="stable")
    .reset_index(drop=True)
)

validation_df = (
    validation_df
    .sort_values("ID", kind="stable")
    .reset_index(drop=True)
)

test_df = (
    test_df
    .sort_values("ID", kind="stable")
    .reset_index(drop=True)
)


# ============================================================
# 7. SAVE CANONICAL FIVE-CLASS SPLITS
# ============================================================

train_path = OUTPUT_DIR / "train.csv"
validation_path = OUTPUT_DIR / "validation.csv"
test_path = OUTPUT_DIR / "test.csv"

train_df.to_csv(
    train_path,
    index=False,
)

validation_df.to_csv(
    validation_path,
    index=False,
)

test_df.to_csv(
    test_path,
    index=False,
)


# ============================================================
# 8. CREATE POSITIVE-ONLY DERIVED SUBSETS
# ============================================================

# IMPORTANT:
#
# These are NOT new/random splits.
#
# They are simply the positive-label records belonging to the
# already-frozen five-class partitions.
#
# This allows us later to calculate:
#
#   Indicator vs Ideation vs Behavior vs Attempt
#
# performance using exactly the same held-out population.

train_positive = train_df[
    train_df["sentiment"].isin(POSITIVE_LABELS)
].copy()

validation_positive = validation_df[
    validation_df["sentiment"].isin(POSITIVE_LABELS)
].copy()

test_positive = test_df[
    test_df["sentiment"].isin(POSITIVE_LABELS)
].copy()


train_positive_path = (
    OUTPUT_DIR / "train_positive.csv"
)

validation_positive_path = (
    OUTPUT_DIR / "validation_positive.csv"
)

test_positive_path = (
    OUTPUT_DIR / "test_positive.csv"
)


train_positive.to_csv(
    train_positive_path,
    index=False,
)

validation_positive.to_csv(
    validation_positive_path,
    index=False,
)

test_positive.to_csv(
    test_positive_path,
    index=False,
)


# ============================================================
# 9. PRINT FIVE-CLASS SPLIT REPORT
# ============================================================

print_split_report(
    "TRAIN — FIVE CLASS",
    train_df,
)

print_split_report(
    "VALIDATION — FIVE CLASS",
    validation_df,
)

print_split_report(
    "TEST — FIVE CLASS",
    test_df,
)


# ============================================================
# 10. PRINT POSITIVE-ONLY SUBSET REPORT
# ============================================================

for name, split in [
    ("TRAIN — POSITIVE ONLY", train_positive),
    ("VALIDATION — POSITIVE ONLY", validation_positive),
    ("TEST — POSITIVE ONLY", test_positive),
]:

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print(f"Total rows: {len(split):,}")

    print("\nCounts:")

    print(
        split["sentiment"]
        .value_counts()
        .reindex(
            POSITIVE_LABELS,
            fill_value=0,
        )
    )


# ============================================================
# 11. VERIFY EXPECTED OVERALL SPLIT SIZES
# ============================================================

# For 28,792 total rows using this 70/15/15 procedure:

assert len(train_df) == 20154, (
    f"Unexpected training size: {len(train_df)}"
)

assert len(validation_df) == 4319, (
    f"Unexpected validation size: {len(validation_df)}"
)

assert len(test_df) == 4319, (
    f"Unexpected test size: {len(test_df)}"
)


# ============================================================
# 12. CREATE REPRODUCIBILITY MANIFEST
# ============================================================

# The manifest documents:
#
# - random seed
# - split methodology
# - dataset hash
# - exact class counts
# - software versions
# - hashes of the resulting split files
#
# This is useful for both the paper and a public repository.

manifest = {
    "experiment": "MentalRoBERTa five-class fine-tuning dataset split",

    "input_file": str(INPUT_FILE),

    "input_file_sha256": sha256_file(INPUT_FILE),

    "random_seed": RANDOM_STATE,

    "sorting_column": "ID",

    "split_unit": "individual post",

    "stratification_column": "sentiment",

    "split_strategy": (
        "Stratified post-level train/validation/test split"
    ),

    "train_fraction": 0.70,
    "validation_fraction": 0.15,
    "test_fraction": 0.15,

    "labels": ALL_LABELS,

    "positive_labels": POSITIVE_LABELS,

    "total_rows": int(len(df)),

    "full_dataset_label_counts": get_label_counts(df),

    "train_rows": int(len(train_df)),
    "validation_rows": int(len(validation_df)),
    "test_rows": int(len(test_df)),

    "train_label_counts": get_label_counts(train_df),

    "validation_label_counts": get_label_counts(
        validation_df
    ),

    "test_label_counts": get_label_counts(test_df),

    "train_positive_rows": int(
        len(train_positive)
    ),

    "validation_positive_rows": int(
        len(validation_positive)
    ),

    "test_positive_rows": int(
        len(test_positive)
    ),

    "software_versions": {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
    },

    "output_file_sha256": {
        "train.csv": sha256_file(train_path),

        "validation.csv": sha256_file(
            validation_path
        ),

        "test.csv": sha256_file(test_path),

        "train_positive.csv": sha256_file(
            train_positive_path
        ),

        "validation_positive.csv": sha256_file(
            validation_positive_path
        ),

        "test_positive.csv": sha256_file(
            test_positive_path
        ),
    },
}


manifest_path = (
    OUTPUT_DIR / "split_manifest.json"
)

with open(
    manifest_path,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        manifest,
        f,
        indent=4,
        ensure_ascii=False,
    )


# ============================================================
# 13. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("SPLIT COMPLETE")
print("=" * 60)

print("\nFive-class canonical splits:")

print(
    f"Train:      {len(train_df):,}"
)

print(
    f"Validation: {len(validation_df):,}"
)

print(
    f"Test:       {len(test_df):,}"
)

print("\nPositive-only derived subsets:")

print(
    f"Train positive:      "
    f"{len(train_positive):,}"
)

print(
    f"Validation positive: "
    f"{len(validation_positive):,}"
)

print(
    f"Test positive:       "
    f"{len(test_positive):,}"
)

print(
    f"\nRandom seed: {RANDOM_STATE}"
)

print(
    f"Manifest saved to: "
    f"{manifest_path.resolve()}"
)

print(
    f"All split files saved to: "
    f"{OUTPUT_DIR.resolve()}"
)

print(
    "\nIMPORTANT: These are now the canonical experiment splits. "
    "Do not regenerate or reshuffle them once training begins."
)
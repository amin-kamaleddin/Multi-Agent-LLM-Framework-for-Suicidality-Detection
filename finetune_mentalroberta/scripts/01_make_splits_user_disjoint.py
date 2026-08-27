from pathlib import Path
import hashlib
import json
import platform

import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedGroupKFold


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = Path("data/raw/suicidality_full_dataset_posneg_balanced.csv")

OUTPUT_DIR = Path("data/splits")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
GROUP_COLUMN = "users"

# 20 user-disjoint folds -> 14 train, 3 validation, 3 test.
# This targets ~70/15/15 of POSTS while keeping every user in
# exactly one split. Exact post percentages are not required.
N_FOLDS = 20
TRAIN_FOLDS = set(range(0, 14))
VALIDATION_FOLDS = set(range(14, 17))
TEST_FOLDS = set(range(17, 20))

TARGET_TRAIN_FRACTION = 0.70
TARGET_VALIDATION_FRACTION = 0.15
TARGET_TEST_FRACTION = 0.15

ALL_LABELS = [
    "Unsupportive",
    "Indicator",
    "Ideation",
    "Behavior",
    "Attempt",
]

POSITIVE_LABELS = [
    "Indicator",
    "Ideation",
    "Behavior",
    "Attempt",
]

EXPECTED_TOTAL_ROWS = 28792

EXPECTED_LABEL_COUNTS = {
    "Unsupportive": 14396,
    "Indicator": 4550,
    "Ideation": 7022,
    "Behavior": 2039,
    "Attempt": 785,
}


# ============================================================
# HELPERS
# ============================================================

def sha256_file(path: Path) -> str:
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def get_label_counts(dataframe: pd.DataFrame) -> dict:
    counts = (
        dataframe["sentiment"]
        .value_counts()
        .reindex(ALL_LABELS, fill_value=0)
    )
    return {label: int(counts[label]) for label in ALL_LABELS}


def get_label_proportions(dataframe: pd.DataFrame) -> dict:
    proportions = (
        dataframe["sentiment"]
        .value_counts(normalize=True)
        .reindex(ALL_LABELS, fill_value=0)
    )
    return {label: float(proportions[label]) for label in ALL_LABELS}


def get_unique_user_count(dataframe: pd.DataFrame) -> int:
    return int(dataframe[GROUP_COLUMN].nunique())


def print_split_report(
    name: str,
    dataframe: pd.DataFrame,
    total_rows: int,
    total_users: int,
) -> None:
    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    row_fraction = len(dataframe) / total_rows
    user_fraction = dataframe[GROUP_COLUMN].nunique() / total_users

    print(f"Total rows:   {len(dataframe):,} ({row_fraction:.2%})")
    print(
        f"Unique users: {dataframe[GROUP_COLUMN].nunique():,} "
        f"({user_fraction:.2%})"
    )

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

required_columns = {"ID", "users", "text", "sentiment"}
missing_columns = required_columns - set(df.columns)

assert not missing_columns, (
    f"Required columns are missing: {missing_columns}"
)

assert len(df) == EXPECTED_TOTAL_ROWS, (
    f"Expected {EXPECTED_TOTAL_ROWS:,} rows, but found {len(df):,}."
)

assert df["ID"].notna().all(), "Missing values found in ID column."
assert df["ID"].is_unique, "Duplicate IDs found."
assert df["text"].notna().all(), "Missing text values found."
assert df["sentiment"].notna().all(), "Missing sentiment labels found."

# User identity is now required because the split unit is the user.
assert df[GROUP_COLUMN].notna().all(), (
    f"Missing values found in {GROUP_COLUMN} column. "
    "A user-disjoint split cannot be created until these are resolved."
)

assert df[GROUP_COLUMN].astype(str).str.strip().ne("").all(), (
    f"Blank user identifiers found in {GROUP_COLUMN} column."
)

unexpected_labels = set(df["sentiment"].unique()) - set(ALL_LABELS)
assert not unexpected_labels, f"Unexpected labels found: {unexpected_labels}"

actual_label_counts = get_label_counts(df)
assert actual_label_counts == EXPECTED_LABEL_COUNTS, (
    "\nDataset label counts do not match the expected canonical dataset.\n"
    f"Expected: {EXPECTED_LABEL_COUNTS}\n"
    f"Actual:   {actual_label_counts}"
)

total_unique_users = get_unique_user_count(df)

print("\nDataset validation passed.")
print(f"Unique users in full dataset: {total_unique_users:,}")


# ============================================================
# 3. STABLE ORDERING FOR REPRODUCIBILITY
# ============================================================

df = (
    df
    .sort_values("ID", kind="stable")
    .reset_index(drop=True)
)

print("\nDataset sorted by ID for reproducibility.")


# ============================================================
# 4. CREATE USER-DISJOINT ~70 / 15 / 15 SPLIT
# ============================================================

# StratifiedGroupKFold tries to preserve class proportions while
# ensuring that an entire user belongs to only one fold.
#
# 20 folds are convenient here because:
#   14 / 20 = 70%
#    3 / 20 = 15%
#    3 / 20 = 15%
#
# Since users have different numbers of posts, the resulting POST
# percentages will be approximate, not necessarily exact.

print(
    "\nCreating 20 stratified, user-disjoint folds and "
    "combining them into train/validation/test..."
)

splitter = StratifiedGroupKFold(
    n_splits=N_FOLDS,
    shuffle=True,
    random_state=RANDOM_STATE,
)

fold_assignment = pd.Series(index=df.index, dtype="int64")

for fold_number, (_, fold_indices) in enumerate(
    splitter.split(
        X=df,
        y=df["sentiment"],
        groups=df[GROUP_COLUMN],
    )
):
    fold_assignment.iloc[fold_indices] = fold_number

assert fold_assignment.notna().all(), (
    "ERROR: Some rows were not assigned to a fold."
)

df = df.copy()
df["_split_fold"] = fold_assignment.astype(int)

train_df = df[df["_split_fold"].isin(TRAIN_FOLDS)].copy()
validation_df = df[df["_split_fold"].isin(VALIDATION_FOLDS)].copy()
test_df = df[df["_split_fold"].isin(TEST_FOLDS)].copy()

for split_df in (train_df, validation_df, test_df):
    split_df.drop(columns=["_split_fold"], inplace=True)


# ============================================================
# 5. VERIFY SPLIT INTEGRITY
# ============================================================

train_ids = set(train_df["ID"])
validation_ids = set(validation_df["ID"])
test_ids = set(test_df["ID"])

train_users = set(train_df[GROUP_COLUMN])
validation_users = set(validation_df[GROUP_COLUMN])
test_users = set(test_df[GROUP_COLUMN])

# No POST overlap.
assert train_ids.isdisjoint(validation_ids), (
    "ERROR: Train and validation post IDs overlap."
)
assert train_ids.isdisjoint(test_ids), (
    "ERROR: Train and test post IDs overlap."
)
assert validation_ids.isdisjoint(test_ids), (
    "ERROR: Validation and test post IDs overlap."
)

# No USER overlap. This is the critical leakage-prevention rule.
assert train_users.isdisjoint(validation_users), (
    "ERROR: Train and validation USERS overlap."
)
assert train_users.isdisjoint(test_users), (
    "ERROR: Train and test USERS overlap."
)
assert validation_users.isdisjoint(test_users), (
    "ERROR: Validation and test USERS overlap."
)

# Every original row appears exactly once.
assert (
    len(train_df) + len(validation_df) + len(test_df) == len(df)
), "ERROR: Split sizes do not add up to the full dataset."

all_split_ids = train_ids | validation_ids | test_ids
assert len(all_split_ids) == len(df), (
    "ERROR: Some post IDs are missing or duplicated across splits."
)

# Every original user appears in exactly one split.
all_split_users = train_users | validation_users | test_users
assert len(all_split_users) == total_unique_users, (
    "ERROR: Some users are missing from the split assignment."
)

# Every class remains represented in every split.
for split_name, split_df in [
    ("train", train_df),
    ("validation", validation_df),
    ("test", test_df),
]:
    missing_labels = set(ALL_LABELS) - set(split_df["sentiment"].unique())
    assert not missing_labels, (
        f"ERROR: {split_name} split is missing labels: {missing_labels}"
    )

print("\nSplit integrity checks passed.")
print("\nUSER OVERLAP CHECK:")
print(f"Train ∩ Validation users: {len(train_users & validation_users)}")
print(f"Train ∩ Test users:       {len(train_users & test_users)}")
print(f"Validation ∩ Test users:  {len(validation_users & test_users)}")


# ============================================================
# 6. SORT EACH FINISHED SPLIT BY ID
# ============================================================

train_df = train_df.sort_values("ID", kind="stable").reset_index(drop=True)
validation_df = (
    validation_df.sort_values("ID", kind="stable").reset_index(drop=True)
)
test_df = test_df.sort_values("ID", kind="stable").reset_index(drop=True)


# ============================================================
# 7. SAVE FIVE-CLASS SPLITS
# ============================================================

train_path = OUTPUT_DIR / "train.csv"
validation_path = OUTPUT_DIR / "validation.csv"
test_path = OUTPUT_DIR / "test.csv"

train_df.to_csv(train_path, index=False)
validation_df.to_csv(validation_path, index=False)
test_df.to_csv(test_path, index=False)


# ============================================================
# 8. CREATE POSITIVE-ONLY DERIVED SUBSETS
# ============================================================

# These are not separate random splits. They are positive-label rows
# from the already-created five-class user-disjoint partitions.

train_positive = train_df[
    train_df["sentiment"].isin(POSITIVE_LABELS)
].copy()

validation_positive = validation_df[
    validation_df["sentiment"].isin(POSITIVE_LABELS)
].copy()

test_positive = test_df[
    test_df["sentiment"].isin(POSITIVE_LABELS)
].copy()

train_positive_path = OUTPUT_DIR / "train_positive.csv"
validation_positive_path = OUTPUT_DIR / "validation_positive.csv"
test_positive_path = OUTPUT_DIR / "test_positive.csv"

train_positive.to_csv(train_positive_path, index=False)
validation_positive.to_csv(validation_positive_path, index=False)
test_positive.to_csv(test_positive_path, index=False)


# ============================================================
# 9. PRINT FIVE-CLASS SPLIT REPORT
# ============================================================

print_split_report(
    "TRAIN — FIVE CLASS",
    train_df,
    total_rows=len(df),
    total_users=total_unique_users,
)

print_split_report(
    "VALIDATION — FIVE CLASS",
    validation_df,
    total_rows=len(df),
    total_users=total_unique_users,
)

print_split_report(
    "TEST — FIVE CLASS",
    test_df,
    total_rows=len(df),
    total_users=total_unique_users,
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
    print(f"Unique users: {split[GROUP_COLUMN].nunique():,}")

    print("\nCounts:")
    print(
        split["sentiment"]
        .value_counts()
        .reindex(POSITIVE_LABELS, fill_value=0)
    )


# ============================================================
# 11. REPORT ACTUAL SPLIT FRACTIONS
# ============================================================

train_fraction = len(train_df) / len(df)
validation_fraction = len(validation_df) / len(df)
test_fraction = len(test_df) / len(df)

print("\n" + "=" * 60)
print("ACTUAL POST-LEVEL SPLIT FRACTIONS")
print("=" * 60)

print(
    f"Train:      {train_fraction:.4f} ({train_fraction:.2%}) "
    f"[target {TARGET_TRAIN_FRACTION:.0%}]"
)
print(
    f"Validation: {validation_fraction:.4f} ({validation_fraction:.2%}) "
    f"[target {TARGET_VALIDATION_FRACTION:.0%}]"
)
print(
    f"Test:       {test_fraction:.4f} ({test_fraction:.2%}) "
    f"[target {TARGET_TEST_FRACTION:.0%}]"
)

print(
    "\nNOTE: Exact 70/15/15 post counts are not required. "
    "User disjointness takes priority."
)


# ============================================================
# 12. CREATE REPRODUCIBILITY MANIFEST
# ============================================================

manifest = {
    "experiment": (
        "MentalRoBERTa five-class fine-tuning user-disjoint dataset split"
    ),
    "input_file": str(INPUT_FILE),
    "input_file_sha256": sha256_file(INPUT_FILE),
    "random_seed": RANDOM_STATE,
    "sorting_column": "ID",
    "split_unit": "user",
    "group_column": GROUP_COLUMN,
    "stratification_column": "sentiment",
    "split_strategy": (
        "User-disjoint approximately stratified train/validation/test "
        "split using StratifiedGroupKFold with 20 folds; "
        "14 folds assigned to train, 3 to validation, and 3 to test."
    ),
    "n_stratified_group_folds": N_FOLDS,
    "train_fold_numbers": sorted(TRAIN_FOLDS),
    "validation_fold_numbers": sorted(VALIDATION_FOLDS),
    "test_fold_numbers": sorted(TEST_FOLDS),
    "target_train_fraction": TARGET_TRAIN_FRACTION,
    "target_validation_fraction": TARGET_VALIDATION_FRACTION,
    "target_test_fraction": TARGET_TEST_FRACTION,
    "actual_train_fraction": train_fraction,
    "actual_validation_fraction": validation_fraction,
    "actual_test_fraction": test_fraction,
    "labels": ALL_LABELS,
    "positive_labels": POSITIVE_LABELS,
    "total_rows": int(len(df)),
    "total_unique_users": total_unique_users,
    "full_dataset_label_counts": get_label_counts(df),
    "full_dataset_label_proportions": get_label_proportions(df),
    "train_rows": int(len(train_df)),
    "validation_rows": int(len(validation_df)),
    "test_rows": int(len(test_df)),
    "train_unique_users": get_unique_user_count(train_df),
    "validation_unique_users": get_unique_user_count(validation_df),
    "test_unique_users": get_unique_user_count(test_df),
    "train_label_counts": get_label_counts(train_df),
    "validation_label_counts": get_label_counts(validation_df),
    "test_label_counts": get_label_counts(test_df),
    "train_label_proportions": get_label_proportions(train_df),
    "validation_label_proportions": get_label_proportions(validation_df),
    "test_label_proportions": get_label_proportions(test_df),
    "train_positive_rows": int(len(train_positive)),
    "validation_positive_rows": int(len(validation_positive)),
    "test_positive_rows": int(len(test_positive)),
    "user_overlap_counts": {
        "train_validation": int(len(train_users & validation_users)),
        "train_test": int(len(train_users & test_users)),
        "validation_test": int(len(validation_users & test_users)),
    },
    "software_versions": {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
    },
    "output_file_sha256": {
        "train.csv": sha256_file(train_path),
        "validation.csv": sha256_file(validation_path),
        "test.csv": sha256_file(test_path),
        "train_positive.csv": sha256_file(train_positive_path),
        "validation_positive.csv": sha256_file(validation_positive_path),
        "test_positive.csv": sha256_file(test_positive_path),
    },
}

manifest_path = OUTPUT_DIR / "split_manifest.json"

with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=4, ensure_ascii=False)


# ============================================================
# 13. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("USER-DISJOINT SPLIT COMPLETE")
print("=" * 60)

print("\nFive-class canonical splits:")
print(
    f"Train:      {len(train_df):,} posts / "
    f"{get_unique_user_count(train_df):,} users"
)
print(
    f"Validation: {len(validation_df):,} posts / "
    f"{get_unique_user_count(validation_df):,} users"
)
print(
    f"Test:       {len(test_df):,} posts / "
    f"{get_unique_user_count(test_df):,} users"
)

print("\nPositive-only derived subsets:")
print(f"Train positive:      {len(train_positive):,}")
print(f"Validation positive: {len(validation_positive):,}")
print(f"Test positive:       {len(test_positive):,}")

print("\nUser overlap across train/validation/test: 0")
print(f"\nRandom seed: {RANDOM_STATE}")
print(f"Manifest saved to: {manifest_path.resolve()}")
print(f"All split files saved to: {OUTPUT_DIR.resolve()}")

print(
    "\nIMPORTANT: Once these user-disjoint splits are accepted "
    "and training begins, freeze them. Do not regenerate or reshuffle them."
)

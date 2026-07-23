"""Restore canonical RSD15K fields on the two frozen evaluation cohorts.

The selected records do not change. Positive legacy ``source_row`` values are
spreadsheet row numbers, so their canonical zero-based RSD15K ID is
``source_row - 2``. Negative legacy ``source_row`` values already equal the IDs
assigned by their real or synthetic source pool.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_DIR / "dataset"
COHORT_DIR = DATASET_DIR / "positive-negative-final-datasets"
POSITIVE_COHORT = COHORT_DIR / "dataset1_positive_balanced_100_seed42.csv"
NEGATIVE_COHORT = COHORT_DIR / "negative_real_synthetic_balanced_100_seed42.csv"
POSITIVE_SOURCE = DATASET_DIR / "anonymizing_suicidal_dataset.csv"
REAL_NEGATIVE_SOURCE = (
    COHORT_DIR
    / "Suicidality_real_negative_labels_final2_LARGE - Suicidality_real_negative_labels_final2_LARGE.csv"
)
SYNTHETIC_NEGATIVE_SOURCE = (
    COHORT_DIR
    / "Suicidality_synthetic_negative_labels_final_LARGE - Suicidality_synthetic_negative_labels_final_LARGE.csv"
)
CANONICAL_COLUMNS = ["ID", "users", "text", "sentiment", "time", "real"]


def _positive_rows() -> pd.DataFrame:
    cohort = pd.read_csv(POSITIVE_COHORT, encoding="utf-8-sig")
    source = pd.read_csv(POSITIVE_SOURCE, encoding="utf-8-sig")
    rows = []
    for selected in cohort.to_dict(orient="records"):
        source_id = int(selected["ID"] if "ID" in selected else selected["source_row"] - 2)
        original = source.iloc[source_id]
        if original["text"] != selected["text"] or original["sentiment"] != selected["sentiment"]:
            raise ValueError(f"Positive source mismatch for ID={source_id}")
        rows.append(
            {
                "ID": source_id,
                "users": original["users"],
                "text": original["text"],
                "sentiment": original["sentiment"],
                "time": original["time"],
                "real": True,
            }
        )
    return pd.DataFrame(rows, columns=CANONICAL_COLUMNS)


def _negative_rows() -> pd.DataFrame:
    cohort = pd.read_csv(NEGATIVE_COHORT, encoding="utf-8-sig")
    pools = {
        True: pd.read_csv(REAL_NEGATIVE_SOURCE, encoding="utf-8-sig"),
        False: pd.read_csv(SYNTHETIC_NEGATIVE_SOURCE, encoding="utf-8-sig"),
    }
    rows = []
    for selected in cohort.to_dict(orient="records"):
        is_real = bool(selected["real"])
        source_id = int(selected["ID"] if "ID" in selected else selected["source_row"])
        pool = pools[is_real]
        matches = pool[(pool["ID"] == source_id) & (pool["text"] == selected["text"])]
        if len(matches) != 1:
            raise ValueError(f"Expected one negative source match for ID={source_id}, found {len(matches)}")
        original = matches.iloc[0]
        rows.append(
            {
                "ID": source_id,
                "users": original["users"],
                "text": original["text"],
                "sentiment": original["sentiment"],
                "time": original["time"],
                "real": is_real,
            }
        )
    return pd.DataFrame(rows, columns=CANONICAL_COLUMNS)


def main() -> int:
    positive = _positive_rows()
    negative = _negative_rows()
    if set(positive["ID"]) & set(negative["ID"]):
        raise ValueError("Positive and negative cohort IDs must not overlap")
    positive.to_csv(POSITIVE_COHORT, index=False, encoding="utf-8-sig")
    negative.to_csv(NEGATIVE_COHORT, index=False, encoding="utf-8-sig")
    print(
        {
            "positive_rows": len(positive),
            "negative_rows": len(negative),
            "positive_label_counts": positive["sentiment"].value_counts().to_dict(),
            "negative_real_counts": negative["real"].value_counts().to_dict(),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

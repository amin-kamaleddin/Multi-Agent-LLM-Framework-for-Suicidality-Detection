#!/usr/bin/env python3
"""
Check the `text` column of a dataset CSV for duplicate messages.

Primary check : EXACT (100% identical) text duplicates.
Secondary check: whitespace-normalized duplicates (same text once leading/
                 trailing/collapsed internal whitespace is ignored) -- reported
                 separately so you can decide whether they count as duplicates.

Outputs:
  - console summary
  - exact_duplicate_groups.csv : one row per (group_id, ID, row_index) for exact dupes
  - whitespace_duplicate_groups.csv : same, for whitespace-normalized-only collisions
"""
import argparse
import re
import sys
from pathlib import Path

import pandas as pd


def find_duplicate_groups(df: pd.DataFrame, key_series: pd.Series):
    """Return {normalized_key: [row_index, ...]} for keys that appear >1 time."""
    groups = {}
    for row_idx, key in key_series.items():
        groups.setdefault(key, []).append(row_idx)
    return {k: idxs for k, idxs in groups.items() if len(idxs) > 1}


def main():
    parser = argparse.ArgumentParser(description="Check for duplicate messages in a dataset CSV.")
    parser.add_argument(
        "csv_path",
        nargs="?",
        default="/mnt/user-data/uploads/suicidality_full_dataset_posneg_balanced.csv",
        help="Path to the dataset CSV.",
    )
    parser.add_argument("--text-field", default="text", help="Column holding the message text.")
    parser.add_argument("--id-field", default="ID", help="Column holding the row ID.")
    parser.add_argument("--out-dir", default=".", help="Where to write the report CSVs.")
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        print(f"File not found: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total = len(df)
    tf, idf = args.text_field, args.id_field
    if tf not in df.columns:
        print(f"Column '{tf}' not found. Columns: {list(df.columns)}")
        sys.exit(1)

    print("=" * 70)
    print(f"Duplicate check on: {csv_path.name}")
    print(f"Total rows: {total}")
    print("=" * 70)

    # ---- rows with empty / NaN text (worth knowing about) ----
    na_mask = df[tf].isna()
    if na_mask.any():
        print(f"\n[!] {int(na_mask.sum())} row(s) have empty/NaN text (excluded from dupe checks):")
        for row_idx in df.index[na_mask]:
            print(f"      row {row_idx}  (ID={df.at[row_idx, idf] if idf in df.columns else 'n/a'})")

    work = df[~na_mask].copy()

    # ---------------------------------------------------------------
    # 1) EXACT duplicates -- byte-for-byte identical text
    # ---------------------------------------------------------------
    exact_key = work[tf].astype(str)
    exact_groups = find_duplicate_groups(work, exact_key)

    n_exact_groups = len(exact_groups)
    n_exact_rows = sum(len(v) for v in exact_groups.values())
    n_exact_removable = n_exact_rows - n_exact_groups  # rows beyond the first in each group

    print("\n" + "-" * 70)
    print("PRIMARY CHECK -- EXACT (100% identical) duplicates")
    print("-" * 70)
    if n_exact_groups == 0:
        print("No exact duplicate messages found. Every text value is unique.")
    else:
        print(f"Duplicate groups        : {n_exact_groups}")
        print(f"Rows involved           : {n_exact_rows}")
        print(f"Rows to replace/remove  : {n_exact_removable}  (keeping 1 per group)")
        print("\nDetail (first 40 groups shown):")
        rows_out = []
        for gi, (key, idxs) in enumerate(
            sorted(exact_groups.items(), key=lambda kv: (-len(kv[1]), kv[1][0])), start=1
        ):
            ids = [df.at[i, idf] for i in idxs] if idf in df.columns else ["n/a"] * len(idxs)
            sents = df.loc[idxs, "sentiment"].tolist() if "sentiment" in df.columns else []
            for row_idx, rid in zip(idxs, ids):
                rows_out.append(
                    {
                        "group_id": gi,
                        "group_size": len(idxs),
                        idf: df.at[row_idx, idf] if idf in df.columns else "",
                        "row_index": row_idx,
                        "sentiment": df.at[row_idx, "sentiment"] if "sentiment" in df.columns else "",
                        "text": str(df.at[row_idx, tf]),
                    }
                )
            if gi <= 40:
                preview = re.sub(r"\s+", " ", str(key))[:90]
                sent_note = ""
                if sents and len(set(sents)) > 1:
                    sent_note = f"  [!] labels differ: {sents}"
                print(f"  Group {gi:>4} (x{len(idxs)}): IDs={ids}{sent_note}")
                print(f"             text: \"{preview}{'...' if len(str(key)) > 90 else ''}\"")

        exact_csv = out_dir / "exact_duplicate_groups.csv"
        pd.DataFrame(rows_out).to_csv(exact_csv, index=False)
        print(f"\nFull exact-duplicate report written to: {exact_csv}")

    # ---------------------------------------------------------------
    # 2) WHITESPACE-NORMALIZED duplicates (NOT exact, but same after
    #    stripping/collapsing whitespace). Reported separately.
    # ---------------------------------------------------------------
    def norm_ws(s: str) -> str:
        return re.sub(r"\s+", " ", str(s)).strip()

    ws_key = work[tf].map(norm_ws)
    ws_groups_all = find_duplicate_groups(work, ws_key)

    # keep only groups that are NOT already fully exact-duplicate groups,
    # i.e. groups where whitespace normalization merged >1 distinct exact text
    ws_only_groups = {}
    for key, idxs in ws_groups_all.items():
        distinct_exact = {str(work.at[i, tf]) for i in idxs}
        if len(distinct_exact) > 1:
            ws_only_groups[key] = idxs

    print("\n" + "-" * 70)
    print("SECONDARY CHECK -- whitespace-only differences (near-identical)")
    print("-" * 70)
    if not ws_only_groups:
        print("None. No pairs that become identical only after normalizing whitespace.")
    else:
        n_ws_groups = len(ws_only_groups)
        n_ws_rows = sum(len(v) for v in ws_only_groups.values())
        print(f"Groups identical after whitespace-normalization : {n_ws_groups}")
        print(f"Rows involved                                   : {n_ws_rows}")
        print("(These are NOT 100% identical, so by your rule they may be acceptable.)")
        rows_out = []
        for gi, (key, idxs) in enumerate(
            sorted(ws_only_groups.items(), key=lambda kv: (-len(kv[1]), kv[1][0])), start=1
        ):
            for row_idx in idxs:
                rows_out.append(
                    {
                        "group_id": gi,
                        "group_size": len(idxs),
                        idf: df.at[row_idx, idf] if idf in df.columns else "",
                        "row_index": row_idx,
                        "sentiment": df.at[row_idx, "sentiment"] if "sentiment" in df.columns else "",
                        "text": str(df.at[row_idx, tf]),
                    }
                )
            if gi <= 20:
                ids = [df.at[i, idf] for i in idxs] if idf in df.columns else ["n/a"] * len(idxs)
                print(f"  Group {gi:>4} (x{len(idxs)}): IDs={ids}")
        ws_csv = out_dir / "whitespace_duplicate_groups.csv"
        pd.DataFrame(rows_out).to_csv(ws_csv, index=False)
        print(f"\nFull whitespace-duplicate report written to: {ws_csv}")

    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    unique_exact = work[tf].astype(str).nunique()
    print(f"Total rows (non-empty text) : {len(work)}")
    print(f"Distinct text values        : {unique_exact}")
    print(f"Exact duplicate groups      : {n_exact_groups}")
    print(f"Redundant rows (exact)      : {n_exact_removable}")
    all_unique = n_exact_groups == 0
    print(f"\nAll messages 100% unique?   : {'YES' if all_unique else 'NO'}")
    print("=" * 70)


if __name__ == "__main__":
    main()

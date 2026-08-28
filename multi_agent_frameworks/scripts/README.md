# Scripts

Run the scripts in this directory from `multi_agent_frameworks/`.

## Build the workflow statistical comparison

`build_workflow_comparison.py` regenerates the final comparison of the completed
Luna single-agent and Framework 1–4 runs. It uses suicidal-intent sensitivity
as the primary endpoint, where all four suicidality phases are positive and
`Unsupportive` is negative. Uncertainty and workflow differences use 2,000
paired user-cluster bootstrap resamples by default.

From `multi_agent_frameworks/`:

```console
uv run python scripts/build_workflow_comparison.py
```

The command validates dataset IDs, labels, and user assignments before writing
`evaluation_results/workflow_comparison/workflow_comparison.json` and
`workflow_comparison.md`. Framework 3's archived per-row predictions were
pruned, so its exact binary decisions are reconstructed from its binary
misclassification-ID file. Its clustered four-phase CI is therefore reported
as unavailable rather than inferred from aggregate counts.

## Compare record-ID files

`compare_id_files.py` finds IDs that appear in at least one input file but not
in every input file. In set terms, the result is the union of the IDs minus
their intersection.

Pass two or more UTF-8 `.txt` files:

```console
uv run python scripts/compare_id_files.py path/to/first.txt path/to/second.txt path/to/third.txt
```

Each file must contain exactly one unique record ID per non-blank row. IDs are
kept as strings, so values with leading zeroes are preserved.

The command prints a JSON report containing:

- `non_matching_ids`: IDs that are not present in every input file.
- `missing_ids`: the non-matching IDs absent from each individual file.
- `union_count`, `intersection_count`, and `non_matching_count`: summary counts.

The command exits with an error for fewer than two inputs, repeated input
paths, non-`.txt` paths, missing or empty files, invalid UTF-8, blank or
malformed rows, and duplicate IDs within a file.

## Validate the balanced suicidality dataset

`validate_suicidality_dataset.py` checks one CSV for:

- non-empty `ID`, `text`, and `sentiment` values;
- unique IDs;
- unique normalized text, using CRLF-to-LF conversion and trimming;
- one of the five sentiment labels, matched case-insensitively;
- exactly 50% positive rows (`Indicator`, `Ideation`, `Behavior`, or
  `Attempt`) and 50% negative rows (`Unsupportive`).

From `multi_agent_frameworks/`, validate the default balanced dataset:

```console
uv run python scripts/validate_suicidality_dataset.py
```

Pass another CSV as the positional argument. To keep the full JSON report,
including every violating data-row number and row ID, use `--output`:

```console
uv run python scripts/validate_suicidality_dataset.py dataset/example.csv --output logs/dataset_validation/example.json
```

The exit code is `0` when every check passes, `1` when data-quality rules are
breached, and `2` for an unreadable or structurally invalid CSV. Duplicate
checks flag every row in each duplicate group. A balance violation is reported
as a dataset-level violation because there is no objective way to attribute it
to one particular row.

# Seeded 100-Sample Evaluation Cohorts

This directory contains two small, reproducible cohorts for evaluating the final-label and binary `suicidal_intent` contracts.

| File | Rows | Composition |
|---|---:|---|
| `dataset1_positive_balanced_100_seed42.csv` | 100 | 25 each of `Indicator`, `Ideation`, `Behavior`, and `Attempt`; all rows have `real = true`. |
| `negative_real_synthetic_balanced_100_seed42.csv` | 100 | 100 `Unsupportive` rows: 50 real and 50 synthetic. |

Both files use seed 42 and the schema:

```text
ID,users,text,sentiment,time,real
```

- `ID`: stable, globally non-overlapping sample identifier. Positive IDs use the original zero-based RSD15K row identity; negative IDs are retained from their source pools.
- `users`: original author identifier. This is not unique because one author may have multiple posts.
- `text`: model input.
- `sentiment`: ground-truth final label.
- `time`: original Unix timestamp (`0` where the synthetic source has no real timestamp).
- `real`: provenance flag (`true` for real data, `false` for synthetic data).

The `suicidal_intent` output is the dataset's positive-class grouping flag: all four risk labels map to `true`, while `Unsupportive` maps to `false`. In particular, `Indicator = true` represents membership in that four-label grouping and is not by itself a clinical claim that the author expresses suicidal intent.

Use rows from both cohorts for a binary evaluation containing positive and negative ground-truth support. Evaluating either cohort alone produces valid counts but cannot establish two-class macro performance.

The two much larger CSV files in this directory are source pools and remain ignored; they are not part of this committed evaluation fixture.

Regenerate the canonical fields without changing cohort membership by running `python scripts/regenerate_evaluation_cohorts.py` from `multi_agent_framework1` while the ignored source pools are available locally. Full-dataset comparisons should likewise use RSD15K with its original `ID`, `users`, `text`, `sentiment`, and `time` fields rather than a stripped Dataset1-derived export.

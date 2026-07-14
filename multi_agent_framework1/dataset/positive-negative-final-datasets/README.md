# Seeded 100-Sample Evaluation Cohorts

This directory contains two small, reproducible cohorts for evaluating the final-label and binary `suicidal_intent` contracts.

| File | Rows | Composition |
|---|---:|---|
| `dataset1_positive_balanced_100_seed42.csv` | 100 | 25 each of `Indicator`, `Ideation`, `Behavior`, and `Attempt`; all rows have `real = true`. |
| `negative_real_synthetic_balanced_100_seed42.csv` | 100 | 100 `Unsupportive` rows: 50 real and 50 synthetic. |

Both files use seed 42 and the schema:

```text
source_row,text,sentiment,real
```

- `source_row`: stable row identifier within the cohort.
- `text`: model input.
- `sentiment`: ground-truth final label.
- `real`: provenance flag (`true` for real data, `false` for synthetic data).

Use rows from both cohorts for a binary evaluation containing positive and negative ground-truth support. Evaluating either cohort alone produces valid counts but cannot establish two-class macro performance.

The two much larger CSV files in this directory are source pools and remain ignored; they are not part of this committed evaluation fixture.

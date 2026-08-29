# Framework 4 Secondary-Reviewer Ablation

This analysis quantifies the contribution of the secondary reviewer within the original full-dataset Framework 4 run.

The workflow contains 28,792 records; 28,791 have valid paired pre-review and post-review five-class predictions. 1 record(s) were excluded because a paired comparison could not be reconstructed.

## Reviewer activity

- Reviewer triggered: **1,184 / 28,791** analyzable records (4.11%)
- Final label changed: **199** cases (16.81% of reviewed cases)
- Wrong → correct: **121**
- Correct → wrong: **46**
- Wrong → different wrong: **32**
- Net change in correct classifications: **+75**

## Before/after performance

| Metric | Before reviewer | After reviewer | Difference | 95% user-cluster bootstrap CI |
|---|---:|---:|---:|---:|
| Five-class accuracy | 0.7517 | 0.7543 | +0.0026 | [0.0017, 0.0035] |
| Five-class Macro F1 | 0.5633 | 0.5715 | +0.0082 | [0.0063, 0.0100] |
| Positive-only Macro F1 | 0.4808 | 0.4909 | +0.0101 | [0.0079, 0.0124] |
| Binary sensitivity/recall | 0.8078 | 0.8135 | +0.0058 | [0.0045, 0.0071] |
| Binary specificity | 0.9840 | 0.9824 | -0.0017 | [-0.0024, -0.0010] |
| Binary precision | 0.9806 | 0.9788 | -0.0018 | [-0.0027, -0.0011] |
| Binary F1 | 0.8858 | 0.8885 | +0.0027 | [0.0019, 0.0036] |
| Indicator F1 | 0.1184 | 0.1536 | +0.0351 | [0.0282, 0.0423] |

## Generated publication assets

- `figures/figure1_before_after_selected_metrics.png`
- `figures/figure2_metric_differences_with_ci.png`
- `figures/figure3_reviewer_activity_summary.png`
- `tables/publication_summary_table.csv`
- `tables/publication_summary_table.md`

## Method

Uncertainty was quantified using a paired user-cluster bootstrap with 5,000 replicates (seed 42). Unique users were sampled with replacement, all posts belonging to each sampled user were included, and identical bootstrap samples were used for the pre-review and post-review predictions.

Complete reconstructed predictions, exclusions, transition matrices, per-class results, bootstrap distributions, input hashes, and software metadata are saved with the analysis.
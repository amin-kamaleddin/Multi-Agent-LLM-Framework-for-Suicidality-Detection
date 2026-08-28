# Workflow Statistical Comparison

## Analysis Contract

- Primary endpoint: **suicidal-intent sensitivity**, `TP / (TP + FN)`.
- Positive: Indicator, Ideation, Behavior, or Attempt. Negative: Unsupportive.
- Uncertainty: paired user-cluster percentile bootstrap with 2000 resamples (seed 20260820).
- Status: retrospective. The endpoint was designated before this final comparative analysis, but the model outputs already existed; this is not a prospective preregistration.
- Five-label aggregate accuracy and macro-F1 are not primary endpoints. The conditional four-positive-label macro-F1 is secondary.
- Per-post McNemar tests are omitted because posts from the same user are not independent.

A 95% CI for a paired difference is built from the central 95% of the user-bootstrap difference estimates. If the complete interval is above zero, the candidate has higher sensitivity; if it crosses zero, the comparison is inconclusive.

## Primary Endpoint

| System | Sensitivity (95% CI) | False negatives | Specificity (95% CI) |
|---|---:|---:|---:|
| Single agent | 79.67% (77.53% to 81.60%) | 2927 | 98.64% (98.44% to 98.83%) |
| Framework 1 | 80.39% (78.25% to 82.30%) | 2823 | 98.53% (98.33% to 98.73%) |
| Framework 2 | 78.71% (76.53% to 80.68%) | 3065 | 98.71% (98.53% to 98.90%) |
| Framework 3 | 80.38% (78.27% to 82.32%) | 2824 | 98.67% (98.48% to 98.85%) |
| Framework 4 | 81.35% (79.26% to 83.26%) | 2685 | 98.24% (98.02% to 98.46%) |

## Paired Sensitivity Comparisons

Every difference is candidate minus reference and is calculated inside each shared user resample; it is not obtained by subtracting two separate CIs.

| Reference | Candidate | Difference (percentage points) | 95% CI (percentage points) | Interpretation |
|---|---|---:|---:|---|
| Single agent | Framework 1 | +0.72 | +0.51 to +0.94 | candidate higher |
| Single agent | Framework 2 | -0.96 | -1.26 to -0.66 | reference higher |
| Single agent | Framework 3 | +0.72 | +0.51 to +0.92 | candidate higher |
| Single agent | Framework 4 | +1.68 | +1.41 to +1.95 | candidate higher |
| Framework 1 | Framework 2 | -1.68 | -2.02 to -1.38 | reference higher |
| Framework 1 | Framework 3 | -0.01 | -0.21 to +0.19 | inconclusive |
| Framework 1 | Framework 4 | +0.96 | +0.71 to +1.21 | candidate higher |
| Framework 2 | Framework 3 | +1.67 | +1.38 to +1.98 | candidate higher |
| Framework 2 | Framework 4 | +2.64 | +2.30 to +3.01 | candidate higher |
| Framework 3 | Framework 4 | +0.97 | +0.76 to +1.17 | candidate higher |

## Secondary Endpoints

PPV and NPV below describe the observed 50/50 benchmark and should not be interpreted as deployment predictive values when prevalence differs.
Positive-phase macro-F1 is conditional: it includes only records for which both the actual and predicted labels are one of the four positive phases.

| System | PPV (95% CI) | NPV (95% CI) | Conditional positive-phase macro-F1 (95% CI) |
|---|---:|---:|---:|
| Single agent | 98.32% (97.98% to 98.61%) | 82.91% (80.54% to 85.15%) | 48.12% (46.79% to 49.46%) |
| Framework 1 | 98.20% (97.86% to 98.50%) | 83.40% (81.06% to 85.65%) | 49.92% (48.59% to 51.26%) |
| Framework 2 | 98.39% (98.07% to 98.66%) | 82.26% (79.86% to 84.52%) | 53.14% (51.74% to 54.54%) |
| Framework 3 | 98.37% (98.06% to 98.65%) | 83.42% (81.08% to 85.66%) | 50.87% (49.44% to 52.23%) |
| Framework 4 | 97.88% (97.52% to 98.21%) | 84.04% (81.70% to 86.26%) | 52.93% (51.43% to 54.34%) |

## Data Provenance And Limitations

- **Single agent:** Exact per-record final-label predictions.
- **Framework 1:** Exact per-record final-label predictions.
- **Framework 2:** Exact per-record final-label predictions.
- **Framework 3:** Exact per-record final-label predictions.
- **Framework 4:** Exact per-record final-label predictions.
- Previously quoted counts of 426 and 390 false negatives use a different positive-class definition or different result artifact. They are incompatible with the current contract, under which Indicator is positive.
- Confidence intervals quantify sampling uncertainty within this benchmark; they do not establish performance on a different population, platform, or time period.

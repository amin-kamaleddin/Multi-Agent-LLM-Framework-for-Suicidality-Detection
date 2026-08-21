# Prespecified Statistical Analysis

## Scope

This analysis applies to completed `evaluate` run directories with a canonical `users` column. The bootstrap resamples users, rather than individual posts, with replacement and keeps all posts from a sampled user together. This accounts for within-user dependence in repeated posts.

## Primary endpoint

The primary endpoint is **suicidal-intent sensitivity**:

```text
TP / (TP + FN)
```

The positive class is `Indicator`, `Ideation`, `Behavior`, or `Attempt`; `Unsupportive` is negative. Sensitivity is prespecified because missed suicide-related content is safety-relevant and because it is invariant to the artificial positive/negative balance of an evaluation sample.

The report uses a two-sided 95% percentile CI from 2,000 user-cluster bootstrap resamples, with a recorded deterministic seed. The primary comparison between workflows is the paired difference in sensitivity, `candidate - reference`, using the same user resamples for both runs. A CI wholly above zero is reported as `candidate higher`, wholly below zero as `reference higher`, otherwise `inconclusive`. This report intentionally does not turn bootstrap sign probabilities into p-values.

## Secondary endpoints

- specificity;
- observed-sample PPV and NPV;
- prevalence-adjusted PPV and NPV; and
- positive-phase macro F1 for `Indicator`, `Ideation`, `Behavior`, and `Attempt`.

The four-phase metric follows the repository contract: it includes rows only when both ground truth and prediction are one of those four labels. Its purpose is phase discrimination after a positive decision; missed positives remain visible in the primary sensitivity endpoint.

## Prevalence adjustment

The observed PPV and NPV reflect the evaluated sample's class balance. They should not be treated as deployment values if prevalence differs. For a user-specified target prevalence `p`, each bootstrap sample computes:

```text
PPV(p) = sensitivity * p / (sensitivity * p + (1 - specificity) * (1 - p))
NPV(p) = specificity * (1 - p) / ((1 - sensitivity) * p + specificity * (1 - p))
```

These are transport estimates and assume that sensitivity and specificity transfer to the target population. The default report grid is 1%, 5%, 10%, 20%, and 50% positive prevalence. It is configurable with `--target-prevalences`.

## Use

Run `compile-results` on a completed canonical run:

```powershell
uv run python -m cli compile-results `
  --run-dir logs/runs/<reference-run> `
  --compare-run-dir logs/runs/<candidate-run> `
  --bootstrap-replicates 2000 `
  --bootstrap-seed 20260820 `
  --target-prevalences 0.01,0.05,0.10,0.20,0.50
```

The command writes `statistical_analysis.json`, optionally `paired_comparisons.json`, and adds the results to `compiled_results.csv` and `compiled_results.md`.

## Limitations

The method requires at least two non-null user clusters and exactly matching IDs, ground truth, and user assignments for a paired comparison. CIs quantify sampling uncertainty within this dataset; they do not prove transportability to a different platform, time period, or annotation process.

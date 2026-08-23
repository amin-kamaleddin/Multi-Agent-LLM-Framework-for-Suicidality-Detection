# Four-System Held-Out Test Comparison

All systems were evaluated on the same 4,319 canonical held-out posts.

## Primary Metrics

| System | 5-class Accuracy | 5-class Macro F1 | Positive-only Macro F1 | Binary Recall | Binary F1 |
|---|---:|---:|---:|---:|---:|
| Single-Agent Luna | 0.7458 | 0.5318 | 0.4406 | 0.8028 | 0.8847 |
| Framework 3 (Hierarchical, No Reviewer) | 0.7520 | 0.5575 | 0.4729 | 0.8093 | 0.8880 |
| Framework 4 (Hierarchical + Reviewer) | 0.7555 | 0.5652 | 0.4821 | 0.8194 | 0.8933 |
| Fine-Tuned MentalRoBERTa | 0.8437 | 0.7293 | 0.6741 | 0.9815 | 0.9797 |

## Per-Class F1

| Label | Single-Agent Luna | Framework 3 | Framework 4 | Fine-Tuned MentalRoBERTa |
|---|---:|---:|---:|---:|
| Unsupportive | 0.9042 | 0.9062 | 0.9095 | 0.9796 |
| Indicator | 0.0904 | 0.1107 | 0.1748 | 0.7326 |
| Ideation | 0.7509 | 0.7496 | 0.7513 | 0.7558 |
| Behavior | 0.3801 | 0.4620 | 0.4505 | 0.5959 |
| Attempt | 0.5333 | 0.5591 | 0.5397 | 0.5825 |

## Pairwise Bootstrap Comparisons

Differences are calculated as **System B minus System A**.

| Comparison | 5-class Macro F1 Δ | 95% CI | Positive-only Macro F1 Δ | 95% CI |
|---|---:|---:|---:|---:|
| framework3_minus_single_agent_luna | 0.0257 | [0.0120, 0.0391] | 0.0323 | [0.0157, 0.0491] |
| framework4_minus_framework3 | 0.0077 | [-0.0024, 0.0179] | 0.0091 | [-0.0032, 0.0219] |
| framework4_minus_single_agent_luna | 0.0334 | [0.0190, 0.0478] | 0.0414 | [0.0239, 0.0595] |
| mentalroberta_ft_minus_single_agent_luna | 0.1975 | [0.1744, 0.2214] | 0.2335 | [0.2040, 0.2620] |
| mentalroberta_ft_minus_framework3 | 0.1718 | [0.1489, 0.1951] | 0.2012 | [0.1715, 0.2295] |
| mentalroberta_ft_minus_framework4 | 0.1641 | [0.1410, 0.1870] | 0.1921 | [0.1624, 0.2201] |

## Pairwise McNemar Tests

| Comparison | A-only correct | B-only correct | Raw p | Holm-adjusted p |
|---|---:|---:|---:|---:|
| framework3_minus_single_agent_luna | 53 | 80 | 0.0238011 | 0.0476022 |
| framework4_minus_framework3 | 38 | 53 | 0.141793 | 0.141793 |
| framework4_minus_single_agent_luna | 60 | 102 | 0.00120722 | 0.00362166 |
| mentalroberta_ft_minus_single_agent_luna | 304 | 727 | 1.40814e-40 | 8.44882e-40 |
| mentalroberta_ft_minus_framework3 | 307 | 703 | 2.22703e-36 | 1.11351e-35 |
| mentalroberta_ft_minus_framework4 | 305 | 686 | 2.24703e-34 | 8.98811e-34 |
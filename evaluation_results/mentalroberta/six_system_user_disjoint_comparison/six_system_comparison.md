# Six-System User-Disjoint Test Comparison

All six systems were evaluated on the exact same canonical user-disjoint test partition containing 4,317 posts from 2,036 unique users.

Primary uncertainty estimates use a paired user-cluster bootstrap with 5,000 replicates.

Exact post-level McNemar tests are reported as a secondary sensitivity analysis and are Holm-adjusted across all 15 pairwise comparisons.

## Primary Metrics

| System | 5-class Accuracy | 5-class Macro F1 | 95% CI | Positive-only Macro F1 | 95% CI | Binary Recall | 95% CI | Binary F1 | 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Single-Agent Luna | 0.7487 | 0.5301 | [0.5017, 0.5586] | 0.4404 | [0.4072, 0.4743] | 0.7895 | [0.7463, 0.8305] | 0.8756 | [0.8475, 0.9007] |
| Framework 1 (Parallel, No Evidence) | 0.7517 | 0.5504 | [0.5217, 0.5791] | 0.4669 | [0.4322, 0.5005] | 0.7974 | [0.7557, 0.8373] | 0.8780 | [0.8515, 0.9019] |
| Framework 2 (Parallel + Evidence) | 0.7514 | 0.5759 | [0.5481, 0.6024] | 0.4983 | [0.4640, 0.5294] | 0.7803 | [0.7386, 0.8217] | 0.8716 | [0.8444, 0.8969] |
| Framework 3 (Hierarchical, No Reviewer) | 0.7577 | 0.5641 | [0.5340, 0.5943] | 0.4826 | [0.4460, 0.5183] | 0.8011 | [0.7580, 0.8420] | 0.8825 | [0.8554, 0.9069] |
| Framework 4 (Hierarchical + Reviewer) | 0.7586 | 0.5708 | [0.5401, 0.6004] | 0.4902 | [0.4528, 0.5253] | 0.8113 | [0.7679, 0.8520] | 0.8876 | [0.8604, 0.9116] |
| Fine-Tuned MentalRoBERTa | 0.8439 | 0.7126 | [0.6781, 0.7416] | 0.6523 | [0.6084, 0.6893] | 0.9875 | [0.9813, 0.9928] | 0.9820 | [0.9759, 0.9869] |

## Per-Class F1

| Label | Single-Agent Luna | Framework 1 (Parallel, No Evidence) | Framework 2 (Parallel + Evidence) | Framework 3 (Hierarchical, No Reviewer) | Framework 4 (Hierarchical + Reviewer) | Fine-Tuned MentalRoBERTa |
|---|---:|---:|---:|---:|---:|---:|
| Unsupportive | 0.8980 | 0.8986 | 0.8961 | 0.9025 | 0.9056 | 0.9818 |
| Indicator | 0.1222 | 0.1176 | 0.1442 | 0.1395 | 0.1998 | 0.7133 |
| Ideation | 0.7586 | 0.7597 | 0.7434 | 0.7609 | 0.7559 | 0.7566 |
| Behavior | 0.3232 | 0.4266 | 0.5473 | 0.4528 | 0.4358 | 0.5903 |
| Attempt | 0.5484 | 0.5494 | 0.5487 | 0.5650 | 0.5571 | 0.5209 |

## Pairwise User-Cluster Bootstrap

Differences are defined as **System B minus System A**.

| Comparison | Metric | Observed Δ | 95% CI | CI excludes zero |
|---|---|---:|---:|---:|
| framework1_minus_single_agent_luna | five_class_macro_f1 | 0.0203 | [0.0058, 0.0350] | True |
| framework1_minus_single_agent_luna | positive_only_macro_f1 | 0.0265 | [0.0083, 0.0449] | True |
| framework2_minus_single_agent_luna | five_class_macro_f1 | 0.0458 | [0.0248, 0.0650] | True |
| framework2_minus_single_agent_luna | positive_only_macro_f1 | 0.0578 | [0.0312, 0.0819] | True |
| framework3_minus_single_agent_luna | five_class_macro_f1 | 0.0341 | [0.0191, 0.0490] | True |
| framework3_minus_single_agent_luna | positive_only_macro_f1 | 0.0422 | [0.0236, 0.0606] | True |
| framework4_minus_single_agent_luna | five_class_macro_f1 | 0.0408 | [0.0252, 0.0554] | True |
| framework4_minus_single_agent_luna | positive_only_macro_f1 | 0.0498 | [0.0304, 0.0681] | True |
| mentalroberta_ft_minus_single_agent_luna | five_class_macro_f1 | 0.1825 | [0.1442, 0.2166] | True |
| mentalroberta_ft_minus_single_agent_luna | positive_only_macro_f1 | 0.2119 | [0.1665, 0.2526] | True |
| framework2_minus_framework1 | five_class_macro_f1 | 0.0255 | [0.0073, 0.0413] | True |
| framework2_minus_framework1 | positive_only_macro_f1 | 0.0314 | [0.0086, 0.0511] | True |
| framework3_minus_framework1 | five_class_macro_f1 | 0.0137 | [-0.0005, 0.0274] | False |
| framework3_minus_framework1 | positive_only_macro_f1 | 0.0157 | [-0.0022, 0.0328] | False |
| framework4_minus_framework1 | five_class_macro_f1 | 0.0204 | [0.0061, 0.0345] | True |
| framework4_minus_framework1 | positive_only_macro_f1 | 0.0233 | [0.0055, 0.0410] | True |
| mentalroberta_ft_minus_framework1 | five_class_macro_f1 | 0.1622 | [0.1254, 0.1952] | True |
| mentalroberta_ft_minus_framework1 | positive_only_macro_f1 | 0.1854 | [0.1410, 0.2248] | True |
| framework3_minus_framework2 | five_class_macro_f1 | -0.0118 | [-0.0298, 0.0070] | False |
| framework3_minus_framework2 | positive_only_macro_f1 | -0.0157 | [-0.0380, 0.0079] | False |
| framework4_minus_framework2 | five_class_macro_f1 | -0.0051 | [-0.0226, 0.0133] | False |
| framework4_minus_framework2 | positive_only_macro_f1 | -0.0080 | [-0.0299, 0.0152] | False |
| mentalroberta_ft_minus_framework2 | five_class_macro_f1 | 0.1367 | [0.1018, 0.1701] | True |
| mentalroberta_ft_minus_framework2 | positive_only_macro_f1 | 0.1540 | [0.1117, 0.1944] | True |
| framework4_minus_framework3 | five_class_macro_f1 | 0.0067 | [-0.0023, 0.0147] | False |
| framework4_minus_framework3 | positive_only_macro_f1 | 0.0076 | [-0.0034, 0.0175] | False |
| mentalroberta_ft_minus_framework3 | five_class_macro_f1 | 0.1485 | [0.1102, 0.1816] | True |
| mentalroberta_ft_minus_framework3 | positive_only_macro_f1 | 0.1697 | [0.1246, 0.2105] | True |
| mentalroberta_ft_minus_framework4 | five_class_macro_f1 | 0.1418 | [0.1038, 0.1764] | True |
| mentalroberta_ft_minus_framework4 | positive_only_macro_f1 | 0.1621 | [0.1172, 0.2032] | True |

## Exact McNemar Tests — Secondary

| Comparison | A-only correct | B-only correct | Raw p | Holm-adjusted p |
|---|---:|---:|---:|---:|
| framework1_minus_single_agent_luna | 65 | 78 | 0.315625 | 1 |
| framework2_minus_single_agent_luna | 168 | 180 | 0.55548 | 1 |
| framework3_minus_single_agent_luna | 52 | 91 | 0.00139827 | 0.0139827 |
| framework4_minus_single_agent_luna | 69 | 112 | 0.00172057 | 0.0154852 |
| mentalroberta_ft_minus_single_agent_luna | 268 | 679 | 8.41309e-42 | 1.26196e-40 |
| framework2_minus_framework1 | 147 | 146 | 1 | 1 |
| framework3_minus_framework1 | 46 | 72 | 0.02097 | 0.16776 |
| framework4_minus_framework1 | 65 | 95 | 0.0215738 | 0.16776 |
| mentalroberta_ft_minus_framework1 | 272 | 670 | 1.97721e-39 | 2.57038e-38 |
| framework3_minus_framework2 | 128 | 155 | 0.122063 | 0.610316 |
| framework4_minus_framework2 | 139 | 170 | 0.0877251 | 0.52635 |
| mentalroberta_ft_minus_framework2 | 260 | 659 | 1.21579e-40 | 1.70211e-39 |
| framework4_minus_framework3 | 46 | 50 | 0.759649 | 1 |
| mentalroberta_ft_minus_framework3 | 276 | 648 | 3.43077e-35 | 4.11692e-34 |
| mentalroberta_ft_minus_framework4 | 274 | 642 | 9.36899e-35 | 1.03059e-33 |

## Reproducibility

- Bootstrap seed: `42`
- Bootstrap replicates: `5000`
- Canonical test SHA-256: `1786dbe96e3898bc3ef336d0d021253ed0604ee8d73c1ef2dea409ce2dfe3f64`
- Complete prediction-file hashes and software versions are recorded in `provenance/analysis_provenance.json`.
- Available run configuration, metric, manifest, audit, and training-summary artifacts are hashed in `provenance/input_provenance.csv`.
# RSD-15K Dataset Summary

## Reference

Zheng, Shouwen, Yingzhi Tao, and Taiqi Zhou. "RSD-15K: A Large-Scale User-Level Annotated Dataset for Suicide Risk Detection on Social Media." 2025 IEEE 41st International Conference on Data Engineering Workshops (ICDEW), Hong Kong, Hong Kong, 2025, pp. 190-196. DOI: 10.1109/ICDEW67478.2025.00029.

- Paper: https://arxiv.org/abs/2507.11559
- IEEE page: https://ieeexplore.ieee.org/abstract/document/11108158
- Dataset repository and access instructions: https://github.com/Suicide-DataSet/RSD_15K
- Local project report citation: reference `[19]` in `sources/project-report-suicidal-screening-final-draft.pdf`

## What The Dataset Contains

RSD-15K is a Reddit suicide-risk dataset built from posts in `r/SuicideWatch`. The paper describes it as a large-scale user-level dataset designed for suicide risk detection on social media. It retains posting-time sequence information, which makes it usable for static post classification and for studying risk evolution across a user's timeline.

The paper reports:

- 1,265 users
- 14,613 valid posts after annotation
- timestamps/posting order retained
- four suicide-risk labels: `Indicator`, `Ideation`, `Behavior`, `Attempt`
- data access through the dataset repository's application/download process because of sensitivity and non-commercial research-use terms

The local project report describes the benchmark dataset as `Dataset1`, a cleaned version of RSD-15K with 9,800 entries. That cleaned subset removes duplicate or near-duplicate posts, posts shorter than 50 characters, and users with more than 50 entries.

## Label Meanings

Use these labels exactly in code and evaluation outputs:

- `Indicator`: The post does not indicate the author's own suicidal risk. It may discuss another person's suicidality, general suicide-related topics, or explicit denial of the author's own suicidal intent.
- `Ideation`: The author expresses suicidal thoughts or desire to die, but does not describe concrete self-harm or suicide action.
- `Behavior`: The author describes preparatory or self-harm behavior beyond verbal thoughts, such as gathering means, writing notes, or self-inflicted harm without an explicit fatal attempt.
- `Attempt`: The post mentions a suicide attempt, current or past.

Note: the paper and report sometimes use the spelling `Behaviour`; the repository's agent and evaluation code use `Behavior`.

## Local Dataset Facts Already Documented

The repository currently documents the original local `rsd_15k.csv` statistics in `src/example/suicide_detection/data/README.md`:

- total posts: 14,613
- unique users: 1,265
- average posts per user: 11.55
- average word count: 94.88
- median word count: 55.0

Label distribution:

| Label | Count | Percentage |
| --- | ---: | ---: |
| Ideation | 7,133 | 48.81% |
| Indicator | 4,615 | 31.58% |
| Behavior | 2,056 | 14.07% |
| Attempt | 809 | 5.54% |

## Annotation And Quality Notes

The RSD-15K paper states that annotation used a detailed four-level risk standard adapted from C-SSRS. It describes three annotators and three supervisors with psychology-domain knowledge and English fluency, annotator training, uncertainty reporting for ambiguous samples, daily review, and Fleiss' Kappa on a 30% jointly labeled subset.

Important implication for this project: the label space is clinically motivated but still ambiguous at class boundaries. Existing project results already show repeated confusion between `Indicator` and `Ideation`, and weak performance on `Behavior`.

## Current Repository State

The code expects the main CSV at:

`src/example/suicide_detection/data/rsd_15k.csv`

As of this documentation pass, no `.csv` files are present in the checkout. The data directory contains helper scripts and README documentation, but not the benchmark CSV itself. Before any new benchmark can be trusted, the dataset must be restored or downloaded through the documented access path.

## First Verification Tasks

1. Obtain the intended benchmark CSV and place it at `src/example/suicide_detection/data/rsd_15k.csv`, or pass its location with `--data-path`.
2. Verify columns: `ID`, `users`, `text`, `sentiment`, `time`.
3. Confirm row count and label distribution against the table above.
4. Confirm whether the active benchmark should use the full 14,613-post dataset or the 9,800-entry cleaned `Dataset1` described in the report.
5. Use `ID` as the sample identifier, not `users`, because a single user can have multiple posts.
6. Only after these checks, run small smoke evaluations before any full model comparison.

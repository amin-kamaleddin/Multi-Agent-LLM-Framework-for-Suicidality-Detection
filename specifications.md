# Specifications: Multi-Agent Suicidality Detection

## Purpose

This repository is being shaped into a reproducible benchmark for suicidality detection in text. The current research goal is a three-way performance comparison:

1. rule-based keyword classification as a low-complexity baseline
2. single-agent LLM classification as the existing benchmark baseline
3. multi-agent classification with specialized agents and a final judge/host decision

The immediate starting step is not more model work. It is to find, restore, and understand the benchmark dataset so every later result is grounded in the same data contract.

## Current Source Materials

- `sources/project-report-suicidal-screening-final-draft.pdf`: current project report with background, dataset description, model results, limitations, and future work.
- `sources/research-ideas-and-next-steps.md`: research critique and recommended next steps for strengthening the paper.
- `sources/rsd15k-dataset-summary.md`: dataset-specific summary created from the project report, local data docs, and the RSD-15K paper.

## Current Architecture

The codebase has two main layers.

### Core Evaluation Framework

Location: `src/EvalRing`

Responsibilities:

- load and normalize datasets into `DataSample` records
- run agents over samples
- compute classification metrics
- persist per-sample and run-level artifacts
- retry or resume failed evaluations

Important modules:

- `src/EvalRing/dataset`: dataset loaders such as `CSVDataset` and `DataFrameDataset`
- `src/EvalRing/agent`: `MockAgent`, `RuleBasedAgent`, `OpenAIAgent`, classification utilities, and multi-role orchestration
- `src/EvalRing/evaluator`: classification evaluation, metrics, and result objects
- `src/EvalRing/evaluator/llm_judge`: rubric and LLM-as-judge evaluation utilities

### Suicide Detection Application

Location: `src/example/suicide_detection`

Responsibilities:

- define the suicidality label space
- prompt LLMs for risk classification
- run RSD-15K benchmarks
- support multiple classification modes
- write run artifacts under `_EvalRing`

Important files:

- `llm_agent.py`: task-specific LLM agent and label parsing
- `evaluate_rsd15k_main.py`: main single-run benchmark entry point
- `evaluate_rsd15k_llm_judge.py`: judge-oriented evaluation entry point
- `run_model_suite.py`: sequential multi-model benchmark runner
- `data/README.md`: documented dataset statistics
- `ground_truth/DATA_FLOW.md`: notes on dataset fields and `ID` usage

Current supported agent modes:

- `single-class`: one label returned directly by the model
- `multi-class-chance`: model returns class probabilities
- `base-vs-rest-binary`: binary comparisons are aggregated into one class distribution
- `multi-agent-host`: host model coordinates specialized role agents
- `per-class-score`: listed as a supported mode in CLI/docs

## Existing Work

The project report already benchmarks single-agent LLMs on the suicidality task. The central findings are:

- model performance varies meaningfully across the four labels
- `Ideation` is generally easier for models
- `Behavior` is consistently difficult
- `Indicator` and `Ideation` are often confused
- severe-risk cases can be under-classified
- the current four-way label space may be too flat for reliable zero-shot classification

The strongest current research direction is to move from "which LLM is best" toward "which risk formulation works." The proposed multi-agent system should test whether decomposition into specialized risk agents improves classification accuracy, macro F1, and high-risk recall.

## Dataset Contract

Primary dataset: RSD-15K.

Reference:

Zheng, Shouwen, Yingzhi Tao, and Taiqi Zhou. "RSD-15K: A Large-Scale User-Level Annotated Dataset for Suicide Risk Detection on Social Media." 2025 IEEE 41st International Conference on Data Engineering Workshops (ICDEW), Hong Kong, Hong Kong, 2025, pp. 190-196. DOI: 10.1109/ICDEW67478.2025.00029.

Public reference links:

- https://arxiv.org/abs/2507.11559
- https://github.com/Suicide-DataSet/RSD_15K

Expected local path:

`src/example/suicide_detection/data/rsd_15k.csv`

Expected columns:

- `ID`: unique per-row identifier
- `users`: user identifier, not unique
- `text`: post content
- `sentiment`: ground-truth label
- `time`: Unix timestamp

Expected labels:

- `Indicator`
- `Ideation`
- `Behavior`
- `Attempt`

Important naming note: the paper/report may use `Behaviour`, but the code uses `Behavior`. Evaluation outputs should preserve the code label unless a deliberate migration is made.

Known documented statistics for the original local dataset:

- 14,613 posts
- 1,265 users
- label distribution: 7,133 `Ideation`, 4,615 `Indicator`, 2,056 `Behavior`, 809 `Attempt`

The project report also describes a cleaned 9,800-entry `Dataset1`. Before running new benchmarks, decide whether the benchmark target is the original 14,613-post file or the cleaned 9,800-entry file.

## Current Data Blocker

The repository currently references `rsd_15k.csv`, but no `.csv` files are present in the checkout. The data directory contains helper scripts and documentation only.

This blocks trustworthy testing and benchmarking. Do not interpret model results until the exact dataset file is restored and verified.

## Starting Step: Dataset Grounding

Success criteria for the first milestone:

1. The intended benchmark CSV exists locally.
2. Its schema matches the dataset contract above.
3. Row counts and label counts are recorded.
4. The active benchmark split is named explicitly: original RSD-15K, cleaned 9,800-entry Dataset1, synthetic dataset, or another variant.
5. The runner uses `ID` as `id_field`, not `users`.
6. A tiny smoke run succeeds before full benchmarking.

Recommended checks after the CSV is restored:

```powershell
python src/example/suicide_detection/ground_truth/explore_rsd15k.py
python src/example/suicide_detection/evaluate_rsd15k_main.py --n-samples 2 --max-workers 2 --agent-mode single-class
```

If the restored CSV uses a non-default path, pass it with:

```powershell
python src/example/suicide_detection/evaluate_rsd15k_main.py --data-path path/to/rsd_15k.csv --n-samples 2 --max-workers 2
```

## Next Steps

1. Restore or download the RSD-15K CSV and confirm whether the project is benchmarking the original dataset or the cleaned 9,800-entry subset.
2. Produce a small dataset audit file with row count, label distribution, duplicate `ID` check, duplicate `users` check, missing text count, and timestamp sanity checks.
3. Reconcile dataset naming across docs and scripts so `rsd_15k.csv`, cleaned Dataset1, and synthetic data are not conflated.
4. Implement or configure the rule-based baseline with explicit keyword rules and expected low-performance framing.
5. Rerun or preserve the single-agent LLM baseline as the middle comparison point.
6. Align the multi-agent design with the research hypothesis: one specialized agent per risk paradigm plus a final judge/host.
7. Add risk-sensitive reporting beyond macro F1, especially high-risk recall and a collapsed urgent-action view for `Behavior` plus `Attempt`.
8. Add minimal tests around label normalization and output parsing before expanding experiments.

## Non-Goals For The Immediate Pass

- adding more LLMs
- rewriting the evaluation framework
- changing labels before the dataset audit is complete
- treating synthetic data results as evidence until cross-model or human validation is added
- building a UI/dashboard

## Open Questions

- Which exact file was used for the reported full single-agent runs: full 14,613-post RSD-15K or cleaned 9,800-entry Dataset1?
- Where is the cleaned 9,800-entry Dataset1 stored, and what filename should be canonical?
- Should `Indicator` remain the zero/self-risk-negative class, or should a true neutral/no-risk class be added before the next paper version?
- Should the multi-agent system classify in the current flat four-way label space, or test a hierarchy: self-risk versus no self-risk, ideation versus action, attempt versus non-attempt?

# Suicide Detection Research

This repository benchmarks approaches to suicidality risk detection in text on the RSD-15K dataset. It compares three tiers of classifier complexity:

1. **Rule-based keyword classification** — a low-complexity baseline.
2. **Single-agent LLM classification** — one model call (or a small set of prompting strategies) per sample.
3. **Multi-agent LLM classification** — specialized per-label agents feeding a final judge/host decision.

See [specifications.md](specifications.md) for the current research goals, dataset contract, and open questions.

## Repository layout

The code is split into two independent, separately-installable pipelines, plus supporting research assets.

### `multi_agent_frameworks/`

A standalone `uv` project implementing the rule-based baseline, a single-agent workflow, and four LangGraph-based multi-agent architectures (parallel specialists, parallel evidence collectors, and two hierarchical/reviewer designs), all run through one shared, architecture-neutral evaluator and CLI.

Start with [multi_agent_frameworks/README.md](multi_agent_frameworks/README.md).

### `single_agent_frameworks/`

Two parts:

- `single_agent_frameworks/single_agent_frameworks/`: a general-purpose, domain-agnostic agent evaluation framework (dataset loaders, agent base classes, evaluators, LLM-as-a-judge utilities). It has no suicide-detection-specific code and is intended to be reusable and public-domainable on its own.
- `single_agent_frameworks/example/`: applications built on that framework, most notably `suicide_detection/`, which runs single-agent RSD-15K benchmarks (single-class, multi-class-chance, base-vs-rest-binary, per-class-score, and multi-agent-host modes).

Start with [single_agent_frameworks/README.md](single_agent_frameworks/README.md).

### Other top-level folders

| Path | Contents |
|---|---|
| `docs/` | Design specs, usage guides, and multi-agent architecture diagrams. Some documents predate the current `single_agent_frameworks/` / `multi_agent_frameworks/` split — see each document's own scope notes. |
| `evaluation_results/` | Retained benchmark run outputs (reports, logs, and metrics) from prior evaluation passes. |
| `finetune_mentalroberta/` | Scripts to fine-tune and evaluate a MentalRoBERTa classifier baseline and compare it against the agent-based systems. |
| `negative_label_creation/` | Source material and scripts used to build the negative (non-suicidal) label dataset. |
| `tools/` | Standalone utility scripts (for example, audit and API-key-check helpers). |
| `tests/` | Repository-root test package. |

## Getting started

Each pipeline manages its own dependencies and environment:

```bash
# Multi-agent frameworks (standalone uv project)
cd multi_agent_frameworks
uv sync
uv run python -m cli list-workflows

# Single-agent frameworks
cd single_agent_frameworks
python example/suicide_detection/evaluate_rsd15k_main.py --n-samples 2 --max-workers 2 --agent-mode single-class
```

See the README in each folder for full setup and usage instructions.


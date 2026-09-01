# Single-Agent Frameworks

This folder contains two things:

1. **`single_agent_frameworks/`** (the nested package) — a general-purpose, domain-agnostic framework for running and evaluating single agents against a dataset. It knows nothing about suicide detection specifically: it is a reusable evaluation harness (datasets, agents, evaluators, metrics, LLM-as-a-judge) that any classification task can be built on top of.
2. **`example/`** — concrete applications built on that framework. The primary one is `example/suicide_detection/`, which benchmarks LLMs on the RSD-15K suicide-risk dataset.

> **Legacy name:** the `single_agent_frameworks/single_agent_frameworks` general-usage package was initially developed as a standalone project called **EvalRing**, a (now depublished) open-source generalized agent evaluation framework from one of the authors, during the first weeks of this research project; after the team agreed upon developing a new framework infrastructure exclusive to this project, the legacy code has remained only in this folder, renamed throughout, and is no longer maintained or available under that name.

## Package: `single_agent_frameworks/single_agent_frameworks`

A unified interface for evaluating AI agents with standardized metrics, datasets, and evaluation protocols.

Core components:

- **Datasets** (`single_agent_frameworks/dataset`): `BaseDataset`, `DataSample`, `JSONDataset`, `CSVDataset`, `DataFrameDataset` — load and normalize rows from JSON, CSV, or a pandas `DataFrame` into a common `DataSample` record.
- **Agents** (`single_agent_frameworks/agent`): `BaseAgent`, `AgentResponse`, `MockAgent`, `RuleBasedAgent`, `OpenAIAgent`, plus reusable classification helpers (`resolve_classification_prediction`, `normalize_probability_distribution`, `aggregate_base_vs_rest_probabilities`) and a multi-role host orchestrator.
- **Evaluators** (`single_agent_frameworks/evaluator`): `BaseEvaluator`, `EvaluationResult`, `EvaluationMetrics`, `ClassificationEvaluator` — run an agent over a dataset, compute metrics, and persist results.
- **LLM-as-a-Judge** (`single_agent_frameworks/evaluator/llm_judge`): `Rubric`, `RubricLevel`, `ScoringCriteria`, `JudgeVerdict`, `JudgeTemplate`, `LLMJudge`, `OpenAIJudge`, `LLMJudgeEvaluator` — rubric-based scoring where one LLM judges another's output.
- **Utils** (`single_agent_frameworks/utils`): a global response/result cache, a sequential multi-model suite runner, and visualization generation for suite reports.

### Basic usage

```python
from single_agent_frameworks import Evaluator, ClassificationMetrics
from single_agent_frameworks.agents import create_agent
from single_agent_frameworks.datasets import load_dataset

dataset = load_dataset("path/to/dataset.json")
agent = create_agent("rule_based", name="my_agent", version="1.0")

evaluator = Evaluator(output_dir="results")
result = evaluator.evaluate(agent, dataset, ClassificationMetrics(), "my_evaluation")

print(f"Accuracy: {result.metrics['accuracy']:.4f}")
print(f"F1 Score: {result.metrics['f1_score']:.4f}")
```

## Application: `example/suicide_detection`

Benchmarks LLMs on RSD-15K using the framework above. Supported agent modes:

- `single-class`: the model returns one label directly.
- `multi-class-chance`: the model returns per-class probabilities.
- `base-vs-rest-binary`: one declared base class is compared pairwise against every other class.
- `per-class-score`: the model scores every class independently on a rubric.
- `multi-agent-host`: a host model coordinates specialized sub-agents to reach a consensus label.

Key files:

- `llm_agent.py`: the suicide-detection-specific LLM agent and label parsing.
- `evaluate_rsd15k_main.py`: main single-run benchmark entry point.
- `evaluate_rsd15k_llm_judge.py`: judge-oriented evaluation entry point.
- `run_model_suite.py`: sequential multi-model benchmark runner.
- `data_translator.py`: splits the dataset into single-turn (`*_simple.csv`) and multi-turn (`*_multi_round.csv`) subsets.

See [`docs/USAGE.md`](../docs/USAGE.md) and [`docs/DESIGN_SPEC.md`](../docs/DESIGN_SPEC.md) for the full operational reference, and [`example/suicide_detection/README.md`](example/suicide_detection/README.md) for a complete end-to-end usage guide.

## Setup and quick start

From the repository root:

```bash
cd single_agent_frameworks
pip install -r ../requirements.txt   # or your project's usual dependency install
```

Set at least one API key in a `.env` file at the repository root (or export it in your shell):

- `OPEN_ROUTER_KEY` (preferred — the runner defaults to OpenRouter when set)
- `OPENAI_API_KEY` (fallback)

Run a smoke test:

```bash
python example/suicide_detection/evaluate_rsd15k_main.py --n-samples 2 --max-workers 2 --agent-mode single-class
```

Each run writes a timestamped folder under `example/suicide_detection/_runs/` containing `result.md`, `all_cases.csv`, `all_cases.txt`, `failed_cases.txt`, and `Meta.json`.


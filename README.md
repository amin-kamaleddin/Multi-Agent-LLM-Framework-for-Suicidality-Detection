# EvalRing

EvalRing is a unified evaluation framework for agents across LLMs and agent versions. Its goal is to provide a standardized interface for evaluating AI agents with comprehensive metrics and reproducible results.

Data package: 
https://drive.google.com/drive/folders/1aVvpHiBbPeaj9ZW1qXh2Jrv8pJr-YWr4?usp=sharing

## Features

- **Unified Interface**: Single framework to evaluate different types of AI agents
- **Comprehensive Metrics**: Built-in support for classification, regression, and custom metrics
- **Multiple Data Formats**: Support for JSON, CSV, and pandas DataFrames
- **Extensible Architecture**: Easy to add new agents, metrics, and datasets
- **CLI Tool**: Command-line interface for running evaluations
- **Result Management**: Automatic saving and tracking of evaluation results

## Installation

### From Source

```bash
git clone https://github.com/evalring/evalring.git
cd evalring
pip install -e .
```

### Development Installation

```bash
git clone https://github.com/evalring/evalring.git
cd evalring
pip install -e ".[dev]"
```

## Quick Start

### Basic Usage

```python
from EvalRing import Evaluator, ClassificationMetrics
from EvalRing.agents import MockAgent
from EvalRing.datasets import load_dataset

# Load your dataset
dataset = load_dataset("path/to/your/dataset.json")

# Create an agent (or use your own)
agent = MockAgent("my_agent")

# Set up metrics calculator
metrics_calc = ClassificationMetrics()

# Run evaluation
evaluator = Evaluator(output_dir="results")
result = evaluator.evaluate(agent, dataset, metrics_calc, "my_evaluation")

print(f"Accuracy: {result.metrics['accuracy']:.4f}")
print(f"F1 Score: {result.metrics['f1_score']:.4f}")
```

### Command Line Interface

```bash
# Get dataset information
evalring info --dataset data/my_dataset.json

# Run evaluation
evalring evaluate \
    --dataset data/my_dataset.json \
    --agent configs/my_agent.json \
    --metrics classification \
    --output results/ \
    --task-name "sentiment_analysis"

# Validate agent configuration
evalring validate --config configs/my_agent.json
```

## Core Components

### Agents

EvalRing provides several built-in agent types:

- **AgentTemplate**
- **MockAgent**: Simple random agent for testing, mocking a llm

```python
from EvalRing.agents import create_agent

# Create different types of agents
mock_agent = create_agent("mock", name="test_agent", version="1.0")
rule_agent = create_agent("rule_based", name="sentiment_agent", version="1.0")
ensemble_agent = create_agent("ensemble", name="combo_agent", version="1.0")
```

### Datasets

Support for multiple data formats:

```python
from EvalRing.datasets import JSONDataset, CSVDataset, DataFrameDataset

# Load JSON dataset
json_dataset = JSONDataset("data.json", text_field="text", label_field="label")

# Load CSV dataset
csv_dataset = CSVDataset("data.csv", text_field="review", label_field="sentiment")

# Use pandas DataFrame
import pandas as pd
df = pd.read_csv("data.csv")
df_dataset = DataFrameDataset(df, text_field="text", label_field="target")
```

## Example: LLM Performance in Suicide Detection

Here's a complete example of evaluating an LLM on suicide detection:

```python
from EvalRing import Evaluator, ClassificationMetrics
from EvalRing.datasets import load_dataset
from EvalRing.agents import create_agent

# Load the suicide detection dataset
dataset = load_dataset("examples/suicide_detection/data/dataset.json")

# Create a rule-based agent for demonstration
agent = create_agent(
    "rule_based",
    name="suicide_detection_agent",
    version="1.0",
    positive_keywords=["suicide", "kill", "end", "die", "death"],
    negative_keywords=["happy", "good", "fine", "okay", "well"]
)

# Set up evaluation
evaluator = Evaluator(agent, dataset, output_dir="suicide_detection_results")

# Run evaluation
result = evaluator.evaluate(
    version="1.0",
    task_name="suicide_detection_evaluation"
)

# Display results
print(f"Evaluation completed in {result.duration:.2f} seconds")
print(f"Accuracy: {result.metrics['accuracy']:.4f}")
print(f"Precision: {result.metrics['precision']:.4f}")
print(f"Recall: {result.metrics['recall']:.4f}")
print(f"F1 Score: {result.metrics['f1_score']:.4f}")
```

## Project Structure

```
EvalRing/
├── src/
│   └── EvalRing/
│       ├── __init__.py          # Package initialization
│       ├── core.py              # Core evaluation framework
│       ├── agents.py            # Agent implementations
│       ├── datasets.py          # Dataset loading utilities
│       ├── metrics.py           # Metrics calculation
│       └── cli.py               # Command-line interface
├── examples/                    # Example implementations
│   └── suicide_detection/       # Suicide detection example
├── tests/                       # Unit tests
├── setup.py                     # Package setup
├── pyproject.toml              # Modern Python packaging
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

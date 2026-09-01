# Suicide Detection Evaluation Pipeline

This repository contains the complete pipeline for running and evaluating suicide detection models using the `single_agent_frameworks` framework. Below are the detailed steps for replicating this research and executing the end-to-end evaluation suite.

## 1. Environment Setup

1. **Activate the Environment**
   Ensure your Python virtual environment is activated and the `single_agent_frameworks` package is available. If you haven't installed dependencies, do so from the project root:
   
   ```bash
   pip install -r requirements.txt
   # OR install single_agent_frameworks in editable mode
   pip install -e .
   ```

2. **API Keys**
   The pipeline requires access to models via API. Set one or more of the following environment variables (depending on your model configurations):
   - `OPEN_ROUTER_KEY`
   - `OPENAI_API_KEY`
   - `ANTHROPIC_API_KEY`
   - `GOOGLE_API_KEY`

## 2. Dataset Preparation

The original dataset must be parsed and split into single-turn (`simple`) and multi-turn (`multi_round`) formats for structured evaluation and to keep the original source immutable.

**Run the Data Translator:**
```bash
python src/example/suicide_detection/data_translator.py
```

By default, this will scan for the source data and output two processed `.csv` files into the `src/example/suicide_detection/data/` directory:
- `rsd_15k_simple.csv`
- `rsd_15k_multi_round.csv`

*(If using a custom data location, you can direct it via the `--input-csv` argument).*

## 3. Running the Model Suite

The primary evaluation execution is wrapped by `run_model_suite.py`. This script acts as an automated harness, leveraging `single_agent_frameworks.utils.suite_runner` to systematically traverse through a list of LLMs configured in a JSON file.

### Basic Suite Execution
To test models using default parameters:

```bash
python src/example/suicide_detection/run_model_suite.py
```

### Advanced Suite Execution
For formal publication runs, you will likely customize parameters like sample size and thread pool.

```bash
python src/example/suicide_detection/run_model_suite.py \
    --n-samples 100 \
    --max-workers 50 \
    --agent-mode single-class \
    --models-file model_list_l.json
```

**Key Parameters:**
- `--n-samples`: Number of samples to evaluate. Omit or set appropriately for full dataset runs.
- `--max-workers`: Maximum concurrent workers for API calls (adjust carefully based on your API rate limits).
- `--seed`: Random seed for reproducibility (default `42`).
- `--agent-mode`: Specifies the classification strategy. Supported patterns:
  - `single-class`: Model outputs a single textual label.
  - `multi-class-chance`: Model provides probabilities for multiple classes.
  - `base-vs-rest-binary`: Agent runs pairwise comparisons against a base class.
  - `multi-agent-host`: A designated host model coordinates with specialized roles (e.g. an "expert" and "critic" model).
- `--base-class`: Applicable only when using `base-vs-rest-binary` (e.g., `Indicator`, `Attempt`).
- `--models-file`: The target JSON file indexing all the models you wish to benchmark in the suite.
- `--no-cache`: Force restarts without applying run caches.

## 4. Single-Model Granular Evaluation

To troubleshoot a specific model or debug agent logic, use the inner script `evaluate_rsd15k_main.py` directly. This evaluates whatever the default environment configuration is defined as, instead of sweeping across the model JSON file.

```bash
python src/example/suicide_detection/evaluate_rsd15k_main.py \
    --n-samples 50 \
    --agent-mode multi-class-chance
```

## 5. Resuming and Retrying Runs

In extensive programmatic runs, API instability can occur. The `single_agent_frameworks` framework handles persistent state tracking. A `Meta.json` artifact within the output directory acts as the run ledger.

### Auto-Continue
To resume operations that were interrupted, appending the `--continue` flag will bypass already successful evaluations from the previous payload, ensuring cost efficiency.

```bash
python src/example/suicide_detection/evaluate_rsd15k_main.py --continue
```
*Note: `run_model_suite.py` safely evaluates with `--continue` logic out-of-the-box unless explicitly bypassed with `--no-cache`.*

### Target Failed API Calls
If you only need to rerun inferences that resulted in an `Error` state (bypassing the cache mapping):

```bash
python src/example/suicide_detection/evaluate_rsd15k_main.py --retry-failed
```

## 6. Output Artifacts and Analysis

Upon completion, output traces are bundled under the `_runs/` output cache directory structure (timestamped per run).

**Contents typically include:**
- `Meta.json`: Comprehensive configuration tracing the parameters, inputs, constraints, and total latency.
- Event JSON lines logging the prompts submitted, API replies generated, and token metadata.
- Generated probability matrices (if `multi-class-chance` is the active topology).

These detailed payload traces integrate downstream into `single_agent_frameworks.utils.visualizations` and significance verification libraries for final formal charting and statistics reporting.
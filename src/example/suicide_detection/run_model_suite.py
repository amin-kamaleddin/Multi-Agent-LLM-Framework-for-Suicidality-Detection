"""
Convenience wrapper to execute the sequential model suite for suicide detection.
"""

import sys
from pathlib import Path
import argparse

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent.parent
sys.path.insert(0, str(project_root))

from EvalRing.utils.suite_runner import run_suite

def run_local_suite(
    n_samples: int = 1,
    max_workers: int = 50,
    seed: int = 42,
    agent_mode: str = "single-class",
    base_class: str = "Indicator",
    models_file: str = "model_list_l.json",
    cache: str = "latest",
    continue_runs: bool = True
) -> int:
    eval_script = script_dir / "evaluate_rsd15k_main.py"
    model_list_path = script_dir / models_file
    
    resolved_cache = str(script_dir / "_EvalRing") if cache == "latest" else cache

    return run_suite(
        eval_script=str(eval_script),
        model_list_path=str(model_list_path),
        n_samples=n_samples,
        max_workers=max_workers,
        seed=seed,
        agent_mode=agent_mode,
        base_class=base_class,
        cache=resolved_cache,
        continue_runs=continue_runs
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the suicide detection model suite.")
    parser.add_argument("--n-samples", type=int, default=1, help="Number of samples to evaluate")
    parser.add_argument("--max-workers", type=int, default=50, help="Parallel workers per model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--agent-mode", type=str, default="single-class", help="Agent execution mode")
    parser.add_argument("--base-class", type=str, default="Indicator", help="Base class for binary modes")
    parser.add_argument("--models-file", type=str, default="model_list_l.json", help="Model list filename in this directory")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching")
    args = parser.parse_args()

    code = run_local_suite(
        n_samples=args.n_samples,
        max_workers=args.max_workers,
        seed=args.seed,
        agent_mode=args.agent_mode,
        base_class=args.base_class,
        models_file=args.models_file,
        cache=None if args.no_cache else "latest",
        continue_runs=not args.no_cache
    )
    sys.exit(code)

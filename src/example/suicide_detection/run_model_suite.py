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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the suicide detection model suite.")
    parser.add_argument("--n-samples", type=int, default=1, help="Number of samples to evaluate")
    parser.add_argument("--max-workers", type=int, default=50, help="Parallel workers per model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--agent-mode", type=str, default="single-class", help="Agent execution mode")
    parser.add_argument("--base-class", type=str, default="Indicator", help="Base class for binary modes")
    parser.add_argument("--models-file", type=str, default="model_list.json", help="Model list filename in this directory")
    args = parser.parse_args()

    eval_script = script_dir / "evaluate_rsd15k_main.py"
    model_list_path = script_dir / args.models_file

    code = run_suite(
        eval_script=str(eval_script),
        model_list_path=str(model_list_path),
        n_samples=args.n_samples,
        max_workers=args.max_workers,
        seed=args.seed,
        agent_mode=args.agent_mode,
        base_class=args.base_class,
    )
    sys.exit(code)

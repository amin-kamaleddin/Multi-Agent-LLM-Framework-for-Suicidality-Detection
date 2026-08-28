"""Build the final user-cluster workflow comparison from committed run artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation.workflow_comparison import write_repository_comparison


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build paired user-cluster statistical comparisons for completed Luna workflows."
    )
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository root containing evaluation_results/ (default: inferred from this script).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <repository-root>/evaluation_results/workflow_comparison).",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=2_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_820)
    args = parser.parse_args()
    output_dir = args.output_dir or args.repository_root / "evaluation_results" / "workflow_comparison"
    outputs = write_repository_comparison(
        args.repository_root,
        output_dir,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(json.dumps(outputs, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import json
import os
import csv
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

MODEL_NAMES = [
    "openai/gpt-5-mini",
    "anthropic/claude-haiku-4.5",
    "google/gemini-3-flash-preview",
    "x-ai/grok-3-mini",
]


def _latest_run_dir(runs_root: Path) -> Path | None:
    if not runs_root.exists():
        return None
    run_dirs = [p for p in runs_root.iterdir() if p.is_dir() and p.name.startswith("run_")]
    if not run_dirs:
        return None
    return sorted(run_dirs, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def _read_meta(meta_path: Path) -> dict:
    if not meta_path.exists():
        return {}
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _count_prediction_errors(all_cases_csv: Path) -> int:
    if not all_cases_csv.exists():
        return 0
    count = 0
    with open(all_cases_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            prediction = (row.get("prediction") or "").strip().lower()
            err = (row.get("error") or "").strip()
            if prediction == "error" or err:
                count += 1
    return count


def run_suite(
    n_samples: int = 2,
    max_workers: int = 32,
    seed: int = 0,
    agent_mode: str = "single-class",
    base_class: str = "Indicator",
) -> int:
    script_dir = Path(__file__).resolve().parent
    eval_script = script_dir / "evaluate_rsd15k_main.py"
    runs_root = script_dir / ".EvalRing"
    project_root = script_dir.parent.parent.parent

    load_dotenv(project_root / ".env")

    if not eval_script.exists():
        print(f"ERROR: evaluate script not found: {eval_script}")
        return 1

    if not os.environ.get("OPEN_ROUTER_KEY") and not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: Missing OPEN_ROUTER_KEY / OPENAI_API_KEY in environment.")
        return 1

    summary_rows = []

    print("=" * 80)
    print("Model suite start")
    print(f"timestamp={datetime.now().isoformat()}")
    print(
        f"n_samples={n_samples}, max_workers={max_workers}, seed={seed}, "
        f"agent_mode={agent_mode}, base_class={base_class}"
    )
    print("=" * 80)

    for idx, model_name in enumerate(MODEL_NAMES, start=1):
        print(f"\n[{idx}/{len(MODEL_NAMES)}] Running model: {model_name}")

        env = os.environ.copy()
        env["OPEN_ROUTER_MODEL"] = model_name

        before_dir = _latest_run_dir(runs_root)

        cmd = [
            sys.executable,
            str(eval_script),
            "--n-samples",
            str(n_samples),
            "--max-workers",
            str(max_workers),
            "--seed",
            str(seed),
            "--agent-mode",
            str(agent_mode),
            "--base-class",
            str(base_class),
        ]

        proc = subprocess.run(cmd, cwd=str(script_dir), env=env)

        if proc.returncode != 0:
            print(f"ERROR: Run failed for model '{model_name}' (return code {proc.returncode}).")
            print("Stopping suite (next model will NOT run).")
            return proc.returncode

        after_dir = _latest_run_dir(runs_root)
        if after_dir is None or (before_dir is not None and after_dir == before_dir):
            print(f"ERROR: Could not locate new run directory for model '{model_name}'.")
            print("Stopping suite (next model will NOT run).")
            return 2

        meta = _read_meta(after_dir / "Meta.json")
        execution_failures = meta.get("execution_failures", [])
        prediction_error_count = _count_prediction_errors(after_dir / "all_cases.csv")
        metrics = meta.get("aggregate_metrics", {})
        accuracy = metrics.get("accuracy")
        f1 = metrics.get("f1_score")

        summary_rows.append(
            {
                "model": model_name,
                "run_dir": str(after_dir),
                "accuracy": accuracy,
                "f1_score": f1,
                "execution_failures": len(execution_failures),
                "prediction_errors": prediction_error_count,
                "agent_mode": agent_mode,
                "base_class": base_class,
            }
        )

        print(
            f"OK: accuracy={accuracy}, f1={f1}, execution_failures={len(execution_failures)}, prediction_errors={prediction_error_count}"
        )

        if execution_failures:
            print(
                f"ERROR: Found {len(execution_failures)} execution failures in Meta.json for '{model_name}'."
            )
            print("Stopping suite (next model will NOT run).")
            return 3

        if prediction_error_count > 0:
            print(
                f"ERROR: Found {prediction_error_count} per-sample prediction errors in all_cases.csv for '{model_name}'."
            )
            print("Stopping suite (next model will NOT run).")
            return 4

    suite_report = {
        "timestamp": datetime.now().isoformat(),
        "n_samples": n_samples,
        "max_workers": max_workers,
        "agent_mode": agent_mode,
        "base_class": base_class,
        "models": summary_rows,
    }

    report_path = runs_root / f"suite_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(suite_report, f, indent=2)

    print("\n" + "=" * 80)
    print("Suite completed successfully")
    print(f"Report: {report_path}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(
        run_suite(
            n_samples=1000,
            max_workers=10,
            seed=0,
            agent_mode="single-class",
            base_class="Indicator",
        )
    )

"""
Generic utility to run a suite of models alongside an EvalRing evaluation script.
"""

import json
import os
import csv
import subprocess
import sys
import concurrent.futures
from datetime import datetime
from pathlib import Path
import argparse
import logging

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
try:
    from EvalRing.utils.visualizations import generate_suite_visuals
except ImportError:
    generate_suite_visuals = None


def _load_model_names(model_list_path: Path) -> list[str]:
    """Load model list JSON file and parse requested models preserving their ordering."""
    if not model_list_path.exists():
        logging.warning(f"Model list not found: {model_list_path}")
        return []

    with open(model_list_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    models = data.get("models", [])
    model_names: list[str] = []
    for m in models:
        if isinstance(m, str):
            model_name = m.strip()
            if model_name:
                model_names.append(model_name)
            continue

        if isinstance(m, dict):
            if m.get("available") is False:
                continue
            model_name = (m.get("openrouter_id") or "").strip()
            if model_name:
                model_names.append(model_name)
            continue

    # De-dup while preserving order
    seen: set[str] = set()
    deduped: list[str] = []
    for name in model_names:
        if name in seen:
            continue
        seen.add(name)
        deduped.append(name)

    return deduped


def _read_meta(meta_path: Path) -> dict:
    """Read Meta.json output safely."""
    if not meta_path.exists():
        return {}
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _sum_error_histogram(error_histogram: object) -> int:
    """Sum arbitrary error dictionaries robustly."""
    if not isinstance(error_histogram, dict):
        return 0
    total = 0
    for v in error_histogram.values():
        if isinstance(v, bool):
            total += int(v)
            continue
        if isinstance(v, (int, float)):
            total += int(v)
            continue
        if isinstance(v, str):
            s = v.strip()
            if not s:
                continue
            try:
                total += int(float(s))
            except ValueError:
                continue
    return total


def _parse_boolish_int(value: str | None) -> int | None:
    """Safely parse boolean or integer-like truthy strings."""
    if value is None:
        return None
    s = str(value).strip().lower()
    if s in {"1", "true", "t", "yes", "y"}:
        return 1
    if s in {"0", "false", "f", "no", "n"}:
        return 0
    return None


def _count_incorrect_and_errors(all_cases_csv: Path) -> tuple[int, int]:
    """Parse out incorrect cases directly from the target CSV output."""
    if not all_cases_csv.exists():
        return 0, 0
    incorrect_count = 0
    error_count = 0
    with open(all_cases_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            correct = _parse_boolish_int(row.get("correct"))
            if correct == 0:
                incorrect_count += 1
            prediction = (row.get("prediction") or "").strip().lower()
            err = (row.get("error") or "").strip()
            if prediction == "error" or err:
                error_count += 1
    return incorrect_count, error_count


def _strip_per_sample_results(md_text: str) -> str:
    """Strips the per-sample results portion to compress markdown reports."""
    lines = md_text.splitlines(keepends=True)
    out: list[str] = []
    skipping = False
    for line in lines:
        if not skipping and line.strip() == "## Per-Sample Results":
            skipping = True
            continue
        if skipping:
            if line.startswith("## ") and line.strip() != "## Per-Sample Results":
                skipping = False
                out.append(line)
            continue
        out.append(line)
    return "".join(out)


def run_suite(
    eval_script: str | Path,
    model_list_path: str | Path,
    n_samples: int = 2,
    max_workers: int = 32,
    seed: int = 0,
    agent_mode: str = "single-class",
    base_class: str = "Indicator",
    host_model: str | None = None,
    role_models_json: str | None = None,
    max_host_iterations: int = 10,
    continue_runs: bool = False,
    cache: str | None = None,
    out_dir: str | None = None,
) -> int:
    """
    Primary interface for invoking external evaluation scripts dynamically
    across numerous underlying LLMs defined by the model list.
    """
    eval_script = Path(eval_script).resolve()
    model_list_path = Path(model_list_path).resolve()
    
    script_dir = eval_script.parent
    project_root = script_dir.parent.parent.parent

    model_names = _load_model_names(model_list_path)
    if not model_names:
        logging.error(f"No models loaded from {model_list_path}")
        return 1

    load_dotenv(project_root / ".env")

    if not eval_script.exists():
        logging.error(f"Evaluate script not found: {eval_script}")
        return 1

    if not os.environ.get("OPEN_ROUTER_KEY") and not os.environ.get("OPENAI_API_KEY"):
        logging.error("Missing OPEN_ROUTER_KEY / OPENAI_API_KEY in environment.")
        return 1

    summary_rows = []
    suite_started_at = datetime.now()
    suite_stamp = suite_started_at.strftime('%Y%m%d_%H%M%S')
    
    if out_dir:
        suite_dir = Path(out_dir).resolve() / f"run_suite_{suite_stamp}"
    else:
        suite_dir = script_dir / ".EvalRing" / f"run_suite_{suite_stamp}"
        
    suite_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Model suite start")
    print(f"Timestamp: {suite_started_at.isoformat()}")
    print(f"eval_script: {eval_script}")
    print(
        f"n_samples={n_samples}, max_workers={max_workers}, seed={seed}, "
        f"agent_mode={agent_mode}, base_class={base_class}, host_model={host_model}, "
        f"max_host_iterations={max_host_iterations}, continue_runs={continue_runs}"
    )
    print(f"models_loaded={len(model_names)} from {model_list_path}")
    print(f"suite_dir={suite_dir}")
    print("=" * 80)

    def _run_single_model(model_name: str, idx: int) -> dict:
        print(f"[{idx}/{len(model_names)}] Starting model: {model_name}")
        env = os.environ.copy()
        env["OPEN_ROUTER_MODEL"] = model_name
        env["PYTHONIOENCODING"] = "utf-8"

        cmd = [
            sys.executable,
            str(eval_script),
            "--n-samples", str(n_samples),
            "--max-workers", str(max_workers),
            "--seed", str(seed),
            "--agent-mode", str(agent_mode),
            "--base-class", str(base_class),
            "--max-host-iterations", str(max_host_iterations),
            "--out-dir", str(suite_dir),
        ]
        if continue_runs:
            cmd.append("--continue")
        if cache:
            cmd.extend(["--cache", str(cache)])
        if host_model:
            cmd.extend(["--host-model", str(host_model)])
        if role_models_json:
            cmd.extend(["--role-models-json", str(role_models_json)])

        clean_name = model_name.replace("/", "_")
        log_dir = suite_dir / ".log"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"run_{clean_name}.log"

        with open(log_path, "w", encoding="utf-8") as f_log:
            proc = subprocess.run(cmd, cwd=str(script_dir), env=env, stdout=f_log, stderr=subprocess.STDOUT)

        if proc.returncode != 0:
            print(f"ERROR: Run failed for model '{model_name}' (return code {proc.returncode}). See {log_path.name}")
            return {"error": proc.returncode, "model": model_name}

        # Scan log for output dir
        after_dir = None
        with open(log_path, "r", encoding="utf-8") as f_log:
            for line in f_log:
                if line.startswith("Output:   "):
                    after_dir = Path(line.replace("Output:   ", "").strip())
                    break

        if not after_dir or not after_dir.exists():
            model_suffix = model_name.split("/")[-1].lower()
            possible_dirs = []
            for p in suite_dir.iterdir():
                if p.is_dir() and p.name.startswith("run_"):
                    if model_suffix in p.name.lower():
                        possible_dirs.append(p)
            if possible_dirs:
                after_dir = sorted(possible_dirs, key=lambda d: d.stat().st_mtime, reverse=True)[0]

        if not after_dir or not after_dir.exists():
            print(f"ERROR: Could not locate new run directory for model '{model_name}'.")
            return {"error": 2, "model": model_name}

        meta = _read_meta(after_dir / "Meta.json")
        execution_failures = meta.get("execution_failures", [])
        error_histogram = meta.get("error", {})
        incorrect_count, execution_error_count = _count_incorrect_and_errors(after_dir / "all_cases.csv")
        error_total = _sum_error_histogram(error_histogram)
        metrics = meta.get("aggregate_metrics", {})
        accuracy = metrics.get("accuracy")
        f1 = metrics.get("f1_score")

        print(f"[{idx}/{len(model_names)}] Completed model: {model_name} -> acc: {accuracy}, err: {execution_error_count}")

        return {
            "model": model_name,
            "run_dir": str(after_dir),
            "accuracy": accuracy,
            "f1_score": f1,
            "execution_failures": len(execution_failures),
            "incorrect": incorrect_count,
            "execution_errors": execution_error_count,
            "error_total": error_total,
            "agent_mode": agent_mode,
            "base_class": base_class,
            "host_model": host_model,
            "role_models_json": role_models_json,
            "max_host_iterations": max_host_iterations,
        }

    futures = []
    has_errors = False
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(model_names)) as executor:
        for idx, model_name in enumerate(model_names, start=1):
            futures.append(executor.submit(_run_single_model, model_name, idx))
        
        for future in concurrent.futures.as_completed(futures):
            try:
                res = future.result()
                if "error" in res:
                    has_errors = True
                else:
                    summary_rows.append(res)
            except Exception as e:
                logging.error(f"Executor failed with future: {e}")
                has_errors = True

    if has_errors:
        print("WARNING: One or more models encountered errors during execution.")

    suite_report = {
        "timestamp": suite_started_at.isoformat(),
        "n_samples": n_samples,
        "max_workers": max_workers,
        "agent_mode": agent_mode,
        "base_class": base_class,
        "host_model": host_model,
        "role_models_json": role_models_json,
        "max_host_iterations": max_host_iterations,
        "models": summary_rows,
    }

    report_path = suite_dir / f"suite_report_{suite_stamp}.json"
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(suite_report, f, indent=2)
    except Exception as e:
        logging.error(f"Failed to record JSON suite report: {e}")

    combined_md_path = suite_dir / f"run_suite_{suite_stamp}.md"
    try:
        with open(combined_md_path, "w", encoding="utf-8") as out:
            out.write("# Model Suite Report\n\n")
            out.write(f"**timestamp:** {suite_report['timestamp']}\n\n")
            out.write(
                f"**n_samples:** {n_samples} | **max_workers:** {max_workers} | **seed:** {seed} | "
                f"**agent_mode:** {agent_mode} | **base_class:** {base_class}\n\n"
            )
            for i, row in enumerate(summary_rows, start=1):
                model_name = row.get("model")
                run_dir = Path(str(row.get("run_dir")))
                result_path = run_dir / "result.md"

                out.write("---\n\n")
                out.write(f"## {i}. {model_name}\n\n")
                out.write(f"**run_dir:** {run_dir}\n\n")
                out.write(
                    f"**accuracy:** {row.get('accuracy')} | **f1_score:** {row.get('f1_score')} | "
                    f"**incorrect:** {row.get('incorrect')} | **execution_errors:** {row.get('execution_errors')}\n\n"
                )

                if not result_path.exists():
                    out.write(f"(missing {result_path})\n\n")
                    continue

                text = result_path.read_text(encoding="utf-8", errors="replace")
                text = _strip_per_sample_results(text)
                out.write(text)
                if not text.endswith("\n"):
                    out.write("\n")
    except Exception as e:
        logging.error(f"Failed to generate composite MD document: {e}")

    print("\n" + "=" * 80)
    print("Suite completed successfully" if not has_errors else "Suite completed with some failures")
    print(f"Report: {report_path}")
    print(f"Combined markdown: {combined_md_path}")
    print("=" * 80)

    # -------------------------------------------------------------
    # Auto-trigger visuals
    # -------------------------------------------------------------
    if generate_suite_visuals:
        try:
            print("\nAuto-triggering generic suite visuals...")
            generate_suite_visuals(report_path, model_list_path)
        except Exception as e:
            logging.error(f"Visualization generation failed safely: {e}")
    else:
        print("\nVisualizations skipped (module matplotlib not importable or could not import generator).")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run EvalRing suite.")
    parser.add_argument("--eval-script", type=str, required=True, help="Path to evaluation main script")
    parser.add_argument("--models-file", type=str, required=True, help="Path to model list JSON")
    parser.add_argument("--n-samples", type=int, default=1)
    parser.add_argument("--max-workers", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--agent-mode", type=str, default="single-class")
    parser.add_argument("--base-class", type=str, default="Indicator")
    parser.add_argument("--host-model", type=str, default=None)
    parser.add_argument("--role-models-json", type=str, default=None)
    parser.add_argument("--max-host-iterations", type=int, default=10)
    parser.add_argument("--continue-runs", action="store_true")
    parser.add_argument("--cache", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)

    args = parser.parse_args()
    
    code = run_suite(
        eval_script=args.eval_script,
        model_list_path=args.models_file,
        n_samples=args.n_samples,
        max_workers=args.max_workers,
        seed=args.seed,
        agent_mode=args.agent_mode,
        base_class=args.base_class,
        host_model=args.host_model,
        role_models_json=args.role_models_json,
        max_host_iterations=args.max_host_iterations,
        continue_runs=args.continue_runs,
        cache=args.cache,
        out_dir=args.out_dir,
    )
    sys.exit(code)

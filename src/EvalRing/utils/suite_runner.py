"""
Generic utility to run a suite of models alongside an EvalRing evaluation script.
"""

import json
import os
import csv
import subprocess
import sys
import concurrent.futures
import threading
import time
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
    base_class: str = "Unsupportive",
    host_model: str | None = None,
    role_models_json: str | None = None,
    max_host_iterations: int = 10,
    continue_runs: bool = False,
    cache: str | None = None,
    cache_mode: str = "both",
    ignore_errors: bool = False,
    out_dir: str | None = None,
    data_path: str | None = None,
    no_confirm: bool = False,
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
        suite_dir = Path(out_dir).resolve() / f"run_suite_{suite_stamp}_{agent_mode}"
    else:
        suite_dir = script_dir / "_EvalRing" / f"run_suite_{suite_stamp}_{agent_mode}"
        
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

    if not no_confirm:
        print("\nPre-computing cache hits for the suite...")
        total_cached = 0
        total_uncached = 0
        
        import concurrent.futures
        import json
        
        def _get_cache_info(idx_model):
            idx, model_name = idx_model
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
                "--info-only"
            ]
            if data_path:
                cmd.extend(["--data-path", str(data_path)])
            if cache:
                cmd.extend(["--cache", str(cache)])
            if cache_mode:
                cmd.extend(["--cache-mode", cache_mode])
            if host_model:
                cmd.extend(["--host-model", str(host_model)])
            if role_models_json:
                cmd.extend(["--role-models-json", str(role_models_json)])

            try:
                # We use bufsize=0 and binary mode (no text=True) so we can stream \r unbuffered tightly.
                proc = subprocess.Popen(cmd, cwd=str(script_dir), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
                
                final_line = ""
                
                import threading
                stderr_output = []
                def drain_stderr(pipe):
                    buffer = bytearray()
                    while True:
                        byte = pipe.read(1)
                        if not byte:
                            break
                        buffer.extend(byte)
                        if byte == b'\n' or byte == b'\r':
                            l_clean = buffer.decode("utf-8", errors="replace").strip()
                            buffer.clear()
                            if l_clean:
                                with cache_lock:
                                    cache_progress[model_name] = l_clean
                                update_progress_file()
                                stderr_output.append(l_clean + "\n")
                    if buffer:
                        l_clean = buffer.decode("utf-8", errors="replace").strip()
                        if l_clean:
                            with cache_lock:
                                cache_progress[model_name] = l_clean
                            update_progress_file()
                            stderr_output.append(l_clean + "\n")

                stderr_thread = threading.Thread(target=drain_stderr, args=(proc.stderr,))
                stderr_thread.start()

                # Read stdout in binary and decode line by line natively
                for line_b in proc.stdout:
                    line = line_b.decode("utf-8", errors="replace").strip()
                    if line:
                        final_line = line
                        if not line.startswith("{"):
                            with cache_lock:
                                cache_progress[model_name] = line
                            update_progress_file()
                        
                proc.wait()
                stderr_thread.join()
                if proc.returncode != 0:
                    last_error = ""
                    for line in reversed(stderr_output):
                        if line.strip() and "tqdm" not in line and "overlapping" not in line.lower():
                            if "error" in line.lower() or "exception" in line.lower():
                                last_error = line.strip()
                                break
                            if not last_error:
                                last_error = line.strip()
                    if not last_error:
                        last_error = "Unknown subprocess error."
                    raise subprocess.CalledProcessError(proc.returncode, cmd, stderr=last_error)
                
                import re
                match = re.search(r'(\{.*\})', final_line)
                if match:
                    cache_info = json.loads(match.group(1))
                else:
                    cache_info = json.loads(final_line)

                return model_name, cache_info.get("cached", 0), cache_info.get("uncached", 0), idx, None
            except Exception as e:
                err_msg = str(e)
                if isinstance(e, subprocess.CalledProcessError) and e.stderr:
                    err_msg += f"\nStderr: {e.stderr}"
                return model_name, 0, 0, idx, err_msg
                
        from tqdm import tqdm
        import threading
        
        cache_progress = {m: "Starting..." for m in model_names}
        cache_lock = threading.Lock()
        progress_file = suite_dir / "progress.txt"
        
        def update_progress_file():
            with cache_lock:
                lines = ["Cache Scan Status:", "=" * 80]
                for m_name, status in cache_progress.items():
                    # Format model name to 30 chars
                    m_short = (m_name[:27] + "..") if len(m_name) > 30 else m_name
                    lines.append(f"{m_short:<30} : {status}")
                with open(progress_file, "w", encoding="utf-8") as pf:
                    pf.write("\n".join(lines) + "\n")
                    
        update_progress_file()
        
        # Process in parallel up to max_workers or len(model_names)
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(max_workers, len(model_names))) as executor:
            future_to_model = {executor.submit(_get_cache_info, (i, m)): m for i, m in enumerate(model_names, 1)}
            
            completed = 0
            with tqdm(total=len(model_names), desc="Checking cache info") as pbar:
                for future in concurrent.futures.as_completed(future_to_model):
                    m_name = future_to_model[future]
                    completed += 1
                    try:
                        m, c, u, idx, err = future.result()
                        with cache_lock:
                            if err:
                                cache_progress[m_name] = f"Error: {err}"
                            else:
                                cache_progress[m_name] = f"[Cached: {c:<4} | Fresh: {u:<4}] Done."
                                total_cached += c
                                total_uncached += u
                    except Exception as e:
                        with cache_lock:
                            cache_progress[m_name] = f"Error: {e}"
                            
                    update_progress_file()
                    pbar.set_postfix_str(f"{completed}/{len(model_names)} completed")
                    pbar.update(1)
                
        print("\n" + "=" * 50)
        print(f"Suite Cache Summary:")
        print(f"  Total API calls cached:   {total_cached}")
        print(f"  Total API calls uncached: {total_uncached}")
        print("=" * 50)
        
        while True:
            val = input("Type 'y' to continue with the suite run, or 'q' to quit: ").strip().lower()
            if val == 'y':
                print("Proceeding...")
                break
            elif val == 'q':
                print("Suite aborted.")
                sys.exit(0)
            else:
                print("Invalid input. Please type 'y' to confirm or 'q' to abort.")

    def _run_single_model(model_name: str, idx: int) -> dict:
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
        if data_path:
            cmd.extend(["--data-path", str(data_path)])
        # IMPORTANT: Suite runner must not block on input! Force -nc.
        cmd.append("-nc")
        if continue_runs:
            cmd.append("--continue")
        if cache:
            cmd.extend(["--cache", str(cache)])
        if cache_mode:
            cmd.extend(["--cache-mode", cache_mode])
        if ignore_errors:
            cmd.append("--ignore-errors")
        if ignore_errors:
            cmd.append("--ignore-errors")
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
    stop_monitor = False
    def _monitor_progress():
        import shutil
        
        progress_file = suite_dir / "suite_progress.txt"
        
        while not stop_monitor:
            total_lines = 0
            total_expected = len(model_names) * n_samples
            per_model = []
            
            for m in model_names:
                m_suffix = m.split("/")[-1].lower()
                clean_name = m.replace("/", "_")
                log_path = suite_dir / ".log" / f"run_{clean_name}.log"
                
                cache_hits = 0
                is_done = False
                if log_path.exists():
                    try:
                        with open(log_path, "r", encoding="utf-8") as lf:
                            for log_line in lf:
                                if "overlapping with current run" in log_line and "remaining." in log_line:
                                    try:
                                        parts = log_line.split("Loaded ")
                                        if len(parts) > 1:
                                            cache_hits = int(parts[1].split(" ")[0])
                                    except Exception:
                                        pass
                                if "Evaluation complete" in log_line or "completely bypassed" in log_line:
                                    is_done = True
                    except Exception:
                        pass

                lines = 0
                latest_dir = None
                for p in suite_dir.iterdir():
                    if p.is_dir() and p.name.startswith("run_") and m_suffix in p.name.lower():
                        latest_dir = p
                
                if latest_dir:
                    partial_path = latest_dir / "all_cases_partial.jsonl"
                    if partial_path.exists():
                        try:
                            with open(partial_path, "r", encoding="utf-8") as pf:
                                lines = sum(1 for line in pf if line.strip())
                        except Exception:
                            pass
                
                if is_done:
                    current_prog = n_samples
                else:
                    current_prog = cache_hits + lines
                
                current_prog = min(current_prog, n_samples)
                
                import textwrap
                short_m = textwrap.shorten(m_suffix, width=20, placeholder="..")
                per_model.append(f"{short_m.ljust(20)} : {current_prog} / {n_samples}  [Cache: {cache_hits}, Fresh: {lines}]")
                total_lines += current_prog
            
            # Write detailed view to progress file
            try:
                with open(progress_file, "w", encoding="utf-8") as pf:
                    pf.write(f"Suite Status: {total_lines} / {total_expected} completed\n")
                    pf.write("=" * 40 + "\n")
                    pf.write("\n".join(per_model) + "\n")
            except Exception:
                pass
            
            # Terminal view: simple non-wrapping single line
            max_w = min(120, shutil.get_terminal_size().columns - 2)
            pct = (total_lines / max(1, total_expected)) * 100
            out_str = f"Suite Progress: {total_lines}/{total_expected} ({pct:.1f}%) -> Details in {progress_file.name}"
            
            sys.stdout.write("\r" + out_str.ljust(max_w))
            sys.stdout.flush()
            time.sleep(2)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(model_names)) as executor:
        for idx, model_name in enumerate(model_names, start=1):
            futures.append(executor.submit(_run_single_model, model_name, idx))
            
        monitor_thread = threading.Thread(target=_monitor_progress, daemon=True)
        monitor_thread.start()
        
        for future in concurrent.futures.as_completed(futures):
            try:
                res = future.result()
                if "error" in res:
                    has_errors = True
                else:
                    summary_rows.append(res)
            except Exception as e:
                logging.error(f"\nExecutor failed with future: {e}")
                has_errors = True
                
        stop_monitor = True
        monitor_thread.join(timeout=3)
        sys.stdout.write("\n")

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
    parser.add_argument("--n-samples", type=int, default=29727)
    parser.add_argument("--max-workers", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--agent-mode", type=str, default="single-class")
    parser.add_argument("--base-class", type=str, default="Unsupportive")
    parser.add_argument("--host-model", type=str, default=None)
    parser.add_argument("--role-models-json", type=str, default=None)
    parser.add_argument("--max-host-iterations", type=int, default=10)
    parser.add_argument("--continue-runs", action="store_true")
    parser.add_argument("--cache", type=str, default=None)
    parser.add_argument("--cache-mode", type=str, default="both", choices=["runs_only", "cache_file", "both"])
    parser.add_argument("--ignore-errors", action="store_true", help="Ignore cached errors, forcing them to re-run.")
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
        cache_mode=args.cache_mode,
        ignore_errors=args.ignore_errors,
        out_dir=args.out_dir,
    )
    sys.exit(code)

"""
Utilities for generating visualizations from single_agent_frameworks suite reports.
"""

import json
from pathlib import Path
from typing import Optional
import logging

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def short_name(model_name: str) -> str:
    """Extract a short display name from a full model identifier (e.g., 'openai/gpt-4o' -> 'gpt-4o')."""
    if "/" in model_name:
        return model_name.split("/", 1)[1]
    return model_name


def generate_suite_visuals(report_path: Path, model_list_path: Optional[Path] = None) -> None:
    """
    Parse a suite_report JSON and generate performance and reliability visual charts.
    
    Args:
        report_path: Path to the suite_report JSON file.
        model_list_path: Optional path to a model_list.json to resolve pricing.
    """
    if plt is None:
        logging.warning("matplotlib is not installed. Visualizations cannot be generated.")
        return

    if not isinstance(report_path, Path):
        report_path = Path(report_path)
        
    if not report_path.exists():
        logging.error(f"Report not found: {report_path}")
        return

    run_dir = report_path.parent
    out_dir = run_dir / "visuals"
    try:
        out_dir.mkdir(exist_ok=True)
    except Exception as e:
        logging.error(f"Failed to create visuals directory at {out_dir}: {e}")
        return

    try:
        with report_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logging.error(f"Failed to read report JSON at {report_path}: {e}")
        return

    models = data.get("models", [])
    if not models:
        logging.warning("No models found in suite report JSON to visualize.")
        return

    records = []
    for m in models:
        records.append(
            {
                "model": m["model"],
                "name": short_name(m["model"]),
                "accuracy": float(m.get("accuracy") or 0.0),
                "f1": float(m.get("f1_score") or 0.0),
                "errors": int(m.get("execution_errors") or 0),
            }
        )

    by_f1 = sorted(records, key=lambda x: x["f1"], reverse=True)
    by_err = sorted(records, key=lambda x: x["errors"], reverse=True)

    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        pass # Fallback to default styling safely

    _generate_leaderboard_chart(by_f1, out_dir / "01_leaderboard_f1_accuracy.png")
    _generate_quality_scatter(records, out_dir / "02_quality_tradeoff_scatter.png")
    _generate_reliability_chart(by_err, out_dir / "03_reliability_execution_errors.png")
    
    if model_list_path and model_list_path.exists():
        _generate_cost_scatter(records, model_list_path, out_dir / "04_cost_vs_f1_scatter.png")

    _generate_composite_priority(records, out_dir / "04_composite_priority.png")
    _write_markdown_summary(run_dir, report_path.name, records, by_f1)
    
    print(f"Saved visuals to: {out_dir}")
    print(f"Saved summary to: {run_dir / 'visuals_summary.md'}")


def _generate_leaderboard_chart(by_f1: list, out_path: Path):
    """Generate and save the macro F1 vs Accuracy leaderboard chart."""
    names = [r["name"] for r in by_f1]
    f1_vals = [r["f1"] for r in by_f1]
    acc_vals = [r["accuracy"] for r in by_f1]

    fig, ax = plt.subplots(figsize=(12, 7))
    y = list(range(len(names)))
    bars = ax.barh(y, f1_vals, color="#1976d2", alpha=0.9, label="Macro F1")
    ax.scatter(acc_vals, y, color="#ef6c00", s=70, label="Accuracy", zorder=3)

    for i, (b, f1, acc) in enumerate(zip(bars, f1_vals, acc_vals)):
        ax.text(f1 + 0.004, i, f"F1 {f1:.3f}", va="center", fontsize=9)
        ax.text(acc + 0.004, i - 0.22, f"Acc {acc:.3f}", va="center", fontsize=8, color="#ef6c00")

    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlim(0.30, 1.0) # Adjusted limits generically
    ax.set_xlabel("Score")
    ax.set_title("Model Leaderboard (Sorted by Macro F1)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    try:
        fig.savefig(out_path, dpi=180)
    except Exception as e:
        logging.error(f"Error saving chart {out_path}: {e}")
    plt.close(fig)


def _generate_quality_scatter(records: list, out_path: Path):
    """Generate and save the accuracy vs macro F1 tradeoff scatter chart."""
    fig, ax = plt.subplots(figsize=(9.5, 7))
    for r in records:
        size = 80 + 24 * r["errors"]
        color = "#d32f2f" if r["errors"] > 0 else "#2e7d32"
        ax.scatter(r["accuracy"], r["f1"], s=size, c=color, alpha=0.75, edgecolor="black", linewidth=0.5)
        ax.text(r["accuracy"] + 0.0012, r["f1"] + 0.0012, r["name"], fontsize=8)

    ax.set_xlabel("Accuracy")
    ax.set_ylabel("Macro F1")
    ax.set_title("Quality Tradeoff: Accuracy vs Macro F1 (Bubble Size = Execution Errors)")
    fig.tight_layout()
    try:
        fig.savefig(out_path, dpi=180)
    except Exception as e:
        logging.error(f"Error saving chart {out_path}: {e}")
    plt.close(fig)


def _generate_reliability_chart(by_err: list, out_path: Path):
    """Generate and save a bar chart displaying execution errors."""
    names_err = [r["name"] for r in by_err]
    err_vals = [r["errors"] for r in by_err]
    colors = ["#c62828" if v > 0 else "#90a4ae" for v in err_vals]

    fig, ax = plt.subplots(figsize=(12, 5.5))
    bars = ax.bar(names_err, err_vals, color=colors)
    
    max_err = max(err_vals) if err_vals else 0
    y_max = max(max_err * 1.15, 1.0)
    ax.set_ylim(0, y_max)
    
    for b, v in zip(bars, err_vals):
        offset = y_max * 0.02
        ax.text(b.get_x() + b.get_width() / 2, v + offset, str(v), ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Execution Errors")
    ax.set_title("Reliability Risk by Model (Execution Errors)")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    try:
        fig.savefig(out_path, dpi=180)
    except Exception as e:
        logging.error(f"Error saving chart {out_path}: {e}")
    plt.close(fig)


def _generate_cost_scatter(records: list, model_list_path: Path, out_path: Path):
    """Generate cost vs F1 score visual mapped with pricing data."""
    costs = {}
    try:
        with model_list_path.open("r", encoding="utf-8") as f:
            ml_data = json.load(f)
            for m in ml_data.get("models", []):
                if m.get("openrouter_id") and m.get("pricing"):
                    costs[m["openrouter_id"]] = m["pricing"].get("total_usd_per_1m_tokens", 0.0)
    except Exception as e:
        logging.error(f"Failed to read model list for pricing: {e}")

    for r in records:
        r["cost"] = costs.get(r["model"], 0.0)

    fig, ax = plt.subplots(figsize=(9.5, 7))
    for r in records:
        size = 120
        color = "#5e35b1"
        ax.scatter(r["cost"], r["f1"], s=size, c=color, alpha=0.75, edgecolor="black", linewidth=0.5)
        ax.text(r["cost"] + (0.01 if r["cost"] < 2 else 0.1), r["f1"] + 0.002, r["name"], fontsize=8)

    ax.set_xlabel("Cost (USD per 1M tokens)")
    ax.set_ylabel("Macro F1")
    ax.set_title("Cost vs F1 Score Tradeoff")
    fig.tight_layout()
    try:
        fig.savefig(out_path, dpi=180)
    except Exception as e:
        logging.error(f"Error saving chart {out_path}: {e}")
    plt.close(fig)


def _generate_composite_priority(records: list, out_path: Path):
    """Generate a composite priority chart based on F1 with a small penalty for errors."""
    composite = []
    for r in records:
        score = r["f1"] - 0.0025 * r["errors"]
        composite.append((r["name"], score, r["f1"], r["errors"]))

    composite_sorted = sorted(composite, key=lambda x: x[1], reverse=True)
    names_c = [x[0] for x in composite_sorted]
    score_c = [x[1] for x in composite_sorted]

    fig, ax = plt.subplots(figsize=(12, 6.5))
    bars = ax.barh(names_c, score_c, color="#6a1b9a")
    ax.invert_yaxis()
    for b, (_, s, f1, err) in zip(bars, composite_sorted):
        ax.text(b.get_width() + 0.003, b.get_y() + b.get_height() / 2, f"S {s:.3f} | F1 {f1:.3f} | E {err}", va="center", fontsize=8)

    ax.set_xlim(0.20, 1.0) # Adjusted limits generically
    ax.set_xlabel("Composite Score")
    ax.set_title("Composite Prioritization (F1 with Error Penalty)")
    fig.tight_layout()
    try:
        fig.savefig(out_path, dpi=180)
    except Exception as e:
        logging.error(f"Error saving chart {out_path}: {e}")
    plt.close(fig)


def _write_markdown_summary(run_dir: Path, report_filename: str, records: list, by_f1: list):
    """Write the markdown summary presentation document incorporating all run visual takeaways."""
    top3 = by_f1[:3]
    risky = [r for r in records if r["errors"] > 0]

    lines = []
    lines.append("# Run Suite Visual Pack")
    lines.append("")
    lines.append(f"Generated from `{report_filename}`.")
    lines.append("")
    lines.append("## Key Takeaways")
    lines.append("")
    if top3:
        lines.append(f"- Best Macro F1: **{top3[0]['model']}** ({top3[0]['f1']:.4f})")
    
    if records:
        best_acc = max(records, key=lambda x: x['accuracy'])
        worse_err = max(records, key=lambda x: x['errors'])
        lines.append(f"- Best Accuracy: **{best_acc['model']}** ({best_acc['accuracy']:.4f})")
        lines.append(f"- Highest reliability risk (execution errors): **{worse_err['model']}** ({worse_err['errors']})")
    
    lines.append("")
    lines.append("## Charts")
    lines.append("")
    lines.append("### 1) Leaderboard by Macro F1 + Accuracy")
    lines.append("![Leaderboard](visuals/01_leaderboard_f1_accuracy.png)")
    lines.append("")
    lines.append("### 2) Accuracy vs Macro F1 (Bubble Size = Errors)")
    lines.append("![Tradeoff](visuals/02_quality_tradeoff_scatter.png)")
    lines.append("")
    lines.append("### 3) Execution Error Risk")
    lines.append("![Reliability](visuals/03_reliability_execution_errors.png)")
    lines.append("")
    lines.append("### 4) Cost vs F1 Tradeoff")
    lines.append("![Cost VS F1](visuals/04_cost_vs_f1_scatter.png)")
    lines.append("")
    lines.append("### 5) Composite Prioritization")
    lines.append("![Composite](visuals/04_composite_priority.png)")
    lines.append("")
    if risky:
        lines.append("## Models with Non-Zero Execution Errors")
        lines.append("")
        for r in sorted(risky, key=lambda x: x["errors"], reverse=True):
            lines.append(f"- {r['model']}: {r['errors']}")
        lines.append("")

    try:
        (run_dir / "visuals_summary.md").write_text("\n".join(lines), encoding="utf-8")
    except Exception as e:
        logging.error(f"Error saving markdown summary: {e}")

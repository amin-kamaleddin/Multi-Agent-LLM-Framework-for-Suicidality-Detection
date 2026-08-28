from __future__ import annotations

from pathlib import Path
import hashlib
import json
import platform
from datetime import datetime, timezone

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# FINAL SIX-SYSTEM PUBLICATION FIGURES
# ============================================================
#
# This script performs NO model training and NO LLM inference.
# It reads only the outputs produced by:
#
#   scripts/04_compare_six_systems_user_disjoint.py
#
# All uncertainty intervals shown here come from the paired
# user-cluster bootstrap produced by that analysis.
#
# Terminology intentionally uses:
#   "test partition"
#   "user-disjoint test partition"
#
# and avoids "held-out" terminology.
# ============================================================


# ============================================================
# SETTINGS
# ============================================================

COMPARISON_ROOT = Path(
    "results/six_system_user_disjoint_comparison"
)

TABLE_DIR = (
    COMPARISON_ROOT
    / "tables"
)

STATS_DIR = (
    COMPARISON_ROOT
    / "stats"
)

CM_DIR = (
    COMPARISON_ROOT
    / "confusion_matrices"
)

PROVENANCE_DIR = (
    COMPARISON_ROOT
    / "provenance"
)

FIGURE_DIR = (
    COMPARISON_ROOT
    / "figures"
)

FIGURE_DATA_DIR = (
    FIGURE_DIR
    / "figure_data"
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FIGURE_DATA_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


SUMMARY_FILE = (
    TABLE_DIR
    / "six_system_summary_with_ci.csv"
)

PER_CLASS_FILE = (
    TABLE_DIR
    / "per_class_metrics_long.csv"
)

PAIRWISE_BOOTSTRAP_FILE = (
    STATS_DIR
    / "pairwise_user_cluster_bootstrap.csv"
)

SYSTEM_CI_FILE = (
    STATS_DIR
    / "system_user_cluster_bootstrap_ci.csv"
)

MCNEMAR_FILE = (
    STATS_DIR
    / "pairwise_mcnemar_secondary.csv"
)

ANALYSIS_PROVENANCE_FILE = (
    PROVENANCE_DIR
    / "analysis_provenance.json"
)


SYSTEM_ORDER = [
    "single_agent_luna",
    "framework1",
    "framework2",
    "framework3",
    "framework4",
    "mentalroberta_ft",
]


SYSTEM_SHORT_NAMES = {
    "single_agent_luna":
        "Single Agent",

    "framework1":
        "Framework 1",

    "framework2":
        "Framework 2",

    "framework3":
        "Framework 3",

    "framework4":
        "Framework 4",

    "mentalroberta_ft":
        "MentalRoBERTa",
}


SYSTEM_LONG_NAMES = {
    "single_agent_luna":
        "Single-Agent Luna",

    "framework1":
        "Framework 1 (Parallel, No Evidence)",

    "framework2":
        "Framework 2 (Parallel + Evidence)",

    "framework3":
        "Framework 3 (Hierarchical, No Reviewer)",

    "framework4":
        "Framework 4 (Hierarchical + Reviewer)",

    "mentalroberta_ft":
        "Fine-Tuned MentalRoBERTa",
}


LABEL_ORDER = [
    "Unsupportive",
    "Indicator",
    "Ideation",
    "Behavior",
    "Attempt",
]


PRIMARY_METRICS = {
    "five_class_macro_f1":
        "Five-class Macro F1",

    "positive_only_macro_f1":
        "Positive-only Macro F1",

    "binary_recall":
        "Binary Recall",

    "binary_f1":
        "Binary F1",
}


# ============================================================
# JOURNAL-STYLE MATPLOTLIB DEFAULTS
# ============================================================

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 600,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})


# ============================================================
# HELPERS
# ============================================================

def sha256_file(
    path: Path,
) -> str:

    digest = hashlib.sha256()

    with open(
        path,
        "rb",
    ) as handle:

        for chunk in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):

            digest.update(
                chunk
            )

    return digest.hexdigest()


def save_figure(
    figure,
    stem: str,
):

    """
    Save every publication figure in:
      - PNG (600 dpi)
      - PDF (vector)
      - SVG (vector)
    """

    png_path = (
        FIGURE_DIR
        / f"{stem}.png"
    )

    pdf_path = (
        FIGURE_DIR
        / f"{stem}.pdf"
    )

    svg_path = (
        FIGURE_DIR
        / f"{stem}.svg"
    )

    figure.savefig(
        png_path,
        bbox_inches="tight",
        dpi=600,
    )

    figure.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    figure.savefig(
        svg_path,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    return [
        png_path,
        pdf_path,
        svg_path,
    ]


def require_columns(
    dataframe,
    required,
    name,
):

    missing = (
        set(
            required
        )
        - set(
            dataframe.columns
        )
    )

    if missing:

        raise ValueError(
            f"{name} is missing required columns: "
            f"{sorted(missing)}"
        )


def ordered_system_frame(
    dataframe,
):

    frame = (
        dataframe
        .copy()
    )

    frame[
        "system"
    ] = pd.Categorical(
        frame[
            "system"
        ],
        categories=SYSTEM_ORDER,
        ordered=True,
    )

    return (
        frame
        .sort_values(
            "system"
        )
        .reset_index(
            drop=True
        )
    )


def metric_ci_frame(
    system_ci_df,
    metric_name,
):

    frame = (
        system_ci_df[
            system_ci_df[
                "metric"
            ]
            == metric_name
        ]
        .copy()
    )

    frame = ordered_system_frame(
        frame
    )

    frame[
        "short_name"
    ] = (
        frame[
            "system"
        ]
        .astype(
            object
        )
        .map(
            SYSTEM_SHORT_NAMES
        )
    )

    return frame


def draw_point_ci(
    axis,
    values,
    lower,
    upper,
    labels,
    title,
    x_label,
    reference=None,
):

    y_positions = np.arange(
        len(
            values
        )
    )

    lower_error = (
        values
        - lower
    )

    upper_error = (
        upper
        - values
    )

    axis.errorbar(
        values,
        y_positions,
        xerr=np.vstack(
            [
                lower_error,
                upper_error,
            ]
        ),
        fmt="o",
        capsize=4,
        markersize=6,
        linewidth=1.2,
    )

    axis.set_yticks(
        y_positions
    )

    axis.set_yticklabels(
        labels
    )

    axis.invert_yaxis()

    axis.set_xlabel(
        x_label
    )

    axis.set_title(
        title
    )

    axis.grid(
        axis="x",
        alpha=0.25,
        linewidth=0.6,
    )

    if reference is not None:

        axis.axvline(
            reference,
            linestyle="--",
            linewidth=1,
        )


def annotate_points(
    axis,
    x_values,
    y_values,
    decimals=3,
):

    for x, y in zip(
        x_values,
        y_values,
    ):

        axis.annotate(
            f"{x:.{decimals}f}",
            (
                x,
                y,
            ),
            xytext=(
                5,
                0,
            ),
            textcoords="offset points",
            va="center",
            fontsize=8,
        )


# ============================================================
# LOAD / VALIDATE INPUTS
# ============================================================

required_files = [
    SUMMARY_FILE,
    PER_CLASS_FILE,
    PAIRWISE_BOOTSTRAP_FILE,
    SYSTEM_CI_FILE,
    MCNEMAR_FILE,
    ANALYSIS_PROVENANCE_FILE,
]


for path in required_files:

    if not path.exists():

        raise FileNotFoundError(
            f"Required analysis output not found:\n"
            f"{path.resolve()}\n\n"
            "Run the six-system comparison script first."
        )


summary_df = pd.read_csv(
    SUMMARY_FILE
)

per_class_df = pd.read_csv(
    PER_CLASS_FILE
)

pairwise_df = pd.read_csv(
    PAIRWISE_BOOTSTRAP_FILE
)

system_ci_df = pd.read_csv(
    SYSTEM_CI_FILE
)

mcnemar_df = pd.read_csv(
    MCNEMAR_FILE
)


with open(
    ANALYSIS_PROVENANCE_FILE,
    "r",
    encoding="utf-8",
) as handle:

    analysis_provenance = json.load(
        handle
    )


require_columns(
    summary_df,
    [
        "system",
        "five_class_accuracy",
        "five_class_macro_f1",
        "positive_only_macro_f1",
        "binary_recall",
        "binary_f1",
    ],
    "six_system_summary_with_ci.csv",
)


require_columns(
    per_class_df,
    [
        "system",
        "label",
        "precision",
        "recall",
        "f1",
        "support",
    ],
    "per_class_metrics_long.csv",
)


require_columns(
    pairwise_df,
    [
        "comparison",
        "system_a",
        "system_b",
        "metric",
        "observed_difference_b_minus_a",
        "ci_95_lower",
        "ci_95_upper",
        "ci_excludes_zero",
    ],
    "pairwise_user_cluster_bootstrap.csv",
)


require_columns(
    system_ci_df,
    [
        "system",
        "metric",
        "point_estimate",
        "ci_95_lower",
        "ci_95_upper",
    ],
    "system_user_cluster_bootstrap_ci.csv",
)


assert (
    set(
        summary_df[
            "system"
        ]
    )
    == set(
        SYSTEM_ORDER
    )
), (
    "Summary file does not contain exactly "
    "the expected six systems."
)


summary_df = ordered_system_frame(
    summary_df
)


# ============================================================
# FIGURE 1
# PRIMARY FIVE-CLASS MACRO F1 WITH 95% CI
# ============================================================

macro_frame = metric_ci_frame(
    system_ci_df,
    "five_class_macro_f1",
)


fig, ax = plt.subplots(
    figsize=(
        7.2,
        4.4,
    )
)


values = (
    macro_frame[
        "point_estimate"
    ]
    .to_numpy()
)

lower = (
    macro_frame[
        "ci_95_lower"
    ]
    .to_numpy()
)

upper = (
    macro_frame[
        "ci_95_upper"
    ]
    .to_numpy()
)


draw_point_ci(
    ax,
    values,
    lower,
    upper,
    macro_frame[
        "short_name"
    ],
    "Five-Class Macro F1 Across Six Systems",
    "Macro F1",
)


annotate_points(
    ax,
    values,
    np.arange(
        len(
            values
        )
    ),
)


fig.tight_layout()


save_figure(
    fig,
    "figure_01_five_class_macro_f1_ci",
)


macro_frame.to_csv(
    FIGURE_DATA_DIR
    / "figure_01_five_class_macro_f1_ci.csv",
    index=False,
)


# ============================================================
# FIGURE 2
# POSITIVE-ONLY MACRO F1 WITH 95% CI
# ============================================================

positive_frame = metric_ci_frame(
    system_ci_df,
    "positive_only_macro_f1",
)


fig, ax = plt.subplots(
    figsize=(
        7.2,
        4.4,
    )
)


values = (
    positive_frame[
        "point_estimate"
    ]
    .to_numpy()
)

lower = (
    positive_frame[
        "ci_95_lower"
    ]
    .to_numpy()
)

upper = (
    positive_frame[
        "ci_95_upper"
    ]
    .to_numpy()
)


draw_point_ci(
    ax,
    values,
    lower,
    upper,
    positive_frame[
        "short_name"
    ],
    "Positive-Only Macro F1 Across Six Systems",
    "Macro F1",
)


annotate_points(
    ax,
    values,
    np.arange(
        len(
            values
        )
    ),
)


fig.tight_layout()


save_figure(
    fig,
    "figure_02_positive_only_macro_f1_ci",
)


positive_frame.to_csv(
    FIGURE_DATA_DIR
    / "figure_02_positive_only_macro_f1_ci.csv",
    index=False,
)


# ============================================================
# FIGURE 3
# BINARY RECALL WITH 95% CI
# ============================================================

binary_recall_frame = metric_ci_frame(
    system_ci_df,
    "binary_recall",
)


fig, ax = plt.subplots(
    figsize=(
        7.2,
        4.4,
    )
)


values = (
    binary_recall_frame[
        "point_estimate"
    ]
    .to_numpy()
)

lower = (
    binary_recall_frame[
        "ci_95_lower"
    ]
    .to_numpy()
)

upper = (
    binary_recall_frame[
        "ci_95_upper"
    ]
    .to_numpy()
)


draw_point_ci(
    ax,
    values,
    lower,
    upper,
    binary_recall_frame[
        "short_name"
    ],
    "Binary Suicidal-Risk Recall Across Six Systems",
    "Recall",
)


annotate_points(
    ax,
    values,
    np.arange(
        len(
            values
        )
    ),
)


fig.tight_layout()


save_figure(
    fig,
    "figure_03_binary_recall_ci",
)


binary_recall_frame.to_csv(
    FIGURE_DATA_DIR
    / "figure_03_binary_recall_ci.csv",
    index=False,
)


# ============================================================
# FIGURE 4
# BINARY F1 WITH 95% CI
# ============================================================

binary_f1_frame = metric_ci_frame(
    system_ci_df,
    "binary_f1",
)


fig, ax = plt.subplots(
    figsize=(
        7.2,
        4.4,
    )
)


values = (
    binary_f1_frame[
        "point_estimate"
    ]
    .to_numpy()
)

lower = (
    binary_f1_frame[
        "ci_95_lower"
    ]
    .to_numpy()
)

upper = (
    binary_f1_frame[
        "ci_95_upper"
    ]
    .to_numpy()
)


draw_point_ci(
    ax,
    values,
    lower,
    upper,
    binary_f1_frame[
        "short_name"
    ],
    "Binary Suicidal-Risk F1 Across Six Systems",
    "F1",
)


annotate_points(
    ax,
    values,
    np.arange(
        len(
            values
        )
    ),
)


fig.tight_layout()


save_figure(
    fig,
    "figure_04_binary_f1_ci",
)


binary_f1_frame.to_csv(
    FIGURE_DATA_DIR
    / "figure_04_binary_f1_ci.csv",
    index=False,
)


# ============================================================
# FIGURE 5
# PRIMARY METRIC PANEL
# ============================================================

fig, axes = plt.subplots(
    nrows=2,
    ncols=2,
    figsize=(
        12,
        8,
    ),
)


panel_specs = [
    (
        axes[
            0,
            0,
        ],
        "five_class_macro_f1",
        "Five-Class Macro F1",
        "Macro F1",
    ),

    (
        axes[
            0,
            1,
        ],
        "positive_only_macro_f1",
        "Positive-Only Macro F1",
        "Macro F1",
    ),

    (
        axes[
            1,
            0,
        ],
        "binary_recall",
        "Binary Recall",
        "Recall",
    ),

    (
        axes[
            1,
            1,
        ],
        "binary_f1",
        "Binary F1",
        "F1",
    ),
]


for (
    axis,
    metric_name,
    title,
    x_label,
) in panel_specs:

    frame = metric_ci_frame(
        system_ci_df,
        metric_name,
    )

    values = (
        frame[
            "point_estimate"
        ]
        .to_numpy()
    )

    lower = (
        frame[
            "ci_95_lower"
        ]
        .to_numpy()
    )

    upper = (
        frame[
            "ci_95_upper"
        ]
        .to_numpy()
    )

    draw_point_ci(
        axis,
        values,
        lower,
        upper,
        frame[
            "short_name"
        ],
        title,
        x_label,
    )


fig.suptitle(
    "Primary Performance Metrics on the User-Disjoint Test Partition"
)


fig.tight_layout(
    rect=[
        0,
        0,
        1,
        0.96,
    ]
)


save_figure(
    fig,
    "figure_05_primary_metrics_panel",
)


# ============================================================
# FIGURE 6
# FIVE-CLASS PER-CLASS F1
#
# NOTE:
# Per-class confidence intervals were not produced by the
# six-system analysis script. This figure therefore shows
# point estimates only and does not fabricate uncertainty.
# ============================================================

per_class_plot = (
    per_class_df[
        [
            "system",
            "label",
            "f1",
        ]
    ]
    .copy()
)


per_class_plot[
    "system"
] = pd.Categorical(
    per_class_plot[
        "system"
    ],
    categories=SYSTEM_ORDER,
    ordered=True,
)


per_class_plot[
    "label"
] = pd.Categorical(
    per_class_plot[
        "label"
    ],
    categories=LABEL_ORDER,
    ordered=True,
)


per_class_plot = (
    per_class_plot
    .sort_values(
        [
            "label",
            "system",
        ]
    )
)


pivot_f1 = (
    per_class_plot
    .pivot(
        index="label",
        columns="system",
        values="f1",
    )
    .reindex(
        LABEL_ORDER
    )
    [
        SYSTEM_ORDER
    ]
)


fig, ax = plt.subplots(
    figsize=(
        10.5,
        5.6,
    )
)


x = np.arange(
    len(
        LABEL_ORDER
    )
)


n_systems = len(
    SYSTEM_ORDER
)


width = 0.12


offsets = (
    np.arange(
        n_systems
    )
    - (
        n_systems
        - 1
    )
    / 2
) * width


for index, system in enumerate(
    SYSTEM_ORDER
):

    ax.bar(
        x
        + offsets[
            index
        ],
        pivot_f1[
            system
        ]
        .to_numpy(),
        width=width,
        label=SYSTEM_SHORT_NAMES[
            system
        ],
    )


ax.set_xticks(
    x
)


ax.set_xticklabels(
    LABEL_ORDER
)


ax.set_ylabel(
    "F1"
)


ax.set_xlabel(
    "Class"
)


ax.set_title(
    "Per-Class F1 Across Six Systems"
)


ax.set_ylim(
    0,
    1.05,
)


ax.grid(
    axis="y",
    alpha=0.25,
    linewidth=0.6,
)


ax.legend(
    ncol=3,
    frameon=False,
)


fig.tight_layout()


save_figure(
    fig,
    "figure_06_per_class_f1",
)


pivot_f1.to_csv(
    FIGURE_DATA_DIR
    / "figure_06_per_class_f1.csv"
)


# ============================================================
# FIGURE 7
# MENTALROBERTA VS EACH LLM SYSTEM
# FOREST PLOT — FIVE-CLASS MACRO F1 DIFFERENCE
# ============================================================

mr_macro = (
    pairwise_df[
        (
            pairwise_df[
                "system_b"
            ]
            == "mentalroberta_ft"
        )
        &
        (
            pairwise_df[
                "metric"
            ]
            == "five_class_macro_f1"
        )
    ]
    .copy()
)


mr_macro[
    "system_a"
] = pd.Categorical(
    mr_macro[
        "system_a"
    ],
    categories=[
        "single_agent_luna",
        "framework1",
        "framework2",
        "framework3",
        "framework4",
    ],
    ordered=True,
)


mr_macro = (
    mr_macro
    .sort_values(
        "system_a"
    )
    .reset_index(
        drop=True
    )
)


mr_macro[
    "comparison_label"
] = (
    mr_macro[
        "system_a"
    ]
    .astype(
        object
    )
    .map(
        lambda x:
            f"MentalRoBERTa − "
            f"{SYSTEM_SHORT_NAMES[x]}"
    )
)


fig, ax = plt.subplots(
    figsize=(
        7.8,
        4.5,
    )
)


values = (
    mr_macro[
        "observed_difference_b_minus_a"
    ]
    .to_numpy()
)

lower = (
    mr_macro[
        "ci_95_lower"
    ]
    .to_numpy()
)

upper = (
    mr_macro[
        "ci_95_upper"
    ]
    .to_numpy()
)


draw_point_ci(
    ax,
    values,
    lower,
    upper,
    mr_macro[
        "comparison_label"
    ],
    "MentalRoBERTa Advantage in Five-Class Macro F1",
    "Difference in Macro F1",
    reference=0.0,
)


annotate_points(
    ax,
    values,
    np.arange(
        len(
            values
        )
    ),
)


fig.tight_layout()


save_figure(
    fig,
    "figure_07_mentalroberta_vs_llm_macro_f1_forest",
)


mr_macro.to_csv(
    FIGURE_DATA_DIR
    / "figure_07_mentalroberta_vs_llm_macro_f1_forest.csv",
    index=False,
)


# ============================================================
# FIGURE 8
# MENTALROBERTA VS EACH LLM SYSTEM
# FOREST PLOT — POSITIVE-ONLY MACRO F1 DIFFERENCE
# ============================================================

mr_positive = (
    pairwise_df[
        (
            pairwise_df[
                "system_b"
            ]
            == "mentalroberta_ft"
        )
        &
        (
            pairwise_df[
                "metric"
            ]
            == "positive_only_macro_f1"
        )
    ]
    .copy()
)


mr_positive[
    "system_a"
] = pd.Categorical(
    mr_positive[
        "system_a"
    ],
    categories=[
        "single_agent_luna",
        "framework1",
        "framework2",
        "framework3",
        "framework4",
    ],
    ordered=True,
)


mr_positive = (
    mr_positive
    .sort_values(
        "system_a"
    )
    .reset_index(
        drop=True
    )
)


mr_positive[
    "comparison_label"
] = (
    mr_positive[
        "system_a"
    ]
    .astype(
        object
    )
    .map(
        lambda x:
            f"MentalRoBERTa − "
            f"{SYSTEM_SHORT_NAMES[x]}"
    )
)


fig, ax = plt.subplots(
    figsize=(
        7.8,
        4.5,
    )
)


values = (
    mr_positive[
        "observed_difference_b_minus_a"
    ]
    .to_numpy()
)

lower = (
    mr_positive[
        "ci_95_lower"
    ]
    .to_numpy()
)

upper = (
    mr_positive[
        "ci_95_upper"
    ]
    .to_numpy()
)


draw_point_ci(
    ax,
    values,
    lower,
    upper,
    mr_positive[
        "comparison_label"
    ],
    "MentalRoBERTa Advantage in Positive-Only Macro F1",
    "Difference in Macro F1",
    reference=0.0,
)


annotate_points(
    ax,
    values,
    np.arange(
        len(
            values
        )
    ),
)


fig.tight_layout()


save_figure(
    fig,
    "figure_08_mentalroberta_vs_llm_positive_f1_forest",
)


mr_positive.to_csv(
    FIGURE_DATA_DIR
    / "figure_08_mentalroberta_vs_llm_positive_f1_forest.csv",
    index=False,
)


# ============================================================
# FIGURE 9
# ALL LLM-ONLY PAIRWISE FIVE-CLASS MACRO F1 DIFFERENCES
#
# This helps distinguish whether differences among Frameworks
# 1-4 and Single Agent are supported by the user-cluster
# bootstrap rather than relying only on ranking.
# ============================================================

llm_systems = [
    "single_agent_luna",
    "framework1",
    "framework2",
    "framework3",
    "framework4",
]


llm_pairwise = (
    pairwise_df[
        (
            pairwise_df[
                "system_a"
            ]
            .isin(
                llm_systems
            )
        )
        &
        (
            pairwise_df[
                "system_b"
            ]
            .isin(
                llm_systems
            )
        )
        &
        (
            pairwise_df[
                "metric"
            ]
            == "five_class_macro_f1"
        )
    ]
    .copy()
)


llm_pairwise[
    "label"
] = (
    llm_pairwise.apply(
        lambda row:
            (
                f"{SYSTEM_SHORT_NAMES[row['system_b']]} − "
                f"{SYSTEM_SHORT_NAMES[row['system_a']]}"
            ),
        axis=1,
    )
)


llm_pairwise = (
    llm_pairwise
    .sort_values(
        "observed_difference_b_minus_a"
    )
    .reset_index(
        drop=True
    )
)


fig, ax = plt.subplots(
    figsize=(
        8.5,
        6.5,
    )
)


values = (
    llm_pairwise[
        "observed_difference_b_minus_a"
    ]
    .to_numpy()
)

lower = (
    llm_pairwise[
        "ci_95_lower"
    ]
    .to_numpy()
)

upper = (
    llm_pairwise[
        "ci_95_upper"
    ]
    .to_numpy()
)


draw_point_ci(
    ax,
    values,
    lower,
    upper,
    llm_pairwise[
        "label"
    ],
    "Pairwise Five-Class Macro F1 Differences Among LLM Systems",
    "Difference in Macro F1",
    reference=0.0,
)


fig.tight_layout()


save_figure(
    fig,
    "figure_09_llm_pairwise_macro_f1_forest",
)


llm_pairwise.to_csv(
    FIGURE_DATA_DIR
    / "figure_09_llm_pairwise_macro_f1_forest.csv",
    index=False,
)


# ============================================================
# FIGURE 10
# OVERALL ACCURACY AND MACRO F1
#
# Compact descriptive figure. Confidence intervals are not
# added to accuracy here because the primary CI figures above
# already show bootstrap uncertainty explicitly.
# ============================================================

descriptive = (
    summary_df[
        [
            "system",
            "five_class_accuracy",
            "five_class_macro_f1",
        ]
    ]
    .copy()
)


descriptive[
    "short_name"
] = (
    descriptive[
        "system"
    ]
    .astype(
        object
    )
    .map(
        SYSTEM_SHORT_NAMES
    )
)


x = np.arange(
    len(
        descriptive
    )
)


width = 0.34


fig, ax = plt.subplots(
    figsize=(
        9.5,
        5.2,
    )
)


ax.bar(
    x
    - width / 2,
    descriptive[
        "five_class_accuracy"
    ],
    width,
    label="Accuracy",
)


ax.bar(
    x
    + width / 2,
    descriptive[
        "five_class_macro_f1"
    ],
    width,
    label="Macro F1",
)


ax.set_xticks(
    x
)


ax.set_xticklabels(
    descriptive[
        "short_name"
    ],
    rotation=25,
    ha="right",
)


ax.set_ylabel(
    "Score"
)


ax.set_ylim(
    0,
    1.0,
)


ax.set_title(
    "Five-Class Accuracy and Macro F1"
)


ax.grid(
    axis="y",
    alpha=0.25,
    linewidth=0.6,
)


ax.legend(
    frameon=False,
)


fig.tight_layout()


save_figure(
    fig,
    "figure_10_accuracy_and_macro_f1",
)


descriptive.to_csv(
    FIGURE_DATA_DIR
    / "figure_10_accuracy_and_macro_f1.csv",
    index=False,
)


# ============================================================
# FIGURES 11-16
# CONFUSION MATRICES — RAW COUNTS
# ============================================================

for figure_number, system in enumerate(
    SYSTEM_ORDER,
    start=11,
):

    path = (
        CM_DIR
        / (
            f"confusion_matrix_"
            f"{system}.csv"
        )
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Missing confusion matrix:\n"
            f"{path.resolve()}"
        )


    cm_df = pd.read_csv(
        path,
        index_col=0,
    )


    matrix = (
        cm_df
        .to_numpy(
            dtype=float
        )
    )


    fig, ax = plt.subplots(
        figsize=(
            6.6,
            5.7,
        )
    )


    image = ax.imshow(
        matrix,
        aspect="auto",
    )


    ax.set_xticks(
        np.arange(
            len(
                LABEL_ORDER
            )
        )
    )


    ax.set_yticks(
        np.arange(
            len(
                LABEL_ORDER
            )
        )
    )


    ax.set_xticklabels(
        LABEL_ORDER,
        rotation=35,
        ha="right",
    )


    ax.set_yticklabels(
        LABEL_ORDER
    )


    ax.set_xlabel(
        "Predicted label"
    )


    ax.set_ylabel(
        "True label"
    )


    ax.set_title(
        (
            f"Confusion Matrix — "
            f"{SYSTEM_SHORT_NAMES[system]}"
        )
    )


    for row_index in range(
        matrix.shape[
            0
        ]
    ):

        for column_index in range(
            matrix.shape[
                1
            ]
        ):

            ax.text(
                column_index,
                row_index,
                f"{int(matrix[row_index, column_index])}",
                ha="center",
                va="center",
                fontsize=8,
            )


    fig.colorbar(
        image,
        ax=ax,
        fraction=0.046,
        pad=0.04,
        label="Count",
    )


    fig.tight_layout()


    save_figure(
        fig,
        (
            f"figure_{figure_number:02d}_"
            f"confusion_matrix_counts_{system}"
        ),
    )


# ============================================================
# FIGURES 17-22
# CONFUSION MATRICES — ROW-NORMALIZED
#
# These are often easier to read in papers because each row
# shows the distribution of predictions conditional on the
# true class, reducing domination by the majority class.
# ============================================================

for figure_number, system in enumerate(
    SYSTEM_ORDER,
    start=17,
):

    path = (
        CM_DIR
        / (
            f"confusion_matrix_"
            f"{system}.csv"
        )
    )


    cm_df = pd.read_csv(
        path,
        index_col=0,
    )


    matrix = (
        cm_df
        .to_numpy(
            dtype=float
        )
    )


    row_sums = (
        matrix.sum(
            axis=1,
            keepdims=True,
        )
    )


    normalized = np.divide(
        matrix,
        row_sums,
        out=np.zeros_like(
            matrix,
            dtype=float,
        ),
        where=row_sums != 0,
    )


    normalized_df = pd.DataFrame(
        normalized,
        index=LABEL_ORDER,
        columns=LABEL_ORDER,
    )


    normalized_df.to_csv(
        FIGURE_DATA_DIR
        / (
            f"figure_{figure_number:02d}_"
            f"confusion_matrix_normalized_{system}.csv"
        )
    )


    fig, ax = plt.subplots(
        figsize=(
            6.6,
            5.7,
        )
    )


    image = ax.imshow(
        normalized,
        aspect="auto",
        vmin=0,
        vmax=1,
    )


    ax.set_xticks(
        np.arange(
            len(
                LABEL_ORDER
            )
        )
    )


    ax.set_yticks(
        np.arange(
            len(
                LABEL_ORDER
            )
        )
    )


    ax.set_xticklabels(
        LABEL_ORDER,
        rotation=35,
        ha="right",
    )


    ax.set_yticklabels(
        LABEL_ORDER
    )


    ax.set_xlabel(
        "Predicted label"
    )


    ax.set_ylabel(
        "True label"
    )


    ax.set_title(
        (
            "Row-Normalized Confusion Matrix — "
            f"{SYSTEM_SHORT_NAMES[system]}"
        )
    )


    for row_index in range(
        normalized.shape[
            0
        ]
    ):

        for column_index in range(
            normalized.shape[
                1
            ]
        ):

            ax.text(
                column_index,
                row_index,
                (
                    f"{normalized[row_index, column_index]:.2f}"
                ),
                ha="center",
                va="center",
                fontsize=8,
            )


    fig.colorbar(
        image,
        ax=ax,
        fraction=0.046,
        pad=0.04,
        label="Proportion within true class",
    )


    fig.tight_layout()


    save_figure(
        fig,
        (
            f"figure_{figure_number:02d}_"
            f"confusion_matrix_normalized_{system}"
        ),
    )


# ============================================================
# FIGURE 23
# BINARY ERROR COUNTS
# ============================================================

binary_counts = (
    summary_df[
        [
            "system",
            "binary_tn",
            "binary_fp",
            "binary_fn",
            "binary_tp",
        ]
    ]
    .copy()
)


binary_counts[
    "short_name"
] = (
    binary_counts[
        "system"
    ]
    .astype(
        object
    )
    .map(
        SYSTEM_SHORT_NAMES
    )
)


x = np.arange(
    len(
        binary_counts
    )
)


width = 0.18


fig, ax = plt.subplots(
    figsize=(
        10.2,
        5.4,
    )
)


series = [
    (
        "binary_tn",
        "True negatives",
    ),

    (
        "binary_fp",
        "False positives",
    ),

    (
        "binary_fn",
        "False negatives",
    ),

    (
        "binary_tp",
        "True positives",
    ),
]


offsets = [
    -1.5 * width,
    -0.5 * width,
    0.5 * width,
    1.5 * width,
]


for (
    column,
    label,
), offset in zip(
    series,
    offsets,
):

    ax.bar(
        x
        + offset,
        binary_counts[
            column
        ],
        width,
        label=label,
    )


ax.set_xticks(
    x
)


ax.set_xticklabels(
    binary_counts[
        "short_name"
    ],
    rotation=25,
    ha="right",
)


ax.set_ylabel(
    "Number of test posts"
)


ax.set_title(
    "Binary Positive-vs-Unsupportive Classification Counts"
)


ax.grid(
    axis="y",
    alpha=0.25,
    linewidth=0.6,
)


ax.legend(
    ncol=2,
    frameon=False,
)


fig.tight_layout()


save_figure(
    fig,
    "figure_23_binary_confusion_counts",
)


binary_counts.to_csv(
    FIGURE_DATA_DIR
    / "figure_23_binary_confusion_counts.csv",
    index=False,
)


# ============================================================
# FIGURE 24
# ERROR TRADE-OFF: BINARY FALSE NEGATIVES VS FALSE POSITIVES
#
# Useful clinically because false negatives and false positives
# have different practical implications.
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        7.0,
        5.4,
    )
)


ax.scatter(
    summary_df[
        "binary_fp"
    ],
    summary_df[
        "binary_fn"
    ],
    s=55,
)


for _, row in summary_df.iterrows():

    ax.annotate(
        SYSTEM_SHORT_NAMES[
            row[
                "system"
            ]
        ],
        (
            row[
                "binary_fp"
            ],
            row[
                "binary_fn"
            ],
        ),
        xytext=(
            5,
            5,
        ),
        textcoords="offset points",
        fontsize=8,
    )


ax.set_xlabel(
    "False positives"
)


ax.set_ylabel(
    "False negatives"
)


ax.set_title(
    "Binary Error Trade-Off Across Six Systems"
)


ax.grid(
    alpha=0.25,
    linewidth=0.6,
)


fig.tight_layout()


save_figure(
    fig,
    "figure_24_binary_error_tradeoff",
)


summary_df[
    [
        "system",
        "binary_fp",
        "binary_fn",
    ]
].to_csv(
    FIGURE_DATA_DIR
    / "figure_24_binary_error_tradeoff.csv",
    index=False,
)


# ============================================================
# FIGURE 25
# LLM SYSTEM RANKING WITH UNCERTAINTY
#
# Deliberately excludes MentalRoBERTa so smaller differences
# among the LLM architectures remain visually legible.
# ============================================================

llm_macro_frame = (
    macro_frame[
        macro_frame[
            "system"
        ]
        != "mentalroberta_ft"
    ]
    .copy()
)


llm_macro_frame = (
    llm_macro_frame
    .sort_values(
        "point_estimate"
    )
    .reset_index(
        drop=True
    )
)


fig, ax = plt.subplots(
    figsize=(
        7.0,
        4.2,
    )
)


values = (
    llm_macro_frame[
        "point_estimate"
    ]
    .to_numpy()
)

lower = (
    llm_macro_frame[
        "ci_95_lower"
    ]
    .to_numpy()
)

upper = (
    llm_macro_frame[
        "ci_95_upper"
    ]
    .to_numpy()
)


draw_point_ci(
    ax,
    values,
    lower,
    upper,
    llm_macro_frame[
        "short_name"
    ],
    "Five-Class Macro F1 Among LLM Systems",
    "Macro F1",
)


annotate_points(
    ax,
    values,
    np.arange(
        len(
            values
        )
    ),
)


fig.tight_layout()


save_figure(
    fig,
    "figure_25_llm_macro_f1_ci",
)


llm_macro_frame.to_csv(
    FIGURE_DATA_DIR
    / "figure_25_llm_macro_f1_ci.csv",
    index=False,
)


# ============================================================
# FIGURE 26
# MCNEMAR SECONDARY ANALYSIS — DISCORDANT CORRECTNESS COUNTS
#
# This is descriptive only. Primary inference remains the
# user-cluster bootstrap.
# ============================================================

mcnemar_plot = (
    mcnemar_df[
        [
            "comparison",
            "system_a",
            "system_b",
            "system_a_only_correct",
            "system_b_only_correct",
            "holm_adjusted_p_value",
        ]
    ]
    .copy()
)


mcnemar_plot[
    "label"
] = (
    mcnemar_plot.apply(
        lambda row:
            (
                f"{SYSTEM_SHORT_NAMES[row['system_b']]} vs "
                f"{SYSTEM_SHORT_NAMES[row['system_a']]}"
            ),
        axis=1,
    )
)


mcnemar_plot = (
    mcnemar_plot
    .sort_values(
        "system_b_only_correct"
    )
    .reset_index(
        drop=True
    )
)


y = np.arange(
    len(
        mcnemar_plot
    )
)


fig, ax = plt.subplots(
    figsize=(
        9.5,
        7.0,
    )
)


ax.barh(
    y,
    -mcnemar_plot[
        "system_a_only_correct"
    ],
    label="System A only correct",
)


ax.barh(
    y,
    mcnemar_plot[
        "system_b_only_correct"
    ],
    label="System B only correct",
)


ax.axvline(
    0,
    linewidth=1,
)


ax.set_yticks(
    y
)


ax.set_yticklabels(
    mcnemar_plot[
        "label"
    ]
)


ax.set_xlabel(
    "Discordant correctness count"
)


ax.set_title(
    "Pairwise Discordant Correctness Counts"
)


ax.legend(
    frameon=False,
)


ax.grid(
    axis="x",
    alpha=0.25,
    linewidth=0.6,
)


fig.tight_layout()


save_figure(
    fig,
    "figure_26_pairwise_discordant_correctness",
)


mcnemar_plot.to_csv(
    FIGURE_DATA_DIR
    / "figure_26_pairwise_discordant_correctness.csv",
    index=False,
)


# ============================================================
# FIGURE MANIFEST
# ============================================================

figure_files = sorted(
    [
        path
        for path in FIGURE_DIR.iterdir()
        if (
            path.is_file()
            and path.suffix.lower()
            in {
                ".png",
                ".pdf",
                ".svg",
            }
        )
    ]
)


manifest = {

    "figure_generation":
        (
            "Publication-style figures for six-system "
            "user-disjoint test-partition comparison"
        ),

    "created_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "terminology_note":
        (
            "Figures use 'test partition' and "
            "'user-disjoint test partition' terminology "
            "and intentionally avoid 'held-out'."
        ),

    "uncertainty_note":
        (
            "All displayed 95% confidence intervals are "
            "the percentile intervals from the paired "
            "user-cluster bootstrap produced by the "
            "six-system comparison analysis."
        ),

    "per_class_uncertainty_note":
        (
            "Per-class F1 figure contains point estimates "
            "only because the comparison analysis did not "
            "produce per-class bootstrap confidence intervals."
        ),

    "primary_inference":
        analysis_provenance.get(
            "primary_inference"
        ),

    "bootstrap_method":
        analysis_provenance.get(
            "bootstrap_method"
        ),

    "bootstrap_iterations":
        analysis_provenance.get(
            "bootstrap_iterations"
        ),

    "random_seed":
        analysis_provenance.get(
            "random_seed"
        ),

    "source_files": {

        "summary": {
            "path":
                str(
                    SUMMARY_FILE
                ),
            "sha256":
                sha256_file(
                    SUMMARY_FILE
                ),
        },

        "per_class": {
            "path":
                str(
                    PER_CLASS_FILE
                ),
            "sha256":
                sha256_file(
                    PER_CLASS_FILE
                ),
        },

        "system_ci": {
            "path":
                str(
                    SYSTEM_CI_FILE
                ),
            "sha256":
                sha256_file(
                    SYSTEM_CI_FILE
                ),
        },

        "pairwise_bootstrap": {
            "path":
                str(
                    PAIRWISE_BOOTSTRAP_FILE
                ),
            "sha256":
                sha256_file(
                    PAIRWISE_BOOTSTRAP_FILE
                ),
        },

        "mcnemar": {
            "path":
                str(
                    MCNEMAR_FILE
                ),
            "sha256":
                sha256_file(
                    MCNEMAR_FILE
                ),
        },
    },

    "software": {
        "python":
            platform.python_version(),

        "matplotlib":
            matplotlib.__version__,

        "numpy":
            np.__version__,

        "pandas":
            pd.__version__,
    },

    "output_formats": [
        "PNG (600 dpi)",
        "PDF (vector)",
        "SVG (vector)",
    ],

    "figures": [
        {
            "path":
                str(
                    path
                ),

            "sha256":
                sha256_file(
                    path
                ),
        }

        for path
        in figure_files
    ],
}


manifest_path = (
    FIGURE_DIR
    / "figure_manifest.json"
)


with open(
    manifest_path,
    "w",
    encoding="utf-8",
) as handle:

    json.dump(
        manifest,
        handle,
        indent=4,
        ensure_ascii=False,
    )


# ============================================================
# README FOR FIGURES
# ============================================================

readme_lines = [

    "# Publication Figures",

    "",

    (
        "These figures are generated entirely from the "
        "saved outputs of the six-system user-disjoint "
        "test-partition comparison."
    ),

    "",

    (
        "No model training or LLM inference is performed "
        "by this script."
    ),

    "",

    "## Confidence intervals",

    "",

    (
        "All plotted 95% confidence intervals are the "
        "paired user-cluster bootstrap percentile intervals "
        "produced by the six-system comparison analysis."
    ),

    "",

    (
        "The same sampled user clusters were used for every "
        "system within each bootstrap replicate."
    ),

    "",

    "## Main manuscript candidates",

    "",

    (
        "- Figure 01: Five-class Macro F1 with 95% CI"
    ),

    (
        "- Figure 02: Positive-only Macro F1 with 95% CI"
    ),

    (
        "- Figure 05: Four-panel primary-metrics summary"
    ),

    (
        "- Figure 06: Per-class F1 comparison"
    ),

    (
        "- Figure 07: MentalRoBERTa vs each LLM system, "
        "five-class Macro F1 differences with 95% CI"
    ),

    (
        "- Figure 08: MentalRoBERTa vs each LLM system, "
        "positive-only Macro F1 differences with 95% CI"
    ),

    (
        "- Figure 09: Pairwise LLM-system differences "
        "with 95% CI"
    ),

    "",

    "## Supplementary candidates",

    "",

    (
        "- Figures 11-16: Raw-count confusion matrices"
    ),

    (
        "- Figures 17-22: Row-normalized confusion matrices"
    ),

    (
        "- Figure 23: Binary confusion counts"
    ),

    (
        "- Figure 24: Binary false-positive/false-negative "
        "trade-off"
    ),

    (
        "- Figure 25: LLM-only Macro F1 with 95% CI"
    ),

    (
        "- Figure 26: Pairwise discordant correctness "
        "counts from the secondary McNemar analysis"
    ),

    "",

    (
        "PNG files are saved at 600 dpi. PDF and SVG "
        "versions are also produced for vector-quality "
        "journal submission and editing."
    ),
]


with open(
    FIGURE_DIR
    / "README.md",
    "w",
    encoding="utf-8",
) as handle:

    handle.write(
        "\n".join(
            readme_lines
        )
    )


# ============================================================
# FINAL
# ============================================================

print(
    "\n"
    + "=" * 88
)


print(
    "PUBLICATION FIGURE GENERATION COMPLETE"
)


print(
    "=" * 88
)


print(
    f"\nFigure directory:\n"
    f"{FIGURE_DIR.resolve()}"
)


print(
    "\nCreated:"
)


print(
    "- 26 numbered figure families"
)


print(
    "- PNG versions at 600 dpi"
)


print(
    "- PDF vector versions"
)


print(
    "- SVG vector versions"
)


print(
    "- figure_data/ CSV files for auditable plotting data"
)


print(
    "- figure_manifest.json with source/output hashes"
)


print(
    "- README.md with manuscript/supplement suggestions"
)


print(
    "\nNo model training or LLM inference "
    "was performed by this script."
)

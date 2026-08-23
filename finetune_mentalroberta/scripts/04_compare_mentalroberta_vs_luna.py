from pathlib import Path
import ast
import json

import numpy as np
import pandas as pd

from scipy.stats import binomtest
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


# ============================================================
# SETTINGS
# ============================================================

SEED = 42
N_BOOTSTRAP = 5000

TEST_FILE = Path(
    "data/splits/test.csv"
)

MENTAL_PREDICTIONS_FILE = Path(
    "results/test/test_predictions.csv"
)

LUNA_PREDICTIONS_FILE = Path(
    "../multi_agent_frameworks/"
    "logs/runs/"
    "20260822_144023_single_agent_luna_frozen_test/"
    "predictions.csv"
)

OUTPUT_DIR = Path(
    "results/comparison"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


ALL_LABELS = [
    "Unsupportive",
    "Indicator",
    "Ideation",
    "Behavior",
    "Attempt",
]

POSITIVE_LABELS = [
    "Indicator",
    "Ideation",
    "Behavior",
    "Attempt",
]


# ============================================================
# HELPERS
# ============================================================

def normalize_id(value):
    """
    Normalize IDs so values such as 49, 49.0,
    and '49' match correctly.
    """

    if pd.isna(value):
        return None

    value_str = str(value).strip()

    try:
        value_float = float(value_str)

        if value_float.is_integer():
            return str(int(value_float))

    except ValueError:
        pass

    return value_str


def normalize_label(value):
    """
    Normalize expected label strings.
    """

    if pd.isna(value):
        return None

    value = str(value).strip()

    lookup = {
        label.lower(): label
        for label in ALL_LABELS
    }

    return lookup.get(
        value.lower(),
        value,
    )


def find_column(
    df,
    candidates,
    description,
):
    """
    Find the first matching column from a list
    of possible column names.
    """

    for candidate in candidates:
        if candidate in df.columns:
            return candidate

    lower_map = {
        column.lower(): column
        for column in df.columns
    }

    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[
                candidate.lower()
            ]

    raise ValueError(
        f"\nCould not identify {description}.\n"
        f"Available columns:\n"
        f"{list(df.columns)}"
    )


def extract_luna_prediction_column(
    luna_df,
):
    """
    Try common evaluator output schemas.
    """

    direct_candidates = [
        "predicted_label",
        "prediction",
        "label_name",
        "predicted_class",
        "output_label",
        "final_output.label_name",
        "final_label",
    ]

    for candidate in direct_candidates:
        if candidate in luna_df.columns:
            return (
                luna_df[candidate]
                .map(normalize_label)
            )

    # Case-insensitive search
    lower_map = {
        column.lower(): column
        for column in luna_df.columns
    }

    for candidate in direct_candidates:
        if candidate.lower() in lower_map:
            actual = lower_map[
                candidate.lower()
            ]

            return (
                luna_df[actual]
                .map(normalize_label)
            )

    # Fallback:
    # some CSVs may store final_output as JSON/dict text
    for candidate in [
        "final_output",
        "output",
        "result",
    ]:
        if candidate not in luna_df.columns:
            continue

        extracted = []

        for raw_value in luna_df[candidate]:

            label = None

            if isinstance(raw_value, dict):
                label = raw_value.get(
                    "label_name"
                )

            elif pd.notna(raw_value):

                text = str(
                    raw_value
                ).strip()

                try:
                    parsed = json.loads(
                        text
                    )

                except Exception:
                    try:
                        parsed = ast.literal_eval(
                            text
                        )

                    except Exception:
                        parsed = None

                if isinstance(
                    parsed,
                    dict,
                ):
                    label = parsed.get(
                        "label_name"
                    )

            extracted.append(
                normalize_label(label)
            )

        result = pd.Series(
            extracted,
            index=luna_df.index,
        )

        if result.notna().all():
            return result

    raise ValueError(
        "\nCould not automatically identify "
        "Luna's prediction column.\n\n"
        f"Available Luna columns:\n"
        f"{list(luna_df.columns)}"
    )


def calculate_metrics(
    y_true,
    y_pred,
    labels,
):
    """
    Aggregate classification metrics.
    """

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    (
        macro_precision,
        macro_recall,
        macro_f1,
        _,
    ) = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )

    (
        _,
        _,
        weighted_f1,
        _,
    ) = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="weighted",
        zero_division=0,
    )

    return {
        "accuracy": float(
            accuracy
        ),
        "macro_precision": float(
            macro_precision
        ),
        "macro_recall": float(
            macro_recall
        ),
        "macro_f1": float(
            macro_f1
        ),
        "weighted_f1": float(
            weighted_f1
        ),
    }


def calculate_per_class(
    y_true,
    y_pred,
):
    """
    Per-class precision, recall, F1, support.
    """

    (
        precision,
        recall,
        f1,
        support,
    ) = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=ALL_LABELS,
        average=None,
        zero_division=0,
    )

    rows = []

    for index, label in enumerate(
        ALL_LABELS
    ):
        rows.append(
            {
                "label": label,
                "precision": float(
                    precision[index]
                ),
                "recall": float(
                    recall[index]
                ),
                "f1": float(
                    f1[index]
                ),
                "support": int(
                    support[index]
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def bootstrap_metric_difference(
    y_true,
    pred_a,
    pred_b,
    metric_function,
    n_bootstrap=N_BOOTSTRAP,
    seed=SEED,
):
    """
    Paired, stratified bootstrap.

    Every bootstrap replicate samples the same
    observations for both classifiers.

    Sampling is stratified by ground-truth class
    so the original class distribution is
    preserved.
    """

    rng = np.random.default_rng(
        seed
    )

    y_true = np.asarray(
        y_true
    )

    pred_a = np.asarray(
        pred_a
    )

    pred_b = np.asarray(
        pred_b
    )

    unique_labels = np.unique(
        y_true
    )

    label_indices = {
        label: np.where(
            y_true == label
        )[0]
        for label in unique_labels
    }

    differences = np.empty(
        n_bootstrap,
        dtype=float,
    )

    for iteration in range(
        n_bootstrap
    ):

        sampled_indices = []

        for label in unique_labels:

            indices = label_indices[
                label
            ]

            sampled = rng.choice(
                indices,
                size=len(indices),
                replace=True,
            )

            sampled_indices.append(
                sampled
            )

        sampled_indices = np.concatenate(
            sampled_indices
        )

        true_sample = y_true[
            sampled_indices
        ]

        pred_a_sample = pred_a[
            sampled_indices
        ]

        pred_b_sample = pred_b[
            sampled_indices
        ]

        metric_a = metric_function(
            true_sample,
            pred_a_sample,
        )

        metric_b = metric_function(
            true_sample,
            pred_b_sample,
        )

        differences[
            iteration
        ] = (
            metric_a
            - metric_b
        )

    lower = np.percentile(
        differences,
        2.5,
    )

    upper = np.percentile(
        differences,
        97.5,
    )

    return {
        "mean_difference": float(
            differences.mean()
        ),
        "ci_95_lower": float(
            lower
        ),
        "ci_95_upper": float(
            upper
        ),
        "bootstrap_iterations": int(
            n_bootstrap
        ),
        "differences": differences,
    }


# ============================================================
# VERIFY FILES
# ============================================================

print("\n" + "=" * 72)
print("MENTALROBERTA VS GPT-5.6 LUNA")
print("PAIRED HELD-OUT TEST COMPARISON")
print("=" * 72)


for path in [
    TEST_FILE,
    MENTAL_PREDICTIONS_FILE,
    LUNA_PREDICTIONS_FILE,
]:

    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n"
            f"{path.resolve()}"
        )


# ============================================================
# LOAD CANONICAL TEST DATA
# ============================================================

print(
    "\nLoading canonical held-out test set..."
)

test_df = pd.read_csv(
    TEST_FILE
)

assert len(test_df) == 4319, (
    f"Expected 4,319 test rows, "
    f"found {len(test_df):,}."
)

test_df["sample_id"] = (
    test_df["ID"]
    .map(normalize_id)
)

test_df["true_label"] = (
    test_df["sentiment"]
    .map(normalize_label)
)

assert (
    test_df["sample_id"]
    .notna()
    .all()
)

assert (
    test_df["sample_id"]
    .is_unique
)

assert set(
    test_df["true_label"]
) <= set(
    ALL_LABELS
)


# ============================================================
# LOAD MENTALROBERTA PREDICTIONS
# ============================================================

print(
    "Loading MentalRoBERTa predictions..."
)

mental_df = pd.read_csv(
    MENTAL_PREDICTIONS_FILE
)

mental_id_column = find_column(
    mental_df,
    [
        "ID",
        "sample_id",
        "id",
    ],
    "MentalRoBERTa ID column",
)

mental_prediction_column = (
    find_column(
        mental_df,
        [
            "predicted_label",
            "prediction",
            "predicted_class",
        ],
        "MentalRoBERTa prediction column",
    )
)

mental_compact = pd.DataFrame(
    {
        "sample_id": (
            mental_df[
                mental_id_column
            ]
            .map(normalize_id)
        ),

        "mentalroberta_prediction": (
            mental_df[
                mental_prediction_column
            ]
            .map(normalize_label)
        ),
    }
)


# ============================================================
# LOAD LUNA PREDICTIONS
# ============================================================

print(
    "Loading Luna predictions..."
)

luna_df = pd.read_csv(
    LUNA_PREDICTIONS_FILE
)

luna_id_column = find_column(
    luna_df,
    [
        "sample_id",
        "ID",
        "id",
    ],
    "Luna sample ID column",
)

luna_predictions = (
    extract_luna_prediction_column(
        luna_df
    )
)

luna_compact = pd.DataFrame(
    {
        "sample_id": (
            luna_df[
                luna_id_column
            ]
            .map(normalize_id)
        ),

        "luna_prediction": (
            luna_predictions
        ),
    }
)


# ============================================================
# VALIDATE PREDICTIONS
# ============================================================

for name, dataframe, prediction_column in [
    (
        "MentalRoBERTa",
        mental_compact,
        "mentalroberta_prediction",
    ),
    (
        "Luna",
        luna_compact,
        "luna_prediction",
    ),
]:

    assert (
        dataframe["sample_id"]
        .notna()
        .all()
    ), (
        f"{name} contains missing IDs."
    )

    assert (
        dataframe["sample_id"]
        .is_unique
    ), (
        f"{name} contains duplicate IDs."
    )

    unknown_predictions = (
        set(
            dataframe[
                prediction_column
            ].dropna()
        )
        - set(ALL_LABELS)
    )

    assert not unknown_predictions, (
        f"{name} contains unexpected labels: "
        f"{unknown_predictions}"
    )

    assert (
        dataframe[
            prediction_column
        ]
        .notna()
        .all()
    ), (
        f"{name} contains missing predictions."
    )


# ============================================================
# ALIGN BY EXACT SAMPLE ID
# ============================================================

paired_df = (
    test_df[
        [
            "sample_id",
            "true_label",
            "text",
        ]
    ]
    .merge(
        mental_compact,
        on="sample_id",
        how="left",
        validate="one_to_one",
    )
    .merge(
        luna_compact,
        on="sample_id",
        how="left",
        validate="one_to_one",
    )
)


assert len(
    paired_df
) == 4319


assert (
    paired_df[
        "mentalroberta_prediction"
    ]
    .notna()
    .all()
), (
    "Some canonical test IDs are missing "
    "from MentalRoBERTa predictions."
)


assert (
    paired_df[
        "luna_prediction"
    ]
    .notna()
    .all()
), (
    "Some canonical test IDs are missing "
    "from Luna predictions."
)


mental_id_set = set(
    mental_compact["sample_id"]
)

luna_id_set = set(
    luna_compact["sample_id"]
)

canonical_id_set = set(
    paired_df["sample_id"]
)


assert (
    mental_id_set
    == canonical_id_set
), (
    "MentalRoBERTa IDs do not exactly match "
    "the canonical held-out test set."
)


assert (
    luna_id_set
    == canonical_id_set
), (
    "Luna IDs do not exactly match "
    "the canonical held-out test set."
)


print(
    "\nID alignment verified:"
)

print(
    "4,319 / 4,319 identical held-out "
    "samples found for both systems."
)


# ============================================================
# CORRECTNESS FLAGS
# ============================================================

paired_df[
    "mentalroberta_correct"
] = (
    paired_df[
        "mentalroberta_prediction"
    ]
    == paired_df[
        "true_label"
    ]
)


paired_df[
    "luna_correct"
] = (
    paired_df[
        "luna_prediction"
    ]
    == paired_df[
        "true_label"
    ]
)


paired_df[
    "systems_agree"
] = (
    paired_df[
        "mentalroberta_prediction"
    ]
    == paired_df[
        "luna_prediction"
    ]
)


# ============================================================
# FIVE-CLASS METRICS
# ============================================================

y_true = paired_df[
    "true_label"
].to_numpy()

mental_pred = paired_df[
    "mentalroberta_prediction"
].to_numpy()

luna_pred = paired_df[
    "luna_prediction"
].to_numpy()


mental_metrics = calculate_metrics(
    y_true,
    mental_pred,
    ALL_LABELS,
)

luna_metrics = calculate_metrics(
    y_true,
    luna_pred,
    ALL_LABELS,
)


# ============================================================
# POSITIVE-ONLY METRICS
# ============================================================

# IMPORTANT:
#
# Select by GROUND-TRUTH positive label.
#
# A positive case predicted Unsupportive remains
# an error and is NOT removed.

positive_mask = np.isin(
    y_true,
    POSITIVE_LABELS,
)

positive_true = y_true[
    positive_mask
]

mental_positive_pred = mental_pred[
    positive_mask
]

luna_positive_pred = luna_pred[
    positive_mask
]


assert len(
    positive_true
) == 2160


mental_positive_metrics = (
    calculate_metrics(
        positive_true,
        mental_positive_pred,
        POSITIVE_LABELS,
    )
)

luna_positive_metrics = (
    calculate_metrics(
        positive_true,
        luna_positive_pred,
        POSITIVE_LABELS,
    )
)


# ============================================================
# BINARY POSITIVE VS UNSUPPORTIVE
# ============================================================

def to_binary(labels):

    return np.where(
        np.asarray(labels)
        == "Unsupportive",
        0,
        1,
    )


binary_true = to_binary(
    y_true
)

mental_binary_pred = to_binary(
    mental_pred
)

luna_binary_pred = to_binary(
    luna_pred
)


def binary_metrics(
    true,
    pred,
):

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            true,
            pred,
            labels=[1],
            average="binary",
            pos_label=1,
            zero_division=0,
        )
    )

    return {
        "accuracy": float(
            accuracy_score(
                true,
                pred,
            )
        ),
        "precision": float(
            precision
        ),
        "recall": float(
            recall
        ),
        "f1": float(
            f1
        ),
    }


mental_binary_metrics = (
    binary_metrics(
        binary_true,
        mental_binary_pred,
    )
)

luna_binary_metrics = (
    binary_metrics(
        binary_true,
        luna_binary_pred,
    )
)


# ============================================================
# PER-CLASS METRICS
# ============================================================

mental_per_class = (
    calculate_per_class(
        y_true,
        mental_pred,
    )
)

luna_per_class = (
    calculate_per_class(
        y_true,
        luna_pred,
    )
)


per_class_comparison = (
    mental_per_class[
        [
            "label",
            "precision",
            "recall",
            "f1",
            "support",
        ]
    ]
    .rename(
        columns={
            "precision":
                "mentalroberta_precision",
            "recall":
                "mentalroberta_recall",
            "f1":
                "mentalroberta_f1",
        }
    )
    .merge(
        luna_per_class[
            [
                "label",
                "precision",
                "recall",
                "f1",
            ]
        ].rename(
            columns={
                "precision":
                    "luna_precision",
                "recall":
                    "luna_recall",
                "f1":
                    "luna_f1",
            }
        ),
        on="label",
        how="inner",
    )
)


per_class_comparison[
    "f1_difference_mental_minus_luna"
] = (
    per_class_comparison[
        "mentalroberta_f1"
    ]
    - per_class_comparison[
        "luna_f1"
    ]
)


# ============================================================
# MCNEMAR'S EXACT TEST
# ============================================================

mental_correct = paired_df[
    "mentalroberta_correct"
].to_numpy()

luna_correct = paired_df[
    "luna_correct"
].to_numpy()


both_correct = int(
    np.sum(
        mental_correct
        & luna_correct
    )
)

mental_only_correct = int(
    np.sum(
        mental_correct
        & ~luna_correct
    )
)

luna_only_correct = int(
    np.sum(
        ~mental_correct
        & luna_correct
    )
)

both_wrong = int(
    np.sum(
        ~mental_correct
        & ~luna_correct
    )
)


discordant_pairs = (
    mental_only_correct
    + luna_only_correct
)


if discordant_pairs == 0:

    mcnemar_p = 1.0

else:

    mcnemar_result = (
        binomtest(
            k=min(
                mental_only_correct,
                luna_only_correct,
            ),
            n=discordant_pairs,
            p=0.5,
            alternative="two-sided",
        )
    )

    mcnemar_p = float(
        mcnemar_result.pvalue
    )


# ============================================================
# PAIRED STRATIFIED BOOTSTRAP
# ============================================================

print(
    f"\nRunning "
    f"{N_BOOTSTRAP:,} paired bootstrap "
    f"replicates..."
)


def five_class_macro_f1(
    true,
    pred,
):

    return f1_score(
        true,
        pred,
        labels=ALL_LABELS,
        average="macro",
        zero_division=0,
    )


bootstrap_five_class = (
    bootstrap_metric_difference(
        y_true,
        mental_pred,
        luna_pred,
        metric_function=(
            five_class_macro_f1
        ),
    )
)


def positive_macro_f1(
    true,
    pred,
):

    return f1_score(
        true,
        pred,
        labels=POSITIVE_LABELS,
        average="macro",
        zero_division=0,
    )


bootstrap_positive = (
    bootstrap_metric_difference(
        positive_true,
        mental_positive_pred,
        luna_positive_pred,
        metric_function=(
            positive_macro_f1
        ),
        seed=SEED + 1,
    )
)


def binary_positive_f1(
    true,
    pred,
):

    return f1_score(
        true,
        pred,
        pos_label=1,
        average="binary",
        zero_division=0,
    )


bootstrap_binary = (
    bootstrap_metric_difference(
        binary_true,
        mental_binary_pred,
        luna_binary_pred,
        metric_function=(
            binary_positive_f1
        ),
        seed=SEED + 2,
    )
)


print(
    "Bootstrap complete."
)


# ============================================================
# CONFUSION MATRICES
# ============================================================

mental_cm = confusion_matrix(
    y_true,
    mental_pred,
    labels=ALL_LABELS,
)

luna_cm = confusion_matrix(
    y_true,
    luna_pred,
    labels=ALL_LABELS,
)


mental_cm_df = pd.DataFrame(
    mental_cm,
    index=[
        f"true_{label}"
        for label in ALL_LABELS
    ],
    columns=[
        f"pred_{label}"
        for label in ALL_LABELS
    ],
)


luna_cm_df = pd.DataFrame(
    luna_cm,
    index=[
        f"true_{label}"
        for label in ALL_LABELS
    ],
    columns=[
        f"pred_{label}"
        for label in ALL_LABELS
    ],
)


# ============================================================
# SUMMARY TABLE
# ============================================================

summary_rows = []


def add_comparison(
    section,
    metric,
    mental_value,
    luna_value,
):

    summary_rows.append(
        {
            "section": section,
            "metric": metric,
            "mentalroberta": (
                mental_value
            ),
            "luna": luna_value,
            "difference_mental_minus_luna":
                mental_value
                - luna_value,
        }
    )


for metric in [
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_f1",
]:

    add_comparison(
        "five_class",
        metric,
        mental_metrics[
            metric
        ],
        luna_metrics[
            metric
        ],
    )


for metric in [
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_f1",
]:

    add_comparison(
        "positive_only",
        metric,
        mental_positive_metrics[
            metric
        ],
        luna_positive_metrics[
            metric
        ],
    )


for metric in [
    "accuracy",
    "precision",
    "recall",
    "f1",
]:

    add_comparison(
        "binary_positive_vs_unsupportive",
        metric,
        mental_binary_metrics[
            metric
        ],
        luna_binary_metrics[
            metric
        ],
    )


summary_df = pd.DataFrame(
    summary_rows
)


# ============================================================
# SAVE OUTPUT FILES
# ============================================================

paired_predictions_path = (
    OUTPUT_DIR
    / "paired_predictions.csv"
)

summary_path = (
    OUTPUT_DIR
    / "model_comparison_summary.csv"
)

per_class_path = (
    OUTPUT_DIR
    / "per_class_comparison.csv"
)

mental_cm_path = (
    OUTPUT_DIR
    / "confusion_matrix_mentalroberta.csv"
)

luna_cm_path = (
    OUTPUT_DIR
    / "confusion_matrix_luna.csv"
)

statistics_path = (
    OUTPUT_DIR
    / "paired_statistical_tests.json"
)

bootstrap_path = (
    OUTPUT_DIR
    / "bootstrap_differences.csv"
)


paired_df.to_csv(
    paired_predictions_path,
    index=False,
)

summary_df.to_csv(
    summary_path,
    index=False,
)

per_class_comparison.to_csv(
    per_class_path,
    index=False,
)

mental_cm_df.to_csv(
    mental_cm_path
)

luna_cm_df.to_csv(
    luna_cm_path
)


bootstrap_df = pd.DataFrame(
    {
        "five_class_macro_f1_difference":
            bootstrap_five_class[
                "differences"
            ],

        "positive_only_macro_f1_difference":
            bootstrap_positive[
                "differences"
            ],

        "binary_f1_difference":
            bootstrap_binary[
                "differences"
            ],
    }
)

bootstrap_df.to_csv(
    bootstrap_path,
    index=False,
)


statistical_results = {
    "comparison": (
        "MentalRoBERTa minus GPT-5.6 Luna"
    ),

    "paired_samples": int(
        len(paired_df)
    ),

    "mcnemar_exact_test": {
        "both_correct": (
            both_correct
        ),
        "mentalroberta_correct_luna_wrong":
            mental_only_correct,
        "luna_correct_mentalroberta_wrong":
            luna_only_correct,
        "both_wrong": (
            both_wrong
        ),
        "discordant_pairs": (
            discordant_pairs
        ),
        "two_sided_exact_p_value": (
            mcnemar_p
        ),
    },

    "five_class_macro_f1": {
        "mentalroberta": (
            mental_metrics[
                "macro_f1"
            ]
        ),
        "luna": (
            luna_metrics[
                "macro_f1"
            ]
        ),
        "observed_difference": (
            mental_metrics[
                "macro_f1"
            ]
            - luna_metrics[
                "macro_f1"
            ]
        ),
        "bootstrap_95_ci": [
            bootstrap_five_class[
                "ci_95_lower"
            ],
            bootstrap_five_class[
                "ci_95_upper"
            ],
        ],
    },

    "positive_only_macro_f1": {
        "mentalroberta": (
            mental_positive_metrics[
                "macro_f1"
            ]
        ),
        "luna": (
            luna_positive_metrics[
                "macro_f1"
            ]
        ),
        "observed_difference": (
            mental_positive_metrics[
                "macro_f1"
            ]
            - luna_positive_metrics[
                "macro_f1"
            ]
        ),
        "bootstrap_95_ci": [
            bootstrap_positive[
                "ci_95_lower"
            ],
            bootstrap_positive[
                "ci_95_upper"
            ],
        ],
    },

    "binary_positive_f1": {
        "mentalroberta": (
            mental_binary_metrics[
                "f1"
            ]
        ),
        "luna": (
            luna_binary_metrics[
                "f1"
            ]
        ),
        "observed_difference": (
            mental_binary_metrics[
                "f1"
            ]
            - luna_binary_metrics[
                "f1"
            ]
        ),
        "bootstrap_95_ci": [
            bootstrap_binary[
                "ci_95_lower"
            ],
            bootstrap_binary[
                "ci_95_upper"
            ],
        ],
    },

    "bootstrap": {
        "iterations": (
            N_BOOTSTRAP
        ),
        "seed": SEED,
        "method": (
            "paired stratified bootstrap "
            "by ground-truth class"
        ),
    },
}


with open(
    statistics_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        statistical_results,
        f,
        indent=4,
    )


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 72)
print("FIVE-CLASS COMPARISON")
print("=" * 72)

print(
    f"{'Metric':<20}"
    f"{'MentalRoBERTa':>18}"
    f"{'Luna':>12}"
    f"{'Difference':>14}"
)

print("-" * 64)

for metric, display_name in [
    ("accuracy", "Accuracy"),
    ("macro_precision", "Macro Precision"),
    ("macro_recall", "Macro Recall"),
    ("macro_f1", "Macro F1"),
    ("weighted_f1", "Weighted F1"),
]:

    mental_value = (
        mental_metrics[
            metric
        ]
    )

    luna_value = (
        luna_metrics[
            metric
        ]
    )

    print(
        f"{display_name:<20}"
        f"{mental_value:>18.4f}"
        f"{luna_value:>12.4f}"
        f"{mental_value - luna_value:>14.4f}"
    )


print("\n" + "=" * 72)
print("POSITIVE-ONLY COMPARISON")
print("=" * 72)

print(
    f"MentalRoBERTa Macro F1: "
    f"{mental_positive_metrics['macro_f1']:.6f}"
)

print(
    f"Luna Macro F1:          "
    f"{luna_positive_metrics['macro_f1']:.6f}"
)

print(
    f"Difference:             "
    f"{mental_positive_metrics['macro_f1'] - luna_positive_metrics['macro_f1']:.6f}"
)


print("\n" + "=" * 72)
print("MCNEMAR EXACT TEST")
print("=" * 72)

print(
    f"Both correct:                  "
    f"{both_correct:,}"
)

print(
    f"MentalRoBERTa only correct:    "
    f"{mental_only_correct:,}"
)

print(
    f"Luna only correct:             "
    f"{luna_only_correct:,}"
)

print(
    f"Both wrong:                    "
    f"{both_wrong:,}"
)

print(
    f"Exact two-sided p-value:        "
    f"{mcnemar_p:.8g}"
)


print("\n" + "=" * 72)
print("PAIRED BOOTSTRAP — MACRO F1 DIFFERENCE")
print("=" * 72)

print(
    "Five-class:"
)

print(
    f"Observed difference: "
    f"{mental_metrics['macro_f1'] - luna_metrics['macro_f1']:.6f}"
)

print(
    f"95% CI: "
    f"[{bootstrap_five_class['ci_95_lower']:.6f}, "
    f"{bootstrap_five_class['ci_95_upper']:.6f}]"
)


print(
    "\nPositive-only:"
)

print(
    f"Observed difference: "
    f"{mental_positive_metrics['macro_f1'] - luna_positive_metrics['macro_f1']:.6f}"
)

print(
    f"95% CI: "
    f"[{bootstrap_positive['ci_95_lower']:.6f}, "
    f"{bootstrap_positive['ci_95_upper']:.6f}]"
)


print("\n" + "=" * 72)
print("PER-CLASS F1")
print("=" * 72)

print(
    per_class_comparison[
        [
            "label",
            "mentalroberta_f1",
            "luna_f1",
            "f1_difference_mental_minus_luna",
        ]
    ].to_string(
        index=False
    )
)


print("\n" + "=" * 72)
print("COMPARISON COMPLETE")
print("=" * 72)

print(
    f"\nOutputs saved to:\n"
    f"{OUTPUT_DIR.resolve()}"
)

print(
    "\nFiles:"
)

print(
    f"- {paired_predictions_path.name}"
)

print(
    f"- {summary_path.name}"
)

print(
    f"- {per_class_path.name}"
)

print(
    f"- {mental_cm_path.name}"
)

print(
    f"- {luna_cm_path.name}"
)

print(
    f"- {statistics_path.name}"
)

print(
    f"- {bootstrap_path.name}"
)

print(
    "\nNo models were trained or modified."
)
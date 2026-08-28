from __future__ import annotations

from pathlib import Path
import hashlib
import json
import platform
from datetime import datetime, timezone
from itertools import combinations

import numpy as np
import pandas as pd
import scipy
import sklearn

from scipy.stats import binomtest

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

SEED = 42

N_BOOTSTRAP = 5000


# ============================================================
# CANONICAL TEST PARTITION
# ============================================================

TEST_FILE = Path(
    "data/splits/test.csv"
)

SPLIT_MANIFEST_FILE = Path(
    "data/splits/split_manifest.json"
)


# ============================================================
# CANONICAL PUBLICATION ARTIFACT DIRECTORY
#
# IMPORTANT:
# All final system outputs used for publication analysis
# are loaded from the Git-tracked evaluation_results tree.
#
# The analysis intentionally does NOT read from:
#
#     multi_agent_frameworks/logs/runs/
#
# because that directory is operational/debugging output
# and may be excluded from Git.
# ============================================================

EVALUATION_RESULTS_DIR = Path(
    "../evaluation_results"
)


# ============================================================
# EXACT FINAL RUN DIRECTORIES
# ============================================================

SINGLE_AGENT_DIR = (
    EVALUATION_RESULTS_DIR
    / "20260827_194855_single_agent_luna_user_disjoint_test"
)

FRAMEWORK1_DIR = (
    EVALUATION_RESULTS_DIR
    / "20260827_200922_framework1_luna_user_disjoint_test"
)

FRAMEWORK2_DIR = (
    EVALUATION_RESULTS_DIR
    / "20260827_214548_framework2_luna_user_disjoint_test_rerun"
)

FRAMEWORK3_DIR = (
    EVALUATION_RESULTS_DIR
    / "20260827_230650_framework3_luna_user_disjoint_test"
)

FRAMEWORK4_DIR = (
    EVALUATION_RESULTS_DIR
    / "20260828_003121_framework4_luna_user_disjoint_test"
)

MENTALROBERTA_DIR = (
    EVALUATION_RESULTS_DIR
    / "mentalroberta"
)

# ============================================================
# FINAL PREDICTION FILES
# ============================================================

SINGLE_AGENT_FILE = SINGLE_AGENT_DIR / "predictions.csv"
FRAMEWORK1_FILE = FRAMEWORK1_DIR / "predictions.csv"
FRAMEWORK2_FILE = FRAMEWORK2_DIR / "predictions.csv"
FRAMEWORK3_FILE = FRAMEWORK3_DIR / "predictions.csv"
FRAMEWORK4_FILE = FRAMEWORK4_DIR / "predictions.csv"
MENTALROBERTA_FILE = MENTALROBERTA_DIR / "test_predictions.csv"


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

OUTPUT_ROOT = Path(
    "results/six_system_user_disjoint_comparison"
)


TABLE_DIR = (
    OUTPUT_ROOT
    / "tables"
)


STATS_DIR = (
    OUTPUT_ROOT
    / "stats"
)


CM_DIR = (
    OUTPUT_ROOT
    / "confusion_matrices"
)


PROVENANCE_DIR = (
    OUTPUT_ROOT
    / "provenance"
)


FIGURE_DIR = (
    OUTPUT_ROOT
    / "figures"
)


for directory in [
    OUTPUT_ROOT,
    TABLE_DIR,
    STATS_DIR,
    CM_DIR,
    PROVENANCE_DIR,
    FIGURE_DIR,
]:

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# LABEL DEFINITIONS
# ============================================================

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
# SYSTEM DEFINITIONS
# ============================================================

SYSTEMS = {

    "single_agent_luna": {

        "display_name":
            "Single-Agent Luna",

        "path":
            SINGLE_AGENT_FILE,

        "prediction_candidates": [
            "prediction",
            "predicted_label",
        ],

        "artifact_directory":
            SINGLE_AGENT_DIR,
    },


    "framework1": {

        "display_name":
            "Framework 1 "
            "(Parallel, No Evidence)",

        "path":
            FRAMEWORK1_FILE,

        "prediction_candidates": [
            "prediction",
            "predicted_label",
        ],

        "artifact_directory":
            FRAMEWORK1_DIR,
    },


    "framework2": {

        "display_name":
            "Framework 2 "
            "(Parallel + Evidence)",

        "path":
            FRAMEWORK2_FILE,

        "prediction_candidates": [
            "prediction",
            "predicted_label",
        ],

        "artifact_directory":
            FRAMEWORK2_DIR,
    },


    "framework3": {

        "display_name":
            "Framework 3 "
            "(Hierarchical, No Reviewer)",

        "path":
            FRAMEWORK3_FILE,

        "prediction_candidates": [
            "prediction",
            "predicted_label",
        ],

        "artifact_directory":
            FRAMEWORK3_DIR,
    },


    "framework4": {

        "display_name":
            "Framework 4 "
            "(Hierarchical + Reviewer)",

        "path":
            FRAMEWORK4_FILE,

        "prediction_candidates": [
            "prediction",
            "predicted_label",
        ],

        "artifact_directory":
            FRAMEWORK4_DIR,
    },


    "mentalroberta_ft": {

        "display_name":
            "Fine-Tuned MentalRoBERTa",

        "path":
            MENTALROBERTA_FILE,

        "prediction_candidates": [
            "predicted_label",
            "prediction",
        ],

        "artifact_directory":
            MENTALROBERTA_DIR,
    },
}


SYSTEM_ORDER = [
    "single_agent_luna",
    "framework1",
    "framework2",
    "framework3",
    "framework4",
    "mentalroberta_ft",
]


# Six systems produce 6 choose 2 = 15 pairwise comparisons.

PAIRWISE_COMPARISONS = list(
    combinations(
        SYSTEM_ORDER,
        2,
    )
)


# ============================================================
# GENERAL HELPERS
# ============================================================

def sha256_file(
    path: Path,
) -> str:

    sha256 = hashlib.sha256()

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

            sha256.update(
                chunk
            )

    return sha256.hexdigest()


def normalize_id(
    value,
):

    """
    Normalize IDs so values such as:

        49
        49.0
        "49"

    all resolve to the same string representation.
    """

    if pd.isna(
        value
    ):
        return None

    value_string = str(
        value
    ).strip()

    try:

        numeric = float(
            value_string
        )

        if numeric.is_integer():

            return str(
                int(
                    numeric
                )
            )

    except ValueError:
        pass

    return value_string


def normalize_label(
    value,
):

    """
    Normalize whitespace and capitalization
    while preserving the canonical label names.
    """

    if pd.isna(
        value
    ):
        return None

    value = str(
        value
    ).strip()

    lookup = {
        label.lower():
            label

        for label
        in ALL_LABELS
    }

    return lookup.get(
        value.lower(),
        value,
    )


def find_column(
    dataframe,
    candidates,
    description,
):

    """
    Locate a column using multiple possible
    schema names.
    """

    for candidate in candidates:

        if candidate in dataframe.columns:

            return candidate


    lower_map = {

        column.lower():
            column

        for column
        in dataframe.columns
    }


    for candidate in candidates:

        if (
            candidate.lower()
            in lower_map
        ):

            return lower_map[
                candidate.lower()
            ]


    raise ValueError(
        f"\nUnable to find {description}.\n"
        f"Available columns:\n"
        f"{list(dataframe.columns)}"
    )


def save_json(
    path: Path,
    payload,
):

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            payload,
            handle,
            indent=4,
            ensure_ascii=False,
        )


# ============================================================
# METRIC HELPERS
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
    labels,
):

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
        weighted_precision,
        weighted_recall,
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

        "accuracy":
            float(
                accuracy
            ),

        "macro_precision":
            float(
                macro_precision
            ),

        "macro_recall":
            float(
                macro_recall
            ),

        "macro_f1":
            float(
                macro_f1
            ),

        "weighted_precision":
            float(
                weighted_precision
            ),

        "weighted_recall":
            float(
                weighted_recall
            ),

        "weighted_f1":
            float(
                weighted_f1
            ),
    }


def calculate_binary_metrics(
    y_true,
    y_pred,
):

    """
    Collapse the five-class taxonomy into:

        0 = Unsupportive
        1 = Indicator / Ideation / Behavior / Attempt
    """

    true_binary = np.where(
        np.asarray(
            y_true
        )
        == "Unsupportive",
        0,
        1,
    )


    pred_binary = np.where(
        np.asarray(
            y_pred
        )
        == "Unsupportive",
        0,
        1,
    )


    (
        precision,
        recall,
        f1,
        _,
    ) = precision_recall_fscore_support(
        true_binary,
        pred_binary,
        average="binary",
        pos_label=1,
        zero_division=0,
    )


    cm = confusion_matrix(
        true_binary,
        pred_binary,
        labels=[
            0,
            1,
        ],
    )


    return {

        "accuracy":
            float(
                accuracy_score(
                    true_binary,
                    pred_binary,
                )
            ),

        "precision":
            float(
                precision
            ),

        "recall":
            float(
                recall
            ),

        "f1":
            float(
                f1
            ),

        "tn":
            int(
                cm[
                    0,
                    0,
                ]
            ),

        "fp":
            int(
                cm[
                    0,
                    1,
                ]
            ),

        "fn":
            int(
                cm[
                    1,
                    0,
                ]
            ),

        "tp":
            int(
                cm[
                    1,
                    1,
                ]
            ),
    }


def calculate_per_class_metrics(
    y_true,
    y_pred,
):

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

                "label":
                    label,

                "precision":
                    float(
                        precision[
                            index
                        ]
                    ),

                "recall":
                    float(
                        recall[
                            index
                        ]
                    ),

                "f1":
                    float(
                        f1[
                            index
                        ]
                    ),

                "support":
                    int(
                        support[
                            index
                        ]
                    ),
            }
        )


    return pd.DataFrame(
        rows
    )


# ============================================================
# MCNEMAR
# ============================================================

def exact_mcnemar(
    correct_a,
    correct_b,
):

    """
    Exact paired McNemar test.

    This is retained as a SECONDARY sensitivity
    analysis.

    The primary uncertainty analysis below uses
    user-cluster bootstrap because multiple posts
    may belong to the same test user.
    """

    correct_a = np.asarray(
        correct_a,
        dtype=bool,
    )


    correct_b = np.asarray(
        correct_b,
        dtype=bool,
    )


    both_correct = int(
        np.sum(
            correct_a
            & correct_b
        )
    )


    a_only = int(
        np.sum(
            correct_a
            & ~correct_b
        )
    )


    b_only = int(
        np.sum(
            ~correct_a
            & correct_b
        )
    )


    both_wrong = int(
        np.sum(
            ~correct_a
            & ~correct_b
        )
    )


    discordant = (
        a_only
        + b_only
    )


    if discordant == 0:

        p_value = 1.0

    else:

        p_value = float(
            binomtest(
                k=min(
                    a_only,
                    b_only,
                ),
                n=discordant,
                p=0.5,
                alternative="two-sided",
            ).pvalue
        )


    return {

        "both_correct":
            both_correct,

        "a_only_correct":
            a_only,

        "b_only_correct":
            b_only,

        "both_wrong":
            both_wrong,

        "discordant_pairs":
            discordant,

        "p_value":
            p_value,
    }


def holm_adjust(
    p_values,
):

    """
    Holm-Bonferroni multiplicity correction.
    """

    p_values = np.asarray(
        p_values,
        dtype=float,
    )


    number_of_tests = len(
        p_values
    )


    sorted_indices = np.argsort(
        p_values
    )


    adjusted = np.empty(
        number_of_tests,
        dtype=float,
    )


    running_max = 0.0


    for rank, index in enumerate(
        sorted_indices
    ):

        multiplier = (
            number_of_tests
            - rank
        )


        candidate = min(
            1.0,
            p_values[
                index
            ]
            * multiplier,
        )


        running_max = max(
            running_max,
            candidate,
        )


        adjusted[
            index
        ] = running_max


    return adjusted


# ============================================================
# USER-CLUSTER BOOTSTRAP
# ============================================================

def cluster_bootstrap_indices(
    users,
    rng,
):

    """
    Sample unique USERS with replacement.

    Every selected user contributes every post
    belonging to that user.

    If a user is sampled multiple times, all of
    that user's posts appear multiple times in
    the bootstrap replicate.
    """

    users = np.asarray(
        users
    )


    unique_users = np.unique(
        users
    )


    sampled_users = rng.choice(
        unique_users,
        size=len(
            unique_users
        ),
        replace=True,
    )


    indices_by_user = {

        user:
            np.where(
                users == user
            )[0]

        for user
        in unique_users
    }


    sampled_indices = np.concatenate(
        [
            indices_by_user[
                user
            ]

            for user
            in sampled_users
        ]
    )


    return sampled_indices


def metric_bundle(
    y_true,
    y_pred,
):

    """
    Metrics used for bootstrap uncertainty.
    """

    five = calculate_metrics(
        y_true,
        y_pred,
        ALL_LABELS,
    )


    positive_mask = np.isin(
        y_true,
        POSITIVE_LABELS,
    )


    positive = calculate_metrics(
        y_true[
            positive_mask
        ],
        y_pred[
            positive_mask
        ],
        POSITIVE_LABELS,
    )


    binary = calculate_binary_metrics(
        y_true,
        y_pred,
    )


    return {

        "five_class_macro_f1":
            five[
                "macro_f1"
            ],

        "five_class_accuracy":
            five[
                "accuracy"
            ],

        "positive_only_macro_f1":
            positive[
                "macro_f1"
            ],

        "binary_recall":
            binary[
                "recall"
            ],

        "binary_f1":
            binary[
                "f1"
            ],
    }


def user_cluster_bootstrap(
    y_true,
    users,
    predictions,
    n_bootstrap,
    seed,
):

    """
    Paired nonparametric user-cluster bootstrap.

    The same sampled users are used for ALL
    systems within each replicate.

    This preserves paired comparisons while
    accounting for repeated posts from individual
    users.
    """

    rng = np.random.default_rng(
        seed
    )


    y_true = np.asarray(
        y_true
    )


    users = np.asarray(
        users
    )


    predictions = {

        name:
            np.asarray(
                values
            )

        for name, values
        in predictions.items()
    }


    metric_names = [

        "five_class_macro_f1",

        "five_class_accuracy",

        "positive_only_macro_f1",

        "binary_recall",

        "binary_f1",
    ]


    system_store = {

        system: {

            metric:
                np.empty(
                    n_bootstrap,
                    dtype=float,
                )

            for metric
            in metric_names
        }

        for system
        in SYSTEM_ORDER
    }


    pair_store = {

        (
            f"{system_b}"
            f"_minus_"
            f"{system_a}"
        ): {

            metric:
                np.empty(
                    n_bootstrap,
                    dtype=float,
                )

            for metric
            in metric_names
        }

        for system_a, system_b
        in PAIRWISE_COMPARISONS
    }


    for iteration in range(
        n_bootstrap
    ):

        sampled_indices = (
            cluster_bootstrap_indices(
                users,
                rng,
            )
        )


        sampled_true = (
            y_true[
                sampled_indices
            ]
        )


        sampled_scores = {}


        for system in SYSTEM_ORDER:

            sampled_prediction = (
                predictions[
                    system
                ][
                    sampled_indices
                ]
            )


            scores = metric_bundle(
                sampled_true,
                sampled_prediction,
            )


            sampled_scores[
                system
            ] = scores


            for metric in metric_names:

                system_store[
                    system
                ][
                    metric
                ][
                    iteration
                ] = (
                    scores[
                        metric
                    ]
                )


        for (
            system_a,
            system_b,
        ) in PAIRWISE_COMPARISONS:

            key = (
                f"{system_b}"
                f"_minus_"
                f"{system_a}"
            )


            for metric in metric_names:

                pair_store[
                    key
                ][
                    metric
                ][
                    iteration
                ] = (

                    sampled_scores[
                        system_b
                    ][
                        metric
                    ]

                    -

                    sampled_scores[
                        system_a
                    ][
                        metric
                    ]
                )


    return (
        system_store,
        pair_store,
    )


def summarize_distribution(
    values,
):

    """
    Percentile bootstrap interval + bootstrap SE.
    """

    values = np.asarray(
        values,
        dtype=float,
    )


    return {

        "bootstrap_mean":
            float(
                np.mean(
                    values
                )
            ),

        "bootstrap_se":
            float(
                np.std(
                    values,
                    ddof=1,
                )
            ),

        "ci_95_lower":
            float(
                np.percentile(
                    values,
                    2.5,
                )
            ),

        "ci_95_upper":
            float(
                np.percentile(
                    values,
                    97.5,
                )
            ),

        "iterations":
            int(
                len(
                    values
                )
            ),
    }


# ============================================================
# PROVENANCE HELPERS
# ============================================================

def artifact_metadata(
    path: Path,
    artifact_type: str,
    system_name: str,
):

    return {

        "system":
            system_name,

        "artifact_type":
            artifact_type,

        "path":
            str(
                path
            ),

        "exists":
            bool(
                path.exists()
            ),

        "sha256":
            (
                sha256_file(
                    path
                )

                if (
                    path.exists()
                    and path.is_file()
                )

                else None
            ),
    }


def find_optional_artifact(
    directory: Path,
    candidate_names,
):

    """
    Return the first matching optional provenance
    artifact if one exists.

    This permits MentalRoBERTa and the LLM runner
    to retain their original artifact filenames.
    """

    for filename in candidate_names:

        candidate = (
            directory
            / filename
        )

        if candidate.exists():

            return candidate

    return None


# ============================================================
# START
# ============================================================

print(
    "\n"
    + "=" * 92
)

print(
    "SIX-SYSTEM USER-DISJOINT "
    "TEST-PARTITION COMPARISON"
)

print(
    "=" * 92
)


print(
    "\nSystems:"
)


for system_name in SYSTEM_ORDER:

    print(
        "- "
        + SYSTEMS[
            system_name
        ][
            "display_name"
        ]
    )


# ============================================================
# 1. VERIFY REQUIRED INPUTS
# ============================================================

required_files = [

    TEST_FILE,

    SPLIT_MANIFEST_FILE,

    SINGLE_AGENT_FILE,

    FRAMEWORK1_FILE,

    FRAMEWORK2_FILE,

    FRAMEWORK3_FILE,

    FRAMEWORK4_FILE,

    MENTALROBERTA_FILE,
]


for required_file in required_files:

    if not required_file.exists():

        raise FileNotFoundError(
            "\nRequired file not found:\n"
            f"{required_file.resolve()}\n\n"
            "This comparison intentionally reads "
            "from the Git-tracked canonical "
            "evaluation_results directory."
        )


print(
    "\nAll required files found."
)


# ============================================================
# 2. LOAD AND VERIFY SPLIT MANIFEST
# ============================================================

with open(
    SPLIT_MANIFEST_FILE,
    "r",
    encoding="utf-8",
) as handle:

    split_manifest = json.load(
        handle
    )


assert (
    split_manifest.get(
        "split_unit"
    )
    == "user"
), (
    "split_manifest.json does not describe "
    "a user-level split."
)


assert (
    split_manifest.get(
        "group_column"
    )
    == "users"
), (
    "split_manifest.json does not identify "
    "'users' as the grouping column."
)


expected_test_hash = (
    split_manifest
    .get(
        "output_file_sha256",
        {},
    )
    .get(
        "test.csv"
    )
)


if expected_test_hash is None:

    raise ValueError(
        "split_manifest.json does not contain "
        "the canonical SHA-256 for test.csv."
    )


actual_test_hash = (
    sha256_file(
        TEST_FILE
    )
)


assert (
    actual_test_hash
    == expected_test_hash
), (
    "test.csv does not match the canonical "
    "split manifest SHA-256."
)


expected_test_rows = int(
    split_manifest[
        "test_rows"
    ]
)


expected_test_users = int(
    split_manifest[
        "test_unique_users"
    ]
)


expected_positive_rows = int(
    split_manifest[
        "test_positive_rows"
    ]
)


# ============================================================
# 3. VERIFY ZERO CROSS-PARTITION USER OVERLAP
# ============================================================

overlap_fields = [

    "train_validation_user_overlap",

    "train_test_user_overlap",

    "validation_test_user_overlap",
]


for field in overlap_fields:

    value = split_manifest.get(
        field
    )

    if value is not None:

        assert (
            int(
                value
            )
            == 0
        ), (
            f"{field} is not zero."
        )


print(
    "Canonical split manifest verified."
)


# ============================================================
# 4. LOAD CANONICAL TEST DATA
# ============================================================

test_df = pd.read_csv(
    TEST_FILE
)


required_test_columns = {
    "ID",
    "users",
    "text",
    "sentiment",
}


missing_test_columns = (
    required_test_columns
    - set(
        test_df.columns
    )
)


assert (
    not missing_test_columns
), (
    "test.csv is missing required columns: "
    f"{missing_test_columns}"
)


assert (
    len(
        test_df
    )
    == expected_test_rows
), (
    f"Expected {expected_test_rows:,} test rows, "
    f"found {len(test_df):,}."
)


assert (
    test_df[
        "users"
    ]
    .nunique()
    == expected_test_users
), (
    "Unique test-user count differs from "
    "the split manifest."
)


assert (
    test_df[
        "ID"
    ]
    .is_unique
), (
    "Canonical test IDs are not unique."
)


assert (
    test_df[
        "ID"
    ]
    .notna()
    .all()
), (
    "Canonical test.csv contains missing IDs."
)


assert (
    test_df[
        "users"
    ]
    .notna()
    .all()
), (
    "Canonical test.csv contains missing users."
)


# Deterministic row order for analysis.

test_df = (
    test_df
    .sort_values(
        "ID",
        kind="stable",
    )
    .reset_index(
        drop=True
    )
)


test_df[
    "sample_id"
] = (
    test_df[
        "ID"
    ]
    .map(
        normalize_id
    )
)


test_df[
    "true_label"
] = (
    test_df[
        "sentiment"
    ]
    .map(
        normalize_label
    )
)


assert (
    test_df[
        "sample_id"
    ]
    .is_unique
)


assert (
    test_df[
        "sample_id"
    ]
    .notna()
    .all()
)


unexpected_truth_labels = (
    set(
        test_df[
            "true_label"
        ]
    )
    - set(
        ALL_LABELS
    )
)


assert (
    not unexpected_truth_labels
), (
    "Unexpected ground-truth labels found: "
    f"{unexpected_truth_labels}"
)


actual_positive_rows = int(
    test_df[
        "true_label"
    ]
    .isin(
        POSITIVE_LABELS
    )
    .sum()
)


assert (
    actual_positive_rows
    == expected_positive_rows
), (
    "Positive test-row count differs from "
    "the split manifest."
)


print(
    f"\nCanonical test partition:"
)


print(
    f"- Posts: "
    f"{len(test_df):,}"
)


print(
    f"- Unique users: "
    f"{test_df['users'].nunique():,}"
)


print(
    f"- Positive posts: "
    f"{actual_positive_rows:,}"
)


# ============================================================
# 5. LOAD ALL SIX SYSTEM PREDICTIONS
# ============================================================

system_compact_frames = {}

provenance_rows = []


provenance_rows.append(
    artifact_metadata(
        TEST_FILE,
        "canonical_test_partition",
        "shared",
    )
)


provenance_rows.append(
    artifact_metadata(
        SPLIT_MANIFEST_FILE,
        "split_manifest",
        "shared",
    )
)


for system_name in SYSTEM_ORDER:

    config = SYSTEMS[
        system_name
    ]


    prediction_path = (
        config[
            "path"
        ]
    )


    print(
        f"\nLoading "
        f"{config['display_name']}..."
    )


    dataframe = pd.read_csv(
        prediction_path
    )


    assert (
        len(
            dataframe
        )
        == expected_test_rows
    ), (
        f"{system_name} contains "
        f"{len(dataframe):,} rows; "
        f"expected {expected_test_rows:,}."
    )


    id_column = find_column(
        dataframe,
        [
            "sample_id",
            "ID",
            "id",
        ],
        f"{system_name} ID column",
    )


    prediction_column = find_column(
        dataframe,
        config[
            "prediction_candidates"
        ],
        f"{system_name} prediction column",
    )


    compact = pd.DataFrame(
        {

            "sample_id":
                dataframe[
                    id_column
                ]
                .map(
                    normalize_id
                ),

            f"{system_name}_prediction":
                dataframe[
                    prediction_column
                ]
                .map(
                    normalize_label
                ),
        }
    )


    # --------------------------------------------
    # Verify evaluator-reported truth when present.
    # --------------------------------------------

    truth_candidates = [
        "ground_truth",
        "true_label",
        "sentiment",
        "label",
    ]


    truth_column = None


    for candidate in truth_candidates:

        if candidate in dataframe.columns:

            truth_column = candidate

            break


    if truth_column is not None:

        compact[
            f"{system_name}_reported_truth"
        ] = (
            dataframe[
                truth_column
            ]
            .map(
                normalize_label
            )
        )


    assert (
        compact[
            "sample_id"
        ]
        .is_unique
    ), (
        f"{system_name} contains duplicate IDs."
    )


    assert (
        compact[
            "sample_id"
        ]
        .notna()
        .all()
    ), (
        f"{system_name} contains missing IDs."
    )


    prediction_values = set(
        compact[
            f"{system_name}_prediction"
        ]
        .dropna()
    )


    unexpected_predictions = (
        prediction_values
        - set(
            ALL_LABELS
        )
    )


    assert (
        not unexpected_predictions
    ), (
        f"{system_name} contains unexpected "
        f"prediction labels: "
        f"{unexpected_predictions}"
    )


    assert (
        compact[
            f"{system_name}_prediction"
        ]
        .notna()
        .all()
    ), (
        f"{system_name} contains missing predictions."
    )


    system_compact_frames[
        system_name
    ] = compact


    # --------------------------------------------
    # Prediction-file provenance.
    # --------------------------------------------

    provenance_rows.append(
        artifact_metadata(
            prediction_path,
            "predictions",
            system_name,
        )
    )


    # --------------------------------------------
    # Supporting run artifacts.
    #
    # These do not affect metric calculations but
    # their existence and hashes are recorded when
    # available for reproducibility/auditability.
    # --------------------------------------------

    artifact_directory = (
        config[
            "artifact_directory"
        ]
    )


    optional_artifact_groups = {

        "metrics": [
            "metrics.json",
            "test_metrics.json",
            "evaluation_metrics.json",
        ],

        "run_config": [
            "run_config.json",
            "evaluation_config.json",
        ],

        "run_manifest": [
            "manifest.json",
            "evaluation_manifest.json",
        ],

        "training_summary": [
            "training_summary.json",
        ],

        "dataset_audit": [
            "dataset_audit.json",
        ],

        "trace_summary": [
            "trace_summary.json",
        ],
    }


    for (
        artifact_type,
        candidate_names,
    ) in (
        optional_artifact_groups.items()
    ):

        found_path = (
            find_optional_artifact(
                artifact_directory,
                candidate_names,
            )
        )


        if found_path is not None:

            provenance_rows.append(
                artifact_metadata(
                    found_path,
                    artifact_type,
                    system_name,
                )
            )


provenance_df = pd.DataFrame(
    provenance_rows
)


# ============================================================
# 6. EXACT ID ALIGNMENT
# ============================================================

paired_df = (
    test_df[
        [
            "sample_id",
            "users",
            "true_label",
            "text",
        ]
    ]
    .copy()
)


for system_name in SYSTEM_ORDER:

    paired_df = (
        paired_df
        .merge(
            system_compact_frames[
                system_name
            ],
            on="sample_id",
            how="left",
            validate="one_to_one",
        )
    )


assert (
    len(
        paired_df
    )
    == expected_test_rows
)


canonical_ids = set(
    paired_df[
        "sample_id"
    ]
)


for system_name in SYSTEM_ORDER:

    system_ids = set(
        system_compact_frames[
            system_name
        ][
            "sample_id"
        ]
    )


    assert (
        system_ids
        == canonical_ids
    ), (
        f"{system_name} IDs do not exactly "
        "match canonical test.csv."
    )


    prediction_column = (
        f"{system_name}_prediction"
    )


    assert (
        paired_df[
            prediction_column
        ]
        .notna()
        .all()
    )


    reported_truth_column = (
        f"{system_name}_reported_truth"
    )


    if (
        reported_truth_column
        in paired_df.columns
    ):

        mismatch = (

            paired_df[
                reported_truth_column
            ]

            !=

            paired_df[
                "true_label"
            ]
        )


        assert (
            not mismatch.any()
        ), (
            f"{system_name}'s evaluator-reported "
            "ground truth differs from canonical "
            "test.csv."
        )


print(
    "\n"
    + "=" * 92
)


print(
    "ALIGNMENT VERIFIED"
)


print(
    "=" * 92
)


print(
    f"All six systems contain the exact same "
    f"{expected_test_rows:,} test records."
)


print(
    "All available evaluator-reported "
    "ground truth matches canonical test.csv."
)


# ============================================================
# 7. CORRECTNESS FLAGS
# ============================================================

for system_name in SYSTEM_ORDER:

    paired_df[
        f"{system_name}_correct"
    ] = (

        paired_df[
            f"{system_name}_prediction"
        ]

        ==

        paired_df[
            "true_label"
        ]
    )


# ============================================================
# 8. POINT ESTIMATES
# ============================================================

y_true = (
    paired_df[
        "true_label"
    ]
    .to_numpy()
)


users = (
    paired_df[
        "users"
    ]
    .to_numpy()
)


predictions = {

    system_name:
        paired_df[
            f"{system_name}_prediction"
        ]
        .to_numpy()

    for system_name
    in SYSTEM_ORDER
}


positive_mask = np.isin(
    y_true,
    POSITIVE_LABELS,
)


positive_true = (
    y_true[
        positive_mask
    ]
)


assert (
    len(
        positive_true
    )
    == expected_positive_rows
)


five_class_metrics = {}

positive_metrics = {}

binary_metrics = {}

per_class_frames = []


for system_name in SYSTEM_ORDER:

    prediction = predictions[
        system_name
    ]


    five_class_metrics[
        system_name
    ] = calculate_metrics(
        y_true,
        prediction,
        ALL_LABELS,
    )


    positive_metrics[
        system_name
    ] = calculate_metrics(
        positive_true,
        prediction[
            positive_mask
        ],
        POSITIVE_LABELS,
    )


    binary_metrics[
        system_name
    ] = calculate_binary_metrics(
        y_true,
        prediction,
    )


    per_class = (
        calculate_per_class_metrics(
            y_true,
            prediction,
        )
    )


    per_class.insert(
        0,
        "system",
        system_name,
    )


    per_class.insert(
        1,
        "display_name",
        SYSTEMS[
            system_name
        ][
            "display_name"
        ],
    )


    per_class_frames.append(
        per_class
    )


per_class_long_df = pd.concat(
    per_class_frames,
    ignore_index=True,
)


per_class_f1_df = (
    per_class_long_df
    .pivot(
        index="label",
        columns="system",
        values="f1",
    )
    .reindex(
        ALL_LABELS
    )
    .reset_index()
)


per_class_f1_df = (
    per_class_f1_df[
        [
            "label",
            *SYSTEM_ORDER,
        ]
    ]
)


# ============================================================
# 9. CONFUSION MATRICES
# ============================================================

for system_name in SYSTEM_ORDER:

    cm = confusion_matrix(
        y_true,
        predictions[
            system_name
        ],
        labels=ALL_LABELS,
    )


    cm_df = pd.DataFrame(
        cm,
        index=[
            f"true_{label}"
            for label
            in ALL_LABELS
        ],
        columns=[
            f"pred_{label}"
            for label
            in ALL_LABELS
        ],
    )


    cm_df.to_csv(
        CM_DIR
        / (
            f"confusion_matrix_"
            f"{system_name}.csv"
        )
    )


# ============================================================
# 10. USER-CLUSTER BOOTSTRAP
#
# PRIMARY INFERENTIAL ANALYSIS
# ============================================================

print(
    "\n"
    + "=" * 92
)


print(
    "PAIRED USER-CLUSTER BOOTSTRAP"
)


print(
    "=" * 92
)


print(
    f"\nReplicates: "
    f"{N_BOOTSTRAP:,}"
)


print(
    f"Seed: "
    f"{SEED}"
)


print(
    f"Unique test users: "
    f"{len(np.unique(users)):,}"
)


(
    system_bootstrap,
    pair_bootstrap,
) = user_cluster_bootstrap(

    y_true=y_true,

    users=users,

    predictions=predictions,

    n_bootstrap=N_BOOTSTRAP,

    seed=SEED,
)


print(
    "User-cluster bootstrap complete."
)


# ============================================================
# 11. OBSERVED METRIC LOOKUP
# ============================================================

metric_point_lookup = {

    system: {

        "five_class_macro_f1":
            five_class_metrics[
                system
            ][
                "macro_f1"
            ],

        "five_class_accuracy":
            five_class_metrics[
                system
            ][
                "accuracy"
            ],

        "positive_only_macro_f1":
            positive_metrics[
                system
            ][
                "macro_f1"
            ],

        "binary_recall":
            binary_metrics[
                system
            ][
                "recall"
            ],

        "binary_f1":
            binary_metrics[
                system
            ][
                "f1"
            ],
    }

    for system
    in SYSTEM_ORDER
}


# ============================================================
# 12. SYSTEM-LEVEL BOOTSTRAP CIs
# ============================================================

ci_rows = []


for system_name in SYSTEM_ORDER:

    for (
        metric_name,
        distribution,
    ) in (
        system_bootstrap[
            system_name
        ]
        .items()
    ):

        distribution_summary = (
            summarize_distribution(
                distribution
            )
        )


        ci_rows.append(
            {

                "system":
                    system_name,

                "display_name":
                    SYSTEMS[
                        system_name
                    ][
                        "display_name"
                    ],

                "metric":
                    metric_name,

                "point_estimate":
                    metric_point_lookup[
                        system_name
                    ][
                        metric_name
                    ],

                **distribution_summary,
            }
        )


system_ci_df = pd.DataFrame(
    ci_rows
)


# ============================================================
# 13. ALL 15 PAIRWISE USER-CLUSTER BOOTSTRAP
# ============================================================

pair_rows = []


for (
    system_a,
    system_b,
) in PAIRWISE_COMPARISONS:

    key = (
        f"{system_b}"
        f"_minus_"
        f"{system_a}"
    )


    for (
        metric_name,
        distribution,
    ) in (
        pair_bootstrap[
            key
        ]
        .items()
    ):

        distribution_summary = (
            summarize_distribution(
                distribution
            )
        )


        observed_difference = (

            metric_point_lookup[
                system_b
            ][
                metric_name
            ]

            -

            metric_point_lookup[
                system_a
            ][
                metric_name
            ]
        )


        pair_rows.append(
            {

                "comparison":
                    key,

                "system_a":
                    system_a,

                "system_b":
                    system_b,

                "metric":
                    metric_name,

                "observed_difference_b_minus_a":
                    observed_difference,

                **distribution_summary,

                "ci_excludes_zero":
                    bool(
                        (
                            distribution_summary[
                                "ci_95_lower"
                            ]
                            > 0
                        )
                        or
                        (
                            distribution_summary[
                                "ci_95_upper"
                            ]
                            < 0
                        )
                    ),
            }
        )


pairwise_cluster_bootstrap_df = (
    pd.DataFrame(
        pair_rows
    )
)


# ============================================================
# 14. EXACT MCNEMAR
#
# SECONDARY SENSITIVITY ANALYSIS
# ============================================================

mcnemar_rows = []


for (
    system_a,
    system_b,
) in PAIRWISE_COMPARISONS:

    result = exact_mcnemar(

        paired_df[
            f"{system_a}_correct"
        ],

        paired_df[
            f"{system_b}_correct"
        ],
    )


    mcnemar_rows.append(
        {

            "system_a":
                system_a,

            "system_b":
                system_b,

            "comparison":
                (
                    f"{system_b}"
                    f"_minus_"
                    f"{system_a}"
                ),

            "system_a_only_correct":
                result[
                    "a_only_correct"
                ],

            "system_b_only_correct":
                result[
                    "b_only_correct"
                ],

            "both_correct":
                result[
                    "both_correct"
                ],

            "both_wrong":
                result[
                    "both_wrong"
                ],

            "discordant_pairs":
                result[
                    "discordant_pairs"
                ],

            "raw_p_value":
                result[
                    "p_value"
                ],

            "accuracy_difference_b_minus_a":
                (
                    five_class_metrics[
                        system_b
                    ][
                        "accuracy"
                    ]

                    -

                    five_class_metrics[
                        system_a
                    ][
                        "accuracy"
                    ]
                ),
        }
    )


mcnemar_df = pd.DataFrame(
    mcnemar_rows
)


mcnemar_df[
    "holm_adjusted_p_value"
] = holm_adjust(
    mcnemar_df[
        "raw_p_value"
    ]
    .to_numpy()
)


mcnemar_df[
    "significant_after_holm_0_05"
] = (
    mcnemar_df[
        "holm_adjusted_p_value"
    ]
    < 0.05
)


mcnemar_df[
    "inference_note"
] = (
    "Secondary post-level sensitivity analysis. "
    "Paired user-cluster bootstrap confidence "
    "intervals are the primary inferential analysis."
)


# ============================================================
# 15. OVERALL SIX-SYSTEM SUMMARY
# ============================================================

summary_rows = []


for system_name in SYSTEM_ORDER:

    five = (
        five_class_metrics[
            system_name
        ]
    )


    positive = (
        positive_metrics[
            system_name
        ]
    )


    binary = (
        binary_metrics[
            system_name
        ]
    )


    summary_rows.append(
        {

            "system":
                system_name,

            "display_name":
                SYSTEMS[
                    system_name
                ][
                    "display_name"
                ],


            # --------------------------------------
            # Five-class
            # --------------------------------------

            "five_class_accuracy":
                five[
                    "accuracy"
                ],

            "five_class_macro_precision":
                five[
                    "macro_precision"
                ],

            "five_class_macro_recall":
                five[
                    "macro_recall"
                ],

            "five_class_macro_f1":
                five[
                    "macro_f1"
                ],

            "five_class_weighted_f1":
                five[
                    "weighted_f1"
                ],


            # --------------------------------------
            # Positive-only
            # --------------------------------------

            "positive_only_accuracy":
                positive[
                    "accuracy"
                ],

            "positive_only_macro_precision":
                positive[
                    "macro_precision"
                ],

            "positive_only_macro_recall":
                positive[
                    "macro_recall"
                ],

            "positive_only_macro_f1":
                positive[
                    "macro_f1"
                ],


            # --------------------------------------
            # Binary positive vs Unsupportive
            # --------------------------------------

            "binary_accuracy":
                binary[
                    "accuracy"
                ],

            "binary_precision":
                binary[
                    "precision"
                ],

            "binary_recall":
                binary[
                    "recall"
                ],

            "binary_f1":
                binary[
                    "f1"
                ],

            "binary_tn":
                binary[
                    "tn"
                ],

            "binary_fp":
                binary[
                    "fp"
                ],

            "binary_fn":
                binary[
                    "fn"
                ],

            "binary_tp":
                binary[
                    "tp"
                ],
        }
    )


summary_df = pd.DataFrame(
    summary_rows
)


# ============================================================
# 16. ADD BOOTSTRAP CIs TO SUMMARY
# ============================================================

for (
    metric_name,
    output_prefix,
) in [

    (
        "five_class_macro_f1",
        "five_class_macro_f1",
    ),

    (
        "positive_only_macro_f1",
        "positive_only_macro_f1",
    ),

    (
        "binary_recall",
        "binary_recall",
    ),

    (
        "binary_f1",
        "binary_f1",
    ),
]:

    metric_ci = (
        system_ci_df[
            system_ci_df[
                "metric"
            ]
            == metric_name
        ]
        [
            [
                "system",
                "bootstrap_se",
                "ci_95_lower",
                "ci_95_upper",
            ]
        ]
        .rename(
            columns={

                "bootstrap_se":
                    (
                        f"{output_prefix}"
                        "_bootstrap_se"
                    ),

                "ci_95_lower":
                    (
                        f"{output_prefix}"
                        "_ci95_lower"
                    ),

                "ci_95_upper":
                    (
                        f"{output_prefix}"
                        "_ci95_upper"
                    ),
            }
        )
    )


    summary_df = (
        summary_df
        .merge(
            metric_ci,
            on="system",
            how="left",
            validate="one_to_one",
        )
    )


# ============================================================
# 17. SAVE TABLES
# ============================================================

paired_predictions_path = (
    TABLE_DIR
    / "paired_predictions_six_systems.csv"
)


summary_path = (
    TABLE_DIR
    / "six_system_summary_with_ci.csv"
)


per_class_long_path = (
    TABLE_DIR
    / "per_class_metrics_long.csv"
)


per_class_f1_path = (
    TABLE_DIR
    / "per_class_f1_comparison.csv"
)


system_ci_path = (
    STATS_DIR
    / "system_user_cluster_bootstrap_ci.csv"
)


pairwise_bootstrap_path = (
    STATS_DIR
    / "pairwise_user_cluster_bootstrap.csv"
)


mcnemar_path = (
    STATS_DIR
    / "pairwise_mcnemar_secondary.csv"
)


provenance_path = (
    PROVENANCE_DIR
    / "input_provenance.csv"
)


master_json_path = (
    OUTPUT_ROOT
    / "six_system_results.json"
)


markdown_path = (
    OUTPUT_ROOT
    / "six_system_comparison.md"
)


paired_df.to_csv(
    paired_predictions_path,
    index=False,
)


summary_df.to_csv(
    summary_path,
    index=False,
)


per_class_long_df.to_csv(
    per_class_long_path,
    index=False,
)


per_class_f1_df.to_csv(
    per_class_f1_path,
    index=False,
)


system_ci_df.to_csv(
    system_ci_path,
    index=False,
)


pairwise_cluster_bootstrap_df.to_csv(
    pairwise_bootstrap_path,
    index=False,
)


mcnemar_df.to_csv(
    mcnemar_path,
    index=False,
)


provenance_df.to_csv(
    provenance_path,
    index=False,
)


# ============================================================
# 18. SAVE RAW BOOTSTRAP DISTRIBUTIONS
# ============================================================

bootstrap_payload = {}


for (
    system_name,
    metrics,
) in system_bootstrap.items():

    for (
        metric_name,
        values,
    ) in metrics.items():

        bootstrap_payload[
            (
                f"system__"
                f"{system_name}__"
                f"{metric_name}"
            )
        ] = values


for (
    comparison_name,
    metrics,
) in pair_bootstrap.items():

    for (
        metric_name,
        values,
    ) in metrics.items():

        bootstrap_payload[
            (
                f"pair__"
                f"{comparison_name}__"
                f"{metric_name}"
            )
        ] = values


raw_bootstrap_path = (
    STATS_DIR
    / "raw_user_cluster_bootstrap.npz"
)


np.savez_compressed(
    raw_bootstrap_path,
    **bootstrap_payload,
)


# ============================================================
# 19. ANALYSIS PROVENANCE
# ============================================================

analysis_provenance = {

    "analysis_name":
        (
            "Six-system user-disjoint "
            "test-partition comparison"
        ),

    "created_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "random_seed":
        SEED,

    "bootstrap_iterations":
        N_BOOTSTRAP,

    "bootstrap_method":
        (
            "Paired nonparametric user-cluster bootstrap. "
            "Unique test users are sampled with replacement. "
            "All posts belonging to each sampled user are "
            "included. Identical sampled user clusters are "
            "used for all systems within each replicate."
        ),

    "confidence_interval_method":
        (
            "95% percentile intervals from the "
            "paired user-cluster bootstrap."
        ),

    "primary_inference":
        (
            "User-cluster bootstrap confidence intervals."
        ),

    "secondary_inference":
        (
            "Exact paired McNemar tests on post-level "
            "correctness with Holm-Bonferroni correction "
            "across all 15 pairwise system comparisons."
        ),

    "test_file":
        str(
            TEST_FILE
        ),

    "test_file_sha256":
        actual_test_hash,

    "split_manifest_file":
        str(
            SPLIT_MANIFEST_FILE
        ),

    "split_manifest_sha256":
        sha256_file(
            SPLIT_MANIFEST_FILE
        ),

    "test_rows":
        expected_test_rows,

    "test_users":
        expected_test_users,

    "positive_test_rows":
        expected_positive_rows,

    "split_unit":
        split_manifest.get(
            "split_unit"
        ),

    "group_column":
        split_manifest.get(
            "group_column"
        ),

    "all_labels":
        ALL_LABELS,

    "positive_labels":
        POSITIVE_LABELS,

    "system_order":
        SYSTEM_ORDER,

    "number_of_systems":
        len(
            SYSTEM_ORDER
        ),

    "number_of_pairwise_comparisons":
        len(
            PAIRWISE_COMPARISONS
        ),

    "python_version":
        platform.python_version(),

    "platform":
        platform.platform(),

    "numpy_version":
        np.__version__,

    "pandas_version":
        pd.__version__,

    "scipy_version":
        scipy.__version__,

    "sklearn_version":
        sklearn.__version__,

    "prediction_files": {

        system_name: {

            "display_name":
                SYSTEMS[
                    system_name
                ][
                    "display_name"
                ],

            "path":
                str(
                    SYSTEMS[
                        system_name
                    ][
                        "path"
                    ]
                ),

            "sha256":
                sha256_file(
                    SYSTEMS[
                        system_name
                    ][
                        "path"
                    ]
                ),
        }

        for system_name
        in SYSTEM_ORDER
    },
}


analysis_provenance_path = (
    PROVENANCE_DIR
    / "analysis_provenance.json"
)


save_json(
    analysis_provenance_path,
    analysis_provenance,
)


# ============================================================
# 20. MASTER MACHINE-READABLE RESULTS
# ============================================================

master_results = {

    "experiment":
        (
            "Six-system paired comparison on "
            "canonical user-disjoint test partition"
        ),

    "test_rows":
        expected_test_rows,

    "test_users":
        expected_test_users,

    "positive_test_rows":
        expected_positive_rows,

    "test_file_sha256":
        actual_test_hash,

    "random_seed":
        SEED,

    "bootstrap_iterations":
        N_BOOTSTRAP,

    "primary_inference":
        (
            "paired user-cluster bootstrap"
        ),

    "systems": {

        system_name: {

            "display_name":
                SYSTEMS[
                    system_name
                ][
                    "display_name"
                ],

            "prediction_file":
                str(
                    SYSTEMS[
                        system_name
                    ][
                        "path"
                    ]
                ),

            "prediction_file_sha256":
                sha256_file(
                    SYSTEMS[
                        system_name
                    ][
                        "path"
                    ]
                ),

            "five_class_metrics":
                five_class_metrics[
                    system_name
                ],

            "positive_only_metrics":
                positive_metrics[
                    system_name
                ],

            "binary_metrics":
                binary_metrics[
                    system_name
                ],
        }

        for system_name
        in SYSTEM_ORDER
    },

    "positive_only_definition":
        (
            "All records whose ground-truth label "
            "is Indicator, Ideation, Behavior, or "
            "Attempt. Predictions of Unsupportive "
            "remain errors and are not excluded."
        ),

    "cluster_bootstrap_system_ci":
        (
            system_ci_df
            .to_dict(
                orient="records"
            )
        ),

    "cluster_bootstrap_pairwise":
        (
            pairwise_cluster_bootstrap_df
            .to_dict(
                orient="records"
            )
        ),

    "mcnemar_secondary":
        (
            mcnemar_df
            .to_dict(
                orient="records"
            )
        ),
}


save_json(
    master_json_path,
    master_results,
)


# ============================================================
# 21. MARKDOWN REPORT
# ============================================================

markdown_lines = []


markdown_lines.append(
    "# Six-System User-Disjoint Test Comparison"
)


markdown_lines.append(
    ""
)


markdown_lines.append(
    (
        f"All six systems were evaluated on the exact "
        f"same canonical user-disjoint test partition "
        f"containing {expected_test_rows:,} posts from "
        f"{expected_test_users:,} unique users."
    )
)


markdown_lines.append(
    ""
)


markdown_lines.append(
    (
        "Primary uncertainty estimates use a paired "
        f"user-cluster bootstrap with {N_BOOTSTRAP:,} "
        "replicates."
    )
)


markdown_lines.append(
    ""
)


markdown_lines.append(
    (
        "Exact post-level McNemar tests are reported "
        "as a secondary sensitivity analysis and are "
        "Holm-adjusted across all 15 pairwise comparisons."
    )
)


markdown_lines.append(
    ""
)


markdown_lines.append(
    "## Primary Metrics"
)


markdown_lines.append(
    ""
)


markdown_lines.append(
    (
        "| System | 5-class Accuracy | "
        "5-class Macro F1 | 95% CI | "
        "Positive-only Macro F1 | 95% CI | "
        "Binary Recall | 95% CI | "
        "Binary F1 | 95% CI |"
    )
)


markdown_lines.append(
    (
        "|---|---:|---:|---:|---:|---:|"
        "---:|---:|---:|---:|"
    )
)


for _, row in summary_df.iterrows():

    markdown_lines.append(
        (
            f"| {row['display_name']} "
            f"| {row['five_class_accuracy']:.4f} "
            f"| {row['five_class_macro_f1']:.4f} "
            f"| [{row['five_class_macro_f1_ci95_lower']:.4f}, "
            f"{row['five_class_macro_f1_ci95_upper']:.4f}] "
            f"| {row['positive_only_macro_f1']:.4f} "
            f"| [{row['positive_only_macro_f1_ci95_lower']:.4f}, "
            f"{row['positive_only_macro_f1_ci95_upper']:.4f}] "
            f"| {row['binary_recall']:.4f} "
            f"| [{row['binary_recall_ci95_lower']:.4f}, "
            f"{row['binary_recall_ci95_upper']:.4f}] "
            f"| {row['binary_f1']:.4f} "
            f"| [{row['binary_f1_ci95_lower']:.4f}, "
            f"{row['binary_f1_ci95_upper']:.4f}] |"
        )
    )


markdown_lines.append(
    ""
)


markdown_lines.append(
    "## Per-Class F1"
)


markdown_lines.append(
    ""
)


header = (
    "| Label | "
    + " | ".join(
        SYSTEMS[
            system
        ][
            "display_name"
        ]
        for system
        in SYSTEM_ORDER
    )
    + " |"
)


separator = (
    "|---|"
    + "---:|" * len(
        SYSTEM_ORDER
    )
)


markdown_lines.append(
    header
)


markdown_lines.append(
    separator
)


for label in ALL_LABELS:

    row = (
        per_class_f1_df[
            per_class_f1_df[
                "label"
            ]
            == label
        ]
        .iloc[
            0
        ]
    )


    values = " | ".join(
        f"{row[system]:.4f}"
        for system
        in SYSTEM_ORDER
    )


    markdown_lines.append(
        f"| {label} | "
        f"{values} |"
    )


markdown_lines.append(
    ""
)


markdown_lines.append(
    "## Pairwise User-Cluster Bootstrap"
)


markdown_lines.append(
    ""
)


markdown_lines.append(
    (
        "Differences are defined as "
        "**System B minus System A**."
    )
)


markdown_lines.append(
    ""
)


primary_pairwise = (
    pairwise_cluster_bootstrap_df[
        pairwise_cluster_bootstrap_df[
            "metric"
        ]
        .isin(
            [
                "five_class_macro_f1",
                "positive_only_macro_f1",
            ]
        )
    ]
)


markdown_lines.append(
    (
        "| Comparison | Metric | Observed Δ | "
        "95% CI | CI excludes zero |"
    )
)


markdown_lines.append(
    "|---|---|---:|---:|---:|"
)


for _, row in primary_pairwise.iterrows():

    markdown_lines.append(
        (
            f"| {row['comparison']} "
            f"| {row['metric']} "
            f"| {row['observed_difference_b_minus_a']:.4f} "
            f"| [{row['ci_95_lower']:.4f}, "
            f"{row['ci_95_upper']:.4f}] "
            f"| {row['ci_excludes_zero']} |"
        )
    )


markdown_lines.append(
    ""
)


markdown_lines.append(
    "## Exact McNemar Tests — Secondary"
)


markdown_lines.append(
    ""
)


markdown_lines.append(
    (
        "| Comparison | A-only correct | "
        "B-only correct | Raw p | "
        "Holm-adjusted p |"
    )
)


markdown_lines.append(
    "|---|---:|---:|---:|---:|"
)


for _, row in mcnemar_df.iterrows():

    markdown_lines.append(
        (
            f"| {row['comparison']} "
            f"| {int(row['system_a_only_correct'])} "
            f"| {int(row['system_b_only_correct'])} "
            f"| {row['raw_p_value']:.6g} "
            f"| {row['holm_adjusted_p_value']:.6g} |"
        )
    )


markdown_lines.append(
    ""
)


markdown_lines.append(
    "## Reproducibility"
)


markdown_lines.append(
    ""
)


markdown_lines.append(
    f"- Bootstrap seed: `{SEED}`"
)


markdown_lines.append(
    (
        f"- Bootstrap replicates: "
        f"`{N_BOOTSTRAP}`"
    )
)


markdown_lines.append(
    (
        f"- Canonical test SHA-256: "
        f"`{actual_test_hash}`"
    )
)


markdown_lines.append(
    (
        "- Complete prediction-file hashes and "
        "software versions are recorded in "
        "`provenance/analysis_provenance.json`."
    )
)


markdown_lines.append(
    (
        "- Available run configuration, metric, "
        "manifest, audit, and training-summary "
        "artifacts are hashed in "
        "`provenance/input_provenance.csv`."
    )
)


with open(
    markdown_path,
    "w",
    encoding="utf-8",
) as handle:

    handle.write(
        "\n".join(
            markdown_lines
        )
    )


# ============================================================
# 22. PRINT PRIMARY RESULTS
# ============================================================

print(
    "\n"
    + "=" * 112
)


print(
    "PRIMARY SIX-SYSTEM RESULTS"
)


print(
    "=" * 112
)


print(
    f"\n{'System':<46}"
    f"{'Accuracy':>10}"
    f"{'Macro F1':>12}"
    f"{'Pos F1':>12}"
    f"{'Bin Recall':>12}"
    f"{'Bin F1':>10}"
)


print(
    "-" * 102
)


for system_name in SYSTEM_ORDER:

    five = (
        five_class_metrics[
            system_name
        ]
    )


    positive = (
        positive_metrics[
            system_name
        ]
    )


    binary = (
        binary_metrics[
            system_name
        ]
    )


    print(
        f"{SYSTEMS[system_name]['display_name']:<46}"
        f"{five['accuracy']:>10.4f}"
        f"{five['macro_f1']:>12.4f}"
        f"{positive['macro_f1']:>12.4f}"
        f"{binary['recall']:>12.4f}"
        f"{binary['f1']:>10.4f}"
    )


# ============================================================
# 23. PRINT SYSTEM CIs
# ============================================================

print(
    "\n"
    + "=" * 112
)


print(
    "95% USER-CLUSTER BOOTSTRAP CIs — "
    "FIVE-CLASS MACRO F1"
)


print(
    "=" * 112
)


macro_ci = (
    system_ci_df[
        system_ci_df[
            "metric"
        ]
        == "five_class_macro_f1"
    ]
)


for _, row in macro_ci.iterrows():

    print(
        f"{row['display_name']:<46}"
        f"{row['point_estimate']:.4f} "
        f"[{row['ci_95_lower']:.4f}, "
        f"{row['ci_95_upper']:.4f}]"
    )


# ============================================================
# 24. KEY MENTALROBERTA VS FRAMEWORK 4
# ============================================================

print(
    "\n"
    + "=" * 112
)


print(
    "KEY COMPARISON: "
    "MENTALROBERTA VS FRAMEWORK 4"
)


print(
    "=" * 112
)


key_comparison = (
    "mentalroberta_ft"
    "_minus_"
    "framework4"
)


for metric in [

    "five_class_macro_f1",

    "positive_only_macro_f1",

    "binary_recall",

    "binary_f1",
]:

    row = (
        pairwise_cluster_bootstrap_df[
            (
                pairwise_cluster_bootstrap_df[
                    "comparison"
                ]
                == key_comparison
            )
            &
            (
                pairwise_cluster_bootstrap_df[
                    "metric"
                ]
                == metric
            )
        ]
        .iloc[
            0
        ]
    )


    print(
        f"\n{metric}:"
    )


    print(
        "Observed difference "
        "(MentalRoBERTa - Framework 4): "
        f"{row['observed_difference_b_minus_a']:.6f}"
    )


    print(
        "95% user-cluster bootstrap CI: "
        f"[{row['ci_95_lower']:.6f}, "
        f"{row['ci_95_upper']:.6f}]"
    )


key_mcnemar = (
    mcnemar_df[
        mcnemar_df[
            "comparison"
        ]
        == key_comparison
    ]
    .iloc[
        0
    ]
)


print(
    "\nSecondary exact McNemar raw p-value: "
    f"{key_mcnemar['raw_p_value']:.8g}"
)


print(
    "Secondary Holm-adjusted p-value: "
    f"{key_mcnemar['holm_adjusted_p_value']:.8g}"
)


# ============================================================
# 25. FINAL
# ============================================================

print(
    "\n"
    + "=" * 92
)


print(
    "SIX-SYSTEM COMPARISON COMPLETE"
)


print(
    "=" * 92
)


print(
    f"\nOutput directory:\n"
    f"{OUTPUT_ROOT.resolve()}"
)


print(
    "\nMain outputs:"
)


print(
    "- tables/six_system_summary_with_ci.csv"
)


print(
    "- tables/per_class_metrics_long.csv"
)


print(
    "- tables/per_class_f1_comparison.csv"
)


print(
    "- tables/paired_predictions_six_systems.csv"
)


print(
    "- stats/system_user_cluster_bootstrap_ci.csv"
)


print(
    "- stats/pairwise_user_cluster_bootstrap.csv"
)


print(
    "- stats/pairwise_mcnemar_secondary.csv"
)


print(
    "- stats/raw_user_cluster_bootstrap.npz"
)


print(
    "- provenance/input_provenance.csv"
)


print(
    "- provenance/analysis_provenance.json"
)


print(
    "- six_system_results.json"
)


print(
    "- six_system_comparison.md"
)


print(
    "- one confusion-matrix CSV per system"
)


print(
    "\nNo model training or LLM inference "
    "was performed by this script."
)
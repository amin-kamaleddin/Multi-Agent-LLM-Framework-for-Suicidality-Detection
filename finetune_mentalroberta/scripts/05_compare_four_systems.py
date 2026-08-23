from pathlib import Path
import hashlib
import json
from itertools import combinations

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

SPLIT_MANIFEST_FILE = Path(
    "data/splits/split_manifest.json"
)

# ------------------------------------------------------------
# Prediction files from the four frozen systems
# ------------------------------------------------------------

MENTALROBERTA_FILE = Path(
    "results/test/test_predictions.csv"
)

SINGLE_LUNA_FILE = Path(
    "../multi_agent_frameworks/logs/runs/"
    "20260822_144023_single_agent_luna_frozen_test/"
    "predictions.csv"
)

FRAMEWORK3_FILE = Path(
    "../multi_agent_frameworks/logs/runs/"
    "20260822_233218_framework3_luna_frozen_test/"
    "predictions.csv"
)

FRAMEWORK4_FILE = Path(
    "../multi_agent_frameworks/logs/runs/"
    "20260823_002216_framework4_luna_frozen_test/"
    "predictions.csv"
)


OUTPUT_DIR = Path(
    "results/four_system_comparison"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# LABELS
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
# SYSTEM NAMES
# ============================================================

SYSTEMS = {
    "single_agent_luna": {
        "display_name": "Single-Agent Luna",
        "path": SINGLE_LUNA_FILE,
        "prediction_candidates": [
            "prediction",
            "predicted_label",
        ],
    },

    "framework3": {
        "display_name": (
            "Framework 3 "
            "(Hierarchical, No Reviewer)"
        ),
        "path": FRAMEWORK3_FILE,
        "prediction_candidates": [
            "prediction",
            "predicted_label",
        ],
    },

    "framework4": {
        "display_name": (
            "Framework 4 "
            "(Hierarchical + Reviewer)"
        ),
        "path": FRAMEWORK4_FILE,
        "prediction_candidates": [
            "prediction",
            "predicted_label",
        ],
    },

    "mentalroberta_ft": {
        "display_name": (
            "Fine-Tuned MentalRoBERTa"
        ),
        "path": MENTALROBERTA_FILE,
        "prediction_candidates": [
            "predicted_label",
            "prediction",
        ],
    },
}


SYSTEM_ORDER = [
    "single_agent_luna",
    "framework3",
    "framework4",
    "mentalroberta_ft",
]


# ============================================================
# IMPORTANT PAIRWISE COMPARISONS
# ============================================================

# Difference is always:
#
# second system - first system
#
# So:
# framework3_minus_single
# means Framework 3 metric minus Single-Agent Luna metric.

PAIRWISE_COMPARISONS = [
    (
        "single_agent_luna",
        "framework3",
    ),
    (
        "framework3",
        "framework4",
    ),
    (
        "single_agent_luna",
        "framework4",
    ),
    (
        "single_agent_luna",
        "mentalroberta_ft",
    ),
    (
        "framework3",
        "mentalroberta_ft",
    ),
    (
        "framework4",
        "mentalroberta_ft",
    ),
]


# ============================================================
# HELPERS
# ============================================================

def sha256_file(path: Path) -> str:
    sha256 = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            sha256.update(chunk)

    return sha256.hexdigest()


def normalize_id(value):
    """
    Normalize IDs so 49, 49.0, and "49"
    all become the same value.
    """

    if pd.isna(value):
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
                int(numeric)
            )

    except ValueError:
        pass

    return value_string


def normalize_label(value):
    """
    Normalize label capitalization/whitespace.
    """

    if pd.isna(value):
        return None

    value = str(
        value
    ).strip()

    lookup = {
        label.lower(): label
        for label in ALL_LABELS
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
    Find a column using a list of possible names.
    """

    for candidate in candidates:
        if candidate in dataframe.columns:
            return candidate

    lower_map = {
        column.lower(): column
        for column in dataframe.columns
    }

    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[
                candidate.lower()
            ]

    raise ValueError(
        f"\nUnable to find {description}.\n"
        f"Available columns:\n"
        f"{list(dataframe.columns)}"
    )


def calculate_metrics(
    y_true,
    y_pred,
    labels,
):
    """
    Calculate aggregate classification metrics.
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

        "weighted_precision": float(
            weighted_precision
        ),

        "weighted_recall": float(
            weighted_recall
        ),

        "weighted_f1": float(
            weighted_f1
        ),
    }


def calculate_binary_metrics(
    y_true,
    y_pred,
):
    """
    Collapse five labels into:

    0 = Unsupportive
    1 = Positive
    """

    true_binary = np.where(
        np.asarray(y_true)
        == "Unsupportive",
        0,
        1,
    )

    pred_binary = np.where(
        np.asarray(y_pred)
        == "Unsupportive",
        0,
        1,
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            true_binary,
            pred_binary,
            average="binary",
            pos_label=1,
            zero_division=0,
        )
    )

    cm = confusion_matrix(
        true_binary,
        pred_binary,
        labels=[0, 1],
    )

    return {
        "accuracy": float(
            accuracy_score(
                true_binary,
                pred_binary,
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

        "tn": int(
            cm[0, 0]
        ),

        "fp": int(
            cm[0, 1]
        ),

        "fn": int(
            cm[1, 0]
        ),

        "tp": int(
            cm[1, 1]
        ),
    }


def calculate_per_class_metrics(
    y_true,
    y_pred,
):
    """
    Precision / recall / F1 for each of the
    five original classes.
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


def exact_mcnemar(
    correct_a,
    correct_b,
):
    """
    Exact paired McNemar test.

    Tests whether discordant correctness
    outcomes are symmetric.
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
        result = binomtest(
            k=min(
                a_only,
                b_only,
            ),
            n=discordant,
            p=0.5,
            alternative="two-sided",
        )

        p_value = float(
            result.pvalue
        )

    return {
        "both_correct": both_correct,
        "a_only_correct": a_only,
        "b_only_correct": b_only,
        "both_wrong": both_wrong,
        "discordant_pairs": discordant,
        "p_value": p_value,
    }


def holm_adjust(
    p_values,
):
    """
    Holm-Bonferroni adjustment for the six
    pairwise McNemar comparisons.

    No extra statistics package required.
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
            p_values[index]
            * multiplier,
        )

        running_max = max(
            running_max,
            candidate,
        )

        adjusted[index] = (
            running_max
        )

    return adjusted


def make_stratified_bootstrap_indices(
    y_true,
    rng,
):
    """
    Generate one bootstrap sample while preserving
    the exact number of observations from each
    ground-truth class.
    """

    y_true = np.asarray(
        y_true
    )

    sampled_parts = []

    for label in np.unique(
        y_true
    ):

        indices = np.where(
            y_true == label
        )[0]

        sampled = rng.choice(
            indices,
            size=len(indices),
            replace=True,
        )

        sampled_parts.append(
            sampled
        )

    return np.concatenate(
        sampled_parts
    )


def bootstrap_all_systems(
    y_true,
    predictions,
    labels,
    n_bootstrap,
    seed,
):
    """
    Paired stratified bootstrap.

    Crucially, every system is evaluated on the
    SAME resampled observations for each bootstrap
    replicate.

    Returns pairwise metric differences:
        system B - system A
    """

    rng = np.random.default_rng(
        seed
    )

    y_true = np.asarray(
        y_true
    )

    predictions = {
        key: np.asarray(value)
        for key, value
        in predictions.items()
    }

    difference_store = {
        f"{system_b}_minus_{system_a}":
            np.empty(
                n_bootstrap,
                dtype=float,
            )

        for system_a, system_b
        in PAIRWISE_COMPARISONS
    }

    for iteration in range(
        n_bootstrap
    ):

        sampled_indices = (
            make_stratified_bootstrap_indices(
                y_true,
                rng,
            )
        )

        sampled_true = (
            y_true[
                sampled_indices
            ]
        )

        sampled_scores = {}

        for system_name in SYSTEM_ORDER:

            sampled_prediction = (
                predictions[
                    system_name
                ][
                    sampled_indices
                ]
            )

            sampled_scores[
                system_name
            ] = f1_score(
                sampled_true,
                sampled_prediction,
                labels=labels,
                average="macro",
                zero_division=0,
            )

        for system_a, system_b in (
            PAIRWISE_COMPARISONS
        ):

            key = (
                f"{system_b}"
                f"_minus_"
                f"{system_a}"
            )

            difference_store[
                key
            ][
                iteration
            ] = (
                sampled_scores[
                    system_b
                ]
                - sampled_scores[
                    system_a
                ]
            )

    return difference_store


def summarize_bootstrap(
    differences,
):
    """
    Summarize bootstrap difference distribution.
    """

    return {
        "mean_difference": float(
            np.mean(
                differences
            )
        ),

        "ci_95_lower": float(
            np.percentile(
                differences,
                2.5,
            )
        ),

        "ci_95_upper": float(
            np.percentile(
                differences,
                97.5,
            )
        ),

        "iterations": int(
            len(
                differences
            )
        ),
    }


# ============================================================
# START
# ============================================================

print("\n" + "=" * 78)
print("FOUR-SYSTEM HELD-OUT TEST COMPARISON")
print("=" * 78)

print(
    "\nSystems:"
)

for system_name in SYSTEM_ORDER:
    print(
        f"- {SYSTEMS[system_name]['display_name']}"
    )


# ============================================================
# 1. VERIFY REQUIRED FILES
# ============================================================

required_files = [
    TEST_FILE,
    SPLIT_MANIFEST_FILE,
]

required_files.extend(
    SYSTEMS[
        system_name
    ]["path"]
    for system_name
    in SYSTEM_ORDER
)


for required_file in required_files:

    if not required_file.exists():

        raise FileNotFoundError(
            f"\nRequired file not found:\n"
            f"{required_file.resolve()}"
        )


print(
    "\nAll required files found."
)


# ============================================================
# 2. VERIFY CANONICAL TEST HASH
# ============================================================

with open(
    SPLIT_MANIFEST_FILE,
    "r",
    encoding="utf-8",
) as f:

    split_manifest = json.load(f)


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

actual_test_hash = (
    sha256_file(
        TEST_FILE
    )
)


if expected_test_hash is not None:

    assert (
        actual_test_hash
        == expected_test_hash
    ), (
        "test.csv no longer matches "
        "the canonical split manifest."
    )


print(
    "Canonical test.csv hash verified."
)


# ============================================================
# 3. LOAD CANONICAL TEST SET
# ============================================================

test_df = pd.read_csv(
    TEST_FILE
)


assert len(
    test_df
) == 4319


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

assert set(
    test_df[
        "true_label"
    ]
) <= set(
    ALL_LABELS
)


print(
    f"\nCanonical test rows: "
    f"{len(test_df):,}"
)


# ============================================================
# 4. LOAD ALL FOUR PREDICTION FILES
# ============================================================

system_compact_frames = {}


for system_name in SYSTEM_ORDER:

    config = SYSTEMS[
        system_name
    ]

    print(
        f"\nLoading "
        f"{config['display_name']}..."
    )

    dataframe = pd.read_csv(
        config[
            "path"
        ]
    )


    assert len(
        dataframe
    ) == 4319, (
        f"{system_name} has "
        f"{len(dataframe):,} rows; "
        f"expected 4,319."
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


    prediction_column = (
        find_column(
            dataframe,
            config[
                "prediction_candidates"
            ],
            (
                f"{system_name} "
                f"prediction column"
            ),
        )
    )


    compact = pd.DataFrame(
        {
            "sample_id": (
                dataframe[
                    id_column
                ]
                .map(
                    normalize_id
                )
            ),

            f"{system_name}_prediction":
                (
                    dataframe[
                        prediction_column
                    ]
                    .map(
                        normalize_label
                    )
                ),
        }
    )


    # --------------------------------------------
    # Verify evaluator ground truth if present
    # --------------------------------------------

    if "ground_truth" in dataframe.columns:

        compact[
            f"{system_name}_reported_truth"
        ] = (
            dataframe[
                "ground_truth"
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
        f"{system_name} contains "
        f"duplicate IDs."
    )


    assert (
        compact[
            "sample_id"
        ]
        .notna()
        .all()
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
        f"{system_name} has unexpected "
        f"labels: "
        f"{unexpected_predictions}"
    )


    assert (
        compact[
            f"{system_name}_prediction"
        ]
        .notna()
        .all()
    ), (
        f"{system_name} contains "
        f"missing predictions."
    )


    system_compact_frames[
        system_name
    ] = compact


# ============================================================
# 5. ALIGN ALL FOUR SYSTEMS BY ID
# ============================================================

paired_df = test_df[
    [
        "sample_id",
        "true_label",
        "text",
    ]
].copy()


for system_name in SYSTEM_ORDER:

    paired_df = paired_df.merge(
        system_compact_frames[
            system_name
        ],
        on="sample_id",
        how="left",
        validate="one_to_one",
    )


assert len(
    paired_df
) == 4319


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
        f"{system_name} IDs do not "
        f"exactly match the canonical test set."
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
            != paired_df[
                "true_label"
            ]
        )

        assert not mismatch.any(), (
            f"{system_name}'s reported "
            f"ground truth does not match "
            f"canonical test.csv."
        )


print("\n" + "=" * 78)
print("ALIGNMENT VERIFIED")
print("=" * 78)

print(
    "All four systems contain the exact "
    "same 4,319 held-out IDs."
)

print(
    "Evaluator ground truth matches "
    "canonical test.csv."
)


# ============================================================
# 6. CORRECTNESS FLAGS
# ============================================================

for system_name in SYSTEM_ORDER:

    paired_df[
        f"{system_name}_correct"
    ] = (
        paired_df[
            f"{system_name}_prediction"
        ]
        == paired_df[
            "true_label"
        ]
    )


# ============================================================
# 7. FIVE-CLASS METRICS
# ============================================================

y_true = paired_df[
    "true_label"
].to_numpy()


predictions = {
    system_name: (
        paired_df[
            f"{system_name}_prediction"
        ]
        .to_numpy()
    )

    for system_name in SYSTEM_ORDER
}


five_class_metrics = {}


for system_name in SYSTEM_ORDER:

    five_class_metrics[
        system_name
    ] = calculate_metrics(
        y_true,
        predictions[
            system_name
        ],
        ALL_LABELS,
    )


# ============================================================
# 8. POSITIVE-ONLY METRICS
# ============================================================

# IMPORTANT:
#
# Positive-only means:
# select by GROUND-TRUTH positive class.
#
# We keep all 2,160 positive cases.
#
# If a system predicted Unsupportive,
# that remains an error.

positive_mask = np.isin(
    y_true,
    POSITIVE_LABELS,
)


positive_true = y_true[
    positive_mask
]


assert len(
    positive_true
) == 2160


positive_predictions = {
    system_name: (
        predictions[
            system_name
        ][
            positive_mask
        ]
    )

    for system_name in SYSTEM_ORDER
}


positive_metrics = {}


for system_name in SYSTEM_ORDER:

    positive_metrics[
        system_name
    ] = calculate_metrics(
        positive_true,
        positive_predictions[
            system_name
        ],
        POSITIVE_LABELS,
    )


# ============================================================
# 9. BINARY METRICS
# ============================================================

binary_metrics = {}


for system_name in SYSTEM_ORDER:

    binary_metrics[
        system_name
    ] = calculate_binary_metrics(
        y_true,
        predictions[
            system_name
        ],
    )


# ============================================================
# 10. PER-CLASS METRICS
# ============================================================

per_class_frames = []


for system_name in SYSTEM_ORDER:

    frame = (
        calculate_per_class_metrics(
            y_true,
            predictions[
                system_name
            ],
        )
    )

    frame.insert(
        0,
        "system",
        system_name,
    )

    frame.insert(
        1,
        "display_name",
        SYSTEMS[
            system_name
        ][
            "display_name"
        ],
    )

    per_class_frames.append(
        frame
    )


per_class_long_df = pd.concat(
    per_class_frames,
    ignore_index=True,
)


# Wide F1 table

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
# 11. CONFUSION MATRICES
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
            for label in ALL_LABELS
        ],
        columns=[
            f"pred_{label}"
            for label in ALL_LABELS
        ],
    )

    cm_df.to_csv(
        OUTPUT_DIR
        / (
            f"confusion_matrix_"
            f"{system_name}.csv"
        )
    )


# ============================================================
# 12. CREATE OVERALL SUMMARY TABLE
# ============================================================

summary_rows = []


for system_name in SYSTEM_ORDER:

    five = five_class_metrics[
        system_name
    ]

    positive = positive_metrics[
        system_name
    ]

    binary = binary_metrics[
        system_name
    ]

    summary_rows.append(
        {
            "system": system_name,

            "display_name": SYSTEMS[
                system_name
            ][
                "display_name"
            ],

            # Five class
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

            # Positive-only
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

            # Binary
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
# 13. PAIRWISE MCNEMAR TESTS
# ============================================================

mcnemar_rows = []


for system_a, system_b in (
    PAIRWISE_COMPARISONS
):

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
            "system_a": (
                system_a
            ),

            "system_b": (
                system_b
            ),

            "comparison": (
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
    ].to_numpy()
)


mcnemar_df[
    "significant_after_holm_0_05"
] = (
    mcnemar_df[
        "holm_adjusted_p_value"
    ]
    < 0.05
)


# ============================================================
# 14. PAIRED STRATIFIED BOOTSTRAP:
#     FIVE-CLASS MACRO F1
# ============================================================

print("\n" + "=" * 78)
print("PAIRED BOOTSTRAP")
print("=" * 78)

print(
    f"\nRunning "
    f"{N_BOOTSTRAP:,} paired stratified "
    f"bootstrap replicates..."
)

print(
    "Metric 1/2: Five-class Macro F1"
)


bootstrap_five = (
    bootstrap_all_systems(
        y_true=y_true,
        predictions=predictions,
        labels=ALL_LABELS,
        n_bootstrap=N_BOOTSTRAP,
        seed=SEED,
    )
)


print(
    "Metric 2/2: Positive-only Macro F1"
)


bootstrap_positive = (
    bootstrap_all_systems(
        y_true=positive_true,
        predictions=positive_predictions,
        labels=POSITIVE_LABELS,
        n_bootstrap=N_BOOTSTRAP,
        seed=SEED + 1,
    )
)


print(
    "Bootstrap complete."
)


# ============================================================
# 15. SUMMARIZE BOOTSTRAP RESULTS
# ============================================================

bootstrap_summary_rows = []


for system_a, system_b in (
    PAIRWISE_COMPARISONS
):

    key = (
        f"{system_b}"
        f"_minus_"
        f"{system_a}"
    )


    five_summary = (
        summarize_bootstrap(
            bootstrap_five[
                key
            ]
        )
    )


    positive_summary = (
        summarize_bootstrap(
            bootstrap_positive[
                key
            ]
        )
    )


    observed_five_difference = (
        five_class_metrics[
            system_b
        ][
            "macro_f1"
        ]
        -
        five_class_metrics[
            system_a
        ][
            "macro_f1"
        ]
    )


    observed_positive_difference = (
        positive_metrics[
            system_b
        ][
            "macro_f1"
        ]
        -
        positive_metrics[
            system_a
        ][
            "macro_f1"
        ]
    )


    bootstrap_summary_rows.append(
        {
            "comparison": key,

            "system_a": (
                system_a
            ),

            "system_b": (
                system_b
            ),

            "five_class_observed_difference":
                observed_five_difference,

            "five_class_ci_95_lower":
                five_summary[
                    "ci_95_lower"
                ],

            "five_class_ci_95_upper":
                five_summary[
                    "ci_95_upper"
                ],

            "positive_only_observed_difference":
                observed_positive_difference,

            "positive_only_ci_95_lower":
                positive_summary[
                    "ci_95_lower"
                ],

            "positive_only_ci_95_upper":
                positive_summary[
                    "ci_95_upper"
                ],
        }
    )


bootstrap_summary_df = (
    pd.DataFrame(
        bootstrap_summary_rows
    )
)


# ============================================================
# 16. SAVE RAW BOOTSTRAP DISTRIBUTIONS
# ============================================================

bootstrap_raw_columns = {}


for key, differences in (
    bootstrap_five.items()
):

    bootstrap_raw_columns[
        f"five_class__{key}"
    ] = differences


for key, differences in (
    bootstrap_positive.items()
):

    bootstrap_raw_columns[
        f"positive_only__{key}"
    ] = differences


bootstrap_raw_df = pd.DataFrame(
    bootstrap_raw_columns
)


# ============================================================
# 17. SAVE ALL OUTPUTS
# ============================================================

paired_predictions_path = (
    OUTPUT_DIR
    / "paired_predictions_four_systems.csv"
)

summary_path = (
    OUTPUT_DIR
    / "four_system_summary.csv"
)

per_class_long_path = (
    OUTPUT_DIR
    / "per_class_metrics_long.csv"
)

per_class_f1_path = (
    OUTPUT_DIR
    / "per_class_f1_comparison.csv"
)

mcnemar_path = (
    OUTPUT_DIR
    / "pairwise_mcnemar_tests.csv"
)

bootstrap_summary_path = (
    OUTPUT_DIR
    / "pairwise_bootstrap_summary.csv"
)

bootstrap_raw_path = (
    OUTPUT_DIR
    / "bootstrap_distributions.csv"
)

json_path = (
    OUTPUT_DIR
    / "four_system_results.json"
)

markdown_path = (
    OUTPUT_DIR
    / "four_system_comparison.md"
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

mcnemar_df.to_csv(
    mcnemar_path,
    index=False,
)

bootstrap_summary_df.to_csv(
    bootstrap_summary_path,
    index=False,
)

bootstrap_raw_df.to_csv(
    bootstrap_raw_path,
    index=False,
)


# ============================================================
# 18. MASTER JSON
# ============================================================

master_results = {
    "experiment": (
        "Four-system paired comparison "
        "on canonical held-out test set"
    ),

    "test_rows": int(
        len(
            paired_df
        )
    ),

    "positive_test_rows": int(
        positive_mask.sum()
    ),

    "test_file_sha256": (
        actual_test_hash
    ),

    "random_seed": SEED,

    "bootstrap_iterations": (
        N_BOOTSTRAP
    ),

    "systems": {
        system_name: {
            "display_name": (
                SYSTEMS[
                    system_name
                ][
                    "display_name"
                ]
            ),

            "prediction_file": str(
                SYSTEMS[
                    system_name
                ][
                    "path"
                ]
            ),

            "five_class_metrics": (
                five_class_metrics[
                    system_name
                ]
            ),

            "positive_only_metrics": (
                positive_metrics[
                    system_name
                ]
            ),

            "binary_metrics": (
                binary_metrics[
                    system_name
                ]
            ),
        }

        for system_name
        in SYSTEM_ORDER
    },

    "mcnemar_pairwise": (
        mcnemar_df
        .to_dict(
            orient="records"
        )
    ),

    "bootstrap_pairwise": (
        bootstrap_summary_df
        .to_dict(
            orient="records"
        )
    ),

    "positive_only_definition": (
        "All records whose ground-truth label "
        "is Indicator, Ideation, Behavior, or "
        "Attempt. Predictions of Unsupportive "
        "remain errors and are not excluded."
    ),
}


with open(
    json_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        master_results,
        f,
        indent=4,
        ensure_ascii=False,
    )


# ============================================================
# 19. MARKDOWN REPORT
# ============================================================

markdown_lines = []

markdown_lines.append(
    "# Four-System Held-Out Test Comparison"
)

markdown_lines.append("")

markdown_lines.append(
    "All systems were evaluated on the same "
    "4,319 canonical held-out posts."
)

markdown_lines.append("")

markdown_lines.append(
    "## Primary Metrics"
)

markdown_lines.append("")

markdown_lines.append(
    "| System | 5-class Accuracy | "
    "5-class Macro F1 | "
    "Positive-only Macro F1 | "
    "Binary Recall | Binary F1 |"
)

markdown_lines.append(
    "|---|---:|---:|---:|---:|---:|"
)


for system_name in SYSTEM_ORDER:

    five = five_class_metrics[
        system_name
    ]

    positive = positive_metrics[
        system_name
    ]

    binary = binary_metrics[
        system_name
    ]

    markdown_lines.append(
        f"| "
        f"{SYSTEMS[system_name]['display_name']} "
        f"| {five['accuracy']:.4f} "
        f"| {five['macro_f1']:.4f} "
        f"| {positive['macro_f1']:.4f} "
        f"| {binary['recall']:.4f} "
        f"| {binary['f1']:.4f} |"
    )


markdown_lines.append("")

markdown_lines.append(
    "## Per-Class F1"
)

markdown_lines.append("")

markdown_lines.append(
    "| Label | Single-Agent Luna | "
    "Framework 3 | Framework 4 | "
    "Fine-Tuned MentalRoBERTa |"
)

markdown_lines.append(
    "|---|---:|---:|---:|---:|"
)


for label in ALL_LABELS:

    row = (
        per_class_f1_df[
            per_class_f1_df[
                "label"
            ]
            == label
        ]
        .iloc[0]
    )

    markdown_lines.append(
        f"| {label} "
        f"| {row['single_agent_luna']:.4f} "
        f"| {row['framework3']:.4f} "
        f"| {row['framework4']:.4f} "
        f"| {row['mentalroberta_ft']:.4f} |"
    )


markdown_lines.append("")

markdown_lines.append(
    "## Pairwise Bootstrap Comparisons"
)

markdown_lines.append("")

markdown_lines.append(
    "Differences are calculated as "
    "**System B minus System A**."
)

markdown_lines.append("")

markdown_lines.append(
    "| Comparison | 5-class Macro F1 Δ | "
    "95% CI | Positive-only Macro F1 Δ | "
    "95% CI |"
)

markdown_lines.append(
    "|---|---:|---:|---:|---:|"
)


for _, row in (
    bootstrap_summary_df
    .iterrows()
):

    markdown_lines.append(
        f"| {row['comparison']} "
        f"| {row['five_class_observed_difference']:.4f} "
        f"| "
        f"[{row['five_class_ci_95_lower']:.4f}, "
        f"{row['five_class_ci_95_upper']:.4f}] "
        f"| {row['positive_only_observed_difference']:.4f} "
        f"| "
        f"[{row['positive_only_ci_95_lower']:.4f}, "
        f"{row['positive_only_ci_95_upper']:.4f}] |"
    )


markdown_lines.append("")

markdown_lines.append(
    "## Pairwise McNemar Tests"
)

markdown_lines.append("")

markdown_lines.append(
    "| Comparison | A-only correct | "
    "B-only correct | Raw p | "
    "Holm-adjusted p |"
)

markdown_lines.append(
    "|---|---:|---:|---:|---:|"
)


for _, row in (
    mcnemar_df
    .iterrows()
):

    markdown_lines.append(
        f"| {row['comparison']} "
        f"| {int(row['system_a_only_correct'])} "
        f"| {int(row['system_b_only_correct'])} "
        f"| {row['raw_p_value']:.6g} "
        f"| {row['holm_adjusted_p_value']:.6g} |"
    )


with open(
    markdown_path,
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "\n".join(
            markdown_lines
        )
    )


# ============================================================
# 20. PRINT PRIMARY RESULTS
# ============================================================

print("\n" + "=" * 78)
print("PRIMARY FOUR-SYSTEM RESULTS")
print("=" * 78)


print(
    f"\n{'System':<40}"
    f"{'Accuracy':>10}"
    f"{'Macro F1':>12}"
    f"{'Pos F1':>12}"
    f"{'Bin Recall':>12}"
    f"{'Bin F1':>10}"
)

print("-" * 96)


for system_name in SYSTEM_ORDER:

    five = five_class_metrics[
        system_name
    ]

    positive = positive_metrics[
        system_name
    ]

    binary = binary_metrics[
        system_name
    ]

    print(
        f"{SYSTEMS[system_name]['display_name']:<40}"
        f"{five['accuracy']:>10.4f}"
        f"{five['macro_f1']:>12.4f}"
        f"{positive['macro_f1']:>12.4f}"
        f"{binary['recall']:>12.4f}"
        f"{binary['f1']:>10.4f}"
    )


# ============================================================
# 21. PRINT PER-CLASS F1
# ============================================================

print("\n" + "=" * 78)
print("PER-CLASS F1")
print("=" * 78)


print(
    per_class_f1_df.to_string(
        index=False
    )
)


# ============================================================
# 22. PRINT KEY MENTALROBERTA COMPARISON
# ============================================================

print("\n" + "=" * 78)
print("KEY COMPARISON: MENTALROBERTA VS FRAMEWORK 4")
print("=" * 78)


key_row = (
    bootstrap_summary_df[
        bootstrap_summary_df[
            "comparison"
        ]
        == (
            "mentalroberta_ft"
            "_minus_"
            "framework4"
        )
    ]
    .iloc[0]
)


print(
    "\nFive-class Macro F1 difference:"
)

print(
    f"{key_row['five_class_observed_difference']:.6f}"
)

print(
    "95% CI:"
)

print(
    f"["
    f"{key_row['five_class_ci_95_lower']:.6f}, "
    f"{key_row['five_class_ci_95_upper']:.6f}"
    f"]"
)


print(
    "\nPositive-only Macro F1 difference:"
)

print(
    f"{key_row['positive_only_observed_difference']:.6f}"
)

print(
    "95% CI:"
)

print(
    f"["
    f"{key_row['positive_only_ci_95_lower']:.6f}, "
    f"{key_row['positive_only_ci_95_upper']:.6f}"
    f"]"
)


key_mcnemar = (
    mcnemar_df[
        mcnemar_df[
            "comparison"
        ]
        == (
            "mentalroberta_ft"
            "_minus_"
            "framework4"
        )
    ]
    .iloc[0]
)


print(
    "\nMcNemar exact p-value:"
)

print(
    key_mcnemar[
        "raw_p_value"
    ]
)


print(
    "Holm-adjusted p-value:"
)

print(
    key_mcnemar[
        "holm_adjusted_p_value"
    ]
)


# ============================================================
# 23. FINAL
# ============================================================

print("\n" + "=" * 78)
print("FOUR-SYSTEM COMPARISON COMPLETE")
print("=" * 78)

print(
    f"\nOutput directory:\n"
    f"{OUTPUT_DIR.resolve()}"
)

print(
    "\nMain outputs:"
)

print(
    "- four_system_summary.csv"
)

print(
    "- per_class_f1_comparison.csv"
)

print(
    "- per_class_metrics_long.csv"
)

print(
    "- pairwise_mcnemar_tests.csv"
)

print(
    "- pairwise_bootstrap_summary.csv"
)

print(
    "- paired_predictions_four_systems.csv"
)

print(
    "- four_system_results.json"
)

print(
    "- four_system_comparison.md"
)

print(
    "- one confusion matrix CSV per system"
)

print(
    "\nNo model training or LLM calls were performed."
)
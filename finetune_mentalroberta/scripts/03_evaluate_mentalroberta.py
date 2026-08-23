from pathlib import Path
import hashlib
import json
import platform
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)
import transformers


# ============================================================
# SETTINGS
# ============================================================

SEED = 42

MODEL_DIR = Path(
    "models/mentalroberta_5class/best_model"
)

TEST_FILE = Path(
    "data/splits/test.csv"
)

SPLIT_MANIFEST_FILE = Path(
    "data/splits/split_manifest.json"
)

TRAINING_SUMMARY_FILE = Path(
    "results/training/training_summary.json"
)

OUTPUT_DIR = Path(
    "results/test"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MAX_LENGTH = 512
BATCH_SIZE = 8


# ============================================================
# CANONICAL LABEL MAPPING
# ============================================================

LABEL2ID = {
    "Unsupportive": 0,
    "Indicator": 1,
    "Ideation": 2,
    "Behavior": 3,
    "Attempt": 4,
}

ID2LABEL = {
    0: "Unsupportive",
    1: "Indicator",
    2: "Ideation",
    3: "Behavior",
    4: "Attempt",
}

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

ALL_LABEL_IDS = [
    LABEL2ID[label]
    for label in ALL_LABELS
]

POSITIVE_LABEL_IDS = [
    LABEL2ID[label]
    for label in POSITIVE_LABELS
]


# ============================================================
# REPRODUCIBILITY
# ============================================================

np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

if (
    hasattr(torch.backends, "mps")
    and torch.backends.mps.is_available()
    and hasattr(torch, "mps")
    and hasattr(torch.mps, "manual_seed")
):
    torch.mps.manual_seed(SEED)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def sha256_file(path: Path) -> str:
    """
    Calculate SHA-256 hash of a file.
    """
    sha256 = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            sha256.update(chunk)

    return sha256.hexdigest()


def detect_device():
    """
    Use CUDA if available, otherwise Apple MPS,
    otherwise CPU.
    """

    if torch.cuda.is_available():
        return (
            torch.device("cuda"),
            f"CUDA: {torch.cuda.get_device_name(0)}",
        )

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return (
            torch.device("mps"),
            "Apple MPS",
        )

    return (
        torch.device("cpu"),
        "CPU",
    )


def make_json_serializable(value):
    """
    Convert NumPy / PyTorch scalar values into
    normal Python values for JSON output.
    """

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    if torch.is_tensor(value):
        return value.detach().cpu().tolist()

    return value


def calculate_metrics(
    y_true,
    y_pred,
    labels,
    target_names,
):
    """
    Calculate aggregate and per-class metrics.
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

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy),

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

        "classification_report": report,
    }


def print_metrics_summary(
    title,
    metrics,
):
    """
    Print main aggregate metrics.
    """

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    print(
        f"Accuracy:        "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Macro Precision: "
        f"{metrics['macro_precision']:.4f}"
    )

    print(
        f"Macro Recall:    "
        f"{metrics['macro_recall']:.4f}"
    )

    print(
        f"Macro F1:        "
        f"{metrics['macro_f1']:.4f}"
    )

    print(
        f"Weighted F1:     "
        f"{metrics['weighted_f1']:.4f}"
    )


# ============================================================
# PYTORCH DATASET
# ============================================================

class TestDataset(Dataset):

    def __init__(
        self,
        encodings,
        labels,
        ids,
    ):
        self.encodings = encodings
        self.labels = labels
        self.ids = ids

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        item = {
            key: value[idx]
            for key, value
            in self.encodings.items()
        }

        item["labels"] = (
            self.labels[idx]
        )

        item["row_index"] = idx

        return item


# ============================================================
# START
# ============================================================

print("\n" + "=" * 70)
print("MENTALROBERTA — FINAL HELD-OUT TEST")
print("=" * 70)

print(
    "\nIMPORTANT:"
)

print(
    "This script performs NO training and "
    "NO model selection."
)

print(
    "The frozen model is evaluated on "
    "test.csv exactly as saved."
)


# ============================================================
# 1. VERIFY REQUIRED FILES
# ============================================================

for required_path in [
    MODEL_DIR,
    TEST_FILE,
    SPLIT_MANIFEST_FILE,
]:

    if not required_path.exists():

        raise FileNotFoundError(
            f"Required path not found: "
            f"{required_path}"
        )


# ============================================================
# 2. VERIFY TEST FILE HASH
# ============================================================

with open(
    SPLIT_MANIFEST_FILE,
    "r",
    encoding="utf-8",
) as f:

    split_manifest = json.load(f)


expected_test_hash = (
    split_manifest
    .get("output_file_sha256", {})
    .get("test.csv")
)

actual_test_hash = sha256_file(
    TEST_FILE
)

if expected_test_hash is None:

    raise ValueError(
        "No test.csv hash was found in "
        "split_manifest.json."
    )

assert (
    actual_test_hash == expected_test_hash
), (
    "ERROR: test.csv does not match the "
    "canonical held-out test split."
)

print(
    "\nCanonical test.csv hash verified."
)


# ============================================================
# 3. LOAD TEST DATA
# ============================================================

test_df = pd.read_csv(
    TEST_FILE
)

print(
    f"\nTest rows loaded: "
    f"{len(test_df):,}"
)

assert len(test_df) == 4319, (
    f"Expected 4,319 test rows, "
    f"found {len(test_df):,}."
)


# ============================================================
# 4. TEST DATA INTEGRITY
# ============================================================

required_columns = {
    "ID",
    "text",
    "sentiment",
}

missing_columns = (
    required_columns
    - set(test_df.columns)
)

assert not missing_columns, (
    f"Missing columns: {missing_columns}"
)

assert test_df["ID"].is_unique, (
    "Duplicate IDs found in test.csv."
)

assert test_df["text"].notna().all(), (
    "Missing text found in test.csv."
)

assert test_df["sentiment"].notna().all(), (
    "Missing labels found in test.csv."
)

unexpected_labels = (
    set(test_df["sentiment"].unique())
    - set(ALL_LABELS)
)

assert not unexpected_labels, (
    f"Unexpected labels found: "
    f"{unexpected_labels}"
)

print(
    "Test data integrity checks passed."
)

print(
    "\nTest label distribution:"
)

print(
    test_df["sentiment"]
    .value_counts()
    .reindex(ALL_LABELS)
)


# ============================================================
# 5. LOAD FROZEN MODEL
# ============================================================

device, device_description = (
    detect_device()
)

print(
    f"\nDevice: {device_description}"
)

print(
    "\nLoading frozen MentalRoBERTa model..."
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR
)

model = (
    AutoModelForSequenceClassification
    .from_pretrained(
        MODEL_DIR
    )
)

model.to(device)

model.eval()


# ============================================================
# 6. VERIFY MODEL LABEL CONFIGURATION
# ============================================================

assert model.config.num_labels == 5, (
    f"Expected a five-class model, "
    f"found {model.config.num_labels} labels."
)

print(
    "Frozen five-class model loaded."
)


# ============================================================
# 7. TOKENIZE TEST DATA
# ============================================================

print(
    "\nTokenizing held-out test set..."
)

texts = (
    test_df["text"]
    .astype(str)
    .tolist()
)

true_labels = (
    test_df["sentiment"]
    .map(LABEL2ID)
    .astype(int)
    .tolist()
)

ids = test_df["ID"].tolist()


encodings = tokenizer(
    texts,
    truncation=True,
    max_length=MAX_LENGTH,
    padding=False,
)


test_dataset = TestDataset(
    encodings=encodings,
    labels=true_labels,
    ids=ids,
)


data_collator = (
    DataCollatorWithPadding(
        tokenizer=tokenizer
    )
)


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=data_collator,
    num_workers=0,
    pin_memory=False,
)


# ============================================================
# 8. RUN FINAL HELD-OUT INFERENCE
# ============================================================

print("\n" + "=" * 70)
print("RUNNING FINAL TEST INFERENCE")
print("=" * 70)

print(
    f"\nExamples:   {len(test_dataset):,}"
)

print(
    f"Batch size: {BATCH_SIZE}"
)

all_logits = []
all_predictions = []
all_probabilities = []
all_true_labels = []
all_row_indices = []


with torch.inference_mode():

    for batch_number, batch in enumerate(
        test_loader,
        start=1,
    ):

        labels = batch.pop(
            "labels"
        )

        row_indices = batch.pop(
            "row_index"
        )

        batch = {
            key: value.to(device)
            for key, value
            in batch.items()
        }

        outputs = model(
            **batch
        )

        logits = outputs.logits

        probabilities = torch.softmax(
            logits,
            dim=-1,
        )

        predictions = torch.argmax(
            probabilities,
            dim=-1,
        )

        all_logits.append(
            logits.cpu()
        )

        all_probabilities.append(
            probabilities.cpu()
        )

        all_predictions.append(
            predictions.cpu()
        )

        all_true_labels.append(
            labels.cpu()
        )

        all_row_indices.append(
            row_indices.cpu()
        )

        if (
            batch_number % 50 == 0
            or batch_number == len(test_loader)
        ):

            processed = min(
                batch_number * BATCH_SIZE,
                len(test_dataset),
            )

            print(
                f"Processed "
                f"{processed:,} / "
                f"{len(test_dataset):,}"
            )


# ============================================================
# 9. COMBINE INFERENCE OUTPUTS
# ============================================================

all_logits = torch.cat(
    all_logits,
    dim=0,
).numpy()

all_probabilities = torch.cat(
    all_probabilities,
    dim=0,
).numpy()

all_predictions = torch.cat(
    all_predictions,
    dim=0,
).numpy()

all_true_labels = torch.cat(
    all_true_labels,
    dim=0,
).numpy()

all_row_indices = torch.cat(
    all_row_indices,
    dim=0,
).numpy()


# Ensure original deterministic order.

order = np.argsort(
    all_row_indices
)

all_logits = all_logits[order]

all_probabilities = (
    all_probabilities[order]
)

all_predictions = (
    all_predictions[order]
)

all_true_labels = (
    all_true_labels[order]
)


assert len(all_predictions) == len(
    test_df
)

assert np.array_equal(
    all_true_labels,
    np.array(true_labels),
), (
    "Prediction rows do not align with "
    "the original test rows."
)

print(
    "\nInference complete."
)


# ============================================================
# 10. CREATE PER-ROW PREDICTION FILE
# ============================================================

prediction_df = test_df.copy()

prediction_df["true_label"] = (
    [
        ID2LABEL[int(label_id)]
        for label_id
        in all_true_labels
    ]
)

prediction_df["predicted_label"] = (
    [
        ID2LABEL[int(label_id)]
        for label_id
        in all_predictions
    ]
)

prediction_df["correct"] = (
    all_true_labels
    == all_predictions
)

prediction_df["confidence"] = (
    all_probabilities.max(
        axis=1
    )
)


for label_name, label_id in (
    LABEL2ID.items()
):

    prediction_df[
        f"prob_{label_name}"
    ] = (
        all_probabilities[
            :,
            label_id
        ]
    )


predictions_path = (
    OUTPUT_DIR
    / "test_predictions.csv"
)

prediction_df.to_csv(
    predictions_path,
    index=False,
)

print(
    f"\nPer-row predictions saved to:\n"
    f"{predictions_path.resolve()}"
)


# ============================================================
# 11. FULL FIVE-CLASS METRICS
# ============================================================

full_metrics = calculate_metrics(
    y_true=all_true_labels,
    y_pred=all_predictions,
    labels=ALL_LABEL_IDS,
    target_names=ALL_LABELS,
)

print_metrics_summary(
    "FULL FIVE-CLASS HELD-OUT TEST",
    full_metrics,
)


# ============================================================
# 12. FULL FIVE-CLASS CONFUSION MATRIX
# ============================================================

full_confusion_matrix = (
    confusion_matrix(
        all_true_labels,
        all_predictions,
        labels=ALL_LABEL_IDS,
    )
)

full_cm_df = pd.DataFrame(
    full_confusion_matrix,
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

full_cm_path = (
    OUTPUT_DIR
    / "confusion_matrix_5class.csv"
)

full_cm_df.to_csv(
    full_cm_path
)


# ============================================================
# 13. POSITIVE-ONLY ANALYSIS
# ============================================================

# IMPORTANT:
#
# We are NOT running the model again.
#
# We simply select the records in the canonical
# test set whose true labels are one of:
#
# Indicator
# Ideation
# Behavior
# Attempt
#
# If MentalRoBERTa predicted "Unsupportive" for
# one of these records, that prediction remains
# an error. Nothing is changed or reclassified.

positive_mask = np.isin(
    all_true_labels,
    POSITIVE_LABEL_IDS,
)

positive_true = (
    all_true_labels[
        positive_mask
    ]
)

positive_pred = (
    all_predictions[
        positive_mask
    ]
)

positive_prediction_df = (
    prediction_df[
        positive_mask
    ]
    .copy()
)

assert len(
    positive_prediction_df
) == 2160, (
    f"Expected 2,160 positive test rows, "
    f"found {len(positive_prediction_df):,}."
)


positive_metrics = calculate_metrics(
    y_true=positive_true,
    y_pred=positive_pred,
    labels=POSITIVE_LABEL_IDS,
    target_names=POSITIVE_LABELS,
)


print_metrics_summary(
    "POSITIVE-ONLY TEST SUBSET",
    positive_metrics,
)


# ============================================================
# 14. POSITIVE-SUBSET PREDICTION DISTRIBUTION
# ============================================================

# This is useful because the model is still allowed
# to predict Unsupportive for a genuinely positive row.

positive_prediction_distribution = (
    positive_prediction_df[
        "predicted_label"
    ]
    .value_counts()
    .reindex(
        ALL_LABELS,
        fill_value=0,
    )
)

print(
    "\nPredictions made on positive-only "
    "ground-truth cases:"
)

print(
    positive_prediction_distribution
)


positive_predictions_path = (
    OUTPUT_DIR
    / "test_predictions_positive_only.csv"
)

positive_prediction_df.to_csv(
    positive_predictions_path,
    index=False,
)


# ============================================================
# 15. POSITIVE-ONLY CONFUSION MATRIX
# ============================================================

# Use all five prediction columns here.
#
# That allows us to see positive examples incorrectly
# classified as Unsupportive.

positive_confusion_matrix = (
    confusion_matrix(
        positive_true,
        positive_pred,
        labels=ALL_LABEL_IDS,
    )
)

positive_cm_df = pd.DataFrame(
    positive_confusion_matrix,
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

# Unsupportive is not a ground-truth label in this
# subset, so that row will contain zeros.
#
# Keep it for consistency and transparency.

positive_cm_path = (
    OUTPUT_DIR
    / "confusion_matrix_positive_subset.csv"
)

positive_cm_df.to_csv(
    positive_cm_path
)


# ============================================================
# 16. DERIVE BINARY POSITIVE VS UNSUPPORTIVE PERFORMANCE
# ============================================================

# Because this five-class classifier can also make
# an Unsupportive prediction, we can collapse its
# predictions into:
#
# Unsupportive
# vs
# Positive (any of the four RSD classes)
#
# without rerunning the model.

binary_true = np.where(
    all_true_labels
    == LABEL2ID["Unsupportive"],
    0,
    1,
)

binary_pred = np.where(
    all_predictions
    == LABEL2ID["Unsupportive"],
    0,
    1,
)


binary_accuracy = accuracy_score(
    binary_true,
    binary_pred,
)

(
    binary_precision,
    binary_recall,
    binary_f1,
    _,
) = precision_recall_fscore_support(
    binary_true,
    binary_pred,
    labels=[1],
    average="binary",
    pos_label=1,
    zero_division=0,
)

binary_cm = confusion_matrix(
    binary_true,
    binary_pred,
    labels=[0, 1],
)


binary_metrics = {
    "accuracy": float(
        binary_accuracy
    ),

    "positive_precision": float(
        binary_precision
    ),

    "positive_recall": float(
        binary_recall
    ),

    "positive_f1": float(
        binary_f1
    ),

    "confusion_matrix": {
        "true_unsupportive_pred_unsupportive":
            int(binary_cm[0, 0]),

        "true_unsupportive_pred_positive":
            int(binary_cm[0, 1]),

        "true_positive_pred_unsupportive":
            int(binary_cm[1, 0]),

        "true_positive_pred_positive":
            int(binary_cm[1, 1]),
    },
}


print("\n" + "=" * 70)
print("DERIVED BINARY POSITIVE VS UNSUPPORTIVE")
print("=" * 70)

print(
    f"Accuracy:  "
    f"{binary_accuracy:.4f}"
)

print(
    f"Precision: "
    f"{binary_precision:.4f}"
)

print(
    f"Recall:    "
    f"{binary_recall:.4f}"
)

print(
    f"F1:        "
    f"{binary_f1:.4f}"
)

print(
    "\nBinary confusion matrix "
    "[[TN, FP], [FN, TP]]:"
)

print(binary_cm)


# ============================================================
# 17. SAVE CLASSIFICATION REPORTS
# ============================================================

full_report_df = (
    pd.DataFrame(
        full_metrics[
            "classification_report"
        ]
    )
    .transpose()
)

full_report_path = (
    OUTPUT_DIR
    / "classification_report_5class.csv"
)

full_report_df.to_csv(
    full_report_path
)


positive_report_df = (
    pd.DataFrame(
        positive_metrics[
            "classification_report"
        ]
    )
    .transpose()
)

positive_report_path = (
    OUTPUT_DIR
    / "classification_report_positive_subset.csv"
)

positive_report_df.to_csv(
    positive_report_path
)


# ============================================================
# 18. LOAD TRAINING SUMMARY FOR PROVENANCE
# ============================================================

training_summary = None

if TRAINING_SUMMARY_FILE.exists():

    with open(
        TRAINING_SUMMARY_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        training_summary = json.load(f)


# ============================================================
# 19. MODEL FILE HASH
# ============================================================

model_weight_files = list(
    MODEL_DIR.glob("*.safetensors")
)

if not model_weight_files:

    model_weight_files = list(
        MODEL_DIR.glob("pytorch_model*.bin")
    )


model_weight_hashes = {}

for model_file in model_weight_files:

    model_weight_hashes[
        model_file.name
    ] = sha256_file(
        model_file
    )


# ============================================================
# 20. SAVE MASTER TEST RESULTS JSON
# ============================================================

results = {
    "experiment": (
        "MentalRoBERTa five-class final "
        "held-out evaluation"
    ),

    "evaluated_utc": (
        datetime.now(
            timezone.utc
        ).isoformat()
    ),

    "random_seed": SEED,

    "model_directory": str(
        MODEL_DIR
    ),

    "model_weight_sha256": (
        model_weight_hashes
    ),

    "test_file": str(
        TEST_FILE
    ),

    "test_file_sha256": (
        actual_test_hash
    ),

    "test_rows": int(
        len(test_df)
    ),

    "positive_test_rows": int(
        positive_mask.sum()
    ),

    "device": device_description,

    "max_length": MAX_LENGTH,

    "batch_size": BATCH_SIZE,

    "labels": LABEL2ID,

    "five_class_metrics": (
        full_metrics
    ),

    "positive_only_metrics": (
        positive_metrics
    ),

    "positive_subset_prediction_distribution": {
        label: int(
            positive_prediction_distribution[
                label
            ]
        )
        for label
        in ALL_LABELS
    },

    "binary_positive_vs_unsupportive": (
        binary_metrics
    ),

    "five_class_confusion_matrix": (
        full_confusion_matrix.tolist()
    ),

    "five_class_confusion_matrix_label_order": (
        ALL_LABELS
    ),

    "training_summary": (
        training_summary
    ),

    "software_versions": {
        "python": (
            platform.python_version()
        ),

        "torch": (
            torch.__version__
        ),

        "transformers": (
            transformers.__version__
        ),

        "pandas": (
            pd.__version__
        ),

        "numpy": (
            np.__version__
        ),

        "scikit_learn": (
            sklearn.__version__
        ),
    },
}


results_path = (
    OUTPUT_DIR
    / "test_results.json"
)

with open(
    results_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        results,
        f,
        indent=4,
        ensure_ascii=False,
        default=make_json_serializable,
    )


# ============================================================
# 21. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL HELD-OUT EVALUATION COMPLETE")
print("=" * 70)

print(
    "\nFive-class Macro F1:"
)

print(
    f"{full_metrics['macro_f1']:.6f}"
)

print(
    "\nPositive-only Macro F1:"
)

print(
    f"{positive_metrics['macro_f1']:.6f}"
)

print(
    "\nFive-class Accuracy:"
)

print(
    f"{full_metrics['accuracy']:.6f}"
)

print(
    "\nResults saved to:"
)

print(
    OUTPUT_DIR.resolve()
)

print(
    "\nMain files:"
)

print(
    f"- {predictions_path.name}"
)

print(
    f"- {positive_predictions_path.name}"
)

print(
    f"- {full_report_path.name}"
)

print(
    f"- {positive_report_path.name}"
)

print(
    f"- {full_cm_path.name}"
)

print(
    f"- {positive_cm_path.name}"
)

print(
    f"- {results_path.name}"
)

print(
    "\nThe frozen model was not modified."
)

print(
    "No further model selection should be "
    "performed using this test set."
)
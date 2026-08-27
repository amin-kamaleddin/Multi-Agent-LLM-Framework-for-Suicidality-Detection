from pathlib import Path
import hashlib
import json
import os
import platform
import random
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn.functional as F

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset

import transformers
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)


# ============================================================
# SETTINGS
# ============================================================

SEED = 42

MODEL_NAME = "mental/mental-roberta-base"
MODEL_REVISION = "610e9d35d8f2550d713579077242d5e523209cea"

TRAIN_FILE = Path("data/splits/train.csv")
VALIDATION_FILE = Path("data/splits/validation.csv")
SPLIT_MANIFEST_FILE = Path("data/splits/split_manifest.json")

MODEL_ROOT = Path("models/mentalroberta_5class_user_disjoint")
CHECKPOINT_DIR = MODEL_ROOT / "checkpoints"
FINAL_MODEL_DIR = MODEL_ROOT / "best_model"

RESULTS_DIR = Path("results/training_user_disjoint")

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Five-class label mapping
# ------------------------------------------------------------

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

NUM_LABELS = len(LABEL2ID)

# ------------------------------------------------------------
# Model/training hyperparameters
# ------------------------------------------------------------

MAX_LENGTH = 512

LEARNING_RATE = 2e-5

NUM_TRAIN_EPOCHS = 5

TRAIN_BATCH_SIZE = 4
EVAL_BATCH_SIZE = 8

# Effective train batch size:
#
# 4 examples/device
# x 4 gradient accumulation
# = 16 examples per optimizer update
GRADIENT_ACCUMULATION_STEPS = 4

WEIGHT_DECAY = 0.01

WARMUP_RATIO = 0.10

EARLY_STOPPING_PATIENCE = 2


# ============================================================
# REPRODUCIBILITY
# ============================================================

os.environ["PYTHONHASHSEED"] = str(SEED)

random.seed(SEED)
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

# Hugging Face helper: seeds Python, NumPy, and PyTorch.
set_seed(SEED)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def sha256_file(path: Path) -> str:
    """
    Calculate the SHA-256 hash of a file.
    """
    sha256 = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            sha256.update(chunk)

    return sha256.hexdigest()


def detect_hardware() -> str:
    """
    Report the main accelerator available to PyTorch.
    """

    if torch.cuda.is_available():
        return f"CUDA: {torch.cuda.get_device_name(0)}"

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return "Apple MPS"

    return "CPU"


# ============================================================
# DATASET CLASS
# ============================================================

class MentalHealthDataset(Dataset):
    """
    Simple PyTorch Dataset containing tokenized posts
    and integer class labels.
    """

    def __init__(
        self,
        encodings,
        labels,
    ):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {
            key: value[idx]
            for key, value in self.encodings.items()
        }

        item["labels"] = self.labels[idx]

        return item


# ============================================================
# 1. VERIFY CANONICAL SPLIT FILES
# ============================================================

print("\n" + "=" * 70)
print("MENTALROBERTA FIVE-CLASS FINE-TUNING")
print("=" * 70)

print(f"\nRandom seed: {SEED}")
print(f"Base model: {MODEL_NAME}")
print(f"Hardware detected: {detect_hardware()}")

for required_file in [
    TRAIN_FILE,
    VALIDATION_FILE,
    SPLIT_MANIFEST_FILE,
]:
    if not required_file.exists():
        raise FileNotFoundError(
            f"Required file not found: {required_file}"
        )

print("\nCanonical split files found.")


# ============================================================
# 2. VERIFY SPLIT HASHES AGAINST MANIFEST
# ============================================================

with open(
    SPLIT_MANIFEST_FILE,
    "r",
    encoding="utf-8",
) as f:
    split_manifest = json.load(f)

expected_train_hash = (
    split_manifest
    .get("output_file_sha256", {})
    .get("train.csv")
)

expected_validation_hash = (
    split_manifest
    .get("output_file_sha256", {})
    .get("validation.csv")
)

actual_train_hash = sha256_file(TRAIN_FILE)
actual_validation_hash = sha256_file(
    VALIDATION_FILE
)

if expected_train_hash is not None:
    assert (
        actual_train_hash == expected_train_hash
    ), (
        "train.csv no longer matches the canonical "
        "split recorded in split_manifest.json."
    )

if expected_validation_hash is not None:
    assert (
        actual_validation_hash
        == expected_validation_hash
    ), (
        "validation.csv no longer matches the canonical "
        "split recorded in split_manifest.json."
    )

print("Split hashes verified.")


# ============================================================
# 3. LOAD TRAIN + VALIDATION ONLY
# ============================================================

print("\nLoading training and validation data...")

train_df = pd.read_csv(TRAIN_FILE)

validation_df = pd.read_csv(
    VALIDATION_FILE
)

print(
    f"Training rows:   {len(train_df):,}"
)

print(
    f"Validation rows: {len(validation_df):,}"
)

# IMPORTANT:
#
# test.csv is deliberately NOT loaded here.
# It stays untouched until the final evaluation script.


# ============================================================
# 4. DATA INTEGRITY CHECKS
# ============================================================

required_columns = {
    "ID",
    "users",
    "text",
    "sentiment",
}

for name, dataframe in [
    ("train", train_df),
    ("validation", validation_df),
]:
    missing_columns = (
        required_columns - set(dataframe.columns)
    )

    assert not missing_columns, (
        f"{name} is missing columns: "
        f"{missing_columns}"
    )

    assert dataframe["ID"].is_unique, (
        f"Duplicate IDs found in {name}."
    )

    assert dataframe["text"].notna().all(), (
        f"Missing text found in {name}."
    )

    assert dataframe["sentiment"].notna().all(), (
        f"Missing labels found in {name}."
    )

    unknown_labels = (
        set(dataframe["sentiment"].unique())
        - set(LABEL2ID.keys())
    )

    assert not unknown_labels, (
        f"Unknown labels in {name}: "
        f"{unknown_labels}"
    )

# Ensure train and validation remain disjoint at both the post and user levels.

assert set(train_df["ID"]).isdisjoint(
    set(validation_df["ID"])
), "Train and validation post IDs overlap."

assert train_df["users"].notna().all(), (
    "Missing user IDs found in train."
)

assert validation_df["users"].notna().all(), (
    "Missing user IDs found in validation."
)

train_users = set(train_df["users"])
validation_users = set(validation_df["users"])

assert train_users.isdisjoint(validation_users), (
    "CRITICAL: Train and validation USERS overlap."
)

assert split_manifest.get("split_unit") == "user", (
    "split_manifest.json does not describe a user-level split."
)

assert split_manifest.get("group_column") == "users", (
    "split_manifest.json does not identify 'users' as the grouping column."
)

manifest_overlap = split_manifest.get("user_overlap_counts", {})

assert manifest_overlap.get("train_validation") == 0, (
    "Manifest reports train/validation user overlap."
)

assert manifest_overlap.get("train_test") == 0, (
    "Manifest reports train/test user overlap."
)

assert manifest_overlap.get("validation_test") == 0, (
    "Manifest reports validation/test user overlap."
)

print("Data integrity checks passed.")
print(f"Unique train users:      {train_df['users'].nunique():,}")
print(f"Unique validation users: {validation_df['users'].nunique():,}")
print("Train/validation user overlap: 0")


# ============================================================
# 5. CONVERT LABELS TO INTEGER IDs
# ============================================================

train_labels = (
    train_df["sentiment"]
    .map(LABEL2ID)
    .astype(int)
    .tolist()
)

validation_labels = (
    validation_df["sentiment"]
    .map(LABEL2ID)
    .astype(int)
    .tolist()
)


# ============================================================
# 6. CALCULATE BALANCED CLASS WEIGHTS
# ============================================================

# IMPORTANT:
#
# Class weights are calculated ONLY from train.csv.
# Validation and test data have no influence.

train_labels_np = np.array(
    train_labels,
    dtype=np.int64,
)

classes = np.arange(
    NUM_LABELS,
    dtype=np.int64,
)

class_weights_np = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=train_labels_np,
)

class_weights = torch.tensor(
    class_weights_np,
    dtype=torch.float32,
)

print("\nTraining label distribution:")

training_counts = (
    train_df["sentiment"]
    .value_counts()
    .reindex(LABEL2ID.keys())
)

print(training_counts)

print("\nCalculated class weights:")

for label, label_id in LABEL2ID.items():
    print(
        f"{label:14s}: "
        f"{class_weights_np[label_id]:.6f}"
    )


# ============================================================
# 7. USE THE SAME PINNED HUGGING FACE MODEL REVISION
# ============================================================

# Keep the exact same base-model revision as the original experiment.
# This makes the user-disjoint split the substantive experimental change.

print("\nUsing pinned Hugging Face model revision...")
print(f"Model revision: {MODEL_REVISION}")


# ============================================================
# 8. LOAD TOKENIZER
# ============================================================

print("\nLoading MentalRoBERTa tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    revision=MODEL_REVISION,
)


# ============================================================
# 9. TOKENIZE TRAIN + VALIDATION
# ============================================================

print(
    "\nTokenizing training data..."
)

train_encodings = tokenizer(
    train_df["text"].astype(str).tolist(),
    truncation=True,
    max_length=MAX_LENGTH,
    padding=False,
)

print(
    "Tokenizing validation data..."
)

validation_encodings = tokenizer(
    validation_df["text"]
    .astype(str)
    .tolist(),
    truncation=True,
    max_length=MAX_LENGTH,
    padding=False,
)


train_dataset = MentalHealthDataset(
    train_encodings,
    train_labels,
)

validation_dataset = MentalHealthDataset(
    validation_encodings,
    validation_labels,
)


# Dynamic padding:
#
# Each batch is padded only to its longest post,
# rather than padding every post to 512 tokens.
# This saves substantial memory/computation.

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer
)


# ============================================================
# 10. LOAD BASE MODEL + FIVE-CLASS HEAD
# ============================================================

print(
    "\nLoading MentalRoBERTa model..."
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    revision=MODEL_REVISION,
    num_labels=NUM_LABELS,
    label2id=LABEL2ID,
    id2label=ID2LABEL,
)

print(
    "\nFive-class classification head initialized."
)


# ============================================================
# 11. DEFINE WEIGHTED CROSS-ENTROPY LOSS
# ============================================================

def weighted_loss_function(
    outputs,
    labels,
    num_items_in_batch=None,
):
    """
    Balanced weighted cross-entropy.

    Rare classes such as Attempt receive greater
    training weight than common classes such as
    Unsupportive.

    Hugging Face Trainer passes:
        outputs
        labels
        num_items_in_batch
    """

    logits = outputs["logits"]

    weights = class_weights.to(
        device=logits.device,
        dtype=logits.dtype,
    )

    # Use summed loss so gradient accumulation can be
    # normalized across the complete accumulated batch.

    loss_sum = F.cross_entropy(
        logits,
        labels,
        weight=weights,
        reduction="sum",
    )

    if num_items_in_batch is None:
        denominator = labels.numel()
    elif torch.is_tensor(num_items_in_batch):
        denominator = num_items_in_batch.to(
            device=loss_sum.device,
            dtype=loss_sum.dtype,
        )
    else:
        denominator = float(
            num_items_in_batch
        )

    return loss_sum / denominator


# ============================================================
# 12. VALIDATION METRICS
# ============================================================

def compute_metrics(eval_prediction):
    """
    Metrics used during validation.

    Macro F1 is the PRIMARY metric and is used
    to select the best checkpoint.
    """

    logits, labels = eval_prediction

    predictions = np.argmax(
        logits,
        axis=-1,
    )

    accuracy = accuracy_score(
        labels,
        predictions,
    )

    (
        macro_precision,
        macro_recall,
        macro_f1,
        _,
    ) = precision_recall_fscore_support(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )

    (
        _,
        _,
        weighted_f1,
        _,
    ) = precision_recall_fscore_support(
        labels,
        predictions,
        average="weighted",
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
        "weighted_f1": float(
            weighted_f1
        ),
    }


# ============================================================
# 13. TRAINING ARGUMENTS
# ============================================================

training_args = TrainingArguments(
    # --------------------------------------------
    # Output
    # --------------------------------------------

    output_dir=str(
        CHECKPOINT_DIR
    ),

    # --------------------------------------------
    # Reproducibility
    # --------------------------------------------

    seed=SEED,
    data_seed=SEED,

    # --------------------------------------------
    # Training
    # --------------------------------------------

    learning_rate=LEARNING_RATE,

    num_train_epochs=NUM_TRAIN_EPOCHS,

    per_device_train_batch_size=(
        TRAIN_BATCH_SIZE
    ),

    per_device_eval_batch_size=(
        EVAL_BATCH_SIZE
    ),

    gradient_accumulation_steps=(
        GRADIENT_ACCUMULATION_STEPS
    ),

    weight_decay=WEIGHT_DECAY,

    warmup_ratio=WARMUP_RATIO,

    lr_scheduler_type="linear",

    optim="adamw_torch",

    max_grad_norm=1.0,

    # --------------------------------------------
    # Validation
    # --------------------------------------------

    eval_strategy="epoch",

    # --------------------------------------------
    # Checkpointing
    # --------------------------------------------

    save_strategy="epoch",

    save_total_limit=2,

    load_best_model_at_end=True,

    metric_for_best_model="macro_f1",

    greater_is_better=True,

    # --------------------------------------------
    # Logging
    # --------------------------------------------

    logging_strategy="steps",

    logging_steps=50,

    report_to="none",

    # --------------------------------------------
    # Memory / Mac compatibility
    # --------------------------------------------

    eval_accumulation_steps=16,

    dataloader_num_workers=0,

    dataloader_pin_memory=False,

    # Keep full float32 for stability on MPS.
    fp16=False,
    bf16=False,

    # Save model using safetensors.
    #save_safetensors=True,
)


# ============================================================
# 14. CREATE TRAINER
# ============================================================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    compute_loss_func=weighted_loss_function,
    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=(
                EARLY_STOPPING_PATIENCE
            )
        )
    ],
)


print(
    f"\nTrainer device: {trainer.args.device}"
)

effective_batch_size = (
    TRAIN_BATCH_SIZE
    * GRADIENT_ACCUMULATION_STEPS
)

print(
    f"Effective training batch size: "
    f"{effective_batch_size}"
)


# ============================================================
# 15. SAVE PRE-TRAINING MANIFEST
# ============================================================

training_manifest = {
    "experiment": (
        "MentalRoBERTa five-class "
        "suicidality classification"
    ),

    "created_utc": datetime.now(
        timezone.utc
    ).isoformat(),

    "base_model": MODEL_NAME,

    "base_model_revision": MODEL_REVISION,

    "task": "five-class sequence classification",

    "split_method": "user-disjoint",

    "group_column": "users",

    "train_unique_users": int(train_df["users"].nunique()),

    "validation_unique_users": int(validation_df["users"].nunique()),

    "train_validation_user_overlap": 0,

    "labels": LABEL2ID,

    "primary_validation_metric": "macro_f1",

    "random_seed": SEED,

    "train_file": str(TRAIN_FILE),

    "validation_file": str(
        VALIDATION_FILE
    ),

    "train_file_sha256": (
        actual_train_hash
    ),

    "validation_file_sha256": (
        actual_validation_hash
    ),

    "train_rows": int(
        len(train_df)
    ),

    "validation_rows": int(
        len(validation_df)
    ),

    "class_weights": {
        label: float(
            class_weights_np[label_id]
        )
        for label, label_id
        in LABEL2ID.items()
    },

    "hyperparameters": {
        "max_length": MAX_LENGTH,
        "learning_rate": LEARNING_RATE,
        "max_epochs": NUM_TRAIN_EPOCHS,
        "train_batch_size": (
            TRAIN_BATCH_SIZE
        ),
        "eval_batch_size": (
            EVAL_BATCH_SIZE
        ),
        "gradient_accumulation_steps": (
            GRADIENT_ACCUMULATION_STEPS
        ),
        "effective_train_batch_size": (
            effective_batch_size
        ),
        "weight_decay": WEIGHT_DECAY,
        "warmup_ratio": WARMUP_RATIO,
        "scheduler": "linear",
        "optimizer": "adamw_torch",
        "early_stopping_patience": (
            EARLY_STOPPING_PATIENCE
        ),
        "loss": (
            "balanced weighted "
            "cross-entropy"
        ),
    },

    "hardware": {
        "detected_accelerator": (
            detect_hardware()
        ),
        "trainer_device": str(
            trainer.args.device
        ),
        "machine": (
            platform.machine()
        ),
        "platform": (
            platform.platform()
        ),
    },

    "software_versions": {
        "python": (
            platform.python_version()
        ),
        "torch": torch.__version__,
        "transformers": (
            transformers.__version__
        ),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit_learn": (
            sklearn.__version__
        ),
    },
}

training_manifest_path = (
    RESULTS_DIR
    / "training_manifest.json"
)

with open(
    training_manifest_path,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        training_manifest,
        f,
        indent=4,
        ensure_ascii=False,
    )

print(
    "\nTraining manifest saved to:"
)

print(
    training_manifest_path.resolve()
)


# ============================================================
# 16. TRAIN
# ============================================================

print("\n" + "=" * 70)
print("STARTING FINE-TUNING")
print("=" * 70)

print(
    "\nNOTE: test.csv is NOT being used."
)

train_result = trainer.train()


# ============================================================
# 17. SAVE TRAINING METRICS
# ============================================================

trainer.log_metrics(
    "train",
    train_result.metrics,
)

trainer.save_metrics(
    "train",
    train_result.metrics,
)

trainer.save_state()


# ============================================================
# 18. FINAL VALIDATION OF BEST CHECKPOINT
# ============================================================

print("\n" + "=" * 70)
print("BEST CHECKPOINT VALIDATION")
print("=" * 70)

validation_metrics = trainer.evaluate()

trainer.log_metrics(
    "validation",
    validation_metrics,
)

trainer.save_metrics(
    "validation",
    validation_metrics,
)


# ============================================================
# 19. SAVE BEST MODEL
# ============================================================

FINAL_MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

trainer.save_model(
    str(FINAL_MODEL_DIR)
)

tokenizer.save_pretrained(
    str(FINAL_MODEL_DIR)
)


# ============================================================
# 20. SAVE LABEL MAPPING
# ============================================================

label_mapping_path = (
    FINAL_MODEL_DIR
    / "label_mapping.json"
)

with open(
    label_mapping_path,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        {
            "label2id": LABEL2ID,
            "id2label": ID2LABEL,
        },
        f,
        indent=4,
    )


# ============================================================
# 21. SAVE FINAL TRAINING SUMMARY
# ============================================================

summary = {
    "base_model": MODEL_NAME,

    "base_model_revision": (
        MODEL_REVISION
    ),

    "random_seed": SEED,

    "best_checkpoint": (
        trainer.state.best_model_checkpoint
    ),

    "best_validation_macro_f1": (
        trainer.state.best_metric
    ),

    "validation_metrics": {
        key: float(value)
        for key, value
        in validation_metrics.items()
        if isinstance(
            value,
            (int, float),
        )
    },

    "class_weights": {
        label: float(
            class_weights_np[label_id]
        )
        for label, label_id
        in LABEL2ID.items()
    },

    "final_model_directory": str(
        FINAL_MODEL_DIR
    ),

    "test_set_used": False,
}

summary_path = (
    RESULTS_DIR
    / "training_summary.json"
)

with open(
    summary_path,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        summary,
        f,
        indent=4,
        ensure_ascii=False,
    )


# ============================================================
# 22. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("FINE-TUNING COMPLETE")
print("=" * 70)

print(
    "\nBest checkpoint:"
)

print(
    trainer.state.best_model_checkpoint
)

print(
    "\nBest validation Macro F1:"
)

print(
    trainer.state.best_metric
)

print(
    "\nFinal model saved to:"
)

print(
    FINAL_MODEL_DIR.resolve()
)

print(
    "\nTraining summary saved to:"
)

print(
    summary_path.resolve()
)

print(
    "\nIMPORTANT:"
)

print(
    "test.csv has NOT been evaluated."
)

print(
    "The model is now frozen and ready "
    "for the final test-partition evaluation."
)
# ============================================================
# GESTURE CONTROLLED SYSTEM
# MODEL TRAINING
# ============================================================

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATASET_FILE = (
    PROJECT_ROOT
    / "system"
    / "dataset"
    / "gesture_dataset.csv"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
)

MODEL_FILE = (
    MODEL_DIR
    / "gesture_model.pkl"
)

METRICS_FILE = (
    MODEL_DIR
    / "training_metrics.json"
)

CONFUSION_FILE = (
    MODEL_DIR
    / "confusion_matrix.png"
)


# ============================================================
# CONFIG
# ============================================================

RANDOM_STATE = 42

TEST_SIZE = 0.20

EXPECTED_SAMPLES = 3500

EXPECTED_FEATURES = 42

EXPECTED_CLASSES = [
    "INDEX",
    "PINCH",
    "TWO_FINGERS",
    "OPEN_PALM",
    "FIST",
    "THUMBS_UP",
    "THUMBS_DOWN",
]


# ============================================================
# PRINT HEADER
# ============================================================

def print_header():

    print()
    print("=" * 70)
    print(
        "             GESTURE MODEL TRAINING"
    )
    print("=" * 70)
    print()


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():

    print("[1/6] Loading dataset...")

    if not DATASET_FILE.exists():

        raise FileNotFoundError(
            f"\nDataset not found:\n"
            f"{DATASET_FILE}\n"
        )

    df = pd.read_csv(
        DATASET_FILE
    )

    print(
        f"      Rows found: {len(df)}"
    )

    print(
        f"      Columns found: {len(df.columns)}"
    )

    return df


# ============================================================
# VALIDATE DATASET
# ============================================================

def validate_dataset(df):

    print()
    print("[2/6] Validating dataset...")

    # --------------------------------------------------------
    # Check label column
    # --------------------------------------------------------

    if "label" not in df.columns:

        raise ValueError(
            "Dataset must contain a 'label' column."
        )

    # --------------------------------------------------------
    # Feature columns
    # --------------------------------------------------------

    feature_columns = [
        column
        for column in df.columns
        if column != "label"
    ]

    if len(feature_columns) != EXPECTED_FEATURES:

        raise ValueError(
            f"Expected {EXPECTED_FEATURES} "
            f"features, found {len(feature_columns)}."
        )

    # --------------------------------------------------------
    # Sample count
    # --------------------------------------------------------

    if len(df) != EXPECTED_SAMPLES:

        print(
            f"      WARNING: Expected "
            f"{EXPECTED_SAMPLES} samples."
        )

        print(
            f"      Found {len(df)} samples."
        )

    # --------------------------------------------------------
    # Check classes
    # --------------------------------------------------------

    classes = sorted(
        df["label"]
        .astype(str)
        .unique()
        .tolist()
    )

    expected_sorted = sorted(
        EXPECTED_CLASSES
    )

    if classes != expected_sorted:

        raise ValueError(
            "\nUnexpected gesture classes.\n"
            f"Expected: {expected_sorted}\n"
            f"Found:    {classes}\n"
        )

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    missing = int(
        df.isnull().sum().sum()
    )

    if missing > 0:

        raise ValueError(
            f"Dataset contains {missing} "
            f"missing values."
        )

    # --------------------------------------------------------
    # Numeric features
    # --------------------------------------------------------

    X = df[
        feature_columns
    ].apply(
        pd.to_numeric,
        errors="coerce"
    )

    if X.isnull().any().any():

        raise ValueError(
            "Some feature values are not numeric."
        )

    # --------------------------------------------------------
    # Infinite values
    # --------------------------------------------------------

    if not np.isfinite(
        X.to_numpy(
            dtype=np.float32
        )
    ).all():

        raise ValueError(
            "Dataset contains NaN or infinite values."
        )

    # --------------------------------------------------------
    # Class distribution
    # --------------------------------------------------------

    print()
    print("      Class distribution:")

    counts = (
        df["label"]
        .value_counts()
        .reindex(
            EXPECTED_CLASSES
        )
    )

    for label, count in counts.items():

        print(
            f"        {label:<15} {count}"
        )

    print()

    # --------------------------------------------------------
    # Balance check
    # --------------------------------------------------------

    if counts.min() != counts.max():

        print(
            "      WARNING: Dataset is not perfectly balanced."
        )

    else:

        print(
            "      ✓ Dataset is perfectly balanced."
        )

    return (
        X.to_numpy(
            dtype=np.float32
        ),
        df["label"].astype(str).to_numpy(),
        feature_columns
    )


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(X, y):

    print()
    print("[3/6] Splitting dataset...")

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y
        )
    )

    print(
        f"      Training samples: {len(X_train)}"
    )

    print(
        f"      Testing samples:  {len(X_test)}"
    )

    print()
    print("[4/6] Training SVM model...")
    print()
    print(
        "      Please wait..."
    )

    # ========================================================
    # PIPELINE
    # ========================================================
    #
    # StandardScaler
    #       ↓
    # SVM RBF
    #
    # Saving the entire pipeline means the inference code
    # automatically gets the same preprocessing.
    #
    # ========================================================

    model = Pipeline(
        [
            (
                "scaler",
                StandardScaler()
            ),

            (
                "classifier",
                SVC(
                    kernel="rbf",
                    C=10.0,
                    gamma="scale",
                    probability=True,
                    class_weight=None,
                    random_state=RANDOM_STATE
                )
            )
        ]
    )

    model.fit(
        X_train,
        y_train
    )

    print(
        "      ✓ Training completed."
    )

    return (
        model,
        X_train,
        X_test,
        y_train,
        y_test
    )


# ============================================================
# EVALUATE
# ============================================================

def evaluate_model(
    model,
    X_train,
    X_test,
    y_train,
    y_test
):

    print()
    print("[5/6] Evaluating model...")

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    train_predictions = (
        model.predict(
            X_train
        )
    )

    test_predictions = (
        model.predict(
            X_test
        )
    )

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    train_accuracy = (
        accuracy_score(
            y_train,
            train_predictions
        )
    )

    test_accuracy = (
        accuracy_score(
            y_test,
            test_predictions
        )
    )

    print()
    print(
        f"      Training Accuracy: "
        f"{train_accuracy * 100:.2f}%"
    )

    print(
        f"      Testing Accuracy:  "
        f"{test_accuracy * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    report = classification_report(
        y_test,
        test_predictions,
        labels=EXPECTED_CLASSES,
        output_dict=True,
        zero_division=0
    )

    print()
    print(
        "      Classification Report"
    )

    print(
        "-" * 70
    )

    print(
        classification_report(
            y_test,
            test_predictions,
            labels=EXPECTED_CLASSES,
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        test_predictions,
        labels=EXPECTED_CLASSES
    )

    return (
        train_accuracy,
        test_accuracy,
        report,
        cm
    )


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

def save_confusion_matrix(cm):

    print(
        "      Creating confusion matrix..."
    )

    plt.figure(
        figsize=(10, 8)
    )

    plt.imshow(
        cm,
        interpolation="nearest"
    )

    plt.title(
        "Gesture Recognition Confusion Matrix"
    )

    plt.colorbar()

    tick_positions = np.arange(
        len(EXPECTED_CLASSES)
    )

    plt.xticks(
        tick_positions,
        EXPECTED_CLASSES,
        rotation=45,
        ha="right"
    )

    plt.yticks(
        tick_positions,
        EXPECTED_CLASSES
    )

    plt.xlabel(
        "Predicted Gesture"
    )

    plt.ylabel(
        "Actual Gesture"
    )

    # --------------------------------------------------------
    # Numbers inside matrix
    # --------------------------------------------------------

    threshold = (
        cm.max() / 2.0
        if cm.size
        else 0
    )

    for i in range(
        cm.shape[0]
    ):

        for j in range(
            cm.shape[1]
        ):

            plt.text(
                j,
                i,
                str(cm[i, j]),
                horizontalalignment="center",
                verticalalignment="center",
                color=(
                    "white"
                    if cm[i, j] > threshold
                    else "black"
                )
            )

    plt.tight_layout()

    plt.savefig(
        CONFUSION_FILE,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"      ✓ Saved: {CONFUSION_FILE}"
    )


# ============================================================
# SAVE MODEL
# ============================================================

def save_model(model):

    print()
    print(
        "      Saving trained model..."
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Remove old model first.

    if MODEL_FILE.exists():

        MODEL_FILE.unlink()

    joblib.dump(
        model,
        MODEL_FILE
    )

    print(
        f"      ✓ Saved: {MODEL_FILE}"
    )


# ============================================================
# SAVE METRICS
# ============================================================

def save_metrics(
    train_accuracy,
    test_accuracy,
    report
):

    metrics = {

        "dataset": {
            "file": str(
                DATASET_FILE
            ),
            "samples": EXPECTED_SAMPLES,
            "features": EXPECTED_FEATURES,
            "classes": EXPECTED_CLASSES
        },

        "training": {
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "model": "SVM RBF",
            "C": 10.0,
            "gamma": "scale"
        },

        "accuracy": {
            "training": round(
                float(train_accuracy),
                6
            ),
            "testing": round(
                float(test_accuracy),
                6
            )
        },

        "classification_report": report
    }

    with open(
        METRICS_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4
        )

    print(
        f"      ✓ Saved: {METRICS_FILE}"
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

def print_summary(
    train_accuracy,
    test_accuracy
):

    print()
    print("=" * 70)
    print(
        "                  TRAINING COMPLETE"
    )
    print("=" * 70)

    print()

    print(
        f"Training Accuracy : "
        f"{train_accuracy * 100:.2f}%"
    )

    print(
        f"Testing Accuracy  : "
        f"{test_accuracy * 100:.2f}%"
    )

    print()

    print(
        f"Model:"
    )

    print(
        MODEL_FILE
    )

    print()

    print(
        f"Metrics:"
    )

    print(
        METRICS_FILE
    )

    print()

    print(
        f"Confusion Matrix:"
    )

    print(
        CONFUSION_FILE
    )

    print()

    if test_accuracy >= 0.95:

        print(
            "✓ EXCELLENT MODEL"
        )

    elif test_accuracy >= 0.90:

        print(
            "✓ GOOD MODEL"
        )

    elif test_accuracy >= 0.80:

        print(
            "⚠ MODEL IS USABLE, BUT CAN BE IMPROVED"
        )

    else:

        print(
            "⚠ ACCURACY IS LOW — CHECK DATASET / FEATURES"
        )

    print()

    print(
        "Next step: connect this fresh model to main.py."
    )

    print()

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():

    print_header()

    try:

        # ----------------------------------------------------
        # Load
        # ----------------------------------------------------

        df = load_dataset()

        # ----------------------------------------------------
        # Validate
        # ----------------------------------------------------

        (
            X,
            y,
            feature_columns
        ) = validate_dataset(
            df
        )

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        (
            model,
            X_train,
            X_test,
            y_train,
            y_test
        ) = train_model(
            X,
            y
        )

        # ----------------------------------------------------
        # Evaluate
        # ----------------------------------------------------

        (
            train_accuracy,
            test_accuracy,
            report,
            cm
        ) = evaluate_model(
            model,
            X_train,
            X_test,
            y_train,
            y_test
        )

        # ----------------------------------------------------
        # Save confusion matrix
        # ----------------------------------------------------

        save_confusion_matrix(
            cm
        )

        # ----------------------------------------------------
        # Save model
        # ----------------------------------------------------

        save_model(
            model
        )

        # ----------------------------------------------------
        # Save metrics
        # ----------------------------------------------------

        save_metrics(
            train_accuracy,
            test_accuracy,
            report
        )

        # ----------------------------------------------------
        # Summary
        # ----------------------------------------------------

        print_summary(
            train_accuracy,
            test_accuracy
        )

    except Exception as error:

        print()
        print("=" * 70)
        print(
            "                    ERROR"
        )
        print("=" * 70)

        print()

        print(
            str(error)
        )

        print()

        print(
            "Training stopped."
        )

        print(
            "Fix the above error and run again."
        )

        print()

        raise


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    main()
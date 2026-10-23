"""
SignaSense — Model Training Script
====================================
Loads processed data, builds the LSTM model, trains it with early stopping
and checkpointing, evaluates on the test set, and saves a classification report.

Usage:
    conda activate signasense
    python src/train.py
    python src/train.py --epochs 300 --batch-size 16
"""

import os
import sys
import json
import argparse

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config.settings import (
    DATA_PROCESSED_DIR,
    MODEL_CHECKPOINT_DIR,
    BEST_MODEL_PATH,
    LABEL_MAP_PATH,
    BATCH_SIZE,
    EPOCHS,
    EARLY_STOPPING_PATIENCE,
)
from src.model import build_model


def load_processed_data():
    """Load the preprocessed train/val/test splits."""
    data = {}
    for split in ["train", "val", "test"]:
        X = np.load(os.path.join(DATA_PROCESSED_DIR, f"X_{split}.npy"))
        y = np.load(os.path.join(DATA_PROCESSED_DIR, f"y_{split}.npy"))
        data[split] = (X, y)
        print(f"  {split:5s}: X={X.shape}, y={y.shape}")
    return data


def train(epochs: int = EPOCHS, batch_size: int = BATCH_SIZE):
    """Full training pipeline."""
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    from sklearn.metrics import classification_report, confusion_matrix

    print("=" * 60)
    print("  SignaSense — Model Training")
    print("=" * 60)

    # 1. Load data
    print("\n[1/5] Loading processed data...")
    data = load_processed_data()
    X_train, y_train = data["train"]
    X_val, y_val = data["val"]
    X_test, y_test = data["test"]

    # 2. Load label map
    print("\n[2/5] Loading label map...")
    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        label_map = json.load(f)
    num_classes = len(label_map)
    class_names = [label_map[str(i)] for i in range(num_classes)]
    print(f"  Classes ({num_classes}): {class_names}")

    # 3. Build model
    print("\n[3/5] Building LSTM model...")
    model = build_model(num_classes)
    model.summary()

    # 4. Train
    print(f"\n[4/5] Training for up to {epochs} epochs (early stopping patience={EARLY_STOPPING_PATIENCE})...")
    os.makedirs(MODEL_CHECKPOINT_DIR, exist_ok=True)

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=BEST_MODEL_PATH,
            monitor="val_categorical_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    # 5. Evaluate on test set
    print("\n[5/5] Evaluating on test set...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n  Test Loss     : {test_loss:.4f}")
    print(f"  Test Accuracy : {test_accuracy:.4f} ({test_accuracy * 100:.1f}%)")

    # Classification report
    y_pred = model.predict(X_test, verbose=0)
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_true_labels = np.argmax(y_test, axis=1)

    print("\n" + "=" * 60)
    print("  Classification Report")
    print("=" * 60)
    report = classification_report(
        y_true_labels, y_pred_labels,
        target_names=class_names,
        digits=3,
        zero_division=0,
    )
    print(report)

    # Save the report to a file
    report_path = os.path.join(MODEL_CHECKPOINT_DIR, "classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Test Loss     : {test_loss:.4f}\n")
        f.write(f"Test Accuracy : {test_accuracy:.4f}\n\n")
        f.write(report)
    print(f"[INFO] Classification report saved to: {report_path}")

    # Confusion matrix
    cm = confusion_matrix(y_true_labels, y_pred_labels)
    cm_path = os.path.join(MODEL_CHECKPOINT_DIR, "confusion_matrix.npy")
    np.save(cm_path, cm)
    print(f"[INFO] Confusion matrix saved to: {cm_path}")

    # Save training history
    history_path = os.path.join(MODEL_CHECKPOINT_DIR, "training_history.json")
    history_dict = {
        key: [float(v) for v in values]
        for key, values in history.history.items()
    }
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history_dict, f, indent=2)
    print(f"[INFO] Training history saved to: {history_path}")

    print(f"\n[INFO] Best model saved to: {BEST_MODEL_PATH}")
    print("[INFO] Training complete!")

    return model, history


def main():
    parser = argparse.ArgumentParser(description="SignaSense Model Training")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help=f"Max epochs (default: {EPOCHS})")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help=f"Batch size (default: {BATCH_SIZE})")
    args = parser.parse_args()
    train(epochs=args.epochs, batch_size=args.batch_size)


if __name__ == "__main__":
    main()

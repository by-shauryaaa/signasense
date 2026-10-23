"""
SignaSense — Data Preprocessor
================================
Loads raw .npy sequences from data/raw/, encodes labels, applies optional
augmentation, splits into train/val/test sets, and saves processed arrays.

Usage:
    conda activate signasense
    python src/data_preprocessor.py
    python src/data_preprocessor.py --augment
"""

import os
import sys
import json
import argparse

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config.settings import (
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
    SEQUENCE_LENGTH,
    TOTAL_FEATURES,
    TRAIN_SPLIT,
    VAL_SPLIT,
    LABEL_MAP_PATH,
    POSE_FEATURES,
    HAND_FEATURES,
)


def load_raw_data():
    """
    Walk data/raw/<sign_name>/*.npy and load all sequences.

    Returns
    -------
    sequences : list[np.ndarray]
        Each element is shape (SEQUENCE_LENGTH, TOTAL_FEATURES).
    labels : list[str]
        Corresponding sign name for each sequence.
    """
    sequences = []
    labels = []
    skipped = 0

    sign_dirs = sorted(
        [d for d in os.listdir(DATA_RAW_DIR) if os.path.isdir(os.path.join(DATA_RAW_DIR, d))]
    )

    if not sign_dirs:
        print(f"[ERROR] No sign directories found in {DATA_RAW_DIR}")
        print("        Run data_collector.py first to record samples.")
        sys.exit(1)

    for sign_name in sign_dirs:
        sign_path = os.path.join(DATA_RAW_DIR, sign_name)
        npy_files = sorted([f for f in os.listdir(sign_path) if f.endswith(".npy")])

        for npy_file in npy_files:
            filepath = os.path.join(sign_path, npy_file)
            arr = np.load(filepath)

            if arr.shape == (SEQUENCE_LENGTH, TOTAL_FEATURES):
                sequences.append(arr)
                labels.append(sign_name)
            else:
                print(f"  [WARN] Skipping {filepath} — shape {arr.shape} != expected ({SEQUENCE_LENGTH}, {TOTAL_FEATURES})")
                skipped += 1

    print(f"[INFO] Loaded {len(sequences)} sequences across {len(sign_dirs)} signs. Skipped: {skipped}")
    return sequences, labels


def create_label_map(labels: list[str]) -> dict:
    """
    Create a bidirectional label map: index ↔ sign name.

    Returns
    -------
    label_map : dict
        {"0": "hello", "1": "thank_you", ...}
    label_to_idx : dict
        {"hello": 0, "thank_you": 1, ...}
    """
    unique_labels = sorted(set(labels))
    label_map = {str(i): name for i, name in enumerate(unique_labels)}
    label_to_idx = {name: i for i, name in enumerate(unique_labels)}
    return label_map, label_to_idx


def augment_sequence(sequence: np.ndarray) -> list[np.ndarray]:
    """
    Generate augmented copies of a single sequence.

    Augmentations applied:
        1. Gaussian noise (σ = 0.005)
        2. Temporal shift (roll ±2 frames)
        3. Coordinate scaling (×0.9 to ×1.1)
        4. Horizontal mirror (negate x-coords, swap left↔right hand)

    Parameters
    ----------
    sequence : np.ndarray of shape (SEQUENCE_LENGTH, TOTAL_FEATURES)

    Returns
    -------
    augmented : list of np.ndarray
        List of augmented sequences (does NOT include the original).
    """
    augmented = []

    # 1. Gaussian noise
    noisy = sequence + np.random.normal(0, 0.005, sequence.shape)
    augmented.append(noisy)

    # 2. Temporal shift (+2 frames)
    shifted = np.roll(sequence, 2, axis=0)
    augmented.append(shifted)

    # 3. Coordinate scaling
    scale = np.random.uniform(0.9, 1.1)
    scaled = sequence * scale
    augmented.append(scaled)

    # 4. Horizontal mirror: negate x-coordinates, swap left and right hands
    mirrored = sequence.copy()
    # Pose x-coordinates are at indices 0, 4, 8, ... (every 4th starting from 0)
    for i in range(0, POSE_FEATURES, 4):  # x is every 4th value in pose section
        mirrored[:, i] = 1.0 - mirrored[:, i]  # flip x across center (0.5)

    # Left hand x-coordinates: indices 132, 135, 138, ... (every 3rd starting from 132)
    # Right hand x-coordinates: indices 195, 198, 201, ... (every 3rd starting from 195)
    lh_start = POSE_FEATURES
    rh_start = POSE_FEATURES + HAND_FEATURES
    for i in range(0, HAND_FEATURES, 3):
        mirrored[:, lh_start + i] = 1.0 - mirrored[:, lh_start + i]
        mirrored[:, rh_start + i] = 1.0 - mirrored[:, rh_start + i]

    # Swap left and right hand data
    left_data = mirrored[:, lh_start:rh_start].copy()
    right_data = mirrored[:, rh_start:rh_start + HAND_FEATURES].copy()
    mirrored[:, lh_start:rh_start] = right_data
    mirrored[:, rh_start:rh_start + HAND_FEATURES] = left_data
    augmented.append(mirrored)

    return augmented


def preprocess(use_augmentation: bool = False):
    """
    Full preprocessing pipeline:
    1. Load raw .npy data
    2. Build label map
    3. Optionally augment
    4. One-hot encode labels
    5. Stratified split into train / val / test
    6. Save everything to data/processed/
    """
    # 1. Load
    sequences, labels = load_raw_data()

    # 2. Label map
    label_map, label_to_idx = create_label_map(labels)
    num_classes = len(label_map)
    print(f"[INFO] Number of classes: {num_classes}")
    print(f"[INFO] Classes: {list(label_to_idx.keys())}")

    # 3. Augment (optional)
    if use_augmentation:
        aug_sequences = []
        aug_labels = []
        for seq, lbl in zip(sequences, labels):
            aug_sequences.append(seq)
            aug_labels.append(lbl)
            for aug_seq in augment_sequence(seq):
                aug_sequences.append(aug_seq)
                aug_labels.append(lbl)
        sequences = aug_sequences
        labels = aug_labels
        print(f"[INFO] After augmentation: {len(sequences)} sequences (×5 original)")

    # 4. Convert to arrays
    X = np.array(sequences, dtype=np.float32)
    y_indices = np.array([label_to_idx[lbl] for lbl in labels])

    # One-hot encode
    y = np.zeros((len(y_indices), num_classes), dtype=np.float32)
    y[np.arange(len(y_indices)), y_indices] = 1.0

    print(f"[INFO] X shape: {X.shape}")
    print(f"[INFO] y shape: {y.shape}")

    # 5. Stratified split: train (80%) / temp (20%) → then temp into val (50%) / test (50%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(1.0 - TRAIN_SPLIT), random_state=42, stratify=y_indices,
    )

    # Split the remaining 20% into validation and test (each 10% of total)
    temp_indices = np.argmax(y_temp, axis=1)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=temp_indices,
    )

    print(f"\n[INFO] Split sizes:")
    print(f"  Train : {X_train.shape[0]} samples")
    print(f"  Val   : {X_val.shape[0]} samples")
    print(f"  Test  : {X_test.shape[0]} samples")

    # 6. Save
    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)

    np.save(os.path.join(DATA_PROCESSED_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(DATA_PROCESSED_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(DATA_PROCESSED_DIR, "X_val.npy"), X_val)
    np.save(os.path.join(DATA_PROCESSED_DIR, "y_val.npy"), y_val)
    np.save(os.path.join(DATA_PROCESSED_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(DATA_PROCESSED_DIR, "y_test.npy"), y_test)

    with open(LABEL_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2, ensure_ascii=False)

    print(f"\n[INFO] Saved processed data to: {DATA_PROCESSED_DIR}")
    print(f"[INFO] Label map saved to: {LABEL_MAP_PATH}")


def main():
    parser = argparse.ArgumentParser(description="SignaSense Data Preprocessor")
    parser.add_argument(
        "--augment", action="store_true",
        help="Apply data augmentation (noise, shift, scale, mirror)",
    )
    args = parser.parse_args()
    preprocess(use_augmentation=args.augment)


if __name__ == "__main__":
    main()

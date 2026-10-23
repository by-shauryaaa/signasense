"""
SignaSense — Synthetic Data Generator
======================================
Generates synthetic landmark sequences for all signs to allow immediate
end-to-end testing of data preprocessing, model training, and inference
without requiring immediate manual webcam recording.

Usage:
    conda activate signasense
    python scripts/generate_synthetic_data.py
    python scripts/generate_synthetic_data.py --samples-per-sign 30
    python scripts/generate_synthetic_data.py --clean
"""

import os
import sys
import shutil
import argparse
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config.settings import (
    DATA_RAW_DIR,
    SEQUENCE_LENGTH,
    TOTAL_FEATURES,
    POSE_FEATURES,
    HAND_FEATURES,
    SIGNS,
)


def generate_sign_trajectory(sign_idx: int, sample_idx: int, is_idle: bool = False) -> np.ndarray:
    """
    Generate a 30x258 sequence with distinctive temporal and spatial patterns.
    """
    np.random.seed(sign_idx * 1000 + sample_idx)
    frames = np.zeros((SEQUENCE_LENGTH, TOTAL_FEATURES), dtype=np.float32)

    if is_idle:
        # Idle: hands resting, low variance noise around zero
        noise = np.random.normal(0.0, 0.01, size=(SEQUENCE_LENGTH, TOTAL_FEATURES)).astype(np.float32)
        # Add basic stationary upper body pose landmarks
        base_pose = np.linspace(0.4, 0.6, POSE_FEATURES, dtype=np.float32)
        frames[:, :POSE_FEATURES] = base_pose + noise[:, :POSE_FEATURES]
        frames[:, POSE_FEATURES:] = np.clip(noise[:, POSE_FEATURES:], -0.05, 0.05)
        return frames

    # Active sign: generate distinct frequency, phase, and amplitude per sign
    freq = 1.0 + (sign_idx % 4) * 0.5
    phase = (sign_idx * 0.7) % np.pi
    t = np.linspace(0, 2 * np.pi * freq, SEQUENCE_LENGTH)

    # Base pose (head and shoulders)
    base_pose = np.linspace(0.3, 0.7, POSE_FEATURES, dtype=np.float32)
    frames[:, :POSE_FEATURES] = base_pose + 0.02 * np.sin(t[:, None] + phase)

    # Left hand [132:195] and Right hand [195:258]
    lh_start = POSE_FEATURES
    lh_end = POSE_FEATURES + HAND_FEATURES
    rh_start = lh_end
    rh_end = TOTAL_FEATURES

    # Give each sign distinct spatial offsets and wave motions
    offset = (sign_idx + 1) * 0.05
    hand_wave = 0.15 * np.sin(t[:, None] * 1.5 + phase) + offset
    hand_noise = np.random.normal(0.0, 0.02, size=(SEQUENCE_LENGTH, HAND_FEATURES))

    # Depending on sign index, hand can be dominant right, dominant left, or two-handed
    if sign_idx % 3 == 0:
        # Two-handed sign
        frames[:, lh_start:lh_end] = np.clip(hand_wave + hand_noise, 0.0, 1.0)
        frames[:, rh_start:rh_end] = np.clip(hand_wave + hand_noise, 0.0, 1.0)
    elif sign_idx % 3 == 1:
        # Right hand dominant
        frames[:, lh_start:lh_end] = 0.0
        frames[:, rh_start:rh_end] = np.clip(hand_wave * 1.2 + hand_noise, 0.0, 1.0)
    else:
        # Left hand dominant
        frames[:, lh_start:lh_end] = np.clip(hand_wave * 1.2 + hand_noise, 0.0, 1.0)
        frames[:, rh_start:rh_end] = 0.0

    return frames.astype(np.float32)


def generate_dataset(samples_per_sign: int = 30, signs: list[str] = None):
    """Generate synthetic sequences for the specified signs."""
    target_signs = signs or SIGNS
    print("=" * 60)
    print("  SignaSense — Generating Synthetic Dataset")
    print("=" * 60)
    print(f"  Signs count       : {len(target_signs)}")
    print(f"  Samples per sign  : {samples_per_sign}")
    print(f"  Sequence shape    : ({SEQUENCE_LENGTH}, {TOTAL_FEATURES})")
    print(f"  Destination       : {DATA_RAW_DIR}")
    print("=" * 60)

    total_created = 0
    for sign_idx, sign in enumerate(target_signs):
        sign_dir = os.path.join(DATA_RAW_DIR, sign)
        os.makedirs(sign_dir, exist_ok=True)
        is_idle = (sign == "_idle_")

        for sample_idx in range(samples_per_sign):
            file_path = os.path.join(sign_dir, f"{sample_idx}.npy")
            seq = generate_sign_trajectory(sign_idx, sample_idx, is_idle=is_idle)
            np.save(file_path, seq)
            total_created += 1

        print(f"  [{sign_idx + 1:2d}/{len(target_signs):2d}] Generated {samples_per_sign} samples for '{sign}'")

    print("=" * 60)
    print(f"  Total created: {total_created} .npy sequence files")
    print("  Ready for preprocessing with: python src/data_preprocessor.py --augment")
    print("=" * 60)


def clean_dataset():
    """Remove all data from data/raw/."""
    if os.path.exists(DATA_RAW_DIR):
        signs = [d for d in os.listdir(DATA_RAW_DIR) if os.path.isdir(os.path.join(DATA_RAW_DIR, d))]
        for sign in signs:
            shutil.rmtree(os.path.join(DATA_RAW_DIR, sign))
        print(f"[INFO] Cleared {len(signs)} sign folders from {DATA_RAW_DIR}")
    else:
        print("[INFO] No raw directory found.")


def main():
    parser = argparse.ArgumentParser(description="SignaSense Synthetic Data Generator")
    parser.add_argument("--samples-per-sign", type=int, default=30, help="Number of samples per sign (default: 30)")
    parser.add_argument("--clean", action="store_true", help="Remove all synthetic files from data/raw/")
    args = parser.parse_args()

    if args.clean:
        clean_dataset()
    else:
        generate_dataset(samples_per_sign=args.samples_per_sign)


if __name__ == "__main__":
    main()

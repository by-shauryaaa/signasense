53537           `   1`"""
SignaSense — Data Collector
============================
Interactive webcam script for recording sign language gesture samples.

For each sign in the vocabulary, records N sequences of M frames each.
Each sequence is saved as a .npy file of shape (SEQUENCE_LENGTH, 258).

Usage:
    conda activate signasense
    python src/data_collector.py
    python src/data_collector.py --signs hello thank_you yes no
    python src/data_collector.py --num-sequences 50
"""

import os
import sys
import argparse
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config.settings import (
    DATA_RAW_DIR,
    SEQUENCE_LENGTH,
    NUM_SEQUENCES_PER_SIGN,
    SIGNS,
)
from src.landmark_extractor import (
    create_holistic_model,
    process_frame,
    extract_keypoints,
    draw_landmarks,
)


def ensure_dirs(signs: list[str]):
    """Create data/raw/<sign>/ directories if they don't exist."""
    for sign in signs:
        path = os.path.join(DATA_RAW_DIR, sign)
        os.makedirs(path, exist_ok=True)


def count_existing_samples(sign: str) -> int:
    """Return how many .npy files already exist for a given sign."""
    sign_dir = os.path.join(DATA_RAW_DIR, sign)
    if not os.path.isdir(sign_dir):
        return 0
    return len([f for f in os.listdir(sign_dir) if f.endswith(".npy")])


def collect_data(signs: list[str], num_sequences: int):
    """
    Main collection loop.

    For each sign:
      1. Show the sign name on screen.
      2. Wait for 's' keypress to start recording.
      3. Capture SEQUENCE_LENGTH frames, extract keypoints.
      4. Save the (SEQUENCE_LENGTH, 258) array as .npy.
      5. Repeat for num_sequences sequences.
    """
    ensure_dirs(signs)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    print("=" * 60)
    print("  SignaSense — Data Collection")
    print("=" * 60)
    print(f"  Signs to record : {len(signs)}")
    print(f"  Sequences/sign  : {num_sequences}")
    print(f"  Frames/sequence : {SEQUENCE_LENGTH}")
    print(f"  Save directory  : {DATA_RAW_DIR}")
    print("=" * 60)
    print()

    with create_holistic_model() as holistic:
        for sign_idx, sign in enumerate(signs):
            existing = count_existing_samples(sign)
            print(f"\n[{sign_idx + 1}/{len(signs)}] Sign: '{sign}'")
            print(f"  Existing samples: {existing}")

            for seq_idx in range(num_sequences):
                global_seq_idx = existing + seq_idx
                save_path = os.path.join(DATA_RAW_DIR, sign, f"{global_seq_idx}.npy")

                # ── Wait screen: show instructions ────────────────────────
                waiting = True
                while waiting:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    results, image = process_frame(frame, holistic)
                    draw_landmarks(image, results)

                    # Instruction overlay
                    overlay = image.copy()
                    cv2.rectangle(overlay, (0, 0), (image.shape[1], 90), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)

                    cv2.putText(
                        image,
                        f"SIGN: {sign.upper()}",
                        (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA,
                    )
                    cv2.putText(
                        image,
                        f"Sequence {seq_idx + 1}/{num_sequences} — Press 's' to START | 'n' to SKIP | 'q' to QUIT",
                        (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
                    )

                    cv2.imshow("SignaSense — Data Collection", image)
                    key = cv2.waitKey(10) & 0xFF

                    if key == ord("s"):
                        waiting = False
                    elif key == ord("n"):
                        waiting = False
                        save_path = None  # skip this sequence
                    elif key == ord("q"):
                        cap.release()
                        cv2.destroyAllWindows()
                        print("\n[INFO] Collection aborted by user.")
                        return

                if save_path is None:
                    print(f"    Sequence {seq_idx + 1} — SKIPPED")
                    continue

                # ── Countdown ─────────────────────────────────────────────
                for countdown in range(3, 0, -1):
                    ret, frame = cap.read()
                    if ret:
                        results, image = process_frame(frame, holistic)
                        draw_landmarks(image, results)
                        cv2.putText(
                            image, str(countdown), (image.shape[1] // 2 - 30, image.shape[0] // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 0, 255), 5, cv2.LINE_AA,
                        )
                        cv2.imshow("SignaSense — Data Collection", image)
                    cv2.waitKey(500)

                # ── Record frames ─────────────────────────────────────────
                sequence = []
                for frame_idx in range(SEQUENCE_LENGTH):
                    ret, frame = cap.read()
                    if not ret:
                        break

                    results, image = process_frame(frame, holistic)
                    keypoints = extract_keypoints(results)
                    sequence.append(keypoints)

                    # Visual feedback
                    draw_landmarks(image, results)
                    progress = int((frame_idx + 1) / SEQUENCE_LENGTH * 100)
                    cv2.putText(
                        image,
                        f"RECORDING: {sign.upper()} [{frame_idx + 1}/{SEQUENCE_LENGTH}]",
                        (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA,
                    )
                    # Progress bar
                    bar_w = image.shape[1] - 40
                    cv2.rectangle(image, (20, 50), (20 + bar_w, 65), (50, 50, 50), -1)
                    cv2.rectangle(image, (20, 50), (20 + int(bar_w * progress / 100), 65), (0, 0, 255), -1)

                    cv2.imshow("SignaSense — Data Collection", image)
                    cv2.waitKey(1)

                # ── Save ──────────────────────────────────────────────────
                sequence_array = np.array(sequence)
                if sequence_array.shape == (SEQUENCE_LENGTH, 258):
                    np.save(save_path, sequence_array)
                    print(f"    Sequence {seq_idx + 1} — SAVED → {save_path}  shape={sequence_array.shape}")
                else:
                    print(f"    Sequence {seq_idx + 1} — ERROR: unexpected shape {sequence_array.shape}, discarded.")

                # Brief pause between sequences
                time.sleep(0.3)

    cap.release()
    cv2.destroyAllWindows()
    print("\n[INFO] Data collection complete!")

    # Summary
    print("\n" + "=" * 60)
    print("  Collection Summary")
    print("=" * 60)
    for sign in signs:
        count = count_existing_samples(sign)
        print(f"  {sign:20s} : {count} samples")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="SignaSense Data Collector")
    parser.add_argument(
        "--signs", nargs="+", default=None,
        help="Specific signs to record (default: all from settings.py)",
    )
    parser.add_argument(
        "--num-sequences", type=int, default=NUM_SEQUENCES_PER_SIGN,
        help=f"Number of sequences per sign (default: {NUM_SEQUENCES_PER_SIGN})",
    )
    args = parser.parse_args()

    signs_to_record = args.signs if args.signs else SIGNS
    collect_data(signs_to_record, args.num_sequences)


if __name__ == "__main__":
    main()

"""
SignaSense — Real-Time Inference Engine
========================================
Opens the webcam and performs live sign language recognition using the
trained LSTM model. Outputs detected signs to the console and optionally
speaks them using text-to-speech.

Pipeline:
    Webcam frame → MediaPipe landmarks → sliding window (30 frames)
    → LSTM prediction → confidence filter → consecutive confirmation
    → sentence buffer → (optional) grammar smoothing → console output

Usage:
    conda activate signasense
    python src/inference.py
    python src/inference.py --no-tts
    python src/inference.py --threshold 0.85
"""

import os
import sys
import json
import time
import argparse
from collections import deque

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config.settings import (
    SEQUENCE_LENGTH,
    BEST_MODEL_PATH,
    LABEL_MAP_PATH,
    CONFIDENCE_THRESHOLD,
    REQUIRED_CONFIRMATIONS,
    IDLE_TIMEOUT_SECONDS,
)
from src.landmark_extractor import (
    create_holistic_model,
    process_frame,
    extract_keypoints,
    draw_landmarks,
)
from src.grammar_smoother import smooth


def load_model():
    """Load the trained Keras model."""
    from tensorflow.keras.models import load_model as keras_load_model

    if not os.path.exists(BEST_MODEL_PATH):
        print(f"[ERROR] Model not found at: {BEST_MODEL_PATH}")
        print("        Run train.py first to train the model.")
        sys.exit(1)

    print(f"[INFO] Loading model from: {BEST_MODEL_PATH}")
    model = keras_load_model(BEST_MODEL_PATH)
    return model


def load_label_map() -> dict:
    """Load the index-to-sign label map."""
    if not os.path.exists(LABEL_MAP_PATH):
        print(f"[ERROR] Label map not found at: {LABEL_MAP_PATH}")
        sys.exit(1)

    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def run_inference(
    video_path: str = None,
    threshold: float = CONFIDENCE_THRESHOLD,
    confirmations: int = REQUIRED_CONFIRMATIONS,
    use_tts: bool = True,
    save_output: str = None,
    no_display: bool = False,
):
    """
    Main real-time inference loop supporting both live webcam and video files.

    Parameters
    ----------
    video_path : str, optional
        Path to a pre-recorded video file (.mp4, .mov, .avi, etc.).
        If None, opens default webcam (index 0).
    threshold : float
        Minimum softmax confidence to accept a prediction (0.0 to 1.0).
    confirmations : int
        Number of consecutive identical predictions required to register a sign.
    use_tts : bool
        If True, speak detected signs using pyttsx3.
    save_output : str, optional
        Path to save the annotated video with landmarks and prediction HUD.
    no_display : bool
        If True, run headlessly without opening an OpenCV window.
    """
    # Load model and labels
    model = load_model()
    label_map = load_label_map()

    # Optional TTS engine
    tts_engine = None
    if use_tts:
        try:
            import pyttsx3
            tts_engine = pyttsx3.init()
            tts_engine.setProperty("rate", 160)
            print("[INFO] Text-to-Speech engine initialized.")
        except Exception as e:
            print(f"[WARN] Could not initialize TTS: {e}")
            tts_engine = None

    # Open video source (webcam or file)
    is_file = False
    if video_path:
        if not os.path.exists(video_path):
            print(f"[ERROR] Video file not found: {video_path}")
            return
        cap = cv2.VideoCapture(video_path)
        is_file = True
        source_name = f"File: {os.path.basename(video_path)}"
    else:
        cap = cv2.VideoCapture(0)
        source_name = "Live Webcam"

    if not cap.isOpened():
        print(f"[ERROR] Cannot open video source: {source_name}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if is_file else None

    # Video writer if output is requested
    writer = None

    # State variables
    window = deque(maxlen=SEQUENCE_LENGTH)     # Sliding window of keypoints
    sentence = []                               # Accumulated sign glosses
    last_prediction = None                      # Last predicted sign
    confirmation_count = 0                      # Consecutive confirmation counter
    last_sign_time = time.time()                # Wall clock time of last sign
    last_sign_frame = 0                         # Frame index of last sign
    idle_submitted = False                      # Flag to prevent repeat auto-submit
    frame_idx = 0

    print("\n" + "=" * 60)
    print("  SignaSense — Sign Language Recognition Engine")
    print("=" * 60)
    print(f"  Source               : {source_name}")
    if is_file and total_frames:
        print(f"  Total frames / FPS   : {total_frames} frames @ {fps:.1f} FPS (~{total_frames / fps:.1f}s)")
    print(f"  Confidence threshold : {threshold}")
    print(f"  Required confirmations: {confirmations}")
    print(f"  TTS enabled          : {use_tts and tts_engine is not None}")
    print(f"  Vocabulary           : {len(label_map)} signs")
    if save_output:
        print(f"  Saving output to     : {save_output}")
    print("=" * 60)
    if not is_file:
        print("\n  Controls:")
        print("    'q'     — Quit")
        print("    'c'     — Clear sentence buffer")
        print("    'Enter' — Speak/submit current sentence")
    print()

    with create_holistic_model() as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            # Initialize video writer on first frame
            if save_output and writer is None:
                fh, fw = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(save_output, fourcc, fps, (fw, fh))

            # 1. Extract keypoints
            results, image = process_frame(frame, holistic)
            keypoints = extract_keypoints(results)
            window.append(keypoints)

            # Draw landmarks
            draw_landmarks(image, results)

            # 2. Predict (only when window is full)
            predicted_sign = None
            confidence = 0.0

            if len(window) == SEQUENCE_LENGTH:
                input_data = np.expand_dims(list(window), axis=0).astype(np.float32)
                prediction = model.predict(input_data, verbose=0)[0]
                predicted_idx = int(np.argmax(prediction))
                confidence = float(prediction[predicted_idx])
                predicted_sign = label_map.get(str(predicted_idx), "???")

                # 3. Confidence filter
                if confidence < threshold:
                    predicted_sign = None

            # 4. Consecutive confirmation
            if predicted_sign is not None and predicted_sign != "_idle_":
                if predicted_sign == last_prediction:
                    confirmation_count += 1
                else:
                    confirmation_count = 1
                    last_prediction = predicted_sign

                if confirmation_count == confirmations:
                    # Register the sign!
                    sentence.append(predicted_sign)
                    last_sign_time = time.time()
                    last_sign_frame = frame_idx
                    idle_submitted = False
                    confirmation_count = 0

                    timestamp_sec = frame_idx / fps if is_file else 0.0
                    time_str = f"[{int(timestamp_sec // 60):02d}:{timestamp_sec % 60:05.2f}] " if is_file else ""
                    print(f"  {time_str}[DETECTED] {predicted_sign.upper()} (confidence: {confidence:.2f})")

                    # Speak individual sign
                    if tts_engine and not is_file:
                        word = predicted_sign.replace("_", " ")
                        tts_engine.say(word)
                        tts_engine.runAndWait()
            else:
                # Reset confirmation on idle or low-confidence
                if predicted_sign == "_idle_" or predicted_sign is None:
                    confirmation_count = 0
                    last_prediction = None

            # 5. Auto-submit on idle timeout
            time_since_sign = (
                (frame_idx - last_sign_frame) / fps if is_file else (time.time() - last_sign_time)
            )
            if sentence and not idle_submitted:
                if time_since_sign > IDLE_TIMEOUT_SECONDS:
                    fluent_text = smooth(sentence)
                    raw_glosses = " ".join(s.upper() for s in sentence)
                    print(f"\n  [SENTENCE] {fluent_text}  (glosses: {raw_glosses})")

                    if tts_engine:
                        tts_engine.say(fluent_text)
                        tts_engine.runAndWait()

                    idle_submitted = True

            # ── Display overlay ───────────────────────────────────────
            h, w = image.shape[:2]

            # Top bar: current prediction
            overlay = image.copy()
            cv2.rectangle(overlay, (0, 0), (w, 100), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)

            if predicted_sign and predicted_sign != "_idle_":
                color = (0, 255, 0) if confidence >= threshold else (0, 165, 255)
                cv2.putText(
                    image, f"Sign: {predicted_sign.upper()}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA,
                )
                cv2.putText(
                    image, f"Confidence: {confidence:.2f}", (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA,
                )

                # Confirmation progress bar
                bar_pct = min(confirmation_count / confirmations, 1.0)
                bar_w_px = int(200 * bar_pct)
                cv2.rectangle(image, (20, 80), (220, 92), (50, 50, 50), -1)
                cv2.rectangle(image, (20, 80), (20 + bar_w_px, 92), (0, 255, 0), -1)
            else:
                waiting_msg = f"Reading video... ({frame_idx}/{total_frames})" if is_file and total_frames else "Waiting for sign..."
                cv2.putText(
                    image, waiting_msg, (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 1, cv2.LINE_AA,
                )

            # Bottom bar: sentence buffer
            if sentence:
                sentence_display = " ".join(s.upper() for s in sentence[-8:])  # Show last 8 words
                overlay2 = image.copy()
                cv2.rectangle(overlay2, (0, h - 50), (w, h), (0, 0, 0), -1)
                cv2.addWeighted(overlay2, 0.6, image, 0.4, 0, image)
                cv2.putText(
                    image, f"Sentence: {sentence_display}", (20, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA,
                )

            # Save annotated frame if requested
            if writer:
                writer.write(image)

            # Display window unless headless
            if not no_display:
                cv2.imshow("SignaSense — Sign Language Recognition", image)
                wait_time = max(1, int(1000 / fps)) if is_file else 5
                key = cv2.waitKey(wait_time) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("c"):
                    sentence.clear()
                    idle_submitted = False
                    print("  [INFO] Sentence buffer cleared.")
                elif key == 13:  # Enter key
                    if sentence:
                        fluent_text = smooth(sentence)
                        raw_glosses = " ".join(s.upper() for s in sentence)
                        print(f"\n  [SUBMIT] {fluent_text}  (glosses: {raw_glosses})")
                        if tts_engine:
                            tts_engine.say(fluent_text)
                            tts_engine.runAndWait()
                        sentence.clear()
                        idle_submitted = False

    # Video complete: flush remaining sentence
    if sentence and not idle_submitted:
        fluent_text = smooth(sentence)
        raw_glosses = " ".join(s.upper() for s in sentence)
        print(f"\n  [FINAL VIDEO TRANSCRIPTION] {fluent_text}  (glosses: {raw_glosses})")
        if tts_engine:
            tts_engine.say(fluent_text)
            tts_engine.runAndWait()

    cap.release()
    if writer:
        writer.release()
        print(f"[INFO] Annotated video saved successfully to: {save_output}")
    if not no_display:
        cv2.destroyAllWindows()
    print("\n[INFO] Inference finished.")


def main():
    parser = argparse.ArgumentParser(description="SignaSense Real-Time Sign Language Recognition")
    parser.add_argument(
        "--video", type=str, default=None,
        help="Path to pre-recorded video file (.mp4, .mov, .avi, etc.). Defaults to webcam if omitted.",
    )
    parser.add_argument(
        "--threshold", type=float, default=CONFIDENCE_THRESHOLD,
        help=f"Confidence threshold (default: {CONFIDENCE_THRESHOLD})",
    )
    parser.add_argument(
        "--confirmations", type=int, default=REQUIRED_CONFIRMATIONS,
        help=f"Required consecutive confirmations (default: {REQUIRED_CONFIRMATIONS})",
    )
    parser.add_argument(
        "--no-tts", action="store_true",
        help="Disable text-to-speech output",
    )
    parser.add_argument(
        "--save-output", type=str, default=None,
        help="Path to save the annotated video output (e.g. output.mp4)",
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Run without displaying GUI window (headless/batch mode)",
    )
    args = parser.parse_args()

    run_inference(
        video_path=args.video,
        threshold=args.threshold,
        confirmations=args.confirmations,
        use_tts=not args.no_tts,
        save_output=args.save_output,
        no_display=args.no_display,
    )


if __name__ == "__main__":
    main()

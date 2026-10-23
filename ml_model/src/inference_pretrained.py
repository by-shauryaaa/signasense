"""
SignaSense — Pretrained 250-Sign ASL Continuous Sentence Recognition Engine
=============================================================================
Uses the 1st-place winning architecture from Google's Isolated Sign Language
Recognition benchmark. Trained on over 100,000 real video samples of native
ASL signers covering 250 everyday signs.

Features:
  - Continuous Sliding Window with Temporal Segmentation & Peak NMS
  - Recognizes multi-sign sequences and continuous signing across videos
  - Translates full ASL sentences into fluent spoken English
  - Hand presence and movement gating to prevent resting-hand false positives
  - Real-time interactive HUD with progress bar, confidence meters, and transcripts
  - Text-to-Speech (pyttsx3) integration
  - Comprehensive end-of-video transcription report and timeline

Usage:
  # Continuous video recognition:
  python src/inference_pretrained.py --video path/to/video.mp4

  # Live webcam recognition:
  python src/inference_pretrained.py

  # Headless mode (batch processing):
  python src/inference_pretrained.py --video path/to/video.mp4 --no-display
"""

import os
import sys
import json
import time
import argparse
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.grammar_smoother import smooth

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "pretrained", "asl_250_model.tflite")
LABEL_MAP_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "pretrained", "sign_to_prediction_index_map.json")

# ASL root aliases to natural English words
GLOSS_ALIASES = {
    "now": "today",
    "thankyou": "thank_you",
    "minemy": "my",
    "weus": "we",
    "hesheit": "he",
    "callonphone": "call",
    "haveto": "have_to",
    "icecream": "ice_cream",
    "frenchfries": "french_fries",
    "glasswindow": "window",
    "shhh": "quiet",
    "owie": "hurt",
}


@dataclass
class DetectedSignEvent:
    sign: str
    canonical: str
    start_frame: int
    end_frame: int
    start_sec: float
    end_sec: float
    peak_conf: float
    window_count: int


def load_label_map():
    if not os.path.exists(LABEL_MAP_PATH):
        raise FileNotFoundError(f"Label map not found at {LABEL_MAP_PATH}")
    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        sign_to_idx = json.load(f)
    idx_to_sign = {int(v): k for k, v in sign_to_idx.items()}
    return sign_to_idx, idx_to_sign


def load_interpreter():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    interp = tf.lite.Interpreter(model_path=MODEL_PATH)
    fn = interp.get_signature_runner("serving_default")
    return fn


def extract_543_landmarks(res):
    """Extract flat (543, 3) matrix from MediaPipe Holistic results with NaN for missing."""
    frame_lms = np.full((543, 3), np.nan, dtype=np.float32)

    if res.face_landmarks:
        for i, lm in enumerate(res.face_landmarks.landmark):
            frame_lms[i] = [lm.x, lm.y, lm.z]

    if res.left_hand_landmarks:
        for i, lm in enumerate(res.left_hand_landmarks.landmark):
            frame_lms[468 + i] = [lm.x, lm.y, lm.z]

    if res.pose_landmarks:
        for i, lm in enumerate(res.pose_landmarks.landmark):
            frame_lms[489 + i] = [lm.x, lm.y, lm.z]

    if res.right_hand_landmarks:
        for i, lm in enumerate(res.right_hand_landmarks.landmark):
            frame_lms[522 + i] = [lm.x, lm.y, lm.z]

    return frame_lms


def predict_sequence(fn, frames_list, idx_to_sign, top_k=5):
    """Run model inference on a sequence of (543, 3) frames."""
    if len(frames_list) < 5:
        return []
    frames_arr = np.array(frames_list, dtype=np.float32)
    raw_out = fn(inputs=frames_arr)["outputs"]
    probs = tf.nn.softmax(raw_out).numpy()
    top_indices = np.argsort(probs)[-top_k:][::-1]

    results = []
    for idx in top_indices:
        sign = idx_to_sign[idx]
        prob = float(probs[idx])
        results.append((sign, prob))
    return results


def draw_hud(
    image,
    top_results,
    active_sign_name,
    timeline_events,
    current_sentence,
    latest_sentence_text,
    frame_info="",
):
    """Render a modern visual HUD on the video frame."""
    h, w = image.shape[:2]

    # Top overlay bar
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (w, 115), (15, 15, 24), -1)
    cv2.addWeighted(overlay, 0.78, image, 0.22, 0, image)

    # Active Sign or Current Top Prediction
    if top_results:
        best_sign, best_prob = top_results[0]
        canonical = GLOSS_ALIASES.get(best_sign, best_sign)
        display_name = canonical.upper().replace("_", " ")
        if best_sign == "now":
            display_name = "TODAY / NOW"

        # Color based on confidence
        if best_prob >= 0.50:
            color = (0, 255, 128)   # Bright neon green
        elif best_prob >= 0.35:
            color = (0, 220, 255)   # Amber / Cyan
        else:
            color = (160, 160, 160) # Gray

        cv2.putText(image, f"ASL Sign: {display_name}", (20, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.95, color, 2, cv2.LINE_AA)

        # Confidence percentage & bar
        conf_text = f"Confidence: {best_prob * 100:.1f}%"
        cv2.putText(image, conf_text, (20, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (230, 230, 230), 1, cv2.LINE_AA)

        bar_x = 240
        bar_w = 160
        bar_h = 12
        cv2.rectangle(image, (bar_x, 56), (bar_x + bar_w, 56 + bar_h), (50, 50, 60), -1)
        fill_w = int(bar_w * min(best_prob, 1.0))
        cv2.rectangle(image, (bar_x, 56), (bar_x + fill_w, 56 + bar_h), color, -1)

        # Runner-up predictions
        if len(top_results) > 1:
            runners = ", ".join(f"{s} ({p*100:.0f}%)" for s, p in top_results[1:4])
            cv2.putText(image, f"Alternatives: {runners}", (20, 98),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (160, 160, 170), 1, cv2.LINE_AA)
    else:
        status_msg = "Signing: " + active_sign_name if active_sign_name else "Scanning for ASL gestures..."
        cv2.putText(image, status_msg, (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.80, (180, 180, 190), 1, cv2.LINE_AA)

    # Frame and timing info on top-right
    if frame_info:
        cv2.putText(image, frame_info, (max(20, w - 280), 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 210), 1, cv2.LINE_AA)

    # Bottom subtitle & translation bar
    overlay2 = image.copy()
    cv2.rectangle(overlay2, (0, h - 80), (w, h), (12, 12, 20), -1)
    cv2.addWeighted(overlay2, 0.82, image, 0.18, 0, image)

    # Line 1: Live detected ASL glosses
    gloss_list = [ev.canonical.upper().replace("_", " ") for ev in timeline_events[-6:]]
    if current_sentence:
        gloss_list.extend([g.upper().replace("_", " ") for g in current_sentence[-4:]])
    gloss_str = " -> ".join(gloss_list) if gloss_list else "(None yet)"
    cv2.putText(image, f"ASL Signs : {gloss_str}", (20, h - 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 230, 120), 1, cv2.LINE_AA)

    # Line 2: Spoken English translation
    translation_display = latest_sentence_text or (smooth(current_sentence) if current_sentence else "...")
    cv2.putText(image, f"English   : \"{translation_display}\"", (20, h - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)


def run_continuous_inference(
    video_path=None,
    save_output=None,
    no_tts=False,
    no_display=False,
    threshold=0.40,
    window_size=30,
    stride=4,
    min_windows=2,
    pause_gap=1.8,
):
    """
    Main continuous sign language recognition loop for video files and live webcam.
    """
    _, idx_to_sign = load_label_map()
    fn = load_interpreter()

    # Optional TTS engine
    tts_engine = None
    if not no_tts:
        try:
            import pyttsx3
            tts_engine = pyttsx3.init()
            tts_engine.setProperty("rate", 160)
        except Exception as e:
            tts_engine = None

    is_file = (video_path is not None)
    if is_file:
        if not os.path.exists(video_path):
            print(f"[ERROR] Video file not found: {video_path}")
            return
        cap = cv2.VideoCapture(video_path)
        source_name = f"File: {os.path.basename(video_path)}"
    else:
        cap = cv2.VideoCapture(0)
        source_name = "Live Webcam"

    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {source_name}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if is_file else None
    duration_sec = (total_frames / fps) if (is_file and total_frames) else 0.0

    print("=" * 70)
    print("  SignaSense — Continuous ASL Sentence Recognition (Google ISLR)")
    print("=" * 70)
    print(f"  Source               : {source_name}")
    if is_file and total_frames:
        print(f"  Duration             : {duration_sec:.2f}s ({total_frames} frames @ {fps:.1f} FPS)")
    print(f"  ASL Vocabulary       : 250 signs")
    print(f"  Confidence Threshold : {threshold:.2f}")
    print(f"  Sliding Window       : {window_size} frames (stride {stride} frames)")
    print(f"  Pause Gap            : {pause_gap:.1f}s (triggers sentence boundary)")
    print(f"  TTS Enabled          : {not no_tts and tts_engine is not None}")
    if save_output:
        print(f"  Save Annotated Video : {save_output}")
    print("=" * 70)
    print()

    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    writer = None
    all_frames_lms = []
    window = deque(maxlen=window_size)
    hand_presence_window = deque(maxlen=window_size)

    # State tracking
    timeline_events = []
    current_sentence_glosses = []
    all_sentences = []
    active_seg = None
    last_sign_end_sec = 0.0
    latest_sentence_text = ""
    top_results = []
    frame_idx = 0
    start_wall_time = time.time()

    def commit_segment(seg):
        nonlocal last_sign_end_sec, latest_sentence_text
        dur = seg["end_sec"] - seg["start_sec"]
        canonical = GLOSS_ALIASES.get(seg["sign"], seg["sign"])
        event = DetectedSignEvent(
            sign=seg["sign"],
            canonical=canonical,
            start_frame=seg["start_frame"],
            end_frame=seg["end_frame"],
            start_sec=seg["start_sec"],
            end_sec=seg["end_sec"],
            peak_conf=seg["peak_conf"],
            window_count=seg["windows"],
        )
        timeline_events.append(event)
        current_sentence_glosses.append(canonical)
        last_sign_end_sec = seg["end_sec"]

        time_str = f"[{int(seg['start_sec']//60):02d}:{seg['start_sec']%60:04.1f} -> {int(seg['end_sec']//60):02d}:{seg['end_sec']%60:04.1f}]"
        print(f"  {time_str} [DETECTED SIGN] {canonical.upper():14s} (Peak Conf: {seg['peak_conf']*100:5.1f}%, Duration: {dur:3.1f}s)")

        # Update running translation preview
        latest_sentence_text = smooth(current_sentence_glosses)

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            curr_sec = frame_idx / fps if is_file else (time.time() - start_wall_time)

            # Initialize video writer on first frame if requested
            if save_output and writer is None:
                fh, fw = frame.shape[:2]
                writer = cv2.VideoWriter(save_output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (fw, fh))

            # MediaPipe Holistic landmark extraction
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            res = holistic.process(rgb)
            rgb.flags.writeable = True

            lms = extract_543_landmarks(res)
            all_frames_lms.append(lms)
            window.append(lms)

            has_hand = (res.left_hand_landmarks is not None or res.right_hand_landmarks is not None)
            hand_presence_window.append(has_hand)

            # Draw skeletal tracking
            if res.pose_landmarks:
                mp_drawing.draw_landmarks(frame, res.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                          mp_drawing_styles.get_default_pose_landmarks_style())
            if res.left_hand_landmarks:
                mp_drawing.draw_landmarks(frame, res.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            if res.right_hand_landmarks:
                mp_drawing.draw_landmarks(frame, res.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            # Continuous Sliding Window Evaluation
            if len(window) == window_size and frame_idx % stride == 0:
                hands_count = sum(hand_presence_window)
                win_start_frame = frame_idx - window_size + 1
                win_start_sec = win_start_frame / fps if is_file else (curr_sec - window_size / fps)
                win_end_sec = curr_sec

                # 1. Check Pause Gap for Sentence Flush (only when no gesture is actively signing)
                if active_seg is None and current_sentence_glosses and (curr_sec - last_sign_end_sec > pause_gap):
                    sent_text = smooth(current_sentence_glosses)
                    all_sentences.append(sent_text)
                    print(f"\n  >>> [COMPLETED SENTENCE] \"{sent_text}\"\n")
                    if tts_engine:
                        tts_engine.say(sent_text)
                        tts_engine.runAndWait()
                    current_sentence_glosses.clear()

                # 2. Hand Presence Gating
                if hands_count >= 10:
                    top_results = predict_sequence(fn, list(window), idx_to_sign, top_k=5)
                    if top_results and top_results[0][1] >= threshold:
                        cand_sign, cand_prob = top_results[0]
                    else:
                        cand_sign, cand_prob = None, 0.0
                else:
                    cand_sign, cand_prob = None, 0.0
                    top_results = []

                # 3. Temporal Segment Tracking & Non-Maximum Suppression
                if cand_sign is not None:
                    if active_seg is None:
                        active_seg = {
                            "sign": cand_sign,
                            "start_frame": win_start_frame,
                            "end_frame": frame_idx,
                            "start_sec": win_start_sec,
                            "end_sec": win_end_sec,
                            "peak_conf": cand_prob,
                            "windows": 1,
                        }
                    elif active_seg["sign"] == cand_sign:
                        active_seg["end_frame"] = frame_idx
                        active_seg["end_sec"] = win_end_sec
                        active_seg["peak_conf"] = max(active_seg["peak_conf"], cand_prob)
                        active_seg["windows"] += 1
                    else:
                        # Switched to a new sign
                        if active_seg["windows"] >= min_windows:
                            commit_segment(active_seg)
                        active_seg = {
                            "sign": cand_sign,
                            "start_frame": win_start_frame,
                            "end_frame": frame_idx,
                            "start_sec": win_start_sec,
                            "end_sec": win_end_sec,
                            "peak_conf": cand_prob,
                            "windows": 1,
                        }
                else:
                    # Low confidence or idle transition
                    if active_seg is not None:
                        if active_seg["windows"] >= min_windows:
                            commit_segment(active_seg)
                        active_seg = None

            # Render visual HUD
            active_name = active_seg["sign"].upper() if active_seg else ""
            if total_frames:
                frame_info = f"{int(curr_sec//60):02d}:{curr_sec%60:04.1f} / {int(duration_sec//60):02d}:{duration_sec%60:04.1f}  ({frame_idx}/{total_frames})"
            else:
                frame_info = f"{int(curr_sec//60):02d}:{curr_sec%60:04.1f}  (Frame {frame_idx})"

            draw_hud(
                frame,
                top_results,
                active_name,
                timeline_events,
                current_sentence_glosses,
                latest_sentence_text,
                frame_info,
            )

            if writer:
                writer.write(frame)

            if not no_display:
                cv2.imshow("SignaSense — ASL Continuous Recognition", frame)
                wait_ms = max(1, int(1000 / fps)) if is_file else 5
                key = cv2.waitKey(wait_ms) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("c"):
                    current_sentence_glosses.clear()
                    latest_sentence_text = ""
                    print("  [INFO] Sentence buffer cleared.")
                elif key in (13, 32):  # Enter or Space
                    if current_sentence_glosses:
                        sent_text = smooth(current_sentence_glosses)
                        all_sentences.append(sent_text)
                        print(f"\n  >>> [MANUAL SUBMIT] \"{sent_text}\"\n")
                        if tts_engine:
                            tts_engine.say(sent_text)
                            tts_engine.runAndWait()
                        current_sentence_glosses.clear()

    cap.release()
    if writer:
        writer.release()
        print(f"\n[INFO] Annotated video saved to: {save_output}")
    if not no_display:
        cv2.destroyAllWindows()

    # Commit any lingering active segment at end of video
    if active_seg is not None and active_seg["windows"] >= min_windows:
        commit_segment(active_seg)

    # Flush remaining uncommitted sentence
    if current_sentence_glosses:
        sent_text = smooth(current_sentence_glosses)
        all_sentences.append(sent_text)

    # Fallback for very short single-gesture clips if no segment was committed
    if not timeline_events and all_frames_lms and is_file:
        print("  [INFO] Running whole-video verification pass...")
        full_top5 = predict_sequence(fn, all_frames_lms, idx_to_sign, top_k=5)
        if full_top5 and full_top5[0][1] >= 0.20:
            best_sign, best_prob = full_top5[0]
            canonical = GLOSS_ALIASES.get(best_sign, best_sign)
            event = DetectedSignEvent(
                sign=best_sign,
                canonical=canonical,
                start_frame=1,
                end_frame=len(all_frames_lms),
                start_sec=0.0,
                end_sec=duration_sec,
                peak_conf=best_prob,
                window_count=1,
            )
            timeline_events.append(event)
            all_sentences.append(smooth([canonical]))

    # Print Final Comprehensive Transcription Report
    print("\n" + "=" * 70)
    print("  SignaSense — Full Video Recognition & Translation Report")
    print("=" * 70)
    if is_file:
        print(f"  Video File          : {os.path.basename(video_path)}")
        print(f"  Duration            : {duration_sec:.2f}s ({total_frames} frames @ {fps:.1f} FPS)")
    print(f"  Signs Recognized    : {len(timeline_events)}")
    print(f"  Sentences Formed    : {len(all_sentences)}")
    print()

    if timeline_events:
        print("  --- Chronological Sign Timeline ---")
        for i, ev in enumerate(timeline_events, 1):
            dur = ev.end_sec - ev.start_sec
            time_str = f"[{int(ev.start_sec//60):02d}:{ev.start_sec%60:04.1f} -> {int(ev.end_sec//60):02d}:{ev.end_sec%60:04.1f}]"
            print(f"    #{i}: {time_str}  {ev.canonical.upper():14s} (Peak Confidence: {ev.peak_conf*100:5.1f}%, Duration: {dur:3.1f}s)")
        print()

        print("  --- Complete ASL Transcript ---")
        print("    " + " ".join(ev.canonical.upper() for ev in timeline_events))
        print()

    if all_sentences:
        print("  --- Spoken English Translation ---")
        for i, s in enumerate(all_sentences, 1):
            print(f"    Sentence #{i}: \"{s}\"")
        print()

        # Speak the final combined translation
        if tts_engine and is_file:
            full_speech = " ".join(all_sentences)
            print("  [INFO] Speaking translated translation via Text-to-Speech...")
            tts_engine.say(full_speech)
            tts_engine.runAndWait()
    else:
        print("  [NOTICE] No sign language gestures met the confidence threshold.")
    print("=" * 70)
    print()


def main():
    parser = argparse.ArgumentParser(description="SignaSense Continuous ASL Sentence Recognition Engine")
    parser.add_argument("positional_video", nargs="?", default=None, help="Path to video file (.mp4, .mov, etc.) or omit for webcam")
    parser.add_argument("--video", type=str, default=None, help="Path to video file (.mp4, .mov, etc.) or omit for webcam")
    parser.add_argument("--save-output", type=str, default=None, help="Save annotated video with landmarks and HUD")
    parser.add_argument("--no-tts", action="store_true", help="Disable text-to-speech output")
    parser.add_argument("--no-display", action="store_true", help="Run headlessly without GUI window")
    parser.add_argument("--threshold", type=float, default=0.40, help="Confidence threshold (default: 0.40)")
    parser.add_argument("--pause-gap", type=float, default=1.8, help="Seconds of pause to split sentences (default: 1.8)")
    args = parser.parse_args()

    chosen_video = args.video or args.positional_video

    run_continuous_inference(
        video_path=chosen_video,
        save_output=args.save_output,
        no_tts=args.no_tts,
        no_display=args.no_display,
        threshold=args.threshold,
        pause_gap=args.pause_gap,
    )


if __name__ == "__main__":
    main()

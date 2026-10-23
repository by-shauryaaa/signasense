"""
SignaSense — Landmark Extractor
================================
Wraps Google MediaPipe Holistic to extract 3D pose + hand landmarks from
a single video frame.  Returns a flat numpy array of 258 floats per frame.

Feature vector layout:
    [0:132]   → Pose   (33 landmarks × 4 values: x, y, z, visibility)
    [132:195] → Left hand  (21 landmarks × 3 values: x, y, z)
    [195:258] → Right hand (21 landmarks × 3 values: x, y, z)
"""

import cv2
import numpy as np
import mediapipe as mp

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config.settings import (
    MP_MIN_DETECTION_CONFIDENCE,
    MP_MIN_TRACKING_CONFIDENCE,
    POSE_FEATURES,
    HAND_FEATURES,
    TOTAL_FEATURES,
)


# ── MediaPipe handles ─────────────────────────────────────────────────────────
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def create_holistic_model():
    """Return a configured MediaPipe Holistic instance (use as context manager)."""
    return mp_holistic.Holistic(
        min_detection_confidence=MP_MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MP_MIN_TRACKING_CONFIDENCE,
    )


def process_frame(frame, holistic_model):
    """
    Run MediaPipe Holistic on a single BGR frame.

    Parameters
    ----------
    frame : np.ndarray
        BGR image from OpenCV (shape H×W×3).
    holistic_model : mp.solutions.holistic.Holistic
        An active Holistic model instance.

    Returns
    -------
    results : mediapipe results object
        Contains .pose_landmarks, .left_hand_landmarks, .right_hand_landmarks
    image : np.ndarray
        The same frame converted to RGB (writeable).
    """
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False          # perf: mark read-only for MediaPipe
    results = holistic_model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # back to BGR for display
    return results, image


def extract_keypoints(results) -> np.ndarray:
    """
    Flatten MediaPipe Holistic results into a single feature vector.

    Parameters
    ----------
    results : mediapipe results object

    Returns
    -------
    keypoints : np.ndarray of shape (258,)
        Concatenation of [pose(132), left_hand(63), right_hand(63)].
        Missing body parts are zero-filled.
    """
    # Pose landmarks: 33 × (x, y, z, visibility)
    if results.pose_landmarks:
        pose = np.array(
            [[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark]
        ).flatten()
    else:
        pose = np.zeros(POSE_FEATURES)

    # Left hand landmarks: 21 × (x, y, z)
    if results.left_hand_landmarks:
        left_hand = np.array(
            [[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]
        ).flatten()
    else:
        left_hand = np.zeros(HAND_FEATURES)

    # Right hand landmarks: 21 × (x, y, z)
    if results.right_hand_landmarks:
        right_hand = np.array(
            [[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]
        ).flatten()
    else:
        right_hand = np.zeros(HAND_FEATURES)

    return np.concatenate([pose, left_hand, right_hand])


def draw_landmarks(image, results):
    """
    Draw pose and hand landmarks on the image (for visual debugging).

    Parameters
    ----------
    image : np.ndarray
        BGR image to draw on (modified in place).
    results : mediapipe results object
    """
    # Pose connections
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=1),
        )

    # Left hand
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=1),
        )

    # Right hand
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=1),
        )


# ── Quick verification script ─────────────────────────────────────────────────
if __name__ == "__main__":
    """
    Run this file directly to verify that MediaPipe extraction works.
    Opens your webcam, overlays landmarks, and prints keypoint shape.
    Press 'q' to quit.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        exit(1)

    print("[INFO] Landmark Extractor verification mode.")
    print("[INFO] Press 'q' to quit.\n")

    with create_holistic_model() as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Failed to grab frame.")
                break

            results, image = process_frame(frame, holistic)
            keypoints = extract_keypoints(results)

            # Overlay landmarks
            draw_landmarks(image, results)

            # Print shape verification
            status = f"Keypoints: {keypoints.shape} | "
            status += f"Pose: {'YES' if results.pose_landmarks else 'NO'} | "
            status += f"LH: {'YES' if results.left_hand_landmarks else 'NO'} | "
            status += f"RH: {'YES' if results.right_hand_landmarks else 'NO'}"

            cv2.putText(
                image, status, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA,
            )

            cv2.imshow("SignaSense — Landmark Verification", image)

            if cv2.waitKey(5) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Done.")

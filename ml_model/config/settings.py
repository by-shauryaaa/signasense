"""
SignaSense Configuration
========================
Central configuration for all pipeline constants.
"""

import os

# ── Project Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODEL_CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "models", "checkpoints")
LABEL_MAP_PATH = os.path.join(DATA_PROCESSED_DIR, "label_map.json")
BEST_MODEL_PATH = os.path.join(MODEL_CHECKPOINT_DIR, "signasense_best.keras")


def ensure_directories():
    """Ensure all required project directories exist."""
    for path in [DATA_RAW_DIR, DATA_PROCESSED_DIR, MODEL_CHECKPOINT_DIR]:
        os.makedirs(path, exist_ok=True)


# ── MediaPipe Settings ─────────────────────────────────────────────────────────
MP_MIN_DETECTION_CONFIDENCE = 0.5
MP_MIN_TRACKING_CONFIDENCE = 0.5

# ── Feature Dimensions ─────────────────────────────────────────────────────────
# Pose: 33 landmarks × 4 values (x, y, z, visibility) = 132
# Left Hand: 21 landmarks × 3 values (x, y, z) = 63
# Right Hand: 21 landmarks × 3 values (x, y, z) = 63
# Total = 258
NUM_POSE_LANDMARKS = 33
NUM_POSE_VALUES = 4  # x, y, z, visibility
NUM_HAND_LANDMARKS = 21
NUM_HAND_VALUES = 3  # x, y, z
POSE_FEATURES = NUM_POSE_LANDMARKS * NUM_POSE_VALUES   # 132
HAND_FEATURES = NUM_HAND_LANDMARKS * NUM_HAND_VALUES    # 63
TOTAL_FEATURES = POSE_FEATURES + (2 * HAND_FEATURES)    # 258

# ── Data Collection ────────────────────────────────────────────────────────────
SEQUENCE_LENGTH = 30          # Frames per sample (~1 second at 30 FPS)
NUM_SEQUENCES_PER_SIGN = 30   # Number of samples to record per sign

# ── Comprehensive Vocabulary (44 Signs) ──────────────────────────────────────────
SIGNS = [
    # Greetings & Politeness
    "hello",
    "goodbye",
    "thank_you",
    "please",
    "sorry",
    "welcome",
    "nice_to_meet_you",
    "how_are_you",

    # Responses & Descriptors
    "yes",
    "no",
    "good",
    "bad",
    "fine",

    # Pronouns & Questions
    "me",
    "you",
    "who",
    "what",
    "where",
    "when",
    "why",
    "how",
    "name",

    # Daily Needs & Actions
    "help",
    "water",
    "drink",
    "eat",
    "more",
    "want",
    "bathroom",
    "sleep",
    "stop",

    # Emotions & Physical States
    "happy",
    "sad",
    "tired",
    "pain",
    "i_love_you",

    # People & Places
    "family",
    "friend",
    "home",
    "work",
    "doctor",

    # Time
    "today",
    "tomorrow",

    # Neutral Calibration State
    "_idle_",
]

# ── Training Hyperparameters ───────────────────────────────────────────────────
TRAIN_SPLIT = 0.80
VAL_SPLIT = 0.10
TEST_SPLIT = 0.10
BATCH_SIZE = 32
EPOCHS = 200
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 30

# ── Inference Settings ─────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.80        # Minimum softmax probability to accept a prediction
REQUIRED_CONFIRMATIONS = 5         # Same sign predicted N times in a row to register
IDLE_TIMEOUT_SECONDS = 3.0         # Seconds of _idle_ before auto-submitting sentence

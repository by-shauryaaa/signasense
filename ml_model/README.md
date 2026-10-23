# SignaSense — Real-Time Sign Language to Text & Speech Engine

SignaSense is an end-to-end, high-performance machine learning engine that translates live American Sign Language (ASL) gestures from a webcam stream into recognized glosses, fluent English sentences, and synthesized speech.

---

## Architecture & Pipeline

```text
[ Webcam Stream (30 FPS) ]
           │
           ▼
[ MediaPipe Holistic Tracker ]
  • Pose Landmarks (33 × 4 = 132 values)
  • Left Hand (21 × 3 = 63 values)
  • Right Hand (21 × 3 = 63 values)
           │
           ▼ (258 floats / frame)
[ 30-Frame Sliding Window ]
           │
           ▼ (Shape: 1, 30, 258)
[ Stacked LSTM Classification Model ]
  • LSTM(64, return_sequences=True) + Dropout(0.3)
  • LSTM(128, return_sequences=True) + Dropout(0.3)
  • LSTM(64, return_sequences=False)
  • Dense(64) → Dropout(0.3) → Dense(32) → Dense(num_classes, softmax)
           │
           ▼
[ Temporal Stabilization Engine ]
  • Softmax confidence thresholding (default: 0.80)
  • Consecutive frame confirmation (prevents flickering)
  • Idle state isolation (`_idle_` class prevents hallucination)
           │
           ▼
[ Grammar Smoothing Engine ]
  • Offline rule-based phrase lookup & sub-sequence reconstruction
  • (Optional) LLM prompt smoothing for unseen gloss combinations
           │
           ▼
[ Output: Live Video Overlay + Console + Offline TTS (pyttsx3) ]
```

---

## Directory Structure

```
SignaSense/
├── config/
│   ├── __init__.py
│   └── settings.py              # Central pipeline constants and hyperparameters
├── data/
│   ├── raw/                     # Raw .npy sequences per sign label (30x258)
│   └── processed/               # Augmented arrays: X_train, y_train, etc., label_map.json
├── models/
│   └── checkpoints/             # Trained signasense_best.keras, reports, confusion matrix
├── scripts/
│   └── generate_synthetic_data.py # Synthetic landmark generator for testing
├── src/
│   ├── __init__.py
│   ├── landmark_extractor.py    # MediaPipe Holistic keypoint extraction (258 features)
│   ├── data_collector.py        # Interactive webcam gesture recorder
│   ├── data_preprocessor.py     # Augmentation, encoding, and train/val/test splitting
│   ├── model.py                 # Stacked LSTM architecture definition
│   ├── train.py                 # Training script with early stopping and checkpointing
│   ├── inference.py             # Live real-time webcam inference with visual HUD & TTS
│   └── grammar_smoother.py      # ASL glosses to natural spoken English
├── tests/
│   └── test_pipeline.py         # Unit and integration test suite
├── requirements.txt
└── README.md
```

---

## Vocabulary (44 Classes)

SignaSense recognizes a comprehensive everyday ASL vocabulary:

| Category | Signs |
|---|---|
| **Greetings & Politeness (8)** | `hello`, `goodbye`, `thank_you`, `please`, `sorry`, `welcome`, `nice_to_meet_you`, `how_are_you` |
| **Responses & Descriptors (5)**| `yes`, `no`, `good`, `bad`, `fine` |
| **Pronouns & Questions (9)**   | `me`, `you`, `who`, `what`, `where`, `when`, `why`, `how`, `name` |
| **Daily Needs & Actions (9)**  | `help`, `water`, `drink`, `eat`, `more`, `want`, `bathroom`, `sleep`, `stop` |
| **Emotions & Health (5)**      | `happy`, `sad`, `tired`, `pain`, `i_love_you` |
| **People & Places (5)**        | `family`, `friend`, `home`, `work`, `doctor` |
| **Time (2)**                   | `today`, `tomorrow` |
| **Resting State (1)**          | `_idle_` (resting calibration to prevent false positives) |

---

## Quick Start

### 1. Environment Setup

SignaSense requires **Python 3.12** for compatibility with MediaPipe 0.10.x and TensorFlow:

```bash
# Activate the conda environment
conda activate signasense
```

Dependencies can be installed via:
```bash
pip install -r requirements.txt
```

### 2. Verify Installation & Pipeline Tests

Run the full pipeline test suite:
```bash
python tests/test_pipeline.py
```
*Expected: 14 tests passing (`OK`).*

### 3. Record Your Own Sign Language Data (Webcam)

To record your own gestures using your webcam:
```bash
python src/data_collector.py
```

- Each sign records 30 sequences of 30 frames (~1 sec at 30 FPS).
- Includes the critical `_idle_` resting state.
- Controls:
  - Press `s` to start recording the current sequence (with visual 3-2-1 countdown).
  - Press `k` to skip to the next sign.
  - Press `q` to safely exit (resumes where you left off).

### 4. Preprocess & Augment Dataset

```bash
python src/data_preprocessor.py --augment
```
Applies:
- Gaussian noise jittering
- Temporal shift (frame offset)
- Landmark coordinate scaling
- Hand mirroring for symmetrical gestures
- Stratified 80/10/10 Train/Validation/Test split

### 5. Train the Model

```bash
python src/train.py --epochs 100 --batch-size 32
```
- Trains with early stopping and model checkpointing.
- Saves the best weights to `models/checkpoints/signasense_best.keras`.
- Generates `classification_report.txt`, `confusion_matrix.npy`, and `training_history.json`.

### 6. Run Live Real-Time Inference

```bash
python src/inference.py
```

Options:
- `--threshold 0.85`: Set minimum confidence threshold (default: 0.80).
- `--confirmations 5`: Require N consecutive frames to confirm a sign (default: 5).
- `--no-tts`: Disable speech synthesis.

Interactive Controls during Live Video:
- `Enter`: Submit and speak current sentence immediately.
- `c`: Clear sentence buffer.
- `q`: Quit.

### 7. Run Inference on a Pre-Recorded Video File

You can translate an existing video of someone signing ASL:

```bash
# Using Python
python src/inference.py --video path/to/asl_video.mp4

# Save annotated video with skeletal tracking + HUD subtitles
python src/inference.py --video path/to/asl_video.mp4 --save-output output_annotated.mp4

# Run headlessly without display window
python src/inference.py --video path/to/asl_video.mp4 --no-display
```

Or using the drag-and-drop launcher:
- Simply drag and drop any video file (`.mp4`, `.mov`, `.avi`) directly onto `run_video_inference.bat`!

---

## Pretrained 250-Sign ASL Continuous Sentence Recognition Engine

For immediate, high-accuracy recognition of real human signers without manual model training:

- **Model**: Trained on **100,000+ real video clips** of native signers across **250 everyday ASL signs** (Google ISLR benchmark).
- **Continuous Sliding Window**: Scans video across time with temporal non-maximum suppression (NMS) to detect full multi-word signing sequences.
- **Natural Language Translation**: Automatically aggregates detected sign glosses, detects pauses between phrases, and translates them into fluent spoken English sentences using grammar smoothing.
- **Timestamped Timeline**: Provides a complete chronological timeline with start and end timestamps, peak confidence scores, and durations for each sign detected in the video.
- **Hand Presence Gating**: Intelligent motion and hand-presence filters prevent idle hands from generating false positives.

### One-Click Windows Launchers:
- **Continuous Video Translation**: Drag & drop any video file (`.mp4`, `.mov`, `.avi`) directly onto `run_video_inference.bat` (or `run_pretrained_video.bat`).
- **Live Continuous Webcam Recognition**: Double-click `run_pretrained_live.bat`.

### Command Line:
```bash
# Continuous video recognition & sentence translation:
python src/inference_pretrained.py --video path/to/video.mp4

# Save annotated video with skeletal tracking & subtitles:
python src/inference_pretrained.py --video path/to/video.mp4 --save-output output.mp4

# Run headlessly (batch mode):
python src/inference_pretrained.py --video path/to/video.mp4 --no-display

# Adjust confidence threshold and sentence pause gap:
python src/inference_pretrained.py --video path/to/video.mp4 --threshold 0.40 --pause-gap 1.8

# Live webcam stream:
python src/inference_pretrained.py
```

---

## Synthetic Data Testing (Zero-Hardware Demo)

If you wish to test the entire training and inference pipeline before recording physical gestures:
```bash
# 1. Generate 30 synthetic sequences per sign
python scripts/generate_synthetic_data.py --samples-per-sign 30

# 2. Preprocess with augmentation
python src/data_preprocessor.py --augment

# 3. Train LSTM model
python src/train.py --epochs 15 --batch-size 32

# 4. Run tests
python tests/test_pipeline.py
```

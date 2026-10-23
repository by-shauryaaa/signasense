# SignaSense

> **Real-Time AI-Powered American Sign Language (ASL) Recognition & Speech Translation**  
> *Bridging the communication gap between Deaf/Hard-of-Hearing individuals and the hearing community.*

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Platform: iOS 17+](https://img.shields.io/badge/Platform-iOS%2017%2B-lightgrey.svg)](ios_app/)
[![Python: 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](ml_model/)
[![Framework: MediaPipe](https://img.shields.io/badge/Framework-MediaPipe-orange.svg)](ml_model/)
[![Framework: CoreML / Vision](https://img.shields.io/badge/Apple-Vision%20%7C%20CoreML-black.svg)](ios_app/)

---

## Overview

**SignaSense** is an intelligent, real-time sign language translation platform that converts American Sign Language (ASL) gestures into natural spoken English and readable text. Built as a dual-stack monorepo, SignaSense pairs a native **SwiftUI iOS application** with an end-to-end **Python Machine Learning engine**.

```text
       ┌────────────────────────┐         ┌────────────────────────┐
       │   iOS Native App       │         │   Python ML Engine     │
       │   (SwiftUI + Vision)   │         │   (MediaPipe + LSTM)   │
       └───────────┬────────────┘         └───────────┬────────────┘
                   │                                  │
                   ▼                                  ▼
      • 21-point Hand Landmarks           • 258 Holistic Skeletal Features
      • Apple Vision Framework            • Pretrained 250-Sign ASL Model
      • Plug-and-play CoreML              • 44-Class Stacked LSTM Pipeline
      • AVSpeechSynthesizer               • Temporal NMS & Grammar Smoother
                   │                                  │
                   └─────────────────┬────────────────┘
                                     ▼
                     [ Real-Time ASL Translation ]
                     [ Text HUD + Voice Synthesis ]
```

---

## Repository Structure

```
SignaSense/
├── README.md                      # Primary project overview and instructions
├── LICENSE                        # MIT License
├── .gitignore                     # Git ignore rules for iOS, ML, and OS files
│
├── ios_app/                       # Native iOS Application
│   ├── App/
│   │   └── SignaSenseApp.swift    # App entry point
│   ├── Views/
│   │   └── ContentView.swift      # Neumorphic UI, camera feed, landmark overlays, TTS
│   ├── Resources/
│   │   ├── Assets.xcassets/       # App icons, colors, and branding
│   │   ├── Info.plist             # Camera & microphone permission manifests
│   │   ├── MacCameraBridge.swift  # Local camera streaming bridge for testing
│   │   └── README_MODEL_SETUP.md  # Guide to plug in Core ML models
│   ├── SignaSense.xcodeproj/      # Xcode project configuration
│   ├── SignaSense_PRD.pdf         # Product Requirements Document
│   └── logo.jpeg                  # Application banner / logo
│
└── ml_model/                      # Python ML & Continuous Recognition Pipeline
    ├── config/
    │   ├── __init__.py
    │   └── settings.py            # Hyperparameters, paths, and class definitions
    ├── data/                      # Dataset directories (ignored by git)
    │   ├── raw/                   # Captured 30x258 .npy landmark sequences
    │   └── processed/             # Augmented training/validation/test sets
    ├── models/
    │   ├── checkpoints/           # Training checkpoints & evaluation metrics
    │   └── pretrained/            # Pretrained 250-sign Google ISLR TFLite engine
    ├── scripts/
    │   └── generate_synthetic_data.py # Zero-hardware synthetic data generator
    ├── src/
    │   ├── landmark_extractor.py  # 258-feature MediaPipe Holistic tracking
    │   ├── data_collector.py      # Interactive webcam gesture recorder
    │   ├── data_preprocessor.py   # Augmentation (jitter, scale, shift, flip)
    │   ├── model.py               # 3-layer Stacked LSTM classification model
    │   ├── train.py               # Model training with early stopping
    │   ├── inference.py           # Live inference HUD with temporal confirmation
    │   ├── inference_pretrained.py# Continuous sentence recognition (250 signs)
    │   └── grammar_smoother.py    # ASL gloss to fluent English grammar engine
    ├── tests/
    │   └── test_pipeline.py       # Comprehensive unit test suite (14 tests)
    ├── videos/
    │   └── sample_test.mp4        # Sample video for automated test verification
    ├── requirements.txt           # Python dependencies (Python 3.12 compatible)
    ├── README.md                  # Detailed ML documentation
    ├── run.ps1                    # Portable PowerShell launcher
    └── *.bat                      # One-click Windows batch launchers
```

---

## Key Features

- **Continuous 250-Sign ASL Engine**: Pretrained on 100,000+ real video clips of native signers (Google ISLR benchmark) in `ml_model/models/pretrained/asl_250_model.tflite`.
- **44-Class Stacked LSTM Pipeline**: Custom trainable model with MediaPipe Holistic skeletal tracking across 258 features (pose + left hand + right hand).
- **Anti-Flicker Temporal Stabilization**: Sliding window analysis with confidence thresholding, consecutive-frame confirmation, and an `_idle_` resting state to eliminate false triggers.
- **ASL Grammar Smoothing**: Reconstructs broken sign glosses into fluent English sentences using rule-based phrase mappings and optional LLM smoothing (`OpenAI`).
- **Real-Time Voice Synthesis**: Offline text-to-speech output (`AVSpeechSynthesizer` on iOS, `pyttsx3` in Python).
- **Native iOS Experience**: Polished neumorphic SwiftUI design, real-time hand-landmark skeletal visualizer, and zero hardcoded prediction fallbacks.

---

## Getting Started

### 1. Python ML Engine (`ml_model/`)

#### Prerequisites
- Python 3.12 (recommended via Conda or venv)
- Webcam (for live recognition)

#### Installation
```bash
# Navigate to the ML directory
cd ml_model

# Recommended: Create and activate a Conda environment
conda create -n signasense python=3.12 -y
conda activate signasense

# Install dependencies
pip install -r requirements.txt
```

#### Run Unit Tests
Verify that all feature extraction and pipeline components are functional:
```bash
python tests/test_pipeline.py
```
*Expected: 14 passing tests (`OK`).*

#### Live Webcam Inference
```bash
# Pretrained 250-sign continuous recognition:
python src/inference_pretrained.py

# Custom 44-sign LSTM model:
python src/inference.py
```

#### Run Inference on Video Files
```bash
# Translate an existing ASL video:
python src/inference_pretrained.py --video videos/sample_test.mp4 --save-output output.mp4
```

---

### 2. iOS Application (`ios_app/`)

#### Prerequisites
- macOS Sonoma (14.0+) or later
- Xcode 15.0+
- iOS 17.0+ deployment target

#### Building & Running
1. Open the project in Xcode:
   ```bash
   open ios_app/SignaSense.xcodeproj
   ```
2. Select an iOS Simulator (e.g., iPhone 15 Pro) or a connected physical iOS device.
3. Press **Cmd + R** to build and run.
4. Grant Camera permissions when prompted.

#### CoreML Model Integration
The iOS app is designed for plug-and-play CoreML support:
1. Export or convert your trained model to `SignClassifier.mlmodel` (or `.mlpackage`).
2. Drag and drop `SignClassifier.mlmodel` into `ios_app/Resources/`.
3. The app's `ModelManager` automatically detects the model on launch and feeds real-time Vision hand pose keypoints directly to it.
4. See [README_MODEL_SETUP.md](ios_app/Resources/README_MODEL_SETUP.md) for full instructions.

---

## ASL Vocabulary Reference

### Pretrained ISLR Model (250 Signs)
Recognizes 250 high-frequency everyday ASL signs from the Google Isolated Sign Language Recognition benchmark, supporting multi-sign continuous sentence recognition.

### Custom LSTM Pipeline (44 Signs)
| Category | Included Signs |
|---|---|
| **Greetings & Politeness (8)** | `hello`, `goodbye`, `thank_you`, `please`, `sorry`, `welcome`, `nice_to_meet_you`, `how_are_you` |
| **Responses & Descriptors (5)**| `yes`, `no`, `good`, `bad`, `fine` |
| **Pronouns & Questions (9)**   | `me`, `you`, `who`, `what`, `where`, `when`, `why`, `how`, `name` |
| **Daily Needs & Actions (9)**  | `help`, `water`, `drink`, `eat`, `more`, `want`, `bathroom`, `sleep`, `stop` |
| **Emotions & Health (5)**      | `happy`, `sad`, `tired`, `pain`, `i_love_you` |
| **People & Places (5)**        | `family`, `friend`, `home`, `work`, `doctor` |
| **Time (2)**                   | `today`, `tomorrow` |
| **Neutral State (1)**          | `_idle_` (resting calibration to isolate natural pauses) |

---

## Contributing

Contributions, bug reports, and suggestions are welcome!
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for details.

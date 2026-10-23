"""
SignaSense — LSTM Model Definition
====================================
Defines the stacked LSTM architecture for sign language gesture classification.

Architecture:
    Input (batch, 30, 258)
    → LSTM(64, return_sequences=True)
    → Dropout(0.3)
    → LSTM(128, return_sequences=True)
    → Dropout(0.3)
    → LSTM(64, return_sequences=False)
    → Dense(64, relu)
    → Dropout(0.3)
    → Dense(32, relu)
    → Dense(num_classes, softmax)
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config.settings import SEQUENCE_LENGTH, TOTAL_FEATURES, LEARNING_RATE


def build_model(num_classes: int):
    """
    Build and compile the stacked LSTM model.

    Parameters
    ----------
    num_classes : int
        Number of sign classes (including _idle_).

    Returns
    -------
    model : tf.keras.Model
        Compiled Keras model ready for training.
    """
    # Import TensorFlow here (heavy import, only when needed)
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam

    model = Sequential([
        Input(shape=(SEQUENCE_LENGTH, TOTAL_FEATURES)),
        # First LSTM layer: learns low-level temporal patterns
        LSTM(64, return_sequences=True, activation="relu"),
        Dropout(0.3),

        # Second LSTM layer: learns higher-level temporal abstractions
        LSTM(128, return_sequences=True, activation="relu"),
        Dropout(0.3),

        # Third LSTM layer: compresses temporal information into a single vector
        LSTM(64, return_sequences=False, activation="relu"),

        # Dense classification head
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(32, activation="relu"),

        # Output layer: probability distribution over sign classes
        Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["categorical_accuracy"],
    )

    return model


def get_model_summary(num_classes: int) -> str:
    """Build model and return its summary as a string."""
    import io
    model = build_model(num_classes)
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + "\n"))
    return stream.getvalue()


if __name__ == "__main__":
    # Quick check: build and print model summary
    print("=" * 60)
    print("  SignaSense — Model Architecture")
    print("=" * 60)
    print(get_model_summary(num_classes=15))

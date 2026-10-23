"""
SignaSense — Pipeline Tests
=============================
Smoke tests for all pipeline components.

Usage:
    conda activate signasense
    python -m pytest tests/test_pipeline.py -v
    # or simply:
    python tests/test_pipeline.py
"""

import os
import sys
import json
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config.settings import (
    SEQUENCE_LENGTH,
    TOTAL_FEATURES,
    POSE_FEATURES,
    HAND_FEATURES,
    BEST_MODEL_PATH,
    LABEL_MAP_PATH,
    DATA_PROCESSED_DIR,
)


class TestLandmarkExtractor(unittest.TestCase):
    """Tests for src/landmark_extractor.py"""

    def test_extract_keypoints_returns_correct_shape(self):
        """extract_keypoints should always return (258,) even with no detections."""
        from src.landmark_extractor import extract_keypoints

        # Create a mock results object with no detections
        class MockResults:
            pose_landmarks = None
            left_hand_landmarks = None
            right_hand_landmarks = None

        keypoints = extract_keypoints(MockResults())
        self.assertEqual(keypoints.shape, (TOTAL_FEATURES,))
        self.assertEqual(keypoints.shape, (258,))

    def test_extract_keypoints_all_zeros_when_no_detection(self):
        """When nothing is detected, all values should be zero."""
        from src.landmark_extractor import extract_keypoints

        class MockResults:
            pose_landmarks = None
            left_hand_landmarks = None
            right_hand_landmarks = None

        keypoints = extract_keypoints(MockResults())
        np.testing.assert_array_equal(keypoints, np.zeros(TOTAL_FEATURES))

    def test_feature_dimensions_are_consistent(self):
        """Verify the dimension constants add up correctly."""
        self.assertEqual(POSE_FEATURES, 33 * 4)   # 132
        self.assertEqual(HAND_FEATURES, 21 * 3)    # 63
        self.assertEqual(TOTAL_FEATURES, POSE_FEATURES + 2 * HAND_FEATURES)  # 258


class TestDataPreprocessor(unittest.TestCase):
    """Tests for data preprocessing logic."""

    def test_augment_sequence_produces_4_copies(self):
        """augment_sequence should return 4 augmented versions."""
        from src.data_preprocessor import augment_sequence

        dummy_seq = np.random.rand(SEQUENCE_LENGTH, TOTAL_FEATURES).astype(np.float32)
        augmented = augment_sequence(dummy_seq)

        self.assertEqual(len(augmented), 4)
        for aug in augmented:
            self.assertEqual(aug.shape, (SEQUENCE_LENGTH, TOTAL_FEATURES))

    def test_augmented_copies_differ_from_original(self):
        """Augmented sequences should not be identical to the original."""
        from src.data_preprocessor import augment_sequence

        dummy_seq = np.random.rand(SEQUENCE_LENGTH, TOTAL_FEATURES).astype(np.float32)
        augmented = augment_sequence(dummy_seq)

        for i, aug in enumerate(augmented):
            # At least one value should differ
            self.assertFalse(
                np.allclose(dummy_seq, aug, atol=1e-10),
                f"Augmentation {i} is identical to original",
            )


class TestGrammarSmoother(unittest.TestCase):
    """Tests for src/grammar_smoother.py"""

    def test_known_phrase(self):
        from src.grammar_smoother import smooth
        result = smooth(["hello", "name", "what"])
        self.assertEqual(result, "Hello, what is your name?")

    def test_single_sign(self):
        from src.grammar_smoother import smooth
        result = smooth(["thank_you"])
        self.assertEqual(result, "Thank you!")

    def test_unknown_combination_returns_something(self):
        """Unknown gloss sequences should produce a non-empty fallback."""
        from src.grammar_smoother import smooth
        result = smooth(["xyz_unknown", "abc_unknown"])
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)

    def test_empty_glosses(self):
        from src.grammar_smoother import smooth
        result = smooth([])
        self.assertEqual(result, "")


class TestModelArchitecture(unittest.TestCase):
    """Tests for src/model.py"""

    def test_model_output_shape(self):
        """Model should output (batch, num_classes)."""
        from src.model import build_model

        num_classes = 15
        model = build_model(num_classes)

        # Create dummy input
        dummy_input = np.random.rand(1, SEQUENCE_LENGTH, TOTAL_FEATURES).astype(np.float32)
        output = model.predict(dummy_input, verbose=0)

        self.assertEqual(output.shape, (1, num_classes))

    def test_model_output_sums_to_one(self):
        """Softmax output should sum to approximately 1.0."""
        from src.model import build_model

        model = build_model(num_classes=10)
        dummy_input = np.random.rand(1, SEQUENCE_LENGTH, TOTAL_FEATURES).astype(np.float32)
        output = model.predict(dummy_input, verbose=0)

        self.assertAlmostEqual(float(np.sum(output)), 1.0, places=5)


class TestTrainedModelIntegration(unittest.TestCase):
    """
    Integration tests that require a trained model and processed data.
    These are skipped if the model hasn't been trained yet.
    """

    @unittest.skipUnless(
        os.path.exists(BEST_MODEL_PATH),
        "Trained model not found — run train.py first",
    )
    def test_model_loads_successfully(self):
        from tensorflow.keras.models import load_model
        model = load_model(BEST_MODEL_PATH)
        self.assertIsNotNone(model)

    @unittest.skipUnless(
        os.path.exists(LABEL_MAP_PATH),
        "Label map not found — run data_preprocessor.py first",
    )
    def test_label_map_consistency(self):
        """Label map should have sequential integer keys starting from 0."""
        with open(LABEL_MAP_PATH, "r") as f:
            label_map = json.load(f)

        keys = sorted(int(k) for k in label_map.keys())
        expected = list(range(len(label_map)))
        self.assertEqual(keys, expected)

    @unittest.skipUnless(
        os.path.exists(BEST_MODEL_PATH) and os.path.exists(LABEL_MAP_PATH),
        "Model or label map not found",
    )
    def test_model_output_matches_label_count(self):
        """Model output dimension should equal the number of labels."""
        from tensorflow.keras.models import load_model

        model = load_model(BEST_MODEL_PATH)
        with open(LABEL_MAP_PATH, "r") as f:
            label_map = json.load(f)

        output_dim = model.output_shape[-1]
        self.assertEqual(output_dim, len(label_map))


if __name__ == "__main__":
    print("=" * 60)
    print("  SignaSense — Pipeline Tests")
    print("=" * 60)
    unittest.main(verbosity=2)

"""Tests for failure analysis diagnostics."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evals.failure_analysis import (
    analyze_latent_capacity,
    analyze_position_accuracy,
    analyze_quantization_error,
    analyze_separator_confusion,
    analyze_tf_fr_gap,
)


class TestFailureAnalysis(unittest.TestCase):
    def test_tf_fr_gap_analysis(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            tf_report_path = tmp_path / "tf_report.json"
            fr_samples_path = tmp_path / "fr_samples.jsonl"
            out_dir = tmp_path / "analysis"

            tf_data = {
                "teacher_forced": {
                    "metrics": {"valid/token_loss": 1.25, "valid/token_accuracy": 0.82}
                }
            }
            tf_report_path.write_text(json.dumps(tf_data), encoding="utf-8")

            samples = [
                {
                    "length_bucket": "short",
                    "token_accuracy": 0.75,
                    "geometry_f1_2px": 0.88,
                    "eos_rate": 1.0,
                },
                {
                    "length_bucket": "short",
                    "token_accuracy": 0.65,
                    "geometry_f1_2px": 0.80,
                    "eos_rate": 1.0,
                },
                {
                    "length_bucket": "long",
                    "token_accuracy": 0.35,
                    "geometry_f1_2px": 0.40,
                    "eos_rate": 0.0,
                },
            ]
            with open(fr_samples_path, "w", encoding="utf-8") as f:
                for s in samples:
                    f.write(json.dumps(s) + "\n")

            # A global score cannot stand in for bucket-aligned measurements.
            with self.assertRaisesRegex(ValueError, "bucket-aligned"):
                analyze_tf_fr_gap(tf_report_path, fr_samples_path, out_dir)

    def test_tf_fr_gap_with_per_bucket_tf(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            tf_report_path = tmp_path / "tf_report.json"
            fr_samples_path = tmp_path / "fr_samples.jsonl"
            out_dir = tmp_path / "analysis"

            tf_data = {
                "teacher_forced": {
                    "metrics": {"valid/token_accuracy": 0.80},
                    "per_bucket": {
                        "short": {"tf_token_accuracy_mean": 0.85},
                        "long": {"tf_token_accuracy_mean": 0.70},
                    },
                }
            }
            tf_report_path.write_text(json.dumps(tf_data), encoding="utf-8")

            samples = [
                {
                    "length_bucket": "short",
                    "token_accuracy": 0.70,
                    "geometry_f1_2px": 0.85,
                },
                {
                    "length_bucket": "long",
                    "token_accuracy": 0.35,
                    "geometry_f1_2px": 0.40,
                },
            ]
            with open(fr_samples_path, "w", encoding="utf-8") as f:
                for s in samples:
                    f.write(json.dumps(s) + "\n")

            res = analyze_tf_fr_gap(tf_report_path, fr_samples_path, out_dir)
            # Bucket-aligned gaps:
            # short: 0.85 - 0.70 = 0.15
            # long: 0.70 - 0.35 = 0.35
            self.assertAlmostEqual(
                res["per_bucket"]["short"]["accuracy_gap_tf_minus_fr"], 0.15
            )
            self.assertAlmostEqual(
                res["per_bucket"]["long"]["accuracy_gap_tf_minus_fr"], 0.35
            )

    def test_position_accuracy_curves(self):
        # 2 samples:
        # Sample 1: length 5, matches at 0, 1, 2, fails at 3, 4
        # Sample 2: length 5, matches at 0, 1, fails at 2, 3, 4
        records = [
            {
                "length_bucket": "short",
                "target_tokens": [1002, 10, 20, 30, 1003],
                "predicted_tokens": [1002, 10, 20, 99, 99],
            },
            {
                "length_bucket": "short",
                "target_tokens": [1002, 10, 20, 30, 1003],
                "predicted_tokens": [1002, 10, 99, 99, 99],
            },
        ]

        res = analyze_position_accuracy(
            records, onset_threshold=0.80, collapse_threshold=0.50
        )
        overall = res["overall"]
        curve = overall["accuracy_curve"]

        # pos 0: 2/2 = 1.0
        # pos 1: 2/2 = 1.0
        # pos 2: 1/2 = 0.5 (onset triggered here: < 0.80)
        # pos 3: 0/2 = 0.0 (collapse triggered here: < 0.50)
        self.assertEqual(curve[0], 1.0)
        self.assertEqual(curve[1], 1.0)
        self.assertEqual(curve[2], 0.5)
        self.assertEqual(curve[3], 0.0)
        self.assertEqual(overall["onset_position"], 2)
        self.assertEqual(overall["collapse_position"], 3)

    def test_separator_confusion_and_cascade(self):
        # SEP = 1001
        records = [
            {
                # Target has SEP at index 2 (correctly predicted), next 2 tokens match
                "target_tokens": [10, 20, 1001, 30, 40],
                "predicted_tokens": [10, 20, 1001, 30, 40],
            },
            {
                # Target has SEP at index 2 (missed: predicted 99), next 2 tokens fail
                "target_tokens": [10, 20, 1001, 30, 40],
                "predicted_tokens": [10, 20, 99, 88, 77],
            },
            {
                # False separator at index 1
                "target_tokens": [10, 20, 30],
                "predicted_tokens": [10, 1001, 30],
            },
        ]

        res = analyze_separator_confusion(records, sep_token_id=1001, cascade_window=2)
        cm = res["confusion_matrix"]
        metrics = res["metrics"]
        cascade = res["cascade_analysis"]

        self.assertEqual(cm["true_positives"], 1)
        self.assertEqual(cm["false_positives"], 1)
        self.assertEqual(cm["false_negatives"], 1)
        self.assertAlmostEqual(metrics["separator_precision"], 0.5)
        self.assertAlmostEqual(metrics["separator_recall"], 0.5)

        # After correct SEP: next tokens [30, 40] match 100% (1.0)
        # After missed SEP: next tokens fail 0% (0.0)
        self.assertAlmostEqual(cascade["accuracy_after_correct_separator"], 1.0)
        self.assertAlmostEqual(cascade["accuracy_after_missed_separator"], 0.0)
        self.assertAlmostEqual(cascade["cascade_penalty"], 1.0)
        self.assertTrue(cascade["cascade_effect_detected"])

    def test_quantization_error_analysis(self):
        # Simple 2-vector codebook
        codebook = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)

        # Stroke-5: 2 drawing points, 1 lift point, 1 sentinel
        # Row 1: (0.1, 0.1) -> closest is (0, 0), error = sqrt(0.02) ~ 0.1414
        # Row 2: (0.9, 0.9) lift -> closest is (1, 1), error = sqrt(0.02) ~ 0.1414
        # Row 3: sentinel
        stroke5 = np.array(
            [
                [0.1, 0.1, 1.0, 0.0, 0.0],
                [0.9, 0.9, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        res = analyze_quantization_error([stroke5], codebook)
        self.assertEqual(res["total_evaluated_points"], 2)
        self.assertEqual(res["total_evaluated_strokes"], 1)
        self.assertAlmostEqual(
            res["point_quantization_error"]["mean"], 0.1414, places=3
        )

    def test_latent_capacity_analysis(self):
        # 10 samples in 4 dimensions, centered with known variance
        np.random.seed(42)
        latents = np.random.randn(10, 4).astype(np.float32)

        res = analyze_latent_capacity(latents)
        self.assertEqual(res["num_samples"], 10)
        self.assertEqual(res["latent_dimension"], 4)
        self.assertGreater(res["effective_dimensionality_pr"], 1.0)
        self.assertLessEqual(res["effective_dimensionality_pr"], 4.0)
        self.assertIn("top_1", res["cumulative_variance_explained"])
        self.assertEqual(len(res["singular_values_top10"]), 4)


if __name__ == "__main__":
    unittest.main()

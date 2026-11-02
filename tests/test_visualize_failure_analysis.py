"""Tests for failure analysis visualization tools."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evals.visualize_failure_analysis import (
    generate_all_plots,
    plot_codebook_usage,
    plot_latent_capacity,
    plot_position_accuracy_curves,
    plot_tf_fr_gap,
)


class TestVisualizeFailureAnalysis(unittest.TestCase):
    def test_plot_position_accuracy_curves(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "pos_curve.png"
            data = {
                "overall": {"accuracy_curve": [1.0, 0.9, 0.75, 0.4]},
                "by_bucket": {
                    "short": {"num_samples": 10, "accuracy_curve": [1.0, 0.95, 0.85, 0.6]},
                    "long": {"num_samples": 5, "accuracy_curve": [1.0, 0.8, 0.45, 0.2]},
                },
            }
            res = plot_position_accuracy_curves(data, out_file)
            self.assertTrue(res.exists())
            self.assertGreater(res.stat().st_size, 1000)

    def test_plot_tf_fr_gap(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "gap.png"
            data = {
                "per_bucket": {
                    "short": {"tf_token_accuracy_mean": 0.85, "fr_token_accuracy_mean": 0.75},
                    "medium": {"tf_token_accuracy_mean": 0.80, "fr_token_accuracy_mean": 0.60},
                    "long": {"tf_token_accuracy_mean": 0.70, "fr_token_accuracy_mean": 0.35},
                }
            }
            res = plot_tf_fr_gap(data, out_file)
            self.assertTrue(res.exists())
            self.assertGreater(res.stat().st_size, 1000)

    def test_plot_codebook_usage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "codebook.png"
            # 100 entries: 70 used, 30 dead
            counts = np.zeros(100, dtype=np.int64)
            counts[:70] = np.random.randint(1, 500, size=70)
            res = plot_codebook_usage(counts, out_file)
            self.assertTrue(res.exists())
            self.assertGreater(res.stat().st_size, 1000)

    def test_plot_latent_capacity(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "scree.png"
            data = {
                "latent_dimension": 128,
                "effective_dimensionality_pr": 42.5,
                "cumulative_variance_explained": {
                    "top_1": 0.15,
                    "top_5": 0.45,
                    "top_10": 0.65,
                    "top_20": 0.80,
                    "top_50": 0.95,
                    "top_100": 0.99,
                },
            }
            res = plot_latent_capacity(data, out_file)
            self.assertTrue(res.exists())
            self.assertGreater(res.stat().st_size, 1000)

    def test_generate_all_plots(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            analysis_dir = Path(tmpdir) / "analysis"
            figures_dir = Path(tmpdir) / "figures"
            analysis_dir.mkdir()

            # Create mock analysis reports
            (analysis_dir / "position_accuracy_curves.json").write_text(
                json.dumps({"overall": {"accuracy_curve": [0.9, 0.8, 0.5]}}), encoding="utf-8"
            )
            (analysis_dir / "tf_vs_fr_gap_analysis.json").write_text(
                json.dumps({"per_bucket": {"short": {"tf_token_accuracy_mean": 0.8, "fr_token_accuracy_mean": 0.7}}}),
                encoding="utf-8",
            )
            np.save(analysis_dir / "codebook_usage_counts.npy", np.array([10, 20, 0, 5]))
            (analysis_dir / "latent_capacity_analysis.json").write_text(
                json.dumps({"cumulative_variance_explained": {"top_1": 0.5}, "effective_dimensionality_pr": 2.0}),
                encoding="utf-8",
            )

            generated = generate_all_plots(analysis_dir, figures_dir)
            self.assertEqual(len(generated), 4)
            for p in generated:
                self.assertTrue(p.exists())
                self.assertGreater(p.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()

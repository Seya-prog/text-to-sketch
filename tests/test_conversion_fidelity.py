"""Tests for TF-to-PyTorch conversion fidelity analysis."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from evals.conversion_fidelity import compute_parity_metrics, evaluate_logits_files


class TestConversionFidelity(unittest.TestCase):
    def test_identical_logits_pass(self):
        # 4 samples, 10 positions, vocab size 100
        logits = np.random.randn(4, 10, 100).astype(np.float32)
        res = compute_parity_metrics(logits, logits)

        self.assertEqual(res["status"], "PASS")
        self.assertAlmostEqual(res["metrics"]["global_cosine_similarity"], 1.0, places=4)
        self.assertAlmostEqual(res["metrics"]["argmax_agreement"], 1.0, places=4)
        self.assertAlmostEqual(res["metrics"]["max_absolute_difference"], 0.0, places=4)
        self.assertEqual(res["num_positions_evaluated"], 40)
        self.assertEqual(res["vocab_size"], 100)

    def test_slightly_perturbed_logits_within_tolerance(self):
        # Small floating-point noise typical of cross-framework differences
        np.random.seed(42)
        reference = np.random.randn(2, 5, 50).astype(np.float32)
        noise = np.random.randn(2, 5, 50).astype(np.float32) * 1e-4
        candidate = reference + noise

        res = compute_parity_metrics(candidate, reference, min_cosine=0.999, min_agreement=0.99)
        self.assertEqual(res["status"], "PASS")
        self.assertGreaterEqual(res["metrics"]["global_cosine_similarity"], 0.999)
        self.assertGreaterEqual(res["metrics"]["argmax_agreement"], 0.99)

    def test_dissimilar_logits_fail(self):
        np.random.seed(42)
        ref = np.random.randn(4, 10, 50).astype(np.float32)
        cand = np.random.randn(4, 10, 50).astype(np.float32)

        res = compute_parity_metrics(cand, ref, min_cosine=0.999, min_agreement=0.99)
        self.assertEqual(res["status"], "FAIL")
        self.assertFalse(res["thresholds"]["cosine_passed"])
        self.assertFalse(res["thresholds"]["agreement_passed"])

    def test_shape_mismatch_raises(self):
        a = np.zeros((2, 5, 10), dtype=np.float32)
        b = np.zeros((2, 6, 10), dtype=np.float32)
        with self.assertRaises(ValueError):
            compute_parity_metrics(a, b)

    def test_evaluate_logits_files_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            ref_path = tmp_path / "ref.npz"
            cand_path = tmp_path / "cand.npz"
            out_dir = tmp_path / "output"

            data = np.random.randn(2, 4, 30).astype(np.float32)
            np.savez(ref_path, logits=data)
            np.savez(cand_path, logits=data)

            report = evaluate_logits_files(cand_path, ref_path, output_dir=out_dir)
            self.assertEqual(report["status"], "PASS")

            report_file = out_dir / "conversion_fidelity_report.json"
            self.assertTrue(report_file.exists())
            saved = json.loads(report_file.read_text(encoding="utf-8"))
            self.assertEqual(saved["status"], "PASS")


if __name__ == "__main__":
    unittest.main()

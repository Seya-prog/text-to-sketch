"""Tests for evals/benchmark_manifest.py."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


class TestAnnotateSample(unittest.TestCase):
    """Unit tests for per-sample annotation."""

    def test_simple_token_sequence(self) -> None:
        from evals.benchmark_manifest import annotate_sample

        # SOS=1002, tokens [5, 10, 1001, 20, 30, 1001, 40], EOS=1003
        tokens = [1002, 5, 10, 1001, 20, 30, 1001, 40, 1003]
        result = annotate_sample("test_0", "valid", tokens)

        self.assertEqual(result["sample_id"], "test_0")
        self.assertEqual(result["split"], "valid")
        # Content tokens: 5, 10, 1001, 20, 30, 1001, 40 = 7
        self.assertEqual(result["token_length"], 7)
        # 2 separators → 3 strokes
        self.assertEqual(result["sep_count"], 2)
        self.assertEqual(result["stroke_count"], 3)
        # Motion tokens: 7 - 2 = 5
        self.assertEqual(result["motion_token_count"], 5)
        # Detail density: 5/3 ≈ 1.67
        self.assertAlmostEqual(result["detail_density"], 1.67, places=2)
        # Separator ratio: 2/7 ≈ 0.2857
        self.assertAlmostEqual(result["separator_ratio"], 0.2857, places=3)

    def test_no_separators(self) -> None:
        from evals.benchmark_manifest import annotate_sample

        tokens = [1002, 5, 10, 20, 1003]
        result = annotate_sample("test_1", "valid", tokens)

        self.assertEqual(result["sep_count"], 0)
        self.assertEqual(result["stroke_count"], 1)
        self.assertEqual(result["motion_token_count"], 3)

    def test_padded_sequence(self) -> None:
        from evals.benchmark_manifest import annotate_sample

        tokens = [1002, 5, 10, 1003, 0, 0, 0]
        result = annotate_sample("test_2", "valid", tokens)

        # Only content tokens: 5, 10
        self.assertEqual(result["token_length"], 2)
        self.assertEqual(result["raw_length"], 7)


class TestLengthBucket(unittest.TestCase):
    """Test length bucketing."""

    def test_short(self) -> None:
        from evals.benchmark_manifest import length_bucket

        self.assertEqual(length_bucket(1), "short")
        self.assertEqual(length_bucket(512), "short")

    def test_medium(self) -> None:
        from evals.benchmark_manifest import length_bucket

        self.assertEqual(length_bucket(513), "medium")
        self.assertEqual(length_bucket(1024), "medium")

    def test_long(self) -> None:
        from evals.benchmark_manifest import length_bucket

        self.assertEqual(length_bucket(1025), "long")
        self.assertEqual(length_bucket(2048), "long")

    def test_extended(self) -> None:
        from evals.benchmark_manifest import length_bucket

        self.assertEqual(length_bucket(2049), "extended")
        self.assertEqual(length_bucket(4096), "extended")

    def test_over_4096(self) -> None:
        from evals.benchmark_manifest import length_bucket

        self.assertEqual(length_bucket(5000), "over_4096")


class TestBuildManifest(unittest.TestCase):
    """Integration test for manifest building from npz files."""

    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp(prefix="test_manifest_")
        self.dataset_root = Path(self._tmp)

        # Create a minimal valid.npz with 3 samples
        seq1 = np.array([1002, 5, 10, 1001, 20, 1003], dtype=np.int64)
        seq2 = np.array([1002, 5, 10, 20, 30, 40, 1003], dtype=np.int64)
        seq3 = np.array([1002, 5, 1001, 10, 1001, 20, 1003], dtype=np.int64)
        # Pad to same length
        max_len = max(len(seq1), len(seq2), len(seq3))
        x = np.zeros((3, max_len), dtype=np.int64)
        x[0, :len(seq1)] = seq1
        x[1, :len(seq2)] = seq2
        x[2, :len(seq3)] = seq3
        y = np.array([0, 0, 0], dtype=np.int64)
        np.savez(self.dataset_root / "valid.npz", x=x, y=y)

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_builds_manifest_from_npz(self) -> None:
        from evals.benchmark_manifest import build_manifest

        records = build_manifest(self.dataset_root, "valid")
        self.assertEqual(len(records), 3)
        self.assertEqual(records[0]["split"], "valid")
        self.assertTrue(records[0]["sample_id"].startswith("valid_"))

    def test_manifest_has_required_fields(self) -> None:
        from evals.benchmark_manifest import build_manifest

        records = build_manifest(self.dataset_root, "valid")
        required = {
            "sample_id", "split", "token_length", "length_bucket",
            "stroke_count", "sep_count", "detail_density", "separator_ratio",
        }
        for record in records:
            self.assertTrue(required.issubset(record.keys()), f"Missing fields in {record}")

    def test_summarize_manifest(self) -> None:
        from evals.benchmark_manifest import build_manifest, summarize_manifest

        records = build_manifest(self.dataset_root, "valid")
        summary = summarize_manifest(records)
        self.assertEqual(summary["total_samples"], 3)
        self.assertIn("length_buckets", summary)
        self.assertIn("token_length", summary)

    def test_write_and_read_manifest(self) -> None:
        from evals.benchmark_manifest import build_manifest, write_manifest

        records = build_manifest(self.dataset_root, "valid")
        output_path = self.dataset_root / "manifest.jsonl"
        write_manifest(records, output_path)

        # Read back
        loaded = []
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                loaded.append(json.loads(line))
        self.assertEqual(len(loaded), 3)


if __name__ == "__main__":
    unittest.main()

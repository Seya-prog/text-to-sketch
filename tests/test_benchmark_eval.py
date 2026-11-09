"""CPU regressions for the complete benchmark-to-diagnostics workflow."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import yaml
from safetensors.torch import save_file

from builders import build_model
from evals.benchmark_common import length_bucket, select_comparison_samples
from evals.benchmark_eval import _load_model_checkpoint
from evals.benchmark_manifest import annotate_sample
from evals.failure_analysis import analyze_tf_fr_gap
from scripts.sketchformer.config import compose_training_config

ROOT = Path(__file__).resolve().parents[1]


class BenchmarkIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data = self.root / "data"
        self.data.mkdir()
        self.sequences = [[6, 1, 5, 7], [6, 2, 1, 5, 7], [6, 3, 4, 5, 7]]
        self.write_data(self.sequences)
        codebook = np.array(
            [[0.1, 0], [0, 0.1], [-0.1, 0], [0, -0.1]], dtype=np.float32
        )
        np.save(self.root / "codebook.npy", codebook)
        config = compose_training_config("configs/train.yaml", experiment="smoke_test")
        config["experiment"].pop("overrides", None)
        config["data"]["dataset"]["root"] = str(self.data)
        config["data"]["sequence"].update(max_length=16, target_max_length=16)
        config["data"]["batching"].update(num_workers=0, persistent_workers=False)
        tokens = dict(
            codebook_size=4,
            motion_token_offset=1,
            pad_token_id=0,
            sep_token_id=5,
            sos_token_id=6,
            eos_token_id=7,
            vocab_size=8,
            codebook_dir=None,
            codebook_path=str(self.root / "codebook.npy"),
        )
        config["data"]["format"]["token_dictionary"] = tokens
        config["model"]["input"].update(max_seq_len=16, token_dictionary=tokens)
        config["model"]["embedding"]["positional_encoding"]["max_length"] = 16
        config["model"]["architecture"].update(
            d_model=16,
            latent_dim=16,
            pool_hidden_dim=16,
            num_encoder_layers=1,
            num_decoder_layers=1,
            num_heads=2,
            dim_feedforward=32,
            dropout=0.0,
        )
        self.config = self.root / "config.yaml"
        self.config.write_text(yaml.safe_dump(config))
        self.checkpoint = self.root / "model.safetensors"
        torch.manual_seed(42)
        save_file(build_model(config["model"]).state_dict(), str(self.checkpoint))
        self.out = self.root / "results"

    def write_data(self, sequences):
        x = np.empty(len(sequences), dtype=object)
        x[:] = [np.array(seq, dtype=np.int64) for seq in sequences]
        np.savez(
            self.data / "valid.npz",
            x=x,
            y=np.zeros(len(x), dtype=np.int64),
            sample_ids=np.array([f"sample-{i}" for i in range(len(x))]),
        )

    def run_benchmark(self, name, *extra, success=True):
        env = dict(os.environ)
        env.update(
            PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"
        )
        command = [
            sys.executable,
            "-B",
            "evals/benchmark_eval.py",
            "--config",
            str(self.config),
            "--checkpoint",
            str(self.checkpoint),
            "--device",
            "cpu",
            "--seed",
            "0",
            "--max-generation-length",
            "8",
            "--num-plots",
            "2",
            "--save-predictions",
            "--save-latents",
            "--output-dir",
            str(self.out),
            "--run-name",
            name,
            *extra,
        ]
        result = subprocess.run(
            command, cwd=ROOT, env=env, text=True, capture_output=True, timeout=60
        )
        if success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0)
        return result

    def test_real_model_both_modes_gap_latents_and_frozen_selection(self):
        self.run_benchmark("baseline")
        report_path = self.out / "baseline_report.json"
        report = json.loads(report_path.read_text())
        self.assertEqual(report["provenance"]["seed"], 0)
        self.assertFalse(report["provenance"]["checkpoint_load"]["missing_keys"])
        self.assertIn("dataset_sha256", report["provenance"])
        self.assertEqual(report["teacher_forced"]["metrics"]["valid/valid_tokens"], 11)
        self.assertAlmostEqual(
            report["teacher_forced"]["per_bucket"]["short"]["tf_token_perplexity"],
            np.exp(report["teacher_forced"]["per_bucket"]["short"]["tf_token_loss"]),
        )
        self.assertIn("fr_exact_match", report["free_running"]["per_bucket"]["short"])
        tf_records = [
            json.loads(line)
            for line in (self.out / report["teacher_forced"]["per_sample_path"])
            .read_text()
            .splitlines()
        ]
        self.assertEqual([r["target_length"] for r in tf_records], [4, 5, 5])
        self.assertEqual(
            {r["sample_id"] for r in tf_records}, {"sample-0", "sample-1", "sample-2"}
        )
        gap = analyze_tf_fr_gap(
            report_path, self.out / "baseline_per_sample.jsonl", self.out / "analysis"
        )
        self.assertTrue(gap["per_sample_gap_available"])
        for row in gap["per_sample_records"]:
            self.assertAlmostEqual(
                row["accuracy_gap_tf_minus_fr"],
                row["tf_token_accuracy"] - row["token_accuracy"],
            )
        self.assertEqual(
            gap["worst_20_samples"],
            sorted(
                gap["per_sample_records"],
                key=lambda r: (-r["accuracy_gap_tf_minus_fr"], r["sample_id"]),
            ),
        )
        with np.load(self.out / "baseline_latents.npz") as latents:
            self.assertEqual(latents["embeddings"].shape, (3, 16))
            self.assertEqual(
                latents["sample_ids"].tolist(), [r["sample_id"] for r in tf_records]
            )
        manifest_path = self.out / "baseline_comparison_manifest.json"
        self.run_benchmark("candidate", "--comparison-manifest", str(manifest_path))
        baseline_selection = json.loads(manifest_path.read_text())["samples"]
        candidate_selection = json.loads(
            (self.out / "candidate_comparison_manifest.json").read_text()
        )["samples"]
        self.assertEqual(baseline_selection, candidate_selection)
        original = report_path.read_bytes()
        result = self.run_benchmark("baseline", success=False)
        self.assertIn("already exist", result.stderr)
        self.assertEqual(report_path.read_bytes(), original)

    def test_overlong_sample_is_rejected_before_creating_report(self):
        self.write_data([[6] + [1] * 16 + [7]])
        result = self.run_benchmark("overlong", success=False)
        self.assertIn("truncation is disabled", result.stderr)
        self.assertIn("sample-0", result.stderr)
        self.assertFalse((self.out / "overlong_report.json").exists())

    def test_incomplete_checkpoint_is_not_silently_accepted(self):
        model = torch.nn.Linear(2, 2)
        path = self.root / "partial.safetensors"
        save_file({"weight": model.weight.detach()}, str(path))
        with self.assertRaises(RuntimeError):
            _load_model_checkpoint(path, model)
        audit = _load_model_checkpoint(path, model, strict=False)
        self.assertEqual(audit["missing_keys"], ["bias"])


class BenchmarkSelectionTest(unittest.TestCase):
    def test_stratified_selection_is_stable_under_loader_reordering(self):
        records = [
            dict(sample_id=f"short-{i}", target_length=100, length_bucket="short")
            for i in range(60)
        ]
        records += [
            dict(sample_id="extended-0", target_length=3000, length_bucket="2049_4096")
        ]
        selected = select_comparison_samples(records, 50, 0)
        self.assertIn("extended-0", {r["sample_id"] for r in selected})
        self.assertEqual(len(selected), 50)
        self.assertEqual(
            selected, select_comparison_samples(list(reversed(records)), 50, 0)
        )
        self.assertNotEqual(selected, select_comparison_samples(records, 50, 1))

    def test_manifest_and_runtime_share_boundary_lengths(self):
        for size in (512, 513, 1024, 1025, 2048, 2049, 4096):
            tokens = [1002] + [1] * (size - 2) + [1003, 0, 0]
            record = annotate_sample("sample", "valid", tokens)
            self.assertEqual(record["sequence_length"], size)
            self.assertEqual(record["length_bucket"], length_bucket(size))

    def test_duplicate_ids_are_rejected(self):
        row = dict(sample_id="duplicate", target_length=100, length_bucket="short")
        with self.assertRaises(ValueError):
            select_comparison_samples([row, row], 2, 42)


if __name__ == "__main__":
    unittest.main()

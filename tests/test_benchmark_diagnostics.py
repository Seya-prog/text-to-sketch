"""Scientific regressions and CLI failure propagation for PR #6."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from evals.benchmark_eval import run_free_running
from evals.conversion_fidelity import compute_parity_metrics, evaluate_logits_files
from evals.failure_analysis import (
    analyze_latent_capacity,
    analyze_position_accuracy,
    analyze_separator_confusion,
    analyze_tf_fr_gap,
)

ROOT = Path(__file__).resolve().parents[1]


class ParityRegressionTest(unittest.TestCase):
    def test_padding_cannot_hide_wrong_meaningful_predictions(self):
        ref = np.tile([1000.0, 0.0], (1, 1000, 1)).astype(np.float32)
        candidate = ref.copy()
        ref[:, :10] = [10.0, 0.0]
        candidate[:, :10] = [0.0, 10.0]
        self.assertEqual(compute_parity_metrics(candidate, ref)["status"], "PASS")
        masked = compute_parity_metrics(candidate, ref, lengths=np.array([10]))
        self.assertEqual(masked["status"], "FAIL")
        self.assertEqual(masked["num_positions_evaluated"], 10)
        self.assertEqual(masked["metrics"]["argmax_agreement"], 0.0)
        mask = np.arange(1000)[None, :] < 10
        self.assertEqual(
            compute_parity_metrics(candidate, ref, valid_mask=mask)["status"], "FAIL"
        )
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            np.savez(root / "candidate.npz", logits=candidate, valid_mask=mask)
            np.savez(root / "reference.npz", logits=ref, lengths=np.array([10]))
            result = evaluate_logits_files(
                root / "candidate.npz", root / "reference.npz"
            )
            self.assertEqual(result["status"], "FAIL")
            self.assertIn("valid_mask_sha256", result)
            np.savez(root / "unmasked.npz", logits=ref)
            with self.assertRaisesRegex(ValueError, "require valid_mask"):
                evaluate_logits_files(root / "unmasked.npz", root / "unmasked.npz")

    def test_invalid_masks_empty_comparisons_and_nonfinite_valid_logits_rejected(self):
        logits = np.ones((1, 4, 3), dtype=np.float32)
        for kwargs in [
            dict(valid_mask=np.zeros((1, 4), dtype=bool)),
            dict(valid_mask=np.ones((1, 3), dtype=bool)),
            dict(lengths=np.array([5])),
            dict(lengths=np.array([1.5])),
            dict(valid_mask=np.ones((1, 4), dtype=int)),
        ]:
            with self.assertRaises(ValueError):
                compute_parity_metrics(logits, logits, **kwargs)
        bad = logits.copy()
        bad[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            compute_parity_metrics(bad, logits)
        # Invalid storage padding is irrelevant when explicitly masked.
        bad[0, 0] = logits[0, 0]
        bad[0, 3] = np.nan
        self.assertEqual(
            compute_parity_metrics(bad, logits, lengths=np.array([3]))["status"], "PASS"
        )

    def test_conversion_failure_exits_nonzero_through_both_analysis_commands(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            ref = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
            np.save(root / "ref.npy", ref)
            np.save(root / "candidate.npy", -ref)
            env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
            for command in ("conversion", "all"):
                args = [
                    sys.executable,
                    "-B",
                    "evals/failure_analysis.py",
                    command,
                    "--candidate-logits",
                    str(root / "candidate.npy"),
                    "--reference-logits",
                    str(root / "ref.npy"),
                    "--output-dir",
                    str(root / command),
                ]
                if command == "all":
                    args += [
                        "--dataset-root",
                        str(root / "absent"),
                        "--codebook-path",
                        str(root / "missing.npy"),
                    ]
                result = subprocess.run(
                    args, cwd=ROOT, env=env, capture_output=True, text=True, timeout=30
                )
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertEqual(
                    json.loads(
                        (root / command / "conversion_fidelity_report.json").read_text()
                    )["status"],
                    "FAIL",
                )


class DiagnosticRegressionTest(unittest.TestCase):
    def test_missing_separator_continuations_count_as_incorrect(self):
        rows = [
            dict(target_tokens=[1001, 1, 2], predicted_tokens=p)
            for p in ([1001, 1, 2], [1001], [99, 99, 99])
        ]
        cascade = analyze_separator_confusion(rows, cascade_window=2)[
            "cascade_analysis"
        ]
        self.assertEqual(cascade["accuracy_after_correct_separator"], 0.5)
        self.assertEqual(cascade["cascade_penalty"], 0.5)
        self.assertEqual(cascade["correct_separator_windows"], 2)
        self.assertEqual(cascade["missed_separator_windows"], 1)
        no_comparison = analyze_separator_confusion(rows[:2])["cascade_analysis"]
        self.assertIsNone(no_comparison["cascade_effect_detected"])
        self.assertIsNone(no_comparison["cascade_penalty"])

    def test_participation_ratio_uses_variance_and_is_scale_invariant(self):
        spectrum = np.r_[10.0, np.ones(127)]
        z = np.vstack([np.diag(spectrum), -np.diag(spectrum)])
        expected = 227.0**2 / (10000.0 + 127.0)
        result = analyze_latent_capacity(z)
        self.assertAlmostEqual(
            result["effective_dimensionality_pr"], expected, places=2
        )
        self.assertEqual(
            result["effective_dimensionality_pr"],
            analyze_latent_capacity(z * 7)["effective_dimensionality_pr"],
        )
        self.assertEqual(
            analyze_latent_capacity(np.ones((3, 4)))["effective_dimensionality_pr"], 0.0
        )
        for invalid in (np.empty((0, 4)), np.full((3, 4), np.nan)):
            with self.assertRaises(ValueError):
                analyze_latent_capacity(invalid)

    def test_gap_joins_samples_normalizes_buckets_and_ranks_actual_gap(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            tf = [
                dict(
                    sample_id="a",
                    target_length=20,
                    target_sha256="same-a",
                    length_bucket="short",
                    tf_token_accuracy=0.9,
                ),
                dict(
                    sample_id="b",
                    target_length=1200,
                    target_sha256="same-b",
                    length_bucket="long",
                    tf_token_accuracy=0.2,
                ),
            ]
            fr = [
                dict(
                    sample_id="b",
                    target_length=1200,
                    target_sha256="same-b",
                    length_bucket="1025_2048",
                    token_accuracy=0.1,
                    geometry_f1_2px=0.0,
                ),
                dict(
                    sample_id="a",
                    target_length=20,
                    target_sha256="same-a",
                    length_bucket="1_512",
                    token_accuracy=0.1,
                    geometry_f1_2px=0.99,
                ),
            ]
            (root / "tf.json").write_text(
                json.dumps({"teacher_forced": {"per_sample_records": tf}})
            )
            fr_path = root / "fr.jsonl"
            fr_path.write_text("\n".join(json.dumps(r) for r in fr))
            report = analyze_tf_fr_gap(root / "tf.json", fr_path, root)
            self.assertAlmostEqual(
                report["per_bucket"]["short"]["accuracy_gap_tf_minus_fr"], 0.8
            )
            self.assertAlmostEqual(
                report["per_bucket"]["long"]["accuracy_gap_tf_minus_fr"], 0.1
            )
            self.assertEqual(report["worst_20_samples"][0]["sample_id"], "a")
            fr[0]["target_sha256"] = "different-input"
            fr_path.write_text("\n".join(json.dumps(r) for r in fr))
            with self.assertRaisesRegex(ValueError, "target_sha256"):
                analyze_tf_fr_gap(root / "tf.json", fr_path, root)

    def test_position_curves_use_shared_buckets_and_report_sample_support(self):
        result = analyze_position_accuracy(
            [{"target_tokens": [1] * 1600, "predicted_tokens": [1] * 1600}]
        )
        self.assertIn("long", result["by_bucket"])
        self.assertEqual(result["overall"]["active_sample_counts"], [1] * 1600)
        with self.assertRaisesRegex(ValueError, "save-predictions"):
            analyze_position_accuracy([{"length_bucket": "short"}])

    def test_real_free_running_collection_includes_rare_buckets(self):
        class IdentityGenerator(torch.nn.Module):
            def generate(self, batch, **kwargs):
                return SimpleNamespace(
                    tokens=batch["targets"], lengths=batch["lengths"]
                )

        def batch(count, length):
            tokens = torch.ones((count, length), dtype=torch.long)
            tokens[:, 0] = 6
            tokens[:, -2] = 5
            tokens[:, -1] = 7
            return dict(
                targets=tokens,
                lengths=torch.full((count,), length),
                sample_ids=[f"{length}:{i}" for i in range(count)],
                source_indices=torch.arange(count),
                source_files=["fixture.npz"] * count,
            )

        summary, examples = run_free_running(
            IdentityGenerator(),
            [batch(60, 4), batch(1, 513), batch(1, 2049)],
            torch.device("cpu"),
            SimpleNamespace(autocast_dtype=None),
            np.array(
                [[0.01, 0.0], [0.0, 0.01], [-0.01, 0.0], [0.0, -0.01]], dtype=np.float32
            ),
            eos_token_id=7,
            sos_token_id=6,
            num_plots=50,
        )
        self.assertEqual(len(examples), 50)
        self.assertIn("513:0", {e.sample_id for e in examples})
        self.assertIn("2049:0", {e.sample_id for e in examples})
        for bucket in ("short", "medium", "extended"):
            self.assertEqual(
                summary["per_bucket"][bucket]["fr_token_accuracy_mean"], 1.0
            )

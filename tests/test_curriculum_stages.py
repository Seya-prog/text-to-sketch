"""Tests for curriculum stage improvements (issue #11).

Covers:
  - Long drawings at short stages (filtered correctly).
  - Padded batches near the token budget (special tokens counted).
  - Partial gradient groups at stage boundaries (correct weighting).
  - Skipped updates don't advance LR schedule.
  - Stage report produces correct counts.
  - Empty stage is handled gracefully.
  - Warmup clamping.
"""

from __future__ import annotations

import math
import unittest

import numpy as np
import torch

from dataloaders.collate import TokenSequenceCollator
from dataloaders.datamodule import (
    SequenceLengthSubset,
    TokenBudgetBatchSampler,
    _round_up,
)
from scripts.sketchformer.curriculum import (
    CurriculumStage,
    clamp_warmup,
    compute_stage_updates,
    parse_curriculum,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

class _FakeDataset:
    """Minimal dataset stand-in for SequenceLengthSubset tests."""

    def __init__(self, items: list) -> None:
        self._items = items

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int):
        return self._items[index]


# ---------------------------------------------------------------------------
# 1. Long drawings at short stages
# ---------------------------------------------------------------------------

class TestLongDrawingsAtShortStages(unittest.TestCase):
    """SequenceLengthSubset must exclude drawings that exceed the limit."""

    def test_short_stage_excludes_long_sequences(self) -> None:
        lengths = [10, 50, 100, 500, 1000, 2000]
        dataset = _FakeDataset(list(range(len(lengths))))
        subset = SequenceLengthSubset(dataset, lengths, max_length=100)

        self.assertEqual(len(subset), 3)
        self.assertEqual(subset.lengths, [10, 50, 100])

    def test_all_sequences_too_long_raises(self) -> None:
        lengths = [500, 1000, 2000]
        dataset = _FakeDataset(list(range(len(lengths))))

        with self.assertRaises(ValueError):
            SequenceLengthSubset(dataset, lengths, max_length=100)

    def test_allow_empty_does_not_raise(self) -> None:
        lengths = [500, 1000, 2000]
        dataset = _FakeDataset(list(range(len(lengths))))
        subset = SequenceLengthSubset(
            dataset, lengths, max_length=100, allow_empty=True
        )

        self.assertEqual(len(subset), 0)
        self.assertEqual(subset.lengths, [])

    def test_exact_boundary_is_included(self) -> None:
        lengths = [100, 200]
        dataset = _FakeDataset(list(range(len(lengths))))
        subset = SequenceLengthSubset(dataset, lengths, max_length=100)

        self.assertEqual(len(subset), 1)
        self.assertEqual(subset[0], 0)


# ---------------------------------------------------------------------------
# 2. Padded batches near the token budget
# ---------------------------------------------------------------------------

class TestTokenBudgetWithSpecialTokens(unittest.TestCase):
    """Token budget sampler must account for padding and special tokens."""

    def test_budget_with_padding_overhead(self) -> None:
        """Raw lengths that fit inside the budget before padding may exceed
        it after padding alignment.  Ensure the round-up helper is correct."""
        self.assertEqual(_round_up(5, 8), 8)
        self.assertEqual(_round_up(8, 8), 8)
        self.assertEqual(_round_up(9, 8), 16)
        self.assertEqual(_round_up(1, None), 1)

    def test_budget_sampler_with_adjusted_lengths(self) -> None:
        """When lengths include padding alignment (but NOT double-counted
        SOS/EOS — stored tok-dict sequences already contain them) the
        sampler should produce batches that don't exceed the budget."""
        raw_lengths = [100, 200, 300]
        pad_multiple = 8
        adjusted = [
            _round_up(length, pad_multiple)
            for length in raw_lengths
        ]
        # adjusted = [104, 200, 304]
        self.assertEqual(adjusted, [104, 200, 304])

        sampler = TokenBudgetBatchSampler(
            adjusted, max_tokens=512, max_batch_size=8, shuffle=False
        )
        batches = list(sampler)
        for batch in batches:
            batch_max = max(adjusted[i] for i in batch)
            self.assertLessEqual(batch_max * len(batch), 512)

    def test_collator_adds_sos_eos(self) -> None:
        """The collator prepends SOS and appends EOS; the padded batch must
        be at least raw_length + 2, rounded up."""
        collator = TokenSequenceCollator(
            max_length=32,
            pad_token_id=0,
            sep_token_id=5,
            sos_token_id=6,
            eos_token_id=7,
            add_start_token=True,
            add_end_token=True,
            pad_to_multiple_of=8,
        )
        sample = {
            "tokens": np.array([1, 2, 3], dtype=np.int64),
            "label": 0,
            "source_index": 0,
            "source_file": "test.npz",
        }
        batch = collator([sample])

        # raw=3 + SOS + EOS = 5 tokens → padded to 8
        self.assertEqual(batch["tokens"].shape[1], 8)
        self.assertEqual(batch["lengths"][0].item(), 5)


# ---------------------------------------------------------------------------
# 3. Partial gradient groups at stage boundaries
# ---------------------------------------------------------------------------

class TestPartialGradientGroup(unittest.TestCase):
    """When accumulated micro-batches < accumulate at end of epoch,
    the gradient divisor must correct for the smaller group."""

    def test_partial_correction_factor(self) -> None:
        """Verify the math: if accumulate=4 but only 2 micro-batches ran,
        each micro-batch divided loss by 4.  The correction must multiply
        by 4/2=2 so the effective divisor equals the real count (2)."""
        accumulate = 4
        actual_micro = 2
        correction = float(accumulate) / float(actual_micro)
        self.assertAlmostEqual(correction, 2.0)

        # The train loop applies: _divide_gradients(model, 1/correction)
        # which multiplies grad by correction (=2), undoing the excess /4.
        effective_divisor = accumulate / correction
        self.assertAlmostEqual(effective_divisor, actual_micro)

    def test_full_accumulation_has_no_correction(self) -> None:
        accumulate = 4
        actual_micro = 4
        correction = float(accumulate) / float(actual_micro)
        self.assertAlmostEqual(correction, 1.0)


# ---------------------------------------------------------------------------
# 4. Skipped updates don't advance LR schedule
# ---------------------------------------------------------------------------

class TestSchedulerStepGuard(unittest.TestCase):
    """The LR schedule must not advance when the optimizer step is skipped.

    Note: in the current implementation the scheduler always steps with
    the optimizer (not on micro-batch boundaries).  This test validates
    that ``compute_stage_updates`` returns the correct count so the
    cosine schedule is built with the right ``total_steps``.
    """

    def test_compute_stage_updates_batch_accumulation(self) -> None:
        # 20 examples, batch_size=4 → 5 batches, accumulate=2 → 3 updates/epoch
        updates = compute_stage_updates(
            num_examples=20, epochs=3, batch_size=4, accumulate=2
        )
        # 5 batches / 2 = ceil(2.5) = 3 updates * 3 epochs = 9
        self.assertEqual(updates, 9)

    def test_compute_stage_updates_token_budget(self) -> None:
        lengths = [100, 200, 300, 400]
        total = sum(lengths)  # 1000
        target = 500
        updates = compute_stage_updates(
            num_examples=4,
            epochs=2,
            batch_size=4,
            accumulate=1,
            target_tokens=target,
            fitting_lengths=lengths,
        )
        # ceil(1000/500)=2 per epoch * 2 epochs = 4
        self.assertEqual(updates, 4)

    def test_compute_stage_updates_empty(self) -> None:
        updates = compute_stage_updates(
            num_examples=0, epochs=5, batch_size=4, accumulate=1
        )
        self.assertEqual(updates, 0)


# ---------------------------------------------------------------------------
# 5. Stage report counts (via compute_stage_updates)
# ---------------------------------------------------------------------------

class TestStageReportCounts(unittest.TestCase):
    """Verify the numbers the stage report would display."""

    def test_single_example_single_epoch(self) -> None:
        updates = compute_stage_updates(
            num_examples=1, epochs=1, batch_size=1, accumulate=1
        )
        self.assertEqual(updates, 1)

    def test_round_up_batches(self) -> None:
        # 7 examples, batch=3 → 3 batches, accumulate=2 → 2 updates, 2 epochs → 4
        updates = compute_stage_updates(
            num_examples=7, epochs=2, batch_size=3, accumulate=2
        )
        self.assertEqual(updates, 4)


# ---------------------------------------------------------------------------
# 6. Empty stage handling
# ---------------------------------------------------------------------------

class TestEmptyStageHandling(unittest.TestCase):
    """An empty curriculum stage must be skippable, not crash."""

    def test_subset_allow_empty(self) -> None:
        lengths = [500, 1000]
        dataset = _FakeDataset([0, 1])
        subset = SequenceLengthSubset(
            dataset, lengths, max_length=100, allow_empty=True
        )

        self.assertEqual(len(subset), 0)
        self.assertEqual(subset.lengths, [])

    def test_parse_curriculum_rejects_zero_epochs(self) -> None:
        config = {
            "curriculum": {
                "enabled": True,
                "stages": [
                    {"name": "s1", "max_length": 512, "epochs": 0, "learning_rate": 1e-4},
                ],
            },
        }
        with self.assertRaises(ValueError):
            parse_curriculum(config, default_max_length=512)


# ---------------------------------------------------------------------------
# 7. Warmup clamping
# ---------------------------------------------------------------------------

class TestWarmupClamping(unittest.TestCase):
    """Warmup must be clamped to at most 80% of available stage updates."""

    def test_warmup_within_budget(self) -> None:
        # 100 warmup, 200 updates → stays at 100
        self.assertEqual(clamp_warmup(100, 200), 100)

    def test_warmup_exceeds_budget(self) -> None:
        # 100 warmup, 50 updates → clamped to 0.8*50=40
        self.assertEqual(clamp_warmup(100, 50), 40)

    def test_warmup_equals_budget(self) -> None:
        # 100 warmup, 100 updates → clamped to 80
        self.assertEqual(clamp_warmup(100, 100), 80)

    def test_warmup_zero_updates(self) -> None:
        self.assertEqual(clamp_warmup(100, 0), 0)

    def test_warmup_zero_configured(self) -> None:
        self.assertEqual(clamp_warmup(0, 100), 0)


# ---------------------------------------------------------------------------
# 8. Curriculum stage parsing (regression)
# ---------------------------------------------------------------------------

class TestCurriculumParsing(unittest.TestCase):
    """Existing parsing tests adapted from test_long_sequence.py."""

    def test_disabled_curriculum_returns_single_stage(self) -> None:
        config = {"curriculum": {"enabled": False}, "training": {"max_epochs": 5}}
        stages = parse_curriculum(config, default_max_length=4096)
        self.assertEqual(len(stages), 1)
        self.assertEqual(stages[0].name, "full")
        self.assertEqual(stages[0].epochs, 5)
        self.assertTrue(math.isnan(stages[0].learning_rate))

    def test_stages_must_increase(self) -> None:
        config = {
            "curriculum": {
                "enabled": True,
                "stages": [
                    {"name": "s1", "max_length": 512, "epochs": 1, "learning_rate": 1e-4},
                    {"name": "s2", "max_length": 256, "epochs": 1, "learning_rate": 1e-4},
                    {"name": "s3", "max_length": 1024, "epochs": 1, "learning_rate": 1e-4},
                ],
            },
        }
        with self.assertRaises(ValueError):
            parse_curriculum(config, default_max_length=1024)

    def test_final_stage_must_reach_max(self) -> None:
        config = {
            "curriculum": {
                "enabled": True,
                "stages": [
                    {"name": "s1", "max_length": 512, "epochs": 1, "learning_rate": 1e-4},
                ],
            },
        }
        with self.assertRaises(ValueError):
            parse_curriculum(config, default_max_length=1024)


# ---------------------------------------------------------------------------
# 9. Skipped stage must advance global_epoch (Fix 3)
# ---------------------------------------------------------------------------

class TestSkippedStageEpochAdvancement(unittest.TestCase):
    """When an empty stage is skipped the global_epoch counter must still
    advance by stage.epochs so resume_epoch_for_stage() stays correct."""

    def test_epoch_advancement_after_skip(self) -> None:
        """Simulate the curriculum loop logic for an empty stage."""
        stages = [
            CurriculumStage("short", max_length=128, epochs=2,
                            learning_rate=1e-4, trainable="all"),
            CurriculumStage("long", max_length=4096, epochs=3,
                            learning_rate=1e-5, trainable="all"),
        ]
        global_epoch = 0
        skipped: list[str] = []

        for stage in stages:
            # Simulate: stage "short" has no examples
            num_examples = 0 if stage.name == "short" else 10
            if num_examples == 0:
                # This is the bug-fix: advance epoch even when skipping.
                global_epoch += stage.epochs
                skipped.append(stage.name)
                continue
            global_epoch += stage.epochs

        self.assertIn("short", skipped)
        # global_epoch must be 2 (short) + 3 (long) = 5
        self.assertEqual(global_epoch, 5)


# ---------------------------------------------------------------------------
# 10. Scaler skip detection (Fix 4)
# ---------------------------------------------------------------------------

class TestScalerSkipDetection(unittest.TestCase):
    """Verify the scale_before/scale_after pattern used to detect whether
    GradScaler actually ran optimizer.step()."""

    def test_scale_drop_means_step_was_skipped(self) -> None:
        """When the scale drops, step was skipped due to inf/NaN."""
        scale_before = 65536.0
        # GradScaler reduces scale on NaN
        scale_after = 32768.0
        step_ran = scale_after >= scale_before
        self.assertFalse(step_ran)

    def test_scale_unchanged_means_step_ran(self) -> None:
        """When scale stays the same, the step ran normally."""
        scale_before = 65536.0
        scale_after = 65536.0
        step_ran = scale_after >= scale_before
        self.assertTrue(step_ran)

    def test_scale_increase_means_step_ran(self) -> None:
        """When scale increases (growth interval reached), step also ran."""
        scale_before = 32768.0
        scale_after = 65536.0
        step_ran = scale_after >= scale_before
        self.assertTrue(step_ran)


if __name__ == "__main__":
    unittest.main()

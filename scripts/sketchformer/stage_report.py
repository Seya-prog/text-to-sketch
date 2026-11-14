"""Report curriculum stage statistics before training.

Prints, for each planned curriculum stage:
  - Number of training and validation drawings that fit the stage limit.
  - Length distribution (min, p25, median, p75, max).
  - Number of optimizer updates the stage can produce.
  - Whether warmup steps exceed available updates.
  - Whether the stage is empty or suspiciously small.

Exits with code 1 if any stage has zero examples.

Usage
-----
    python -m scripts.sketchformer.stage_report --config configs/train.yaml \
        --experiment anime_tok_dict_long

    # or via entry-point once installed:
    tts-stage-report --experiment anime_tok_dict_long
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any


def _add_project_to_path() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists() and (parent / "configs").exists():
            sys.path.insert(0, str(parent))
            return parent
    raise RuntimeError("Could not find project root directory.")


PROJECT_ROOT = _add_project_to_path()

import numpy as np

from builders.config_utils import get_nested
from dataloaders import StrokeSequenceDataModule
from scripts.sketchformer.config import batch_limit, compose_training_config
from scripts.sketchformer.curriculum import compute_stage_updates, parse_curriculum


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument(
        "--min-examples",
        type=int,
        default=10,
        help="Warn if a stage has fewer training examples than this.",
    )
    return parser.parse_args()


def _length_stats(lengths: list[int]) -> dict[str, int]:
    """Return min/p25/median/p75/max for a list of lengths."""
    if not lengths:
        return {"min": 0, "p25": 0, "median": 0, "p75": 0, "max": 0}
    arr = np.array(lengths)
    return {
        "min": int(arr.min()),
        "p25": int(np.percentile(arr, 25)),
        "median": int(np.median(arr)),
        "p75": int(np.percentile(arr, 75)),
        "max": int(arr.max()),
    }


def _count_fitting(lengths: list[int], max_length: int) -> int:
    """Count complete sequences that fit within max_length."""
    return sum(1 for length in lengths if length <= max_length)



def _report_stages(
    config: dict[str, Any],
    *,
    min_examples: int = 10,
) -> bool:
    """Print stage statistics.  Returns True if all stages are viable."""

    max_length = int(get_nested(config, "data.sequence.max_length"))
    stages = parse_curriculum(config["trainer"], default_max_length=max_length)

    datamodule = StrokeSequenceDataModule(
        config["data"],
        project_root=PROJECT_ROOT,
    )
    datamodule.setup("fit")
    assert datamodule.train_dataset is not None
    assert datamodule.valid_dataset is not None

    train_lengths = datamodule.train_dataset.lengths
    valid_lengths = datamodule.valid_dataset.lengths

    batch_size = int(get_nested(config, "data.batching.batch_size", 2))
    accumulate = int(
        get_nested(config, "trainer.training.accumulate_grad_batches", 1)
    )
    target_tokens = int(
        get_nested(config, "trainer.training.target_tokens_per_step", 0) or 0
    )
    warmup_steps = int(get_nested(config, "optimizer.scheduler.warmup_steps", 0))
    limit_train = get_nested(config, "trainer.training.limit_train_batches", 1.0)

    all_viable = True

    # Header.
    print(f"Dataset: train={len(train_lengths)} valid={len(valid_lengths)}")
    print(
        f"Overall length: train min={min(train_lengths)} "
        f"median={int(np.median(train_lengths))} "
        f"max={max(train_lengths)}"
    )
    print(
        f"Overall length: valid min={min(valid_lengths)} "
        f"median={int(np.median(valid_lengths))} "
        f"max={max(valid_lengths)}"
    )
    print(f"Batch size={batch_size}  accumulate={accumulate}  "
          f"target_tokens={target_tokens}  warmup_steps={warmup_steps}")
    print()

    for index, stage in enumerate(stages):
        train_fitting = [l for l in train_lengths if l <= stage.max_length]
        valid_fitting = [l for l in valid_lengths if l <= stage.max_length]

        # Compute updates using the actual loader so that bucket_by_length,
        # max_tokens_per_batch, and limit_train_batches are all reflected.
        is_empty = len(train_fitting) == 0
        if is_empty:
            updates = 0
        else:
            try:
                stage_loader = datamodule.train_dataloader(stage.max_length)
                num_batches = batch_limit(stage_loader, limit_train)
            except Exception:
                # Fall back to the simple calculation if loader fails.
                num_batches = compute_stage_updates(
                    len(train_fitting), 1, batch_size, 1,
                    target_tokens=target_tokens,
                    fitting_lengths=train_fitting,
                )
            updates = math.ceil(num_batches / accumulate) * stage.epochs

        train_stats = _length_stats(train_fitting)
        valid_stats = _length_stats(valid_fitting)

        warmup_ok = warmup_steps <= updates
        is_empty = len(train_fitting) == 0
        is_tiny = 0 < len(train_fitting) < min_examples

        status_flags: list[str] = []
        if is_empty:
            status_flags.append("EMPTY")
            all_viable = False
        if is_tiny:
            status_flags.append(f"TINY(<{min_examples})")
        if not warmup_ok:
            status_flags.append(f"WARMUP_EXCEEDS({warmup_steps}>{updates})")

        flag_str = f"  *** {', '.join(status_flags)} ***" if status_flags else ""

        print(f"Stage {index + 1}/{len(stages)}: {stage.name}{flag_str}")
        print(f"  max_length={stage.max_length}  epochs={stage.epochs}  "
              f"lr={stage.learning_rate:.8g}  trainable={stage.trainable}")
        print(f"  train: {len(train_fitting)} examples  "
              f"(min={train_stats['min']}  p25={train_stats['p25']}  "
              f"median={train_stats['median']}  p75={train_stats['p75']}  "
              f"max={train_stats['max']})")
        print(f"  valid: {len(valid_fitting)} examples  "
              f"(min={valid_stats['min']}  p25={valid_stats['p25']}  "
              f"median={valid_stats['median']}  p75={valid_stats['p75']}  "
              f"max={valid_stats['max']})")
        print(f"  optimizer_updates={updates}  "
              f"warmup_safe={'yes' if warmup_ok else 'NO'}")
        print()

    return all_viable


def main() -> int:
    args = parse_args()
    config = compose_training_config(args.config, experiment=args.experiment)
    if args.data_root:
        config["data"]["dataset"]["root"] = args.data_root

    all_viable = _report_stages(config, min_examples=args.min_examples)
    if not all_viable:
        print("ERROR: One or more stages have zero training examples.")
        print("Adjust stage limits or enable drawing windowing before training.")
        return 1
    print("All stages have training examples.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

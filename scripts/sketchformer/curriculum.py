"""Length curriculum contracts for long-sequence Sketchformer fine-tuning."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch


@dataclass(frozen=True)
class CurriculumStage:
    name: str
    max_length: int
    epochs: int
    learning_rate: float
    trainable: str = "all"


def parse_curriculum(
    trainer_config: Mapping[str, Any],
    *,
    default_max_length: int,
) -> list[CurriculumStage]:
    """Parse configured stages or return one conventional all-model stage."""

    curriculum = trainer_config.get("curriculum", {})
    if not bool(curriculum.get("enabled", False)):
        training = trainer_config.get("training", {})
        return [
            CurriculumStage(
                name="full",
                max_length=int(default_max_length),
                epochs=int(training.get("max_epochs", 1)),
                learning_rate=float("nan"),
                trainable="all",
            )
        ]

    raw_stages = curriculum.get("stages", [])
    if not raw_stages:
        raise ValueError("trainer.curriculum.enabled requires at least one stage")
    stages = [
        CurriculumStage(
            name=str(stage.get("name", f"stage-{index + 1}")),
            max_length=int(stage["max_length"]),
            epochs=int(stage["epochs"]),
            learning_rate=float(stage["learning_rate"]),
            trainable=str(stage.get("trainable", "all")),
        )
        for index, stage in enumerate(raw_stages)
    ]
    previous_length = 0
    for stage in stages:
        if stage.max_length <= previous_length:
            raise ValueError("curriculum max_length values must increase strictly")
        if stage.max_length > default_max_length:
            raise ValueError("curriculum stage exceeds model/data max_length")
        if stage.epochs <= 0 or stage.learning_rate <= 0:
            raise ValueError("curriculum epochs and learning_rate must be positive")
        if stage.trainable not in {"expander", "decoder", "all"}:
            raise ValueError("curriculum trainable must be expander, decoder, or all")
        previous_length = stage.max_length
    if stages[-1].max_length != default_max_length:
        raise ValueError("final curriculum stage must reach configured max_length")
    return stages


def set_trainable_scope(model: torch.nn.Module, scope: str) -> int:
    """Freeze model parameters outside the configured curriculum scope."""

    if scope not in {"expander", "decoder", "all"}:
        raise ValueError("scope must be expander, decoder, or all")
    for parameter in model.parameters():
        parameter.requires_grad = scope == "all"
    if scope == "expander":
        _set_module_trainable(model.latent_expander)
    elif scope == "decoder":
        for module_name in (
            "target_embedding",
            "latent_expander",
            "decoder",
            "reconstruction_head",
        ):
            module = getattr(model, module_name, None)
            if module is not None:
                _set_module_trainable(module)
    count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if count == 0:
        raise ValueError(f"curriculum scope {scope!r} selected no trainable parameters")
    return count


def resume_epoch_for_stage(
    stages: list[CurriculumStage],
    stage_index: int,
    completed_epochs: int,
) -> int:
    """Return the first local epoch to run for a resumed curriculum stage."""

    if stage_index < 0 or stage_index >= len(stages):
        raise IndexError("stage_index is outside the curriculum")
    before = sum(stage.epochs for stage in stages[:stage_index])
    return min(stages[stage_index].epochs, max(0, int(completed_epochs) - before))


def compute_stage_updates(
    num_examples: int,
    epochs: int,
    batch_size: int,
    accumulate: int,
    *,
    target_tokens: int = 0,
    fitting_lengths: Sequence[int] = (),
) -> int:
    """Compute the number of optimizer updates a curriculum stage can produce.

    When *target_tokens* > 0 (token-budget accumulation), updates are derived
    from the total token count divided by the target budget.  Otherwise,
    batch-count-based accumulation is used.
    """
    if num_examples <= 0:
        return 0
    if target_tokens > 0 and fitting_lengths:
        total_tokens = sum(int(length) for length in fitting_lengths)
        updates_per_epoch = max(1, math.ceil(total_tokens / target_tokens))
        return updates_per_epoch * epochs
    batches_per_epoch = max(1, math.ceil(num_examples / max(1, batch_size)))
    updates_per_epoch = max(1, math.ceil(batches_per_epoch / max(1, accumulate)))
    return updates_per_epoch * epochs


def clamp_warmup(configured_warmup: int, stage_updates: int) -> int:
    """Return warmup steps clamped to at most 80 % of available stage updates.

    A warmup that exceeds the stage length means the model never reaches
    the target learning rate.  This helper silently reduces warmup so the
    cosine-decay phase gets at least 20 % of the stage.
    """
    if stage_updates <= 0:
        return 0
    return min(configured_warmup, max(1, int(stage_updates * 0.8)))


def _set_module_trainable(module: torch.nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = True

"""Train the native in-repo Sketchformer model."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _add_project_to_path() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists() and (parent / "configs").exists():
            sys.path.insert(0, str(parent))
            return parent
    raise RuntimeError("Could not find project root directory.")


PROJECT_ROOT = _add_project_to_path()

import torch

from builders import (
    build_loss,
    build_model,
    build_optimizer,
    build_scheduler,
    maybe_compile_model,
)
from builders.config_utils import get_nested
from core import CheckpointCallback, average_logs, move_to_device, set_seed
from core.metrics import reconstruction_metrics
from dataloaders import StrokeSequenceDataModule
from scripts.sketchformer.config import (
    batch_limit,
    compose_training_config,
    format_logs,
    limited,
    parse_batch_limit,
    resolve_device,
)
from scripts.sketchformer.curriculum import (
    clamp_warmup,
    compute_stage_updates,
    parse_curriculum,
    resume_epoch_for_stage,
    set_trainable_scope,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--pretrained", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--limit-train-batches", type=parse_batch_limit, default=None)
    parser.add_argument("--limit-val-batches", type=parse_batch_limit, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    if args.data_root:
        config["data"]["dataset"]["root"] = args.data_root
    if args.output_dir:
        config.setdefault("experiment", {}).setdefault("run", {})[
            "output_dir"
        ] = args.output_dir
    if args.pretrained:
        pretrained = config.setdefault("experiment", {}).setdefault("pretrained", {})
        pretrained["use_converted_sketchformer_weights"] = True
        pretrained["path"] = args.pretrained
    if args.resume:
        config.setdefault("experiment", {}).setdefault("run", {})[
            "resume_from_checkpoint"
        ] = args.resume
    if args.precision:
        config.setdefault("trainer", {}).setdefault("runtime", {})["precision"] = args.precision
    if args.max_epochs is not None:
        config["trainer"]["training"]["max_epochs"] = args.max_epochs
    if args.limit_train_batches is not None:
        config["trainer"]["training"]["limit_train_batches"] = args.limit_train_batches
    if args.limit_val_batches is not None:
        config["trainer"]["training"]["limit_val_batches"] = args.limit_val_batches


def _checkpoint_dir(config: dict[str, Any]) -> Path:
    output_dir = get_nested(config, "experiment.run.output_dir")
    if output_dir:
        return PROJECT_ROOT / output_dir

    output_root = get_nested(config, "paths.output_root", "weights/finetuned")
    run_name = get_nested(config, "experiment.name", "sketchformer-run")
    return PROJECT_ROOT / output_root / run_name


def _total_optimizer_steps(config: dict[str, Any], train_loader: Any) -> int:
    max_epochs = int(get_nested(config, "trainer.training.max_epochs", 1))
    limit_train = get_nested(config, "trainer.training.limit_train_batches", 1.0)
    accumulate = int(get_nested(config, "trainer.training.accumulate_grad_batches", 1))
    batches = batch_limit(train_loader, limit_train)
    return max(1, math.ceil(batches / max(1, accumulate)) * max_epochs)


def _divide_gradients(model: torch.nn.Module, divisor: float) -> None:
    if divisor <= 0:
        raise ValueError("gradient divisor must be positive")
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.div_(divisor)


@dataclass(frozen=True)
class PrecisionRuntime:
    """Autocast and scaler settings resolved from trainer.runtime.precision."""

    requested: str
    effective: str
    autocast_dtype: torch.dtype | None
    scaler: Any


def _configure_torch_runtime(config: dict[str, Any], device: torch.device) -> None:
    """Apply device-level runtime settings that matter on RTX-class GPUs."""

    if device.type != "cuda":
        return

    benchmark = bool(get_nested(config, "trainer.runtime.benchmark", True))
    allow_tf32 = bool(get_nested(config, "trainer.runtime.allow_tf32", True))
    torch.backends.cudnn.benchmark = benchmark
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    if allow_tf32 and hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def _resolve_precision_runtime(
    config: dict[str, Any],
    device: torch.device,
) -> PrecisionRuntime:
    requested = str(get_nested(config, "trainer.runtime.precision", "32-true")).lower()
    effective = requested
    autocast_dtype: torch.dtype | None = None

    if requested in {"32", "32-true", "fp32", "float32"}:
        effective = "32-true"
    elif requested in {"16", "16-mixed", "fp16", "float16"}:
        if device.type != "cuda":
            print(f"[precision] {requested} requested on {device.type}; using 32-true")
            effective = "32-true"
        else:
            effective = "16-mixed"
            autocast_dtype = torch.float16
    elif requested in {"bf16", "bf16-mixed", "bfloat16"}:
        if device.type == "cuda" and not torch.cuda.is_bf16_supported():
            print("[precision] bf16 is not supported on this CUDA device; using 16-mixed")
            effective = "16-mixed"
            autocast_dtype = torch.float16
        elif device.type in {"cuda", "cpu"}:
            effective = "bf16-mixed"
            autocast_dtype = torch.bfloat16
        else:
            print(f"[precision] bf16 requested on {device.type}; using 32-true")
            effective = "32-true"
    else:
        raise ValueError(
            "trainer.runtime.precision must be one of 32-true, 16-mixed, or bf16-mixed"
        )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(device.type == "cuda" and effective == "16-mixed"),
    )
    return PrecisionRuntime(
        requested=requested,
        effective=effective,
        autocast_dtype=autocast_dtype,
        scaler=scaler,
    )


def _autocast_context(device: torch.device, precision: PrecisionRuntime):
    if precision.autocast_dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=precision.autocast_dtype)


def _resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def _load_state_dict_file(path: Path) -> dict[str, Any]:
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return dict(load_file(str(path)))

    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint must be a mapping: {path}")
    if "model" in checkpoint:
        return checkpoint["model"]
    if "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


def _load_pretrained_if_configured(model: torch.nn.Module, config: dict[str, Any]) -> None:
    use_pretrained = bool(
        get_nested(
            config,
            "experiment.pretrained.use_converted_sketchformer_weights",
            False,
        )
    ) or bool(get_nested(config, "model.checkpoint.load_converted_weights", False))
    if not use_pretrained:
        return

    path = get_nested(
        config,
        "experiment.pretrained.path",
        get_nested(config, "model.checkpoint.converted_weights_path"),
    )
    if not path:
        raise ValueError("Pretrained loading is enabled, but no pretrained path is configured")

    checkpoint_path = _resolve_project_path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "Configured pretrained weights do not exist: "
            f"{checkpoint_path}. Convert TensorFlow weights first or disable pretrained loading."
        )

    strict = bool(
        get_nested(
            config,
            "experiment.pretrained.strict",
            get_nested(config, "model.checkpoint.strict", False),
        )
    )
    incompatible = model.load_state_dict(_load_state_dict_file(checkpoint_path), strict=strict)
    print(
        "[pretrained] loaded={} missing_keys={} unexpected_keys={}".format(
            checkpoint_path,
            len(incompatible.missing_keys),
            len(incompatible.unexpected_keys),
        )
    )


def _validate(
    model,
    valid_loader,
    loss_fn,
    config: dict[str, Any],
    device: torch.device,
    precision: PrecisionRuntime,
    *,
    log_prefix: str = "val",
) -> dict[str, torch.Tensor]:
    model.eval()
    logs = []
    limit_val = get_nested(config, "trainer.training.limit_val_batches", 1.0)
    with torch.no_grad():
        for batch in limited(valid_loader, limit_val):
            batch = move_to_device(batch, device)
            with _autocast_context(device, precision):
                output = model(batch)
                loss_output = loss_fn(output, batch)
                metric_output = reconstruction_metrics(output, batch)
            step_logs = loss_output.as_log_dict(prefix=log_prefix)
            step_logs.update(metric_output.as_log_dict(prefix=log_prefix))
            logs.append(step_logs)
    weight_key = (
        f"{log_prefix}/valid_tokens"
        if logs and f"{log_prefix}/valid_tokens" in logs[0]
        else None
    )
    return average_logs(logs, weight_key=weight_key)


def main() -> int:
    args = parse_args()
    config = compose_training_config(args.config, experiment=args.experiment)
    _apply_cli_overrides(config, args)

    seed = int(get_nested(config, "project.seed", 42))
    deterministic = bool(get_nested(config, "trainer.runtime.deterministic", False))
    set_seed(seed, deterministic=deterministic)

    device = resolve_device(args.device)
    _configure_torch_runtime(config, device)
    precision = _resolve_precision_runtime(config, device)
    checkpoint_dir = _checkpoint_dir(config)
    curriculum_stages = parse_curriculum(
        config["trainer"],
        default_max_length=int(get_nested(config, "data.sequence.max_length")),
    )

    if args.dry_run:
        print(f"experiment={get_nested(config, 'experiment.name')}")
        print(f"data_root={get_nested(config, 'data.dataset.root')}")
        print(f"model={get_nested(config, 'model.name')}")
        print(f"device={device}")
        print(f"precision={precision.effective}")
        print(f"checkpoint_dir={checkpoint_dir}")
        print(f"max_epochs={get_nested(config, 'trainer.training.max_epochs')}")
        print(
            "curriculum="
            + ",".join(
                f"{stage.name}:{stage.max_length}x{stage.epochs}:{stage.trainable}"
                for stage in curriculum_stages
            )
        )
        return 0

    datamodule = StrokeSequenceDataModule(
        config["data"],
        project_root=PROJECT_ROOT,
        seed=seed,
    )
    datamodule.setup("fit")
    raw_model = build_model(config["model"])
    _load_pretrained_if_configured(raw_model, config)
    raw_model = raw_model.to(device)
    model = maybe_compile_model(raw_model, config["model"])
    loss_fn = build_loss(config["optimizer"]).to(device)
    resume_path = get_nested(config, "experiment.run.resume_from_checkpoint")
    resume_result = None
    if resume_path:
        from core import load_checkpoint

        resume_result = load_checkpoint(
            _resolve_project_path(resume_path),
            raw_model,
            strict=False,
        )
        print(
            f"[resume] restored epoch={resume_result.epoch} "
            f"step={resume_result.step}"
        )

    checkpoint_callback = CheckpointCallback(
        checkpoint_dir,
        monitor=str(
            get_nested(config, "trainer.checkpointing.monitor", "val/token_loss")
        ),
        mode=str(get_nested(config, "trainer.checkpointing.mode", "min")),
        save_last=bool(get_nested(config, "trainer.checkpointing.save_last", True)),
    )

    log_every = int(get_nested(config, "trainer.training.log_every_n_steps", 10))
    accumulate = max(
        1,
        int(get_nested(config, "trainer.training.accumulate_grad_batches", 1)),
    )
    grad_clip = get_nested(config, "optimizer.gradient.clip_norm", None)
    limit_train = get_nested(config, "trainer.training.limit_train_batches", 1.0)
    target_tokens = int(
        get_nested(config, "trainer.training.target_tokens_per_step", 0) or 0
    )
    early_stopping_patience = int(
        get_nested(config, "trainer.curriculum.early_stopping_patience", 0) or 0
    )
    configured_warmup = int(
        get_nested(config, "optimizer.scheduler.warmup_steps", 0)
    )
    global_step = resume_result.step if resume_result is not None else 0
    global_epoch = resume_result.epoch if resume_result is not None else 0

    # ------------------------------------------------------------------
    # Fixed held-out validation set: always the full validation split
    # so scores are comparable across curriculum stages.
    # ------------------------------------------------------------------
    held_out_loader = datamodule.val_dataloader(max_sequence_length=None)

    # Stage-progression log written alongside checkpoints.
    stage_log_path = checkpoint_dir / "stage_log.jsonl"
    stage_log_path.parent.mkdir(parents=True, exist_ok=True)

    for stage_index, stage in enumerate(curriculum_stages):
        start_stage_epoch = resume_epoch_for_stage(
            curriculum_stages,
            stage_index,
            global_epoch,
        )
        if start_stage_epoch >= stage.epochs:
            print(f"[curriculum] skip completed stage={stage.name}")
            continue

        # ------------------------------------------------------------------
        # Guard against empty stages: the stage report should have
        # caught this, but if not, skip rather than crash.
        # ------------------------------------------------------------------
        try:
            train_loader = datamodule.train_dataloader(stage.max_length)
        except ValueError:
            print(
                f"[curriculum] WARNING stage={stage.name} has no training "
                f"examples at max_length={stage.max_length} — skipping"
            )
            continue

        try:
            valid_loader = datamodule.val_dataloader(stage.max_length)
        except ValueError:
            # Fall back to full validation set if the stage filter is empty.
            valid_loader = held_out_loader

        trainable_parameters = set_trainable_scope(raw_model, stage.trainable)
        optimizer_config = copy.deepcopy(config["optimizer"])
        if not math.isnan(stage.learning_rate):
            optimizer_config.setdefault("optimizer", {})["lr"] = stage.learning_rate
        optimizer = build_optimizer(raw_model, optimizer_config)

        # Compute total optimizer updates for this stage.
        if target_tokens > 0:
            dataset_tokens = sum(int(length) for length in train_loader.dataset.lengths)
            total_steps = max(1, math.ceil(dataset_tokens / target_tokens) * stage.epochs)
        else:
            stage_config = copy.deepcopy(config)
            stage_config["trainer"]["training"]["max_epochs"] = stage.epochs
            total_steps = _total_optimizer_steps(stage_config, train_loader)

        # ------------------------------------------------------------------
        # Fix: clamp warmup so it never exceeds the stage's update budget.
        # ------------------------------------------------------------------
        effective_warmup = clamp_warmup(configured_warmup, total_steps)
        if effective_warmup != configured_warmup:
            print(
                f"[curriculum] warmup clamped {configured_warmup} → "
                f"{effective_warmup} for stage={stage.name} "
                f"(total_steps={total_steps})"
            )
        stage_scheduler_config = copy.deepcopy(optimizer_config)
        stage_scheduler_config.setdefault("scheduler", {})["warmup_steps"] = effective_warmup
        scheduler = build_scheduler(
            optimizer,
            stage_scheduler_config,
            total_steps=total_steps,
        )

        if resume_path and start_stage_epoch > 0:
            from core import load_checkpoint

            load_checkpoint(
                _resolve_project_path(resume_path),
                raw_model,
                optimizer=optimizer,
                scheduler=scheduler,
                strict=False,
            )
        checkpoint_callback.tracker.best = (
            float(resume_result.metrics[checkpoint_callback.monitor])
            if resume_result is not None
            and start_stage_epoch > 0
            and checkpoint_callback.monitor in resume_result.metrics
            else None
        )
        non_improving = 0
        print(
            f"[curriculum] stage={stage.name} max_length={stage.max_length} "
            f"epochs={stage.epochs} trainable={stage.trainable} "
            f"parameters={trainable_parameters} lr={optimizer.param_groups[0]['lr']:.8g} "
            f"total_steps={total_steps} warmup={effective_warmup}"
        )

        for stage_epoch in range(start_stage_epoch, stage.epochs):
            global_epoch += 1
            model.train()
            optimizer.zero_grad(set_to_none=True)
            train_logs = []
            num_batches = batch_limit(train_loader, limit_train)
            accumulated_tokens = 0
            accumulated_micro = 0  # micro-batches accumulated so far

            for batch_index, batch in enumerate(limited(train_loader, limit_train)):
                batch = move_to_device(batch, device)
                with _autocast_context(device, precision):
                    output = model(batch)
                    loss_output = loss_fn(output, batch)
                    batch_tokens = int(
                        loss_output.valid_tokens.item()
                        if loss_output.valid_tokens is not None
                        else batch["valid_mask"].sum().item()
                    )
                    scaled_loss = (
                        loss_output.total * batch_tokens
                        if target_tokens > 0
                        else loss_output.total / accumulate
                    )

                precision.scaler.scale(scaled_loss).backward()
                accumulated_tokens += batch_tokens
                accumulated_micro += 1
                train_logs.append(loss_output.as_log_dict(prefix="train"))

                is_full_accumulation = (
                    (target_tokens > 0 and accumulated_tokens >= target_tokens)
                    or (target_tokens <= 0 and (batch_index + 1) % accumulate == 0)
                )
                is_last_batch = (batch_index + 1) == num_batches

                should_step = is_full_accumulation or is_last_batch

                if should_step:
                    precision.scaler.unscale_(optimizer)
                    if target_tokens > 0:
                        _divide_gradients(raw_model, float(accumulated_tokens))
                    elif not is_full_accumulation and is_last_batch:
                        # -------------------------------------------------------
                        # Fix: partial gradient group at epoch/stage end.
                        # Re-weight so the final partial group has the correct
                        # effective learning-rate contribution.  The loss was
                        # divided by ``accumulate`` in each micro-batch; we
                        # need to multiply by ``accumulate / actual`` to
                        # compensate.
                        # -------------------------------------------------------
                        correction = float(accumulate) / float(accumulated_micro)
                        _divide_gradients(raw_model, 1.0 / correction)
                    if grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), float(grad_clip))
                    precision.scaler.step(optimizer)
                    precision.scaler.update()

                    # -------------------------------------------------------
                    # Fix: only advance the LR schedule when a real optimizer
                    # update happened (not when scaler skipped due to NaN).
                    # -------------------------------------------------------
                    if scheduler.scheduler is not None:
                        scheduler.scheduler.step()

                    optimizer.zero_grad(set_to_none=True)
                    accumulated_tokens = 0
                    accumulated_micro = 0
                    global_step += 1

                if (batch_index + 1) % log_every == 0:
                    current_lr = optimizer.param_groups[0]["lr"]
                    print(
                        f"stage={stage.name} epoch={stage_epoch + 1} "
                        f"batch={batch_index + 1} lr={current_lr:.8g} "
                        f"step={global_step} "
                        f"{format_logs(train_logs[-1])}"
                    )

            train_weight_key = (
                "train/valid_tokens"
                if train_logs and "train/valid_tokens" in train_logs[0]
                else None
            )
            train_epoch_logs = average_logs(train_logs, weight_key=train_weight_key)

            # Stage-filtered validation.
            val_logs = _validate(
                model, valid_loader, loss_fn, config, device, precision,
                log_prefix="val",
            )

            # Fixed held-out validation (same set across all stages).
            held_out_logs = _validate(
                model, held_out_loader, loss_fn, config, device, precision,
                log_prefix="held_out",
            )

            saved = checkpoint_callback.on_validation_end(
                raw_model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=global_epoch,
                step=global_step,
                metrics={key: float(value.detach().cpu()) for key, value in val_logs.items()},
            )
            non_improving = 0 if "best" in saved else non_improving + 1

            # ------------------------------------------------------------------
            # Log: record stage, LR, and update count for post-hoc analysis.
            # ------------------------------------------------------------------
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"stage={stage.name} epoch={stage_epoch + 1} "
                f"train {format_logs(train_epoch_logs)}"
            )
            print(
                f"stage={stage.name} epoch={stage_epoch + 1} "
                f"valid {format_logs(val_logs)}"
            )
            print(
                f"stage={stage.name} epoch={stage_epoch + 1} "
                f"held_out {format_logs(held_out_logs)}"
            )
            print(
                f"stage={stage.name} epoch={stage_epoch + 1} "
                f"lr={current_lr:.8g} global_step={global_step}"
            )

            stage_entry = {
                "stage": stage.name,
                "stage_epoch": stage_epoch + 1,
                "global_epoch": global_epoch,
                "global_step": global_step,
                "lr": current_lr,
                "max_length": stage.max_length,
                "trainable": stage.trainable,
            }
            for key, value in val_logs.items():
                stage_entry[key] = float(value.detach().cpu())
            for key, value in held_out_logs.items():
                stage_entry[key] = float(value.detach().cpu())
            with open(stage_log_path, "a", encoding="utf-8") as log_file:
                log_file.write(json.dumps(stage_entry) + "\n")

            is_final_stage = stage_index == len(curriculum_stages) - 1
            if (
                is_final_stage
                and early_stopping_patience > 0
                and non_improving >= early_stopping_patience
            ):
                print(
                    f"[early-stopping] stage={stage.name} "
                    f"patience={early_stopping_patience}"
                )
                break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

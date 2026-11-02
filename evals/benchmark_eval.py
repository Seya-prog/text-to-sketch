"""Deterministic benchmark evaluation for Sketchformer.

Runs both teacher-forced and free-running evaluation on a specified split,
reports per-bucket metrics, saves a JSON report with full provenance, and
generates a fixed comparison sample set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
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
import torch

from builders import build_loss, build_model, maybe_compile_model
from builders.config_utils import get_nested
from core import average_logs, load_checkpoint, move_to_device
from core.metrics import reconstruction_metrics
from dataloaders import StrokeSequenceDataModule
from metrics.sketchformer.free_running import (
    aggregate_free_running_records,
    free_running_reconstruction_records,
)
from metrics.sketchformer.reconstruction import (
    collect_generated_reconstruction_examples,
    collect_reconstruction_examples,
    write_metrics_report,
)
from prep_data.sketch_token.create_token_dict import load_codebook_from_dir
from scripts.sketchformer.config import (
    compose_training_config,
    format_logs,
    limited,
    parse_batch_limit,
    resolve_device,
)
from scripts.sketchformer.train import (
    _autocast_context,
    _configure_torch_runtime,
    _resolve_precision_runtime,
)


def _file_sha256(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _config_hash(config: dict[str, Any]) -> str:
    """Deterministic hash of the configuration dictionary."""
    config_str = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


def _load_model_checkpoint(path: Path, model: torch.nn.Module, strict: bool = False) -> None:
    """Load model weights supporting both .safetensors and torch checkpoints."""
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state_dict = load_file(str(path))
    else:
        checkpoint = torch.load(path, map_location="cpu")
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
        else:
            state_dict = checkpoint
    model.load_state_dict(state_dict, strict=strict)


def _load_codebook(config: dict[str, Any]) -> np.ndarray:
    """Load the token codebook from config paths."""
    codebook_dir = get_nested(config, "data.format.token_dictionary.codebook_dir")
    codebook_path = get_nested(config, "data.format.token_dictionary.codebook_path")
    if codebook_dir:
        codebook, _metadata = load_codebook_from_dir(PROJECT_ROOT / codebook_dir)
        return codebook
    if codebook_path:
        path = Path(codebook_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return np.load(path)
    raise ValueError(
        "Benchmark evaluation requires "
        "data.format.token_dictionary.codebook_dir or codebook_path"
    )


def run_teacher_forced(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    loader: Any,
    device: torch.device,
    precision: Any,
    *,
    limit_batches: float | int = 1.0,
    split: str = "valid",
) -> dict[str, Any]:
    """Run teacher-forced evaluation and return aggregated metrics."""
    logs: list[dict[str, torch.Tensor]] = []
    model.eval()
    total_samples = 0
    start_time = time.perf_counter()
    bucket_accs: dict[str, list[float]] = {}

    with torch.no_grad():
        for batch in limited(loader, limit_batches):
            batch = move_to_device(batch, device)
            with _autocast_context(device, precision):
                output = model(batch)
                loss_output = loss_fn(output, batch)
                metric_output = reconstruction_metrics(output, batch)
                step_logs = loss_output.as_log_dict(prefix=split)
                step_logs.update(metric_output.as_log_dict(prefix=split))
            logs.append(step_logs)
            batch_size = batch["targets"].shape[0]
            total_samples += batch_size

            # Per-bucket accuracy collection
            token_logits = getattr(getattr(output, "reconstruction", None), "token_logits", None)
            if token_logits is not None:
                targets = batch["targets"].to(device=token_logits.device, dtype=torch.long)
                valid_mask = batch.get("valid_mask")
                if valid_mask is None:
                    valid_mask = torch.ones(targets.shape[:2], dtype=torch.bool, device=targets.device)
                else:
                    valid_mask = valid_mask.to(device=targets.device, dtype=torch.bool)
                preds = torch.argmax(token_logits, dim=-1)
                matches = (preds == targets) & valid_mask
                lengths = batch.get("lengths")
                from evals.benchmark_manifest import length_bucket
                for row_i in range(batch_size):
                    row_mask = valid_mask[row_i]
                    if row_mask.sum() > 0:
                        row_acc = float(matches[row_i][row_mask].float().mean().item())
                        row_len = int(lengths[row_i]) if lengths is not None else int(row_mask.sum().item())
                        b = length_bucket(row_len)
                        bucket_accs.setdefault(b, []).append(row_acc)

    elapsed = time.perf_counter() - start_time
    weight_key = f"{split}/valid_tokens"
    summary = average_logs(
        logs,
        weight_key=weight_key if logs and weight_key in logs[0] else None,
    )

    # Collect per-bucket metrics if per-sample bucket accuracies exist
    per_bucket: dict[str, dict[str, Any]] = {}
    if bucket_accs:
        for b, acc_list in sorted(bucket_accs.items()):
            per_bucket[b] = {
                "count": len(acc_list),
                "tf_token_accuracy_mean": round(float(np.mean(acc_list)), 4),
                "tf_token_accuracy_median": round(float(np.median(acc_list)), 4),
            }

    return {
        "mode": "teacher-forced",
        "total_samples": total_samples,
        "wall_clock_seconds": round(elapsed, 2),
        "seconds_per_sample": round(elapsed / max(total_samples, 1), 4),
        "metrics": {
            key: round(float(value), 6) for key, value in summary.items()
        },
        "per_bucket": per_bucket,
    }


def run_free_running(
    model: torch.nn.Module,
    loader: Any,
    device: torch.device,
    precision: Any,
    codebook: np.ndarray,
    *,
    eos_token_id: int,
    limit_batches: float | int = 1.0,
    split: str = "valid",
    max_generation_length: int | None = None,
    num_plots: int = 50,
    save_predictions: bool = False,
) -> tuple[dict[str, Any], list[Any]]:
    """Run free-running evaluation and return metrics + reconstruction examples."""
    model.eval()
    all_records: list[dict[str, float | int | str]] = []
    examples: list[Any] = []
    total_samples = 0
    start_time = time.perf_counter()

    # Access the raw model for generate() (unwrap compiled model if needed)
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model

    with torch.no_grad():
        for batch in limited(loader, limit_batches):
            batch = move_to_device(batch, device)
            with _autocast_context(device, precision):
                generation = raw_model.generate(
                    batch,
                    max_length=max_generation_length,
                    use_cache=True,
                )
                batch_records = free_running_reconstruction_records(
                    generation.tokens,
                    generation.lengths,
                    batch,
                    codebook,
                    eos_token_id=eos_token_id,
                )
                if save_predictions:
                    targets = torch.as_tensor(batch["targets"]).detach().cpu()
                    t_lens = torch.as_tensor(batch["lengths"]).detach().cpu().long()
                    preds = generation.tokens.detach().cpu().long()
                    p_lens = generation.lengths.detach().cpu().long()
                    for idx, rec in enumerate(batch_records):
                        rec["target_tokens"] = targets[idx, : t_lens[idx]].tolist()
                        rec["predicted_tokens"] = preds[idx, : p_lens[idx]].tolist()
                # Attach sample metadata to records
                sample_ids = batch.get("sample_ids")
                source_files = batch.get("source_files")
                source_indices = batch.get("source_indices")
                for idx, rec in enumerate(batch_records):
                    if sample_ids is not None and idx < len(sample_ids):
                        rec["sample_id"] = str(sample_ids[idx])
                    if source_files is not None and idx < len(source_files):
                        rec["source_file"] = str(source_files[idx])
                    if source_indices is not None and idx < len(source_indices):
                        rec["source_index"] = int(source_indices[idx])

                all_records.extend(batch_records)

            batch_size = batch["targets"].shape[0]
            total_samples += batch_size

            # Collect comparison examples
            remaining = num_plots - len(examples)
            if remaining > 0:
                examples.extend(
                    collect_generated_reconstruction_examples(
                        generation,
                        batch,
                        max_examples=remaining,
                        codebook=codebook,
                    )
                )

    elapsed = time.perf_counter() - start_time
    free_summary = aggregate_free_running_records(all_records, device=device)

    return {
        "mode": "free-running",
        "total_samples": total_samples,
        "wall_clock_seconds": round(elapsed, 2),
        "seconds_per_sample": round(elapsed / max(total_samples, 1), 4),
        "metrics": {
            f"{split}/{key}": round(float(value), 6)
            for key, value in free_summary.items()
        },
        "per_sample_records": all_records,
    }, examples


def build_provenance(
    args: argparse.Namespace,
    config: dict[str, Any],
    codebook_path: Path | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Build a provenance block for full reproducibility."""
    effective_seed = seed if seed is not None else (
        args.seed if getattr(args, "seed", None) is not None else int(config.get("project", {}).get("seed", 42))
    )
    provenance: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checkpoint": args.checkpoint,
        "experiment": args.experiment,
        "split": args.split,
        "seed": int(effective_seed),
        "config_hash": _config_hash(config),
        "data_root": config["data"]["dataset"]["root"],
        "max_generation_length": args.max_generation_length,
        "limit_batches": args.limit_batches,
    }
    if codebook_path and codebook_path.exists():
        provenance["codebook_sha256"] = _file_sha256(codebook_path)
    if args.checkpoint:
        ckpt_path = PROJECT_ROOT / args.checkpoint
        if ckpt_path.exists():
            provenance["checkpoint_sha256"] = _file_sha256(ckpt_path)

    # GPU info
    if torch.cuda.is_available():
        provenance["gpu"] = torch.cuda.get_device_name(0)
        provenance["gpu_memory_total_mb"] = round(
            torch.cuda.get_device_properties(0).total_mem / 1024 / 1024, 0
        )

    return provenance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument(
        "--experiment",
        default="anime_tok_dict_finetune",
        help="Experiment name (defaults to anime_tok_dict_finetune for full 2048 sequence length).",
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", choices=["valid", "test"], default="valid")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", default=None)
    parser.add_argument("--limit-batches", type=parse_batch_limit, default=1.0)
    parser.add_argument("--max-generation-length", type=int, default=None)
    parser.add_argument("--num-plots", type=int, default=50)
    parser.add_argument(
        "--output-dir",
        default="evals/results",
        help="Directory for the JSON report and comparison plots.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Name for this benchmark run (used in output filenames).",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["teacher-forced", "free-running"],
        choices=["teacher-forced", "free-running"],
        help="Evaluation modes to run.",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Include target and predicted token arrays in per-sample records for failure analysis.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the random seed for reproducibility.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = compose_training_config(args.config, experiment=args.experiment)
    if args.data_root:
        config["data"]["dataset"]["root"] = args.data_root
    if args.precision:
        config.setdefault("trainer", {}).setdefault("runtime", {})["precision"] = (
            args.precision
        )

    # Seed for reproducibility
    seed = args.seed or int(config.get("project", {}).get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = resolve_device(args.device)
    _configure_torch_runtime(config, device)
    precision = _resolve_precision_runtime(config, device)

    # Data
    datamodule = StrokeSequenceDataModule(config["data"], project_root=PROJECT_ROOT)
    datamodule.setup("test" if args.split == "test" else "fit")
    loader = (
        datamodule.test_dataloader()
        if args.split == "test"
        else datamodule.val_dataloader()
    )

    # Model
    raw_model = build_model(config["model"])
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.is_absolute():
            ckpt_path = PROJECT_ROOT / ckpt_path
        _load_model_checkpoint(ckpt_path, raw_model, strict=False)
        print(f"[benchmark] loaded checkpoint: {args.checkpoint}")
    else:
        print(
            "[warning] evaluating randomly initialized model; "
            "pass --checkpoint for trained weights"
        )
    model = maybe_compile_model(raw_model.to(device), config["model"])

    # Loss (for teacher-forced mode)
    loss_fn = build_loss(config["optimizer"]).to(device)

    # Codebook (for free-running mode)
    codebook = None
    codebook_path = None
    if "free-running" in args.modes:
        codebook = _load_codebook(config)
        raw_codebook_path = get_nested(
            config, "data.format.token_dictionary.codebook_path"
        )
        if raw_codebook_path:
            codebook_path = PROJECT_ROOT / raw_codebook_path

    # Output setup
    run_name = args.run_name or f"benchmark_{args.split}"
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build report
    config.setdefault("project", {})["seed"] = seed
    report: dict[str, Any] = {
        "provenance": build_provenance(args, config, codebook_path, seed=seed),
    }

    # Track peak GPU memory
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # --- Teacher-forced evaluation ---
    if "teacher-forced" in args.modes:
        print("\n[benchmark] Running teacher-forced evaluation...")
        tf_results = run_teacher_forced(
            model,
            loss_fn,
            loader,
            device,
            precision,
            limit_batches=args.limit_batches,
            split=args.split,
        )
        report["teacher_forced"] = tf_results
        print(f"  → {tf_results['total_samples']} samples in {tf_results['wall_clock_seconds']}s")
        for key, value in tf_results["metrics"].items():
            print(f"  {key}: {value:.6f}")

    # --- Free-running evaluation ---
    examples = []
    if "free-running" in args.modes:
        assert codebook is not None
        print("\n[benchmark] Running free-running evaluation...")
        fr_results, examples = run_free_running(
            model,
            loader,
            device,
            precision,
            codebook,
            eos_token_id=int(
                get_nested(config, "data.format.token_dictionary.eos_token_id")
            ),
            limit_batches=args.limit_batches,
            split=args.split,
            max_generation_length=args.max_generation_length,
            num_plots=args.num_plots,
            save_predictions=args.save_predictions,
        )
        # Remove per-sample records from report (save separately)
        per_sample = fr_results.pop("per_sample_records", [])
        report["free_running"] = fr_results
        print(f"  → {fr_results['total_samples']} samples in {fr_results['wall_clock_seconds']}s")
        for key, value in fr_results["metrics"].items():
            print(f"  {key}: {value:.6f}")

        # Save per-sample records for failure analysis
        per_sample_path = output_dir / f"{run_name}_per_sample.jsonl"
        with open(per_sample_path, "w", encoding="utf-8") as f:
            for record in per_sample:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        print(f"\n[benchmark] wrote per-sample records to {per_sample_path}")

    # GPU memory
    if torch.cuda.is_available():
        peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        report["provenance"]["peak_gpu_memory_mb"] = round(peak_mb, 1)

    # Save report
    report_path = output_dir / f"{run_name}_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"\n[benchmark] wrote report to {report_path}")

    # Save comparison plots
    if examples:
        from metrics.sketchformer.visualisation import save_reconstruction_examples

        plots_dir = output_dir / f"{run_name}_plots"
        saved = save_reconstruction_examples(examples, plots_dir)
        print(f"[benchmark] wrote {len(saved)} comparison plots to {plots_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


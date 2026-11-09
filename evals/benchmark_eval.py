"""Deterministic benchmark evaluation for Sketchformer.

Runs both teacher-forced and free-running evaluation on a specified split,
reports per-bucket metrics, saves a JSON report with full provenance, and
generates a fixed comparison sample set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
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

from evals.benchmark_common import (
    ACCURACY_DEFINITION,
    LENGTH_DEFINITION,
    length_bucket,
    length_bucket_range,
    records_by_id,
    sample_metadata,
    sample_rank,
    select_comparison_samples,
)
from builders import build_loss, build_model, maybe_compile_model
from builders.config_utils import get_nested
from core import move_to_device
from dataloaders import StrokeSequenceDataModule
from metrics.sketchformer.free_running import (
    free_running_reconstruction_records,
)
from metrics.sketchformer.reconstruction import (
    collect_generated_reconstruction_examples,
)
from prep_data.sketch_token.create_token_dict import load_codebook_from_dir
from scripts.sketchformer.config import (
    compose_training_config,
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


def _load_model_checkpoint(
    path: Path, model: torch.nn.Module, strict: bool = True
) -> dict[str, Any]:
    """Load model weights supporting both .safetensors and torch checkpoints."""
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state_dict = load_file(str(path))
    else:
        checkpoint = torch.load(path, map_location="cpu")
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get(
                "model", checkpoint.get("state_dict", checkpoint)
            )
        else:
            state_dict = checkpoint
    if not isinstance(state_dict, dict) or not (
        model.state_dict().keys() & state_dict.keys()
    ):
        raise ValueError("Checkpoint contains no matching model weights")
    incompatible = model.load_state_dict(state_dict, strict=strict)
    return {
        "strict": strict,
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }


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


def _annotate_record(
    batch: dict[str, Any], row: int, sep_token_id: int = 1001
) -> dict[str, Any]:
    length = int(batch["lengths"][row])
    tokens = batch["targets"][row, :length].detach().cpu().numpy().astype(np.int64)
    return {
        **sample_metadata(batch, row),
        "target_length": length,
        "length_bucket": length_bucket(length),
        "length_bucket_range": length_bucket_range(length),
        "target_sha256": hashlib.sha256(tokens.tobytes()).hexdigest(),
        "accuracy_definition": ACCURACY_DEFINITION,
        "stroke_count": int(np.sum(tokens == sep_token_id))
        + int(len(tokens) > 2 and tokens[-2] != sep_token_id),
    }


def _tf_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    count = sum(r["tf_scored_tokens"] for r in records)
    loss = sum(r["tf_token_loss_sum"] for r in records) / count
    return {
        "count": len(records),
        "valid_tokens": count,
        "tf_token_loss": loss,
        "tf_token_perplexity": float(np.exp(min(loss, 50.0))),
        "tf_token_accuracy": sum(r["tf_correct_tokens"] for r in records) / count,
        "tf_token_accuracy_mean": float(
            np.mean([r["tf_token_accuracy"] for r in records])
        ),
        "tf_token_accuracy_median": float(
            np.median([r["tf_token_accuracy"] for r in records])
        ),
    }


def run_teacher_forced(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    loader: Any,
    device: torch.device,
    precision: Any,
    *,
    limit_batches: float | int = 1.0,
    split: str = "valid",
    sos_token_id: int = 1002,
    sep_token_id: int = 1001,
    save_latents: bool = False,
) -> dict[str, Any]:
    """Score shifted loss targets and retain comparable per-sample measurements."""
    model.eval()
    records: list[dict[str, Any]] = []
    latents: list[np.ndarray] = []
    start_time = time.perf_counter()
    with torch.no_grad():
        for batch in limited(loader, limit_batches):
            batch = move_to_device(batch, device)
            with _autocast_context(device, precision):
                output = model(batch)
                loss_fn(
                    output, batch
                )  # Exercise the configured reconstruction objective.
            logits = output.reconstruction.token_logits.float()
            targets = (
                output.loss_targets
                if output.loss_targets is not None
                else batch["targets"]
            )
            mask = (
                output.loss_valid_mask
                if output.loss_valid_mask is not None
                else batch["valid_mask"]
            )
            mask = mask.bool() & (targets != sos_token_id)
            losses = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                reduction="none",
            ).reshape_as(targets)
            matches = logits.argmax(-1) == targets
            for row in range(targets.shape[0]):
                row_mask = mask[row]
                count = int(row_mask.sum())
                if not count:
                    raise ValueError("Benchmark sample has no scorable target tokens")
                loss_sum = float(losses[row][row_mask].sum())
                correct = int(matches[row][row_mask].sum())
                records.append(
                    {
                        **_annotate_record(batch, row, sep_token_id),
                        "tf_scored_tokens": count,
                        "tf_correct_tokens": correct,
                        "tf_token_loss_sum": loss_sum,
                        "tf_token_loss": loss_sum / count,
                        "tf_token_accuracy": correct / count,
                    }
                )
            if save_latents:
                latents.append(output.embedding.detach().float().cpu().numpy())
    if not records:
        raise ValueError("Teacher-forced benchmark evaluated zero samples")
    records_by_id(records)
    per_bucket = {
        bucket: _tf_summary([r for r in records if r["length_bucket"] == bucket])
        for bucket in sorted({r["length_bucket"] for r in records})
    }
    summary = _tf_summary(records)
    elapsed = time.perf_counter() - start_time
    result = {
        "mode": "teacher-forced",
        "total_samples": len(records),
        "wall_clock_seconds": elapsed,
        "seconds_per_sample": elapsed / len(records),
        "metrics": {
            f"{split}/token_loss": summary["tf_token_loss"],
            f"{split}/token_accuracy": summary["tf_token_accuracy"],
            f"{split}/token_perplexity": summary["tf_token_perplexity"],
            f"{split}/token_accuracy_mean": summary["tf_token_accuracy_mean"],
            f"{split}/valid_tokens": summary["valid_tokens"],
        },
        "per_bucket": per_bucket,
        "per_sample_records": records,
    }
    if save_latents:
        result["latent_vectors"] = np.concatenate(latents)
    return result


def _fr_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    result = {"count": len(records)}
    for name in (
        "token_accuracy",
        "exact_match",
        "eos_rate",
        "max_length_hit",
        "geometry_f1_2px",
    ):
        result["fr_" + name] = float(np.mean([r[name] for r in records]))
    result["fr_token_accuracy_mean"] = result["fr_token_accuracy"]
    result["fr_geometry_f1_2px_median"] = float(
        np.median([r["geometry_f1_2px"] for r in records])
    )
    return result


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
    sos_token_id: int = 1002,
    pad_token_id: int = 0,
    sep_token_id: int = 1001,
    seed: int = 42,
    comparison_ids: list[str] | None = None,
) -> tuple[dict[str, Any], list[Any]]:
    """Score complete generations and choose stable, stratified qualitative samples."""
    model.eval()
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    all_records: list[dict[str, Any]] = []
    candidates: dict[str, Any] = {}
    requested = set(comparison_ids) if comparison_ids is not None else None
    start_time = time.perf_counter()
    with torch.no_grad():
        for batch in limited(loader, limit_batches):
            batch = move_to_device(batch, device)
            with _autocast_context(device, precision):
                generation = raw_model.generate(
                    batch, max_length=max_generation_length, use_cache=True
                )
            batch_records = free_running_reconstruction_records(
                generation.tokens,
                generation.lengths,
                batch,
                codebook,
                eos_token_id=eos_token_id,
            )
            targets = batch["targets"].detach().cpu()
            predictions = generation.tokens.detach().cpu()
            for row, record in enumerate(batch_records):
                target = targets[row, : int(batch["lengths"][row])].numpy()
                pred = predictions[row, : int(generation.lengths[row])].numpy()
                aligned = np.full(len(target), -1, dtype=np.int64)
                overlap = min(len(target), len(pred))
                aligned[:overlap] = pred[:overlap]
                mask = (target != sos_token_id) & (target != pad_token_id)
                if not mask.any():
                    raise ValueError("Benchmark sample has no scorable target tokens")
                record.update(_annotate_record(batch, row, sep_token_id))
                record["token_accuracy"] = float(np.mean((target == aligned)[mask]))
                generation_limit = (
                    max_generation_length
                    if max_generation_length is not None
                    else int(batch["lengths"].max())
                )
                record["max_length_hit"] = float(
                    not record["eos_rate"] and len(pred) >= generation_limit
                )
                if save_predictions:
                    record["target_tokens"] = target.tolist()
                    record["predicted_tokens"] = pred.tolist()
                all_records.append(record)
            if num_plots:
                for example in collect_generated_reconstruction_examples(
                    generation,
                    batch,
                    max_examples=targets.shape[0],
                    codebook=codebook,
                ):
                    key = str(example.sample_id)
                    if requested is None or key in requested:
                        candidates[key] = example
                # Bound memory to at most num_plots candidates in each bucket.
                if requested is None:
                    for bucket in {
                        length_bucket(e.length) for e in candidates.values()
                    }:
                        keys = sorted(
                            (
                                k
                                for k, e in candidates.items()
                                if length_bucket(e.length) == bucket
                            ),
                            key=lambda k: sample_rank(k, seed),
                        )
                        for key in keys[num_plots:]:
                            del candidates[key]
    if not all_records:
        raise ValueError("Free-running benchmark evaluated zero samples")
    by_id = records_by_id(all_records)
    if comparison_ids is None:
        selection = select_comparison_samples(all_records, num_plots, seed)
    else:
        if len(requested) != len(comparison_ids):
            raise ValueError("Comparison manifest contains duplicate sample IDs")
        if not requested <= by_id.keys():
            raise ValueError(
                "Comparison manifest includes samples absent from this evaluation"
            )
        selection = [
            {k: by_id[key][k] for k in ("sample_id", "length_bucket", "target_length")}
            for key in comparison_ids
        ]
    examples = [candidates[r["sample_id"]] for r in selection]
    summary = _fr_summary(all_records)
    elapsed = time.perf_counter() - start_time
    metrics = {
        f"{split}/free_running/{k[3:]}": v
        for k, v in summary.items()
        if k.startswith("fr_")
    }
    return {
        "mode": "free-running",
        "total_samples": len(all_records),
        "wall_clock_seconds": elapsed,
        "seconds_per_sample": elapsed / len(all_records),
        "metrics": metrics,
        "per_bucket": {
            b: _fr_summary([r for r in all_records if r["length_bucket"] == b])
            for b in sorted({r["length_bucket"] for r in all_records})
        },
        "per_sample_records": all_records,
        "comparison_samples": selection,
    }, examples


def build_provenance(
    args: argparse.Namespace,
    config: dict[str, Any],
    codebook_path: Path | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Build a provenance block for full reproducibility."""
    effective_seed = (
        seed
        if seed is not None
        else (
            args.seed
            if getattr(args, "seed", None) is not None
            else int(config.get("project", {}).get("seed", 42))
        )
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

    # GPU info for the actual selected device.
    selected_device = resolve_device(getattr(args, "device", "auto"))
    if selected_device.type == "cuda":
        provenance["gpu"] = torch.cuda.get_device_name(selected_device)
        provenance["gpu_memory_total_mb"] = round(
            torch.cuda.get_device_properties(selected_device).total_mem / 1024 / 1024, 0
        )

    provenance["experiment"] = config.get("experiment", {}).get("name", args.experiment)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True, capture_output=True
    )
    provenance["git_commit"] = (
        revision.stdout.strip() if revision.returncode == 0 else None
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )
    provenance["git_dirty"] = (
        bool(status.stdout.strip()) if status.returncode == 0 else None
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
    parser.add_argument(
        "--codebook-path",
        type=Path,
        help="Override the codebook location without changing its token layout",
    )
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
    parser.add_argument(
        "--save-latents",
        action="store_true",
        help="Save per-sample encoder embeddings during teacher-forced evaluation.",
    )
    parser.add_argument(
        "--comparison-manifest",
        type=Path,
        help="Reuse a previously frozen stratified plot sample set.",
    )
    parser.add_argument(
        "--allow-random-init",
        action="store_true",
        help="Allow synthetic smoke tests without a checkpoint.",
    )
    parser.add_argument(
        "--allow-partial-checkpoint",
        action="store_true",
        help="Allow incomplete weights, recording all missing/unexpected keys in provenance.",
    )
    return parser.parse_args()


def _prepare_loader(config: dict[str, Any], split: str, seed: int):
    sequence = config["data"]["sequence"]
    sequence["truncate_long_sequences"] = False
    sequence["add_start_token"] = True
    sequence["add_end_token"] = True
    capacity = int(config["model"]["input"]["max_seq_len"])
    if int(sequence["max_length"]) > capacity:
        raise ValueError(
            "Dataset sequence limit exceeds model capacity; choose a compatible experiment/checkpoint"
        )
    datamodule = StrokeSequenceDataModule(
        config["data"], project_root=PROJECT_ROOT, seed=seed
    )
    # Load only the requested split; evaluation must not require training/test data.
    dataset = datamodule._make_dataset(split)
    dataset.transform = BenchmarkTokenTransform(
        dataset.transform,
        int(config["data"]["format"]["token_dictionary"]["pad_token_id"]),
    )
    for index in range(len(dataset)):
        try:
            dataset[index]
        except ValueError as exc:
            raise ValueError(
                f"Unsupported sample {dataset.index[index].sample_id}: {exc}. "
                "Benchmark truncation is disabled. Use a longer-sequence compatible experiment/checkpoint."
            ) from exc
    if split == "valid":
        datamodule.valid_dataset = dataset
        return datamodule.val_dataloader()
    datamodule.test_dataset = dataset
    return datamodule.test_dataloader()


class BenchmarkTokenTransform:
    """Remove storage padding before the existing non-truncating token transform."""

    def __init__(self, transform, pad_token_id):
        self.transform = transform
        self.pad_token_id = pad_token_id

    def __call__(self, sample):
        sample = dict(sample)
        tokens = np.asarray(sample["tokens"])
        sample["tokens"] = tokens[tokens != self.pad_token_id]
        return self.transform(sample)


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def main() -> int:
    args = parse_args()
    if args.num_plots < 0 or args.limit_batches <= 0:
        raise ValueError(
            "num_plots must be nonnegative and limit_batches must be positive"
        )
    if not args.checkpoint and not args.allow_random_init:
        raise ValueError(
            "Provide --checkpoint; --allow-random-init is for synthetic smoke tests only"
        )
    if args.save_latents and "teacher-forced" not in args.modes:
        raise ValueError("--save-latents requires teacher-forced mode")
    if args.comparison_manifest and (
        "free-running" not in args.modes or args.num_plots == 0
    ):
        raise ValueError(
            "--comparison-manifest requires free-running mode and num_plots > 0"
        )
    config = compose_training_config(args.config, experiment=args.experiment)
    if config["data"]["format"]["type"] not in {"tok_dict", "token", "tokens"}:
        raise ValueError("This benchmark requires tok-dict reconstruction data")
    if args.data_root:
        config["data"]["dataset"]["root"] = args.data_root
    if args.codebook_path:
        config["data"]["format"]["token_dictionary"]["codebook_dir"] = None
        config["data"]["format"]["token_dictionary"]["codebook_path"] = str(
            args.codebook_path.resolve()
        )
    if args.precision:
        config["trainer"]["runtime"]["precision"] = args.precision
    seed = (
        args.seed
        if args.seed is not None
        else int(config.get("project", {}).get("seed", 42))
    )
    config.setdefault("project", {})["seed"] = seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = resolve_device(args.device)
    _configure_torch_runtime(config, device)
    precision = _resolve_precision_runtime(config, device)
    config["trainer"]["runtime"]["precision"] = precision.effective
    capacity = int(config["model"]["input"]["max_seq_len"])
    if (
        args.max_generation_length is not None
        and not 2 <= args.max_generation_length <= capacity
    ):
        raise ValueError(
            f"max_generation_length must be between 2 and model capacity {capacity}"
        )
    run_name = args.run_name or f"benchmark_{args.split}"
    if Path(run_name).name != run_name or run_name in {".", ".."}:
        raise ValueError("run_name must be a filename, without directory components")
    output_dir = PROJECT_ROOT / args.output_dir
    report_path = output_dir / f"{run_name}_report.json"
    if output_dir.exists() and any(
        p.name.startswith(run_name + "_") for p in output_dir.iterdir()
    ):
        raise FileExistsError(
            f"Run artifacts already exist for {run_name}; choose a new --run-name to preserve the baseline"
        )
    loader = _prepare_loader(config, args.split, seed)
    raw_model = build_model(config["model"])
    checkpoint_load = None
    if args.checkpoint:
        checkpoint_load = _load_model_checkpoint(
            PROJECT_ROOT / args.checkpoint,
            raw_model,
            strict=not args.allow_partial_checkpoint,
        )
        print(f"[benchmark] loaded checkpoint: {args.checkpoint}")
    model = maybe_compile_model(raw_model.to(device), config["model"])
    loss_fn = build_loss(config["optimizer"]).to(device)
    codebook = None
    codebook_path = None
    if "free-running" in args.modes:
        codebook = _load_codebook(config)
        expected = int(config["data"]["format"]["token_dictionary"]["codebook_size"])
        if codebook.shape != (expected, 2) or not np.isfinite(codebook).all():
            raise ValueError(
                f"Codebook must contain {expected} finite two-dimensional vectors matching the model token layout"
            )
        directory = get_nested(config, "data.format.token_dictionary.codebook_dir")
        codebook_path = PROJECT_ROOT / (
            str(Path(directory) / "codebook.npy")
            if directory
            else get_nested(config, "data.format.token_dictionary.codebook_path")
        )
    provenance = build_provenance(args, config, codebook_path, seed=seed)
    provenance.update(
        {
            "length_definition": LENGTH_DEFINITION,
            "accuracy_definition": ACCURACY_DEFINITION,
            "effective_precision": precision.effective,
            "device": str(device),
            "checkpoint_load": checkpoint_load,
            "random_initialization": not bool(args.checkpoint),
        }
    )
    files = sorted(loader.dataset.files)
    provenance["dataset_files"] = {p.name: _file_sha256(p) for p in files}
    provenance["dataset_sha256"] = hashlib.sha256(
        json.dumps(provenance["dataset_files"], sort_keys=True).encode(),
    ).hexdigest()
    comparison_ids = None
    selection_seed = seed
    if args.comparison_manifest:
        manifest = json.loads(Path(args.comparison_manifest).read_text())
        if manifest.get("dataset_sha256") != provenance["dataset_sha256"]:
            raise ValueError(
                "Comparison manifest belongs to a different dataset snapshot"
            )
        if manifest.get("length_definition") != LENGTH_DEFINITION:
            raise ValueError("Comparison manifest uses a different length definition")
        comparison_ids = [row["sample_id"] for row in manifest["samples"]]
        selection_seed = int(manifest["seed"])
    report = {"schema_version": 2, "provenance": provenance}
    output_dir.mkdir(parents=True, exist_ok=True)
    tf_records = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    if "teacher-forced" in args.modes:
        print("[benchmark] Running teacher-forced evaluation...")
        tf_results = run_teacher_forced(
            model,
            loss_fn,
            loader,
            device,
            precision,
            limit_batches=args.limit_batches,
            split=args.split,
            sos_token_id=int(
                config["data"]["format"]["token_dictionary"]["sos_token_id"]
            ),
            sep_token_id=int(
                config["data"]["format"]["token_dictionary"]["sep_token_id"]
            ),
            save_latents=args.save_latents,
        )
        tf_records = tf_results.pop("per_sample_records")
        latent_vectors = tf_results.pop("latent_vectors", None)
        tf_path = output_dir / f"{run_name}_teacher_forced_per_sample.jsonl"
        _write_jsonl(tf_path, tf_records)
        tf_results["per_sample_path"] = tf_path.name
        report["teacher_forced"] = tf_results
        if latent_vectors is not None:
            latent_path = output_dir / f"{run_name}_latents.npz"
            np.savez_compressed(
                latent_path,
                embeddings=latent_vectors,
                sample_ids=np.array([r["sample_id"] for r in tf_records]),
                lengths=np.array([r["target_length"] for r in tf_records]),
                stroke_counts=np.array([r["stroke_count"] for r in tf_records]),
                tf_token_accuracy=np.array(
                    [r["tf_token_accuracy"] for r in tf_records]
                ),
            )
            report["teacher_forced"]["latent_vectors_path"] = latent_path.name
    examples = []
    if "free-running" in args.modes:
        print("[benchmark] Running free-running evaluation...")
        fr_results, examples = run_free_running(
            model,
            loader,
            device,
            precision,
            codebook,
            eos_token_id=int(
                config["data"]["format"]["token_dictionary"]["eos_token_id"]
            ),
            sos_token_id=int(
                config["data"]["format"]["token_dictionary"]["sos_token_id"]
            ),
            pad_token_id=int(
                config["data"]["format"]["token_dictionary"]["pad_token_id"]
            ),
            sep_token_id=int(
                config["data"]["format"]["token_dictionary"]["sep_token_id"]
            ),
            limit_batches=args.limit_batches,
            split=args.split,
            max_generation_length=args.max_generation_length,
            num_plots=args.num_plots,
            save_predictions=args.save_predictions,
            seed=seed,
            comparison_ids=comparison_ids,
        )
        per_sample = fr_results.pop("per_sample_records")
        if (
            tf_records
            and records_by_id(tf_records).keys() != records_by_id(per_sample).keys()
        ):
            raise ValueError("TF and FR modes evaluated different samples")
        per_sample_path = output_dir / f"{run_name}_per_sample.jsonl"
        _write_jsonl(per_sample_path, per_sample)
        fr_results["per_sample_path"] = per_sample_path.name
        selection = fr_results.pop("comparison_samples")
        selection_path = output_dir / f"{run_name}_comparison_manifest.json"
        selection_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "seed": selection_seed,
                    "length_definition": LENGTH_DEFINITION,
                    "dataset_sha256": provenance["dataset_sha256"],
                    "samples": selection,
                },
                indent=2,
            )
        )
        fr_results["comparison_manifest_path"] = selection_path.name
        report["free_running"] = fr_results
    if device.type == "cuda":
        provenance["peak_gpu_memory_mb"] = (
            torch.cuda.max_memory_allocated(device) / 1024 / 1024
        )
    if examples:
        from metrics.sketchformer.visualisation import save_reconstruction_examples

        save_reconstruction_examples(examples, output_dir / f"{run_name}_plots")
    report_path.write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(f"[benchmark] wrote report to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

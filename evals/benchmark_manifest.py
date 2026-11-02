"""Benchmark manifest generation for Sketchformer evaluation.

Generates a stratified JSONL manifest annotating every sample in a dataset
split with length bucket, stroke count, detail density, and separator ratio.
"""

from __future__ import annotations

import argparse
import json
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

LENGTH_BUCKETS = (
    (1, 512, "short"),
    (513, 1024, "medium"),
    (1025, 2048, "long"),
    (2049, 4096, "extended"),
)


def length_bucket(length: int) -> str:
    """Return the human-readable bucket name for a sequence length."""
    for low, high, name in LENGTH_BUCKETS:
        if low <= length <= high:
            return name
    return "over_4096"


def length_bucket_range(length: int) -> str:
    """Return the numeric range bucket string (e.g. '1_512')."""
    for low, high, _name in LENGTH_BUCKETS:
        if low <= length <= high:
            return f"{low}_{high}"
    return "over_4096"


def annotate_sample(
    sample_id: str,
    split: str,
    tokens: Any,
    *,
    sep_token_id: int = 1001,
    eos_token_id: int = 1003,
    sos_token_id: int = 1002,
    pad_token_id: int = 0,
) -> dict[str, Any]:
    """Compute stratification metadata for a single token sequence."""
    import numpy as np

    tokens = np.asarray(tokens, dtype=np.int64)

    # Strip special tokens for analysis
    special = {pad_token_id, sos_token_id, eos_token_id}
    content_mask = ~np.isin(tokens, list(special))
    content_tokens = tokens[content_mask]
    token_length = len(content_tokens)

    # Count separators and strokes (repository uses terminal separator after each stroke)
    sep_count = int(np.sum(content_tokens == sep_token_id))
    if len(content_tokens) == 0:
        stroke_count = 0
    elif content_tokens[-1] == sep_token_id:
        stroke_count = sep_count
    else:
        stroke_count = sep_count + 1

    # Motion tokens (excluding separators)
    motion_count = token_length - sep_count

    # Detail density = motion tokens per stroke
    detail_density = motion_count / max(stroke_count, 1)

    # Separator ratio = separator tokens / total content tokens
    separator_ratio = sep_count / max(token_length, 1)

    return {
        "sample_id": sample_id,
        "split": split,
        "token_length": token_length,
        "raw_length": len(tokens),
        "length_bucket": length_bucket(token_length),
        "length_bucket_range": length_bucket_range(token_length),
        "stroke_count": stroke_count,
        "sep_count": sep_count,
        "motion_token_count": motion_count,
        "detail_density": round(detail_density, 2),
        "separator_ratio": round(separator_ratio, 4),
    }


def build_manifest(
    dataset_root: Path,
    split: str,
    *,
    train_pattern: str = "train_*.npz",
    valid_file: str = "valid.npz",
    test_file: str = "test.npz",
    sep_token_id: int = 1001,
    eos_token_id: int = 1003,
    sos_token_id: int = 1002,
    pad_token_id: int = 0,
) -> list[dict[str, Any]]:
    """Build a manifest of annotated samples from npz files."""
    import numpy as np

    dataset_root = Path(dataset_root)
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    if split == "train":
        files = sorted(dataset_root.glob(train_pattern))
    elif split == "valid":
        files = [dataset_root / valid_file]
    elif split == "test":
        files = [dataset_root / test_file]
    else:
        raise ValueError(f"Unknown split: {split}")

    files = [f for f in files if f.exists()]
    if not files:
        raise FileNotFoundError(f"No {split} files found under {dataset_root}")

    records: list[dict[str, Any]] = []
    global_idx = 0

    for file_path in files:
        data = np.load(file_path, allow_pickle=True)
        sequences = data["x"]
        for local_idx in range(len(sequences)):
            sample_id = f"{split}_{global_idx:06d}"
            record = annotate_sample(
                sample_id=sample_id,
                split=split,
                tokens=sequences[local_idx],
                sep_token_id=sep_token_id,
                eos_token_id=eos_token_id,
                sos_token_id=sos_token_id,
                pad_token_id=pad_token_id,
            )
            record["source_file"] = str(file_path.relative_to(dataset_root))
            record["source_index"] = local_idx
            records.append(record)
            global_idx += 1

    return records


def summarize_manifest(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute dataset-level statistics from a manifest."""
    if not records:
        return {"total_samples": 0}

    lengths = [r["token_length"] for r in records]
    strokes = [r["stroke_count"] for r in records]
    densities = [r["detail_density"] for r in records]
    sep_ratios = [r["separator_ratio"] for r in records]

    import numpy as np

    bucket_counts: dict[str, int] = {}
    for r in records:
        bucket = r["length_bucket"]
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1

    return {
        "total_samples": len(records),
        "length_buckets": bucket_counts,
        "token_length": {
            "min": int(np.min(lengths)),
            "max": int(np.max(lengths)),
            "mean": round(float(np.mean(lengths)), 1),
            "median": round(float(np.median(lengths)), 1),
            "std": round(float(np.std(lengths)), 1),
        },
        "stroke_count": {
            "min": int(np.min(strokes)),
            "max": int(np.max(strokes)),
            "mean": round(float(np.mean(strokes)), 1),
            "median": round(float(np.median(strokes)), 1),
        },
        "detail_density": {
            "min": round(float(np.min(densities)), 2),
            "max": round(float(np.max(densities)), 2),
            "mean": round(float(np.mean(densities)), 2),
        },
        "separator_ratio": {
            "min": round(float(np.min(sep_ratios)), 4),
            "max": round(float(np.max(sep_ratios)), 4),
            "mean": round(float(np.mean(sep_ratios)), 4),
        },
    }


def write_manifest(records: list[dict[str, Any]], output_path: Path) -> None:
    """Write manifest records as JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a stratified benchmark manifest for Sketchformer evaluation."
    )
    parser.add_argument(
        "--dataset-root",
        default="data/processed/sketchformer-ready-data/tok-dict",
        help="Root directory containing train/valid/test npz files.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["valid", "test"],
        choices=["train", "valid", "test"],
        help="Which splits to include in the manifest.",
    )
    parser.add_argument(
        "--output",
        default="evals/benchmark_manifest.jsonl",
        help="Output path for the JSONL manifest.",
    )
    parser.add_argument(
        "--summary-output",
        default="evals/benchmark_manifest_summary.json",
        help="Output path for the summary statistics JSON.",
    )
    parser.add_argument("--sep-token-id", type=int, default=1001)
    parser.add_argument("--eos-token-id", type=int, default=1003)
    parser.add_argument("--sos-token-id", type=int, default=1002)
    parser.add_argument("--pad-token-id", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_root = PROJECT_ROOT / args.dataset_root

    all_records: list[dict[str, Any]] = []
    for split in args.splits:
        print(f"[manifest] processing split={split} from {dataset_root}")
        records = build_manifest(
            dataset_root,
            split,
            sep_token_id=args.sep_token_id,
            eos_token_id=args.eos_token_id,
            sos_token_id=args.sos_token_id,
            pad_token_id=args.pad_token_id,
        )
        all_records.extend(records)
        print(f"  → {len(records)} samples")

    manifest_path = PROJECT_ROOT / args.output
    write_manifest(all_records, manifest_path)
    print(f"[manifest] wrote {len(all_records)} records to {manifest_path}")

    summary = summarize_manifest(all_records)
    summary_path = PROJECT_ROOT / args.summary_output
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[manifest] wrote summary to {summary_path}")
    print(json.dumps(summary, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


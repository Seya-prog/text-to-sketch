"""Failure analysis tools for Sketchformer benchmark evaluation.

Analyzes per-sample records from benchmark_eval.py to diagnose:
1. Teacher-forced vs free-running gap (gap)
2. Long-sequence error accumulation (position-curves)
3. Separator token prediction errors & cascade analysis (separators & separator-confusion)
4. Codebook suitability & quantization error (codebook & quantization)
5. TensorFlow-to-PyTorch conversion fidelity (conversion)
6. Latent representation capacity & bottleneck analysis (latent-capacity)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _add_project_to_path() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists() and (parent / "configs").exists():
            sys.path.insert(0, str(parent))
            return parent
    raise RuntimeError("Could not find project root directory.")


PROJECT_ROOT = _add_project_to_path()

from evals.conversion_fidelity import evaluate_logits_files
from evals.benchmark_common import length_bucket, normalize_bucket, records_by_id


# ---------------------------------------------------------------------------
# 1. Teacher-forced vs free-running gap
# ---------------------------------------------------------------------------


def analyze_tf_fr_gap(
    tf_report_path: Path,
    fr_per_sample_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Join identical samples; compare macro accuracies without a global fallback."""
    tf_report = json.loads(tf_report_path.read_text(encoding="utf-8"))
    tf_data = tf_report.get("teacher_forced", {})
    tf_records = tf_data.get("per_sample_records", [])
    if "per_sample_path" in tf_data:
        tf_records = _load_jsonl(tf_report_path.parent / tf_data["per_sample_path"])
    fr_records = _load_jsonl(fr_per_sample_path)
    if not fr_records:
        raise ValueError("No free-running records found")
    joined = []
    tf_per_bucket = {
        normalize_bucket(b): value for b, value in tf_data.get("per_bucket", {}).items()
    }
    if tf_records:
        tf_by_id, fr_by_id = records_by_id(tf_records), records_by_id(fr_records)
        if tf_by_id.keys() != fr_by_id.keys():
            raise ValueError(
                "TF and FR sample IDs differ; compare the same split and batch limit"
            )
        for sample_id in sorted(tf_by_id):
            tf, fr = tf_by_id[sample_id], fr_by_id[sample_id]
            for key in ("target_length", "target_sha256", "accuracy_definition"):
                if key in tf and tf[key] != fr.get(key):
                    raise ValueError(f"TF/FR {key} mismatch for sample {sample_id}")
            if normalize_bucket(tf["length_bucket"]) != normalize_bucket(
                fr["length_bucket"]
            ):
                raise ValueError(f"TF/FR bucket mismatch for sample {sample_id}")
            joined.append(
                {
                    **fr,
                    "length_bucket": normalize_bucket(fr["length_bucket"]),
                    "tf_token_accuracy": float(tf["tf_token_accuracy"]),
                    "tf_token_loss": tf.get("tf_token_loss"),
                    "accuracy_gap_tf_minus_fr": float(tf["tf_token_accuracy"])
                    - float(fr["token_accuracy"]),
                }
            )
    elif not tf_per_bucket:
        raise ValueError(
            "No bucket-aligned or per-sample TF scores; rerun the corrected benchmark"
        )
    buckets = {}
    for record in joined or fr_records:
        bucket = normalize_bucket(record["length_bucket"])
        buckets.setdefault(bucket, []).append(record)
    analysis = {
        "total_samples": len(fr_records),
        "teacher_forced_metrics": tf_data.get("metrics", {}),
        "teacher_forced_per_bucket": tf_per_bucket,
        "per_sample_gap_available": bool(joined),
        "per_bucket": {},
    }
    for bucket, records in sorted(buckets.items()):
        fr_acc = float(np.mean([r["token_accuracy"] for r in records]))
        if joined:
            tf_acc = float(np.mean([r["tf_token_accuracy"] for r in records]))
        else:
            tf_acc = tf_per_bucket.get(bucket, {}).get("tf_token_accuracy_mean")
            if tf_acc is None:
                raise ValueError(
                    f"Missing TF scores for {bucket}; global accuracy cannot replace them"
                )
        f1 = [r["geometry_f1_2px"] for r in records if "geometry_f1_2px" in r]
        eos = [r["eos_rate"] for r in records if "eos_rate" in r]
        analysis["per_bucket"][bucket] = {
            "count": len(records),
            "tf_token_accuracy_mean": tf_acc,
            "fr_token_accuracy_mean": fr_acc,
            "accuracy_gap_tf_minus_fr": float(tf_acc) - fr_acc,
            "fr_token_accuracy_median": float(
                np.median([r["token_accuracy"] for r in records])
            ),
            "fr_geometry_f1_mean": float(np.mean(f1)) if f1 else None,
            "fr_geometry_f1_median": float(np.median(f1)) if f1 else None,
            "fr_eos_rate": float(np.mean(eos)) if eos else None,
        }
    if joined:
        analysis["per_sample_records"] = joined
        analysis["worst_20_samples"] = sorted(
            joined,
            key=lambda r: (-r["accuracy_gap_tf_minus_fr"], str(r["sample_id"])),
        )[:20]
    output_path = Path(output_dir) / "tf_vs_fr_gap_analysis.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(analysis, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(f"[analysis] wrote TF vs FR gap analysis to {output_path}")
    return analysis


# ---------------------------------------------------------------------------
# 2. Position-accuracy curves (Long-Sequence Error Accumulation)
# ---------------------------------------------------------------------------


def analyze_position_accuracy(
    records_or_path: list[dict[str, Any]] | Path,
    output_dir: Path | None = None,
    *,
    onset_threshold: float = 0.80,
    collapse_threshold: float = 0.50,
    sep_token_id: int = 1001,
) -> dict[str, Any]:
    """Compute token accuracy at each sequence position across length buckets.

    Identifies:
    - Onset position: first position where accuracy drops below onset_threshold
    - Collapse position: first position where accuracy drops below collapse_threshold
    - Recovery events: positions where accuracy jumps following separator tokens
    """
    if isinstance(records_or_path, (str, Path)):
        records = _load_jsonl(Path(records_or_path))
    else:
        records = records_or_path

    if not records:
        return {"error": "No prediction records provided"}

    if any("target_tokens" not in r or "predicted_tokens" not in r for r in records):
        raise ValueError(
            "Prediction arrays are required; run benchmark_eval.py with --save-predictions"
        )

    # Separate records by bucket
    buckets: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        target = r.get("target_tokens", [])
        bucket = r.get("length_bucket")
        if not bucket:
            bucket = length_bucket(len(target))
        else:
            bucket = normalize_bucket(bucket)
        buckets.setdefault(bucket, []).append(r)

    def _compute_curve(bucket_records: list[dict[str, Any]]) -> dict[str, Any]:
        targets = [
            np.asarray(r["target_tokens"], dtype=np.int64)
            for r in bucket_records
            if "target_tokens" in r
        ]
        preds = [
            np.asarray(r.get("predicted_tokens", []), dtype=np.int64)
            for r in bucket_records
            if "target_tokens" in r
        ]
        if not targets:
            return {}

        max_len = max(len(t) for t in targets)
        accuracies: list[float] = []
        active_counts: list[int] = []

        for pos in range(max_len):
            valid_targets = []
            valid_preds = []
            for t, p in zip(targets, preds):
                if pos < len(t):
                    valid_targets.append(t[pos])
                    valid_preds.append(p[pos] if pos < len(p) else -1)

            count = len(valid_targets)
            active_counts.append(count)
            if count > 0:
                acc = float(np.mean(np.array(valid_targets) == np.array(valid_preds)))
                accuracies.append(round(acc, 4))
            else:
                accuracies.append(0.0)

        # Detect onset and collapse
        onset_pos = None
        collapse_pos = None
        for pos, acc in enumerate(accuracies):
            if onset_pos is None and acc < onset_threshold:
                onset_pos = pos
            if collapse_pos is None and acc < collapse_threshold:
                collapse_pos = pos

        # Detect recovery events (> +5% jump)
        recoveries = []
        for pos in range(1, len(accuracies)):
            jump = accuracies[pos] - accuracies[pos - 1]
            if jump >= 0.05:
                # Check if previous position had separators
                has_sep = any(
                    pos - 1 < len(t) and t[pos - 1] == sep_token_id for t in targets
                )
                recoveries.append(
                    {
                        "position": pos,
                        "accuracy_jump": round(jump, 4),
                        "after_separator": has_sep,
                    }
                )

        return {
            "num_samples": len(bucket_records),
            "max_sequence_length": max_len,
            "onset_position": onset_pos,
            "collapse_position": collapse_pos,
            "recovery_event_count": len(recoveries),
            "recovery_events": recoveries[:10],
            "accuracy_curve": accuracies,
            "active_sample_counts": active_counts,
        }

    analysis: dict[str, Any] = {
        "total_samples": len(records),
        "overall": _compute_curve(records),
        "by_bucket": {b: _compute_curve(recs) for b, recs in sorted(buckets.items())},
    }

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "position_accuracy_curves.json"
        out_path.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
        print(f"[analysis] wrote position accuracy curves to {out_path}")

    return analysis


# ---------------------------------------------------------------------------
# 3. Separator token prediction & cascade analysis
# ---------------------------------------------------------------------------


def analyze_separator_confusion(
    records_or_path: list[dict[str, Any]] | Path,
    output_dir: Path | None = None,
    *,
    sep_token_id: int = 1001,
    cascade_window: int = 10,
) -> dict[str, Any]:
    """Compute separator confusion matrix and measure stroke cascade effect.

    Measures:
    - TP, FP, FN, TN for separator prediction
    - Precision, Recall, F1
    - Cascade effect: Accuracy in the 10 tokens following correct vs missed separator
    """
    if isinstance(records_or_path, (str, Path)):
        records = _load_jsonl(Path(records_or_path))
    else:
        records = records_or_path

    if not records:
        return {"error": "No prediction records provided"}

    if cascade_window <= 0:
        raise ValueError("cascade_window must be positive")
    if any("target_tokens" not in r or "predicted_tokens" not in r for r in records):
        raise ValueError(
            "Prediction arrays are required; run benchmark_eval.py with --save-predictions"
        )

    tp = 0
    fp = 0
    fn = 0
    tn = 0

    acc_after_correct_sep: list[float] = []
    acc_after_missed_sep: list[float] = []

    for r in records:
        targets = np.asarray(r.get("target_tokens", []), dtype=np.int64)
        preds = np.asarray(r.get("predicted_tokens", []), dtype=np.int64)
        if len(targets) == 0:
            continue

        for pos in range(len(targets)):
            target_is_sep = bool(targets[pos] == sep_token_id)
            pred_is_sep = bool(pos < len(preds) and preds[pos] == sep_token_id)

            if target_is_sep and pred_is_sep:
                tp += 1
                # Cascade window
                window_end = min(pos + 1 + cascade_window, len(targets))
                if window_end > pos + 1:
                    w_targets = targets[pos + 1 : window_end]
                    w_preds = preds[pos + 1 : min(window_end, len(preds))]
                    w_overlap = min(len(w_targets), len(w_preds))
                    matches = np.sum(w_targets[:w_overlap] == w_preds[:w_overlap])
                    acc_after_correct_sep.append(float(matches / len(w_targets)))

            elif not target_is_sep and pred_is_sep:
                fp += 1
            elif target_is_sep and not pred_is_sep:
                fn += 1
                # Cascade window for missed separator
                window_end = min(pos + 1 + cascade_window, len(targets))
                if window_end > pos + 1:
                    w_targets = targets[pos + 1 : window_end]
                    w_preds = preds[pos + 1 : min(window_end, len(preds))]
                    w_overlap = min(len(w_targets), len(w_preds))
                    matches = np.sum(w_targets[:w_overlap] == w_preds[:w_overlap])
                    acc_after_missed_sep.append(float(matches / len(w_targets)))
            else:
                tn += 1

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = (
        float(2 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )
    false_sep_rate = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    missed_sep_rate = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    mean_acc_correct = (
        float(np.mean(acc_after_correct_sep)) if acc_after_correct_sep else 0.0
    )
    mean_acc_missed = (
        float(np.mean(acc_after_missed_sep)) if acc_after_missed_sep else 0.0
    )
    has_comparison = bool(acc_after_correct_sep and acc_after_missed_sep)
    cascade_penalty = mean_acc_correct - mean_acc_missed if has_comparison else None

    analysis = {
        "confusion_matrix": {
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn,
        },
        "metrics": {
            "separator_precision": round(precision, 4),
            "separator_recall": round(recall, 4),
            "separator_f1": round(f1, 4),
            "false_separator_rate": round(false_sep_rate, 4),
            "missed_separator_rate": round(missed_sep_rate, 4),
        },
        "cascade_analysis": {
            "cascade_window_tokens": cascade_window,
            "accuracy_after_correct_separator": round(mean_acc_correct, 4),
            "accuracy_after_missed_separator": round(mean_acc_missed, 4),
            "correct_separator_windows": len(acc_after_correct_sep),
            "missed_separator_windows": len(acc_after_missed_sep),
            "cascade_penalty": round(cascade_penalty, 4) if has_comparison else None,
            "cascade_effect_detected": bool(cascade_penalty > 0.05)
            if has_comparison
            else None,
        },
    }

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "separator_confusion_analysis.json"
        out_path.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
        print(f"[analysis] wrote separator confusion analysis to {out_path}")

    return analysis


# ---------------------------------------------------------------------------
# 4. Separator dataset distribution
# ---------------------------------------------------------------------------


def analyze_separators(
    dataset_root: Path,
    split: str,
    *,
    sep_token_id: int = 1001,
    eos_token_id: int = 1003,
    sos_token_id: int = 1002,
    pad_token_id: int = 0,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Analyze separator token distribution in the dataset."""
    from evals.benchmark_manifest import build_manifest

    records = build_manifest(
        dataset_root,
        split,
        sep_token_id=sep_token_id,
        eos_token_id=eos_token_id,
        sos_token_id=sos_token_id,
        pad_token_id=pad_token_id,
    )

    sep_counts = [r["sep_count"] for r in records]
    sep_ratios = [r["separator_ratio"] for r in records]
    stroke_counts = [r["stroke_count"] for r in records]
    lengths = [r["token_length"] for r in records]

    buckets: dict[str, list[float]] = {}
    for r in records:
        bucket = r["length_bucket"]
        buckets.setdefault(bucket, []).append(r["separator_ratio"])

    analysis = {
        "total_samples": len(records),
        "separator_count": {
            "min": int(np.min(sep_counts)) if sep_counts else 0,
            "max": int(np.max(sep_counts)) if sep_counts else 0,
            "mean": round(float(np.mean(sep_counts)), 1) if sep_counts else 0.0,
            "median": round(float(np.median(sep_counts)), 1) if sep_counts else 0.0,
        },
        "separator_ratio": {
            "min": round(float(np.min(sep_ratios)), 4) if sep_ratios else 0.0,
            "max": round(float(np.max(sep_ratios)), 4) if sep_ratios else 0.0,
            "mean": round(float(np.mean(sep_ratios)), 4) if sep_ratios else 0.0,
        },
        "stroke_count": {
            "min": int(np.min(stroke_counts)) if stroke_counts else 0,
            "max": int(np.max(stroke_counts)) if stroke_counts else 0,
            "mean": round(float(np.mean(stroke_counts)), 1) if stroke_counts else 0.0,
        },
        "sep_ratio_by_bucket": {
            bucket: {
                "count": len(ratios),
                "mean": round(float(np.mean(ratios)), 4),
                "std": round(float(np.std(ratios)), 4),
            }
            for bucket, ratios in sorted(buckets.items())
        },
        "correlation_length_vs_sep_count": round(
            float(np.corrcoef(lengths, sep_counts)[0, 1]), 4
        )
        if len(lengths) > 1
        else 0.0,
    }

    if output_dir:
        output_dir = Path(output_dir)
        output_path = output_dir / "separator_analysis.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
        print(f"[analysis] wrote separator analysis to {output_path}")

    return analysis


# ---------------------------------------------------------------------------
# 5. Codebook coverage & quantization error
# ---------------------------------------------------------------------------


def analyze_codebook(
    codebook_path: Path,
    dataset_root: Path,
    split: str,
    *,
    sep_token_id: int = 1001,
    eos_token_id: int = 1003,
    sos_token_id: int = 1002,
    pad_token_id: int = 0,
    motion_token_offset: int = 1,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Analyze codebook coverage and usage for the anime dataset."""
    codebook = np.load(codebook_path)
    codebook_size = len(codebook)

    if split == "train":
        files = sorted(dataset_root.glob("train_*.npz"))
    elif split == "valid":
        files = [dataset_root / "valid.npz"]
    elif split == "test":
        files = [dataset_root / "test.npz"]
    else:
        raise ValueError(f"Unknown split: {split}")

    files = [f for f in files if f.exists()]
    if not files:
        raise FileNotFoundError(f"No {split} files found under {dataset_root}")

    special_tokens = {pad_token_id, sep_token_id, sos_token_id, eos_token_id}
    usage_counts = np.zeros(codebook_size, dtype=np.int64)
    total_motion_tokens = 0

    for file_path in files:
        data = np.load(file_path, allow_pickle=True)
        sequences = data["x"]
        for seq in sequences:
            tokens = np.asarray(seq, dtype=np.int64)
            motion_mask = np.ones(len(tokens), dtype=bool)
            for st in special_tokens:
                motion_mask &= tokens != st
            motion_tokens = tokens[motion_mask]
            adjusted = motion_tokens - motion_token_offset
            valid = (adjusted >= 0) & (adjusted < codebook_size)
            for idx in adjusted[valid]:
                usage_counts[idx] += 1
            total_motion_tokens += int(valid.sum())

    used_entries = int(np.sum(usage_counts > 0))
    dead_entries = codebook_size - used_entries

    sorted_indices = np.argsort(usage_counts)[::-1]
    top_10 = [
        {"entry": int(idx), "count": int(usage_counts[idx])}
        for idx in sorted_indices[:10]
    ]
    bottom_10_used = [
        {"entry": int(idx), "count": int(usage_counts[idx])}
        for idx in sorted_indices[::-1]
        if usage_counts[idx] > 0
    ][:10]

    nonzero_counts = usage_counts[usage_counts > 0]

    analysis = {
        "codebook_size": codebook_size,
        "total_motion_tokens": total_motion_tokens,
        "used_entries": used_entries,
        "dead_entries": dead_entries,
        "coverage_ratio": round(used_entries / codebook_size, 4),
        "usage_distribution": {
            "min": int(np.min(nonzero_counts)) if len(nonzero_counts) > 0 else 0,
            "max": int(np.max(nonzero_counts)) if len(nonzero_counts) > 0 else 0,
            "mean": round(float(np.mean(nonzero_counts)), 1)
            if len(nonzero_counts) > 0
            else 0.0,
            "median": round(float(np.median(nonzero_counts)), 1)
            if len(nonzero_counts) > 0
            else 0.0,
            "std": round(float(np.std(nonzero_counts)), 1)
            if len(nonzero_counts) > 0
            else 0.0,
        },
        "top_10_entries": top_10,
        "bottom_10_used_entries": bottom_10_used,
        "codebook_vector_stats": {
            "min_norm": round(float(np.min(np.linalg.norm(codebook, axis=1))), 4),
            "max_norm": round(float(np.max(np.linalg.norm(codebook, axis=1))), 4),
            "mean_norm": round(float(np.mean(np.linalg.norm(codebook, axis=1))), 4),
        },
    }

    if output_dir:
        output_dir = Path(output_dir)
        output_path = output_dir / "codebook_analysis.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
        print(f"[analysis] wrote codebook analysis to {output_path}")

        hist_path = output_dir / "codebook_usage_counts.npy"
        np.save(hist_path, usage_counts)
        print(f"[analysis] wrote usage counts to {hist_path}")

    return analysis


def analyze_quantization_error(
    stroke5_arrays: list[np.ndarray],
    codebook: np.ndarray,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Measure per-point quantization error and cumulative endpoint drift across strokes."""
    from scipy.spatial import cKDTree

    codebook = np.asarray(codebook, dtype=np.float32)
    tree = cKDTree(codebook)

    point_errors: list[float] = []
    stroke_drifts: list[float] = []

    for s5 in stroke5_arrays:
        s5 = np.asarray(s5, dtype=np.float32)
        if s5.ndim != 2 or s5.shape[1] != 5:
            continue

        current_stroke_points: list[np.ndarray] = []
        for row in s5:
            if row[4] >= 0.5:  # EOS sentinel
                break

            pt = row[:2]
            dist, idx = tree.query(pt)
            point_errors.append(float(dist))
            current_stroke_points.append(pt)

            if row[3] >= 0.5:  # Pen lift
                if current_stroke_points:
                    original_end = np.sum(current_stroke_points, axis=0)
                    _, q_indices = tree.query(current_stroke_points)
                    quantized_end = np.sum(codebook[q_indices], axis=0)
                    drift = float(np.linalg.norm(original_end - quantized_end))
                    stroke_drifts.append(drift)
                current_stroke_points = []

    analysis = {
        "total_evaluated_points": len(point_errors),
        "total_evaluated_strokes": len(stroke_drifts),
        "point_quantization_error": {
            "mean": round(float(np.mean(point_errors)), 4) if point_errors else 0.0,
            "median": round(float(np.median(point_errors)), 4) if point_errors else 0.0,
            "max": round(float(np.max(point_errors)), 4) if point_errors else 0.0,
            "std": round(float(np.std(point_errors)), 4) if point_errors else 0.0,
        },
        "stroke_endpoint_drift": {
            "mean": round(float(np.mean(stroke_drifts)), 4) if stroke_drifts else 0.0,
            "median": round(float(np.median(stroke_drifts)), 4)
            if stroke_drifts
            else 0.0,
            "max": round(float(np.max(stroke_drifts)), 4) if stroke_drifts else 0.0,
            "std": round(float(np.std(stroke_drifts)), 4) if stroke_drifts else 0.0,
        },
    }

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "quantization_analysis.json"
        out_path.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
        print(f"[analysis] wrote quantization error analysis to {out_path}")

    return analysis


# ---------------------------------------------------------------------------
# 6. Latent representation capacity (128-dim Bottleneck Analysis)
# ---------------------------------------------------------------------------


def analyze_latent_capacity(
    latent_vectors: np.ndarray,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Analyze latent space capacity and effective dimensionality.

    Computes SVD singular values and participation ratio (PR) to determine
    whether the 128-dim bottleneck is under-utilized or saturating:
        PR = (sum(s_i^2))^2 / sum(s_i^4), using covariance eigenvalues.
    """
    Z = np.asarray(latent_vectors, dtype=np.float32)
    if Z.ndim != 2 or min(Z.shape, default=0) == 0 or not np.isfinite(Z).all():
        raise ValueError(f"Expected 2D latent matrix (N, D), got {Z.shape}")

    num_samples, dim = Z.shape
    centered = Z - np.mean(Z, axis=0, keepdims=True)

    # Singular Value Decomposition
    _, s, _ = np.linalg.svd(centered, full_matrices=False)
    variance_explained = (s**2) / np.sum(s**2) if np.sum(s**2) > 0 else np.zeros_like(s)

    # Participation Ratio (effective dimension)
    eigenvalues = s.astype(np.float64) ** 2
    sum_eigenvalues2 = float(np.sum(eigenvalues**2))
    participation_ratio = (
        float(np.sum(eigenvalues) ** 2 / sum_eigenvalues2)
        if sum_eigenvalues2 > 0
        else 0.0
    )

    cumulative_var = np.cumsum(variance_explained)
    top_components = {
        f"top_{k}": round(float(cumulative_var[min(k - 1, len(cumulative_var) - 1)]), 4)
        for k in (1, 5, 10, 20, 50, 100)
        if k <= dim
    }

    analysis = {
        "num_samples": num_samples,
        "latent_dimension": dim,
        "effective_dimensionality_pr": round(float(participation_ratio), 2),
        "effective_dim_ratio": round(float(participation_ratio / dim), 4),
        "cumulative_variance_explained": top_components,
        "singular_values_top10": [round(float(v), 4) for v in s[:10]],
    }

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "latent_capacity_analysis.json"
        out_path.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
        print(f"[analysis] wrote latent capacity analysis to {out_path}")

    return analysis


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file into a list of dicts."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run failure analysis on benchmark evaluation results."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # 1. TF vs FR gap
    gap = sub.add_parser("gap", help="Analyze teacher-forced vs free-running gap")
    gap.add_argument("--tf-report", required=True, help="Path to benchmark report JSON")
    gap.add_argument("--fr-per-sample", required=True, help="Path to per-sample JSONL")
    gap.add_argument("--output-dir", default="evals/results/analysis")

    # 2. Position accuracy curves
    pos = sub.add_parser(
        "position-curves", help="Compute position-accuracy curves and collapse point"
    )
    pos.add_argument(
        "--predictions",
        required=True,
        help="Path to JSONL with target_tokens and predicted_tokens",
    )
    pos.add_argument("--output-dir", default="evals/results/analysis")

    # 3. Separator confusion
    sep_conf = sub.add_parser(
        "separator-confusion", help="Compute separator confusion and cascade penalty"
    )
    sep_conf.add_argument(
        "--predictions",
        required=True,
        help="Path to JSONL with target_tokens and predicted_tokens",
    )
    sep_conf.add_argument("--output-dir", default="evals/results/analysis")

    # 4. Separator dataset distribution
    sep = sub.add_parser("separators", help="Analyze separator token distribution")
    sep.add_argument(
        "--dataset-root", default="data/processed/sketchformer-ready-data/tok-dict"
    )
    sep.add_argument("--split", default="valid")
    sep.add_argument("--output-dir", default="evals/results/analysis")

    # 5. Codebook analysis
    cb = sub.add_parser("codebook", help="Analyze codebook coverage and usage")
    cb.add_argument(
        "--codebook-path", default="data/processed/sketch_token/codebook.npy"
    )
    cb.add_argument(
        "--dataset-root", default="data/processed/sketchformer-ready-data/tok-dict"
    )
    cb.add_argument("--split", default="valid")
    cb.add_argument("--output-dir", default="evals/results/analysis")

    # 6. Quantization error
    quant = sub.add_parser(
        "quantization", help="Measure codebook point error and endpoint drift"
    )
    quant.add_argument(
        "--stroke5-dir", required=True, help="Directory containing Stroke-5 .npz files"
    )
    quant.add_argument(
        "--codebook-path", default="data/processed/sketch_token/codebook.npy"
    )
    quant.add_argument("--output-dir", default="evals/results/analysis")

    # 7. Latent capacity
    lat = sub.add_parser(
        "latent-capacity", help="Analyze 128-dim latent bottleneck effective dimension"
    )
    lat.add_argument(
        "--latent-vectors",
        required=True,
        help="Path to .npy or .npz containing latent embeddings",
    )
    lat.add_argument("--output-dir", default="evals/results/analysis")

    # 8. Conversion fidelity
    conv = sub.add_parser("conversion", help="Verify TF-to-PyTorch logit parity")
    conv.add_argument(
        "--candidate-logits",
        required=True,
        help="Path to PyTorch candidate logits (.npy or .npz)",
    )
    conv.add_argument(
        "--reference-logits",
        required=True,
        help="Path to TF reference logits (.npy or .npz)",
    )
    conv.add_argument("--output-dir", default="evals/results/analysis")
    conv.add_argument("--min-cosine", type=float, default=0.999)
    conv.add_argument("--min-agreement", type=float, default=0.99)

    # 9. Plot figures from analysis
    plot_cmd = sub.add_parser(
        "plot", help="Generate publication figures from analysis results"
    )
    plot_cmd.add_argument("--analysis-dir", default="evals/results/analysis")
    plot_cmd.add_argument("--output-dir", default="evals/results/analysis/figures")

    # 10. Run all analyses
    all_cmd = sub.add_parser("all", help="Run all available analyses")
    all_cmd.add_argument(
        "--dataset-root", default="data/processed/sketchformer-ready-data/tok-dict"
    )
    all_cmd.add_argument(
        "--codebook-path", default="data/processed/sketch_token/codebook.npy"
    )
    all_cmd.add_argument("--split", default="valid")
    all_cmd.add_argument("--tf-report", default=None)
    all_cmd.add_argument("--fr-per-sample", default=None)
    all_cmd.add_argument("--predictions", default=None)
    all_cmd.add_argument("--stroke5-dir", default=None)
    all_cmd.add_argument("--latent-vectors", default=None)
    all_cmd.add_argument("--candidate-logits", default=None)
    all_cmd.add_argument("--reference-logits", default=None)
    all_cmd.add_argument("--output-dir", default="evals/results/analysis")
    all_cmd.add_argument(
        "--plots", action="store_true", help="Generate figures after analysis completes"
    )

    for command in (conv, all_cmd):
        validity = command.add_mutually_exclusive_group()
        validity.add_argument("--valid-mask", type=Path, default=None)
        validity.add_argument("--lengths", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = PROJECT_ROOT / args.output_dir
    exit_code = 0

    if args.command == "gap":
        analyze_tf_fr_gap(Path(args.tf_report), Path(args.fr_per_sample), out_dir)

    elif args.command == "position-curves":
        analyze_position_accuracy(Path(args.predictions), output_dir=out_dir)

    elif args.command == "separator-confusion":
        analyze_separator_confusion(Path(args.predictions), output_dir=out_dir)

    elif args.command == "separators":
        analyze_separators(
            PROJECT_ROOT / args.dataset_root, args.split, output_dir=out_dir
        )

    elif args.command == "codebook":
        analyze_codebook(
            PROJECT_ROOT / args.codebook_path,
            PROJECT_ROOT / args.dataset_root,
            args.split,
            output_dir=out_dir,
        )

    elif args.command == "quantization":
        stroke5_dir = Path(args.stroke5_dir)
        arrays = [
            np.asarray(
                np.load(p, allow_pickle=True)[np.load(p, allow_pickle=True).files[0]],
                dtype=np.float32,
            )
            for p in sorted(stroke5_dir.glob("*.npz"))
        ]
        codebook = np.load(PROJECT_ROOT / args.codebook_path)
        analyze_quantization_error(arrays, codebook, output_dir=out_dir)

    elif args.command == "latent-capacity":
        data = np.load(args.latent_vectors)
        latents = data if isinstance(data, np.ndarray) else data[data.files[0]]
        analyze_latent_capacity(latents, output_dir=out_dir)

    elif args.command == "conversion":
        report = evaluate_logits_files(
            Path(args.candidate_logits),
            Path(args.reference_logits),
            output_dir=out_dir,
            min_cosine=args.min_cosine,
            min_agreement=args.min_agreement,
            valid_mask_path=args.valid_mask,
            lengths_path=args.lengths,
        )
        exit_code = 0 if report["status"] == "PASS" else 1

    elif args.command == "plot":
        from evals.visualize_failure_analysis import generate_all_plots

        generate_all_plots(
            PROJECT_ROOT / args.analysis_dir, PROJECT_ROOT / args.output_dir
        )

    elif args.command == "all":
        dataset_root = PROJECT_ROOT / args.dataset_root
        print("=" * 60)
        print("Running all available failure analyses")
        print("=" * 60)

        # 1. Separator distribution
        print("\n--- 1. Separator Distribution ---")
        if dataset_root.exists():
            analyze_separators(dataset_root, args.split, output_dir=out_dir)
        else:
            print(f"[skip] dataset root not found: {dataset_root}")

        # 2. Codebook coverage
        codebook_path = PROJECT_ROOT / args.codebook_path
        if codebook_path.exists() and dataset_root.exists():
            print("\n--- 2. Codebook Coverage Analysis ---")
            analyze_codebook(
                codebook_path, dataset_root, args.split, output_dir=out_dir
            )
        else:
            print("[skip] codebook or dataset not found")

        # 3. TF vs FR gap
        if args.tf_report and args.fr_per_sample:
            print("\n--- 3. TF vs FR Gap Analysis ---")
            analyze_tf_fr_gap(Path(args.tf_report), Path(args.fr_per_sample), out_dir)

        # 4. Position accuracy curves & Separator confusion
        if args.predictions:
            print("\n--- 4. Position Accuracy Curves ---")
            analyze_position_accuracy(Path(args.predictions), output_dir=out_dir)
            print("\n--- 5. Separator Confusion & Cascade ---")
            analyze_separator_confusion(Path(args.predictions), output_dir=out_dir)

        # 5. Quantization
        if args.stroke5_dir and codebook_path.exists():
            print("\n--- 6. Codebook Quantization Error ---")
            stroke5_dir = Path(args.stroke5_dir)
            arrays = [
                np.asarray(
                    np.load(p, allow_pickle=True)[
                        np.load(p, allow_pickle=True).files[0]
                    ],
                    dtype=np.float32,
                )
                for p in sorted(stroke5_dir.glob("*.npz"))
            ]
            analyze_quantization_error(
                arrays, np.load(codebook_path), output_dir=out_dir
            )

        # 6. Latent capacity
        if args.latent_vectors:
            print("\n--- 7. Latent Bottleneck Capacity ---")
            data = np.load(args.latent_vectors)
            latents = data if isinstance(data, np.ndarray) else data[data.files[0]]
            analyze_latent_capacity(latents, output_dir=out_dir)

        # 7. Conversion fidelity
        if args.candidate_logits and args.reference_logits:
            print("\n--- 8. Conversion Fidelity ---")
            report = evaluate_logits_files(
                Path(args.candidate_logits),
                Path(args.reference_logits),
                output_dir=out_dir,
                valid_mask_path=args.valid_mask,
                lengths_path=args.lengths,
            )
            if report["status"] != "PASS":
                exit_code = 1

        if args.plots:
            print("\n--- 9. Generating Figures ---")
            from evals.visualize_failure_analysis import generate_all_plots

            generate_all_plots(out_dir, out_dir / "figures")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

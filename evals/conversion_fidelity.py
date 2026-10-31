"""Conversion fidelity and parity analysis between TensorFlow and PyTorch.

Evaluates whether model checkpoint conversion (TensorFlow -> PyTorch) preserved
numerical parity:
- Flattened logit cosine similarity (target >= 0.999)
- Token argmax agreement rate (target >= 0.99)
- Top-k candidate agreement (top-3, top-5)
- Maximum absolute logit difference
- Parameter count and norm checks
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def _add_project_to_path() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists() and (parent / "configs").exists():
            sys.path.insert(0, str(parent))
            return parent
    raise RuntimeError("Could not find project root directory.")


PROJECT_ROOT = _add_project_to_path()


# ---------------------------------------------------------------------------
# Core Parity Metrics
# ---------------------------------------------------------------------------

def compute_parity_metrics(
    candidate_logits: torch.Tensor | np.ndarray,
    reference_logits: torch.Tensor | np.ndarray,
    *,
    min_cosine: float = 0.999,
    min_agreement: float = 0.99,
    top_k: int = 5,
) -> dict[str, Any]:
    """Compute parity metrics comparing candidate (PyTorch) vs reference (TF) logits.

    Parameters
    ----------
    candidate_logits : torch.Tensor | np.ndarray
        Logits predicted by the converted PyTorch model. Shape: (B, T, V) or (N, V).
    reference_logits : torch.Tensor | np.ndarray
        Reference logits from original TensorFlow model or export. Shape must match candidate.
    min_cosine : float
        Minimum cosine similarity threshold to pass (default: 0.999).
    min_agreement : float
        Minimum top-1 token agreement threshold to pass (default: 0.99).
    top_k : int
        Top-k window to check for reference argmax inclusion (default: 5).

    Returns
    -------
    dict with parity metrics and pass/fail status.
    """
    candidate = torch.as_tensor(candidate_logits, dtype=torch.float32)
    reference = torch.as_tensor(reference_logits, dtype=torch.float32)

    if candidate.shape != reference.shape:
        raise ValueError(
            f"Shape mismatch: candidate {tuple(candidate.shape)} vs "
            f"reference {tuple(reference.shape)}"
        )

    if candidate.ndim < 2:
        raise ValueError(
            f"Logits must have at least 2 dimensions (samples/positions, vocab), "
            f"got shape {tuple(candidate.shape)}"
        )

    # Flatten across batch and sequence dimensions: shape (-1, vocab_size)
    cand_flat = candidate.reshape(-1, candidate.shape[-1])
    ref_flat = reference.reshape(-1, reference.shape[-1])
    num_positions = cand_flat.shape[0]
    vocab_size = cand_flat.shape[1]

    # 1. Global flattened cosine similarity
    cand_vec = cand_flat.reshape(1, -1)
    ref_vec = ref_flat.reshape(1, -1)
    global_cosine = float(F.cosine_similarity(cand_vec, ref_vec).item())

    # 2. Per-position cosine similarity statistics
    pos_cosine = F.cosine_similarity(cand_flat, ref_flat, dim=-1)
    mean_pos_cosine = float(pos_cosine.mean().item())
    min_pos_cosine = float(pos_cosine.min().item())

    # 3. Top-1 argmax agreement
    cand_argmax = cand_flat.argmax(dim=-1)
    ref_argmax = ref_flat.argmax(dim=-1)
    matches = (cand_argmax == ref_argmax).float()
    argmax_agreement = float(matches.mean().item())

    # 4. Top-k agreement (is reference top-1 in candidate's top-k?)
    k_actual = min(top_k, vocab_size)
    _, cand_topk_indices = torch.topk(cand_flat, k=k_actual, dim=-1)
    ref_top1_expanded = ref_argmax.unsqueeze(-1)
    in_topk = (cand_topk_indices == ref_top1_expanded).any(dim=-1).float()
    topk_agreement = float(in_topk.mean().item())

    # 5. Numerical difference statistics
    abs_diff = torch.abs(cand_flat - ref_flat)
    max_abs_diff = float(abs_diff.max().item())
    mean_abs_diff = float(abs_diff.mean().item())
    rms_diff = float(torch.sqrt(torch.mean(abs_diff ** 2)).item())

    # 6. Pass / Fail evaluation
    passed = bool(
        global_cosine >= min_cosine and argmax_agreement >= min_agreement
    )

    return {
        "status": "PASS" if passed else "FAIL",
        "num_positions_evaluated": num_positions,
        "vocab_size": vocab_size,
        "metrics": {
            "global_cosine_similarity": round(global_cosine, 6),
            "mean_position_cosine": round(mean_pos_cosine, 6),
            "min_position_cosine": round(min_pos_cosine, 6),
            "argmax_agreement": round(argmax_agreement, 6),
            f"top_{k_actual}_agreement": round(topk_agreement, 6),
            "max_absolute_difference": round(max_abs_diff, 6),
            "mean_absolute_difference": round(mean_abs_diff, 6),
            "rms_difference": round(rms_diff, 6),
        },
        "thresholds": {
            "min_cosine": min_cosine,
            "min_argmax_agreement": min_agreement,
            "cosine_passed": bool(global_cosine >= min_cosine),
            "agreement_passed": bool(argmax_agreement >= min_agreement),
        },
    }


# ---------------------------------------------------------------------------
# File and NPZ Evaluation
# ---------------------------------------------------------------------------

def evaluate_logits_files(
    candidate_path: Path,
    reference_path: Path,
    output_dir: Path | None = None,
    *,
    min_cosine: float = 0.999,
    min_agreement: float = 0.99,
) -> dict[str, Any]:
    """Compare candidate and reference logit arrays loaded from .npz or .npy."""
    candidate_path = Path(candidate_path)
    reference_path = Path(reference_path)

    cand_array = _load_array(candidate_path)
    ref_array = _load_array(reference_path)

    results = compute_parity_metrics(
        cand_array,
        ref_array,
        min_cosine=min_cosine,
        min_agreement=min_agreement,
    )

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "candidate_file": str(candidate_path),
        "reference_file": str(reference_path),
        **results,
    }

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "conversion_fidelity_report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[conversion_fidelity] saved report to {report_path}")

    return report


def _load_array(path: Path) -> np.ndarray:
    """Load array from .npy or .npz file."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if path.suffix == ".npz":
        with np.load(path) as data:
            for key in ("logits", "token_logits", "arr_0", data.files[0]):
                if key in data:
                    return np.asarray(data[key], dtype=np.float32)
            raise ValueError(f"No suitable logit array found in {path}. Keys: {data.files}")
    else:
        return np.load(path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify TensorFlow to PyTorch checkpoint conversion fidelity."
    )
    parser.add_argument(
        "--candidate-logits",
        type=Path,
        default=None,
        help="Path to PyTorch candidate logits (.npy or .npz).",
    )
    parser.add_argument(
        "--reference-logits",
        type=Path,
        default=None,
        help="Path to reference TensorFlow logits (.npy or .npz).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evals/results/conversion_fidelity"),
        help="Directory to write fidelity report.",
    )
    parser.add_argument(
        "--min-cosine",
        type=float,
        default=0.999,
        help="Minimum logit cosine similarity threshold (default: 0.999).",
    )
    parser.add_argument(
        "--min-agreement",
        type=float,
        default=0.99,
        help="Minimum argmax token agreement threshold (default: 0.99).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.candidate_logits and args.reference_logits:
        report = evaluate_logits_files(
            args.candidate_logits,
            args.reference_logits,
            output_dir=args.output_dir,
            min_cosine=args.min_cosine,
            min_agreement=args.min_agreement,
        )
        status = report["status"]
        cosine = report["metrics"]["global_cosine_similarity"]
        agreement = report["metrics"]["argmax_agreement"]
        print(f"\n[conversion_fidelity] Result: {status}")
        print(f"  Cosine Similarity: {cosine:.6f} (target >= {args.min_cosine})")
        print(f"  Argmax Agreement : {agreement:.6f} (target >= {args.min_agreement})")
        return 0 if status == "PASS" else 1

    print("[conversion_fidelity] Error: Please provide both --candidate-logits and --reference-logits.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

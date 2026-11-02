"""Publication-ready visualizer for failure analysis diagnostics.

Generates figures from failure analysis reports:
1. Position-accuracy curves with onset (80%) and collapse (50%) thresholds
2. Teacher-forced vs free-running accuracy gap bar chart across length buckets
3. Codebook usage frequency distribution and dead-token analysis
4. Latent representation singular value spectrum (128-dim bottleneck scree plot)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _add_project_to_path() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists() and (parent / "configs").exists():
            sys.path.insert(0, str(parent))
            return parent
    raise RuntimeError("Could not find project root directory.")


PROJECT_ROOT = _add_project_to_path()

# Modern style palette
_COLORS = {
    "primary": "#2b5c8f",
    "secondary": "#d95f02",
    "accent": "#7570b3",
    "green": "#2ca02c",
    "red": "#d62728",
    "gray": "#7f7f7f",
    "grid": "#e0e0e0",
}

_BUCKET_COLORS = {
    "short": "#1b9e77",
    "medium": "#377eb8",
    "long": "#ff7f00",
    "extended": "#e41a1c",
    "overall": "#333333",
}


# ---------------------------------------------------------------------------
# 1. Position-Accuracy Curves Plot
# ---------------------------------------------------------------------------

def plot_position_accuracy_curves(
    data_or_path: dict[str, Any] | Path,
    output_path: Path,
    *,
    onset_threshold: float = 0.80,
    collapse_threshold: float = 0.50,
) -> Path:
    """Plot token accuracy as a function of sequence position with thresholds."""
    if isinstance(data_or_path, (str, Path)):
        data = json.loads(Path(data_or_path).read_text(encoding="utf-8"))
    else:
        data = data_or_path

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)

    by_bucket = data.get("by_bucket", {})
    plotted = False

    for bucket_name, b_data in by_bucket.items():
        curve = b_data.get("accuracy_curve", [])
        if not curve:
            continue
        positions = np.arange(len(curve))
        color = _BUCKET_COLORS.get(bucket_name.lower(), _COLORS["primary"])
        label = f"{bucket_name.capitalize()} (n={b_data.get('num_samples', '?')})"
        ax.plot(positions, curve, label=label, color=color, linewidth=1.8, alpha=0.9)
        plotted = True

    # Plot overall if available
    overall = data.get("overall", {})
    if overall and overall.get("accuracy_curve"):
        curve = overall["accuracy_curve"]
        ax.plot(
            np.arange(len(curve)),
            curve,
            label="Overall",
            color=_BUCKET_COLORS["overall"],
            linewidth=2.2,
            linestyle="-",
        )
        plotted = True

    if not plotted:
        ax.text(0.5, 0.5, "No position accuracy curves available", ha="center", va="center")

    # Threshold horizontal lines
    ax.axhline(
        onset_threshold,
        color=_COLORS["secondary"],
        linestyle="--",
        linewidth=1.2,
        label=f"Onset Threshold ({int(onset_threshold * 100)}%)",
    )
    ax.axhline(
        collapse_threshold,
        color=_COLORS["red"],
        linestyle="--",
        linewidth=1.2,
        label=f"Collapse Threshold ({int(collapse_threshold * 100)}%)",
    )

    ax.set_title("Sequence Position vs. Token Accuracy (Long-Sequence Degradation)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Sequence Token Position", fontsize=10)
    ax.set_ylabel("Token Prediction Accuracy", fontsize=10)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, linestyle=":", alpha=0.6, color=_COLORS["grid"])
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualizer] saved position-accuracy plot to {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# 2. Teacher-Forced vs Free-Running Gap Plot
# ---------------------------------------------------------------------------

def plot_tf_fr_gap(
    data_or_path: dict[str, Any] | Path,
    output_path: Path,
) -> Path:
    """Plot grouped bar chart of Teacher-Forced vs Free-Running accuracy by bucket."""
    if isinstance(data_or_path, (str, Path)):
        data = json.loads(Path(data_or_path).read_text(encoding="utf-8"))
    else:
        data = data_or_path

    per_bucket = data.get("per_bucket", {})
    if not per_bucket:
        fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
        ax.text(0.5, 0.5, "No bucket gap data available", ha="center", va="center")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return output_path

    buckets = list(per_bucket.keys())
    x = np.arange(len(buckets))
    width = 0.35

    tf_vals: list[float] = []
    fr_vals: list[float] = []

    for b in buckets:
        b_info = per_bucket[b]
        tf_acc = b_info.get("tf_token_accuracy_mean")
        fr_acc = b_info.get("fr_token_accuracy_mean", 0.0)
        tf_vals.append(float(tf_acc) if tf_acc is not None else 0.0)
        fr_vals.append(float(fr_acc))

    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=150)

    rects1 = ax.bar(x - width / 2, tf_vals, width, label="Teacher-Forced (TF)", color=_COLORS["primary"], alpha=0.85)
    rects2 = ax.bar(x + width / 2, fr_vals, width, label="Free-Running (FR)", color=_COLORS["secondary"], alpha=0.85)

    # Add gap labels above bars
    for i in range(len(buckets)):
        tf_val = tf_vals[i]
        fr_val = fr_vals[i]
        if tf_val > 0 and fr_val > 0:
            gap = tf_val - fr_val
            pct = gap * 100
            y_pos = max(tf_val, fr_val) + 0.03
            ax.text(x[i], y_pos, f"Δ = -{pct:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold", color=_COLORS["red"])

    ax.set_title("Teacher-Forced vs. Free-Running Accuracy Gap (Exposure Bias)", fontsize=12, fontweight="bold", pad=12)
    ax.set_ylabel("Token Accuracy", fontsize=10)
    ax.set_xlabel("Length Complexity Bucket", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([b.capitalize() for b in buckets], fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", linestyle=":", alpha=0.6, color=_COLORS["grid"])
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualizer] saved TF vs FR gap plot to {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# 3. Codebook Usage Distribution Plot
# ---------------------------------------------------------------------------

def plot_codebook_usage(
    counts_or_path: np.ndarray | Path,
    output_path: Path,
) -> Path:
    """Plot sorted log-frequency distribution of codebook tokens and dead entries."""
    if isinstance(counts_or_path, (str, Path)):
        p = Path(counts_or_path)
        if p.suffix == ".json":
            data = json.loads(p.read_text(encoding="utf-8"))
            counts = np.array([x["count"] for x in data.get("top_10_entries", [])])
        else:
            counts = np.load(p)
    else:
        counts = np.asarray(counts_or_path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)

    total_entries = len(counts)
    used_entries = int(np.sum(counts > 0))
    dead_entries = total_entries - used_entries

    # Panel 1: Donut chart of used vs dead entries
    ax1.pie(
        [used_entries, dead_entries],
        labels=[f"Active ({used_entries})", f"Dead ({dead_entries})"],
        autopct="%1.1f%%",
        startangle=90,
        colors=[_COLORS["green"], _COLORS["red"]],
        wedgeprops=dict(width=0.45, edgecolor="white"),
    )
    ax1.set_title(f"Codebook Active vs Dead Entries (K={total_entries})", fontsize=11, fontweight="bold")

    # Panel 2: Ranked frequency curve (Zipfian plot)
    sorted_counts = np.sort(counts)[::-1]
    rank = np.arange(1, len(sorted_counts) + 1)

    ax2.plot(rank, sorted_counts + 1, color=_COLORS["primary"], linewidth=2)
    ax2.set_yscale("log")
    ax2.set_xscale("log")
    ax2.set_title("Ranked Token Usage (Log-Log Frequency)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Token Rank (Most to Least Frequent)", fontsize=10)
    ax2.set_ylabel("Occurrences (Count + 1)", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.6, color=_COLORS["grid"])

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualizer] saved codebook usage plot to {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# 4. Latent Space Capacity Scree Plot (Bottleneck Analysis)
# ---------------------------------------------------------------------------

def plot_latent_capacity(
    data_or_path: dict[str, Any] | Path,
    output_path: Path,
) -> Path:
    """Plot cumulative variance explained across latent dimensions (Scree plot)."""
    if isinstance(data_or_path, (str, Path)):
        data = json.loads(Path(data_or_path).read_text(encoding="utf-8"))
    else:
        data = data_or_path

    top_var = data.get("cumulative_variance_explained", {})
    eff_dim = data.get("effective_dimensionality_pr", None)
    latent_dim = data.get("latent_dimension", 128)

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)

    if top_var:
        ks = [int(k.replace("top_", "")) for k in top_var.keys()]
        vals = [float(v) * 100 for v in top_var.values()]
        ax.plot(ks, vals, marker="o", color=_COLORS["primary"], linewidth=2, markersize=6)
        for k, v in zip(ks, vals):
            ax.annotate(
                f"{v:.1f}%",
                (k, v),
                textcoords="offset points",
                xytext=(0, 7),
                ha="center",
                fontsize=8,
                fontweight="bold",
            )

    if eff_dim is not None:
        ax.axvline(
            eff_dim,
            color=_COLORS["secondary"],
            linestyle="--",
            linewidth=1.5,
            label=f"Effective Dimension (PR = {eff_dim:.1f} / {latent_dim})",
        )

    ax.set_title("128-Dim Latent Bottleneck: Cumulative Variance Explained", fontsize=11, fontweight="bold", pad=12)
    ax.set_xlabel("Number of Principal Components", fontsize=10)
    ax.set_ylabel("Variance Explained (%)", fontsize=10)
    ax.set_ylim(0, 110)
    ax.grid(True, linestyle=":", alpha=0.6, color=_COLORS["grid"])
    if eff_dim is not None:
        ax.legend(loc="lower right", framealpha=0.9, fontsize=9)

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualizer] saved latent capacity plot to {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Generate All Available Plots
# ---------------------------------------------------------------------------

def generate_all_plots(
    analysis_dir: Path,
    output_dir: Path,
) -> list[Path]:
    """Scan analysis directory and generate figures for all available reports."""
    analysis_dir = Path(analysis_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generated: list[Path] = []

    # 1. Position-accuracy curves
    pos_path = analysis_dir / "position_accuracy_curves.json"
    if pos_path.exists():
        out = plot_position_accuracy_curves(pos_path, output_dir / "figure_position_accuracy_curves.png")
        generated.append(out)

    # 2. TF vs FR gap
    gap_path = analysis_dir / "tf_vs_fr_gap_analysis.json"
    if gap_path.exists():
        out = plot_tf_fr_gap(gap_path, output_dir / "figure_tf_vs_fr_gap.png")
        generated.append(out)

    # 3. Codebook usage
    cb_counts_path = analysis_dir / "codebook_usage_counts.npy"
    if cb_counts_path.exists():
        out = plot_codebook_usage(cb_counts_path, output_dir / "figure_codebook_usage.png")
        generated.append(out)
    else:
        cb_json_path = analysis_dir / "codebook_analysis.json"
        if cb_json_path.exists():
            out = plot_codebook_usage(cb_json_path, output_dir / "figure_codebook_usage.png")
            generated.append(out)

    # 4. Latent capacity
    lat_path = analysis_dir / "latent_capacity_analysis.json"
    if lat_path.exists():
        out = plot_latent_capacity(lat_path, output_dir / "figure_latent_capacity_scree.png")
        generated.append(out)

    return generated


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate publication-ready figures for failure analysis."
    )
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=Path("evals/results/analysis"),
        help="Directory containing failure analysis JSON and NPY files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evals/results/analysis/figures"),
        help="Directory to save generated figures.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print("=" * 60)
    print("Generating Failure Analysis Figures")
    print(f"Source: {args.analysis_dir}")
    print(f"Output: {args.output_dir}")
    print("=" * 60)

    saved = generate_all_plots(args.analysis_dir, args.output_dir)
    print(f"\nDone! Generated {len(saved)} figures in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

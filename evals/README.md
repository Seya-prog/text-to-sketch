# Benchmark Evaluation and Failure Analysis

Phase 1 Weeks 1–2 tooling for evaluating the Sketchformer prediction model.

## Scripts

### 1. Benchmark Manifest (`benchmark_manifest.py`)

Generates a stratified JSONL manifest annotating every sample with length bucket,
stroke count, detail density, and separator ratio.

```bash
python evals/benchmark_manifest.py \
  --dataset-root data/processed/sketchformer-ready-data/tok-dict \
  --splits valid test \
  --output evals/benchmark_manifest.jsonl
```

### 2. Benchmark Evaluation (`benchmark_eval.py`)

Deterministic evaluation harness that runs both teacher-forced and free-running
evaluation, reports per-bucket metrics, and saves a JSON report with full
provenance (checkpoint hash, codebook hash, config hash, seed, timestamp).

```bash
# Full benchmark (both modes) + 50 qualitative comparison plots
python evals/benchmark_eval.py \
  --checkpoint weights/pretrained/sketchformer_tok_dict_init.safetensors \
  --split valid \
  --num-plots 50 \
  --output-dir evals/results \
  --run-name baseline_valid

# Save predictions for failure analysis (position-curves & separator-confusion)
python evals/benchmark_eval.py \
  --checkpoint weights/pretrained/sketchformer_tok_dict_init.safetensors \
  --split valid \
  --save-predictions \
  --output-dir evals/results \
  --run-name baseline_valid

# Teacher-forced only
python evals/benchmark_eval.py \
  --checkpoint weights/pretrained/sketchformer_tok_dict_init.safetensors \
  --modes teacher-forced \
  --split valid

# Free-running only
python evals/benchmark_eval.py \
  --checkpoint weights/pretrained/sketchformer_tok_dict_init.safetensors \
  --modes free-running \
  --split valid
```

### 3. TensorFlow-to-PyTorch Conversion Fidelity (`conversion_fidelity.py`)

Evaluates numerical parity between converted PyTorch model logits and reference
TensorFlow outputs (global cosine similarity $\ge 0.999$, token argmax agreement $\ge 0.99$).

```bash
python evals/conversion_fidelity.py \
  --candidate-logits evals/results/pytorch_logits.npz \
  --reference-logits evals/results/tf_reference_logits.npz \
  --output-dir evals/results/conversion_fidelity
```

### 4. Failure Analysis (`failure_analysis.py`)

Diagnostic analysis suite for investigating model failure modes across all Phase 1 dimensions:

```bash
# Run all analyses and automatically generate publication figures
python evals/failure_analysis.py all \
  --dataset-root data/processed/sketchformer-ready-data/tok-dict \
  --codebook-path data/processed/sketch_token/codebook.npy \
  --split valid \
  --plots \
  --output-dir evals/results/analysis

# 1. TF vs FR gap analysis
python evals/failure_analysis.py gap \
  --tf-report evals/results/baseline_valid_report.json \
  --fr-per-sample evals/results/baseline_valid_per_sample.jsonl

# 2. Position-accuracy curves (identifies onset <80% and collapse <50%)
python evals/failure_analysis.py position-curves \
  --predictions evals/results/baseline_valid_per_sample.jsonl

# 3. Separator confusion & cascade effect (precision, recall, post-sep penalty)
python evals/failure_analysis.py separator-confusion \
  --predictions evals/results/baseline_valid_per_sample.jsonl

# 4. Separator dataset distribution
python evals/failure_analysis.py separators --split valid

# 5. Codebook coverage & dead tokens
python evals/failure_analysis.py codebook --split valid

# 6. Quantization error & cumulative endpoint drift
python evals/failure_analysis.py quantization \
  --stroke5-dir data/processed/stroke5 \
  --codebook-path data/processed/sketch_token/codebook.npy

# 7. 128-dim latent bottleneck effective dimension & SVD spectrum
python evals/failure_analysis.py latent-capacity \
  --latent-vectors evals/results/validation_latents.npy

# 8. Generate figures from an existing analysis directory
python evals/failure_analysis.py plot \
  --analysis-dir evals/results/analysis \
  --output-dir evals/results/analysis/figures
```

### 5. Failure Analysis Visualizer (`visualize_failure_analysis.py`)

Generates standalone publication-ready figures for reports and papers:

```bash
python evals/visualize_failure_analysis.py \
  --analysis-dir evals/results/analysis \
  --output-dir evals/results/analysis/figures
```

Generated plots:
- `figure_position_accuracy_curves.png`: Sequence position vs token accuracy per bucket with Onset (80%) and Collapse (50%) thresholds
- `figure_tf_vs_fr_gap.png`: Grouped bar chart comparing Teacher-Forced vs Free-Running accuracy with gap annotations
- `figure_codebook_usage.png`: Active vs dead token donut chart and Zipfian log-log usage frequency curve
- `figure_latent_capacity_scree.png`: Cumulative variance explained scree plot for the 128-dim bottleneck

## Output Structure

```
evals/
├── benchmark_manifest.jsonl          # Per-sample annotations
├── benchmark_manifest_summary.json   # Dataset-level statistics
└── results/
    ├── baseline_valid_report.json    # Frozen baseline numbers + provenance
    ├── baseline_valid_per_sample.jsonl  # Per-sample free-running records (+ predictions)
    ├── baseline_valid_plots/         # 50 qualitative comparison visualizations
    ├── conversion_fidelity/
    │   └── conversion_fidelity_report.json
    └── analysis/
        ├── tf_vs_fr_gap_analysis.json
        ├── position_accuracy_curves.json
        ├── separator_confusion_analysis.json
        ├── separator_analysis.json
        ├── codebook_analysis.json
        ├── codebook_usage_counts.npy
        ├── quantization_analysis.json
        ├── latent_capacity_analysis.json
        └── figures/
            ├── figure_position_accuracy_curves.png
            ├── figure_tf_vs_fr_gap.png
            ├── figure_codebook_usage.png
            └── figure_latent_capacity_scree.png
```

# Benchmark evaluation and failure analysis

These tools prepare the Phase 1 study. They do not convert TensorFlow checkpoints,
train a model, or supply real baseline measurements. Run the study on the GPU
server with the actual development dataset, codebook, and compatible weights.
The held-out test split is evaluated only when explicitly selected.

## Before running on the server

Install the repository dependencies and editable package in your server's Python
3.10/3.11 environment. Install the PyTorch CUDA wheels appropriate for its NVIDIA
driver. Check `torch.cuda.is_available()` before requesting `--device cuda`.
Run the CPU regression suite first:

```bash
python -B -m unittest discover -s tests -v
```

Use a checkpoint converted for the selected architecture/sequence-length experiment.
For example, `anime_tok_dict_long` needs the corresponding 4096-token initialization
or fine-tuned checkpoint. A 2048-token checkpoint is not interchangeable with it.
Weight loading is strict by default. `--allow-partial-checkpoint` is an explicit
experimental override; missing/unexpected keys are recorded in the report. A file
with no matching weights is rejected. `--allow-random-init` is for synthetic smoke
tests, not the pretrained baseline.

Benchmarking disables truncation regardless of the training configuration and
checks the entire requested split before inference. An overlong sample causes an
error identifying the sample; choose a compatible longer-sequence experiment and
checkpoint. Targets are never shortened to fit the default 2048-token experiment.
`--max-generation-length` limits generation only; missing output tokens still count
as errors against the complete target. Leave it unset for the full baseline.

## 1. Annotate the development split

```bash
python evals/benchmark_manifest.py \
  --dataset-root data/processed/sketchformer-ready-data/tok-dict-4096 \
  --splits valid \
  --output evals/benchmark_manifest.jsonl
```

Sample IDs match the prepared `sample_ids` array, or fall back to `shard_stem:index`,
as in the model dataloader. IDs must be unique. `sequence_length` and bucket assignment
count non-padding model input tokens **including SOS and EOS**. The legacy
`token_length` field describes content tokens excluding SOS, EOS and padding;
`motion_token_count` additionally excludes separators. Numeric legacy bucket names
(e.g. `1_512`) are normalized to `short`, `medium`, `long`, and `extended` by diagnostics.

## 2. Freeze a real baseline

Replace the artifact locations with those on your server. `--codebook-path` changes
the file location; it does not change the checkpoint's token layout or vocabulary.

```bash
python evals/benchmark_eval.py \
  --experiment anime_tok_dict_long \
  --checkpoint weights/pretrained/sketchformer_tok_dict_4096_init.safetensors \
  --data-root data/processed/sketchformer-ready-data/tok-dict-4096 \
  --codebook-path data/processed/sketch_token/codebook.npy \
  --split valid --device cuda --precision 16-mixed \
  --seed 42 --num-plots 50 --save-predictions --save-latents \
  --output-dir evals/results --run-name baseline_valid
```

The default `--modes teacher-forced free-running` evaluates both modes. Either mode
can be selected independently. Teacher-forced evaluation scores the model's shifted
loss targets and mask. Comparable TF/FR per-sample accuracy excludes the supplied
SOS and padding, includes EOS and separators, and counts missing generated tokens
as incorrect. Global TF loss/accuracy are token-weighted; the per-sample accuracy
means used by gap analysis give every sketch equal weight. Perplexity is the
exponential of pooled token loss, not a mean of batch perplexities.

Reports include all TF loss/accuracy/perplexity and FR accuracy/exact-match/EOS/F1
metrics globally and per populated bucket. Empty buckets are absent, not silently
reported as zero quality. Wall-clock timings include loading/metric computation and
are **end-to-end evaluation times**, not isolated GPU kernel latency.

The report records the checkpoint/codebook hashes, split-file hashes and dataset
snapshot hash, resolved config hash, effective seed/precision, device, code revision,
and peak CUDA memory when available. Only the requested split is loaded; no train
split is needed to benchmark validation data. Seed zero is supported.

Run artifacts are never overwritten: use a new `--run-name` for each experiment.
An interrupted run may leave partial artifacts; preserve/inspect them and retry
with a new run name. Archive the baseline artifacts and the exact code revision.

### Frozen qualitative comparisons

The comparison set is selected deterministically across populated length buckets,
with spare capacity redistributed when a bucket has few samples. When fewer than
50 samples exist, all available samples are selected. For subsequent models reuse
exactly the saved IDs:

```bash
python evals/benchmark_eval.py \
  --experiment anime_tok_dict_long \
  --checkpoint weights/finetuned/sketchformer-tok-dict-anime-4096/best.pt \
  --data-root data/processed/sketchformer-ready-data/tok-dict-4096 \
  --codebook-path data/processed/sketch_token/codebook.npy \
  --split valid --device cuda --precision 16-mixed --seed 42 \
  --save-predictions --save-latents --num-plots 50 \
  --comparison-manifest evals/results/baseline_valid_comparison_manifest.json \
  --output-dir evals/results --run-name candidate_valid
```

The dataset snapshot must match; missing comparison IDs cause an error. Loader
ordering and length bucketing cannot change the selected IDs or plot order.

## 3. Diagnose the baseline

```bash
python evals/failure_analysis.py all \
  --dataset-root data/processed/sketchformer-ready-data/tok-dict-4096 \
  --codebook-path data/processed/sketch_token/codebook.npy --split valid \
  --tf-report evals/results/baseline_valid_report.json \
  --fr-per-sample evals/results/baseline_valid_per_sample.jsonl \
  --predictions evals/results/baseline_valid_per_sample.jsonl \
  --latent-vectors evals/results/baseline_valid_latents.npz \
  --plots --output-dir evals/results/baseline_analysis
```

Individual commands are `gap`, `position-curves`, `separator-confusion`,
`separators`, `codebook`, `quantization`, `latent-capacity`, `conversion`, and `plot`.
Use `<command> --help` for required inputs. Position/cascade analysis needs the
prediction arrays exported by `--save-predictions`; absent arrays cause an error.

Gap analysis joins TF and FR by sample ID and verifies the target lengths/hashes
and scoring convention. It reports per-sample gaps and ranks the worst 20 by
TF-minus-FR accuracy, not geometry F1. A global TF score is never substituted for a
missing bucket. Legacy reports with explicit matching bucket summaries can produce
bucket-only analysis, clearly marked as lacking per-sample gaps.

Missing continuation tokens count as errors in separator cascade windows. Counts
are reported for both comparison groups. If either group is absent, the penalty
and effect flag are `null`, since no comparison is supported. Position curves
report the number of active target samples at each position.

Latent export contains `embeddings`, `sample_ids`, `lengths`, `stroke_counts` and
`tf_token_accuracy`, all in the same row order. Effective dimensionality is the
covariance participation ratio: for singular values `s`, `(sum(s**2))**2 / sum(s**4)`.

## 4. Verify conversion fidelity

The comparison tool consumes outputs already exported from the original TensorFlow
model and the converted PyTorch model. Generate both with identical inputs, sample
order, token IDs, masks, decoding convention, and inference mode. The actual
checkpoint converter is `scripts/sketchformer/convert_checkpoint.py`.

For padded sequence logits, export each `.npz` with `logits` shaped `(B,T,V)` and a
boolean `valid_mask` shaped `(B,T)`, or integer `lengths` shaped `(B,)`. Include only
positions that both implementations are meant to predict; exclude padding and any
supplied prefix. Both files' valid positions must agree if both supply metadata.
Sequence exports without validity metadata are rejected. Genuinely unpadded
position arrays may be exported as `(N,V)`.

```bash
python evals/conversion_fidelity.py \
  --candidate-logits evals/results/pytorch_logits.npz \
  --reference-logits evals/results/tf_reference_logits.npz \
  --output-dir evals/results/conversion_fidelity
```

Alternatively pass `--valid-mask mask.npy` or `--lengths lengths.npy` explicitly.
A comparison with zero valid positions, inconsistent masks, or nonfinite valid
logits fails. The report records input file and mask hashes and evaluated positions.
Thresholds default to cosine >= 0.999 and argmax agreement >= 0.99. A failed check
returns a nonzero exit status, including through `failure_analysis.py conversion`
and `failure_analysis.py all`.

## Outputs and follow-up study work

A complete `baseline_valid` run writes:

- `baseline_valid_report.json`: provenance and global/per-bucket metrics.
- `baseline_valid_teacher_forced_per_sample.jsonl`: TF scores and target identity.
- `baseline_valid_per_sample.jsonl`: FR scores and optional target/predicted arrays.
- `baseline_valid_comparison_manifest.json`: frozen stratified comparison IDs.
- `baseline_valid_plots/`: numbered plots in the manifest's selection order.
- `baseline_valid_latents.npz`: optional embeddings with aligned sample metadata.

`failure_analysis.py all --plots` produces diagnostic reports and available figures.
Its optional analyses run only when their inputs are supplied; it does not generate
TensorFlow reference outputs or missing latent/prediction files.

The GPU study still needs actual baseline measurements, verified TensorFlow exports,
per-sample quantization/reconstruction correlations, complexity-colored embedding
visualizations, and a consolidated interpretation with Week 3 recommendations.
The exported IDs/embeddings support that work. None of these empirical findings are
claimed by CPU regression tests, and they must not be inferred from synthetic runs.

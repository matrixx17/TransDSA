# TransDSA

TransDSA converts a pretrained Multi-head Latent Attention (MLA) model into a DeepSeek Sparse Attention (DSA) model, and replaces the normally-trained Indexer with a deterministic, calibration-based token scorer derived from TriAttention's frequency-domain analysis. The result is a sparse-attention model that requires no gradient training of the selector — only a short calibration pass over held-out text.

## Overview

A standard DSA model adds an Indexer module to each attention layer that learns which past tokens to attend to. Training that Indexer is expensive and requires dense-attention supervision. TransDSA takes a different route:

1. **Convert** a TransMLA checkpoint (e.g. `BarraHome/llama3_2-1B-deepseek`) into a DSA-shaped model by grafting an Indexer slot onto every MLA attention layer. No weights are copied or retrained.
2. **Calibrate** the model on a small corpus (128 wikitext-2 samples by default), recording per-head frequency-domain statistics of pre-RoPE queries.
3. **Score** tokens at inference time with `TriAttentionScorer`, a deterministic module that uses the calibration stats plus pre-RoPE key vectors to rank positions by predicted attention importance. No learned parameters.

The scorer operates on the RoPE component of keys only (`qk_rope_head_dim = 64`); the nope component carries no positional information and is not used for selection.

## Repository layout

```
transdsa/
  converter.py      MLA -> DSA conversion; load_dsa_model() reloader
  modeling.py       DSAAttention wrapper, Indexer, tri-scorer integration
  calibrate.py      Per-head frequency statistics collection
  tri_scorer.py     TriAttentionScorer (deterministic, stats-based)
tests/
  check.py          Weight integrity and Indexer-mask sanity checks
  test_tri_scorer.py End-to-end test of scorer attached to every layer
```

## Usage

### 1. Convert an MLA model to DSA

```
python transdsa/converter.py \
    --model-path BarraHome/llama3_2-1B-deepseek \
    --output-path ./outputs/llama3.2-1b-dsa \
    --index-n-heads 8 \
    --index-head-dim 64 \
    --index-topk 256
```

This writes a checkpoint with DSA-aware config fields and the original MLA weights. The Indexer parameter slots exist but are untrained.

### 2. Calibrate frequency statistics

```
python -m transdsa.calibrate \
    --checkpoint ./outputs/llama3.2-1b-dsa \
    --output ./outputs/llama3.2-1b-dsa-stats.pt \
    --device mps
```

Runs 128 wikitext-2 samples through the model with forward pre-hooks on each attention layer, accumulating per-head complex-Q mean and absolute-Q mean in the frequency domain. Saves a `.pt` payload with `metadata` and a `stats` dict keyed by `layer{ii}_head{jj}`.

### 3. Attach the scorer and run inference

```python
from transdsa.converter import load_dsa_model
from transdsa.tri_scorer import TriAttentionScorer

model = load_dsa_model("./outputs/llama3.2-1b-dsa", device="mps")
scorer = TriAttentionScorer(
    stats_path="./outputs/llama3.2-1b-dsa-stats.pt",
    model_path="./outputs/llama3.2-1b-dsa",
).to("mps")

for layer in model.model.layers:
    layer.self_attn.load_tri_scorer(scorer)

# Standard forward pass now uses deterministic tri-scorer selection.
```

Set `layer.self_attn.use_tri_scorer = False` on any layer to fall back to the learned Indexer path.

## Design notes

- **Self-contained scorer.** `TriAttentionScorer` takes only `(stats_path, model_path)` and derives everything else internally: it reads the model config, builds its own `LlamaRotaryEmbedding`, and computes `omega` / `freq_scale_sq` from a rotary probe. The scorer owns its rotary embedder, so it can generate cos/sin tables for any position range without needing them from the caller.
- **Prefill vs decode.** For prefill, pre-RoPE keys are captured directly from `_compute_mla_qkv` and passed to the scorer. For decode, the cache stores post-RoPE keys; `score_tokens_with_cache` extracts the rope portion from the cache and calls `_invert_rope` to recover the pre-RoPE representation before scoring.
- **Device handling.** `TriAttentionScorer` is not an `nn.Module`. It exposes an explicit `.to(device)` that moves all internal tensors including the rotary embedder, and rebuilds its layer-indexed lookup structure.

## Running the tests

```
python tests/check.py
python tests/test_tri_scorer.py --device mps  # if using M-Series Mac
python tests/sanity_sparse_vs_dense.py --device mps --seq-len 1024 --topk 64
python tests/sparsity_sweep.py --device mps --seq-lens 1024 2048 \
    --topks 16 32 64 128 256 512
```

`check.py` verifies that the converter preserves MLA weights, that Indexer shapes are correct, and that the Indexer mask actually affects the output when `topk=1`. `test_tri_scorer.py` builds the scorer, attaches it to every layer, and runs a forward pass against the Indexer baseline. `sanity_sparse_vs_dense.py` compares the tri-scorer, a uniform-random selector, and dense attention on a long prompt. `sparsity_sweep.py` runs the same comparison across a grid of `(seq_len, topk)` values and writes a CSV.

## Findings

### Single prompt, topk=64, seq_len=1024

| Mode | MSE vs dense | Top-1 agreement |
|---|---|---|
| Tri-scorer | 4.33 | 22.5% |
| Random | 5.55 | 8.1% |

### Sparsity sweep (single prompt)

| seq_len | topk | sparsity | tri top-1 | rand top-1 | gap |
|---:|---:|---:|---:|---:|---:|
| 1024 | 16 | 1.6% | 0.155 | 0.078 | +0.077 |
| 1024 | 32 | 3.1% | 0.152 | 0.092 | +0.061 |
| 1024 | 64 | 6.3% | 0.225 | 0.070 | +0.154 |
| 1024 | 128 | 12.5% | 0.193 | 0.140 | +0.054 |
| 1024 | 256 | 25.0% | 0.597 | 0.312 | +0.285 |
| 1024 | 512 | 50.0% | 0.806 | 0.472 | +0.334 |
| 2048 | 16 | 0.8% | 0.159 | 0.069 | +0.090 |
| 2048 | 32 | 1.6% | 0.183 | 0.087 | +0.095 |
| 2048 | 64 | 3.1% | 0.194 | 0.072 | +0.122 |
| 2048 | 128 | 6.3% | 0.187 | 0.094 | +0.092 |
| 2048 | 256 | 12.5% | 0.394 | 0.145 | +0.249 |
| 2048 | 512 | 25.0% | 0.493 | 0.343 | +0.150 |

The tri-scorer beats uniform-random on next-token top-1 agreement in every cell, with gaps of +6 to +33 percentage points. Top-1 agreement saturates around 25-50% of context retained (`topk / seq_len`). At extreme sparsity (≤ 6% of context) all selectors lose substantial fidelity to dense, but tri-scorer remains roughly 2-3x better than random.

MSE numbers are noisier on a single prompt (random selection has high variance per draw) — top-1 agreement is the more reliable metric. A multi-prompt perplexity sweep is the natural next step.

## Status

End-to-end plumbing is verified, and the scorer cleanly beats a random-selection baseline across all tested sparsity levels and at both 1k and 2k context. Remaining work: perplexity harness on wikitext-2 held-out (many prompts, lower variance), longer-context runs (4k+), and batch-size-greater-than-one support in the decode path.

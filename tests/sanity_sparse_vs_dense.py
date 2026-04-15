"""
Sanity check: is the tri-scorer meaningfully better than random selection?

Compares three attention modes on one long prompt:
  1. Dense       — no masking (every key attended)
  2. Tri-scorer  — calibration-based deterministic selection
  3. Random      — uniformly random top-k positions per layer

For each sparse mode we report mean squared error of logits against the
dense baseline.  If tri-scorer MSE is not meaningfully below random MSE,
the calibration / scoring math is not doing useful work and there is no
point running a full sparsity sweep.

Usage
-----
    python tests/sanity_sparse_vs_dense.py --device mps --seq-len 1024 --topk 64
"""
from __future__ import annotations

import argparse
import os
import sys

import transformers

if not hasattr(transformers.utils, "LossKwargs"):
    from typing import TypedDict
    class LossKwargs(TypedDict, total=False):
        pass
    transformers.utils.LossKwargs = LossKwargs

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer

from transdsa.converter import load_dsa_model
from transdsa.tri_scorer import TriAttentionScorer


# ---------------------------------------------------------------------------
# Random scorer — drop-in replacement with the same interface as
# TriAttentionScorer, returning uniformly random top-k indices.
# ---------------------------------------------------------------------------

class RandomScorer:
    """Drop-in scorer that selects top-k indices uniformly at random."""

    def __init__(self, device: str = "cpu", seed: int = 0) -> None:
        self.device = torch.device(device)
        self.generator = torch.Generator(device="cpu").manual_seed(seed)

    def to(self, device):
        self.device = torch.device(device)
        return self

    def _random_topk(self, seq_k: int, topk: int) -> torch.Tensor:
        k = min(topk, seq_k)
        perm = torch.randperm(seq_k, generator=self.generator)[:k]
        return perm.to(self.device)

    def score_tokens(
        self, layer_idx, k_rot_raw, absolute_positions, current_position, topk,
    ):
        seq_k = k_rot_raw.shape[0]
        return self._random_topk(seq_k, topk)

    def score_tokens_with_cache(
        self, layer_idx, cached_key_states, k_rot_raw_current,
        cache_position, seq_q, topk,
    ):
        seq_k = cached_key_states.shape[2]
        return self._random_topk(seq_k, topk)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_with_scorer(model, scorer, input_ids):
    """Attach scorer to every layer, run forward, return logits."""
    for layer in model.model.layers:
        layer.self_attn.load_tri_scorer(scorer)
    with torch.no_grad():
        out = model(input_ids=input_ids)
    return out.logits


def run_dense(model, scorer_cls_for_topk_override, input_ids, seq_len):
    """Run with index_topk set >= seq_len so every key is selected."""
    # Save original topk values, bump them, run, then restore.
    orig_topks = []
    for layer in model.model.layers:
        orig_topks.append(layer.self_attn.dsa_config.index_topk)
        layer.self_attn.dsa_config.index_topk = seq_len + 1

    # Use the tri-scorer path (already attached) — with topk >= seq_k it
    # selects every position, i.e. dense attention.
    with torch.no_grad():
        out = model(input_ids=input_ids)

    for layer, orig in zip(model.model.layers, orig_topks):
        layer.self_attn.dsa_config.index_topk = orig

    return out.logits


def main(args):
    device = args.device

    # -----------------------------------------------------------------
    # Load model + scorers
    # -----------------------------------------------------------------
    print(f"Loading model from {args.checkpoint} on {device}...")
    model = load_dsa_model(args.checkpoint, device=device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint, trust_remote_code=True,
    )

    print(f"Building tri-scorer from {args.stats}...")
    tri_scorer = TriAttentionScorer(
        stats_path=args.stats, model_path=args.checkpoint,
    ).to(device)

    random_scorer = RandomScorer(device=device, seed=args.seed)

    # -----------------------------------------------------------------
    # Build a prompt of the requested length by repeating wikitext-like text
    # -----------------------------------------------------------------
    seed_text = (
        "The history of artificial intelligence began in antiquity, with myths, "
        "stories and rumors of artificial beings endowed with intelligence or "
        "consciousness by master craftsmen. The seeds of modern AI were planted "
        "by philosophers who attempted to describe the process of human thinking "
        "as the mechanical manipulation of symbols. "
    )
    # Repeat enough times to exceed seq_len after tokenization
    prompt = seed_text * 64
    input_ids = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=args.seq_len,
    )["input_ids"].to(device)
    seq_len = input_ids.shape[1]
    print(f"Prompt tokenized to {seq_len} tokens.")

    # -----------------------------------------------------------------
    # Run three modes
    # -----------------------------------------------------------------
    # Dense: attach tri-scorer, bump topk above seq_len so every position
    # is selected (causal mask still applied separately in forward).
    print("\nRunning dense baseline...")
    for layer in model.model.layers:
        layer.self_attn.load_tri_scorer(tri_scorer)
    logits_dense = run_dense(model, None, input_ids, seq_len)

    print(f"Running tri-scorer (topk={args.topk})...")
    for layer in model.model.layers:
        layer.self_attn.dsa_config.index_topk = args.topk
    logits_tri = run_with_scorer(model, tri_scorer, input_ids)

    print(f"Running random scorer (topk={args.topk})...")
    logits_random = run_with_scorer(model, random_scorer, input_ids)

    # -----------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------
    def mse(a, b):
        return (a.float() - b.float()).pow(2).mean().item()

    def top1_agreement(a, b):
        return (a.argmax(dim=-1) == b.argmax(dim=-1)).float().mean().item()

    print("\n" + "=" * 60)
    print(f"Results — seq_len={seq_len}, topk={args.topk}")
    print("=" * 60)
    mse_tri = mse(logits_tri, logits_dense)
    mse_rand = mse(logits_random, logits_dense)
    agree_tri = top1_agreement(logits_tri, logits_dense)
    agree_rand = top1_agreement(logits_random, logits_dense)
    print(f"  Tri-scorer   MSE vs dense: {mse_tri:.6f}   top-1 agreement: {agree_tri:.4f}")
    print(f"  Random       MSE vs dense: {mse_rand:.6f}   top-1 agreement: {agree_rand:.4f}")
    ratio = mse_tri / mse_rand if mse_rand > 0 else float("inf")
    print(f"  Tri/Random MSE ratio: {ratio:.4f}  (<1 means tri-scorer closer to dense)")
    print()
    if ratio < 0.9:
        print("  Tri-scorer is meaningfully closer to dense than random. PROCEED.")
    elif ratio < 1.0:
        print("  Tri-scorer is marginally better than random. Inconclusive.")
    else:
        print("  Tri-scorer is NOT better than random. Investigate before sweeping.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="./outputs/llama3.2-1b-dsa")
    parser.add_argument("--stats", default="./outputs/llama3.2-1b-dsa-stats.pt")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--topk", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args)

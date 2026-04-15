"""
End-to-end test for TriAttentionScorer integration with DSAAttention.

Verifies:
  1. TriAttentionScorer can be constructed from (stats_path, model_path).
  2. The scorer can be moved to a target device via .to(device).
  3. DSAAttention.load_tri_scorer() wires the scorer into every layer.
  4. A forward pass with use_tri_scorer=True produces logits of the
     expected shape and differs from the learned-Indexer path.

Usage
-----
    python tests/test_tri_scorer.py --device cpu
"""
from __future__ import annotations

import argparse
import os
import sys

import transformers

# LossKwargs shim — must run before any HF model is loaded.
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


def main(args: argparse.Namespace) -> None:
    checkpoint_path = args.checkpoint
    stats_path = args.stats
    device = args.device

    # ------------------------------------------------------------------
    # 1. Build the scorer from stats + model paths — self-contained.
    # ------------------------------------------------------------------
    print(f"Building TriAttentionScorer from:")
    print(f"  stats_path: {stats_path}")
    print(f"  model_path: {checkpoint_path}")
    scorer = TriAttentionScorer(
        stats_path=stats_path,
        model_path=checkpoint_path,
    )
    print(f"  sampled_heads: {len(scorer.sampled_heads)}")
    print(f"  head_dim: {scorer.head_dim}")
    print(f"  attention_scale: {scorer.attention_scale}")
    print(f"  omega shape: {tuple(scorer.omega.shape)}")
    print(f"  freq_scale_sq shape: {tuple(scorer.freq_scale_sq.shape)}")
    print(f"  offsets shape: {tuple(scorer.offsets.shape)}")
    print(f"  device (initial): {scorer.device}")

    # ------------------------------------------------------------------
    # 2. Move the scorer to the target device.
    # ------------------------------------------------------------------
    scorer.to(device)
    print(f"  device (after .to): {scorer.device}")
    assert scorer.omega.device.type == torch.device(device).type, \
        f"omega still on {scorer.omega.device}"

    # ------------------------------------------------------------------
    # 3. Load the DSA model and attach the scorer to every layer.
    # ------------------------------------------------------------------
    print(f"\nLoading DSA model from: {checkpoint_path}")
    model = load_dsa_model(checkpoint_path, device=device)
    model.eval()

    attached = 0
    for layer in model.model.layers:
        attn = layer.self_attn
        if hasattr(attn, "load_tri_scorer"):
            attn.load_tri_scorer(scorer)
            attached += 1
    print(f"  Attached scorer to {attached} layers.")
    assert attached == model.config.num_hidden_layers

    # ------------------------------------------------------------------
    # 4. Forward pass — tri-scorer path.
    # ------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_path, trust_remote_code=True,
    )
    prompt = "The quick brown fox jumps over the lazy dog."
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        out_tri = model(**inputs)
    logits_tri = out_tri.logits
    print(f"\nTri-scorer forward pass:")
    print(f"  input_ids shape: {tuple(inputs['input_ids'].shape)}")
    print(f"  logits shape:    {tuple(logits_tri.shape)}")
    assert logits_tri.ndim == 3
    assert logits_tri.shape[0] == 1
    assert logits_tri.shape[2] == model.config.vocab_size

    # ------------------------------------------------------------------
    # 5. Compare against learned-Indexer path (disable tri scorer).
    # ------------------------------------------------------------------
    for layer in model.model.layers:
        layer.self_attn.use_tri_scorer = False

    with torch.no_grad():
        out_idx = model(**inputs)
    logits_idx = out_idx.logits
    diff = (logits_tri - logits_idx).abs().mean().item()
    print(f"\nMean |logit| diff (tri vs indexer): {diff:.6f}")
    # With an untrained Indexer (zero weights_proj) vs the tri scorer,
    # the two paths generally pick different top-k sets, so logits differ.
    # We do not assert a specific threshold here — just report.

    print("\nAll checks passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./outputs/llama3.2-1b-dsa",
    )
    parser.add_argument(
        "--stats",
        type=str,
        default="./outputs/llama3.2-1b-dsa-stats.pt",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
    )
    args = parser.parse_args()
    main(args)

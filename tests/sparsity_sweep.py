"""
Sparsity sweep: compare tri-scorer vs random-selection vs dense across a
range of topk values, at one or more sequence lengths.

For each (seq_len, topk) cell we report:
  - MSE of sparse logits vs dense logits
  - Next-token top-1 agreement with dense
for both tri-scorer and uniform-random selection.

Usage
-----
    python tests/sparsity_sweep.py --device mps \
        --seq-lens 1024 2048 \
        --topks 16 32 64 128 256 512
"""
from __future__ import annotations

import argparse
import csv
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


class RandomScorer:
    """Drop-in scorer returning uniformly random top-k indices."""

    def __init__(self, device="cpu", seed=0):
        self.device = torch.device(device)
        self.generator = torch.Generator(device="cpu").manual_seed(seed)

    def to(self, device):
        self.device = torch.device(device)
        return self

    def _random_topk(self, seq_k, topk):
        k = min(topk, seq_k)
        perm = torch.randperm(seq_k, generator=self.generator)[:k]
        return perm.to(self.device)

    def score_tokens(self, layer_idx, k_rot_raw, absolute_positions,
                     current_position, topk):
        return self._random_topk(k_rot_raw.shape[0], topk)

    def score_tokens_with_cache(self, layer_idx, cached_key_states,
                                k_rot_raw_current, cache_position, seq_q, topk):
        return self._random_topk(cached_key_states.shape[2], topk)


def forward_logits(model, input_ids):
    with torch.no_grad():
        return model(input_ids=input_ids).logits


def set_topk(model, k):
    for layer in model.model.layers:
        layer.self_attn.dsa_config.index_topk = k


def attach(model, scorer):
    for layer in model.model.layers:
        layer.self_attn.load_tri_scorer(scorer)


def mse(a, b):
    return (a.float() - b.float()).pow(2).mean().item()


def top1_agreement(a, b):
    return (a.argmax(dim=-1) == b.argmax(dim=-1)).float().mean().item()


def build_prompt(tokenizer, device, seq_len):
    seed_text = (
        "The history of artificial intelligence began in antiquity, with myths, "
        "stories and rumors of artificial beings endowed with intelligence or "
        "consciousness by master craftsmen. The seeds of modern AI were planted "
        "by philosophers who attempted to describe the process of human thinking "
        "as the mechanical manipulation of symbols. "
    )
    prompt = seed_text * max(1, seq_len // 40)
    ids = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=seq_len,
    )["input_ids"].to(device)
    return ids


def main(args):
    device = args.device
    print(f"Loading model from {args.checkpoint} on {device}...")
    model = load_dsa_model(args.checkpoint, device=device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint, trust_remote_code=True,
    )

    tri = TriAttentionScorer(
        stats_path=args.stats, model_path=args.checkpoint,
    ).to(device)
    rand = RandomScorer(device=device, seed=args.seed)

    rows = []

    for seq_len in args.seq_lens:
        input_ids = build_prompt(tokenizer, device, seq_len)
        actual_len = input_ids.shape[1]
        print(f"\n=== seq_len target {seq_len} (actual {actual_len}) ===")

        # Dense baseline: attach tri-scorer, bump topk so everything selected.
        attach(model, tri)
        set_topk(model, actual_len + 1)
        logits_dense = forward_logits(model, input_ids)

        for topk in args.topks:
            if topk >= actual_len:
                continue
            # Tri
            attach(model, tri)
            set_topk(model, topk)
            logits_tri = forward_logits(model, input_ids)
            # Random
            attach(model, rand)
            set_topk(model, topk)
            logits_rand = forward_logits(model, input_ids)

            mse_tri = mse(logits_tri, logits_dense)
            mse_rand = mse(logits_rand, logits_dense)
            agree_tri = top1_agreement(logits_tri, logits_dense)
            agree_rand = top1_agreement(logits_rand, logits_dense)
            ratio = mse_tri / mse_rand if mse_rand > 0 else float("inf")

            row = {
                "seq_len": actual_len,
                "topk": topk,
                "sparsity_pct": 100.0 * topk / actual_len,
                "mse_tri": mse_tri,
                "mse_rand": mse_rand,
                "mse_ratio": ratio,
                "agree_tri": agree_tri,
                "agree_rand": agree_rand,
                "agree_gap": agree_tri - agree_rand,
            }
            rows.append(row)
            print(
                f"  topk={topk:>4}  "
                f"tri MSE={mse_tri:7.4f}  rand MSE={mse_rand:7.4f}  "
                f"ratio={ratio:5.3f}   "
                f"tri top1={agree_tri:.3f}  rand top1={agree_rand:.3f}  "
                f"gap={agree_tri - agree_rand:+.3f}"
            )

    # Write CSV
    if args.output and rows:
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {len(rows)} rows to {args.output}")
    elif args.output:
        print(f"\nNo rows to write (all topk values >= seq_len).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="./outputs/llama3.2-1b-dsa")
    parser.add_argument("--stats", default="./outputs/llama3.2-1b-dsa-stats.pt")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[1024, 2048])
    parser.add_argument("--topks", type=int, nargs="+",
                        default=[16, 32, 64, 128, 256, 512])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="./outputs/sparsity_sweep.csv")
    args = parser.parse_args()
    main(args)

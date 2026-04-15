"""
Perplexity sweep on wikitext-2 test split.

For each (mode, topk) the script computes mean negative log-likelihood
over a fixed set of fixed-length chunks of the wikitext-2 test split,
then reports perplexity.  This is the multi-prompt, lower-variance
counterpart to sparsity_sweep.py.

Modes
-----
  dense       — tri-scorer attached with topk >= seq_len (every key
                selected after causal masking)
  tri         — tri-scorer at the requested topk
  random      — uniform-random top-k positions

Usage
-----
    python tests/perplexity_sweep.py --device mps \
        --seq-len 1024 --num-chunks 32 \
        --topks 64 128 256 512
"""
from __future__ import annotations

import argparse
import csv
import math
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
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer

from transdsa.converter import load_dsa_model
from transdsa.tri_scorer import TriAttentionScorer


class RandomScorer:
    def __init__(self, device="cpu", seed=0):
        self.device = torch.device(device)
        self.generator = torch.Generator(device="cpu").manual_seed(seed)

    def to(self, device):
        self.device = torch.device(device)
        return self

    def _random_topk(self, seq_k, topk):
        k = min(topk, seq_k)
        return torch.randperm(seq_k, generator=self.generator)[:k].to(self.device)

    def score_tokens(self, layer_idx, k_rot_raw, absolute_positions,
                     current_position, topk):
        return self._random_topk(k_rot_raw.shape[0], topk)

    def score_tokens_with_cache(self, layer_idx, cached_key_states,
                                k_rot_raw_current, cache_position, seq_q, topk):
        return self._random_topk(cached_key_states.shape[2], topk)


def attach(model, scorer):
    for layer in model.model.layers:
        layer.self_attn.load_tri_scorer(scorer)


def set_topk(model, k):
    for layer in model.model.layers:
        layer.self_attn.dsa_config.index_topk = k


def chunk_logprob(model, input_ids):
    """Mean per-token NLL on input_ids, shifted by one for next-token prediction."""
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits  # [1, T, V]
    shift_logits = logits[:, :-1, :].contiguous().float()
    shift_labels = input_ids[:, 1:].contiguous()
    nll = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="mean",
    )
    return nll.item()


def build_chunks(tokenizer, device, seq_len, num_chunks):
    """Concatenate wikitext-2 test split, chunk into fixed-length blocks."""
    print("Loading wikitext-2 test split...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(s for s in ds["text"] if s.strip())
    ids = tokenizer(text, return_tensors="pt")["input_ids"][0]
    print(f"  tokenized: {ids.numel():,} tokens; need "
          f"{num_chunks} x {seq_len} = {num_chunks * seq_len:,}")

    if ids.numel() < num_chunks * seq_len:
        actual = ids.numel() // seq_len
        print(f"  WARNING: only {actual} chunks of length {seq_len} fit")
        num_chunks = actual

    chunks = []
    for i in range(num_chunks):
        start = i * seq_len
        block = ids[start:start + seq_len].unsqueeze(0).to(device)
        chunks.append(block)
    return chunks


def eval_mode(model, chunks, label):
    nlls = []
    for j, ids in enumerate(chunks):
        nll = chunk_logprob(model, ids)
        nlls.append(nll)
        if (j + 1) % 8 == 0 or j == len(chunks) - 1:
            avg = sum(nlls) / len(nlls)
            print(f"    [{label}] chunk {j+1}/{len(chunks)}  "
                  f"running mean NLL {avg:.4f}  ppl {math.exp(avg):.2f}")
    mean_nll = sum(nlls) / len(nlls)
    return mean_nll, math.exp(mean_nll)


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

    chunks = build_chunks(tokenizer, device, args.seq_len, args.num_chunks)
    print(f"  Using {len(chunks)} chunks of length {args.seq_len}\n")

    rows = []

    # Dense baseline
    print("Evaluating dense...")
    attach(model, tri)
    set_topk(model, args.seq_len + 1)
    nll_dense, ppl_dense = eval_mode(model, chunks, "dense")
    rows.append({"mode": "dense", "topk": args.seq_len, "nll": nll_dense,
                 "ppl": ppl_dense})

    for topk in args.topks:
        if topk >= args.seq_len:
            continue
        print(f"\nEvaluating tri-scorer (topk={topk})...")
        attach(model, tri)
        set_topk(model, topk)
        nll_tri, ppl_tri = eval_mode(model, chunks, f"tri/{topk}")
        rows.append({"mode": "tri", "topk": topk, "nll": nll_tri, "ppl": ppl_tri})

        print(f"\nEvaluating random (topk={topk})...")
        attach(model, rand)
        set_topk(model, topk)
        nll_rand, ppl_rand = eval_mode(model, chunks, f"rand/{topk}")
        rows.append({"mode": "random", "topk": topk, "nll": nll_rand,
                     "ppl": ppl_rand})

    # Summary
    print("\n" + "=" * 60)
    print(f"Perplexity on wikitext-2 test  (seq_len={args.seq_len}, "
          f"chunks={len(chunks)})")
    print("=" * 60)
    print(f"  {'mode':<8} {'topk':>6}  {'NLL':>8}  {'PPL':>10}")
    for r in rows:
        print(f"  {r['mode']:<8} {r['topk']:>6}  {r['nll']:>8.4f}  "
              f"{r['ppl']:>10.2f}")

    if args.output and rows:
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="./outputs/llama3.2-1b-dsa")
    parser.add_argument("--stats", default="./outputs/llama3.2-1b-dsa-stats.pt")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--num-chunks", type=int, default=32)
    parser.add_argument("--topks", type=int, nargs="+",
                        default=[64, 128, 256, 512])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="./outputs/perplexity_sweep.csv")
    args = parser.parse_args()
    main(args)

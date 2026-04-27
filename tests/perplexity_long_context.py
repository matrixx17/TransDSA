"""
Long-context perplexity sweep on wikitext-2 test split.

Sweeps multiple seq_lens and per-seq-len topk grids in one run, writing
results incrementally to CSV so a Colab disconnect does not lose data.

Usage
-----
    python tests/perplexity_long_context.py --device cuda \
        --seq-lens 4096 8192 --num-chunks 8

Default topk grid per seq_len:
    4096 -> [256, 512, 1024, 2048]
    8192 -> [512, 1024, 2048, 4096]
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


DEFAULT_TOPKS = {
    4096: [256, 512, 1024, 2048],
    8192: [512, 1024, 2048, 4096],
}


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
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits
    shift_logits = logits[:, :-1, :].contiguous().float()
    shift_labels = input_ids[:, 1:].contiguous()
    nll = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="mean",
    )
    return nll.item()


def build_chunks(tokenizer, device, seq_len, num_chunks, cached_ids=None):
    if cached_ids is None:
        print("Loading wikitext-2 test split...")
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join(s for s in ds["text"] if s.strip())
        cached_ids = tokenizer(text, return_tensors="pt")["input_ids"][0]
        print(f"  tokenized: {cached_ids.numel():,} tokens")

    need = num_chunks * seq_len
    if cached_ids.numel() < need:
        actual = cached_ids.numel() // seq_len
        print(f"  WARNING: only {actual} chunks of length {seq_len} fit "
              f"(needed {num_chunks})")
        num_chunks = actual

    chunks = []
    for i in range(num_chunks):
        start = i * seq_len
        block = cached_ids[start:start + seq_len].unsqueeze(0).to(device)
        chunks.append(block)
    return chunks, cached_ids


def eval_mode(model, chunks, label):
    nlls = []
    for j, ids in enumerate(chunks):
        nll = chunk_logprob(model, ids)
        nlls.append(nll)
        avg = sum(nlls) / len(nlls)
        print(f"    [{label}] chunk {j+1}/{len(chunks)}  "
              f"running mean NLL {avg:.4f}  ppl {math.exp(min(avg, 50)):.2f}",
              flush=True)
    mean_nll = sum(nlls) / len(nlls)
    return mean_nll, math.exp(min(mean_nll, 50))


def append_row(path, row, fieldnames):
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


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

    fieldnames = ["seq_len", "mode", "topk", "retention_pct", "nll", "ppl"]
    if args.output and os.path.exists(args.output) and args.fresh:
        os.remove(args.output)

    cached_ids = None
    for seq_len in args.seq_lens:
        topks = args.topks if args.topks else DEFAULT_TOPKS.get(seq_len, [])
        if not topks:
            print(f"No topks for seq_len={seq_len}, skipping")
            continue

        print(f"\n{'#' * 60}\n# seq_len = {seq_len}\n{'#' * 60}")
        chunks, cached_ids = build_chunks(
            tokenizer, device, seq_len, args.num_chunks, cached_ids=cached_ids,
        )
        if not chunks:
            print(f"No chunks fit at seq_len={seq_len}, skipping")
            continue
        print(f"Using {len(chunks)} chunks of length {seq_len}\n")

        # Dense
        print(f"Evaluating dense (seq_len={seq_len})...")
        attach(model, tri)
        set_topk(model, seq_len + 1)
        nll, ppl = eval_mode(model, chunks, "dense")
        row = {"seq_len": seq_len, "mode": "dense", "topk": seq_len,
               "retention_pct": 100.0, "nll": nll, "ppl": ppl}
        append_row(args.output, row, fieldnames)

        for topk in topks:
            if topk >= seq_len:
                continue
            ret = 100.0 * topk / seq_len

            print(f"\nEvaluating tri (seq_len={seq_len}, topk={topk})...")
            attach(model, tri)
            set_topk(model, topk)
            nll, ppl = eval_mode(model, chunks, f"tri/{topk}")
            append_row(args.output, {
                "seq_len": seq_len, "mode": "tri", "topk": topk,
                "retention_pct": ret, "nll": nll, "ppl": ppl,
            }, fieldnames)

            print(f"\nEvaluating random (seq_len={seq_len}, topk={topk})...")
            attach(model, rand)
            set_topk(model, topk)
            nll, ppl = eval_mode(model, chunks, f"rand/{topk}")
            append_row(args.output, {
                "seq_len": seq_len, "mode": "random", "topk": topk,
                "retention_pct": ret, "nll": nll, "ppl": ppl,
            }, fieldnames)

    print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="./outputs/llama3.2-1b-dsa")
    parser.add_argument("--stats", default="./outputs/llama3.2-1b-dsa-stats.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[4096, 8192])
    parser.add_argument("--num-chunks", type=int, default=8)
    parser.add_argument("--topks", type=int, nargs="+", default=None,
                        help="Override default per-seq-len topk grid.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output",
                        default="./outputs/perplexity_long_context.csv")
    parser.add_argument("--fresh", action="store_true",
                        help="Delete existing output CSV before running.")
    args = parser.parse_args()
    main(args)

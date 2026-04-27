"""
RAG-style evaluation for the TriAttention scorer.

For each hand-written (document, question, answer) triple, we build a
context where the document is embedded inside wikitext filler:

    [filler_prefix] [DOC_START] [document] [DOC_END] [question] [answer]

We then measure, for dense / tri@topk / random@topk:

  1. Answer NLL  — cross-entropy on answer tokens only (the rest of the
     sequence is masked from the loss).
  2. Doc-hit rate — fraction of the scorer's top-k positions that fall
     inside the document span [doc_start, doc_end), aggregated as a
     mean over layers.  Random's expected hit rate is doc_len / seq_len.

Implementation note: the scorer is called once per layer per forward,
with current_pos at the *last* query position.  We therefore measure
hit rate at the position predicting the final answer token — i.e. the
moment with the largest context.

Usage:
    python tests/rag_eval.py --device cuda \
        --num-triples 10 --seq-len 4096 --topks 512 1024 2048
"""
from __future__ import annotations

import argparse
import csv
import json
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


DOC_START = "\n\n--- DOCUMENT START ---\n"
DOC_END = "\n--- DOCUMENT END ---\n\n"


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


class CapturingScorer:
    """Wraps any scorer; records last topk indices per layer call."""
    def __init__(self, inner):
        self.inner = inner
        self.last_topk = {}

    def to(self, device):
        self.inner.to(device)
        return self

    def reset(self):
        self.last_topk = {}

    def score_tokens(self, layer_idx, k_rot_raw, absolute_positions,
                     current_position, topk):
        out = self.inner.score_tokens(
            layer_idx, k_rot_raw, absolute_positions, current_position, topk,
        )
        self.last_topk[layer_idx] = out.detach().cpu()
        return out

    def score_tokens_with_cache(self, layer_idx, cached_key_states,
                                k_rot_raw_current, cache_position, seq_q, topk):
        out = self.inner.score_tokens_with_cache(
            layer_idx, cached_key_states, k_rot_raw_current,
            cache_position, seq_q, topk,
        )
        self.last_topk[layer_idx] = out.detach().cpu()
        return out


def attach(model, scorer):
    for layer in model.model.layers:
        layer.self_attn.load_tri_scorer(scorer)


def set_topk(model, k):
    for layer in model.model.layers:
        layer.self_attn.dsa_config.index_topk = k


def load_filler_text(tokenizer, min_tokens):
    print("Loading wikitext-2 train filler...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(s for s in ds["text"] if s.strip())
    ids = tokenizer(text, return_tensors="pt")["input_ids"][0]
    if ids.numel() < min_tokens:
        raise RuntimeError(f"filler corpus too small: have {ids.numel()}")
    print(f"  filler tokens available: {ids.numel():,}")
    return ids


def build_input(tokenizer, filler_ids, triple, seq_len, filler_offset):
    """Returns (input_ids[1, seq_len], doc_start, doc_end, ans_start, ans_end)."""
    doc_start_ids = tokenizer(DOC_START, add_special_tokens=False,
                              return_tensors="pt")["input_ids"][0]
    doc_end_ids = tokenizer(DOC_END, add_special_tokens=False,
                            return_tensors="pt")["input_ids"][0]
    doc_ids = tokenizer(triple["document"], add_special_tokens=False,
                        return_tensors="pt")["input_ids"][0]
    q_ids = tokenizer("\n" + triple["question"] + "\n",
                      add_special_tokens=False,
                      return_tensors="pt")["input_ids"][0]
    a_ids = tokenizer(triple["answer"], add_special_tokens=False,
                      return_tensors="pt")["input_ids"][0]

    fixed = (doc_start_ids.numel() + doc_ids.numel() + doc_end_ids.numel()
             + q_ids.numel() + a_ids.numel())
    filler_len = seq_len - fixed
    if filler_len < 64:
        raise RuntimeError(
            f"seq_len={seq_len} too small for triple "
            f"(needs at least {fixed + 64})")

    if filler_offset + filler_len > filler_ids.numel():
        filler_offset = 0
    filler_chunk = filler_ids[filler_offset:filler_offset + filler_len]

    parts = [filler_chunk, doc_start_ids, doc_ids, doc_end_ids, q_ids, a_ids]
    full = torch.cat(parts, dim=0)
    assert full.numel() == seq_len, (full.numel(), seq_len)

    doc_start = filler_chunk.numel() + doc_start_ids.numel()
    doc_end = doc_start + doc_ids.numel()
    ans_start = doc_end + doc_end_ids.numel() + q_ids.numel()
    ans_end = seq_len
    return full.unsqueeze(0), doc_start, doc_end, ans_start, ans_end


def answer_nll(logits, input_ids, ans_start, ans_end):
    """NLL over predictions of input_ids[ans_start:ans_end]."""
    pred_logits = logits[0, ans_start - 1:ans_end - 1, :].float()
    labels = input_ids[0, ans_start:ans_end]
    return F.cross_entropy(pred_logits, labels, reduction="mean").item()


def doc_hit_rate(last_topk, doc_start, doc_end):
    if not last_topk:
        return float("nan")
    rates = []
    for _layer, idx in last_topk.items():
        n = idx.numel()
        if n == 0:
            continue
        in_doc = ((idx >= doc_start) & (idx < doc_end)).sum().item()
        rates.append(in_doc / n)
    return sum(rates) / len(rates)


def append_row(path, row, fieldnames):
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def run_one(model, scorer, capturing, mode, topk, input_ids,
            ans_start, ans_end, doc_start, doc_end):
    capturing.reset()
    attach(model, capturing)
    set_topk(model, topk)
    with torch.no_grad():
        out = model(input_ids=input_ids)
    nll = answer_nll(out.logits, input_ids, ans_start, ans_end)
    hit = doc_hit_rate(capturing.last_topk, doc_start, doc_end)
    return nll, hit


def main(args):
    device = args.device
    print(f"Loading model from {args.checkpoint} on {device}...")
    model = load_dsa_model(args.checkpoint, device=device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint, trust_remote_code=True,
    )

    with open(args.triples) as f:
        triples = json.load(f)
    if args.num_triples:
        triples = triples[:args.num_triples]
    print(f"Loaded {len(triples)} triples")

    tri = TriAttentionScorer(
        stats_path=args.stats, model_path=args.checkpoint,
    ).to(device)
    rand = RandomScorer(device=device, seed=args.seed)
    cap_tri = CapturingScorer(tri)
    cap_rand = CapturingScorer(rand)

    filler_ids = load_filler_text(tokenizer, min_tokens=args.seq_len * 2)

    fieldnames = ["triple_idx", "topic", "mode", "topk", "seq_len",
                  "doc_len", "doc_frac", "answer_nll", "doc_hit_rate_mean"]
    if args.output and os.path.exists(args.output) and args.fresh:
        os.remove(args.output)

    for ti, triple in enumerate(triples):
        offset = (ti * 4096) % max(filler_ids.numel() - args.seq_len, 1)
        input_ids, doc_s, doc_e, ans_s, ans_e = build_input(
            tokenizer, filler_ids, triple, args.seq_len, offset,
        )
        input_ids = input_ids.to(device)
        doc_len = doc_e - doc_s
        doc_frac = doc_len / args.seq_len
        ans_len = ans_e - ans_s
        print(f"\n[{ti+1}/{len(triples)}] {triple['topic']}: "
              f"doc=[{doc_s}:{doc_e}] ({doc_len} tok, {doc_frac:.1%}), "
              f"answer={ans_len} tok")

        # Dense
        nll, hit = run_one(model, tri, cap_tri, "dense", args.seq_len + 1,
                           input_ids, ans_s, ans_e, doc_s, doc_e)
        print(f"  dense       NLL {nll:.3f}  hit {hit:.3f}")
        append_row(args.output, {
            "triple_idx": ti, "topic": triple["topic"], "mode": "dense",
            "topk": args.seq_len, "seq_len": args.seq_len,
            "doc_len": doc_len, "doc_frac": doc_frac,
            "answer_nll": nll, "doc_hit_rate_mean": hit,
        }, fieldnames)

        for topk in args.topks:
            nll_t, hit_t = run_one(model, tri, cap_tri, "tri", topk,
                                   input_ids, ans_s, ans_e, doc_s, doc_e)
            print(f"  tri/{topk:<5}   NLL {nll_t:.3f}  hit {hit_t:.3f}  "
                  f"(rand baseline {doc_frac:.3f})")
            append_row(args.output, {
                "triple_idx": ti, "topic": triple["topic"], "mode": "tri",
                "topk": topk, "seq_len": args.seq_len,
                "doc_len": doc_len, "doc_frac": doc_frac,
                "answer_nll": nll_t, "doc_hit_rate_mean": hit_t,
            }, fieldnames)

            nll_r, hit_r = run_one(model, rand, cap_rand, "random", topk,
                                   input_ids, ans_s, ans_e, doc_s, doc_e)
            print(f"  rand/{topk:<5}  NLL {nll_r:.3f}  hit {hit_r:.3f}")
            append_row(args.output, {
                "triple_idx": ti, "topic": triple["topic"], "mode": "random",
                "topk": topk, "seq_len": args.seq_len,
                "doc_len": doc_len, "doc_frac": doc_frac,
                "answer_nll": nll_r, "doc_hit_rate_mean": hit_r,
            }, fieldnames)

    print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="./outputs/llama3.2-1b-dsa")
    parser.add_argument("--stats", default="./outputs/llama3.2-1b-dsa-stats.pt")
    parser.add_argument("--triples", default="./tests/rag_triples.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--num-triples", type=int, default=0,
                        help="0 = use all triples in the file.")
    parser.add_argument("--topks", type=int, nargs="+",
                        default=[512, 1024, 2048])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="./outputs/rag_eval.csv")
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()
    main(args)

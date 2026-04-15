"""
transdsa/calibrate.py

Calibration script for TriAttention-based token scoring in TransDSA.

Runs calibration samples through the DSA model, captures the pre-RoPE
positional query component (q_rot) from each attention layer via forward
hooks, and computes per-head frequency statistics.  These stats are used
by TriAttentionScorer to deterministically score cached tokens without
any learned Indexer parameters.

Usage
-----
    python -m transdsa.calibrate \
        --checkpoint ./outputs/llama3.2-1b-dsa \
        --output ./outputs/llama3.2-1b-dsa-stats.pt \
        --num-samples 128 \
        --device cpu
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Tuple

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Ensure transdsa is importable when run as a script
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transdsa.converter import load_dsa_model
from transdsa.modeling import DSAAttention


# ---------------------------------------------------------------------------
# Complex-pair conversion (copied from pruning_utils for self-containment)
# ---------------------------------------------------------------------------

def _to_complex_pairs(tensor: torch.Tensor, *, style: str = "half") -> torch.Tensor:
    """Convert a real tensor to complex pairs using half-split convention."""
    real_dtype = (
        torch.float32
        if tensor.dtype in (torch.bfloat16, torch.float16)
        else tensor.dtype
    )
    tensor_real = tensor.to(dtype=real_dtype)
    if style == "interleaved":
        real = tensor_real[..., ::2].contiguous()
        imag = tensor_real[..., 1::2].contiguous()
        return torch.complex(real, imag)
    freq_count = tensor.shape[-1] // 2
    real = tensor_real[..., :freq_count].contiguous()
    imag = tensor_real[..., freq_count:].contiguous()
    return torch.complex(real, imag)


# ---------------------------------------------------------------------------
# Layer discovery
# ---------------------------------------------------------------------------

def _find_dsa_layers(model: torch.nn.Module) -> List[Tuple[int, DSAAttention]]:
    """Return (layer_idx, DSAAttention) pairs in layer order."""
    backbone = getattr(model, "model", model)
    layer_list = getattr(backbone, "layers", None)
    if layer_list is None:
        raise RuntimeError("Cannot locate model.model.layers.")
    result = []
    for i, layer_module in enumerate(layer_list):
        attn = getattr(layer_module, "self_attn", None)
        if isinstance(attn, DSAAttention):
            result.append((i, attn))
    if not result:
        raise RuntimeError("No DSAAttention layers found. Was the model converted?")
    return result


# ---------------------------------------------------------------------------
# Main calibration
# ---------------------------------------------------------------------------

def calibrate(
    checkpoint_path: str,
    output_path: str,
    num_samples: int = 128,
    max_length: int = 2048,
    device: str = "cpu",
) -> None:
    device_obj = torch.device(device)
    dtype = torch.bfloat16

    # --- Load DSA model ---
    print(f"Loading DSA model from: {checkpoint_path}", file=sys.stderr)
    model = load_dsa_model(checkpoint_path, device=device)
    model.eval()

    config = model.config
    num_heads = config.num_attention_heads
    qk_head_dim = config.qk_head_dim              # 192
    qk_rope_head_dim = config.qk_rope_head_dim    # 64
    qk_nope_head_dim = config.qk_nope_head_dim    # 128
    num_layers = config.num_hidden_layers

    # --- Load tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_path, trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Load wikitext-2 ---
    print(f"Loading wikitext-2 ({num_samples} samples)...", file=sys.stderr)
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    # Filter out empty / short lines and take the first num_samples
    texts = []
    for row in dataset:
        text = row["text"].strip()
        if len(text) > 50:  # skip empty/header lines
            texts.append(text)
            if len(texts) >= num_samples:
                break
    print(f"  Collected {len(texts)} non-empty samples.", file=sys.stderr)

    # --- Discover DSA layers ---
    dsa_layers = _find_dsa_layers(model)
    print(f"  Found {len(dsa_layers)} DSA attention layers.", file=sys.stderr)

    # --- Accumulators: per (layer, head) running sums ---
    # q_rot is (bsz, num_heads, seq, qk_rope_head_dim=64)
    # After to_complex_pairs: (seq, freq_count) where freq_count = 32
    freq_count = qk_rope_head_dim // 2
    q_complex_sum: Dict[Tuple[int, int], torch.Tensor] = {}
    q_abs_sum: Dict[Tuple[int, int], torch.Tensor] = {}
    token_counts: Dict[Tuple[int, int], int] = {}

    for layer_idx, _ in dsa_layers:
        for head_idx in range(num_heads):
            key = (layer_idx, head_idx)
            q_complex_sum[key] = torch.zeros(freq_count, dtype=torch.complex64)
            q_abs_sum[key] = torch.zeros(freq_count, dtype=torch.float32)
            token_counts[key] = 0

    # --- Register forward hooks to capture pre-RoPE q_rot ---
    # We use a forward_pre_hook with_kwargs=True.  Inside the hook we
    # replicate only the Q projection + split (cheap) to extract q_rot
    # before RoPE is applied.  This avoids running the full KV path.
    captured_q_rot: Dict[int, torch.Tensor] = {}

    def _make_hook(layer_idx: int):
        def hook_fn(module: DSAAttention, args, kwargs):
            # Extract hidden_states from the forward signature
            hidden_states = args[0] if args else kwargs.get("hidden_states")
            if hidden_states is None:
                return

            bsz, seq_q, _ = hidden_states.shape

            # Replicate Q projection (q_lora_rank is None for this model)
            with torch.no_grad():
                if module.q_proj is not None:
                    q = module.q_proj(hidden_states)
                else:
                    qr = module.q_a_proj(hidden_states)
                    if module.q_a_layernorm is not None:
                        qr = module.q_a_layernorm(qr)
                    q = module.q_b_proj(qr)

                q = q.view(bsz, seq_q, module.num_heads, module.qk_head_dim)
                q = q.transpose(1, 2)
                # Split off the RoPE portion (pre-RoPE)
                _, q_rot = torch.split(
                    q, [module.qk_nope_head_dim, module.qk_rope_head_dim], dim=-1
                )
                # q_rot: (bsz, num_heads, seq_q, qk_rope_head_dim)
                captured_q_rot[layer_idx] = q_rot.detach().cpu()

        return hook_fn

    handles = []
    for layer_idx, attn_module in dsa_layers:
        h = attn_module.register_forward_pre_hook(
            _make_hook(layer_idx), with_kwargs=True,
        )
        handles.append(h)

    # --- Run samples through the model ---
    print(f"Running {len(texts)} calibration samples...", file=sys.stderr)
    for sample_idx, text in enumerate(texts):
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        ).to(device_obj)

        with torch.no_grad():
            model(**inputs)

        # Accumulate per-head statistics from this sample
        for layer_idx, _ in dsa_layers:
            q_rot = captured_q_rot.get(layer_idx)
            if q_rot is None:
                continue
            # q_rot: (1, num_heads, seq, qk_rope_head_dim)
            seq_len = q_rot.shape[2]
            for head_idx in range(num_heads):
                q_head = q_rot[0, head_idx]  # (seq, qk_rope_head_dim)
                q_complex = _to_complex_pairs(q_head, style="half")
                # q_complex: (seq, freq_count)
                key = (layer_idx, head_idx)
                q_complex_sum[key] += q_complex.sum(dim=0)
                q_abs_sum[key] += q_complex.abs().sum(dim=0)
                token_counts[key] += seq_len

        captured_q_rot.clear()

        if (sample_idx + 1) % 10 == 0:
            print(f"  Processed {sample_idx + 1}/{len(texts)} samples", file=sys.stderr)

    # Remove hooks
    for h in handles:
        h.remove()

    # --- Compute means ---
    print("Computing per-head means...", file=sys.stderr)
    sampled_heads: List[Tuple[int, int]] = []
    stats_dict: Dict[str, Dict[str, torch.Tensor]] = {}

    for layer_idx, _ in dsa_layers:
        for head_idx in range(num_heads):
            key = (layer_idx, head_idx)
            count = token_counts[key]
            if count == 0:
                continue
            q_mean_complex = q_complex_sum[key] / count
            q_abs_mean = q_abs_sum[key] / count

            stat_key = f"layer{layer_idx:02d}_head{head_idx:02d}"
            stats_dict[stat_key] = {
                "q_mean_real": q_mean_complex.real.cpu(),
                "q_mean_imag": q_mean_complex.imag.cpu(),
                "q_abs_mean": q_abs_mean.cpu(),
            }
            sampled_heads.append((layer_idx, head_idx))

    # --- Build metadata ---
    metadata = {
        "num_traces": len(texts),
        "head_dim": qk_rope_head_dim,
        "dtype": "bfloat16",
        "rope_style": "half",
        "sampled_heads": [[int(l), int(h)] for l, h in sampled_heads],
    }

    payload = {
        "metadata": metadata,
        "stats": stats_dict,
    }

    # --- Save ---
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save(payload, output_path)
    print(
        f"Saved stats to {output_path} "
        f"({len(sampled_heads)} heads, {len(texts)} samples)",
        file=sys.stderr,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calibrate TriAttention frequency stats for TransDSA."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./outputs/llama3.2-1b-dsa",
        help="Path to the DSA-converted model checkpoint.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./outputs/llama3.2-1b-dsa-stats.pt",
        help="Output .pt file path for stats.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=128,
        help="Number of wikitext-2 samples to calibrate on (default: 128).",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Maximum token length per sample (default: 2048).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help='Device (e.g. "cpu", "cuda", "mps").',
    )
    args = parser.parse_args()
    calibrate(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        num_samples=args.num_samples,
        max_length=args.max_length,
        device=args.device,
    )

"""
transdsa/converter.py

Converts a pretrained MLA model (BarraHome/llama3_2-1B-deepseek) into a
DSA model by grafting an Indexer onto every MLA attention layer.

Usage
-----
python transdsa/converter.py \
    --model-path BarraHome/llama3_2-1B-deepseek \
    --output-path ./outputs/llama3.2-1b-dsa \
    --index-n-heads 8 \
    --index-head-dim 64 \
    --index-topk 256
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch
import transformers
import transformers.utils

# ---------------------------------------------------------------------------
# Compatibility shim: LossKwargs was present in transformers 4.x but was
# removed in 5.x.  The BarraHome/llama3_2-1B-deepseek model's remote code
# imports it (modeling_llamamla.py line 9) even though it never uses it.
# Injecting a no-op TypedDict before the model is loaded avoids the
# ImportError without modifying any cached HuggingFace files or upgrading
# transformers (which would break other 5.x APIs the model relies on).
# ---------------------------------------------------------------------------
if not hasattr(transformers.utils, "LossKwargs"):
    from typing import TypedDict
    class LossKwargs(TypedDict, total=False):
        pass
    # Injecting into the module object is sufficient — Python's import system
    # resolves `from transformers.utils import LossKwargs` against the live
    # module dict, so this one assignment covers both import forms.
    transformers.utils.LossKwargs = LossKwargs

from transformers import AutoModelForCausalLM, AutoTokenizer

# Make sure the transdsa package is importable when run as a script from the
# repo root or from inside the transdsa/ directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transdsa.modeling import DSAAttention, DSAConfig


# ---------------------------------------------------------------------------
# MLA layer detection
# ---------------------------------------------------------------------------

def _is_mla_layer(module: torch.nn.Module) -> bool:
    """
    Return True if `module` looks like an MLA attention layer.

    We identify MLA layers by the presence of the KV compression projections
    (kv_a_proj_with_mqa and kv_b_proj) which are unique to MLA and not
    present in standard MHA/GQA.  The q side may or may not be LoRA-split.
    """
    return (
        hasattr(module, "kv_a_proj_with_mqa")
        and hasattr(module, "kv_b_proj")
    )


def find_mla_layers(model: torch.nn.Module) -> list[tuple[str, torch.nn.Module]]:
    """
    Walk the model and return (full_name, module) for every MLA attention layer.
    """
    found = []
    for name, module in model.named_modules():
        if _is_mla_layer(module):
            found.append((name, module))
    return found


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def convert_model(
    model: torch.nn.Module,
    dsa_config: DSAConfig,
) -> tuple[torch.nn.Module, int]:
    """
    Walk `model` in-place and replace every MLA attention layer with a
    DSAAttention that wraps it.

    The MLA weights are never copied — DSAAttention holds references to the
    original Parameter objects.  Only Indexer parameters are new.

    Args:
        model      : The loaded HuggingFace MLA model.
        dsa_config : Indexer hyperparameters.

    Returns:
        (model, n_converted) — the same model object with layers replaced,
        and the number of layers that were converted.
    """
    mla_layers = find_mla_layers(model)

    if not mla_layers:
        raise RuntimeError(
            "No MLA attention layers found in the model.  "
            "Make sure the model was loaded with trust_remote_code=True and "
            "is a TransMLA-converted model (kv_a_proj_with_mqa / kv_b_proj "
            "must be present on each attention module)."
        )

    print(f"Found {len(mla_layers)} MLA attention layer(s) to convert.")

    for full_name, mla_module in mla_layers:
        # Build a DSAAttention that wraps this MLA layer.
        dsa_attn = DSAAttention(mla=mla_module, dsa_config=dsa_config)

        # Navigate to the parent module so we can replace the child attribute.
        # full_name looks like "model.layers.0.self_attn".
        parts = full_name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        child_attr = parts[-1]

        setattr(parent, child_attr, dsa_attn)
        print(f"  Converted: {full_name}")

    return model, len(mla_layers)


# ---------------------------------------------------------------------------
# Config serialisation
# ---------------------------------------------------------------------------

def _update_saved_config(output_path: str, dsa_config: DSAConfig) -> None:
    """
    Read the config.json saved by model.save_pretrained() and add the DSA
    fields so the saved checkpoint is self-describing.
    """
    config_path = os.path.join(output_path, "config.json")
    with open(config_path) as f:
        cfg = json.load(f)

    cfg["index_n_heads"]      = dsa_config.index_n_heads
    cfg["index_head_dim"]     = dsa_config.index_head_dim
    cfg["index_topk"]         = dsa_config.index_topk
    cfg["index_rope_head_dim"] = dsa_config.index_rope_head_dim
    cfg["transdsa_converted"] = True

    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"Updated {config_path} with DSA config fields.")


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

def sanity_check(model: torch.nn.Module, tokenizer, device: str) -> None:
    """
    Run a single forward pass with a short dummy input and verify output shape.
    """
    print("\nRunning sanity check forward pass...")
    model.eval()

    dummy_text = "Hello, world!"
    inputs = tokenizer(dummy_text, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]     # (1, seq_len)
    expected_vocab = model.config.vocab_size

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits             # (1, seq_len, vocab_size)
    assert logits.ndim == 3, f"Expected 3-D logits, got shape {logits.shape}"
    assert logits.shape[0] == 1, f"Batch dim mismatch: {logits.shape}"
    assert logits.shape[1] == input_ids.shape[1], (
        f"Sequence length mismatch: logits {logits.shape[1]} vs "
        f"input {input_ids.shape[1]}"
    )
    assert logits.shape[2] == expected_vocab, (
        f"Vocab size mismatch: {logits.shape[2]} vs {expected_vocab}"
    )

    print(
        f"  Input shape : {tuple(input_ids.shape)}\n"
        f"  Output shape: {tuple(logits.shape)}\n"
        f"  Sanity check passed."
    )


# ---------------------------------------------------------------------------
# DSA-aware loader
# ---------------------------------------------------------------------------

def load_dsa_model(
    checkpoint_path: str,
    device: str = "cpu",
) -> torch.nn.Module:
    """
    Reload a checkpoint saved by convert_model() as a fully DSA-aware model.

    Steps
    -----
    1. Load the base BarraHome MLA architecture via AutoModelForCausalLM
       (trust_remote_code=True) so all MLA layer slots exist.
    2. Run convert_model() to replace every MLA self_attn with DSAAttention,
       creating the Indexer parameter slots.
    3. Load the full state dict from the checkpoint (which contains both the
       original MLA weights *and* the saved indexer.* weights) into the
       now-DSA-aware model.

    Args:
        checkpoint_path : Path to the directory saved by convert_model().
        device          : Device string passed to device_map.

    Returns:
        The converted model with all weights (MLA + Indexer) loaded.
    """
    import json

    # Read DSA hyperparameters from the augmented config.json
    config_path = os.path.join(checkpoint_path, "config.json")
    with open(config_path) as f:
        cfg = json.load(f)

    dsa_config = DSAConfig(
        index_n_heads=cfg["index_n_heads"],
        index_head_dim=cfg["index_head_dim"],
        index_topk=cfg["index_topk"],
        index_rope_head_dim=cfg["index_rope_head_dim"],
    )

    # Step 1: Load base MLA architecture (no DSA weights yet)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=device,
        ignore_mismatched_sizes=False,
    )

    # Step 2: Inject DSAAttention wrappers (creates Indexer parameter slots)
    model, _ = convert_model(model, dsa_config)

    # Step 3: Load the full checkpoint state dict (MLA + indexer.*) into the
    # now-DSA-aware model.  Use strict=False because the checkpoint may
    # contain keys for both the original MLA names and the indexer — all
    # should match after conversion.
    from safetensors.torch import load_file as safetensors_load
    import glob as _glob

    st_files = _glob.glob(os.path.join(checkpoint_path, "*.safetensors"))
    if st_files:
        state_dict = {}
        for sf in sorted(st_files):
            state_dict.update(safetensors_load(sf, device=device))
    else:
        pt_file = os.path.join(checkpoint_path, "pytorch_model.bin")
        state_dict = torch.load(pt_file, map_location=device)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    # lm_head.weight is absent from the checkpoint when the model uses tied
    # embeddings (lm_head shares model.embed_tokens.weight).  That is expected
    # and safe to ignore; PyTorch resolves the tie at runtime.
    EXPECTED_MISSING = {"lm_head.weight"}
    truly_missing = [
        k for k in missing
        if "indexer" not in k and k not in EXPECTED_MISSING
    ]
    if truly_missing:
        raise RuntimeError(
            f"load_dsa_model: missing non-indexer keys in checkpoint:\n"
            + "\n".join(f"  {k}" for k in truly_missing)
        )
    if unexpected:
        raise RuntimeError(
            f"load_dsa_model: unexpected keys in checkpoint:\n"
            + "\n".join(f"  {k}" for k in unexpected)
        )

    # Step 4: Move newly created Indexer parameters to the target device.
    # convert_model() creates Indexer modules on CPU (PyTorch default).
    # The checkpoint doesn't contain indexer weights (they're untrained),
    # so load_state_dict leaves them on CPU.  Ensure the full model —
    # including all Indexer submodules — is on the correct device.
    device_obj = torch.device(device)
    if device_obj.type != "cpu":
        model = model.to(device_obj)

    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    device = args.device

    # ------------------------------------------------------------------
    # 1. Load the pretrained MLA model
    # ------------------------------------------------------------------
    print(f"\nLoading model from: {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=device,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )

    print(f"Model class   : {type(model).__name__}")
    print(f"Config type   : {type(model.config).__name__}")
    print(f"Hidden size   : {model.config.hidden_size}")
    print(f"Num layers    : {model.config.num_hidden_layers}")
    print(f"q_lora_rank   : {model.config.q_lora_rank}")
    print(f"kv_lora_rank  : {model.config.kv_lora_rank}")
    print(f"qk_rope_head_dim: {model.config.qk_rope_head_dim}")

    # ------------------------------------------------------------------
    # 2. Build DSAConfig
    # ------------------------------------------------------------------
    dsa_config = DSAConfig.from_mla_config(
        mla_config=model.config,
        index_n_heads=args.index_n_heads,
        index_head_dim=args.index_head_dim,
        index_topk=args.index_topk,
    )
    print(f"\nDSAConfig: {dsa_config}")

    # ------------------------------------------------------------------
    # 3. Convert: graft Indexers onto every MLA attention layer
    # ------------------------------------------------------------------
    print("\nConverting MLA → DSA...")
    model, n_converted = convert_model(model, dsa_config)
    print(f"Converted {n_converted} layer(s).")

    # ------------------------------------------------------------------
    # 4. Quick parameter summary
    # ------------------------------------------------------------------
    total_params = sum(p.numel() for p in model.parameters())
    indexer_params = sum(
        p.numel()
        for name, p in model.named_parameters()
        if "indexer" in name
    )
    print(f"\nParameter summary:")
    print(f"  Total parameters : {total_params:,}")
    print(f"  Indexer (new)    : {indexer_params:,}  "
          f"({100 * indexer_params / total_params:.2f}% of total)")

    # ------------------------------------------------------------------
    # 5. Sanity check
    # ------------------------------------------------------------------
    sanity_check(model, tokenizer, device)

    # ------------------------------------------------------------------
    # 6. Save
    # ------------------------------------------------------------------
    print(f"\nSaving converted model to: {args.output_path}")
    os.makedirs(args.output_path, exist_ok=True)

    # save_pretrained will only save modules that are proper nn.Module children
    # of the HF model.  DSAAttention's borrowed MLA weights are stored as
    # attributes pointing to the original Parameters, which are still children
    # of the original layer objects — those layers no longer exist in the tree
    # once replaced, so their parameters would be lost.
    #
    # To avoid this: we ensure all parameter tensors are reachable from the
    # model's module tree by checking that the state_dict is complete before
    # saving.  Because DSAAttention stores references to the MLA Parameter
    # objects as plain nn.Module sub-modules (self.wq_a = mla.q_a_proj etc.),
    # PyTorch's parameter traversal will find them correctly.
    model.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)

    # Augment the saved config.json with DSA fields
    _update_saved_config(args.output_path, dsa_config)

    print("\nDone.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a TransMLA model to DeepSeek Sparse Attention (DSA)."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="BarraHome/llama3_2-1B-deepseek",
        help="HuggingFace model ID or local path of the MLA model to convert.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="./outputs/llama3.2-1b-dsa",
        help="Directory to save the converted model.",
    )
    parser.add_argument(
        "--index-n-heads",
        type=int,
        default=8,
        help="Number of Indexer attention heads (default: 8).",
    )
    parser.add_argument(
        "--index-head-dim",
        type=int,
        default=64,
        help="Per-head dimension of Indexer queries/keys (default: 64).",
    )
    parser.add_argument(
        "--index-topk",
        type=int,
        default=256,
        help="Number of tokens the Indexer selects per query position (default: 256).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help='Device for loading and running the model (e.g. "cpu", "cuda", "mps").',
    )

    args = parser.parse_args()
    main(args)

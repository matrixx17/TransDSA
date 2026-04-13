import sys
import os

import transformers

# LossKwargs shim — must run before any HF model is loaded.
if not hasattr(transformers.utils, "LossKwargs"):
    from typing import TypedDict
    class LossKwargs(TypedDict, total=False):
        pass
    transformers.utils.LossKwargs = LossKwargs

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from transdsa.converter import load_dsa_model

# ---------------------------------------------------------------------------
# Load models
# ---------------------------------------------------------------------------
print("Loading original MLA model...")
original = AutoModelForCausalLM.from_pretrained(
    "BarraHome/llama3_2-1B-deepseek",
    trust_remote_code=True,
    dtype=torch.bfloat16,
    device_map="cpu",
)

print("Loading converted DSA model...")
converted = load_dsa_model("./outputs/llama3.2-1b-dsa", device="cpu")

# ---------------------------------------------------------------------------
# Test 1: MLA weight integrity
# Exact parameter names from BarraHome/llama3_2-1B-deepseek layers.0:
#   q_proj, kv_a_proj_with_mqa, kv_b_proj, o_proj
# ---------------------------------------------------------------------------
print("\n=== Test 1: MLA weight integrity ===")
mla_params = ["q_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"]
all_ok = True
orig_params = dict(original.named_parameters())
conv_params = dict(converted.named_parameters())

for name in mla_params:
    for layer_idx in [0, 7, 15]:
        key = f"model.layers.{layer_idx}.self_attn.{name}.weight"
        orig_w = orig_params[key]
        conv_w = conv_params[key]
        ok = torch.equal(orig_w, conv_w)
        print(f"  layer {layer_idx} {name}: {'OK' if ok else 'MISMATCH'}")
        if not ok:
            all_ok = False
print(f"Weight integrity: {'PASSED' if all_ok else 'FAILED'}")

# ---------------------------------------------------------------------------
# Test 2: Indexer exists with correct shapes
# wk:           (index_head_dim=64, hidden_size=2048)
# weights_proj: (index_n_heads=8,   hidden_size=2048)
# wq_a:         (512, 2048)   — present because q_lora_rank is None
# wq_b:         (n_heads*head_dim=512, 512)
# ---------------------------------------------------------------------------
print("\n=== Test 2: Indexer shapes ===")
idx = converted.model.layers[0].self_attn.indexer
print(f"  wq_a shape:         {idx.wq_a.weight.shape}")         # (512, 2048)
print(f"  wq_b shape:         {idx.wq_b.weight.shape}")         # (512, 512)
print(f"  wk shape:           {idx.wk.weight.shape}")           # (64, 2048)
print(f"  k_norm weight:      {idx.k_norm.weight.shape}")       # (64,)
print(f"  weights_proj shape: {idx.weights_proj.weight.shape}") # (8, 2048)

# ---------------------------------------------------------------------------
# Test 3: Index mask is actually applied
# The indexer's topk defaults to 256, which is larger than any short test
# sequence, so every position would be selected and the mask would have no
# effect.  We temporarily reduce topk to 1 (select only one key per query)
# to force the mask to block most positions, then verify:
#   - with zero weights_proj (uniform scores) topk selects the same single
#     position for all queries → heavy masking, deterministic output A
#   - with biased weights_proj (non-uniform scores) topk selects different
#     positions → different masking pattern, output B
# A and B must differ if the index mask is wired up correctly.
# ---------------------------------------------------------------------------
print("\n=== Test 3: Masking effect ===")
tokenizer = AutoTokenizer.from_pretrained(
    "BarraHome/llama3_2-1B-deepseek", trust_remote_code=True
)
inputs = tokenizer("The quick brown fox jumps over the lazy dog", return_tensors="pt")

converted.eval()

# Shrink topk to 1 so the mask always blocks most positions
for layer in converted.model.layers:
    layer.self_attn.indexer.index_topk = 1

# Run A: zero gates (uniform scores → topk picks last position everywhere)
for layer in converted.model.layers:
    layer.self_attn.indexer.weights_proj.weight.data.zero_()

with torch.no_grad():
    out_zeroed = converted(**inputs)

# Run B: non-zero gates (biased scores → topk picks a different position)
for layer in converted.model.layers:
    layer.self_attn.indexer.weights_proj.weight.data.fill_(1.0)

with torch.no_grad():
    out_biased = converted(**inputs)

# Restore topk
for layer in converted.model.layers:
    layer.self_attn.indexer.index_topk = 256

diff = (out_biased.logits - out_zeroed.logits).abs().mean().item()
print(f"  Mean logit diff (biased vs zeroed indexer, topk=1): {diff:.6f}")
print(f"  Masking has effect: {'YES' if diff > 1e-4 else 'NO — check masking logic'}")

print("\nAll checks done.")

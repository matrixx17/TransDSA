"""
transdsa/modeling.py

Defines DSAConfig, Indexer, and DSAAttention for the TransDSA framework.

TransDSA converts a pretrained MLA model into one that uses DeepSeek Sparse
Attention (DSA) by grafting a lightweight Indexer module onto each MLA
attention layer.  All original MLA weights are preserved; only the Indexer
parameters are newly initialized and trained.

Key design notes
----------------
* No fp8, no custom CUDA kernels — plain bf16 PyTorch ops throughout.
* The Indexer uses NON-INTERLEAVED RoPE (rotate_half on first/second halves).
  MLA uses standard apply_rotary_pos_emb from gemma2 (also non-interleaved,
  but a different reshape path).  These must not be mixed up.
* When q_lora_rank is set, the Indexer reuses MLA's latent query `qr`
  (output of wq_a → q_norm) as its own query input — free at prefill time.
  When q_lora_rank is None (as in BarraHome/llama3_2-1B-deepseek), the
  Indexer owns its own wq_a projection from hidden_size and `qr` is x.
* DSAAttention re-implements the MLA forward inline so it can intercept the
  raw attention scores before softmax and apply the sparse index mask.
* HuggingFace rotary embedders return cos/sin of shape (bsz, seq, head_dim).
  apply_rotary_pos_emb_interleave expects this shape and does unsqueeze(1)
  internally.  Our non-interleaved helper must match this convention.
* The Indexer scores query tokens against the *current* x only (seq_q tokens).
  In full-sequence training (seq_q == seq_k) this covers the full context.
  A separate Indexer key cache is needed for true autoregressive decode and
  is left for the inference engine to manage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.cache_utils import Cache
from transformers.models.gemma2.modeling_gemma2 import apply_rotary_pos_emb


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _apply_rotary_non_interleaved(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    Non-interleaved RoPE matching the DeepSeek-V3 Indexer convention.

    Unlike MLA's interleaved RoPE (which interleaves real/imag dimension-wise
    before rotating), the Indexer splits the rope slice into a first half
    ("real") and second half ("imaginary") and applies the standard
    rotate_half formula to that layout.

    Args:
        x   : (..., seq, rope_head_dim) — the rope-only slice of q or k.
              Leading dims can be (bsz, heads) or (bsz, 1).
        cos : (bsz, seq, rope_head_dim)  — from the model's rotary embedder.
        sin : (bsz, seq, rope_head_dim)

    Returns:
        Tensor of same shape as x with rotary embeddings applied.
    """
    half = x.shape[-1] // 2
    x1 = x[..., :half]   # "real" half
    x2 = x[..., half:]   # "imaginary" half
    rotated = torch.cat([-x2, x1], dim=-1)

    # cos/sin: (bsz, seq, rope_head_dim) — need (bsz, 1, seq, rope_head_dim)
    # for broadcasting against (bsz, heads, seq, rope_head_dim).
    cos = cos.unsqueeze(1)  # (bsz, 1, seq, rope_head_dim)
    sin = sin.unsqueeze(1)

    return x * cos + rotated * sin


# ---------------------------------------------------------------------------
# DSAConfig
# ---------------------------------------------------------------------------

@dataclass
class DSAConfig:
    """
    DSA-specific hyperparameters added on top of the base MLA config.

    Stored as a companion object rather than subclassing the HF config, so the
    MLA model's serialization and weight-loading are not disturbed.

    Attributes
    ----------
    index_n_heads : int
        Number of indexer attention heads.  For Llama-3.2 1B use 8.
    index_head_dim : int
        Per-head dimension of the indexer queries and keys.
        For Llama-3.2 1B use 64.
    index_topk : int
        Number of token positions the indexer selects per query position.
        For Llama-3.2 1B use 256 (configurable at inference time).
    index_rope_head_dim : int
        How many of the index_head_dim dimensions receive RoPE.
        Must equal qk_rope_head_dim from the MLA config (64 for this model).
    """
    index_n_heads: int = 8
    index_head_dim: int = 64
    index_topk: int = 256
    index_rope_head_dim: int = 64  # must equal mla_config.qk_rope_head_dim

    @classmethod
    def from_mla_config(
        cls,
        mla_config,
        index_n_heads: int = 8,
        index_head_dim: int = 64,
        index_topk: int = 256,
    ) -> "DSAConfig":
        """Construct a DSAConfig, deriving rope_head_dim from the MLA config."""
        return cls(
            index_n_heads=index_n_heads,
            index_head_dim=index_head_dim,
            index_topk=index_topk,
            index_rope_head_dim=mla_config.qk_rope_head_dim,
        )


# ---------------------------------------------------------------------------
# Indexer
# ---------------------------------------------------------------------------

class Indexer(nn.Module):
    """
    Lightweight token-selection module for DSA.

    Given the current hidden states `x` and the MLA latent query `qr`
    (already computed by wq_a → q_norm in MLA, so reused for free), the
    Indexer scores every key position against every query position and returns
    the top-k most relevant token indices per query position.  These indices
    are used by DSAAttention to zero out non-selected positions in the MLA
    attention score matrix before softmax.

    New parameters per layer
    ------------------------
    wq_a (optional): Linear(hidden_size → q_lora_rank, no bias)
                     Only present when the MLA layer has no q_lora_rank
                     (i.e. uses a full q_proj).  When q_lora_rank IS set,
                     the Indexer receives MLA's pre-computed qr directly and
                     wq_a is absent.
    wq_b        : Linear(q_lora_rank → index_n_heads * index_head_dim, no bias)
    wk          : Linear(hidden_size → index_head_dim, no bias)
    k_norm      : LayerNorm(index_head_dim)   [weight+bias, fp32 internally]
    weights_proj: Linear(hidden_size → index_n_heads, no bias, fp32)
                  Per-head scalar gate modulating per-position score magnitude.

    RoPE convention
    ---------------
    NON-INTERLEAVED RoPE on the first `index_rope_head_dim` dims of q and k.
    This differs from MLA which uses interleaved RoPE.

    Score accumulation
    ------------------
    index_score[b, q, k] = sum_h( gate[b,h,q] * softmax_scale
                                  * dot(q[b,h,q], k[b,1,k]) )
    The per-head gate (weights_proj output, fp32) lets the model learn which
    heads' opinions matter for token selection.
    """

    # Latent dim used when the MLA layer has no q_lora_rank.  Chosen to be
    # comparable to a typical q_lora_rank (512) while staying lightweight.
    _DEFAULT_LATENT_DIM = 512

    def __init__(
        self,
        hidden_size: int,
        q_lora_rank: Optional[int],
        dsa_config: DSAConfig,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Args:
            hidden_size  : Model hidden dimension.
            q_lora_rank  : MLA's q_lora_rank, or None if MLA uses a full q_proj.
                           When None the Indexer adds its own wq_a projection.
            dsa_config   : Indexer hyperparameters.
            dtype        : dtype for new bf16 parameters (wq_a, wq_b, wk).
                           weights_proj is always fp32.
        """
        super().__init__()

        self.n_heads = dsa_config.index_n_heads
        self.head_dim = dsa_config.index_head_dim
        self.rope_head_dim = dsa_config.index_rope_head_dim
        self.index_topk = dsa_config.index_topk
        self.softmax_scale = self.head_dim ** -0.5

        assert self.rope_head_dim <= self.head_dim, (
            f"index_rope_head_dim ({self.rope_head_dim}) must be <= "
            f"index_head_dim ({self.head_dim})"
        )

        # When q_lora_rank is None (MLA uses full q_proj, no latent query),
        # the Indexer creates its own wq_a to produce the latent query.
        # When q_lora_rank is set, qr is passed in from outside (free reuse).
        if q_lora_rank is None:
            latent_dim = self._DEFAULT_LATENT_DIM
            self.wq_a = nn.Linear(hidden_size, latent_dim, bias=False, dtype=dtype)
            nn.init.normal_(self.wq_a.weight, std=0.02)
        else:
            latent_dim = q_lora_rank
            self.wq_a = None  # qr is provided by the caller

        # Query projection: latent_dim → n_heads * head_dim
        self.wq_b = nn.Linear(latent_dim, self.n_heads * self.head_dim, bias=False, dtype=dtype)

        # Key projection: one shared key per token position
        self.wk = nn.Linear(hidden_size, self.head_dim, bias=False, dtype=dtype)

        # Key normalisation — LayerNorm (not RMSNorm), matching DeepSeek-V3
        self.k_norm = nn.LayerNorm(self.head_dim)

        # Per-head scalar gate always in fp32 for numerical stability
        self.weights_proj = nn.Linear(hidden_size, self.n_heads, bias=False, dtype=torch.float32)

        # Initialisation: small queries/keys → near-uniform early scores.
        # Zero gates → indexer has no effect at init (safe for warm-up).
        nn.init.normal_(self.wq_b.weight, std=0.02)
        nn.init.normal_(self.wk.weight, std=0.02)
        nn.init.zeros_(self.weights_proj.weight)

    def forward(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute top-k token indices for sparse attention.

        Args:
            x   : (bsz, seq_q, hidden_size)
                  Current input hidden states.  During training (full-sequence
                  mode) this covers all positions so seq_q == seq_k.
            qr  : (bsz, seq_q, q_lora_rank) or None
                  MLA latent query after wq_a + q_norm, reused when available.
                  Pass None when the MLA layer has no q_lora_rank; the Indexer
                  will compute its own latent query from x via self.wq_a.
            cos : (bsz, seq_q, qk_rope_head_dim)
                  Cosine RoPE embeddings for the current positions.
            sin : (bsz, seq_q, qk_rope_head_dim)
                  Sine RoPE embeddings for the current positions.
            attention_mask : (bsz, 1, seq_q, seq_q) additive mask or None
                  Causal / padding mask (-inf for positions that must not be
                  attended to).  Applied before top-k so they are never selected.

        Returns:
            topk_indices : (bsz, seq_q, k_val)
                  Indices along the seq_q (key) axis of the top-k tokens,
                  where k_val = min(index_topk, seq_q).
        """
        bsz, seq_q, _ = x.shape

        # ---- Query latent ----
        # If MLA has no q_lora_rank, qr is None and we compute our own latent.
        if qr is None:
            qr = self.wq_a(x)   # (bsz, seq_q, _DEFAULT_LATENT_DIM)

        # ---- Query ----
        # (bsz, seq_q, n_heads * head_dim) → (bsz, n_heads, seq_q, head_dim)
        q = self.wq_b(qr).view(bsz, seq_q, self.n_heads, self.head_dim)
        q = q.transpose(1, 2)  # (bsz, n_heads, seq_q, head_dim)

        q_pe   = q[..., :self.rope_head_dim]   # (bsz, n_heads, seq_q, rope_dim)
        q_nope = q[..., self.rope_head_dim:]   # (bsz, n_heads, seq_q, nope_dim)

        # Non-interleaved RoPE on query
        # cos/sin are (bsz, seq_q, rope_head_dim); _apply_rotary_non_interleaved
        # will unsqueeze dim-1 to broadcast over the heads axis.
        q_pe = _apply_rotary_non_interleaved(q_pe, cos, sin)
        q = torch.cat([q_pe, q_nope], dim=-1)  # (bsz, n_heads, seq_q, head_dim)

        # ---- Key ----
        # One shared key vector per token (no per-head key — broadcast instead).
        k = self.wk(x)          # (bsz, seq_q, head_dim)
        k = self.k_norm(k)      # LayerNorm

        # Unsqueeze a pseudo-heads dim so RoPE helper can broadcast.
        # (bsz, seq_q, head_dim) → (bsz, 1, seq_q, head_dim)
        k = k.unsqueeze(1)

        k_pe   = k[..., :self.rope_head_dim]
        k_nope = k[..., self.rope_head_dim:]

        # Non-interleaved RoPE on key (same positions as query)
        k_pe = _apply_rotary_non_interleaved(k_pe, cos, sin)
        k = torch.cat([k_pe, k_nope], dim=-1)  # (bsz, 1, seq_q, head_dim)

        # ---- Scores ----
        # q : (bsz, n_heads, seq_q, head_dim)
        # k : (bsz,       1, seq_q, head_dim)  — shared, broadcast over heads
        # scores : (bsz, n_heads, seq_q, seq_q)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.softmax_scale

        # Per-head scalar gate from weights_proj (fp32).
        # (bsz, seq_q, n_heads) → (bsz, n_heads, seq_q, 1)
        gate = self.weights_proj(x.float())           # fp32
        gate = gate.transpose(1, 2).unsqueeze(-1)     # (bsz, n_heads, seq_q, 1)
        gate = gate * (self.n_heads ** -0.5)

        # Modulate and collapse heads → (bsz, seq_q, seq_q)
        index_score = (scores.float() * gate).sum(dim=1)

        # Apply causal/padding mask before top-k selection
        if attention_mask is not None:
            # attention_mask: (bsz, 1, seq_q, seq_q) additive
            index_score = index_score + attention_mask[:, 0, :, :]

        # Top-k along the key axis
        k_val = min(self.index_topk, seq_q)
        topk_indices = index_score.topk(k_val, dim=-1)[1]  # (bsz, seq_q, k_val)

        return topk_indices


# ---------------------------------------------------------------------------
# DSAAttention
# ---------------------------------------------------------------------------

class DSAAttention(nn.Module):
    """
    Wraps a pretrained MLAAttention and adds a DSA Indexer.

    The MLA weight modules (wq_a, q_norm, wq_b, wkv_a, kv_norm, wkv_b, wo)
    are borrowed by reference — the same Parameter objects are reused, not
    copied.  Only the Indexer introduces new, randomly initialised parameters.

    Forward pass (eager bf16, no fused kernels)
    -------------------------------------------
    When q_lora_rank is set (MLA has wq_a/wq_b):
      1.  qr = q_norm(wq_a(x))               MLA latent query, reused by Indexer
      2.  q  = wq_b(qr)                       Full MLA query
    When q_lora_rank is None (MLA has full q_proj):
      1.  qr = None                           No latent query to reuse
      2.  q  = q_proj(x)                      Full MLA query directly
    Then in both cases:
      3.  k, v = mla_kv(x, cos, sin)          MLA key/value with interleaved RoPE
      4.  k, v updated from KV cache          (decode path)
      5.  topk = Indexer(x, qr, cos, sin)     Non-interleaved RoPE; qr may be None
      6.  scores = q @ k.T * scale            Raw attention scores
      7.  scores += causal_mask               Standard causal masking
      8.  scores += index_mask                -inf for non-top-k positions
      9.  out = softmax(scores) @ v → wo      Standard attention output
    """

    def __init__(
        self,
        mla: nn.Module,
        dsa_config: DSAConfig,
        use_tri_scorer: bool = False,
    ):
        """
        Args:
            mla            : A pretrained MLAAttention instance whose weights are reused.
            dsa_config     : Indexer hyperparameters.
            use_tri_scorer : If True, use TriAttentionScorer instead of the
                             learned Indexer for token selection.  Call
                             load_tri_scorer() after construction to load stats.
        """
        super().__init__()

        self.config = mla.config
        self.layer_idx = mla.layer_idx
        self.dsa_config = dsa_config
        self.use_tri_scorer = use_tri_scorer
        self.tri_scorer = None  # set by load_tri_scorer()

        # ---- Borrow dimensions from MLA ----
        self.num_heads = mla.num_heads
        self.q_lora_rank = mla.q_lora_rank      # None for this model
        self.kv_lora_rank = mla.kv_lora_rank
        self.qk_rope_head_dim = mla.qk_rope_head_dim
        self.qk_nope_head_dim = mla.qk_nope_head_dim
        self.qk_head_dim = mla.qk_head_dim
        self.v_head_dim = mla.v_head_dim
        self.scaling = mla.scaling
        self.attention_dropout = mla.attention_dropout

        # ---- Borrow MLA weight modules using EXACT original attribute names ----
        # Attribute names must match the original MLAAttention exactly so that
        # state_dict() emits keys identical to the pretrained checkpoint.
        # The saved checkpoint will then only have extra indexer.* keys.
        if self.q_lora_rank is not None:
            # LoRA query path
            self.q_a_proj     = mla.q_a_proj
            self.q_a_layernorm = getattr(mla, "q_a_layernorm", None)
            self.q_b_proj     = mla.q_b_proj
            self.q_proj       = None
        else:
            # Full query path (this model)
            self.q_proj       = mla.q_proj
            self.q_a_proj     = None
            self.q_a_layernorm = None
            self.q_b_proj     = None

        self.kv_a_proj_with_mqa = mla.kv_a_proj_with_mqa
        self.kv_a_layernorm     = getattr(mla, "kv_a_layernorm", None)
        self.kv_b_proj          = mla.kv_b_proj
        self.o_proj             = mla.o_proj

        # ---- New: Indexer (only source of new trainable parameters) ----
        # Infer the model's working dtype from the output projection weights so
        # the Indexer's parameters are created in the same dtype (e.g. bfloat16).
        model_dtype = self.o_proj.weight.dtype

        self.indexer = Indexer(
            hidden_size=self.config.hidden_size,
            q_lora_rank=self.q_lora_rank,       # None → Indexer adds its own wq_a
            dsa_config=dsa_config,
            dtype=model_dtype,
        )

    # ------------------------------------------------------------------
    # TriAttention scorer
    # ------------------------------------------------------------------

    def load_tri_scorer(self, scorer: "tri_scorer.TriAttentionScorer") -> None:
        """
        Attach a pre-built TriAttentionScorer to this layer.

        The scorer is shared across layers (it indexes by layer_idx
        internally), so the same object can be passed to every
        DSAAttention instance.

        Args:
            scorer : A TriAttentionScorer loaded via
                     TriAttentionScorer.from_stats().
        """
        self.tri_scorer = scorer
        self.use_tri_scorer = True

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_qr(self, hidden_states: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Compute the MLA latent query if q_lora_rank is set.

        Returns (bsz, seq, q_lora_rank) when the LoRA path is active,
        or None when the model uses a full q_proj (q_lora_rank is None).
        The Indexer handles the None case internally via its own wq_a.
        """
        if self.q_a_proj is None:
            return None
        qr = self.q_a_proj(hidden_states)
        if self.q_a_layernorm is not None:
            qr = self.q_a_layernorm(qr)
        return qr

    def _compute_mla_qkv(
        self,
        hidden_states: torch.Tensor,
        qr: Optional[torch.Tensor],
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute full MLA Q, K, V tensors for the current (query) positions.

        Uses INTERLEAVED RoPE via apply_rotary_pos_emb_interleave, exactly as
        the original MLAAttention forward does.

        Args:
            hidden_states : (bsz, seq_q, hidden_size)
            qr            : (bsz, seq_q, q_lora_rank) or None
            cos           : (bsz, seq_q, qk_rope_head_dim)
            sin           : (bsz, seq_q, qk_rope_head_dim)

        Returns:
            query_states : (bsz, n_heads, seq_q, qk_head_dim)
            key_states   : (bsz, n_heads, seq_q, qk_head_dim)
            value_states : (bsz, n_heads, seq_q, v_head_dim)
            k_rot_raw    : (bsz, 1, seq_q, qk_rope_head_dim) pre-RoPE key rope component
        """
        bsz, seq_q, _ = hidden_states.shape

        # ---- Query ----
        # Mirrors mla.py forward exactly.
        if qr is not None:
            # LoRA path (q_lora_rank is set): qr already has layernorm applied
            q = self.q_b_proj(qr)
        else:
            # Full-rank path (this model: q_proj only, no LoRA)
            q = self.q_proj(hidden_states)
        q = q.view(bsz, seq_q, self.num_heads, self.qk_head_dim).transpose(1, 2)
        # (bsz, n_heads, seq_q, qk_head_dim)

        q_nope, q_rot = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        # ---- KV ----
        # Mirrors mla.py: split compressed_kv into k_pass (latent) and k_rot,
        # then project k_pass through kv_b_proj (no norm in this model).
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        # (bsz, seq_q, kv_lora_rank + qk_rope_head_dim)
        k_pass, k_rot_raw = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )

        if self.kv_a_layernorm is not None:
            k_pass = self.kv_a_layernorm(k_pass)

        kv = self.kv_b_proj(k_pass)
        # (bsz, seq_q, n_heads * (qk_nope_head_dim + v_head_dim))
        kv = kv.view(
            bsz, seq_q, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
        ).transpose(1, 2)
        k_nope, value_states = torch.split(
            kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )
        # k_nope      : (bsz, n_heads, seq_q, qk_nope_head_dim)
        # value_states: (bsz, n_heads, seq_q, v_head_dim)

        # k_rot needs shape (bsz, 1, seq_q, rope_dim) for apply_rotary_pos_emb
        k_rot_raw = k_rot_raw.view(bsz, 1, seq_q, self.qk_rope_head_dim)

        # Standard (non-interleaved) RoPE — matches apply_rotary_pos_emb used in mla.py
        q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot_raw, cos, sin)
        # q_rot: (bsz, n_heads, seq_q, qk_rope_head_dim)
        # k_rot: (bsz,       1, seq_q, qk_rope_head_dim)

        k_rot = k_rot.expand(bsz, self.num_heads, seq_q, self.qk_rope_head_dim)

        query_states = torch.cat([q_nope, q_rot], dim=-1)
        key_states   = torch.cat([k_nope, k_rot], dim=-1)

        return query_states, key_states, value_states, k_rot_raw

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        output_attentions: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Signature matches LlamaDecoderLayer's expectations: accepts
        past_key_values (not past_key_value) and returns exactly 2 values
        (attn_output, attn_weights).

        Args:
            hidden_states       : (bsz, seq_q, hidden_size)
            position_embeddings : (cos, sin), each (bsz, seq_q, qk_rope_head_dim)
            attention_mask      : (bsz, 1, seq_q, seq_k) additive mask or None
            past_key_values     : HuggingFace Cache or None (decode path)
            cache_position      : (seq_q,) integer positions or None
            output_attentions   : Return attention weights if True

        Returns:
            attn_output  : (bsz, seq_q, hidden_size)
            attn_weights : (bsz, n_heads, seq_q, seq_k) or None
        """
        bsz, seq_q, _ = hidden_states.shape
        cos, sin = position_embeddings  # each: (bsz, seq_q, qk_rope_head_dim)

        # ------------------------------------------------------------------
        # 1. MLA latent query — reused by Indexer when available
        #    qr is None when the MLA layer has no q_lora_rank (full q_proj).
        #    The Indexer handles the None case by running its own wq_a.
        # ------------------------------------------------------------------
        qr = self._compute_qr(hidden_states)  # (bsz, seq_q, q_lora_rank) or None

        # ------------------------------------------------------------------
        # 2. Full MLA Q, K, V for the current positions
        # ------------------------------------------------------------------
        query_states, key_states, value_states, k_rot_raw = (
            self._compute_mla_qkv(hidden_states, qr, cos, sin)
        )
        # query_states : (bsz, n_heads, seq_q, qk_head_dim)
        # key_states   : (bsz, n_heads, seq_q, qk_head_dim)
        # value_states : (bsz, n_heads, seq_q, v_head_dim)
        # k_rot_raw    : (bsz, 1, seq_q, qk_rope_head_dim) — pre-RoPE

        # ------------------------------------------------------------------
        # 3. KV cache update (decode path)
        # ------------------------------------------------------------------
        if past_key_values is not None:
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "cache_position": cache_position,
            }
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )
        # key_states / value_states now cover the full context (seq_k >= seq_q)

        seq_k = key_states.shape[2]

        # ------------------------------------------------------------------
        # 4. Token selection — Indexer or TriAttention scorer
        # ------------------------------------------------------------------
        if self.use_tri_scorer and self.tri_scorer is not None:
            # TriAttention deterministic scorer — scores ALL key positions
            # (cached + current) using frequency-domain statistics.
            #
            # For prefill (no cache): k_rot_raw covers all positions.
            # For decode (with cache): need cos/sin for all cached positions
            # to invert RoPE on the cached keys.
            if past_key_values is not None and seq_k > seq_q:
                # Decode path: the scorer generates its own cos/sin tables
                # for the full seq_k range via its internal rotary embedder.
                topk_flat = self.tri_scorer.score_tokens_with_cache(
                    layer_idx=self.layer_idx,
                    cached_key_states=key_states,
                    k_rot_raw_current=k_rot_raw,
                    cache_position=cache_position,
                    seq_q=seq_q,
                    topk=self.dsa_config.index_topk,
                )
                # topk_flat: [k_val] — indices into seq_k
                # Expand to (bsz, seq_q, k_val) for mask construction
                k_val = topk_flat.shape[0]
                topk_indices = topk_flat.unsqueeze(0).unsqueeze(0).expand(
                    bsz, seq_q, k_val,
                )
            else:
                # Prefill path: all keys are current, k_rot_raw covers
                # everything.  Score per query position.
                # k_rot_raw: (bsz, 1, seq_q, rope_dim) — same for all heads
                k_rot_2d = k_rot_raw[0, 0]  # (seq_q, rope_dim)
                positions = (
                    cache_position
                    if cache_position is not None
                    else torch.arange(
                        seq_q,
                        device=hidden_states.device,
                        dtype=torch.long,
                    )
                )
                current_pos = int(positions[-1].item())
                topk_flat = self.tri_scorer.score_tokens(
                    layer_idx=self.layer_idx,
                    k_rot_raw=k_rot_2d,
                    absolute_positions=positions,
                    current_position=current_pos,
                    topk=self.dsa_config.index_topk,
                )
                # topk_flat: [k_val]
                k_val = topk_flat.shape[0]
                topk_indices = topk_flat.unsqueeze(0).unsqueeze(0).expand(
                    bsz, seq_q, k_val,
                )
        else:
            # Learned Indexer — scores current window only
            # Slice attention_mask to the current-window square.
            if attention_mask is not None and seq_k > seq_q:
                indexer_mask = attention_mask[:, :, :, -seq_q:]
            else:
                indexer_mask = attention_mask

            topk_indices = self.indexer(
                x=hidden_states,
                qr=qr,
                cos=cos,
                sin=sin,
                attention_mask=indexer_mask,
            )
        # topk_indices: (bsz, seq_q, k_val)

        # ------------------------------------------------------------------
        # 5. Raw attention scores (eager — we own the compute here)
        # ------------------------------------------------------------------
        # (bsz, n_heads, seq_q, seq_k)
        scores = torch.matmul(query_states, key_states.transpose(-2, -1))
        scores = scores * self.scaling

        # ------------------------------------------------------------------
        # 6. Causal / padding mask
        # ------------------------------------------------------------------
        if attention_mask is not None:
            scores = scores + attention_mask  # additive; -inf for invalid

        # ------------------------------------------------------------------
        # 7. DSA index mask — zero out non-selected positions
        #
        # topk_indices are offsets into either the current window
        # (Indexer) or the full seq_k (TriAttention scorer).
        # When using the Indexer and seq_k > seq_q, shift indices.
        # When using the tri_scorer, indices already refer to seq_k.
        # ------------------------------------------------------------------
        index_mask = torch.full(
            (bsz, seq_q, seq_k),
            float("-inf"),
            device=scores.device,
            dtype=scores.dtype,
        )
        if self.use_tri_scorer and self.tri_scorer is not None:
            # Indices already refer to the full seq_k dimension
            index_mask.scatter_(-1, topk_indices, 0.0)
        else:
            # Indexer indices are offsets into the current window
            offset = seq_k - seq_q
            shifted_indices = topk_indices + offset
            index_mask.scatter_(-1, shifted_indices, 0.0)

        # Broadcast over heads: (bsz, 1, seq_q, seq_k)
        scores = scores + index_mask.unsqueeze(1)

        # ------------------------------------------------------------------
        # 8. Softmax → weighted sum
        # ------------------------------------------------------------------
        attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )

        if self.training and self.attention_dropout > 0.0:
            attn_weights = F.dropout(attn_weights, p=self.attention_dropout)

        # (bsz, n_heads, seq_q, v_head_dim)
        attn_output = torch.matmul(attn_weights, value_states)

        # ------------------------------------------------------------------
        # 9. Output projection
        # ------------------------------------------------------------------
        # (bsz, n_heads, seq_q, v_head_dim) → (bsz, seq_q, n_heads*v_head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, seq_q, self.num_heads * self.v_head_dim)
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights if output_attentions else None

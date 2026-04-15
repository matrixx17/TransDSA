"""
transdsa/tri_scorer.py

Deterministic token scorer for TransDSA based on TriAttention's
frequency-domain analysis.  Replaces the learned Indexer with a
calibration-based scorer that requires no training.

The scorer uses pre-computed per-head frequency statistics (from
calibrate.py) to rank token positions by predicted attention
importance.  Only the RoPE component of keys (qk_rope_head_dim=64)
is used for scoring — the nope component carries no positional
information.

For prefill, k_rot_raw is available directly before RoPE is applied.
For decode with cached keys, the RoPE portion is extracted from the
cache and inverted to recover the pre-RoPE representation.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# Functions lifted from triattention/pruning_utils.py (self-contained copy)
# ---------------------------------------------------------------------------

def _rotate_half(x: torch.Tensor, *, style: str = "half") -> torch.Tensor:
    """Rotate-half for RoPE inversion."""
    if style == "interleaved":
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)
    d = x.shape[-1] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    return torch.cat((-x2, x1), dim=-1)


def _invert_rope(
    rotated: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    scale: float,
    *,
    style: str = "half",
) -> torch.Tensor:
    """Invert RoPE to recover the pre-rotation representation."""
    if scale == 0:
        raise ValueError("attention scaling factor must be non-zero")
    scale_t = torch.tensor(scale, device=rotated.device, dtype=rotated.dtype)
    base = rotated / scale_t
    cos_unit = cos / scale_t
    sin_unit = sin / scale_t
    if style == "interleaved":
        even = base[..., ::2]
        odd = base[..., 1::2]
        cos_even = cos_unit[..., ::2]
        cos_odd = cos_unit[..., 1::2]
        sin_even = sin_unit[..., ::2]
        sin_odd = sin_unit[..., 1::2]
        det = cos_even * cos_odd + sin_even * sin_odd
        det = det.clamp_min(1e-12)
        orig_even = (even * cos_odd + odd * sin_even) / det
        orig_odd = (odd * cos_even - even * sin_odd) / det
        restored = torch.empty_like(base)
        restored[..., ::2] = orig_even
        restored[..., 1::2] = orig_odd
        return restored
    return base * cos_unit - _rotate_half(base, style=style) * sin_unit


def _to_complex_pairs(
    tensor: torch.Tensor, *, style: str = "half",
) -> torch.Tensor:
    """Convert a real [seq, dim] tensor to complex pairs [seq, dim//2]."""
    if tensor.size(-1) % 2 != 0:
        raise ValueError("Head dimension must be even to form complex pairs")
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


def _build_geometric_offsets(
    max_length: int, device: torch.device,
) -> torch.Tensor:
    """Geometric offset grid for score aggregation."""
    if max_length < 1:
        raise ValueError("offset_max_length must be >= 1")
    offsets: List[float] = []
    value = 1
    while value <= max_length:
        offsets.append(float(value))
        value *= 2
    return torch.tensor(offsets, device=device, dtype=torch.float32)


def _compute_frequency_scaling(
    rotary: torch.nn.Module,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Compute per-frequency scaling from the rotary embedder."""
    position_ids = torch.zeros(1, 1, device=device, dtype=torch.long)
    probe = torch.zeros(1, 1, head_dim, device=device, dtype=dtype)
    cos, sin = rotary(probe, position_ids)
    cos0 = cos[0, 0]
    sin0 = sin[0, 0]
    scale = torch.sqrt(cos0[0::2].pow(2) + sin0[0::2].pow(2))
    return scale.to(device=device, dtype=torch.float32)


def compute_frequency_statistics_from_means(
    q_mean_complex: torch.Tensor,
    q_abs_mean: torch.Tensor,
    k_unrot: torch.Tensor,
    *,
    style: str = "half",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute amplitude, phase, and extra term from calibration Q stats and
    un-rotated key vectors.

    Args:
        q_mean_complex : [freq_count] complex — mean Q in frequency domain
        q_abs_mean     : [freq_count] real — mean |Q| in frequency domain
        k_unrot        : [seq, rope_head_dim] real — pre-RoPE key vectors

    Returns:
        amp   : [seq, freq_count]
        phi   : [seq, freq_count]
        extra : [seq, freq_count]
    """
    k_complex = _to_complex_pairs(k_unrot, style=style)
    q_mean_abs = torch.abs(q_mean_complex)
    k_abs = torch.abs(k_complex)
    relative = q_mean_complex.unsqueeze(0) * torch.conj(k_complex)
    phi = torch.atan2(relative.imag, relative.real)
    amp = q_mean_abs.unsqueeze(0) * k_abs
    extra = (q_abs_mean - q_mean_abs).unsqueeze(0) * k_abs
    return amp, phi, extra


def score_keys_for_round(
    key_indices: torch.Tensor,
    round_start: int,
    amp: torch.Tensor,
    phi: torch.Tensor,
    omega: torch.Tensor,
    extra: torch.Tensor,
    offsets: torch.Tensor,
    aggregation: str,
    freq_scale_sq: torch.Tensor,
) -> torch.Tensor:
    """
    Score cached key positions using frequency-domain statistics.

    Args:
        key_indices   : [seq] integer positions of cached tokens
        round_start   : current generation position (query position)
        amp           : [seq, freq_count]
        phi           : [seq, freq_count]
        omega         : [freq_count] — rotary frequencies
        extra         : [seq, freq_count]
        offsets       : [num_offsets] — geometric offset grid
        aggregation   : "mean" or "max"
        freq_scale_sq : [freq_count] — squared frequency scaling

    Returns:
        scores : [seq]
    """
    if key_indices.numel() == 0:
        return torch.empty(0, device=amp.device, dtype=torch.float32)

    base_delta = round_start - key_indices.to(
        device=amp.device, dtype=torch.float32,
    )
    delta_grid = base_delta.unsqueeze(1) + offsets.unsqueeze(0)

    freq_scale_sq = freq_scale_sq.to(device=amp.device, dtype=torch.float32)
    phase = (
        delta_grid.unsqueeze(2) * omega.view(1, 1, -1) + phi.unsqueeze(1)
    )

    cos_phase = torch.cos(phase)
    scale = freq_scale_sq.view(1, 1, -1)

    base_scores = (amp.unsqueeze(1) * scale * cos_phase).sum(dim=2)
    additive = (extra * freq_scale_sq.view(1, -1)).sum(dim=1, keepdim=True)
    combined = base_scores + additive

    if aggregation == "mean":
        return combined.mean(dim=1)
    return combined.max(dim=1).values


# ---------------------------------------------------------------------------
# HeadFrequencyStats
# ---------------------------------------------------------------------------

@dataclass
class HeadFrequencyStats:
    q_mean_complex: torch.Tensor   # [freq_count] complex64
    q_abs_mean: torch.Tensor       # [freq_count] float32


# ---------------------------------------------------------------------------
# TriAttentionScorer
# ---------------------------------------------------------------------------

class TriAttentionScorer:
    """
    Deterministic token scorer using TriAttention frequency statistics.

    Scores key positions based on their predicted attention importance,
    computed from calibrated per-head Q statistics and the pre-RoPE key
    vectors.  No learned parameters — all scoring is deterministic from
    the calibration file.

    Usage from DSAAttention:
        scorer = TriAttentionScorer.from_stats(stats_path, model, device)
        topk = scorer.score_tokens(layer_idx, k_rot_raw, positions, topk)
    """

    def __init__(
        self,
        head_stats: Dict[Tuple[int, int], HeadFrequencyStats],
        sampled_heads: List[Tuple[int, int]],
        omega: torch.Tensor,
        freq_scale_sq: torch.Tensor,
        offsets: torch.Tensor,
        attention_scale: float,
        head_dim: int,
        device: torch.device,
        aggregation: str = "mean",
    ) -> None:
        self.head_stats = head_stats
        self.sampled_heads = sampled_heads
        self.omega = omega
        self.freq_scale_sq = freq_scale_sq
        self.offsets = offsets
        self.attention_scale = attention_scale
        self.head_dim = head_dim
        self.device = device
        self.aggregation = aggregation

        # Pre-index heads by layer for fast lookup
        self._heads_by_layer: Dict[int, List[Tuple[int, HeadFrequencyStats]]] = {}
        for layer_idx, head_idx in sampled_heads:
            key = (layer_idx, head_idx)
            if key in head_stats:
                self._heads_by_layer.setdefault(layer_idx, []).append(
                    (head_idx, head_stats[key])
                )

    @classmethod
    def from_stats(
        cls,
        stats_path: str,
        model: torch.nn.Module,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
        offset_max_length: int = 65536,
        aggregation: str = "mean",
    ) -> "TriAttentionScorer":
        """
        Load calibration stats and build a scorer tied to the given model.

        Args:
            stats_path         : Path to the .pt file from calibrate.py
            model              : The loaded HF model (needed for rotary_emb)
            device             : Device for scorer tensors
            dtype              : Working dtype for rotary probing
            offset_max_length  : Max length for geometric offset grid
            aggregation        : "mean" or "max" for offset aggregation
        """
        payload = torch.load(stats_path, map_location=device, weights_only=False)
        metadata = payload["metadata"]
        stats_raw = payload["stats"]

        # Parse sampled heads
        sampled_heads = [
            (int(pair[0]), int(pair[1]))
            for pair in metadata["sampled_heads"]
        ]

        # Reconstruct HeadFrequencyStats
        head_stats: Dict[Tuple[int, int], HeadFrequencyStats] = {}
        for layer_idx, head_idx in sampled_heads:
            key = f"layer{layer_idx:02d}_head{head_idx:02d}"
            entry = stats_raw.get(key)
            if entry is None:
                continue
            q_mean_complex = torch.complex(
                entry["q_mean_real"].to(device=device, dtype=torch.float32),
                entry["q_mean_imag"].to(device=device, dtype=torch.float32),
            )
            q_abs_mean = entry["q_abs_mean"].to(
                device=device, dtype=torch.float32,
            )
            head_stats[(layer_idx, head_idx)] = HeadFrequencyStats(
                q_mean_complex=q_mean_complex,
                q_abs_mean=q_abs_mean,
            )

        # Get rotary embedder from model
        backbone = getattr(model, "model", model)
        if hasattr(backbone, "rotary_emb"):
            rotary = backbone.rotary_emb
        else:
            rotary = backbone.layers[0].self_attn.rotary_emb
        attention_scale = float(
            getattr(rotary, "attention_scaling", 1.0)
        )

        head_dim = int(metadata["head_dim"])

        # Build omega from inv_freq
        inv_freq = rotary.inv_freq.to(device=device, dtype=torch.float32)
        freq_count = max(1, head_dim // 2)
        omega = inv_freq[:freq_count]

        # Build frequency scaling and offset grid
        freq_scale = _compute_frequency_scaling(
            rotary, head_dim, dtype, device,
        )
        freq_scale_sq = freq_scale.pow(2)
        offsets = _build_geometric_offsets(offset_max_length, device)

        return cls(
            head_stats=head_stats,
            sampled_heads=sampled_heads,
            omega=omega,
            freq_scale_sq=freq_scale_sq,
            offsets=offsets,
            attention_scale=attention_scale,
            head_dim=head_dim,
            device=device,
            aggregation=aggregation,
        )

    def score_tokens(
        self,
        layer_idx: int,
        k_rot_raw: torch.Tensor,
        absolute_positions: torch.Tensor,
        current_position: int,
        topk: int,
    ) -> torch.Tensor:
        """
        Score key positions and return top-k indices.

        Args:
            layer_idx          : Which transformer layer (0-indexed)
            k_rot_raw          : [seq_k, qk_rope_head_dim] pre-RoPE key vectors
                                 (the RoPE component only, before rotation)
            absolute_positions : [seq_k] integer absolute positions of each key
            current_position   : The current query position (for delta computation)
            topk               : Number of top positions to return

        Returns:
            topk_indices : [topk] indices into the seq_k dimension
        """
        heads_for_layer = self._heads_by_layer.get(layer_idx)
        if not heads_for_layer:
            # No calibration data for this layer — fall back to most recent
            seq_k = k_rot_raw.shape[0]
            k_val = min(topk, seq_k)
            return torch.arange(seq_k - k_val, seq_k, device=self.device)

        # k_rot_raw: [seq_k, rope_head_dim]
        k_rot_raw_dev = k_rot_raw.to(device=self.device, dtype=torch.float32)

        per_head_scores: List[torch.Tensor] = []
        for head_idx, stats in heads_for_layer:
            amp, phi, extra = compute_frequency_statistics_from_means(
                stats.q_mean_complex,
                stats.q_abs_mean,
                k_rot_raw_dev,
                style="half",
            )
            head_scores = score_keys_for_round(
                key_indices=absolute_positions,
                round_start=current_position,
                amp=amp,
                phi=phi,
                omega=self.omega,
                extra=extra,
                offsets=self.offsets,
                aggregation=self.aggregation,
                freq_scale_sq=self.freq_scale_sq,
            )
            per_head_scores.append(head_scores)

        # Aggregate across heads: mean
        scores = torch.stack(per_head_scores, dim=0).mean(dim=0)  # [seq_k]
        k_val = min(topk, scores.shape[0])
        return scores.topk(k_val, dim=-1).indices  # [k_val]

    def score_tokens_with_cache(
        self,
        layer_idx: int,
        cached_key_states: torch.Tensor,
        k_rot_raw_current: torch.Tensor,
        cos_full: torch.Tensor,
        sin_full: torch.Tensor,
        cache_position: Optional[torch.LongTensor],
        seq_q: int,
        topk: int,
    ) -> torch.Tensor:
        """
        Score all key positions (cached + current) and return top-k indices.

        Handles the decode path where past keys are post-RoPE in the cache
        and current keys are available pre-RoPE.

        Args:
            layer_idx           : Transformer layer index
            cached_key_states   : [bsz, n_heads, seq_k, qk_head_dim] full keys
                                  from cache (post-RoPE for the rope portion)
            k_rot_raw_current   : [bsz, 1, seq_q, rope_head_dim] pre-RoPE keys
                                  for current positions
            cos_full            : [bsz, seq_k, rope_head_dim] cos table for ALL
                                  positions in the cache
            sin_full            : [bsz, seq_k, rope_head_dim] sin table for ALL
                                  positions in the cache
            cache_position      : [seq_q] absolute positions of current tokens
            seq_q               : Number of current query positions
            topk                : Number of top positions to return

        Returns:
            topk_indices : [topk] indices into seq_k
        """
        seq_k = cached_key_states.shape[2]
        qk_head_dim = cached_key_states.shape[3]
        rope_dim = self.head_dim  # qk_rope_head_dim = 64

        # --- Determine absolute positions for all seq_k tokens ---
        if cache_position is not None:
            # cache_position gives positions for current tokens;
            # earlier cached positions are 0..seq_k-seq_q-1
            current_pos = cache_position[-1].item()
            all_positions = torch.arange(
                seq_k, device=self.device, dtype=torch.long,
            )
        else:
            current_pos = seq_k - 1
            all_positions = torch.arange(
                seq_k, device=self.device, dtype=torch.long,
            )

        # --- Recover pre-RoPE keys for ALL positions ---
        # Extract the RoPE portion from cached keys (last rope_dim dims)
        # cached_key_states: [bsz, n_heads, seq_k, qk_head_dim]
        # The rope portion is the last rope_dim dimensions
        k_rot_cached = cached_key_states[0, 0, :, -rope_dim:]  # [seq_k, rope_dim]

        # For past positions (0..seq_k-seq_q-1): invert RoPE
        # For current positions (seq_k-seq_q..seq_k-1): use k_rot_raw directly
        num_past = seq_k - seq_q

        if num_past > 0:
            # Invert RoPE on past cached keys
            # cos_full/sin_full: [bsz, seq_k, rope_dim] — need [num_past, rope_dim]
            cos_past = cos_full[0, :num_past, :]  # [num_past, rope_dim]
            sin_past = sin_full[0, :num_past, :]
            k_rot_past = k_rot_cached[:num_past]  # [num_past, rope_dim]
            k_unrot_past = _invert_rope(
                k_rot_past, cos_past, sin_past,
                self.attention_scale, style="half",
            )
            # Current tokens: use pre-RoPE directly
            k_unrot_current = k_rot_raw_current[0, 0, :, :].to(
                dtype=torch.float32,
            )  # [seq_q, rope_dim]
            k_unrot = torch.cat(
                [k_unrot_past, k_unrot_current], dim=0,
            )  # [seq_k, rope_dim]
        else:
            # Prefill only — all keys are current
            k_unrot = k_rot_raw_current[0, 0, :, :].to(
                dtype=torch.float32,
            )  # [seq_k, rope_dim]

        return self.score_tokens(
            layer_idx=layer_idx,
            k_rot_raw=k_unrot,
            absolute_positions=all_positions,
            current_position=int(current_pos),
            topk=topk,
        )

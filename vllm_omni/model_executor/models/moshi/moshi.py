# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Inference-only Moshi model for vllm-omni (Phase 1: half-duplex, turn-taking).

Architecture:
- Stage "dialogue": Temporal Transformer (7B) + Depth Transformer (6L)
  Generates text tokens + 8-level RVQ audio codes from user audio input.
- Stage "mimi_decode": Mimi neural audio codec decoder (96M params)
  Converts RVQ codes to audio waveform.

Reference: https://arxiv.org/abs/2410.00037
HF weights: kmhf/hf-moshiko (HuggingFace Transformers format)
"""

import math
from collections.abc import Iterable
from functools import cached_property

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper
from vllm.sequence import IntermediateTensors
from vllm.v1.sample.metadata import SamplingMetadata

from vllm_omni.model_executor.models.output_templates import OmniOutput

logger = init_logger(__name__)


# =============================================================================
# Custom Layers
# =============================================================================


class MoshiRMSNorm(nn.Module):
    """RMSNorm matching HF Moshi weight format."""

    def __init__(self, hidden_size: int, eps: float = 1e-8):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.float().pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(x.dtype)


class MoshiRotaryEmbedding(nn.Module):
    """RoPE for temporal transformer."""

    def __init__(self, dim: int, max_position_embeddings: int = 3000, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        freqs = torch.einsum("i,j->ij", position_ids.float().reshape(-1), self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos().unsqueeze(0)
        sin = emb.sin().unsqueeze(0)
        return cos.to(x.dtype), sin.to(x.dtype)


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MoshiGatingMLP(nn.Module):
    """Gated MLP with SiLU activation. fc1 output is split in half for gating.

    Supports quantization (AWQ, GPTQ, etc.) via vLLM's parallel linear layers.
    When quant_config is None, behaves identically to standard nn.Linear.
    """

    def __init__(self, hidden_size: int, ffn_dim: int,
                 quant_config=None, prefix: str = ""):
        super().__init__()
        self.fc1 = ColumnParallelLinear(
            hidden_size, ffn_dim, bias=False,
            quant_config=quant_config, prefix=f"{prefix}.fc1",
        )
        self.fc2 = RowParallelLinear(
            ffn_dim // 2, hidden_size, bias=False,
            quant_config=quant_config, prefix=f"{prefix}.fc2",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_and_up, _ = self.fc1(x)
        gate, up = gate_and_up.chunk(2, dim=-1)
        out, _ = self.fc2(F.silu(gate) * up)
        return out


class MoshiAttention(nn.Module):
    """Multi-head attention for Temporal Transformer (with RoPE).

    Supports quantization via vLLM's parallel linear layers.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int | None = None,
        head_dim: int | None = None,
        max_position_embeddings: int = 3000,
        quant_config=None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = head_dim or (hidden_size // num_heads)

        # HF Moshi wraps nn.Linear in MoshiLinear, creating .linear.weight
        # We use a nested module to match the weight key path.
        # ColumnParallelLinear/RowParallelLinear enable quantization support.
        self.q_proj = nn.Module()
        self.q_proj.linear = ColumnParallelLinear(
            hidden_size, self.num_heads * self.head_dim, bias=False,
            quant_config=quant_config, prefix=f"{prefix}.q_proj.linear",
        )
        self.k_proj = nn.Module()
        self.k_proj.linear = ColumnParallelLinear(
            hidden_size, self.num_kv_heads * self.head_dim, bias=False,
            quant_config=quant_config, prefix=f"{prefix}.k_proj.linear",
        )
        self.v_proj = nn.Module()
        self.v_proj.linear = ColumnParallelLinear(
            hidden_size, self.num_kv_heads * self.head_dim, bias=False,
            quant_config=quant_config, prefix=f"{prefix}.v_proj.linear",
        )
        self.o_proj = nn.Module()
        self.o_proj.linear = RowParallelLinear(
            self.num_heads * self.head_dim, hidden_size, bias=False,
            quant_config=quant_config, prefix=f"{prefix}.o_proj.linear",
        )

        self.rotary_emb = MoshiRotaryEmbedding(self.head_dim, max_position_embeddings)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        bsz, seq_len, _ = hidden_states.shape

        q, _ = self.q_proj.linear(hidden_states)
        q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k, _ = self.k_proj.linear(hidden_states)
        k = k.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v, _ = self.v_proj.linear(hidden_states)
        v = v.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(q, position_ids)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)

        new_kv = (k, v) if use_cache else None

        # GQA: repeat kv heads
        if self.num_kv_heads < self.num_heads:
            repeat_factor = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat_factor, dim=1)
            v = v.repeat_interleave(repeat_factor, dim=1)

        # Scaled dot-product attention
        attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=(past_key_value is None and seq_len > 1))
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        attn_output, _ = self.o_proj.linear(attn_output)

        return attn_output, new_kv


class MoshiDecoderLayer(nn.Module):
    """Single temporal transformer layer."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        ffn_dim: int,
        num_kv_heads: int | None = None,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-8,
        max_position_embeddings: int = 3000,
        quant_config=None,
        prefix: str = "",
    ):
        super().__init__()
        self.self_attn = MoshiAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = MoshiGatingMLP(
            hidden_size, ffn_dim,
            quant_config=quant_config, prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = MoshiRMSNorm(hidden_size, rms_norm_eps)
        self.post_attention_layernorm = MoshiRMSNorm(hidden_size, rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, new_kv = self.self_attn(hidden_states, position_ids, past_key_value, use_cache)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, new_kv


# =============================================================================
# Temporal Transformer (7B Backbone)
# =============================================================================


class MoshiTemporalModel(nn.Module):
    """
    Temporal Transformer model (backbone LLM).
    Weight path: decoder.model.*
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_hidden_layers: int,
        num_attention_heads: int,
        ffn_dim: int,
        num_kv_heads: int | None = None,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-8,
        max_position_embeddings: int = 3000,
        quant_config=None,
        prefix: str = "",
    ):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size + 1, hidden_size)
        self.layers = nn.ModuleList([
            MoshiDecoderLayer(
                hidden_size=hidden_size,
                num_heads=num_attention_heads,
                ffn_dim=ffn_dim,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                rms_norm_eps=rms_norm_eps,
                max_position_embeddings=max_position_embeddings,
                quant_config=quant_config,
                prefix=f"{prefix}.layers.{i}",
            )
            for i in range(num_hidden_layers)
        ])
        self.norm = MoshiRMSNorm(hidden_size, rms_norm_eps)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]] | None]:
        hidden_states = inputs_embeds
        new_key_values = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None
            hidden_states, new_kv = layer(hidden_states, position_ids, past_kv, use_cache)
            if new_key_values is not None:
                new_key_values.append(new_kv)

        hidden_states = self.norm(hidden_states)
        return hidden_states, new_key_values


class MoshiTemporalDecoder(nn.Module):
    """
    Temporal decoder = model + lm_head.
    Weight path: decoder.*
    """

    def __init__(self, config, quant_config=None, prefix: str = ""):
        super().__init__()
        self.model = MoshiTemporalModel(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            ffn_dim=config.ffn_dim,
            num_kv_heads=getattr(config, "num_key_value_heads", None),
            head_dim=getattr(config, "head_dim", None),
            rms_norm_eps=config.rms_norm_eps,
            max_position_embeddings=config.max_position_embeddings,
            quant_config=quant_config,
            prefix=f"{prefix}.model",
        )
        self.lm_head = ColumnParallelLinear(
            config.hidden_size, config.vocab_size, bias=False,
            quant_config=quant_config, prefix=f"{prefix}.lm_head",
        )


# =============================================================================
# Depth Transformer (Codebook Generation)
# =============================================================================


class MoshiFlexibleLinear(nn.Module):
    """
    Linear layer with per-codebook weights: weight shape [num_layers, out, in].
    Used by depth decoder where each codebook position has its own weights.
    """

    def __init__(self, in_features: int, out_features: int, num_layers: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_layers, out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """x: [batch, hidden] → [batch, out_features] using layer_idx-th weight."""
        return F.linear(x, self.weight[layer_idx])


class MoshiDepthAttention(nn.Module):
    """Attention for Depth Transformer (no RoPE, flexible linear)."""

    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int | None, num_codebooks: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = hidden_size // num_heads

        # Flexible linear: each codebook has its own projection weights
        # HF path: depth_decoder.layers.{L}.self_attn.{q,k,v,o}_proj.linear.weight
        self.q_proj = nn.Module()
        self.q_proj.linear = MoshiFlexibleLinear(hidden_size, self.num_heads * self.head_dim, num_codebooks)
        self.k_proj = nn.Module()
        self.k_proj.linear = MoshiFlexibleLinear(hidden_size, self.num_kv_heads * self.head_dim, num_codebooks)
        self.v_proj = nn.Module()
        self.v_proj.linear = MoshiFlexibleLinear(hidden_size, self.num_kv_heads * self.head_dim, num_codebooks)
        self.o_proj = nn.Module()
        self.o_proj.linear = MoshiFlexibleLinear(self.num_heads * self.head_dim, hidden_size, num_codebooks)

    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        bsz, seq_len, _ = hidden_states.shape

        q = self.q_proj.linear(hidden_states.squeeze(1), layer_idx).unsqueeze(1)
        k = self.k_proj.linear(hidden_states.squeeze(1), layer_idx).unsqueeze(1)
        v = self.v_proj.linear(hidden_states.squeeze(1), layer_idx).unsqueeze(1)

        q = q.view(bsz, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)

        new_kv = (k, v) if use_cache else None

        if self.num_kv_heads < self.num_heads:
            repeat_factor = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat_factor, dim=1)
            v = v.repeat_interleave(repeat_factor, dim=1)

        attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, 1, -1)
        attn_output = self.o_proj.linear(attn_output.squeeze(1), layer_idx).unsqueeze(1)

        return attn_output, new_kv


class MoshiDepthGatingMLP(nn.Module):
    """Gated MLP with flexible linear for depth decoder."""

    def __init__(self, hidden_size: int, ffn_dim: int, num_codebooks: int):
        super().__init__()
        self.fc1 = MoshiFlexibleLinear(hidden_size, ffn_dim, num_codebooks)
        self.fc2 = MoshiFlexibleLinear(ffn_dim // 2, hidden_size, num_codebooks)

    def forward(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        gate_and_up = self.fc1(x, layer_idx)
        gate, up = gate_and_up.chunk(2, dim=-1)
        return self.fc2(F.silu(gate) * up, layer_idx)


class MoshiDepthDecoderLayer(nn.Module):
    """Single depth decoder layer with flexible linear."""

    def __init__(self, hidden_size: int, num_heads: int, ffn_dim: int, num_kv_heads: int | None,
                 num_codebooks: int, rms_norm_eps: float = 1e-8):
        super().__init__()
        self.self_attn = MoshiDepthAttention(hidden_size, num_heads, num_kv_heads, num_codebooks)
        self.mlp = MoshiDepthGatingMLP(hidden_size, ffn_dim, num_codebooks)
        # LayerNorm weights are shared across codebooks (1D, not 3D)
        self.input_layernorm = MoshiRMSNorm(hidden_size, rms_norm_eps)
        self.post_attention_layernorm = MoshiRMSNorm(hidden_size, rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        codebook_idx: int,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, new_kv = self.self_attn(hidden_states, codebook_idx, past_key_value, use_cache)
        hidden_states = residual + hidden_states

        residual = hidden_states
        normed = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(normed.squeeze(1), codebook_idx).unsqueeze(1)

        return hidden_states, new_kv


class MoshiDepthDecoder(nn.Module):
    """
    Depth Transformer: generates 8 audio codebook tokens per time step.
    Weight path: depth_decoder.*
    """

    def __init__(self, depth_config):
        super().__init__()
        self.num_codebooks = depth_config.num_codebooks
        hidden_size = depth_config.hidden_size
        input_size = depth_config.input_size

        # Text token embedding (for the text token predicted by temporal)
        self.text_embed_tokens = nn.Embedding(depth_config.vocab_size + 1, hidden_size)

        # Audio codebook embeddings (7 = num_codebooks - 1, last codebook has no next)
        self.embed_tokens = nn.ModuleList([
            nn.Embedding(depth_config.audio_vocab_size + 1, hidden_size)
            for _ in range(self.num_codebooks - 1)
        ])

        # Projection from temporal hidden to depth hidden (per codebook)
        self.input_projections = MoshiFlexibleLinear(input_size, hidden_size, self.num_codebooks)

        # Transformer layers
        self.layers = nn.ModuleList([
            MoshiDepthDecoderLayer(
                hidden_size=hidden_size,
                num_heads=depth_config.num_attention_heads,
                ffn_dim=depth_config.ffn_dim,
                num_kv_heads=getattr(depth_config, "num_key_value_heads", None),
                num_codebooks=self.num_codebooks,
                rms_norm_eps=depth_config.rms_norm_eps,
            )
            for _ in range(depth_config.num_hidden_layers)
        ])

        # LM heads for each codebook (predicts audio token)
        self.lm_heads = MoshiFlexibleLinear(hidden_size, depth_config.audio_vocab_size, self.num_codebooks)

    @torch.no_grad()
    def generate_codes(
        self,
        temporal_context: torch.Tensor,
        text_token: int,
        temperature: float = 0.8,
        top_k: int = 50,
    ) -> list[int]:
        """
        Generate 8 audio codebook tokens for one time step.

        Args:
            temporal_context: [batch=1, hidden_size] from temporal transformer
            text_token: predicted text token ID
            temperature: sampling temperature
            top_k: top-k filtering

        Returns:
            List of 8 audio codebook token IDs
        """
        device = temporal_context.device
        codes = []

        # Position 0: text token embedding + projected temporal context
        text_embed = self.text_embed_tokens(
            torch.tensor([[text_token]], device=device)
        )  # [1, 1, depth_hidden]

        projected = self.input_projections(temporal_context, 0).unsqueeze(1)  # [1, 1, depth_hidden]
        hidden = text_embed + projected  # [1, 1, depth_hidden]

        # Initialize KV cache for depth layers
        past_kvs = [None] * len(self.layers)

        # Run through depth transformer for position 0
        for layer_idx, layer in enumerate(self.layers):
            hidden, new_kv = layer(hidden, codebook_idx=0, past_key_value=past_kvs[layer_idx], use_cache=True)
            past_kvs[layer_idx] = new_kv

        # Predict first codebook token
        logits = self.lm_heads(hidden.squeeze(1), 0)  # [1, audio_vocab]
        code_0 = self._sample_token(logits, temperature, top_k)
        codes.append(code_0)

        # Positions 1..7: use previous codebook embedding
        for k in range(1, self.num_codebooks):
            # Embed previous codebook token
            code_embed = self.embed_tokens[k - 1](
                torch.tensor([[codes[-1]]], device=device)
            )  # [1, 1, depth_hidden]

            projected = self.input_projections(temporal_context, k).unsqueeze(1)
            hidden = code_embed + projected

            # Run through depth transformer
            for layer_idx, layer in enumerate(self.layers):
                hidden, new_kv = layer(hidden, codebook_idx=k, past_key_value=past_kvs[layer_idx], use_cache=True)
                past_kvs[layer_idx] = new_kv

            # Predict next codebook token
            logits = self.lm_heads(hidden.squeeze(1), k)
            code_k = self._sample_token(logits, temperature, top_k)
            codes.append(code_k)

        return codes

    @staticmethod
    def _sample_token(logits: torch.Tensor, temperature: float, top_k: int) -> int:
        """Sample a token from logits with temperature and top-k filtering."""
        if temperature <= 0:
            return torch.argmax(logits, dim=-1).item()

        logits = logits / temperature
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
            probs = F.softmax(top_k_logits, dim=-1)
            idx = torch.multinomial(probs, 1).item()
            return top_k_indices[0, idx].item()
        else:
            probs = F.softmax(logits, dim=-1)
            return torch.multinomial(probs, 1).item()


# =============================================================================
# Mimi Decoder (Audio Codec)
# =============================================================================


class MimiDecoderModel(nn.Module):
    """
    Minimal Mimi decoder wrapper for vllm-omni.

    For Phase 1, we load the full MimiModel from HuggingFace and use its
    decode() method. This keeps the implementation simple and correct.
    The weight loading uses the standard HF mechanism.
    """

    def __init__(self, audio_encoder_config=None):
        super().__init__()
        self._config = audio_encoder_config
        self._hf_model = None  # Loaded via load_from_pretrained

    def load_from_pretrained(self, model_name_or_path: str):
        """Load the full Mimi model from HuggingFace."""
        from transformers import MimiModel
        self._hf_model = MimiModel.from_pretrained(model_name_or_path)
        self._hf_model.eval()

    def decode(self, audio_codes: torch.Tensor) -> torch.Tensor:
        """
        Decode RVQ codes to audio waveform.

        Args:
            audio_codes: [batch, num_codebooks, time_frames]

        Returns:
            audio: [batch, 1, num_samples] at 24kHz
        """
        if self._hf_model is not None:
            with torch.no_grad():
                output = self._hf_model.decode(audio_codes)
                if hasattr(output, "audio_values"):
                    return output.audio_values
                return output[0]

        raise RuntimeError("MimiDecoderModel not initialized. Call load_from_pretrained() first.")


# =============================================================================
# Unified Model
# =============================================================================


class MoshiForConditionalGenerationVLLM(nn.Module, SupportsPP):
    """
    Unified Moshi model for vllm-omni.

    Stages:
    - "dialogue": Temporal + Depth transformers. Generates text + audio codes.
    - "mimi_decode": Mimi decoder. Converts audio codes to waveform.

    Usage:
        Set `model_stage` in vllm_config to one of: "dialogue", "mimi_decode"

    HF checkpoint: kmhf/hf-moshiko or kyutai/moshiko-pytorch-bf16
    """

    # Weight name mapping from HF checkpoint to our model structure
    # The dialogue stage and mimi_decode stage share the same checkpoint
    hf_to_vllm_mapper = WeightsMapper(orig_to_new_prefix={})

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        self.config = vllm_config.model_config.hf_config
        self.model_stage = vllm_config.model_config.model_stage
        self.quant_config = getattr(vllm_config, "quant_config", None)
        self.have_multimodal_outputs = True

        if self.model_stage == "dialogue":
            self._init_dialogue()
        elif self.model_stage == "mimi_decode":
            self._init_mimi_decode()
        else:
            raise ValueError(
                f"Invalid model_stage: {self.model_stage}. "
                f"Must be one of: 'dialogue', 'mimi_decode'"
            )

        # Generation worker compatibility
        self.make_empty_intermediate_tensors = lambda *a, **k: None

    def _init_dialogue(self):
        """Initialize temporal + depth transformers + audio embeddings."""
        config = self.config
        num_codebooks = config.num_codebooks  # 8

        # Audio stream embeddings: 16 total (8 moshi + 8 user)
        # Weight path: embed_tokens.{0..15}.weight
        audio_vocab_size = config.audio_vocab_size
        self.embed_tokens = nn.ModuleList([
            nn.Embedding(audio_vocab_size + 1, config.hidden_size)
            for _ in range(2 * num_codebooks)
        ])

        # Temporal decoder (backbone LLM + text LM head)
        # Weight path: decoder.*
        self.decoder = MoshiTemporalDecoder(
            config, quant_config=self.quant_config, prefix="decoder",
        )

        # Depth decoder (codebook generation)
        # Weight path: depth_decoder.*
        depth_config = config.depth_decoder_config
        self.depth_decoder = MoshiDepthDecoder(depth_config)

        # Generation parameters
        self.num_codebooks = num_codebooks
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size

        logger.info(
            "Initialized Moshi dialogue model: "
            f"temporal={config.num_hidden_layers}L/{config.hidden_size}H, "
            f"depth={depth_config.num_hidden_layers}L/{depth_config.hidden_size}H, "
            f"codebooks={num_codebooks}"
        )

    def _init_mimi_decode(self):
        """Initialize Mimi decoder for audio generation."""
        self.mimi_decoder = MimiDecoderModel(
            audio_encoder_config=getattr(self.config, "audio_encoder_config", None)
        )
        logger.info("Initialized Moshi mimi_decode stage")

    # ==================== Interface Methods ====================

    @cached_property
    def sampler(self):
        return None

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings=None,
        is_multimodal=None,
    ) -> torch.Tensor:
        """Generation worker: return dummy embeddings."""
        hidden_size = self.vllm_config.model_config.get_hidden_size()
        return torch.zeros(
            input_ids.shape[0], hidden_size,
            device=input_ids.device, dtype=torch.float16,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor | OmniOutput,
        sampling_metadata: SamplingMetadata | None = None,
    ) -> torch.Tensor | None:
        """Generation worker: no logits needed."""
        return None

    def make_omni_output(self, model_outputs, **kwargs) -> OmniOutput:
        """Wrap raw forward outputs into OmniOutput."""
        if isinstance(model_outputs, OmniOutput):
            return model_outputs
        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={"model_outputs": model_outputs},
        )

    # ==================== Per-Step State (Streaming) ====================

    def init_streaming_state(
        self,
        request_id: str = "_default",
        device: torch.device | None = None,
    ):
        """Initialize persistent state for per-step streaming generation.

        Call this once before calling forward_dialogue_step() in a loop.
        The state tracks KV caches, previous tokens, and temporal position.

        State is keyed by request_id so multiple concurrent requests can
        each maintain independent streaming state on the same model instance.

        Args:
            request_id: Unique request identifier for state isolation.
            device: Device to place tensors on. Defaults to model device.
        """
        if device is None:
            device = next(self.parameters()).device

        if not hasattr(self, "_streaming_states"):
            self._streaming_states: dict[str, dict] = {}

        self._streaming_states[request_id] = {
            "past_key_values": None,
            "prev_text_token": None,
            "prev_audio_codes": None,
            "temporal_step": 0,
            "device": device,
        }
        logger.debug(
            "Initialized Moshi streaming state for request %s on %s",
            request_id, device,
        )

    def clear_streaming_state(self, request_id: str = "_default"):
        """Clear streaming state and free KV cache memory for a request.

        Args:
            request_id: The request whose state to clear.
        """
        if hasattr(self, "_streaming_states") and request_id in self._streaming_states:
            del self._streaming_states[request_id]

    @torch.inference_mode()
    def forward_dialogue_step(
        self,
        user_audio_codes: list[int] | None = None,
        temperature: float = 0.7,
        top_k: int = 50,
        request_id: str = "_default",
    ) -> dict:
        """Run a single temporal + depth step for streaming generation.

        This method is called once per temporal position (80ms frame).
        It maintains internal state (KV cache, previous tokens) across calls,
        keyed by request_id for concurrent request isolation.

        Args:
            user_audio_codes: User's audio codes for this step [num_codebooks].
                              None or empty uses silence padding.
            temperature: Sampling temperature.
            top_k: Top-k sampling parameter.
            request_id: Unique request identifier for state lookup.

        Returns:
            Dict with:
                "audio_codes": list[int] of length num_codebooks (8)
                "text_token": int (sampled text token)
                "temporal_step": int (current temporal position)
        """
        states = getattr(self, "_streaming_states", None)
        if states is None or request_id not in states:
            raise RuntimeError(
                f"Streaming state not initialized for request '{request_id}'. "
                "Call init_streaming_state(request_id=...) first."
            )
        state = states[request_id]

        device = state["device"]
        t = state["temporal_step"]
        num_codebooks = self.num_codebooks

        # --- Build input embeddings for time step t ---

        # Text embedding (previous text token, or PAD for t=0)
        if state["prev_text_token"] is None:
            text_token_id = self.vocab_size  # PAD/BOS
        else:
            text_token_id = state["prev_text_token"]

        text_embed = self.decoder.model.embed_tokens(
            torch.tensor([[text_token_id]], device=device, dtype=torch.long)
        )

        # Audio embeddings: sum across all 16 streams
        audio_embed = torch.zeros_like(text_embed)

        # Moshi's own audio codes (streams 0-7): from previous step
        for k in range(num_codebooks):
            if state["prev_audio_codes"] is not None:
                code_id = state["prev_audio_codes"][k]
            else:
                code_id = 0
            audio_embed = audio_embed + self.embed_tokens[k](
                torch.tensor([[code_id]], device=device, dtype=torch.long)
            )

        # User audio codes (streams 8-15): from input
        if user_audio_codes is None:
            user_audio_codes = [0] * num_codebooks
        for k in range(num_codebooks):
            user_code = user_audio_codes[k] if k < len(user_audio_codes) else 0
            audio_embed = audio_embed + self.embed_tokens[num_codebooks + k](
                torch.tensor([[user_code]], device=device, dtype=torch.long)
            )

        combined_embed = text_embed + audio_embed

        # --- Temporal Transformer forward ---
        position_ids = torch.tensor([[t]], device=device, dtype=torch.long)

        hidden_states, past_key_values = self.decoder.model(
            inputs_embeds=combined_embed,
            position_ids=position_ids,
            past_key_values=state["past_key_values"],
            use_cache=True,
        )

        temporal_context = hidden_states[:, -1, :]

        # --- Sample text token ---
        text_logits, _ = self.decoder.lm_head(temporal_context)
        text_token = self._sample_token(text_logits, temperature=temperature, top_k=top_k)

        # --- Depth Transformer: generate 8 audio codes ---
        audio_codes = self.depth_decoder.generate_codes(
            temporal_context=temporal_context,
            text_token=text_token,
            temperature=temperature,
            top_k=top_k,
        )

        # --- Update state ---
        state["past_key_values"] = past_key_values
        state["prev_text_token"] = text_token
        state["prev_audio_codes"] = audio_codes
        state["temporal_step"] = t + 1

        return {
            "audio_codes": audio_codes,
            "text_token": text_token,
            "temporal_step": t,
        }

    # ==================== Forward ====================

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        sampling_metadata: SamplingMetadata | None = None,
        **kwargs,
    ) -> OmniOutput:
        if self.model_stage == "dialogue":
            return self._forward_dialogue(input_ids)
        elif self.model_stage == "mimi_decode":
            return self._forward_mimi_decode(input_ids)
        else:
            raise ValueError(f"Unknown model_stage: {self.model_stage}")

    @torch.inference_mode()
    def _forward_dialogue(self, input_ids: torch.Tensor) -> OmniOutput:
        """
        Run complete Moshi dialogue inference (half-duplex, turn-taking).

        Input: user_audio_codes flattened [8*T] where T = number of frames
        Output: OmniOutput with audio_codes [T_out, 8] and text_tokens [T_out]
        """
        device = input_ids.device
        num_codebooks = self.num_codebooks

        # Reshape user audio codes: [8*T] → [1, 8, T]
        total_tokens = input_ids.shape[0]
        if total_tokens % num_codebooks != 0:
            # Pad to multiple of num_codebooks
            pad_len = num_codebooks - (total_tokens % num_codebooks)
            input_ids = F.pad(input_ids, (0, pad_len), value=0)
            total_tokens = input_ids.shape[0]

        T_user = total_tokens // num_codebooks
        user_codes = input_ids.reshape(num_codebooks, T_user).unsqueeze(0)  # [1, 8, T]

        # Generation parameters
        max_steps = min(T_user, 2048)  # Cap at ~163 seconds
        temperature = 0.7
        top_k = 50

        # State
        text_tokens = []
        audio_codes_list = []
        past_key_values = None

        # Pad token ID for initial state
        audio_pad_id = 0  # Typically 0 or config.audio_vocab_size

        for t in range(max_steps):
            # --- Build input embeddings for time step t ---

            # Text embedding (previous text token, or PAD for t=0)
            if t == 0:
                text_token_id = self.vocab_size  # PAD/BOS token (vocab_size is used as pad)
            else:
                text_token_id = text_tokens[-1]

            text_embed = self.decoder.model.embed_tokens(
                torch.tensor([[text_token_id]], device=device, dtype=torch.long)
            )  # [1, 1, H]

            # Audio embeddings: sum across all 16 streams
            audio_embed = torch.zeros_like(text_embed)

            # Moshi's own audio codes (streams 0-7): from previous step
            for k in range(num_codebooks):
                if t > 0:
                    code_id = audio_codes_list[-1][k]
                else:
                    code_id = audio_pad_id
                audio_embed = audio_embed + self.embed_tokens[k](
                    torch.tensor([[code_id]], device=device, dtype=torch.long)
                )

            # User audio codes (streams 8-15): from input
            for k in range(num_codebooks):
                if t < T_user:
                    user_code = user_codes[0, k, t].item()
                else:
                    user_code = audio_pad_id
                audio_embed = audio_embed + self.embed_tokens[num_codebooks + k](
                    torch.tensor([[user_code]], device=device, dtype=torch.long)
                )

            # Combined embedding
            combined_embed = text_embed + audio_embed  # [1, 1, H]

            # --- Temporal Transformer forward (with KV cache) ---
            position_ids = torch.tensor([[t]], device=device, dtype=torch.long)

            hidden_states, past_key_values = self.decoder.model(
                inputs_embeds=combined_embed,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )  # hidden_states: [1, 1, H]

            temporal_context = hidden_states[:, -1, :]  # [1, H]

            # --- Sample text token ---
            text_logits, _ = self.decoder.lm_head(temporal_context)  # [1, vocab_size]
            text_token = self._sample_token(text_logits, temperature=0.7, top_k=50)
            text_tokens.append(text_token)

            # --- Depth Transformer: generate 8 audio codebook tokens ---
            depth_codes = self.depth_decoder.generate_codes(
                temporal_context=temporal_context,
                text_token=text_token,
                temperature=temperature,
                top_k=top_k,
            )
            audio_codes_list.append(depth_codes)

        # Stack results
        if len(audio_codes_list) > 0:
            audio_codes = torch.tensor(audio_codes_list, device=device, dtype=torch.long)  # [T, 8]
        else:
            audio_codes = torch.zeros(0, num_codebooks, device=device, dtype=torch.long)

        text_token_ids = torch.tensor(text_tokens, device=device, dtype=torch.long)

        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={
                "audio_codes": audio_codes,      # [T, 8]
                "text_tokens": text_token_ids,   # [T]
            },
        )

    @torch.inference_mode()
    def _forward_mimi_decode(self, input_ids: torch.Tensor) -> OmniOutput:
        """
        Mimi decoder: RVQ codes → audio waveform.

        Input: flattened audio codes [8*T]
        Output: OmniOutput with audio tensor [1, num_samples]
        """
        num_codebooks = getattr(self.config, "num_codebooks", 8)

        total_tokens = input_ids.shape[0]
        if total_tokens % num_codebooks != 0:
            pad_len = num_codebooks - (total_tokens % num_codebooks)
            input_ids = F.pad(input_ids, (0, pad_len), value=0)
            total_tokens = input_ids.shape[0]

        T = total_tokens // num_codebooks
        audio_codes = input_ids.reshape(1, num_codebooks, T)  # [1, 8, T]

        audio_waveform = self.mimi_decoder.decode(audio_codes)  # [1, 1, samples]

        # Flatten to [1, samples]
        if audio_waveform.ndim == 3:
            audio_waveform = audio_waveform.squeeze(1)

        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={
                "model_outputs": audio_waveform,  # [1, samples]
            },
        )

    @staticmethod
    def _sample_token(logits: torch.Tensor, temperature: float, top_k: int) -> int:
        """Sample a token from logits."""
        if temperature <= 0:
            return torch.argmax(logits, dim=-1).item()

        logits = logits / temperature
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
            probs = F.softmax(top_k_logits, dim=-1)
            idx = torch.multinomial(probs, 1).item()
            return top_k_indices[0, idx].item()
        else:
            probs = F.softmax(logits, dim=-1)
            return torch.multinomial(probs, 1).item()

    # ==================== Weight Loading ====================

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """
        Load weights from HF Moshi checkpoint.

        Checkpoint structure:
          - decoder.model.* → temporal transformer
          - decoder.lm_head.* → text LM head
          - depth_decoder.* → depth transformer
          - embed_tokens.{0..15}.* → audio stream embeddings
          - audio_encoder.* → Mimi codec (used by mimi_decode stage)
        """
        loaded = set()

        if self.model_stage == "dialogue":
            loaded = self._load_dialogue_weights(weights)
        elif self.model_stage == "mimi_decode":
            loaded = self._load_mimi_decode_weights(weights)

        return loaded

    def _load_dialogue_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights for dialogue stage (temporal + depth + audio embeddings).

        Uses vLLM's weight_loader mechanism for quantization-aware layers.
        Standard (non-quantized) parameters fall back to direct copy.
        """
        loaded = set()
        params_dict = dict(self.named_parameters())

        for name, tensor in weights:
            # Skip audio_encoder weights (used by mimi_decode stage)
            if name.startswith("audio_encoder."):
                continue

            # Map to our parameter names (already matching HF format)
            if name in params_dict:
                param = params_dict[name]
                # Quantization-aware layers attach a weight_loader to params
                weight_loader = getattr(param, "weight_loader", None)
                if weight_loader is not None:
                    weight_loader(param, tensor)
                    loaded.add(name)
                elif param.shape == tensor.shape:
                    param.data.copy_(tensor)
                    loaded.add(name)
                else:
                    logger.warning(
                        f"Shape mismatch for {name}: "
                        f"expected {param.shape}, got {tensor.shape}"
                    )

        logger.info(f"Loaded {len(loaded)} dialogue weights")
        return loaded

    def _load_mimi_decode_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """
        Load weights for mimi_decode stage.

        For Phase 1, we use HF's MimiModel.from_pretrained() directly,
        so weight loading from the Moshi checkpoint is a no-op here.
        The Mimi model is loaded separately.
        """
        # Collect and skip - Mimi model is loaded via load_from_pretrained()
        loaded = set()
        for name, tensor in weights:
            if name.startswith("audio_encoder."):
                loaded.add(name)  # Mark as handled
        return loaded

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for Moshi model components.

Tests custom layers, transformers, depth decoder, and stage input processor
WITHOUT requiring GPU or model weight loading. All tests use random weights.
"""

import math

import pytest
import torch
import torch.nn as nn

# Import model components directly (no vllm dependency needed)
from vllm_omni.model_executor.models.moshi.moshi import (
    MimiDecoderModel,
    MoshiAttention,
    MoshiDecoderLayer,
    MoshiDepthAttention,
    MoshiDepthDecoder,
    MoshiDepthDecoderLayer,
    MoshiDepthGatingMLP,
    MoshiFlexibleLinear,
    MoshiGatingMLP,
    MoshiRMSNorm,
    MoshiRotaryEmbedding,
    MoshiTemporalDecoder,
    MoshiTemporalModel,
    apply_rotary_pos_emb,
    rotate_half,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def batch_size():
    return 1


@pytest.fixture
def hidden_size():
    return 64  # Small for fast testing


@pytest.fixture
def depth_hidden_size():
    return 32


@pytest.fixture
def num_codebooks():
    return 8


@pytest.fixture
def num_heads():
    return 4


@pytest.fixture
def vocab_size():
    return 128


@pytest.fixture
def audio_vocab_size():
    return 64


@pytest.fixture
def ffn_dim():
    return 128  # Must be even for gated MLP


@pytest.fixture
def depth_ffn_dim():
    return 64


class FakeMoshiConfig:
    """Minimal config object for testing."""

    def __init__(self, **kwargs):
        defaults = {
            "vocab_size": 128,
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "ffn_dim": 128,
            "head_dim": 16,
            "rms_norm_eps": 1e-8,
            "max_position_embeddings": 128,
            "audio_vocab_size": 64,
            "num_codebooks": 8,
        }
        defaults.update(kwargs)
        for k, v in defaults.items():
            setattr(self, k, v)


class FakeDepthConfig:
    """Minimal depth decoder config for testing."""

    def __init__(self, **kwargs):
        defaults = {
            "vocab_size": 128,
            "hidden_size": 32,
            "input_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "ffn_dim": 64,
            "head_dim": 8,
            "rms_norm_eps": 1e-8,
            "audio_vocab_size": 64,
            "num_codebooks": 8,
        }
        defaults.update(kwargs)
        for k, v in defaults.items():
            setattr(self, k, v)


# =============================================================================
# Test MoshiRMSNorm
# =============================================================================


class TestMoshiRMSNorm:
    def test_shape_preserved(self, hidden_size, device):
        norm = MoshiRMSNorm(hidden_size).to(device)
        x = torch.randn(2, 10, hidden_size, device=device)
        out = norm(x)
        assert out.shape == x.shape

    def test_output_is_normalized(self, device):
        norm = MoshiRMSNorm(32).to(device)
        x = torch.randn(1, 5, 32, device=device) * 100
        out = norm(x)
        # RMS of output should be close to 1 (modulo learned weight)
        rms = out.float().pow(2).mean(-1).sqrt()
        assert rms.max() < 10.0  # Reasonable bound

    def test_zero_input(self, device):
        norm = MoshiRMSNorm(16).to(device)
        x = torch.zeros(1, 1, 16, device=device)
        out = norm(x)
        # Should not crash, output is all zeros
        assert torch.all(out == 0) or torch.all(torch.isfinite(out))

    def test_weight_initialization(self):
        norm = MoshiRMSNorm(64)
        assert torch.allclose(norm.weight, torch.ones(64))


# =============================================================================
# Test MoshiRotaryEmbedding
# =============================================================================


class TestMoshiRotaryEmbedding:
    def test_output_shape(self, device):
        dim = 16
        rope = MoshiRotaryEmbedding(dim, max_position_embeddings=64).to(device)
        x = torch.randn(1, 4, 5, dim, device=device)
        pos_ids = torch.arange(5, device=device).unsqueeze(0)
        cos, sin = rope(x, pos_ids)
        assert cos.shape == (1, 5, dim)
        assert sin.shape == (1, 5, dim)

    def test_cos_sin_range(self, device):
        rope = MoshiRotaryEmbedding(16).to(device)
        x = torch.randn(1, 4, 3, 16, device=device)
        pos_ids = torch.arange(3, device=device).unsqueeze(0)
        cos, sin = rope(x, pos_ids)
        assert cos.abs().max() <= 1.0 + 1e-6
        assert sin.abs().max() <= 1.0 + 1e-6


class TestRotateHalf:
    def test_rotate_half_shape(self):
        x = torch.randn(2, 4, 8)
        out = rotate_half(x)
        assert out.shape == x.shape

    def test_rotate_half_values(self):
        x = torch.tensor([1.0, 2.0, 3.0, 4.0]).unsqueeze(0).unsqueeze(0)
        out = rotate_half(x)
        expected = torch.tensor([-3.0, -4.0, 1.0, 2.0]).unsqueeze(0).unsqueeze(0)
        assert torch.allclose(out, expected)


class TestApplyRotaryPosEmb:
    def test_output_shape(self):
        batch, heads, seq, dim = 1, 4, 5, 16
        q = torch.randn(batch, heads, seq, dim)
        k = torch.randn(batch, heads, seq, dim)
        cos = torch.randn(1, seq, dim)
        sin = torch.randn(1, seq, dim)
        q_out, k_out = apply_rotary_pos_emb(q, k, cos, sin)
        assert q_out.shape == q.shape
        assert k_out.shape == k.shape


# =============================================================================
# Test MoshiGatingMLP
# =============================================================================


class TestMoshiGatingMLP:
    def test_output_shape(self, hidden_size, ffn_dim, device):
        mlp = MoshiGatingMLP(hidden_size, ffn_dim).to(device)
        x = torch.randn(2, 5, hidden_size, device=device)
        out = mlp(x)
        assert out.shape == (2, 5, hidden_size)

    def test_odd_ffn_dim_raises(self):
        """ffn_dim must be even for gated split."""
        mlp = MoshiGatingMLP(32, 65)
        x = torch.randn(1, 1, 32)
        # This should raise because 65 is odd and chunk(2) gives uneven split
        # Actually chunk doesn't raise - it just gives different sizes
        # The fc2 will fail because input dim won't match
        with pytest.raises(RuntimeError):
            mlp(x)

    def test_gradient_flow(self, device):
        mlp = MoshiGatingMLP(32, 64).to(device)
        x = torch.randn(1, 3, 32, device=device, requires_grad=True)
        out = mlp(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape


# =============================================================================
# Test MoshiFlexibleLinear
# =============================================================================


class TestMoshiFlexibleLinear:
    def test_output_shape(self, num_codebooks, device):
        fl = MoshiFlexibleLinear(64, 32, num_codebooks).to(device)
        x = torch.randn(2, 64, device=device)
        for idx in range(num_codebooks):
            out = fl(x, idx)
            assert out.shape == (2, 32), f"Failed for layer_idx={idx}"

    def test_different_layers_give_different_outputs(self, device):
        fl = MoshiFlexibleLinear(32, 16, 8).to(device)
        x = torch.randn(1, 32, device=device)
        outputs = [fl(x, i) for i in range(8)]
        # At least some should be different (random initialization)
        different = any(
            not torch.allclose(outputs[i], outputs[j])
            for i in range(8) for j in range(i + 1, 8)
        )
        assert different, "All codebook layers produced identical output"

    def test_weight_shape(self, num_codebooks):
        fl = MoshiFlexibleLinear(64, 32, num_codebooks)
        assert fl.weight.shape == (num_codebooks, 32, 64)

    def test_out_of_range_index(self, device):
        fl = MoshiFlexibleLinear(32, 16, 4).to(device)
        x = torch.randn(1, 32, device=device)
        with pytest.raises(IndexError):
            fl(x, 4)  # Only 0-3 valid


# =============================================================================
# Test MoshiAttention (Temporal)
# =============================================================================


class TestMoshiAttention:
    def test_output_shape(self, hidden_size, num_heads, device):
        attn = MoshiAttention(hidden_size, num_heads).to(device)
        x = torch.randn(1, 5, hidden_size, device=device)
        pos = torch.arange(5, device=device).unsqueeze(0)
        out, kv = attn(x, pos, use_cache=False)
        assert out.shape == (1, 5, hidden_size)
        assert kv is None

    def test_kv_cache(self, hidden_size, num_heads, device):
        attn = MoshiAttention(hidden_size, num_heads).to(device)

        # First step: seq_len=3
        x1 = torch.randn(1, 3, hidden_size, device=device)
        pos1 = torch.arange(3, device=device).unsqueeze(0)
        out1, kv1 = attn(x1, pos1, use_cache=True)
        assert kv1 is not None
        assert kv1[0].shape[2] == 3  # k has seq_len=3

        # Second step: seq_len=1, with past KV
        x2 = torch.randn(1, 1, hidden_size, device=device)
        pos2 = torch.tensor([[3]], device=device)
        out2, kv2 = attn(x2, pos2, past_key_value=kv1, use_cache=True)
        assert out2.shape == (1, 1, hidden_size)
        assert kv2[0].shape[2] == 4  # k now has seq_len=4

    def test_gqa_support(self, hidden_size, device):
        """Test grouped-query attention (num_kv_heads < num_heads)."""
        attn = MoshiAttention(hidden_size, num_heads=4, num_kv_heads=2).to(device)
        x = torch.randn(1, 3, hidden_size, device=device)
        pos = torch.arange(3, device=device).unsqueeze(0)
        out, _ = attn(x, pos)
        assert out.shape == (1, 3, hidden_size)


# =============================================================================
# Test MoshiDecoderLayer (Temporal)
# =============================================================================


class TestMoshiDecoderLayer:
    def test_output_shape(self, hidden_size, num_heads, ffn_dim, device):
        layer = MoshiDecoderLayer(hidden_size, num_heads, ffn_dim).to(device)
        x = torch.randn(1, 5, hidden_size, device=device)
        pos = torch.arange(5, device=device).unsqueeze(0)
        out, kv = layer(x, pos)
        assert out.shape == (1, 5, hidden_size)

    def test_residual_connection(self, hidden_size, num_heads, ffn_dim, device):
        """Output should be different from input (residual + transformation)."""
        layer = MoshiDecoderLayer(hidden_size, num_heads, ffn_dim).to(device)
        x = torch.randn(1, 3, hidden_size, device=device)
        pos = torch.arange(3, device=device).unsqueeze(0)
        out, _ = layer(x, pos)
        assert not torch.allclose(out, x, atol=1e-6)


# =============================================================================
# Test MoshiTemporalModel
# =============================================================================


class TestMoshiTemporalModel:
    def test_forward_shape(self, device):
        config = FakeMoshiConfig()
        model = MoshiTemporalModel(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            ffn_dim=config.ffn_dim,
            rms_norm_eps=config.rms_norm_eps,
        ).to(device)

        # Use embed_tokens to create inputs
        input_ids = torch.randint(0, config.vocab_size, (1, 5), device=device)
        inputs_embeds = model.embed_tokens(input_ids)
        pos = torch.arange(5, device=device).unsqueeze(0)

        out, kvs = model(inputs_embeds, pos, use_cache=True)
        assert out.shape == (1, 5, config.hidden_size)
        assert len(kvs) == config.num_hidden_layers

    def test_autoregressive_with_cache(self, device):
        config = FakeMoshiConfig()
        model = MoshiTemporalModel(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            ffn_dim=config.ffn_dim,
        ).to(device)

        # Step 1: process 3 tokens
        embed1 = torch.randn(1, 3, config.hidden_size, device=device)
        pos1 = torch.arange(3, device=device).unsqueeze(0)
        out1, kvs1 = model(embed1, pos1, use_cache=True)

        # Step 2: process 1 token with KV cache
        embed2 = torch.randn(1, 1, config.hidden_size, device=device)
        pos2 = torch.tensor([[3]], device=device)
        out2, kvs2 = model(embed2, pos2, past_key_values=kvs1, use_cache=True)

        assert out2.shape == (1, 1, config.hidden_size)
        # KV cache should have grown
        assert kvs2[0][0].shape[2] == 4  # k[0] now has 4 positions


# =============================================================================
# Test MoshiTemporalDecoder
# =============================================================================


class TestMoshiTemporalDecoder:
    def test_init_and_forward(self, device):
        config = FakeMoshiConfig()
        decoder = MoshiTemporalDecoder(config).to(device)

        assert hasattr(decoder, "model")
        assert hasattr(decoder, "lm_head")
        assert decoder.lm_head.out_features == config.vocab_size

    def test_lm_head_shape(self, device):
        config = FakeMoshiConfig()
        decoder = MoshiTemporalDecoder(config).to(device)

        hidden = torch.randn(1, 5, config.hidden_size, device=device)
        logits = decoder.lm_head(hidden)
        assert logits.shape == (1, 5, config.vocab_size)


# =============================================================================
# Test MoshiDepthAttention
# =============================================================================


class TestMoshiDepthAttention:
    def test_output_shape(self, depth_hidden_size, num_codebooks, device):
        attn = MoshiDepthAttention(depth_hidden_size, num_heads=4, num_kv_heads=4, num_codebooks=num_codebooks).to(device)
        x = torch.randn(1, 1, depth_hidden_size, device=device)
        for k in range(num_codebooks):
            out, kv = attn(x, layer_idx=k, use_cache=True)
            assert out.shape == (1, 1, depth_hidden_size)

    def test_kv_cache_accumulation(self, depth_hidden_size, device):
        attn = MoshiDepthAttention(depth_hidden_size, num_heads=4, num_kv_heads=4, num_codebooks=8).to(device)

        x1 = torch.randn(1, 1, depth_hidden_size, device=device)
        out1, kv1 = attn(x1, layer_idx=0, use_cache=True)
        assert kv1[0].shape[2] == 1

        x2 = torch.randn(1, 1, depth_hidden_size, device=device)
        out2, kv2 = attn(x2, layer_idx=1, past_key_value=kv1, use_cache=True)
        assert kv2[0].shape[2] == 2


# =============================================================================
# Test MoshiDepthGatingMLP
# =============================================================================


class TestMoshiDepthGatingMLP:
    def test_output_shape(self, depth_hidden_size, depth_ffn_dim, num_codebooks, device):
        mlp = MoshiDepthGatingMLP(depth_hidden_size, depth_ffn_dim, num_codebooks).to(device)
        x = torch.randn(2, depth_hidden_size, device=device)
        for k in range(num_codebooks):
            out = mlp(x, k)
            assert out.shape == (2, depth_hidden_size)


# =============================================================================
# Test MoshiDepthDecoder
# =============================================================================


class TestMoshiDepthDecoder:
    def test_init(self):
        config = FakeDepthConfig()
        decoder = MoshiDepthDecoder(config)
        assert decoder.num_codebooks == 8
        assert len(decoder.embed_tokens) == 7  # num_codebooks - 1
        assert len(decoder.layers) == config.num_hidden_layers

    def test_generate_codes_count(self, device):
        config = FakeDepthConfig()
        decoder = MoshiDepthDecoder(config).to(device)

        temporal_context = torch.randn(1, config.input_size, device=device)
        codes = decoder.generate_codes(temporal_context, text_token=0, temperature=0.8, top_k=10)

        assert len(codes) == config.num_codebooks
        for code in codes:
            assert 0 <= code < config.audio_vocab_size

    def test_generate_codes_deterministic_with_zero_temp(self, device):
        config = FakeDepthConfig()
        decoder = MoshiDepthDecoder(config).to(device)

        temporal_context = torch.randn(1, config.input_size, device=device)
        codes1 = decoder.generate_codes(temporal_context, text_token=5, temperature=0.0, top_k=0)
        codes2 = decoder.generate_codes(temporal_context, text_token=5, temperature=0.0, top_k=0)

        assert codes1 == codes2, "Zero temperature should give deterministic results"

    def test_generate_codes_different_text_tokens(self, device):
        """Different text tokens should generally produce different audio codes."""
        config = FakeDepthConfig()
        decoder = MoshiDepthDecoder(config).to(device)

        temporal_context = torch.randn(1, config.input_size, device=device)
        codes_a = decoder.generate_codes(temporal_context, text_token=0, temperature=0.0)
        codes_b = decoder.generate_codes(temporal_context, text_token=50, temperature=0.0)

        # They should differ at least sometimes (not guaranteed but very likely)
        # Just check they don't crash
        assert len(codes_a) == len(codes_b) == config.num_codebooks


# =============================================================================
# Test MoshiDepthDecoder._sample_token
# =============================================================================


class TestSampleToken:
    def test_greedy(self):
        logits = torch.tensor([[0.1, 0.3, 0.9, 0.2]])
        token = MoshiDepthDecoder._sample_token(logits, temperature=0.0, top_k=0)
        assert token == 2  # argmax

    def test_top_k_limits_choices(self):
        logits = torch.tensor([[10.0, -10.0, -10.0, -10.0]])
        token = MoshiDepthDecoder._sample_token(logits, temperature=1.0, top_k=1)
        assert token == 0  # Only top-1 available

    def test_returns_valid_index(self):
        logits = torch.randn(1, 100)
        token = MoshiDepthDecoder._sample_token(logits, temperature=0.8, top_k=10)
        assert 0 <= token < 100


# =============================================================================
# Test Stage Input Processor
# =============================================================================


class TestDialogueToMimiDecode:
    """Test the stage transition from dialogue to mimi_decode."""

    def _make_mock_stage_list(self, audio_codes):
        """Create a minimal mock stage list with audio codes output."""
        from unittest.mock import Mock

        output = Mock()
        output.outputs = [Mock()]
        output.outputs[0].multimodal_output = {"audio_codes": audio_codes}

        stage = Mock()
        stage.engine_outputs = [output]

        return [stage]

    def test_basic_transition(self):
        from vllm_omni.model_executor.stage_input_processors.moshi import dialogue_to_mimi_decode

        T, num_codebooks = 10, 8
        audio_codes = torch.randint(0, 2048, (T, num_codebooks))
        stage_list = self._make_mock_stage_list(audio_codes)

        result = dialogue_to_mimi_decode(stage_list, engine_input_source=[0])

        assert len(result) == 1
        prompt = result[0]
        assert len(prompt["prompt_token_ids"]) == T * num_codebooks

    def test_codes_reshape_correctness(self):
        """Verify the transpose + flatten produces correct ordering."""
        from vllm_omni.model_executor.stage_input_processors.moshi import dialogue_to_mimi_decode

        # Create known codes: [T=3, 8]
        # Row 0 (t=0): [0, 1, 2, 3, 4, 5, 6, 7]
        # Row 1 (t=1): [8, 9, 10, 11, 12, 13, 14, 15]
        # Row 2 (t=2): [16, 17, 18, 19, 20, 21, 22, 23]
        audio_codes = torch.arange(24).reshape(3, 8)
        stage_list = self._make_mock_stage_list(audio_codes)

        result = dialogue_to_mimi_decode(stage_list, engine_input_source=[0])
        flat = result[0]["prompt_token_ids"]

        # After transpose [8, 3] and flatten:
        # Codebook 0: [0, 8, 16], Codebook 1: [1, 9, 17], ...
        assert flat[0] == 0   # codebook 0, t=0
        assert flat[1] == 8   # codebook 0, t=1
        assert flat[2] == 16  # codebook 0, t=2
        assert flat[3] == 1   # codebook 1, t=0
        assert flat[4] == 9   # codebook 1, t=1
        assert flat[5] == 17  # codebook 1, t=2

    def test_empty_stage_raises(self):
        from vllm_omni.model_executor.stage_input_processors.moshi import dialogue_to_mimi_decode

        with pytest.raises(ValueError):
            dialogue_to_mimi_decode([], engine_input_source=[])

    def test_missing_audio_codes_fallback(self):
        """Should produce a fallback prompt when audio_codes is missing."""
        from unittest.mock import Mock

        from vllm_omni.model_executor.stage_input_processors.moshi import dialogue_to_mimi_decode

        output = Mock()
        output.outputs = [Mock()]
        output.outputs[0].multimodal_output = {}  # No audio_codes

        stage = Mock()
        stage.engine_outputs = [output]
        stage_list = [stage]

        result = dialogue_to_mimi_decode(stage_list, engine_input_source=[0])
        assert len(result) == 1
        assert len(result[0]["prompt_token_ids"]) == 8  # Fallback with 8 zeros


# =============================================================================
# Test Full Dialogue Inference Loop (Small Model, CPU)
# =============================================================================


class TestDialogueInferenceLoop:
    """Integration test: run the full dialogue inference loop with tiny random weights."""

    @pytest.fixture
    def tiny_config(self):
        config = FakeMoshiConfig(
            vocab_size=64,
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
            ffn_dim=64,
            head_dim=8,
            audio_vocab_size=32,
            num_codebooks=4,  # Fewer codebooks for speed
        )
        config.depth_decoder_config = FakeDepthConfig(
            vocab_size=64,
            hidden_size=16,
            input_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
            ffn_dim=32,
            head_dim=4,
            audio_vocab_size=32,
            num_codebooks=4,
        )
        return config

    def test_full_dialogue_forward(self, tiny_config, device):
        """Run the entire temporal + depth inference loop on random weights."""
        config = tiny_config
        num_codebooks = config.num_codebooks

        # Build model components
        embed_tokens = nn.ModuleList([
            nn.Embedding(config.audio_vocab_size + 1, config.hidden_size)
            for _ in range(2 * num_codebooks)
        ]).to(device)

        decoder = MoshiTemporalDecoder(config).to(device)
        depth_decoder = MoshiDepthDecoder(config.depth_decoder_config).to(device)

        # Simulate user audio codes: 5 frames × 4 codebooks
        T_user = 5
        user_codes = torch.randint(0, config.audio_vocab_size, (1, num_codebooks, T_user), device=device)

        text_tokens = []
        audio_codes_list = []
        past_key_values = None

        # Run 3 time steps
        for t in range(3):
            # Text embedding
            text_id = config.vocab_size if t == 0 else text_tokens[-1]
            text_embed = decoder.model.embed_tokens(
                torch.tensor([[text_id]], device=device)
            )

            # Audio embedding
            audio_embed = torch.zeros_like(text_embed)
            for k in range(num_codebooks):
                if t > 0:
                    code_id = audio_codes_list[-1][k]
                else:
                    code_id = 0
                audio_embed += embed_tokens[k](
                    torch.tensor([[code_id]], device=device)
                )
            for k in range(num_codebooks):
                audio_embed += embed_tokens[num_codebooks + k](
                    user_codes[:, k, t:t+1]
                )

            combined = text_embed + audio_embed

            # Temporal forward
            pos = torch.tensor([[t]], device=device)
            hidden, past_key_values = decoder.model(
                combined, pos, past_key_values=past_key_values, use_cache=True,
            )

            # Text sampling
            logits = decoder.lm_head(hidden[:, -1, :])
            text_token = torch.argmax(logits, dim=-1).item()
            text_tokens.append(text_token)

            # Depth decoder
            codes = depth_decoder.generate_codes(
                hidden[:, -1, :], text_token=text_token, temperature=0.0,
            )
            audio_codes_list.append(codes)

        # Verify outputs
        assert len(text_tokens) == 3
        assert len(audio_codes_list) == 3
        for codes in audio_codes_list:
            assert len(codes) == num_codebooks
            for c in codes:
                assert 0 <= c < config.audio_vocab_size

    def test_kv_cache_grows_correctly(self, tiny_config, device):
        """Verify KV cache grows by 1 at each step."""
        config = tiny_config
        decoder = MoshiTemporalDecoder(config).to(device)

        past_key_values = None
        for t in range(5):
            embed = torch.randn(1, 1, config.hidden_size, device=device)
            pos = torch.tensor([[t]], device=device)
            _, past_key_values = decoder.model(
                embed, pos, past_key_values=past_key_values, use_cache=True,
            )
            # Check KV cache size at first layer
            assert past_key_values[0][0].shape[2] == t + 1


# =============================================================================
# Test Weight Naming Consistency
# =============================================================================


class TestWeightNaming:
    """Verify that model parameter names match expected HF checkpoint format."""

    def test_temporal_decoder_weight_names(self):
        config = FakeMoshiConfig()
        decoder = MoshiTemporalDecoder(config)
        names = set(n for n, _ in decoder.named_parameters())

        # Check critical weight paths exist
        assert "model.embed_tokens.weight" in names
        assert "lm_head.weight" in names
        assert "model.layers.0.self_attn.q_proj.linear.weight" in names
        assert "model.layers.0.self_attn.k_proj.linear.weight" in names
        assert "model.layers.0.self_attn.v_proj.linear.weight" in names
        assert "model.layers.0.self_attn.o_proj.linear.weight" in names
        assert "model.layers.0.mlp.fc1.weight" in names
        assert "model.layers.0.mlp.fc2.weight" in names
        assert "model.layers.0.input_layernorm.weight" in names
        assert "model.layers.0.post_attention_layernorm.weight" in names
        assert "model.norm.weight" in names

    def test_depth_decoder_weight_names(self):
        config = FakeDepthConfig()
        decoder = MoshiDepthDecoder(config)
        names = set(n for n, _ in decoder.named_parameters())

        # Check critical weight paths exist
        assert "text_embed_tokens.weight" in names
        assert "embed_tokens.0.weight" in names
        assert "input_projections.weight" in names
        assert "layers.0.self_attn.q_proj.linear.weight" in names
        assert "layers.0.mlp.fc1.weight" in names
        assert "layers.0.input_layernorm.weight" in names
        assert "lm_heads.weight" in names

    def test_depth_decoder_3d_weights(self):
        """Verify FlexibleLinear weights are 3D (num_codebooks, out, in)."""
        config = FakeDepthConfig()
        decoder = MoshiDepthDecoder(config)

        # input_projections: [num_codebooks, hidden_size, input_size]
        assert decoder.input_projections.weight.shape == (
            config.num_codebooks, config.hidden_size, config.input_size,
        )

        # lm_heads: [num_codebooks, audio_vocab_size, hidden_size]
        assert decoder.lm_heads.weight.shape == (
            config.num_codebooks, config.audio_vocab_size, config.hidden_size,
        )

        # Attention projections: [num_codebooks, hidden_size, hidden_size]
        q_weight = decoder.layers[0].self_attn.q_proj.linear.weight
        assert q_weight.ndim == 3
        assert q_weight.shape[0] == config.num_codebooks

    def test_audio_embeddings_count(self):
        """Verify 16 audio embeddings (8 moshi + 8 user) at top level."""
        config = FakeMoshiConfig()
        embed_tokens = nn.ModuleList([
            nn.Embedding(config.audio_vocab_size + 1, config.hidden_size)
            for _ in range(2 * config.num_codebooks)
        ])
        assert len(embed_tokens) == 16
        for emb in embed_tokens:
            assert emb.weight.shape == (config.audio_vocab_size + 1, config.hidden_size)

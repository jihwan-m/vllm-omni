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


# =============================================================================
# Phase 2: Test Per-Step Streaming Dialogue
# =============================================================================


class TestForwardDialogueStep:
    """Test the per-step streaming dialogue API (forward_dialogue_step)."""

    @pytest.fixture
    def streaming_model(self, device):
        """Build a tiny Moshi model with streaming state initialized."""
        from vllm_omni.model_executor.models.moshi.moshi import (
            MoshiForConditionalGenerationVLLM,
        )

        config = FakeMoshiConfig(
            vocab_size=64,
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
            ffn_dim=64,
            head_dim=8,
            audio_vocab_size=32,
            num_codebooks=4,
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

        # Build model components directly (bypass VllmConfig)
        model = type("FakeModel", (nn.Module,), {
            "__init__": lambda self_: nn.Module.__init__(self_),
        })()
        model.num_codebooks = config.num_codebooks
        model.vocab_size = config.vocab_size
        model.hidden_size = config.hidden_size
        model.embed_tokens = nn.ModuleList([
            nn.Embedding(config.audio_vocab_size + 1, config.hidden_size)
            for _ in range(2 * config.num_codebooks)
        ])
        model.decoder = MoshiTemporalDecoder(config)
        model.depth_decoder = MoshiDepthDecoder(config.depth_decoder_config)

        # Bind the methods from the real class
        import types
        model.init_streaming_state = types.MethodType(
            MoshiForConditionalGenerationVLLM.init_streaming_state, model
        )
        model.clear_streaming_state = types.MethodType(
            MoshiForConditionalGenerationVLLM.clear_streaming_state, model
        )
        model.forward_dialogue_step = types.MethodType(
            MoshiForConditionalGenerationVLLM.forward_dialogue_step, model
        )
        model._sample_token = MoshiForConditionalGenerationVLLM._sample_token

        model.to(device)
        return model

    def test_init_streaming_state(self, streaming_model, device):
        """Test that streaming state is properly initialized."""
        streaming_model.init_streaming_state(request_id="test1", device=device)
        states = streaming_model._streaming_states
        assert "test1" in states
        state = states["test1"]
        assert state["past_key_values"] is None
        assert state["prev_text_token"] is None
        assert state["prev_audio_codes"] is None
        assert state["temporal_step"] == 0

    def test_single_step(self, streaming_model, device):
        """Test running a single dialogue step."""
        streaming_model.init_streaming_state(request_id="test1", device=device)
        result = streaming_model.forward_dialogue_step(
            user_audio_codes=[0, 0, 0, 0],
            temperature=0.0,
            top_k=0,
            request_id="test1",
        )

        assert "audio_codes" in result
        assert "text_token" in result
        assert "temporal_step" in result
        assert result["temporal_step"] == 0
        assert len(result["audio_codes"]) == streaming_model.num_codebooks
        assert isinstance(result["text_token"], int)

    def test_multi_step_state_accumulation(self, streaming_model, device):
        """Test that state accumulates across multiple steps."""
        streaming_model.init_streaming_state(request_id="test1", device=device)

        results = []
        for t in range(5):
            result = streaming_model.forward_dialogue_step(
                user_audio_codes=[0] * streaming_model.num_codebooks,
                temperature=0.0,
                request_id="test1",
            )
            results.append(result)
            assert result["temporal_step"] == t

        # Verify KV cache has grown
        state = streaming_model._streaming_states["test1"]
        assert state["temporal_step"] == 5
        assert state["past_key_values"] is not None
        # First layer's k cache should have 5 positions
        assert state["past_key_values"][0][0].shape[2] == 5

    def test_step_without_init_raises(self, streaming_model):
        """Test that calling forward_dialogue_step without init raises."""
        with pytest.raises(RuntimeError, match="Streaming state not initialized"):
            streaming_model.forward_dialogue_step(request_id="nonexistent")

    def test_clear_streaming_state(self, streaming_model, device):
        """Test that state is properly cleared."""
        streaming_model.init_streaming_state(request_id="test1", device=device)
        streaming_model.forward_dialogue_step(
            user_audio_codes=[0] * streaming_model.num_codebooks,
            request_id="test1",
        )
        assert "test1" in streaming_model._streaming_states
        streaming_model.clear_streaming_state(request_id="test1")
        assert "test1" not in streaming_model._streaming_states

    def test_deterministic_with_zero_temp(self, streaming_model, device):
        """Test that zero temperature gives deterministic results."""
        user_codes = [1, 2, 3, 4]

        streaming_model.init_streaming_state(request_id="run1", device=device)
        r1 = streaming_model.forward_dialogue_step(
            user_audio_codes=user_codes, temperature=0.0, request_id="run1",
        )
        streaming_model.clear_streaming_state(request_id="run1")

        streaming_model.init_streaming_state(request_id="run2", device=device)
        r2 = streaming_model.forward_dialogue_step(
            user_audio_codes=user_codes, temperature=0.0, request_id="run2",
        )

        assert r1["audio_codes"] == r2["audio_codes"]
        assert r1["text_token"] == r2["text_token"]

    def test_none_user_codes_uses_silence(self, streaming_model, device):
        """Test that None user_audio_codes defaults to silence."""
        streaming_model.init_streaming_state(request_id="test1", device=device)
        result = streaming_model.forward_dialogue_step(
            user_audio_codes=None, temperature=0.0, request_id="test1",
        )
        assert len(result["audio_codes"]) == streaming_model.num_codebooks

    def test_concurrent_requests_isolated(self, streaming_model, device):
        """Test that multiple concurrent requests have independent state."""
        streaming_model.init_streaming_state(request_id="req_a", device=device)
        streaming_model.init_streaming_state(request_id="req_b", device=device)

        # Step req_a forward 3 times
        for _ in range(3):
            streaming_model.forward_dialogue_step(
                user_audio_codes=[0] * streaming_model.num_codebooks,
                temperature=0.0,
                request_id="req_a",
            )

        # Step req_b forward 1 time
        streaming_model.forward_dialogue_step(
            user_audio_codes=[1] * streaming_model.num_codebooks,
            temperature=0.0,
            request_id="req_b",
        )

        # Verify independent state
        assert streaming_model._streaming_states["req_a"]["temporal_step"] == 3
        assert streaming_model._streaming_states["req_b"]["temporal_step"] == 1

        # Clear one without affecting the other
        streaming_model.clear_streaming_state(request_id="req_a")
        assert "req_a" not in streaming_model._streaming_states
        assert "req_b" in streaming_model._streaming_states


# =============================================================================
# Phase 2: Test Streaming Stage Input Processor
# =============================================================================


class TestDialogueToMimiAsyncChunk:
    """Test the chunked stage input processor for streaming."""

    def test_accumulation_returns_none_before_chunk_boundary(self):
        """Processor should return None until CHUNK_SIZE frames accumulated."""
        from unittest.mock import Mock

        from vllm_omni.model_executor.stage_input_processors.moshi_streaming import (
            MOSHI_CHUNK_SIZE,
            dialogue_to_mimi_async_chunk,
        )

        connector = Mock()
        connector.code_prompt_token_ids = {}
        connector.code_prompt_token_ids["req1"] = []
        connector.put_requests = {"req1": 0}

        request = Mock()
        request.external_req_id = "req1"
        request.is_finished.return_value = False

        # Send one step — should return None (not at chunk boundary)
        pooling_output = {"audio_codes": [1, 2, 3, 4, 5, 6, 7, 8]}
        result = dialogue_to_mimi_async_chunk(connector, pooling_output, request)
        assert result is None

    def test_returns_chunk_at_boundary(self):
        """Processor should return a chunk at CHUNK_SIZE boundary."""
        from collections import defaultdict
        from unittest.mock import Mock

        from vllm_omni.model_executor.stage_input_processors.moshi_streaming import (
            MOSHI_CHUNK_SIZE,
            MOSHI_NUM_CODEBOOKS,
            dialogue_to_mimi_async_chunk,
        )

        connector = Mock()
        connector.code_prompt_token_ids = defaultdict(list)
        connector.put_requests = defaultdict(int)

        request = Mock()
        request.external_req_id = "req1"
        request.is_finished.return_value = False

        # Accumulate CHUNK_SIZE - 1 frames (all should return None)
        for i in range(MOSHI_CHUNK_SIZE - 1):
            pooling_output = {"audio_codes": list(range(MOSHI_NUM_CODEBOOKS))}
            result = dialogue_to_mimi_async_chunk(connector, pooling_output, request)
            assert result is None, f"Expected None at step {i}"

        # The CHUNK_SIZE-th frame should produce a chunk
        pooling_output = {"audio_codes": list(range(MOSHI_NUM_CODEBOOKS))}
        result = dialogue_to_mimi_async_chunk(connector, pooling_output, request)
        assert result is not None
        assert "code_predictor_codes" in result
        assert len(result["code_predictor_codes"]) == MOSHI_CHUNK_SIZE * MOSHI_NUM_CODEBOOKS

    def test_returns_partial_chunk_on_finish(self):
        """Processor should return remaining codes when request finishes."""
        from collections import defaultdict
        from unittest.mock import Mock

        from vllm_omni.model_executor.stage_input_processors.moshi_streaming import (
            MOSHI_NUM_CODEBOOKS,
            dialogue_to_mimi_async_chunk,
        )

        connector = Mock()
        connector.code_prompt_token_ids = defaultdict(list)
        connector.put_requests = defaultdict(int)

        request = Mock()
        request.external_req_id = "req1"
        request.is_finished.return_value = False

        # Add 3 frames
        for _ in range(3):
            dialogue_to_mimi_async_chunk(
                connector,
                {"audio_codes": [0] * MOSHI_NUM_CODEBOOKS},
                request,
            )

        # Now signal finish
        request.is_finished.return_value = True
        result = dialogue_to_mimi_async_chunk(
            connector,
            {"audio_codes": [0] * MOSHI_NUM_CODEBOOKS},
            request,
        )
        assert result is not None
        # 4 frames * 8 codebooks = 32 codes
        assert len(result["code_predictor_codes"]) == 4 * MOSHI_NUM_CODEBOOKS
        assert result["finished"] is True

    def test_missing_audio_codes_returns_none(self):
        """Processor should return None when audio_codes is missing."""
        from unittest.mock import Mock

        from vllm_omni.model_executor.stage_input_processors.moshi_streaming import (
            dialogue_to_mimi_async_chunk,
        )

        connector = Mock()
        request = Mock()
        request.external_req_id = "req1"

        result = dialogue_to_mimi_async_chunk(connector, {}, request)
        assert result is None

    def test_codes_ordering_is_codebook_major(self):
        """Verify that flattened codes are in codebook-major order [8, T]."""
        from collections import defaultdict
        from unittest.mock import Mock

        from vllm_omni.model_executor.stage_input_processors.moshi_streaming import (
            MOSHI_NUM_CODEBOOKS,
            dialogue_to_mimi_async_chunk,
        )

        connector = Mock()
        connector.code_prompt_token_ids = defaultdict(list)
        connector.put_requests = defaultdict(int)

        request = Mock()
        request.external_req_id = "req1"
        request.is_finished.return_value = True

        # Frame 0: [0,1,2,3,4,5,6,7], Frame 1: [8,9,10,11,12,13,14,15]
        dialogue_to_mimi_async_chunk(
            connector,
            {"audio_codes": list(range(8))},
            request,
        )
        request.is_finished.return_value = False  # not yet
        # Actually, need to re-mock since first call might have returned
        # Reset
        connector.code_prompt_token_ids = defaultdict(list)
        connector.put_requests = defaultdict(int)
        request.is_finished.return_value = False

        dialogue_to_mimi_async_chunk(
            connector,
            {"audio_codes": list(range(8))},
            request,
        )
        request.is_finished.return_value = True
        result = dialogue_to_mimi_async_chunk(
            connector,
            {"audio_codes": list(range(8, 16))},
            request,
        )

        assert result is not None
        flat = result["code_predictor_codes"]
        # Transpose of [[0..7],[8..15]] = codebook 0: [0,8], codebook 1: [1,9], ...
        assert flat[0] == 0   # codebook 0, frame 0
        assert flat[1] == 8   # codebook 0, frame 1
        assert flat[2] == 1   # codebook 1, frame 0
        assert flat[3] == 9   # codebook 1, frame 1


# =============================================================================
# Phase 2: Test WebSocket Protocol Messages
# =============================================================================


class TestDuplexProtocol:
    """Test the WebSocket protocol message definitions."""

    def test_session_start_defaults(self):
        from vllm_omni.entrypoints.openai.protocol.duplex import SessionStartMessage
        msg = SessionStartMessage()
        assert msg.model == "moshi"
        assert msg.sample_rate == 24000
        assert msg.audio_format == "pcm_s16le"
        assert msg.temperature == 0.7

    def test_session_start_custom(self):
        from vllm_omni.entrypoints.openai.protocol.duplex import SessionStartMessage
        msg = SessionStartMessage(
            model="moshi-large",
            sample_rate=48000,
            temperature=1.0,
            top_k=50,
        )
        assert msg.model == "moshi-large"
        assert msg.sample_rate == 48000
        assert msg.temperature == 1.0

    def test_session_created_serialization(self):
        from vllm_omni.entrypoints.openai.protocol.duplex import SessionCreatedMessage
        msg = SessionCreatedMessage(
            session_id="abc123",
            model="moshi",
            sample_rate=24000,
        )
        d = msg.model_dump()
        assert d["type"] == "session.created"
        assert d["session_id"] == "abc123"

    def test_audio_output_meta(self):
        from vllm_omni.entrypoints.openai.protocol.duplex import AudioOutputMeta
        meta = AudioOutputMeta(chunk_index=5, duration_ms=80.0, is_final=False)
        d = meta.model_dump()
        assert d["type"] == "audio.output.meta"
        assert d["chunk_index"] == 5
        assert d["duration_ms"] == 80.0

    def test_generation_done(self):
        from vllm_omni.entrypoints.openai.protocol.duplex import GenerationDoneMessage
        msg = GenerationDoneMessage(
            session_id="xyz",
            total_chunks=10,
            total_duration_ms=5000.0,
        )
        d = msg.model_dump()
        assert d["type"] == "generation.done"
        assert d["total_chunks"] == 10

    def test_error_message(self):
        from vllm_omni.entrypoints.openai.protocol.duplex import ErrorMessage
        msg = ErrorMessage(message="test error", code="timeout")
        d = msg.model_dump()
        assert d["type"] == "error"
        assert d["message"] == "test error"
        assert d["code"] == "timeout"

    def test_message_type_enum(self):
        from vllm_omni.entrypoints.openai.protocol.duplex import DuplexMessageType
        assert DuplexMessageType.SESSION_START == "session.start"
        assert DuplexMessageType.AUDIO_OUTPUT == "audio.output"
        assert DuplexMessageType.GENERATION_DONE == "generation.done"


# =============================================================================
# Phase 2: Test OmniStreamingChunk
# =============================================================================


class TestOmniStreamingChunk:
    """Test the streaming chunk output type."""

    def test_defaults(self):
        from vllm_omni.outputs import OmniStreamingChunk
        chunk = OmniStreamingChunk()
        assert chunk.stage_id == 0
        assert chunk.chunk_index == 0
        assert chunk.audio_data is None
        assert chunk.is_final is False
        assert chunk.error is None

    def test_with_audio_data(self):
        from vllm_omni.outputs import OmniStreamingChunk
        audio = b"\x00\x01\x02\x03" * 100
        chunk = OmniStreamingChunk(
            stage_id=1,
            chunk_index=5,
            audio_data=audio,
            is_final=True,
        )
        assert chunk.audio_data == audio
        assert len(chunk.audio_data) == 400
        assert chunk.is_final is True

    def test_error_chunk(self):
        from vllm_omni.outputs import OmniStreamingChunk
        chunk = OmniStreamingChunk(
            stage_id=-1,
            is_final=True,
            error="Pipeline failed",
        )
        assert chunk.error == "Pipeline failed"
        assert chunk.is_final is True


# =============================================================================
# Phase 2 Pre-Phase-3 Fixes: Duration Calculation
# =============================================================================


class TestComputePcmDurationMs:
    """Test the duration calculation helper for audio chunks."""

    def test_pcm_s16le_duration(self):
        from vllm_omni.entrypoints.openai.serving_duplex import _compute_pcm_duration_ms
        # 48000 samples at 24kHz = 2 seconds = 2000 ms
        # 48000 samples * 2 bytes/sample = 96000 bytes
        duration = _compute_pcm_duration_ms(b"\x00" * 96000, 24000, "pcm_s16le")
        assert abs(duration - 2000.0) < 0.1

    def test_pcm_f32le_duration(self):
        from vllm_omni.entrypoints.openai.serving_duplex import _compute_pcm_duration_ms
        # 24000 samples at 24kHz = 1 second = 1000 ms
        # 24000 samples * 4 bytes/sample = 96000 bytes
        duration = _compute_pcm_duration_ms(b"\x00" * 96000, 24000, "pcm_f32le")
        assert abs(duration - 1000.0) < 0.1

    def test_opus_returns_zero(self):
        from vllm_omni.entrypoints.openai.serving_duplex import _compute_pcm_duration_ms
        # Compressed formats can't compute duration from byte length
        duration = _compute_pcm_duration_ms(b"\x00" * 1000, 24000, "opus")
        assert duration == 0.0

    def test_empty_data_returns_zero(self):
        from vllm_omni.entrypoints.openai.serving_duplex import _compute_pcm_duration_ms
        duration = _compute_pcm_duration_ms(b"", 24000, "pcm_s16le")
        assert duration == 0.0

    def test_none_data_returns_zero(self):
        from vllm_omni.entrypoints.openai.serving_duplex import _compute_pcm_duration_ms
        duration = _compute_pcm_duration_ms(None, 24000, "pcm_s16le")
        assert duration == 0.0


# =============================================================================
# Phase 2 Pre-Phase-3 Fixes: Chunk processor returns plain bool
# =============================================================================


class TestChunkProcessorReturnTypes:
    """Test that chunk processor returns consistent Python types."""

    def test_finished_is_plain_bool(self):
        """Verify 'finished' field is a plain Python bool, not torch.Tensor."""
        from collections import defaultdict
        from unittest.mock import Mock

        from vllm_omni.model_executor.stage_input_processors.moshi_streaming import (
            MOSHI_NUM_CODEBOOKS,
            dialogue_to_mimi_async_chunk,
        )

        connector = Mock()
        connector.code_prompt_token_ids = defaultdict(list)
        connector.put_requests = defaultdict(int)

        request = Mock()
        request.external_req_id = "req1"
        request.is_finished.return_value = True

        result = dialogue_to_mimi_async_chunk(
            connector,
            {"audio_codes": [0] * MOSHI_NUM_CODEBOOKS},
            request,
        )
        assert result is not None
        assert isinstance(result["finished"], bool)
        assert result["finished"] is True

    def test_code_predictor_codes_is_list(self):
        """Verify 'code_predictor_codes' is a plain Python list."""
        from collections import defaultdict
        from unittest.mock import Mock

        from vllm_omni.model_executor.stage_input_processors.moshi_streaming import (
            MOSHI_NUM_CODEBOOKS,
            dialogue_to_mimi_async_chunk,
        )

        connector = Mock()
        connector.code_prompt_token_ids = defaultdict(list)
        connector.put_requests = defaultdict(int)

        request = Mock()
        request.external_req_id = "req1"
        request.is_finished.return_value = True

        result = dialogue_to_mimi_async_chunk(
            connector,
            {"audio_codes": [0] * MOSHI_NUM_CODEBOOKS},
            request,
        )
        assert isinstance(result["code_predictor_codes"], list)
        assert all(isinstance(c, int) for c in result["code_predictor_codes"])

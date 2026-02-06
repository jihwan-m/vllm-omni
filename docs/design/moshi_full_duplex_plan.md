# Moshi Full-Duplex Implementation Plan for vLLM-Omni

## Executive Summary

Moshi (by Kyutai)와 Mimi codec을 vllm-omni에 통합하여, 궁극적으로 **full-duplex 실시간 음성 대화**를 지원하는 것을 목표로 합니다. 3단계(Phase)로 나누어 점진적으로 구현합니다.

---

## Architecture Comparison

### Current: Qwen3-Omni (Forward-Only Pipeline)

```
User Audio ──→ [Thinker] ──→ [Talker] ──→ [Code2Wav] ──→ Audio Out
                 Stage 0       Stage 1       Stage 2
                (7B, AR)      (AR, RVQ)    (generation)

Timeline: |---- full text ----|-- codec codes --|-- waveform --|
                (sequential, no overlap)
```

### Target: Moshi (Full-Duplex, Interleaved)

```
User Audio ──→ [Mimi Encoder] ──→ tokens ──┐
                                            ↓
              [Temporal Transformer (7B)] ←─┤ (every 80ms)
                        │                   │
                        ↓                   │
              [Depth Transformer (6L)]      │
                        │                   │
                        ↓                   │
              [Mimi Decoder] ──→ Audio Out  │
                        │                   │
                        └───── loop ────────┘

Timeline: |--80ms--|--80ms--|--80ms--|--80ms--| ...
           (simultaneous input + output at each step)
```

---

## Phase 1: Half-Duplex Moshi (Turn-Taking)

> **Goal**: Moshi 모델을 vllm-omni에 등록하고, turn-taking 방식으로 동작시킵니다.
> 사용자 오디오를 Mimi로 인코딩 → Moshi가 응답 생성 → Mimi로 디코딩

### 1.1 새로 생성할 파일

```
vllm_omni/model_executor/models/moshi/
├── __init__.py                          # Export MoshiForConditionalGeneration
├── moshi.py                             # Unified model class (stage dispatcher)
├── moshi_temporal.py                    # Temporal Transformer wrapper
├── moshi_depth.py                       # Depth Transformer (inner 8-step AR loop)
└── mimi.py                              # Mimi encoder/decoder wrapper

vllm_omni/model_executor/stage_configs/
└── moshi.yaml                           # 2-stage config (moshi_dialogue + mimi_decode)

vllm_omni/model_executor/stage_input_processors/
└── moshi.py                             # dialogue_to_mimi_decode() transition

tests/e2e/offline_inference/
└── test_moshi.py                        # Basic inference test
```

### 1.2 수정할 파일

| File | Change |
|------|--------|
| `vllm_omni/model_executor/models/registry.py` | `_OMNI_MODELS`에 Moshi 아키텍처 등록 |
| `vllm_omni/entrypoints/openai/serving_chat.py` | Moshi 오디오 출력 처리 (기존 패턴 활용) |

### 1.3 Model Implementation Details

#### `moshi.py` — Unified Model Class

```python
class MoshiForConditionalGeneration(nn.Module, SupportsMultiModal, SupportsPP):
    """
    Unified Moshi model. model_stage에 따라 다른 동작:
    - "dialogue": Temporal + Depth transformer (text+audio 생성)
    - "mimi_decode": Mimi decoder (tokens → waveform)
    """

    def __init__(self, *, vllm_config, prefix=""):
        super().__init__()
        self.have_multimodal_outputs = True
        self.model_stage = vllm_config.model_config.model_stage

        if self.model_stage == "dialogue":
            self._init_dialogue(vllm_config, prefix)
        elif self.model_stage == "mimi_decode":
            self._init_mimi_decoder(vllm_config, prefix)

    def _init_dialogue(self, vllm_config, prefix):
        """Temporal (7B) + Depth (6L) transformers"""
        config = vllm_config.model_config.hf_config
        self.temporal = MoshiTemporalTransformer(config, vllm_config)
        self.depth = MoshiDepthTransformer(config)
        self.mimi_encoder = MimiEncoder(config)  # For pre-processing user audio
        self.text_lm_head = nn.Linear(config.hidden_size, config.text_vocab_size)

    def _init_mimi_decoder(self, vllm_config, prefix):
        """Mimi streaming decoder"""
        self.mimi_decoder = MimiDecoder(vllm_config.model_config.hf_config)

    def forward(self, input_ids, positions, ...):
        if self.model_stage == "dialogue":
            return self._forward_dialogue(input_ids, positions, ...)
        elif self.model_stage == "mimi_decode":
            return self._forward_mimi_decode(input_ids, positions, ...)

    def _forward_dialogue(self, input_ids, positions, ...):
        """
        Phase 1에서는 전체 시퀀스를 한 번에 처리:
        1. 사용자 오디오 토큰 (Mimi encode 결과)을 prompt로 받음
        2. Temporal Transformer가 text + temporal context 생성
        3. Depth Transformer가 매 step마다 8개 codebook 토큰 생성
        4. 전체 codebook 토큰 시퀀스를 multimodal_output으로 반환
        """
        # Step 1: Temporal transformer forward (standard AR)
        hidden_states = self.temporal(input_ids, positions, ...)

        # Step 2: Text token sampling
        text_logits = self.text_lm_head(hidden_states)

        # Step 3: Depth transformer loop (8 codebooks per step)
        # This runs internally as a loop, not visible to vLLM scheduler
        all_audio_codes = []
        for t in range(seq_len):
            temporal_context = hidden_states[:, t, :]
            codes_t = self.depth.generate_step(temporal_context)  # [8] codes
            all_audio_codes.append(codes_t)

        audio_codes = torch.stack(all_audio_codes, dim=0)  # [T, 8]

        return OmniOutput(
            text_hidden_states=hidden_states,
            multimodal_outputs={
                "audio_codes": audio_codes,       # [T, 8] RVQ codes
                "text_logits": text_logits,        # For text streaming
            },
        )

    def _forward_mimi_decode(self, input_ids, ...):
        """Mimi decoder: RVQ codes → audio waveform"""
        codes = input_ids.reshape(-1, 8).T  # [8, T]
        audio = self.mimi_decoder(codes)     # [1, samples]

        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={"audio": audio},
        )
```

#### `moshi_temporal.py` — Temporal Transformer

```python
class MoshiTemporalTransformer(nn.Module):
    """
    Helium 7B backbone. 표준 Transformer decoder.
    Phase 1: 기본 vLLM AR 추론과 호환.

    Moshi의 multi-stream embedding을 처리:
    - Stream 0: text token embedding
    - Stream 1-8: Moshi audio codebook embeddings
    - Stream 9-16: User audio codebook embeddings (from prompt)
    """

    def __init__(self, config, vllm_config):
        # HF MoshiForConditionalGeneration에서 temporal transformer 부분 로드
        # 17개 stream의 embedding layer
        self.embed_tokens = nn.ModuleList([
            nn.Embedding(vocab_size, config.hidden_size)
            for _ in range(config.num_codebooks * 2 + 1)  # 17 streams
        ])
        self.layers = nn.ModuleList([...])  # 32 transformer layers

    def forward(self, multi_stream_ids, positions, ...):
        """
        Input: multi_stream_ids [batch, 17, seq_len]
          - Channel 0: text tokens
          - Channel 1-8: Moshi audio codes (from previous generation, or empty)
          - Channel 9-16: User audio codes (from Mimi encoder)

        Output: hidden_states [batch, seq_len, hidden_size]
        """
        # Sum embeddings across all 17 streams
        combined = sum(
            self.embed_tokens[i](multi_stream_ids[:, i, :])
            for i in range(17)
        )

        for layer in self.layers:
            combined = layer(combined, positions, ...)

        return combined
```

#### `moshi_depth.py` — Depth Transformer

```python
class MoshiDepthTransformer(nn.Module):
    """
    6-layer Transformer. 매 time step마다 8개 audio codebook 토큰을
    bottom-to-top 순서로 autoregressive하게 생성.

    Context length = num_codebooks (8). 매우 작으므로
    PagedAttention 불필요 — 단순 버퍼로 관리.
    """

    def __init__(self, config):
        self.num_codebooks = config.num_codebooks  # 8
        self.layers = nn.ModuleList([...])  # 6 layers
        self.codebook_heads = nn.ModuleList([
            nn.Linear(config.depth_hidden_size, config.audio_vocab_size)
            for _ in range(self.num_codebooks)
        ])
        self.codebook_embeds = nn.ModuleList([
            nn.Embedding(config.audio_vocab_size, config.depth_hidden_size)
            for _ in range(self.num_codebooks)
        ])

    def generate_step(self, temporal_context: torch.Tensor) -> torch.Tensor:
        """
        한 time step에서 8개 codebook 토큰 생성.

        Args:
            temporal_context: [batch, hidden_size] from Temporal Transformer

        Returns:
            codes: [batch, 8] audio codebook tokens
        """
        codes = []
        hidden = temporal_context.unsqueeze(1)  # [batch, 1, hidden]

        for k in range(self.num_codebooks):
            for layer in self.layers:
                hidden = layer(hidden)

            logits = self.codebook_heads[k](hidden[:, -1, :])
            code_k = torch.argmax(logits, dim=-1)  # Greedy or sample
            codes.append(code_k)

            # Feed back as input for next codebook
            code_embed = self.codebook_embeds[k](code_k).unsqueeze(1)
            hidden = torch.cat([hidden, code_embed], dim=1)

        return torch.stack(codes, dim=-1)  # [batch, 8]
```

#### `mimi.py` — Mimi Encoder/Decoder

```python
class MimiEncoder(nn.Module):
    """Audio → 8-level RVQ tokens at 12.5Hz"""

    def __init__(self, config):
        # Load from kyutai/mimi checkpoint
        self.encoder_conv = ...    # Causal CNN
        self.encoder_transformer = ...  # Transformer
        self.quantizer = ...       # RVQ with 8 codebooks

    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio: [batch, 1, samples] at 24kHz
        Returns:
            codes: [batch, 8, T] where T = samples / 1920 (80ms frames)
        """
        latents = self.encoder_conv(audio)
        latents = self.encoder_transformer(latents)
        codes = self.quantizer.encode(latents)
        return codes


class MimiDecoder(nn.Module):
    """8-level RVQ tokens → Audio waveform"""

    def __init__(self, config):
        self.decoder_transformer = ...
        self.decoder_conv = ...
        self.quantizer = ...

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            codes: [batch, 8, T]
        Returns:
            audio: [batch, 1, samples] at 24kHz
        """
        latents = self.quantizer.decode(codes)
        latents = self.decoder_transformer(latents)
        audio = self.decoder_conv(latents)
        return audio
```

### 1.4 Stage Configuration

```yaml
# vllm_omni/model_executor/stage_configs/moshi.yaml
#
# 2-stage Moshi pipeline (half-duplex, turn-taking)
# Stage 0: Moshi dialogue (Temporal + Depth) — generates text + audio codes
# Stage 1: Mimi decoder — converts audio codes to waveform

async_chunk: false

stage_args:
  - stage_id: 0
    stage_type: llm
    runtime:
      devices: "0"
      max_batch_size: 1     # Moshi는 single-request에 최적화
    engine_args:
      model_stage: dialogue
      model_arch: MoshiForConditionalGeneration
      worker_type: ar        # Temporal Transformer는 standard AR
      scheduler_cls: vllm_omni.core.sched.omni_ar_scheduler.OmniARScheduler
      gpu_memory_utilization: 0.85
      enforce_eager: true
      trust_remote_code: true
      engine_output_type: latent
      enable_prefix_caching: false
      max_num_batched_tokens: 8192
      tensor_parallel_size: 1
    final_output: true
    final_output_type: text    # Inner monologue text
    is_comprehension: true
    default_sampling_params:
      temperature: 0.7
      top_k: 50
      max_tokens: 4096        # ~327초 (4096 / 12.5Hz)
      detokenize: true

  - stage_id: 1
    stage_type: llm
    runtime:
      devices: "0"            # Mimi는 96M params — GPU 공유 가능
      max_batch_size: 1
    engine_args:
      model_stage: mimi_decode
      model_arch: MoshiForConditionalGeneration
      worker_type: generation
      scheduler_cls: vllm_omni.core.sched.omni_generation_scheduler.OmniGenerationScheduler
      gpu_memory_utilization: 0.05
      enforce_eager: true
      trust_remote_code: true
      engine_output_type: audio
      async_scheduling: false
      max_num_batched_tokens: 500000
    engine_input_source: [0]
    custom_process_input_func: vllm_omni.model_executor.stage_input_processors.moshi.dialogue_to_mimi_decode
    final_output: true
    final_output_type: audio
    default_sampling_params:
      temperature: 0.0
      max_tokens: 65536
```

### 1.5 Stage Input Processor

```python
# vllm_omni/model_executor/stage_input_processors/moshi.py

def dialogue_to_mimi_decode(
    stage_list, engine_input_source, prompt=None, requires_multimodal_data=False
):
    """
    Moshi dialogue 출력 (audio_codes [T, 8])을
    Mimi decoder 입력 (flattened token IDs)으로 변환.
    """
    source = stage_list[engine_input_source[0]].engine_outputs
    results = []

    for output in source:
        audio_codes = output.outputs[0].multimodal_output["audio_codes"]
        # [T, 8] → [T*8] flattened for token-based processing
        flat_codes = audio_codes.long().reshape(-1).tolist()

        results.append(OmniTokensPrompt(
            prompt_token_ids=flat_codes,
        ))

    return results
```

### 1.6 Registry Update

```python
# registry.py에 추가
"MoshiForConditionalGeneration": (
    "moshi",
    "moshi",
    "MoshiForConditionalGeneration",
),
```

### 1.7 Phase 1 Data Flow

```
User sends audio in request
        │
        ↓
[Mimi Encoder] (in model's forward, pre-processing)
        │ user_audio_codes [8, T]
        ↓
[Temporal Transformer] (AR, vLLM PagedAttention)
        │ hidden_states [T, hidden_size]
        ↓
[Depth Transformer] (inner loop, 8 steps per time step)
        │ audio_codes [T, 8]
        ↓
    Stage 0 output: OmniOutput(multimodal_outputs={"audio_codes": ...})
        │
        ↓ dialogue_to_mimi_decode()
        │
[Mimi Decoder] (generation worker)
        │ audio waveform [1, samples]
        ↓
    Stage 1 output: OmniOutput(multimodal_outputs={"audio": ...})
        │
        ↓
    Client receives audio response
```

---

## Phase 2: Chunked Streaming Output

> **Goal**: 생성된 오디오를 chunk 단위로 클라이언트에 전달 + WebSocket 엔드포인트 추가

### 2.1 새로 생성할 파일

```
vllm_omni/entrypoints/openai/serving_duplex.py    # WebSocket endpoint handler
vllm_omni/entrypoints/openai/protocol/duplex.py   # Duplex protocol definitions
vllm_omni/model_executor/stage_configs/
└── moshi_streaming.yaml                           # async_chunk enabled config
```

### 2.2 수정할 파일

| File | Change |
|------|--------|
| `vllm_omni/outputs.py` | `final_output_type`을 `str \| list[str]`로 확장 |
| `vllm_omni/entrypoints/openai/protocol/audio.py` | SSE stream format 지원 해제 |
| `vllm_omni/entrypoints/openai/api_server.py` | WebSocket route 등록 |
| `vllm_omni/entrypoints/openai/serving_chat.py` | Dual output (text+audio) 처리 |
| `vllm_omni/model_executor/stage_input_processors/moshi.py` | async chunk 전환 함수 추가 |

### 2.3 Key Changes

#### `outputs.py` — Dual Output Type

```python
@dataclass
class OmniRequestOutput:
    # AS-IS
    final_output_type: str = "text"

    # TO-BE: 복수 출력 타입 지원
    final_output_type: str | list[str] = "text"
    # 예: ["text", "audio"] for Moshi
```

#### `protocol/duplex.py` — WebSocket Protocol

```python
@dataclass
class DuplexAudioFrame:
    """Single audio frame for WebSocket streaming"""
    timestamp_ms: float
    audio_data: bytes        # PCM16 24kHz
    text_token: str | None   # Inner monologue (optional)
    is_final: bool = False

@dataclass
class DuplexSessionConfig:
    """Configuration for a duplex session"""
    model: str
    sample_rate: int = 24000
    frame_duration_ms: int = 80
    audio_format: str = "pcm"
    include_text: bool = True  # Include inner monologue
```

#### `serving_duplex.py` — WebSocket Handler

```python
class OmniServingDuplex:
    """WebSocket-based audio streaming for Moshi-like models"""

    async def handle_duplex_session(self, websocket: WebSocket):
        await websocket.accept()

        # Phase 2: Output streaming only (input is still turn-taking)
        # Phase 3: Bidirectional streaming

        # 1. Receive initial audio from client
        audio_data = await self._receive_audio(websocket)

        # 2. Start generation
        async for chunk in self.engine.generate_stream(...):
            if chunk.final_output_type == "audio":
                audio_bytes = self._encode_audio_chunk(chunk)
                await websocket.send_bytes(audio_bytes)

            if chunk.finished:
                await websocket.send_json({"type": "done"})
                break
```

#### `moshi_streaming.yaml` — Async Chunk Config

```yaml
async_chunk: true

stage_args:
  - stage_id: 0
    # ... same as Phase 1 but with:
    engine_args:
      enforce_eager: false  # Enable CUDA graphs for speed
    # Async chunk: send codes incrementally to Mimi decoder
    custom_process_next_stage_input_func: >
      vllm_omni.model_executor.stage_input_processors.moshi.dialogue_to_mimi_decode_streaming

  - stage_id: 1
    # ... same as Phase 1 but with chunked decode
```

### 2.4 Phase 2 Data Flow

```
Client ──(WebSocket)──→ [Server receives full audio]
                              │
                              ↓
                    [Moshi generates codes]
                              │
                    ┌─────────┼─────────┐
                    ↓         ↓         ↓
                 chunk1    chunk2    chunk3  ...
                    │         │         │
                    ↓         ↓         ↓
              [Mimi decode] each chunk incrementally
                    │         │         │
                    ↓         ↓         ↓
Client ←──(WebSocket)── audio frames streamed back
```

---

## Phase 3: Full-Duplex Bidirectional Pipeline

> **Goal**: 생성 중 실시간으로 사용자 오디오를 주입하여 full-duplex 대화 구현

### 3.1 새로 생성할 파일

```
vllm_omni/core/sched/omni_duplex_scheduler.py      # Duplex-aware scheduler
vllm_omni/worker/gpu_duplex_worker.py               # Interruptible worker
vllm_omni/worker/gpu_duplex_model_runner.py          # Model runner with input injection
vllm_omni/distributed/omni_connectors/
└── connectors/duplex_connector.py                   # Bidirectional connector
vllm_omni/model_executor/stage_configs/
└── moshi_duplex.yaml                                # Full-duplex stage config
```

### 3.2 수정할 파일

| File | Change | Complexity |
|------|--------|------------|
| `vllm_omni/entrypoints/omni.py` | Duplex orchestrator loop 추가 | ★★★★☆ |
| `vllm_omni/entrypoints/omni_stage.py` | Duplex stage worker 모드 | ★★★★☆ |
| `vllm_omni/distributed/omni_connectors/adapter.py` | Backward connector 함수 | ★★★☆☆ |
| `vllm_omni/distributed/omni_connectors/connectors/base.py` | Bidirectional interface | ★★☆☆☆ |
| `vllm_omni/distributed/omni_connectors/factory.py` | Duplex connector 생성 | ★★☆☆☆ |
| `vllm_omni/entrypoints/openai/serving_duplex.py` | Bidirectional WebSocket | ★★★☆☆ |
| `vllm_omni/model_executor/models/moshi/moshi.py` | Streaming input injection | ★★★★★ |

### 3.3 Core Component Details

#### `omni_duplex_scheduler.py` — Duplex Scheduler

```python
class OmniDuplexScheduler:
    """
    Scheduler that supports:
    1. Standard AR scheduling (Temporal Transformer)
    2. Mid-generation input injection (user audio tokens every 80ms)
    3. Pause/resume for duplex operation
    """

    # New request states
    class DuplexRequestStatus(Enum):
        WAITING = "waiting"
        RUNNING = "running"
        PAUSED_FOR_INPUT = "paused_for_input"   # NEW
        FINISHED = "finished"

    def schedule(self) -> SchedulerOutput:
        """Standard scheduling + check for pending input injections"""
        output = super().schedule()

        # Check for pending audio input injections
        for req_id in self.duplex_requests:
            pending = self.input_buffer.get(req_id)
            if pending:
                output.input_injections[req_id] = pending
                self.input_buffer.pop(req_id)

        return output

    def inject_input(self, req_id: str, audio_tokens: list[int]):
        """
        외부에서 호출: 사용자 오디오 토큰을 생성 중인 request에 주입.
        다음 schedule() 호출 시 SchedulerOutput에 포함됨.
        """
        self.input_buffer[req_id] = audio_tokens

    def pause_request(self, req_id: str):
        """Request를 일시정지 (다음 사용자 입력 대기)"""
        req = self.requests[req_id]
        req.status = self.DuplexRequestStatus.PAUSED_FOR_INPUT

    def resume_request(self, req_id: str):
        """일시정지된 request 재개"""
        req = self.requests[req_id]
        req.status = self.DuplexRequestStatus.RUNNING
```

#### `gpu_duplex_worker.py` — Interruptible Worker

```python
class GPUDuplexWorker(OmniWorkerMixin, GPUWorker):
    """
    Duplex worker that can:
    1. Execute model forward pass
    2. Check for input injections between steps
    3. Extend KV cache with new user audio tokens
    """

    def __init__(self, ...):
        super().__init__(...)
        self.model_runner = GPUDuplexModelRunner(self.vllm_config, self.device)

    def execute_model(self, scheduler_output: SchedulerOutput):
        """
        Modified execution loop:
        - Standard forward pass
        - After each step, check for input_injections in scheduler_output
        - If present, extend the request's input sequence with new tokens
        """
        output = self.model_runner.execute_model(scheduler_output)

        # Process input injections
        if hasattr(scheduler_output, 'input_injections'):
            for req_id, new_tokens in scheduler_output.input_injections.items():
                self.model_runner.inject_tokens(req_id, new_tokens)

        return output
```

#### `gpu_duplex_model_runner.py` — Duplex Model Runner

```python
class GPUDuplexModelRunner(GPUARModelRunner):
    """
    Model runner that supports mid-generation token injection.

    Key difference from GPUARModelRunner:
    - KV cache can be extended with new prompt tokens during generation
    - Multi-stream embedding (17 streams) management
    - 80ms step-based execution
    """

    def inject_tokens(self, req_id: str, new_user_audio_tokens: list[int]):
        """
        실행 중인 request에 새 사용자 오디오 토큰을 주입.

        Mechanism:
        1. new_user_audio_tokens는 Mimi encoder 출력 (8 codebook values per frame)
        2. 이를 stream 9-16에 해당하는 embedding으로 변환
        3. 다음 forward step에서 Temporal Transformer에 입력

        KV cache 영향:
        - Temporal Transformer의 KV cache는 확장 (새 토큰 추가)
        - PagedAttention block 추가 할당 필요
        - Depth Transformer는 영향 없음 (매 step 리셋)
        """
        request = self._get_request(req_id)

        # Reshape: [8*N] → [N, 8] (N frames, 8 codebooks each)
        frames = torch.tensor(new_user_audio_tokens).reshape(-1, 8)

        # Store in request's pending input buffer
        request.pending_user_frames = frames

    def execute_model(self, scheduler_output):
        """
        Modified forward that processes pending user frames.
        """
        # For each request with pending frames:
        # 1. Encode pending frames into stream 9-16 embeddings
        # 2. Prepend to current generation position
        # 3. Run forward pass
        # 4. Clear pending frames
        ...
```

#### Bidirectional Connector

```python
# distributed/omni_connectors/connectors/duplex_connector.py

class DuplexConnector(OmniConnectorBase):
    """
    Bidirectional connector supporting both forward and backward data flow.

    Forward: Stage N → Stage N+1 (standard)
    Backward: External → Stage N (audio input injection)
    """

    def __init__(self, stage_id, config):
        super().__init__()
        self.forward_buffer = {}   # Standard forward transfer
        self.backward_buffer = {}  # Input injection buffer

    def put_backward(self, target_stage: str, req_id: str, data: Any):
        """
        사용자 오디오 토큰을 생성 중인 스테이지에 주입.
        WebSocket handler → Orchestrator → Connector → Worker
        """
        key = f"{req_id}_backward"
        self.backward_buffer[key] = data
        return True, len(self.serialize_obj(data)), {}

    def get_backward(self, stage_id: str, req_id: str):
        """Worker가 backward 데이터를 폴링"""
        key = f"{req_id}_backward"
        data = self.backward_buffer.pop(key, None)
        if data:
            return data, len(self.serialize_obj(data))
        return None
```

#### Modified Orchestrator Loop

```python
# In omni.py, new method for duplex mode

def _run_duplex_generation(self, ...):
    """
    Full-duplex orchestration loop:
    - Forward: standard stage pipeline
    - Backward: audio input injection from WebSocket
    - Concurrent: both directions at the same time
    """

    while not session_ended:
        made_progress = False

        # === Forward path: collect outputs from stages ===
        for stage_id, stage in enumerate(self.stage_list):
            result = stage.try_collect()
            if result is None:
                continue

            made_progress = True
            # ... standard forward processing ...

            # If this is a streaming audio output, yield immediately
            if stage.final_output and result.get("is_chunk"):
                yield OmniRequestOutput(
                    stage_id=stage_id,
                    final_output_type=stage.final_output_type,
                    request_output=engine_outputs,
                    finished=False,  # Streaming, not done yet
                )

        # === Backward path: inject user audio tokens ===
        for req_id, pending_audio in self.pending_audio_inputs.items():
            if pending_audio:
                audio_tokens = pending_audio.pop(0)
                # Send to dialogue stage via backward connector
                connector = self.backward_connectors.get(("external", "0"))
                if connector:
                    connector.put_backward("0", req_id, {
                        "user_audio_tokens": audio_tokens,
                        "timestamp": time.time(),
                    })

        if not made_progress:
            await asyncio.sleep(0.005)  # 5ms polling
```

### 3.4 Full-Duplex WebSocket Handler

```python
# serving_duplex.py (Phase 3 완성)

class OmniServingDuplex:
    async def handle_duplex_session(self, websocket: WebSocket):
        await websocket.accept()
        session_id = str(uuid.uuid4())

        # Concurrent send/receive
        async def receive_loop():
            """Continuously receive user audio and inject into model"""
            while True:
                try:
                    audio_bytes = await asyncio.wait_for(
                        websocket.receive_bytes(), timeout=0.08  # 80ms
                    )
                    # Mimi encode
                    audio_tensor = self._bytes_to_tensor(audio_bytes)
                    user_codes = self.mimi_encoder.encode(audio_tensor)

                    # Inject into running generation
                    self.engine.inject_audio(session_id, user_codes)

                except asyncio.TimeoutError:
                    # No user audio this frame — send silence tokens
                    self.engine.inject_audio(session_id, self.silence_codes)
                except WebSocketDisconnect:
                    break

        async def send_loop():
            """Stream generated audio back to client"""
            async for chunk in self.engine.generate_duplex(session_id, ...):
                if chunk.final_output_type == "audio":
                    audio_bytes = self._tensor_to_bytes(
                        chunk.multimodal_output["audio"]
                    )
                    await websocket.send_bytes(audio_bytes)

                if chunk.multimodal_output.get("text_token"):
                    await websocket.send_json({
                        "type": "text",
                        "token": chunk.multimodal_output["text_token"],
                    })

        # Run both loops concurrently
        await asyncio.gather(
            receive_loop(),
            send_loop(),
        )
```

### 3.5 `moshi_duplex.yaml`

```yaml
# Full-duplex Moshi configuration
async_chunk: true
duplex_mode: true          # NEW: enable bidirectional data flow

stage_args:
  - stage_id: 0
    stage_type: llm
    runtime:
      devices: "0"
      max_batch_size: 1
    engine_args:
      model_stage: dialogue
      model_arch: MoshiForConditionalGeneration
      worker_type: duplex                    # NEW worker type
      worker_cls: vllm_omni.worker.gpu_duplex_worker.GPUDuplexWorker
      scheduler_cls: vllm_omni.core.sched.omni_duplex_scheduler.OmniDuplexScheduler
      gpu_memory_utilization: 0.85
      enforce_eager: false
      engine_output_type: latent
      enable_prefix_caching: false
      max_num_batched_tokens: 8192
      step_interval_ms: 80                   # NEW: 12.5Hz stepping
    final_output: true
    final_output_type: ["text", "audio"]     # NEW: dual output
    is_comprehension: true
    accepts_streaming_input: true             # NEW: can receive input during generation
    default_sampling_params:
      temperature: 0.7
      top_k: 50
      max_tokens: 0      # 0 = unlimited (continuous generation)

  - stage_id: 1
    stage_type: llm
    runtime:
      devices: "0"
      max_batch_size: 1
    engine_args:
      model_stage: mimi_decode
      model_arch: MoshiForConditionalGeneration
      worker_type: generation
      scheduler_cls: vllm_omni.core.sched.omni_generation_scheduler.OmniGenerationScheduler
      gpu_memory_utilization: 0.05
      engine_output_type: audio
      chunked_decode: true                   # Decode in 80ms chunks
    engine_input_source: [0]
    custom_process_next_stage_input_func: >
      vllm_omni.model_executor.stage_input_processors.moshi.dialogue_to_mimi_decode_streaming
    final_output: true
    final_output_type: audio
```

### 3.6 Phase 3 Full-Duplex Data Flow

```
                    ┌──────── WebSocket ────────┐
                    │                            │
                    ↓                            ↑
            [Audio Input]                  [Audio Output]
                    │                            ↑
                    ↓                            │
            [Mimi Encoder]               [Mimi Decoder]
                    │                            ↑
                    ↓                            │
         user_audio_tokens              audio_codes chunk
                    │                            ↑
                    ↓ (backward inject)          │ (forward stream)
          ┌─────────────────────────────────────────┐
          │         Moshi Dialogue Stage             │
          │                                          │
          │  ┌──────────────────────────┐           │
          │  │  Temporal Transformer    │           │
          │  │  (7B, AR, PagedAttn)     │           │
          │  │                          │           │
          │  │  streams 0-8: moshi      │           │
          │  │  streams 9-16: user ←────│── inject  │
          │  └─────────┬────────────────┘           │
          │            │ temporal_context            │
          │            ↓                             │
          │  ┌──────────────────────────┐           │
          │  │  Depth Transformer       │           │
          │  │  (6L, 8-step AR loop)    │           │
          │  │  → 8 codebook tokens     │           │
          │  └─────────┬────────────────┘           │
          │            │ audio_codes                 │
          └────────────┼────────────────────────────┘
                       │
                       ↓ (every 80ms)
               [Mimi Decoder] → Audio chunk → WebSocket → Client
```

---

## Implementation Order & Dependencies

```
Phase 1 ─────────────────────────────────────────────────
  │
  ├─ 1a. Mimi encoder/decoder wrapper (mimi.py)
  ├─ 1b. Temporal Transformer wrapper (moshi_temporal.py)
  ├─ 1c. Depth Transformer wrapper (moshi_depth.py)
  │       ↓
  ├─ 1d. Unified model class (moshi.py)
  │       ↓
  ├─ 1e. Registry + stage config + input processor
  │       ↓
  └─ 1f. End-to-end test (test_moshi.py)

Phase 2 ─────────────────────────────────────────────────
  │  (depends on Phase 1)
  │
  ├─ 2a. Dual output type in outputs.py
  ├─ 2b. Duplex protocol definitions
  │       ↓
  ├─ 2c. WebSocket endpoint + handler
  ├─ 2d. moshi_streaming.yaml + async chunk processors
  │       ↓
  └─ 2e. Streaming audio test

Phase 3 ─────────────────────────────────────────────────
  │  (depends on Phase 2)
  │
  ├─ 3a. DuplexConnector (bidirectional)
  ├─ 3b. OmniDuplexScheduler (pause/resume/inject)
  │       ↓
  ├─ 3c. GPUDuplexWorker + ModelRunner
  │       ↓
  ├─ 3d. Modified orchestrator loop (omni.py)
  │       ↓
  ├─ 3e. Bidirectional WebSocket handler
  ├─ 3f. moshi_duplex.yaml
  │       ↓
  └─ 3g. Full-duplex integration test
```

---

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| HF Moshi weights incompatible with custom wrapper | High | HF `MoshiForConditionalGeneration`를 base class로 사용, weight mapping 검증 |
| Depth Transformer 내부 루프가 vLLM 성능 저하 | Medium | CUDA graph 적용, batch=1 최적화 |
| KV cache 동적 확장 시 OOM | High | User audio token budget 제한, pre-allocated block pool |
| 80ms latency budget 초과 | Medium | Mimi를 같은 GPU에 배치, 통신 overhead 최소화 |
| upstream vLLM 호환성 깨짐 | High | Phase 3 변경사항을 별도 모듈로 격리, 기존 코드 수정 최소화 |

---

## Success Criteria

| Phase | Metric | Target |
|-------|--------|--------|
| Phase 1 | End-to-end inference | 사용자 오디오 입력 → Moshi 응답 오디오 출력 |
| Phase 1 | Latency | < 5s for 10s input audio |
| Phase 2 | First audio chunk | < 500ms after generation starts |
| Phase 2 | Streaming throughput | Real-time (12.5 frames/sec sustained) |
| Phase 3 | Full-duplex latency | < 200ms (theoretical minimum: 160ms) |
| Phase 3 | Concurrent I/O | Continuous audio input during generation |

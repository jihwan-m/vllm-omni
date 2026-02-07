# Moshi Full-Duplex Implementation Plan: Phase 2 & Phase 3

> **Status**: Phase 1 (half-duplex, turn-taking) is COMPLETED and tested (45/45 unit tests passing).
> This document details Phase 2 (chunked streaming + WebSocket) and Phase 3 (true full-duplex).

---

## Current State (Phase 1 Completed)

### What Exists

| Component | File | Status |
|-----------|------|--------|
| Temporal Transformer (7B) | `models/moshi/moshi.py` (MoshiTemporalDecoder) | Done |
| Depth Transformer (6L) | `models/moshi/moshi.py` (MoshiDepthDecoder) | Done |
| Mimi Decoder stub | `models/moshi/moshi.py` (MimiDecoderModel) | Stub only |
| Stage config (2-stage) | `stage_configs/moshi.yaml` | Done |
| Stage input processor | `stage_input_processors/moshi.py` | Done |
| Model registry entry | `models/registry.py` | Done |
| Unit tests (45 tests) | `tests/model_executor/models/test_moshi.py` | All passing |

### Current Architecture (Turn-Taking)

```
User speaks → [complete] → System processes → [complete] → System responds → [complete]

  ┌──────────────────────────────────────────────────────────────────────┐
  │ Stage 0: dialogue (generation worker)                                │
  │ ┌─────────────────────┐    ┌──────────────────┐                     │
  │ │ Temporal Transformer │───→│ Depth Transformer │──→ audio_codes[T,8]│
  │ │ (7B, 32 layers)     │    │ (6L, 8-step AR)  │                     │
  │ └─────────────────────┘    └──────────────────┘                     │
  └──────────────────────────────────┬───────────────────────────────────┘
                                     │ SHM Connector
  ┌──────────────────────────────────▼───────────────────────────────────┐
  │ Stage 1: mimi_decode (generation worker)                             │
  │ ┌──────────┐                                                         │
  │ │ Mimi Dec │──→ waveform tensor                                      │
  │ └──────────┘                                                         │
  └──────────────────────────────────────────────────────────────────────┘
```

### Fundamental Architectural Constraints Identified

| # | Constraint | Impact | Where |
|---|-----------|--------|-------|
| C1 | **One-shot execution**: Generation workers finish in single scheduler cycle | No mid-generation state for resumption | `omni_generation_scheduler.py:357-371` |
| C2 | **Forward-only connectors**: SHM connector is unidirectional per edge | No feedback from downstream to upstream | `shm_connector.py`, `adapter.py` |
| C3 | **Blocking stage worker**: `in_q.get()` blocks the entire stage loop | Cannot receive input while generating | `omni_stage.py:783` |
| C4 | **Discrete chunk boundaries**: No true streaming, only materialized chunks | Must wait for complete chunk before forwarding | `shm_connector.py:110`, `adapter.py:243` |
| C5 | **Immediate resource freeing**: Scheduler frees KV/blocks after completion | Cannot resume a finished request | `omni_generation_scheduler.py:371` |
| C6 | **No request interruption**: No abort signal within GPUGenerationWorker | Running requests cannot be stopped for new input | `gpu_generation_worker.py` |
| C7 | **Deferred sampling**: `execute_model()` returns None, state stored | Cannot stream pooler outputs during forward | `gpu_generation_model_runner.py:273-284` |

---

## Phase 2: Chunked Streaming Output + WebSocket

> **Goal**: Stream Moshi's audio output to the client in real-time via WebSocket while
> Mimi decodes chunks incrementally. Input remains turn-based (user speaks, then listens).
> This addresses **C4** (chunking) and adds client-facing streaming, without modifying
> the core scheduler or worker lifecycle.

### 2.0 Architecture Overview

```
                          Phase 2 Architecture
                          ─────────────────────

  Client ←──── WebSocket ────→ API Server
    │                              │
    │  1. Send audio input         │
    │  ──────────────────→         │
    │                              │
    │                    ┌─────────▼──────────┐
    │                    │ Orchestrator (omni) │
    │                    └─────────┬──────────┘
    │                              │
    │              ┌───────────────▼───────────────┐
    │              │ Stage 0: dialogue              │
    │              │ Temporal→Depth (generation)    │
    │              │ async_chunk: true               │
    │              │   └─ puts audio_code chunks    │
    │              │      every N steps via SHM     │
    │              └───────────────┬───────────────┘
    │                              │ SHM (chunked)
    │              ┌───────────────▼───────────────┐
    │              │ Stage 1: mimi_decode           │
    │              │ Mimi decoder (generation)      │
    │              │   └─ produces waveform chunks  │
    │              └───────────────┬───────────────┘
    │                              │
    │  2. Stream audio chunks      │
    │  ←──────────────────         │
    │  (PCM/Opus frames)           │
    │                              │
    │  3. "done" signal            │
    │  ←──────────────────         │
```

### 2.1 New Files to Create

```
vllm_omni/entrypoints/openai/
├── serving_duplex.py              # WebSocket handler (audio streaming)
├── protocol/
│   └── duplex.py                  # WebSocket message protocol definitions

vllm_omni/model_executor/stage_configs/
└── moshi_streaming.yaml           # async_chunk stage config for Moshi

vllm_omni/model_executor/stage_input_processors/
└── moshi_streaming.py             # Chunked stage input processors

tests/
├── entrypoints/test_serving_duplex.py   # WebSocket handler tests
└── model_executor/models/test_moshi_streaming.py  # Streaming pipeline tests
```

### 2.2 Files to Modify

| File | Change | Constraint Addressed |
|------|--------|---------------------|
| `entrypoints/openai/api_server.py` | Add WebSocket route `/v1/audio/duplex` | New endpoint |
| `entrypoints/omni.py` | Add `_run_streaming_generation()` that yields partial outputs | C4 |
| `model_executor/models/moshi/moshi.py` | Dialogue stage outputs intermediate chunks | C4 |
| `engine/output_processor.py` | Support streaming partial audio to client | C4 |
| `protocol/audio.py` | Remove SSE block, add streaming format support | Existing limitation |

### 2.3 Detailed Component Design

#### 2.3.1 WebSocket Protocol (`protocol/duplex.py`)

```python
"""WebSocket message protocol for real-time audio streaming."""
from enum import Enum
from pydantic import BaseModel

class DuplexMessageType(str, Enum):
    # Client → Server
    SESSION_START = "session.start"
    AUDIO_INPUT = "audio.input"           # Binary: raw PCM audio chunk
    AUDIO_INPUT_DONE = "audio.input.done" # Signal: user finished speaking
    SESSION_END = "session.end"

    # Server → Client
    SESSION_CREATED = "session.created"
    AUDIO_OUTPUT = "audio.output"         # Binary: PCM/Opus audio chunk
    TEXT_TOKEN = "text.token"             # Intermediate text token
    GENERATION_DONE = "generation.done"   # Signal: response complete
    ERROR = "error"

class SessionStartMessage(BaseModel):
    type: str = DuplexMessageType.SESSION_START
    model: str = "moshi"
    sample_rate: int = 24000
    audio_format: str = "pcm_s16le"       # Input audio format
    output_format: str = "pcm_s16le"      # Output audio format
    temperature: float = 0.7
    top_k: int = 25

class SessionCreatedMessage(BaseModel):
    type: str = DuplexMessageType.SESSION_CREATED
    session_id: str
    model: str
    sample_rate: int

class AudioOutputChunk(BaseModel):
    """Metadata for audio output (binary data sent as separate WebSocket frame)."""
    type: str = DuplexMessageType.AUDIO_OUTPUT
    chunk_index: int
    duration_ms: float
    is_final: bool = False
```

**Design Rationale**:
- Binary WebSocket frames for audio data (no base64 overhead)
- JSON text frames for control messages
- Stateless protocol — each session is independent
- `chunk_index` enables client-side reordering if needed

#### 2.3.2 WebSocket Handler (`serving_duplex.py`)

```python
"""WebSocket endpoint for real-time Moshi audio streaming."""
import asyncio
import json
import uuid

from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from vllm_omni.entrypoints.openai.protocol.duplex import (
    DuplexMessageType,
    SessionCreatedMessage,
)

class MoshiDuplexHandler:
    """Manages a single WebSocket session for Moshi audio I/O."""

    def __init__(self, engine_client, model_config):
        self.engine_client = engine_client
        self.model_config = model_config

    async def handle(self, websocket: WebSocket):
        await websocket.accept()
        session_id = str(uuid.uuid4())

        try:
            # 1. Wait for session.start
            config = await self._recv_session_start(websocket)

            # 2. Send session.created
            await websocket.send_json(
                SessionCreatedMessage(
                    session_id=session_id,
                    model=config.model,
                    sample_rate=config.sample_rate,
                ).model_dump()
            )

            # 3. Receive audio input until audio.input.done
            audio_buffer = await self._collect_audio_input(websocket)

            # 4. Run Moshi pipeline, streaming output chunks
            await self._stream_generation(
                websocket, session_id, audio_buffer, config
            )

        except WebSocketDisconnect:
            pass
        finally:
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.close()

    async def _collect_audio_input(self, websocket: WebSocket) -> bytes:
        """Collect all audio input until client signals done."""
        chunks = []
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.receive":
                if "bytes" in message:
                    chunks.append(message["bytes"])
                elif "text" in message:
                    data = json.loads(message["text"])
                    if data.get("type") == DuplexMessageType.AUDIO_INPUT_DONE:
                        break
        return b"".join(chunks)

    async def _stream_generation(
        self, websocket, session_id, audio_bytes, config
    ):
        """Run Moshi inference and stream audio chunks back via WebSocket."""
        # Encode audio input via Mimi encoder
        audio_tokens = self._encode_audio_input(audio_bytes, config)

        # Submit to pipeline with streaming=True
        chunk_index = 0
        async for audio_chunk in self.engine_client.generate_streaming(
            request_id=session_id,
            audio_tokens=audio_tokens,
            config=config,
        ):
            # Send binary audio frame
            await websocket.send_bytes(audio_chunk.audio_data)

            # Send metadata text frame
            await websocket.send_json({
                "type": DuplexMessageType.AUDIO_OUTPUT,
                "chunk_index": chunk_index,
                "duration_ms": audio_chunk.duration_ms,
                "is_final": audio_chunk.is_final,
            })
            chunk_index += 1

        # Signal completion
        await websocket.send_json({
            "type": DuplexMessageType.GENERATION_DONE,
            "session_id": session_id,
            "total_chunks": chunk_index,
        })
```

#### 2.3.3 Streaming Orchestrator Extension (`omni.py`)

The current orchestrator's `_run_generation()` yields `OmniRequestOutput` only when a stage
has `final_output=True` and the request is complete. For Phase 2, we add a streaming variant.

```python
# In class Omni, add:

def _run_streaming_generation(
    self,
    prompts,
    sampling_params_list,
) -> Generator[OmniStreamingChunk, None, None]:
    """Run generation with intermediate chunk yielding.

    Unlike _run_generation(), this yields partial audio chunks from
    the final stage as they become available via the async_chunk pipeline,
    rather than waiting for full completion.
    """
    # ... (seed stage-0 same as _run_generation) ...

    while not completed:
        for stage_id, stage in enumerate(self.stage_list):
            result = stage.try_collect()
            if result is None:
                continue

            # Check if this is an intermediate chunk (not final)
            is_chunk = result.get("is_chunk", False)
            is_final = result.get("is_final", False)

            if is_chunk or is_final:
                engine_outputs = _load(result, ...)
                yield OmniStreamingChunk(
                    stage_id=stage_id,
                    audio_data=engine_outputs.get("audio_chunk"),
                    chunk_index=result.get("chunk_index", 0),
                    is_final=is_final,
                )

            if is_final:
                completed = True

            # Forward to next stage (existing logic)
            if not getattr(stage, "final_output", False):
                next_stage_id = stage_id + 1
                # ... (existing forwarding logic) ...
```

**Key Difference from `_run_generation()`**:
- Yields `OmniStreamingChunk` (lightweight, partial audio data) instead of `OmniRequestOutput`
- Does not wait for full request completion before yielding
- Compatible with existing stage/connector infrastructure via `async_chunk`

#### 2.3.4 Moshi Streaming Stage Config (`moshi_streaming.yaml`)

```yaml
# Stage config for Moshi with async chunked streaming
model: "kmhf/hf-moshiko"
model_arch: MoshiForConditionalGenerationVLLM
async_chunk: true

stage_args:
  - stage_id: 0
    engine_args:
      model_stage: dialogue
      model_arch: MoshiForConditionalGenerationVLLM
      worker_type: generation
      scheduler_cls: >-
        vllm_omni.core.sched.omni_generation_scheduler.OmniGenerationScheduler
      gpu_memory_utilization: 0.85
      trust_remote_code: true
      dtype: bfloat16
    final_output: false
    final_output_type: audio
    custom_process_next_stage_input_func: >-
      vllm_omni.model_executor.stage_input_processors.moshi_streaming.dialogue_to_mimi_async_chunk

  - stage_id: 1
    engine_args:
      model_stage: mimi_decode
      model_arch: MoshiForConditionalGenerationVLLM
      worker_type: generation
      scheduler_cls: >-
        vllm_omni.core.sched.omni_generation_scheduler.OmniGenerationScheduler
      gpu_memory_utilization: 0.05
      trust_remote_code: true
      dtype: bfloat16
    final_output: true
    final_output_type: audio
```

#### 2.3.5 Chunked Stage Input Processor (`moshi_streaming.py`)

```python
"""Chunked stage input processors for Moshi streaming pipeline."""
import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

# Moshi operates at 12.5Hz frame rate (80ms per frame)
# We accumulate N frames before sending a chunk to Mimi decoder
MOSHI_CHUNK_SIZE = 25  # 25 frames = 2 seconds of audio (matches Qwen3 pattern)
MOSHI_NUM_CODEBOOKS = 8


def dialogue_to_mimi_async_chunk(
    connector,
    pooling_output: dict,
    request,
) -> dict | None:
    """Accumulate audio code chunks from dialogue stage for Mimi decoding.

    Called by the scheduler's put_chunk() after each generation step.
    Returns None to defer (accumulate more), or a dict to send.

    Audio codes shape per step: [num_codebooks] (8 integer codes)
    Accumulated to [chunk_size, num_codebooks] before sending.
    """
    request_id = request.external_req_id

    # Extract single-step audio codes from pooling output
    audio_codes = pooling_output.get("audio_codes")
    if audio_codes is None:
        return None

    # Accumulate codes
    connector.code_prompt_token_ids[request_id].append(audio_codes)
    accumulated = len(connector.code_prompt_token_ids[request_id])

    is_finished = request.is_finished()
    at_chunk_boundary = (accumulated % MOSHI_CHUNK_SIZE == 0)

    if not at_chunk_boundary and not is_finished:
        return None  # Wait for more codes

    # Determine chunk window
    chunk_length = accumulated % MOSHI_CHUNK_SIZE
    if chunk_length == 0:
        chunk_length = MOSHI_CHUNK_SIZE

    # Get the codes for this chunk
    codes = connector.code_prompt_token_ids[request_id][-chunk_length:]

    # Stack [chunk_length, 8] → transpose [8, chunk_length] → flatten
    codes_tensor = torch.tensor(codes)  # [chunk_length, 8]
    flat_codes = codes_tensor.t().reshape(-1).tolist()  # [8 * chunk_length]

    return {
        "code_predictor_codes": flat_codes,
        "finished": torch.tensor(is_finished, dtype=torch.bool),
        "chunk_index": connector.put_requests[request_id],
    }
```

#### 2.3.6 Dialogue Stage: Per-Step Audio Code Emission

The current `_forward_dialogue()` in `moshi.py` runs the full temporal+depth loop
internally and returns all audio codes at once. For streaming, we need it to emit
codes incrementally via the pooling output mechanism.

**Changes to `moshi.py`**:

```python
# In MoshiForConditionalGenerationVLLM._forward_dialogue():
# Current: runs all T steps, returns complete audio_codes [T, 8]
# New: model tracks internal state, returns codes for current step only

def forward(self, ...):
    if self.model_stage == "dialogue":
        return self._forward_dialogue_step(
            input_ids, positions, intermediate_tensors, ...
        )

def _forward_dialogue_step(self, input_ids, positions, ...):
    """Single temporal+depth step (called once per scheduler cycle).

    Instead of running the full AR loop internally, the scheduler
    calls this method repeatedly. Each call:
    1. Runs one temporal transformer step (with KV cache)
    2. Runs the 8-step depth AR loop for that position
    3. Returns audio codes [8] via pooling output
    """
    # ... single temporal step + depth step ...
    # Returns via OmniOutput with multimodal_outputs={"audio_codes": codes}
```

**Design Decision**: The dialogue stage switches from running the full AR loop
internally (`generation` worker doing everything in one shot) to an `ar` worker
pattern where the scheduler drives the loop step-by-step. This enables the
`put_chunk()` mechanism to emit intermediate codes after each step.

**This requires changing the dialogue stage's `worker_type` from `generation` to `ar`.**

### 2.4 Implementation Steps (Ordered)

```
Step 1: WebSocket Infrastructure
├── 2.4.1 Create protocol/duplex.py (message types)
├── 2.4.2 Create serving_duplex.py (WebSocket handler)
├── 2.4.3 Register WebSocket route in api_server.py
└── 2.4.4 Tests for WebSocket protocol

Step 2: Streaming Pipeline
├── 2.4.5 Refactor _forward_dialogue() to per-step emission
├── 2.4.6 Create moshi_streaming.yaml (async_chunk config)
├── 2.4.7 Create moshi_streaming.py (chunked processor)
├── 2.4.8 Add _run_streaming_generation() to orchestrator
└── 2.4.9 Integration tests

Step 3: Client-Facing Streaming
├── 2.4.10 Wire WebSocket handler to streaming orchestrator
├── 2.4.11 Add OmniStreamingChunk output type
├── 2.4.12 Update output_processor.py for partial audio
└── 2.4.13 End-to-end streaming test
```

### 2.5 Dependency Graph

```
                protocol/duplex.py
                       │
                       ▼
              serving_duplex.py ←──── api_server.py (route)
                       │
                       ▼
           _run_streaming_generation()  (omni.py)
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
   moshi.py        moshi_        output_processor.py
  (per-step)    streaming.py   (partial audio emit)
                (chunk proc)
                       │
                       ▼
             moshi_streaming.yaml
```

### 2.6 Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| AR worker KV cache management for Moshi's internal state | High | Moshi's depth decoder has its own KV cache that resets per temporal step; temporal KV cache is managed by vLLM's PagedAttention. Need to verify depth cache is handled outside PagedAttention. |
| Chunk boundary alignment with Mimi decoder | Medium | Mimi expects contiguous RVQ codes. Ensure chunk boundaries don't split mid-frame (each frame = 8 codebooks). |
| WebSocket backpressure if client is slow | Medium | Buffer limit with drop-oldest policy; send `audio.output` metadata after binary frame so client can detect drops. |
| Thread safety in streaming orchestrator | Medium | Use asyncio.Queue for chunk delivery; orchestrator runs in its own thread, WebSocket handler awaits on queue. |

---

## Phase 3: True Full-Duplex

> **Goal**: User and system speak simultaneously. User audio is continuously
> encoded and injected into the Temporal Transformer while it generates
> response audio. This addresses **C1-C7** (all constraints).

### 3.0 Architecture Overview

```
              Phase 3 Full-Duplex Architecture
              ─────────────────────────────────

  ┌─────────────────────────────────────────────────────────────────┐
  │                    WebSocket Session                             │
  │                                                                 │
  │  Client ◄─── audio frames ───► Server                          │
  │         ◄─── text tokens  ───►                                  │
  │         ◄─── control msgs ───►                                  │
  └───────────────────┬─────────────────────────────────────────────┘
                      │
  ┌───────────────────▼─────────────────────────────────────────────┐
  │              DuplexSessionManager                                │
  │  ┌────────────────┐  ┌─────────────────────┐                    │
  │  │ Input Pipeline  │  │ Output Pipeline      │                   │
  │  │ (async task)    │  │ (async task)         │                   │
  │  │                 │  │                      │                   │
  │  │ audio_in ──→    │  │    ──→ audio_out     │                   │
  │  │ mimi_encode ──→ │  │    ──→ mimi_decode   │                   │
  │  │ token_buffer ──→│  │    ──→ ws.send       │                   │
  │  └────────┬────────┘  └──────────▲───────────┘                   │
  │           │                      │                               │
  └───────────┼──────────────────────┼───────────────────────────────┘
              │                      │
  ┌───────────▼──────────────────────┼───────────────────────────────┐
  │         DuplexDialogueStage                                      │
  │                                                                  │
  │  ┌──────────────────────────────────────────┐                    │
  │  │  Input Ring Buffer                        │                   │
  │  │  [user_codes_t-2, user_codes_t-1, ...]   │                   │
  │  └──────────────────┬───────────────────────┘                    │
  │                     │                                            │
  │  ┌──────────────────▼───────────────────────┐                    │
  │  │  Temporal Transformer (7B)                │                   │
  │  │  Input: text_embed + moshi_audio_embed    │                   │
  │  │         + user_audio_embed (from buffer)  │                   │
  │  │  KV Cache: managed by AR scheduler        │                   │
  │  │  Output: hidden_state → text logits       │───→ text tokens   │
  │  └──────────────────┬───────────────────────┘                    │
  │                     │                                            │
  │  ┌──────────────────▼───────────────────────┐                    │
  │  │  Depth Transformer (6L)                   │                   │
  │  │  8-step AR loop per temporal step         │                   │
  │  │  Output: audio_codes[8]                   │───→ audio codes   │
  │  └──────────────────────────────────────────┘        │           │
  └──────────────────────────────────────────────────────┼───────────┘
                                                         │
  ┌──────────────────────────────────────────────────────▼───────────┐
  │  Mimi Decoder (streaming)                                        │
  │  Converts RVQ codes → 24kHz PCM waveform chunks                  │
  └──────────────────────────────────────────────────────────────────┘
```

### 3.1 Architectural Changes Required

#### 3.1.1 Bidirectional Connector (`BidirectionalConnector`)

**Problem** (C2): Current connectors only support `put(from→to)` / `get(from→to)`.
Full-duplex needs client audio to flow backward into the dialogue stage while
audio output flows forward.

**Solution**: Create a `BidirectionalChannel` that wraps two unidirectional SHM
connectors plus a shared ring buffer for continuous audio input.

```python
class BidirectionalChannel:
    """Two-way communication channel between session manager and stage."""

    def __init__(self, session_id: str, buffer_size: int = 128):
        # Forward: stage → session (audio output codes)
        self.forward_queue = asyncio.Queue(maxsize=buffer_size)

        # Backward: session → stage (user audio codes)
        self.backward_buffer = RingBuffer(capacity=buffer_size)

        # Control: session ↔ stage (interrupt, config changes)
        self.control_queue = asyncio.Queue(maxsize=16)

        self.session_id = session_id
        self._closed = False

    # --- Forward (stage → client) ---
    async def put_output(self, audio_codes: list[int], text_token: int | None):
        """Stage puts generated output codes (non-blocking with backpressure)."""
        await self.forward_queue.put(OutputFrame(audio_codes, text_token))

    async def get_output(self) -> OutputFrame | None:
        """Session manager gets output for client (async, blocks until available)."""
        return await self.forward_queue.get()

    # --- Backward (client → stage) ---
    def put_input(self, user_audio_codes: list[int]):
        """Session manager puts user audio codes (from Mimi encoder)."""
        self.backward_buffer.write(user_audio_codes)

    def get_input(self, position: int) -> list[int] | None:
        """Stage reads user audio codes at given temporal position."""
        return self.backward_buffer.read(position)

    # --- Control ---
    async def send_interrupt(self):
        """Signal the stage to stop current generation."""
        await self.control_queue.put(ControlSignal.INTERRUPT)

    def poll_control(self) -> ControlSignal | None:
        """Stage polls for control signals (non-blocking)."""
        try:
            return self.control_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None
```

**New file**: `vllm_omni/distributed/omni_connectors/bidirectional.py`

#### 3.1.2 Ring Buffer for Continuous Audio Input

```python
class RingBuffer:
    """Lock-free ring buffer for continuous audio token injection.

    Writer (session manager) appends user audio codes at the current position.
    Reader (dialogue stage) reads codes at any position within the window.
    Thread-safe via atomic position counter.
    """

    def __init__(self, capacity: int = 256, num_codebooks: int = 8):
        self.capacity = capacity
        self.num_codebooks = num_codebooks
        # Shared memory buffer: [capacity, num_codebooks]
        self.buffer = torch.zeros(capacity, num_codebooks, dtype=torch.long)
        self.write_pos = 0  # Atomic counter
        self.valid_from = 0  # Oldest valid position

    def write(self, codes: list[int]):
        """Append one frame of user audio codes [num_codebooks]."""
        idx = self.write_pos % self.capacity
        self.buffer[idx] = torch.tensor(codes, dtype=torch.long)
        self.write_pos += 1
        if self.write_pos - self.valid_from > self.capacity:
            self.valid_from = self.write_pos - self.capacity

    def read(self, position: int) -> torch.Tensor | None:
        """Read codes at temporal position. Returns None if evicted."""
        if position < self.valid_from or position >= self.write_pos:
            return None
        idx = position % self.capacity
        return self.buffer[idx].clone()
```

#### 3.1.3 Interruptible Dialogue Worker

**Problem** (C3, C6): The stage worker blocks on `in_q.get()` and has no
mechanism to inject new audio or interrupt generation mid-step.

**Solution**: Replace the blocking batch loop with an event-driven loop that
checks both the input queue AND the bidirectional channel each step.

```python
class DuplexStageWorker:
    """Event-driven stage worker for full-duplex Moshi.

    Replaces the standard _stage_worker() blocking loop.
    Runs temporal+depth steps in a continuous loop, checking for
    user audio input and control signals between steps.
    """

    def __init__(self, model, channel: BidirectionalChannel, ...):
        self.model = model
        self.channel = channel
        self.temporal_step = 0
        self.kv_cache = None

    def run(self):
        """Main event loop: generate continuously, check for input each step."""
        while not self.channel._closed:
            # 1. Check for control signals
            signal = self.channel.poll_control()
            if signal == ControlSignal.INTERRUPT:
                self._handle_interrupt()
                continue

            # 2. Get user audio codes for current step (from ring buffer)
            user_codes = self.channel.get_input(self.temporal_step)
            if user_codes is None:
                # No input yet — use silence padding
                user_codes = torch.zeros(self.num_codebooks, dtype=torch.long)

            # 3. Run one temporal step
            hidden, text_logits, self.kv_cache = self.model.temporal_step(
                text_token=self.prev_text_token,
                moshi_audio_codes=self.prev_audio_codes,
                user_audio_codes=user_codes,
                position=self.temporal_step,
                past_key_values=self.kv_cache,
            )

            # 4. Sample text token
            text_token = torch.argmax(text_logits, dim=-1).item()

            # 5. Run depth decoder (8-step AR)
            audio_codes = self.model.depth_decoder.generate_codes(
                hidden[:, -1, :], text_token=text_token, ...
            )

            # 6. Put output codes into forward channel (non-blocking)
            self.channel.put_output_sync(audio_codes, text_token)

            # 7. Advance state
            self.prev_text_token = text_token
            self.prev_audio_codes = audio_codes
            self.temporal_step += 1

    def _handle_interrupt(self):
        """Clear KV cache and reset state for new turn."""
        self.kv_cache = None
        self.temporal_step = 0
        self.prev_text_token = None
        self.prev_audio_codes = None
```

#### 3.1.4 Duplex Session Manager

**Problem**: The orchestrator currently manages request-response lifecycles.
Full-duplex requires persistent sessions with concurrent I/O.

```python
class DuplexSessionManager:
    """Manages a persistent full-duplex session.

    Coordinates three concurrent tasks:
    1. Input pipeline: WebSocket → Mimi encoder → ring buffer
    2. Output pipeline: forward channel → Mimi decoder → WebSocket
    3. Dialogue worker: continuous temporal+depth generation loop
    """

    def __init__(self, websocket, model, mimi_encoder, mimi_decoder):
        self.websocket = websocket
        self.channel = BidirectionalChannel(session_id=str(uuid.uuid4()))
        self.worker = DuplexStageWorker(model, self.channel)
        self.mimi_encoder = mimi_encoder
        self.mimi_decoder = mimi_decoder

    async def run(self):
        """Launch all three concurrent tasks."""
        async with asyncio.TaskGroup() as tg:
            tg.create_task(self._input_pipeline())
            tg.create_task(self._output_pipeline())
            tg.create_task(asyncio.to_thread(self.worker.run))

    async def _input_pipeline(self):
        """Receive audio from WebSocket, encode with Mimi, write to ring buffer."""
        async for message in self.websocket.iter_bytes():
            # Decode PCM → Mimi encoder → RVQ codes [8]
            pcm_chunk = self._decode_pcm(message)
            codes = self.mimi_encoder.encode_chunk(pcm_chunk)

            # Write to ring buffer (consumed by dialogue worker)
            self.channel.put_input(codes)

    async def _output_pipeline(self):
        """Read generated codes from channel, decode with Mimi, send to client."""
        chunk_index = 0
        while True:
            frame = await self.channel.get_output()
            if frame is None:
                break

            # Decode RVQ codes → PCM audio
            pcm_data = self.mimi_decoder.decode_chunk(frame.audio_codes)

            # Send to client
            await self.websocket.send_bytes(pcm_data)
            await self.websocket.send_json({
                "type": "audio.output",
                "chunk_index": chunk_index,
                "text_token": frame.text_token,
            })
            chunk_index += 1
```

#### 3.1.5 Scheduler Modifications

**Problem** (C1, C5): The scheduler frees resources after one cycle and doesn't
support persistent requests.

**Solution**: Add a `RUNNING_CONTINUOUS` state for requests that persist across
scheduler cycles without being freed.

```python
# In omni_generation_scheduler.py, add:

class RequestStatus(Enum):
    WAITING = "waiting"
    RUNNING = "running"
    RUNNING_CONTINUOUS = "running_continuous"  # NEW: persists across cycles
    FINISHED_STOPPED = "finished_stopped"

# In schedule():
def schedule(self) -> SchedulerOutput:
    for request in self.running:
        if request.status == RequestStatus.RUNNING_CONTINUOUS:
            # Don't free resources; re-schedule for next step
            # Refresh input from bidirectional channel
            user_codes = request.duplex_channel.get_input(request.step)
            request.inject_input(user_codes)
            scheduled_running_reqs.append(request)

# In update_from_output():
def update_from_output(self, scheduler_output, model_runner_output):
    for request in ...:
        if request.status == RequestStatus.RUNNING_CONTINUOUS:
            # Emit partial output but DON'T free resources
            request.step += 1
            # Put output codes to channel
            request.duplex_channel.put_output_sync(
                audio_codes=model_runner_output.audio_codes,
                text_token=model_runner_output.text_token,
            )
            # DO NOT call _free_request()
```

### 3.2 New Files to Create

```
vllm_omni/distributed/omni_connectors/
├── bidirectional.py               # BidirectionalChannel + RingBuffer
└── connectors/
    └── duplex_shm_connector.py    # SHM-backed bidirectional connector

vllm_omni/entrypoints/
├── duplex_session.py              # DuplexSessionManager
└── duplex_stage_worker.py         # DuplexStageWorker (event-driven loop)

vllm_omni/core/sched/
└── omni_duplex_scheduler.py       # Scheduler with RUNNING_CONTINUOUS support

vllm_omni/model_executor/models/moshi/
├── mimi_encoder.py                # Streaming Mimi encoder (audio → RVQ codes)
└── mimi_streaming.py              # Streaming Mimi decoder (RVQ codes → audio)

vllm_omni/model_executor/stage_configs/
└── moshi_duplex.yaml              # Full-duplex stage config

tests/
├── distributed/test_bidirectional.py    # Bidirectional channel tests
├── entrypoints/test_duplex_session.py   # Session manager tests
└── e2e/test_moshi_duplex.py             # End-to-end duplex test
```

### 3.3 Files to Modify

| File | Change | Constraints Addressed |
|------|--------|----------------------|
| `omni_stage.py` | Add `_duplex_stage_worker()` entry point alongside `_stage_worker()` | C3, C6 |
| `omni_generation_scheduler.py` | Add `RUNNING_CONTINUOUS` state, skip `_free_request()` | C1, C5 |
| `gpu_generation_worker.py` | Add `inject_input()` for mid-generation input | C3 |
| `gpu_generation_model_runner.py` | Support partial output emission during forward | C7 |
| `moshi.py` | Add `temporal_step()` single-step API alongside full forward | C1 |
| `omni.py` | Add `_run_duplex_session()` alongside `_run_generation()` | All |
| `adapter.py` | Support bidirectional channel in addition to unidirectional SHM | C2 |
| `serving_duplex.py` | Upgrade from turn-taking (Phase 2) to true full-duplex | All |
| `protocol/duplex.py` | Add continuous I/O message types | New |

### 3.4 Implementation Steps (Ordered)

```
Step 1: Core Infrastructure (no model changes)
├── 3.4.1 Implement RingBuffer with tests
├── 3.4.2 Implement BidirectionalChannel with tests
├── 3.4.3 Add RUNNING_CONTINUOUS to scheduler
└── 3.4.4 Unit tests for all above

Step 2: Moshi Model Per-Step API
├── 3.4.5 Add temporal_step() to MoshiTemporalDecoder
├── 3.4.6 Add streaming Mimi encoder (mimi_encoder.py)
├── 3.4.7 Add streaming Mimi decoder (mimi_streaming.py)
├── 3.4.8 Unit tests for per-step inference
└── 3.4.9 Verify temporal+depth step produces same output as full loop

Step 3: Event-Driven Worker
├── 3.4.10 Implement DuplexStageWorker
├── 3.4.11 Add _duplex_stage_worker() to omni_stage.py
├── 3.4.12 Implement inject_input() in GPUGenerationWorker
└── 3.4.13 Integration test: worker + channel

Step 4: Session Management
├── 3.4.14 Implement DuplexSessionManager
├── 3.4.15 Add _run_duplex_session() to orchestrator
├── 3.4.16 Update serving_duplex.py for continuous I/O
└── 3.4.17 End-to-end duplex test

Step 5: Optimization
├── 3.4.18 CUDA stream overlap for input encoding + generation
├── 3.4.19 Opus encoding for WebSocket output compression
├── 3.4.20 Latency measurement and profiling
└── 3.4.21 Backpressure tuning
```

### 3.5 Dependency Graph

```
                    RingBuffer
                       │
                       ▼
              BidirectionalChannel ◄──── duplex_shm_connector.py
                    │       │
          ┌─────────┘       └──────────┐
          ▼                            ▼
  DuplexStageWorker           DuplexSessionManager
          │                            │
          ▼                            ▼
  omni_duplex_scheduler       serving_duplex.py (upgrade)
  (RUNNING_CONTINUOUS)                 │
          │                            ▼
          ▼                     api_server.py (ws route)
  moshi.py (temporal_step)
          │
          ├──→ mimi_encoder.py
          └──→ mimi_streaming.py
```

### 3.6 Latency Budget

Moshi operates at 12.5Hz (80ms per frame). The total pipeline per frame must
fit within this budget:

```
┌─────────────────────────────────────────────────────────────────────┐
│                   80ms Frame Budget                                  │
├─────────┬──────────────┬───────────────┬──────────────┬─────────────┤
│ Mimi    │ Temporal     │ Depth (8-step │ Mimi         │ WebSocket   │
│ Encode  │ Transformer  │ AR loop)      │ Decode       │ Send        │
│ ~3ms    │ ~25ms        │ ~15ms         │ ~3ms         │ ~2ms        │
│         │ (1 step,7B)  │ (6L×8 steps)  │              │             │
├─────────┴──────────────┴───────────────┴──────────────┴─────────────┤
│ Total: ~48ms → ~32ms headroom for scheduling + IPC                  │
└─────────────────────────────────────────────────────────────────────┘

Note: Latency numbers are estimates for A100 GPU. Actual numbers depend on:
- Batch size (target: 1 for real-time)
- Model precision (bf16)
- KV cache size (grows linearly with conversation length)
```

### 3.7 Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| KV cache grows unbounded during long conversations | **Critical** | Implement sliding window attention or periodic KV eviction (Moshi supports 3000 position limit). Reset KV cache every ~4 minutes. |
| Depth decoder latency exceeds frame budget at scale | High | Profile on target hardware. Depth decoder is lightweight (6L, ~300M params) but 8 sequential AR steps are unavoidable. Consider batching depth steps if possible. |
| Race condition between input writer and stage reader | High | RingBuffer uses atomic position counters. Reader only accesses positions < write_pos. Torch tensor operations are thread-safe for reads. |
| WebSocket disconnection during active generation | Medium | DuplexSessionManager catches `WebSocketDisconnect`, sends `INTERRUPT` to channel, worker cleans up KV cache within 1 scheduler cycle. |
| RUNNING_CONTINUOUS leaks GPU memory if session not cleaned | Medium | Add session timeout (configurable, default 10 min). Session manager sends periodic keepalive; worker auto-terminates on timeout. |
| Mimi encoder/decoder streaming mode accuracy | Medium | Mimi is designed for streaming (12.5Hz, causal convolutions). Use HuggingFace MimiModel with `use_cache=True` for streaming decode. |
| Scheduler starvation with many concurrent duplex sessions | Medium | Limit concurrent duplex sessions per GPU. Duplex requests have priority over batch requests. |

---

## Phased Migration Path

```
 Phase 1 (DONE)         Phase 2                    Phase 3
 ─────────────          ──────────                  ──────────
 Turn-taking            Chunked streaming           True full-duplex
 2-stage pipeline       2-stage + async_chunk       Event-driven loop
 Generation worker      AR worker + streaming       Duplex worker
 HTTP response          WebSocket (half-duplex)     WebSocket (full-duplex)
 No client streaming    Server→Client streaming     Bidirectional streaming

 Constraints:           Addresses:                  Addresses:
 All (C1-C7)            C4 (chunking)               C1-C7 (all)
                        + New WebSocket              Bidirectional I/O
                                                     Continuous generation
```

### Migration Safety

- **Phase 2 is backward compatible**: Existing turn-taking mode still works.
  New streaming mode is opt-in via `moshi_streaming.yaml` config.
- **Phase 3 is a separate code path**: `DuplexStageWorker` does not modify
  the existing `_stage_worker()`. New scheduler state `RUNNING_CONTINUOUS`
  is only activated for duplex sessions.
- **Each phase has independent tests**: Unit tests don't require previous
  phases. E2E tests build incrementally.

---

## Estimated Scope

| Phase | New Files | Modified Files | Lines (est.) | Key Deliverable |
|-------|-----------|---------------|-------------|-----------------|
| Phase 2 | 5 | 5 | ~1500 | WebSocket + chunked audio streaming |
| Phase 3 | 8 | 9 | ~3000 | True full-duplex real-time conversation |

### Phase 2 Priority Order

1. **`protocol/duplex.py`** — Foundation, no dependencies
2. **`moshi_streaming.py`** — Chunk processor, depends on existing connector
3. **`moshi_streaming.yaml`** — Config, depends on chunk processor
4. **`moshi.py` refactor** — Per-step emission, critical path
5. **`serving_duplex.py`** — WebSocket handler, depends on protocol
6. **`omni.py` streaming** — Orchestrator, depends on all above
7. **`api_server.py` route** — Wiring, final step

### Phase 3 Priority Order

1. **`RingBuffer`** — No dependencies, independently testable
2. **`BidirectionalChannel`** — Depends on RingBuffer
3. **`mimi_encoder.py` + `mimi_streaming.py`** — Model components, testable independently
4. **`moshi.py` `temporal_step()`** — Per-step API, critical
5. **`omni_duplex_scheduler.py`** — Core scheduler change
6. **`DuplexStageWorker`** — Depends on channel + scheduler
7. **`DuplexSessionManager`** — Depends on worker + channel
8. **`serving_duplex.py` upgrade** — Final wiring

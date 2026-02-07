# Moshi Full-Duplex Audio Streaming

End-to-end guide for running real-time, full-duplex audio inference with
the [Moshi](https://arxiv.org/abs/2410.00037) speech dialogue model via
vLLM-Omni's WebSocket API.

**Full-duplex** means the client can send audio and receive audio at the same
time — the model continuously generates speech while listening to the user,
enabling natural conversational overlap, barge-in, and back-channeling.

## Prerequisites

| Requirement | FP16 (default) | INT4 (quantized) |
|---|---|---|
| OS | Linux | Linux |
| Python | 3.10+ (3.12 recommended) | 3.10+ (3.12 recommended) |
| GPU | 1x H100-80G / A100-80G | 1x RTX 4070 Ti Super (16 GB+) |
| CUDA | 12.1+ | 12.1+ |
| VRAM | ~35 GB | ~8–12 GB |

## 1. Installation

### 1.1 Create a Python environment

```bash
uv venv --python 3.12 --seed
source .venv/bin/activate
```

### 1.2 Install vLLM (backend engine)

```bash
# NVIDIA CUDA
uv pip install vllm==0.15.0 --torch-backend=auto

# AMD ROCm (alternative)
uv pip install vllm==0.15.0 --extra-index-url https://wheels.vllm.ai/rocm/0.15.0/rocm700
```

### 1.3 Install vLLM-Omni

```bash
git clone https://github.com/vllm-project/vllm-omni.git
cd vllm-omni
uv pip install -e .
```

This installs all core dependencies listed in `pyproject.toml`, including:

- `omegaconf` — YAML stage config parsing
- `librosa`, `soundfile` — audio I/O and resampling
- `torch` — model inference (installed via vLLM)

### 1.4 Install dev/test dependencies (optional)

```bash
uv pip install -e ".[dev]"
```

### 1.5 Verify installation

```bash
python -c "from vllm_omni.entrypoints.openai.api_server import omni_run_server; print('OK')"
```

## 2. Model Setup

Moshi uses the `kmhf/hf-moshiko` checkpoint (HuggingFace Transformers format).
The model will be downloaded automatically on first launch, or you can
pre-download it:

```bash
# Pre-download (optional)
huggingface-cli download kmhf/hf-moshiko
```

### Model architecture summary

| Component | Description |
|---|---|
| Temporal Transformer | 7B parameters, 32 layers. Processes 17 input streams (text + 8 Moshi audio + 8 user audio). |
| Depth Transformer | 6 layers. Generates 8 RVQ codebook tokens per frame. |
| Mimi Codec | Audio tokenizer. Encodes/decodes PCM ↔ RVQ codes at 12.5 Hz (80 ms per frame). |
| Frame rate | 12.5 Hz — one forward pass per 80 ms of audio |
| Codebooks | 8 RVQ levels per frame |

### Quantized model (for 16 GB GPUs)

The temporal transformer supports INT4 quantization (AWQ, GPTQ) which
reduces weights from ~14 GB to ~3.5 GB, enabling inference on consumer
GPUs like the RTX 4070 Ti Super (16 GB) or RTX 4090 (24 GB).

To create an AWQ-quantized checkpoint:

```bash
pip install autoawq

python -c "
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model = AutoAWQForCausalLM.from_pretrained('kmhf/hf-moshiko')
model.quantize(
    tokenizer=None,
    quant_config={'w_bit': 4, 'q_group_size': 128, 'zero_point': True},
)
model.save_quantized('kmhf/hf-moshiko-awq')
"
```

Only the temporal transformer (7B, 32 layers) is quantized. The depth
transformer uses 3D per-codebook weights (`MoshiFlexibleLinear`) which
remain in FP16 — its footprint is negligible.

## 3. Stage Configuration

Full-duplex mode uses a dedicated stage config that bypasses the standard
multi-stage pipeline. The config is shipped at
[`stage_configs/moshi_duplex.yaml`](../../../../vllm_omni/model_executor/stage_configs/moshi_duplex.yaml).

Key settings:

```yaml
# Enable full-duplex mode (bypasses standard scheduler)
duplex: true

stage_args:
  - stage_id: 0
    stage_type: llm
    runtime:
      devices: "0"             # GPU index
      max_batch_size: 1        # Full-duplex: one session per GPU
    engine_args:
      model_arch: MoshiForConditionalGenerationVLLM
      gpu_memory_utilization: 0.90
      enforce_eager: true      # No CUDA graphs (step-by-step generation)
      tensor_parallel_size: 1
    default_sampling_params:
      temperature: 0.7
      top_k: 50

duplex_config:
  max_duration_s: 240          # Max conversation: 4 minutes
  ring_buffer_capacity: 256    # ~20 s of user audio history
  output_queue_size: 128       # ~10 s of buffered output
  num_codebooks: 8             # Moshi RVQ codebooks
  max_temporal_steps: 2500     # KV cache reset threshold
  session_timeout_s: 300       # 5-minute idle timeout
```

To customize, copy the file and pass the path to the server:

```bash
cp vllm_omni/model_executor/stage_configs/moshi_duplex.yaml my_config.yaml
# Edit my_config.yaml as needed
```

## 4. Start the Server

```bash
vllm serve kmhf/hf-moshiko \
  --omni \
  --port 8000 \
  --host 0.0.0.0 \
  --stage-configs-path vllm_omni/model_executor/stage_configs/moshi_duplex.yaml \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.90 \
  --trust-remote-code
```

### Start with quantized model (16 GB GPU)

```bash
vllm serve kmhf/hf-moshiko-awq \
  --omni \
  --port 8000 \
  --host 0.0.0.0 \
  --stage-configs-path vllm_omni/model_executor/stage_configs/moshi_duplex_quantized.yaml \
  --quantization awq \
  --gpu-memory-utilization 0.95 \
  --trust-remote-code
```

The quantized stage config (`moshi_duplex_quantized.yaml`) has tighter
limits tuned for 16 GB VRAM: shorter max duration (120 s), smaller ring
buffer (128 frames), and lower KV cache limit (1500 steps).

### CLI flags reference

| Flag | Default | Description |
|---|---|---|
| `--omni` | (required) | Enable vLLM-Omni multi-modal mode |
| `--port` | `8000` | HTTP/WebSocket listen port |
| `--host` | `0.0.0.0` | Listen address |
| `--stage-configs-path` | auto-detect | Path to stage YAML config |
| `--stage-init-timeout` | `300` | Per-stage init timeout (seconds) |
| `--init-timeout` | `600` | Total init timeout (seconds) |
| `--tensor-parallel-size` | `1` | GPU parallelism for the model |
| `--gpu-memory-utilization` | `0.90` | Fraction of GPU memory to use |
| `--trust-remote-code` | `false` | Allow remote code in model weights |
| `--enforce-eager` | `false` | Disable CUDA graphs (recommended for duplex) |
| `--worker-backend` | `multi_process` | `multi_process` or `ray` |

Wait until you see:

```
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Verify the server

```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

## 5. WebSocket Protocol

### 5.1 Endpoint

```
ws://localhost:8000/v1/audio/duplex
```

### 5.2 Connection lifecycle

```
┌─────────┐                              ┌──────────┐
│  Client  │                              │  Server   │
└────┬─────┘                              └─────┬─────┘
     │  ─── WebSocket connect ──────────────▶   │
     │  ◀── 101 Switching Protocols ────────    │
     │                                          │
     │  ─── session.start (JSON) ───────────▶   │
     │  ◀── session.created (JSON) ─────────    │
     │                                          │
     │  ─── audio.input (binary) ───────────▶   │  ┐
     │  ─── audio.input (binary) ───────────▶   │  │ concurrent
     │  ◀── audio.output (binary) ──────────    │  │ bidirectional
     │  ◀── audio.output.meta (JSON) ───────    │  │ streaming
     │  ─── audio.input (binary) ───────────▶   │  │
     │  ◀── audio.output (binary) ──────────    │  │
     │  ◀── audio.output.meta (JSON) ───────    │  ┘
     │                                          │
     │  ─── interrupt (JSON) ───────────────▶   │  (optional)
     │                                          │
     │  ─── session.end (JSON) ─────────────▶   │
     │  ◀── generation.done (JSON) ─────────    │
     │  ◀── WebSocket close ────────────────    │
```

### 5.3 Message reference

#### Client → Server

**`session.start`** — Initialize a session (must be the first message):

```json
{
  "type": "session.start",
  "model": "moshi",
  "sample_rate": 24000,
  "audio_format": "pcm_s16le",
  "output_format": "pcm_s16le",
  "temperature": 0.7,
  "top_k": 25,
  "max_duration_s": 30.0,
  "duplex": true
}
```

| Field | Type | Default | Description |
|---|---|---|---|
| `model` | string | `"moshi"` | Model identifier |
| `sample_rate` | int | `24000` | Audio sample rate (Hz) |
| `audio_format` | string | `"pcm_s16le"` | Input audio format: `pcm_s16le`, `pcm_f32le` |
| `output_format` | string | `"pcm_s16le"` | Output format: `pcm_s16le`, `pcm_f32le`, `opus` |
| `temperature` | float | `0.7` | Sampling temperature (0.0–2.0) |
| `top_k` | int | `25` | Top-k sampling parameter |
| `max_duration_s` | float | `30.0` | Max output duration in seconds (1.0–300.0) |
| `duplex` | bool | `false` | **Set `true` for full-duplex mode** |

**`audio.input`** (binary) — Raw PCM audio chunk. Send continuously as audio
is captured from the microphone.

**`audio.input.done`** — Signal user finished speaking (optional in full-duplex):

```json
{"type": "audio.input.done"}
```

**`interrupt`** — Stop generation and reset KV cache (Phase 3):

```json
{"type": "interrupt"}
```

**`session.end`** — Terminate the session:

```json
{"type": "session.end"}
```

#### Server → Client

**`session.created`** — Session accepted:

```json
{
  "type": "session.created",
  "session_id": "a1b2c3d4-...",
  "model": "moshi",
  "sample_rate": 24000,
  "duplex": true
}
```

**`audio.output`** (binary) — Generated PCM audio chunk. Play immediately.

**`audio.output.meta`** — Metadata for the preceding audio chunk:

```json
{
  "type": "audio.output.meta",
  "chunk_index": 0,
  "duration_ms": 80.0,
  "is_final": false,
  "temporal_step": 42,
  "text_token": 1537
}
```

| Field | Type | Description |
|---|---|---|
| `chunk_index` | int | Sequential chunk counter (0-based) |
| `duration_ms` | float | Audio duration of this chunk in ms |
| `is_final` | bool | Whether this is the last audio chunk |
| `temporal_step` | int \| null | Model temporal position for this frame |
| `text_token` | int \| null | Sampled text token ID (if any) |

**`text.token`** — Intermediate text token:

```json
{
  "type": "text.token",
  "token_id": 1537,
  "text": "hello",
  "step": 42
}
```

**`generation.done`** — Generation complete:

```json
{
  "type": "generation.done",
  "session_id": "a1b2c3d4-...",
  "total_chunks": 375,
  "total_duration_ms": 30000.0
}
```

**`error`** — Error message:

```json
{
  "type": "error",
  "message": "Pipeline timeout: no chunk received",
  "code": "internal_error",
  "recoverable": false,
  "details": null
}
```

## 6. Client Examples

### 6.1 Python WebSocket client (full-duplex)

A complete example using the `websockets` library:

```bash
pip install websockets sounddevice numpy
```

```python
"""Full-duplex Moshi client — send mic audio, play received audio."""

import asyncio
import json
import struct

import numpy as np
import sounddevice as sd
import websockets

SERVER_URL = "ws://localhost:8000/v1/audio/duplex"
SAMPLE_RATE = 24000
CHUNK_DURATION_S = 0.08  # 80ms to match Moshi frame rate
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION_S)


async def main():
    async with websockets.connect(SERVER_URL) as ws:
        # 1. Start session in full-duplex mode
        await ws.send(json.dumps({
            "type": "session.start",
            "model": "moshi",
            "sample_rate": SAMPLE_RATE,
            "audio_format": "pcm_s16le",
            "output_format": "pcm_s16le",
            "temperature": 0.7,
            "top_k": 25,
            "max_duration_s": 60.0,
            "duplex": True,
        }))

        # 2. Wait for session.created
        response = json.loads(await ws.recv())
        assert response["type"] == "session.created", response
        print(f"Session created: {response['session_id']} (duplex={response['duplex']})")

        # 3. Run input and output concurrently
        stop_event = asyncio.Event()

        async def send_audio():
            """Capture mic audio and send to server."""
            loop = asyncio.get_event_loop()

            def callback(indata, frames, time_info, status):
                """sounddevice callback — queue PCM bytes."""
                pcm = (indata[:, 0] * 32767).astype(np.int16).tobytes()
                asyncio.run_coroutine_threadsafe(ws.send(pcm), loop)

            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=CHUNK_SAMPLES,
                callback=callback,
            ):
                await stop_event.wait()

            # Signal end of input
            await ws.send(json.dumps({"type": "session.end"}))

        async def recv_audio():
            """Receive server audio and play it."""
            stream = sd.OutputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="int16",
            )
            stream.start()

            try:
                async for message in ws:
                    if isinstance(message, bytes):
                        # Binary frame = audio output
                        pcm = np.frombuffer(message, dtype=np.int16)
                        stream.write(pcm.reshape(-1, 1))
                    else:
                        data = json.loads(message)
                        if data["type"] == "generation.done":
                            print(f"Done: {data['total_chunks']} chunks, "
                                  f"{data['total_duration_ms']:.0f} ms")
                            stop_event.set()
                            break
                        elif data["type"] == "audio.output.meta":
                            pass  # Metadata for the preceding audio chunk
                        elif data["type"] == "text.token":
                            print(data.get("text", ""), end="", flush=True)
                        elif data["type"] == "error":
                            print(f"Error: {data['message']}")
                            stop_event.set()
                            break
            finally:
                stream.stop()
                stream.close()

        # Run both pipelines concurrently
        await asyncio.gather(send_audio(), recv_audio())


if __name__ == "__main__":
    asyncio.run(main())
```

### 6.2 Minimal half-duplex client

For simpler use cases (speak first, then listen):

```python
"""Half-duplex Moshi client — send all audio, then receive response."""

import asyncio
import json

import websockets

SERVER_URL = "ws://localhost:8000/v1/audio/duplex"


async def main():
    # Read a PCM file (24kHz, 16-bit, mono)
    with open("input.pcm", "rb") as f:
        audio_bytes = f.read()

    async with websockets.connect(SERVER_URL) as ws:
        # 1. Start session (half-duplex: duplex=False)
        await ws.send(json.dumps({
            "type": "session.start",
            "model": "moshi",
            "sample_rate": 24000,
            "duplex": False,
        }))
        response = json.loads(await ws.recv())
        print(f"Session: {response['session_id']}")

        # 2. Send all audio in chunks
        chunk_size = 24000 * 2  # 1 second of pcm_s16le
        for i in range(0, len(audio_bytes), chunk_size):
            await ws.send(audio_bytes[i:i + chunk_size])

        # 3. Signal input complete
        await ws.send(json.dumps({"type": "audio.input.done"}))

        # 4. Receive generated audio
        output_chunks = []
        async for message in ws:
            if isinstance(message, bytes):
                output_chunks.append(message)
            else:
                data = json.loads(message)
                if data["type"] == "generation.done":
                    print(f"Received {data['total_chunks']} chunks")
                    break

        # 5. Write output audio
        with open("output.pcm", "wb") as f:
            for chunk in output_chunks:
                f.write(chunk)
        print(f"Saved {len(output_chunks)} chunks to output.pcm")


if __name__ == "__main__":
    asyncio.run(main())
```

### 6.3 Testing with `websocat`

Quick test from the command line:

```bash
# Install websocat
# https://github.com/vi/websocat

# Connect and send session.start
echo '{"type":"session.start","model":"moshi","sample_rate":24000,"duplex":true}' \
  | websocat ws://localhost:8000/v1/audio/duplex
```

## 7. Architecture Overview

Full-duplex inference uses three concurrent components, coordinated by
the `DuplexSessionManager`:

```
                     WebSocket (/v1/audio/duplex)
                            │          ▲
                     binary │          │ binary + JSON
                    (PCM in)│          │(PCM out + meta)
                            ▼          │
                ┌───────────────────────────────────┐
                │      DuplexSessionManager          │
                │                                    │
                │  ┌─────────────┐ ┌──────────────┐  │
                │  │   Input     │ │   Output     │  │
                │  │  Pipeline   │ │  Pipeline    │  │
                │  │  (async)    │ │  (async)     │  │
                │  └──────┬──────┘ └──────▲───────┘  │
                │         │               │          │
                └─────────┼───────────────┼──────────┘
                          │               │
              put_input() │               │ get_output()
              (ring buf)  │               │ (async queue)
                          ▼               │
                ┌───────────────────────────────────┐
                │      BidirectionalChannel          │
                │                                    │
                │  backward_buffer    forward_queue   │
                │  (RingBuffer)       (asyncio.Queue) │
                │                                    │
                │         control_queue               │
                │  (INTERRUPT / PAUSE / RESUME / SHUTDOWN) │
                └──────────┬──────────────▲──────────┘
                           │              │
               get_input() │              │ put_output_sync()
               (ring buf)  │              │ (call_soon_threadsafe)
                           ▼              │
                ┌───────────────────────────────────┐
                │      DuplexStageWorker             │
                │      (dedicated thread)            │
                │                                    │
                │  for each temporal step (80 ms):   │
                │    1. poll control signals          │
                │    2. read user audio from buffer   │
                │    3. model.forward_dialogue_step() │
                │    4. emit OutputFrame              │
                └───────────────────────────────────┘
                           │
                           ▼
                ┌───────────────────────────────────┐
                │  MoshiForConditionalGenerationVLLM │
                │                                    │
                │  Temporal Transformer (7B, 32L)    │
                │  ──▶ Depth Transformer (6L)        │
                │  ──▶ 8 RVQ audio codes + text tok  │
                └───────────────────────────────────┘
```

### Data flow

1. **Input pipeline** receives binary PCM frames from WebSocket, converts
   them to audio codes (via Mimi encoder), and writes to the RingBuffer.
2. **DuplexStageWorker** runs in a dedicated thread. Each iteration reads
   user audio codes at the current temporal position from the RingBuffer,
   runs `forward_dialogue_step()` on the model, and pushes the resulting
   `OutputFrame` (8 audio codes + text token) to the forward queue.
3. **Output pipeline** reads `OutputFrame`s from the forward queue, converts
   audio codes back to PCM (via Mimi decoder), and sends binary + JSON
   metadata to the client.

### Key classes

| Class | File | Role |
|---|---|---|
| `MoshiDuplexHandler` | `entrypoints/openai/serving_duplex.py` | WebSocket handler, routes to half/full-duplex |
| `DuplexSessionManager` | `entrypoints/duplex_session.py` | Coordinates 3 concurrent tasks via `asyncio.TaskGroup` |
| `DuplexStageWorker` | `entrypoints/duplex_stage_worker.py` | Event-driven generation loop (runs in thread) |
| `BidirectionalChannel` | `distributed/omni_connectors/bidirectional.py` | Forward queue + backward ring buffer + control queue |
| `RingBuffer` | `distributed/omni_connectors/bidirectional.py` | Thread-safe circular buffer for user audio codes |
| `MoshiForConditionalGenerationVLLM` | `model_executor/models/moshi/moshi.py` | Temporal + Depth transformer with per-request state |

## 8. Configuration Tuning

### Sampling parameters

| Parameter | Default | Effect |
|---|---|---|
| `temperature` | `0.7` | Higher → more varied speech; lower → more monotone |
| `top_k` | `25` (`50` in stage config) | Limits vocabulary per sampling step |

### Duplex config knobs

| Parameter | Default | When to change |
|---|---|---|
| `ring_buffer_capacity` | `256` (20 s) | Increase for longer conversations with heavy overlap |
| `output_queue_size` | `128` (10 s) | Increase if client consumes slowly |
| `max_temporal_steps` | `2500` (200 s) | Decrease to lower VRAM; increase if model supports more |
| `max_duration_s` | `240` (4 min) | Set per-conversation limit |
| `session_timeout_s` | `300` (5 min) | Idle timeout before server closes the session |

### GPU memory

The dialogue model requires ~35 GB VRAM. Key levers:

- `gpu_memory_utilization: 0.90` — fraction of VRAM allocated for the model
- `enforce_eager: true` — required for step-by-step generation (no CUDA graphs)
- `tensor_parallel_size: 1` — increase for multi-GPU setups (splits the model)

## 9. Running Tests

```bash
# From repo root, run Moshi-specific tests
cd tests/model_executor/models

# Create a minimal pytest.ini to prevent parent conftest from loading
echo "[pytest]" > pytest.ini

python -m pytest test_moshi.py -v

# Clean up
rm pytest.ini
```

The test suite covers:

| Test area | Count | Description |
|---|---|---|
| Model streaming | 8 | `forward_dialogue_step`, state init/clear, concurrent requests |
| Ring buffer | 9 | Write/read, eviction, thread safety, reset |
| Bidirectional channel | 4 | Forward/backward paths, control signals, close |
| Output frame / signals | 3 | Data class defaults and enum values |
| Duplex stage worker | 3 | Run loop, shutdown handling, silence fallback |
| Protocol messages | 7 | All message types, duplex flag, new Phase 3 fields |
| Serving (half-duplex) | ~60 | PCM duration, session flow, chunk processing |
| **Total** | **101** | |

## 10. Limitations and Roadmap

### Current limitations

- **Mimi codec is a placeholder.** The `_process_audio_input()` and
  `_codes_to_pcm()` methods currently produce silence. Production use
  requires integrating the actual Mimi encoder/decoder.
- **Single session per GPU.** `max_batch_size: 1` in the stage config.
  Concurrent sessions require separate GPU allocations.
- **No Opus codec support.** Output format `opus` is accepted but duration
  cannot be inferred from byte length.

### Roadmap

- **Mimi encoder/decoder integration** — Replace placeholder with real
  audio tokenizer for production-quality audio.
- **CUDA stream overlap** — Pipeline Mimi encode/decode with transformer
  forward pass for lower latency.
- **Opus encoding** — Server-side Opus encoding for bandwidth-efficient
  streaming.
- **Latency profiling** — End-to-end latency measurement and optimization.
- **Backpressure tuning** — Adaptive PAUSE/RESUME thresholds based on
  network conditions.
- **Multi-session support** — KV cache sharing or model batching for
  multiple concurrent duplex sessions.

## 11. Troubleshooting

### Server won't start

```
ModuleNotFoundError: No module named 'vllm'
```

Install vLLM first: `uv pip install vllm==0.15.0 --torch-backend=auto`

### WebSocket connection refused

Verify the server is running and the port is correct:

```bash
curl http://localhost:8000/health
```

### "duplex=True requested but no direct model reference"

The server falls back to half-duplex if `app.state.duplex_model` is not set.
This means the stage config was not loaded in duplex mode. Ensure
`--stage-configs-path` points to `moshi_duplex.yaml` (which has `duplex: true`).

### Out of memory (OOM)

- Reduce `gpu_memory_utilization` (e.g., `0.80`)
- Reduce `max_temporal_steps` (shorter conversations = smaller KV cache)
- Use `tensor_parallel_size: 2` with 2 GPUs

### High latency

- Ensure `enforce_eager: true` is set (CUDA graphs add overhead for
  step-by-step generation)
- Check that no other processes are using the GPU
- Monitor the output queue size — if it fills up, the client is consuming
  too slowly

### Tests fail with `ModuleNotFoundError: No module named 'vllm'`

The parent `tests/conftest.py` imports `vllm`. Run tests from inside
`tests/model_executor/models/` with a local `pytest.ini` as shown in
[Section 9](#9-running-tests).

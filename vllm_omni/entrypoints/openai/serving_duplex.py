# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
WebSocket endpoint for real-time Moshi audio streaming (Phase 2: half-duplex).

Protocol:
  1. Client connects via WebSocket to /v1/audio/duplex
  2. Client sends session.start (JSON) with audio config
  3. Server sends session.created (JSON)
  4. Client sends audio.input (binary frames) + audio.input.done
  5. Server streams audio.output (binary) + audio.output.meta (JSON) chunks
  6. Server sends generation.done
"""
import asyncio
import json

from vllm.logger import init_logger
from vllm.utils import random_uuid

from vllm_omni.entrypoints.openai.protocol.duplex import (
    AudioOutputMeta,
    DuplexMessageType,
    ErrorMessage,
    GenerationDoneMessage,
    SessionCreatedMessage,
    SessionStartMessage,
)
from vllm_omni.outputs import OmniStreamingChunk

logger = init_logger(__name__)

# Timeout for waiting on pipeline chunks (seconds).
# Prevents indefinite blocking if pipeline hangs.
_CHUNK_RECV_TIMEOUT = 60.0

# Maximum number of chunks that can be buffered in the async queue
# before the pipeline thread blocks (backpressure).
_MAX_CHUNK_QUEUE_SIZE = 64


class MoshiDuplexHandler:
    """Manages a single WebSocket session for Moshi audio streaming.

    Supports two modes:
      - **Half-duplex** (Phase 2, duplex=False): Client sends all audio
        first, then server streams audio output back.
      - **Full-duplex** (Phase 3, duplex=True): Client and server send
        audio concurrently via DuplexSessionManager.
    """

    def __init__(self, engine_client, model=None):
        self.engine_client = engine_client
        self.model = model  # Direct model ref for full-duplex mode

    async def handle(self, websocket):
        """Main entry point for a WebSocket connection."""
        await websocket.accept()
        session_id = random_uuid()
        logger.info("Duplex session %s: connected", session_id)

        try:
            # 1. Receive and validate session.start
            config = await self._recv_session_start(websocket, session_id)
            if config is None:
                return

            # 2. Route to appropriate handler based on duplex mode
            if config.duplex and self.model is not None:
                await self._handle_full_duplex(
                    websocket, session_id, config,
                )
            else:
                if config.duplex and self.model is None:
                    logger.warning(
                        "Duplex session %s: duplex=True requested but no "
                        "direct model reference; falling back to half-duplex",
                        session_id,
                    )
                await self._handle_half_duplex(
                    websocket, session_id, config,
                )

        except Exception as e:
            logger.exception("Duplex session %s: error", session_id)
            try:
                await websocket.send_json(
                    ErrorMessage(
                        message=str(e),
                        code="internal_error",
                    ).model_dump()
                )
            except Exception:
                pass
        finally:
            logger.info("Duplex session %s: closed", session_id)

    async def _handle_full_duplex(self, websocket, session_id, config):
        """Full-duplex mode: concurrent input/output via DuplexSessionManager."""
        from vllm_omni.entrypoints.duplex_session import DuplexSessionManager

        # Send session.created with duplex=True
        await websocket.send_json(
            SessionCreatedMessage(
                session_id=session_id,
                model=config.model,
                sample_rate=config.sample_rate,
                duplex=True,
            ).model_dump()
        )
        logger.info(
            "Duplex session %s: full-duplex mode (model=%s, sr=%d, temp=%.2f)",
            session_id, config.model, config.sample_rate, config.temperature,
        )

        session = DuplexSessionManager(
            websocket=websocket,
            model=self.model,
            session_id=session_id,
            config=config,
        )
        await session.run()

    async def _handle_half_duplex(self, websocket, session_id, config):
        """Half-duplex mode: collect input, then stream output."""
        # Send session.created
        await websocket.send_json(
            SessionCreatedMessage(
                session_id=session_id,
                model=config.model,
                sample_rate=config.sample_rate,
                duplex=False,
            ).model_dump()
        )
        logger.info(
            "Duplex session %s: half-duplex mode (model=%s, sr=%d, temp=%.2f)",
            session_id, config.model, config.sample_rate, config.temperature,
        )

        # Collect audio input until audio.input.done
        audio_buffer = await self._collect_audio_input(websocket, session_id)
        if audio_buffer is None:
            return

        logger.info(
            "Duplex session %s: received %d bytes of audio input",
            session_id, len(audio_buffer),
        )

        # Run Moshi pipeline, streaming output chunks
        await self._stream_generation(
            websocket, session_id, audio_buffer, config,
        )

    async def _recv_session_start(self, websocket, session_id):
        """Wait for the session.start message from the client."""
        try:
            raw = await asyncio.wait_for(websocket.receive_text(), timeout=10.0)
            data = json.loads(raw)
        except asyncio.TimeoutError:
            await websocket.send_json(
                ErrorMessage(
                    message="Timeout waiting for session.start",
                    code="timeout",
                ).model_dump()
            )
            await websocket.close(code=1008, reason="Timeout")
            return None
        except Exception as e:
            logger.warning("Duplex session %s: invalid session.start: %s", session_id, e)
            await websocket.close(code=1003, reason="Invalid message")
            return None

        msg_type = data.get("type")
        if msg_type != DuplexMessageType.SESSION_START:
            await websocket.send_json(
                ErrorMessage(
                    message=f"Expected session.start, got {msg_type}",
                    code="protocol_error",
                ).model_dump()
            )
            await websocket.close(code=1002, reason="Protocol error")
            return None

        try:
            return SessionStartMessage(**data)
        except Exception as e:
            await websocket.send_json(
                ErrorMessage(
                    message=f"Invalid session.start parameters: {e}",
                    code="invalid_params",
                ).model_dump()
            )
            await websocket.close(code=1002, reason="Invalid parameters")
            return None

    async def _collect_audio_input(self, websocket, session_id) -> bytes | None:
        """Collect binary audio frames until audio.input.done is received."""
        chunks = []
        max_input_bytes = 50 * 1024 * 1024  # 50 MB limit
        total_bytes = 0

        while True:
            message = await websocket.receive()
            ws_type = message.get("type", "")

            if ws_type == "websocket.disconnect":
                logger.info("Duplex session %s: client disconnected during input", session_id)
                return None

            if "bytes" in message and message["bytes"]:
                # Binary frame = audio data
                chunk = message["bytes"]
                total_bytes += len(chunk)
                if total_bytes > max_input_bytes:
                    await websocket.send_json(
                        ErrorMessage(
                            message="Audio input exceeds maximum size",
                            code="input_too_large",
                        ).model_dump()
                    )
                    await websocket.close(code=1009, reason="Input too large")
                    return None
                chunks.append(chunk)

            elif "text" in message and message["text"]:
                # JSON control message
                try:
                    data = json.loads(message["text"])
                except json.JSONDecodeError:
                    continue

                msg_type = data.get("type")
                if msg_type == DuplexMessageType.AUDIO_INPUT_DONE:
                    break
                elif msg_type == DuplexMessageType.SESSION_END:
                    logger.info("Duplex session %s: client ended session during input", session_id)
                    return None

        return b"".join(chunks)

    async def _stream_generation(
        self,
        websocket,
        session_id: str,
        audio_bytes: bytes,
        config: SessionStartMessage,
    ):
        """Run Moshi inference and stream audio chunks back via WebSocket.

        Converts raw audio bytes to tokens, submits to the pipeline,
        and streams generated audio chunks as they become available.
        """
        # Convert raw PCM bytes to audio token IDs for the Moshi pipeline
        audio_token_ids = self._pcm_to_token_ids(
            audio_bytes, config.audio_format, config.sample_rate
        )

        # Submit to the streaming pipeline
        chunk_index = 0
        total_duration_ms = 0.0

        async for chunk in self._generate_streaming(
            session_id=session_id,
            audio_token_ids=audio_token_ids,
            config=config,
        ):
            if chunk.audio_data is not None:
                # Send binary audio frame
                await websocket.send_bytes(chunk.audio_data)

                # Compute duration based on output audio format
                duration_ms = _compute_pcm_duration_ms(
                    chunk.audio_data, config.sample_rate, config.output_format,
                )
                total_duration_ms += duration_ms

                await websocket.send_json(
                    AudioOutputMeta(
                        chunk_index=chunk_index,
                        duration_ms=duration_ms,
                        is_final=chunk.is_final,
                    ).model_dump()
                )
                chunk_index += 1

        # Send generation.done
        await websocket.send_json(
            GenerationDoneMessage(
                session_id=session_id,
                total_chunks=chunk_index,
                total_duration_ms=total_duration_ms,
            ).model_dump()
        )
        logger.info(
            "Duplex session %s: generation complete (%d chunks, %.1f ms)",
            session_id, chunk_index, total_duration_ms,
        )

    async def _generate_streaming(
        self,
        session_id: str,
        audio_token_ids: list[int],
        config: SessionStartMessage,
    ):
        """Yield OmniStreamingChunk objects from the engine pipeline.

        This wraps the synchronous pipeline orchestrator in an async generator,
        running the blocking generation in a thread executor.
        """
        import concurrent.futures

        loop = asyncio.get_event_loop()

        # Bounded queue for cross-thread chunk delivery (backpressure)
        chunk_queue: asyncio.Queue = asyncio.Queue(
            maxsize=_MAX_CHUNK_QUEUE_SIZE,
        )

        # Event to signal cancellation to the pipeline thread
        cancel_event = asyncio.Event()

        def _run_pipeline():
            """Run in thread: execute pipeline, push chunks to queue."""
            try:
                for chunk in self.engine_client.generate_streaming(
                    request_id=session_id,
                    audio_token_ids=audio_token_ids,
                    temperature=config.temperature,
                    top_k=config.top_k,
                    max_duration_s=config.max_duration_s,
                ):
                    if cancel_event.is_set():
                        logger.info(
                            "Duplex session %s: pipeline cancelled", session_id,
                        )
                        break
                    loop.call_soon_threadsafe(chunk_queue.put_nowait, chunk)
                # Signal completion
                loop.call_soon_threadsafe(chunk_queue.put_nowait, None)
            except Exception as e:
                loop.call_soon_threadsafe(
                    chunk_queue.put_nowait,
                    OmniStreamingChunk(
                        stage_id=-1,
                        is_final=True,
                        error=str(e),
                    ),
                )

        # Start pipeline in background thread
        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix=f"moshi-pipeline-{session_id[:8]}",
        )
        future = executor.submit(_run_pipeline)

        try:
            while True:
                try:
                    chunk = await asyncio.wait_for(
                        chunk_queue.get(), timeout=_CHUNK_RECV_TIMEOUT,
                    )
                except asyncio.TimeoutError:
                    logger.error(
                        "Duplex session %s: timeout waiting for pipeline chunk",
                        session_id,
                    )
                    raise RuntimeError("Pipeline timeout: no chunk received")
                if chunk is None:
                    break
                if chunk.error:
                    raise RuntimeError(f"Pipeline error: {chunk.error}")
                yield chunk
        finally:
            # Signal cancellation and wait for the thread to finish
            cancel_event.set()
            executor.shutdown(wait=True, cancel_futures=True)

    @staticmethod
    def _pcm_to_token_ids(
        audio_bytes: bytes,
        audio_format: str,
        sample_rate: int,
    ) -> list[int]:
        """Convert raw PCM audio bytes to mock token IDs.

        In the full implementation, this would run the Mimi encoder
        to convert audio waveform -> RVQ codes -> flattened token IDs.
        For Phase 2, we compute the frame count from input duration
        and return placeholder token IDs (silence).

        TODO(Phase 3): Replace with actual Mimi encoder integration.
        """
        if audio_format == "pcm_s16le":
            # 16-bit signed LE: 2 bytes per sample
            num_samples = len(audio_bytes) // 2
        elif audio_format == "pcm_f32le":
            # 32-bit float LE: 4 bytes per sample
            num_samples = len(audio_bytes) // 4
        else:
            num_samples = len(audio_bytes) // 2

        # Calculate number of Moshi frames (12.5 Hz = 80ms per frame)
        frames_per_second = 12.5
        duration_s = num_samples / sample_rate
        num_frames = int(duration_s * frames_per_second)
        num_codebooks = 8

        # Return placeholder token IDs (zeros = silence)
        # In production, Mimi encoder would fill these with real RVQ codes
        return [0] * (num_frames * num_codebooks)


def _compute_pcm_duration_ms(
    audio_data: bytes,
    sample_rate: int,
    output_format: str,
) -> float:
    """Compute duration in milliseconds for an audio data chunk.

    For uncompressed PCM formats, duration can be computed exactly from
    byte length. For compressed formats (opus), duration cannot be
    determined from byte length alone, so we return 0.0.

    Args:
        audio_data: Raw audio bytes.
        sample_rate: Sample rate in Hz.
        output_format: Audio format identifier.

    Returns:
        Duration in milliseconds, or 0.0 if format is compressed.
    """
    if not audio_data:
        return 0.0

    if output_format == "pcm_s16le":
        bytes_per_sample = 2
    elif output_format == "pcm_f32le":
        bytes_per_sample = 4
    else:
        # Compressed formats (opus, etc.): byte length != duration
        return 0.0

    num_samples = len(audio_data) / bytes_per_sample
    return (num_samples / sample_rate) * 1000

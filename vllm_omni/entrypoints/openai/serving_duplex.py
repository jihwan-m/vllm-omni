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
import struct

from vllm.logger import init_logger
from vllm.utils import random_uuid

from vllm_omni.entrypoints.openai.protocol.duplex import (
    AudioInputDoneMessage,
    AudioOutputMeta,
    DuplexMessageType,
    ErrorMessage,
    GenerationDoneMessage,
    SessionCreatedMessage,
    SessionStartMessage,
)
from vllm_omni.outputs import OmniStreamingChunk

logger = init_logger(__name__)


class MoshiDuplexHandler:
    """Manages a single WebSocket session for Moshi audio streaming.

    This handler implements half-duplex (turn-taking) audio streaming:
    the client sends all audio input first, then the server streams
    audio output back in chunks.
    """

    def __init__(self, engine_client):
        self.engine_client = engine_client

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

            # 2. Send session.created
            await websocket.send_json(
                SessionCreatedMessage(
                    session_id=session_id,
                    model=config.model,
                    sample_rate=config.sample_rate,
                ).model_dump()
            )
            logger.info(
                "Duplex session %s: created (model=%s, sr=%d, temp=%.2f)",
                session_id, config.model, config.sample_rate, config.temperature,
            )

            # 3. Collect audio input until audio.input.done
            audio_buffer = await self._collect_audio_input(websocket, session_id)
            if audio_buffer is None:
                return

            logger.info(
                "Duplex session %s: received %d bytes of audio input",
                session_id, len(audio_buffer),
            )

            # 4. Run Moshi pipeline, streaming output chunks
            await self._stream_generation(
                websocket, session_id, audio_buffer, config,
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

                # Send metadata for the audio frame
                duration_ms = len(chunk.audio_data) / (
                    config.sample_rate * 2  # 16-bit = 2 bytes per sample
                ) * 1000 if chunk.audio_data else 0
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

        # Create a queue for cross-thread chunk delivery
        chunk_queue = asyncio.Queue()

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
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        executor.submit(_run_pipeline)

        try:
            while True:
                chunk = await chunk_queue.get()
                if chunk is None:
                    break
                if chunk.error:
                    raise RuntimeError(f"Pipeline error: {chunk.error}")
                yield chunk
        finally:
            executor.shutdown(wait=False)

    @staticmethod
    def _pcm_to_token_ids(
        audio_bytes: bytes,
        audio_format: str,
        sample_rate: int,
    ) -> list[int]:
        """Convert raw PCM audio bytes to mock token IDs.

        In the full implementation, this would run the Mimi encoder
        to convert audio waveform → RVQ codes → flattened token IDs.
        For Phase 2, we pass the raw audio length as a placeholder
        and let the pipeline handle encoding.
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

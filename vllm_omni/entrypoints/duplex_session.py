# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Full-duplex session manager for Moshi (Phase 3).

Coordinates three concurrent tasks for a single WebSocket session:

  1. **Input pipeline** (async):  WebSocket binary frames → PCM decode →
     (placeholder Mimi encode) → ring buffer writes.

  2. **Output pipeline** (async):  Forward queue reads → (placeholder
     Mimi decode) → PCM encode → WebSocket binary frames + JSON metadata.

  3. **Dialogue worker** (thread):  Continuous temporal+depth generation
     loop driven by DuplexStageWorker. Reads from ring buffer, writes
     to forward queue.

All three run concurrently via asyncio.TaskGroup, with the worker thread
bridged via asyncio.to_thread().
"""
from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

from vllm.logger import init_logger

from vllm_omni.distributed.omni_connectors.bidirectional import (
    BidirectionalChannel,
    ControlSignal,
)
from vllm_omni.entrypoints.duplex_stage_worker import DuplexStageWorker

if TYPE_CHECKING:
    from vllm_omni.entrypoints.openai.protocol.duplex import SessionStartMessage

logger = init_logger(__name__)

# Moshi frame rate: 12.5 Hz → 80ms per frame
_FRAME_RATE_HZ = 12.5
_FRAME_DURATION_MS = 80.0

# Number of codebooks per Moshi audio frame
_NUM_CODEBOOKS = 8

# Session timeout: close session if no output for this long
_SESSION_TIMEOUT_S = 300.0  # 5 minutes


class DuplexSessionManager:
    """Manages a persistent full-duplex Moshi session.

    Created per WebSocket connection. Coordinates concurrent input
    (user audio → model) and output (model → user audio) pipelines
    with the dialogue generation worker.

    Args:
        websocket: The WebSocket connection (FastAPI/Starlette).
        model: MoshiForConditionalGenerationVLLM instance.
        session_id: Unique session identifier.
        config: Session configuration from client's session.start message.
    """

    def __init__(
        self,
        websocket,
        model,
        session_id: str,
        config: SessionStartMessage,
    ):
        self.websocket = websocket
        self.model = model
        self.session_id = session_id
        self.config = config

        # Create bidirectional channel
        max_steps = int(config.max_duration_s * _FRAME_RATE_HZ)
        self.channel = BidirectionalChannel(
            session_id=session_id,
            buffer_capacity=max(max_steps, 256),
            output_queue_size=128,
            num_codebooks=_NUM_CODEBOOKS,
        )

        # Create worker (runs in thread)
        self.worker = DuplexStageWorker(
            model=model,
            channel=self.channel,
            request_id=session_id,
            temperature=config.temperature,
            top_k=config.top_k,
            max_steps=max_steps,
        )

        # Tracking
        self._input_frames = 0
        self._output_frames = 0
        self._user_done = False

    async def run(self):
        """Launch all three concurrent tasks.

        Returns when:
          - Client disconnects
          - Worker finishes (max steps or error)
          - Session timeout
        """
        loop = asyncio.get_event_loop()
        self.channel.bind_loop(loop)

        logger.info(
            "DuplexSession %s: starting (model=%s, max_steps=%d)",
            self.session_id, self.config.model, self.worker.max_steps,
        )

        try:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(self._input_pipeline())
                tg.create_task(self._output_pipeline())
                tg.create_task(asyncio.to_thread(self.worker.run))
        except* Exception as eg:
            # TaskGroup raises ExceptionGroup; log all
            for exc in eg.exceptions:
                if not isinstance(exc, asyncio.CancelledError):
                    logger.exception(
                        "DuplexSession %s: task error: %s",
                        self.session_id, exc,
                    )
        finally:
            self.channel.close()
            logger.info(
                "DuplexSession %s: ended (in=%d frames, out=%d frames)",
                self.session_id, self._input_frames, self._output_frames,
            )

    # ----------------------------------------------------------------
    # Input pipeline: WebSocket → ring buffer
    # ----------------------------------------------------------------

    async def _input_pipeline(self):
        """Receive user audio from WebSocket and write to ring buffer.

        Accepts:
          - Binary frames: raw PCM audio data
          - JSON text frames: control messages (audio.input.done, interrupt,
            session.end)
        """
        try:
            while not self.channel.closed:
                message = await self.websocket.receive()
                ws_type = message.get("type", "")

                if ws_type == "websocket.disconnect":
                    logger.info(
                        "DuplexSession %s: client disconnected",
                        self.session_id,
                    )
                    break

                if "bytes" in message and message["bytes"]:
                    self._process_audio_input(message["bytes"])

                elif "text" in message and message["text"]:
                    action = self._process_control_input(message["text"])
                    if action == "stop":
                        break

        except Exception as e:
            logger.warning(
                "DuplexSession %s: input pipeline error: %s",
                self.session_id, e,
            )
        finally:
            self._user_done = True
            # Signal worker to stop after processing remaining input
            if not self.channel.closed:
                await self.channel.send_control(ControlSignal.SHUTDOWN)

    def _process_audio_input(self, audio_bytes: bytes):
        """Convert raw PCM bytes to Moshi audio codes and write to buffer.

        In the full implementation, this would:
          1. Decode PCM → float32 waveform
          2. Run Mimi encoder → RVQ codes [8]
          3. Write codes to ring buffer

        For Phase 3 initial implementation, we compute frame count from
        PCM length and write placeholder codes (zeros = silence).
        TODO(Production): Integrate actual Mimi encoder.
        """
        # PCM s16le: 2 bytes per sample, mono
        bytes_per_sample = 2
        sample_rate = self.config.sample_rate
        num_samples = len(audio_bytes) // bytes_per_sample

        # How many 80ms frames in this chunk?
        duration_s = num_samples / sample_rate
        num_frames = int(duration_s * _FRAME_RATE_HZ)

        for _ in range(max(num_frames, 1)):
            # Placeholder: write silence codes
            # TODO: Replace with actual Mimi encoder output
            self.channel.put_input([0] * _NUM_CODEBOOKS)
            self._input_frames += 1

    def _process_control_input(self, text: str) -> str | None:
        """Process a JSON control message from the client.

        Returns "stop" if the session should end.
        """
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return None

        msg_type = data.get("type", "")

        if msg_type == "audio.input.done":
            logger.info(
                "DuplexSession %s: user finished input (%d frames)",
                self.session_id, self._input_frames,
            )
            self._user_done = True
            return None  # Worker continues generating until done

        if msg_type == "session.end":
            logger.info(
                "DuplexSession %s: client ended session",
                self.session_id,
            )
            return "stop"

        if msg_type == "interrupt":
            logger.info(
                "DuplexSession %s: client interrupted",
                self.session_id,
            )
            # Non-blocking: signal will be picked up by worker
            asyncio.get_event_loop().call_soon(
                self.channel.control_queue.put_nowait,
                ControlSignal.INTERRUPT,
            )
            return None

        return None

    # ----------------------------------------------------------------
    # Output pipeline: forward queue → WebSocket
    # ----------------------------------------------------------------

    async def _output_pipeline(self):
        """Read generated output frames and stream to client via WebSocket.

        Sends:
          - Binary frame: raw PCM audio bytes (from Mimi decoder)
          - JSON frame: metadata (chunk_index, text_token, temporal_step)
        """
        chunk_index = 0
        total_duration_ms = 0.0

        try:
            while not self.channel.closed:
                frame = await self.channel.get_output(
                    timeout=_SESSION_TIMEOUT_S,
                )
                if frame is None:
                    # Worker finished or timeout
                    break

                # Convert audio codes to PCM bytes via Mimi decoder
                # TODO(Production): Replace with actual Mimi decoder
                pcm_data = self._codes_to_pcm(frame.audio_codes)

                if pcm_data:
                    # Send binary audio frame
                    await self.websocket.send_bytes(pcm_data)

                # Send JSON metadata
                meta = {
                    "type": "audio.output.meta",
                    "chunk_index": chunk_index,
                    "temporal_step": frame.temporal_step,
                    "duration_ms": _FRAME_DURATION_MS,
                    "is_final": False,
                }
                if frame.text_token is not None:
                    meta["text_token"] = frame.text_token

                await self.websocket.send_json(meta)

                chunk_index += 1
                total_duration_ms += _FRAME_DURATION_MS
                self._output_frames += 1

        except Exception as e:
            logger.warning(
                "DuplexSession %s: output pipeline error: %s",
                self.session_id, e,
            )
        finally:
            # Send generation.done
            try:
                await self.websocket.send_json({
                    "type": "generation.done",
                    "session_id": self.session_id,
                    "total_chunks": chunk_index,
                    "total_duration_ms": total_duration_ms,
                })
            except Exception:
                pass

    def _codes_to_pcm(self, audio_codes: list[int]) -> bytes:
        """Convert Moshi audio codes to PCM audio bytes.

        In the full implementation, this would run the Mimi decoder
        to convert RVQ codes → 24kHz float32 waveform → PCM bytes.

        For Phase 3 initial implementation, we generate 80ms of silence
        as a placeholder.
        TODO(Production): Integrate actual Mimi decoder.
        """
        if not audio_codes:
            return b""

        # 80ms at the configured sample rate
        sample_rate = self.config.sample_rate
        num_samples = int(sample_rate * _FRAME_DURATION_MS / 1000)
        # PCM s16le: 2 bytes per sample, all zeros = silence
        return b"\x00" * (num_samples * 2)

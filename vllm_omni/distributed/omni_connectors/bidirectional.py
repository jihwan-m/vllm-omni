# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Bidirectional communication primitives for full-duplex Moshi (Phase 3).

Provides:
  - RingBuffer: Thread-safe circular buffer for continuous audio token
    injection. Writer (session manager) appends user audio codes; reader
    (dialogue worker) reads at any position within the window.
  - OutputFrame / ControlSignal: Data types for channel communication.
  - BidirectionalChannel: Two-way channel between session manager and
    the duplex stage worker, combining a forward queue (audio output),
    a backward ring buffer (user audio input), and a control queue.
"""
from __future__ import annotations

import asyncio
import enum
import threading
from dataclasses import dataclass, field

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)


# ============================================================================
# Ring Buffer
# ============================================================================


class RingBuffer:
    """Thread-safe ring buffer for continuous audio token injection.

    The writer (async session manager / input pipeline) appends user audio
    codes one frame at a time. The reader (duplex stage worker thread)
    reads codes at any temporal position within the valid window.

    Thread safety: A threading.Lock protects write_pos and valid_from
    updates. The torch buffer is read-safe because the reader only
    accesses positions that are fully written (pos < write_pos).

    Attributes:
        capacity: Maximum number of frames stored (older frames evicted).
        num_codebooks: Number of RVQ codebook tokens per frame (8 for Moshi).
    """

    def __init__(self, capacity: int = 256, num_codebooks: int = 8):
        self.capacity = capacity
        self.num_codebooks = num_codebooks
        self.buffer = torch.zeros(capacity, num_codebooks, dtype=torch.long)
        self._write_pos = 0
        self._valid_from = 0
        self._lock = threading.Lock()

    @property
    def write_pos(self) -> int:
        """Current write position (number of frames written total)."""
        return self._write_pos

    @property
    def valid_from(self) -> int:
        """Oldest valid position still in the buffer."""
        return self._valid_from

    def write(self, codes: list[int]) -> int:
        """Append one frame of user audio codes.

        Args:
            codes: List of length num_codebooks (e.g., [c0, c1, ..., c7]).

        Returns:
            The position this frame was written to.

        Raises:
            ValueError: If len(codes) != num_codebooks.
        """
        if len(codes) != self.num_codebooks:
            raise ValueError(
                f"Expected {self.num_codebooks} codes, got {len(codes)}"
            )
        with self._lock:
            idx = self._write_pos % self.capacity
            self.buffer[idx] = torch.tensor(codes, dtype=torch.long)
            pos = self._write_pos
            self._write_pos += 1
            if self._write_pos - self._valid_from > self.capacity:
                self._valid_from = self._write_pos - self.capacity
            return pos

    def read(self, position: int) -> list[int] | None:
        """Read codes at a given temporal position.

        Returns None if the position has been evicted or not yet written.

        Args:
            position: Absolute temporal position to read.

        Returns:
            List of num_codebooks ints, or None if position is invalid.
        """
        with self._lock:
            if position < self._valid_from or position >= self._write_pos:
                return None
            idx = position % self.capacity
            return self.buffer[idx].tolist()

    def available(self) -> int:
        """Number of frames available for reading."""
        with self._lock:
            return self._write_pos - self._valid_from

    def reset(self):
        """Clear the buffer and reset positions."""
        with self._lock:
            self.buffer.zero_()
            self._write_pos = 0
            self._valid_from = 0


# ============================================================================
# Data Types
# ============================================================================


@dataclass
class OutputFrame:
    """A single frame of generated output from the dialogue stage.

    Attributes:
        audio_codes: RVQ codebook tokens [num_codebooks] for this frame.
        text_token: Sampled text token ID (or None if no text emitted).
        temporal_step: Which temporal position this frame corresponds to.
    """
    audio_codes: list[int]
    text_token: int | None = None
    temporal_step: int = 0


class ControlSignal(enum.Enum):
    """Control signals between session manager and stage worker."""
    INTERRUPT = "interrupt"       # Stop generation, reset state
    PAUSE = "pause"               # Pause generation (backpressure)
    RESUME = "resume"             # Resume after pause
    SHUTDOWN = "shutdown"         # Terminate worker


# ============================================================================
# Bidirectional Channel
# ============================================================================


class BidirectionalChannel:
    """Two-way communication channel for full-duplex Moshi sessions.

    Connects three concurrent components:
      - Input pipeline (async): writes user audio to backward_buffer
      - Output pipeline (async): reads generated audio from forward_queue
      - Dialogue worker (thread): reads user audio, writes generated audio

    Forward path (worker → client):
      worker calls put_output_sync() → forward_queue → session reads get_output()

    Backward path (client → worker):
      session calls put_input() → backward_buffer → worker reads get_input()

    Control path (bidirectional):
      session calls send_control() → control_queue → worker calls poll_control()
    """

    def __init__(
        self,
        session_id: str,
        buffer_capacity: int = 256,
        output_queue_size: int = 128,
        num_codebooks: int = 8,
    ):
        self.session_id = session_id
        self.num_codebooks = num_codebooks

        # Forward: stage worker → session manager (generated audio)
        self.forward_queue: asyncio.Queue[OutputFrame | None] = asyncio.Queue(
            maxsize=output_queue_size,
        )

        # Backward: session manager → stage worker (user audio)
        self.backward_buffer = RingBuffer(
            capacity=buffer_capacity,
            num_codebooks=num_codebooks,
        )

        # Control: session manager ↔ stage worker
        self.control_queue: asyncio.Queue[ControlSignal] = asyncio.Queue(
            maxsize=16,
        )

        self._closed = False
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop):
        """Bind the event loop for thread-safe async queue operations.

        Must be called from the async context before worker thread starts.
        """
        self._loop = loop

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self):
        """Mark channel as closed. Worker should check this each step."""
        self._closed = True
        # Push sentinel to unblock consumers
        if self._loop is not None:
            self._loop.call_soon_threadsafe(
                self.forward_queue.put_nowait, None,
            )

    # --- Forward path (worker → client) ---

    def put_output_sync(self, frame: OutputFrame):
        """Worker thread puts a generated output frame (thread-safe).

        Uses call_soon_threadsafe to enqueue from the worker thread
        into the async forward_queue.
        """
        if self._closed:
            return
        if self._loop is None:
            raise RuntimeError("Channel not bound to event loop")
        self._loop.call_soon_threadsafe(
            self.forward_queue.put_nowait, frame,
        )

    async def get_output(self, timeout: float = 5.0) -> OutputFrame | None:
        """Session manager gets next output frame (async).

        Returns None if channel is closed or timeout reached.
        """
        try:
            return await asyncio.wait_for(
                self.forward_queue.get(), timeout=timeout,
            )
        except asyncio.TimeoutError:
            return None

    # --- Backward path (client → worker) ---

    def put_input(self, user_audio_codes: list[int]) -> int:
        """Session manager writes a user audio frame to the ring buffer.

        Returns the position the frame was written to.
        """
        return self.backward_buffer.write(user_audio_codes)

    def get_input(self, position: int) -> list[int] | None:
        """Worker reads user audio codes at a given temporal position.

        Returns None if position hasn't been written yet or was evicted.
        The worker should use silence padding when None is returned.
        """
        return self.backward_buffer.read(position)

    @property
    def input_write_pos(self) -> int:
        """How many input frames have been written so far."""
        return self.backward_buffer.write_pos

    # --- Control path ---

    async def send_control(self, signal: ControlSignal):
        """Session manager sends a control signal to the worker."""
        await self.control_queue.put(signal)

    def poll_control(self) -> ControlSignal | None:
        """Worker polls for control signals (non-blocking).

        Called by the worker thread between temporal steps.
        """
        try:
            return self.control_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
WebSocket message protocol for real-time Moshi audio streaming.

Supports both Phase 2 (half-duplex, turn-taking) and Phase 3 (full-duplex,
concurrent bidirectional audio).

Message types:
  Client → Server:
    - session.start (JSON): Initialize session with model/audio config
    - audio.input (binary): Raw PCM audio chunk
    - audio.input.done (JSON): Signal user finished speaking (optional in P3)
    - interrupt (JSON): Request generation interruption (Phase 3)
    - session.end (JSON): Terminate session

  Server → Client:
    - session.created (JSON): Session accepted with assigned ID
    - audio.output (binary): PCM/Opus audio chunk
    - audio.output.meta (JSON): Metadata for preceding audio chunk
    - text.token (JSON): Intermediate text token (optional)
    - generation.done (JSON): Response generation complete
    - error (JSON): Error message
"""
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# =============================================================================
# Message Types
# =============================================================================


class DuplexMessageType(str, Enum):
    """All possible WebSocket message types."""

    # Client → Server
    SESSION_START = "session.start"
    AUDIO_INPUT = "audio.input"
    AUDIO_INPUT_DONE = "audio.input.done"
    INTERRUPT = "interrupt"
    SESSION_END = "session.end"

    # Server → Client
    SESSION_CREATED = "session.created"
    AUDIO_OUTPUT = "audio.output"
    AUDIO_OUTPUT_META = "audio.output.meta"
    TEXT_TOKEN = "text.token"
    GENERATION_DONE = "generation.done"
    ERROR = "error"


# =============================================================================
# Client → Server Messages
# =============================================================================


class SessionStartMessage(BaseModel):
    """Client sends this to initialize a streaming session."""

    type: str = DuplexMessageType.SESSION_START
    model: str = "moshi"
    sample_rate: int = Field(default=24000, description="Input audio sample rate in Hz")
    audio_format: str = Field(
        default="pcm_s16le",
        description="Input audio format (pcm_s16le, pcm_f32le)",
    )
    output_format: str = Field(
        default="pcm_s16le",
        description="Output audio format (pcm_s16le, pcm_f32le, opus)",
    )
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_k: int = Field(default=25, ge=0)
    max_duration_s: float = Field(
        default=30.0,
        ge=1.0,
        le=300.0,
        description="Maximum output audio duration in seconds",
    )
    # Phase 3: enable full-duplex (concurrent input during generation)
    duplex: bool = Field(
        default=False,
        description="Enable full-duplex mode (Phase 3). When True, client "
        "can send audio.input frames while receiving audio.output.",
    )


class AudioInputDoneMessage(BaseModel):
    """Client sends this to signal that audio input is complete.

    In Phase 2 (half-duplex), this marks the end of the input turn.
    In Phase 3 (full-duplex), this is optional — the client can continue
    sending audio.input even after sending this (for barge-in scenarios).
    """

    type: str = DuplexMessageType.AUDIO_INPUT_DONE


class InterruptMessage(BaseModel):
    """Client sends this to interrupt ongoing generation (Phase 3).

    The server will stop generation after finishing the current temporal
    step, clear KV cache, and prepare for a new turn.
    """

    type: str = DuplexMessageType.INTERRUPT


class SessionEndMessage(BaseModel):
    """Client sends this to terminate the session."""

    type: str = DuplexMessageType.SESSION_END


# =============================================================================
# Server → Client Messages
# =============================================================================


class SessionCreatedMessage(BaseModel):
    """Server acknowledges session creation."""

    type: str = DuplexMessageType.SESSION_CREATED
    session_id: str
    model: str
    sample_rate: int
    duplex: bool = False


class AudioOutputMeta(BaseModel):
    """Metadata for a preceding binary audio chunk."""

    type: str = DuplexMessageType.AUDIO_OUTPUT_META
    chunk_index: int
    duration_ms: float
    is_final: bool = False
    temporal_step: int | None = None
    text_token: int | None = None


class TextTokenMessage(BaseModel):
    """Intermediate text token from the temporal transformer."""

    type: str = DuplexMessageType.TEXT_TOKEN
    token_id: int
    text: str = ""
    step: int = 0


class GenerationDoneMessage(BaseModel):
    """Server signals that audio generation is complete."""

    type: str = DuplexMessageType.GENERATION_DONE
    session_id: str
    total_chunks: int
    total_duration_ms: float = 0.0


class ErrorMessage(BaseModel):
    """Server sends this on error."""

    type: str = DuplexMessageType.ERROR
    message: str
    code: str = "internal_error"
    recoverable: bool = False
    details: dict[str, Any] | None = None

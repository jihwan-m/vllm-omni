# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Event-driven stage worker for full-duplex Moshi (Phase 3).

Replaces the standard _stage_worker() blocking batch loop with a
continuous temporal step loop. Each iteration:
  1. Polls for control signals (interrupt, pause, shutdown)
  2. Reads user audio codes from the ring buffer at the current position
  3. Runs one temporal + depth step via forward_dialogue_step()
  4. Puts the generated output frame into the forward queue
  5. Advances temporal position

The worker runs in its own thread, communicating with the async
session manager via BidirectionalChannel.
"""
from __future__ import annotations

import time

import torch
from vllm.logger import init_logger

from vllm_omni.distributed.omni_connectors.bidirectional import (
    BidirectionalChannel,
    ControlSignal,
    OutputFrame,
)

logger = init_logger(__name__)

# Maximum temporal steps before forcing KV cache reset.
# Moshi supports ~3000 positions; reset at 2500 for safety.
_MAX_TEMPORAL_STEPS = 2500

# How long to sleep (seconds) when paused, to avoid busy-waiting.
_PAUSE_SLEEP = 0.05

# How long to wait for user audio before using silence (seconds).
# At 12.5 Hz, one frame = 80ms. We wait up to ~40ms for input to arrive.
_INPUT_WAIT_MS = 40


class DuplexStageWorker:
    """Event-driven stage worker for full-duplex Moshi dialogue.

    Drives the Moshi model step-by-step, reading user audio from the
    ring buffer and emitting generated audio+text to the forward queue.

    The worker does NOT use the standard vllm scheduler or stage worker
    loop. Instead, it directly calls model.forward_dialogue_step() in
    a tight loop, giving it full control over timing and state.

    Args:
        model: MoshiForConditionalGenerationVLLM (or compatible).
               Must have init_streaming_state(), forward_dialogue_step(),
               and clear_streaming_state() methods.
        channel: BidirectionalChannel for I/O with session manager.
        request_id: Unique session identifier for state isolation.
        temperature: Sampling temperature.
        top_k: Top-k sampling parameter.
        max_steps: Maximum temporal steps (0 = use _MAX_TEMPORAL_STEPS).
        device: Torch device for model inference.
    """

    def __init__(
        self,
        model,
        channel: BidirectionalChannel,
        request_id: str,
        temperature: float = 0.7,
        top_k: int = 50,
        max_steps: int = 0,
        device: torch.device | None = None,
    ):
        self.model = model
        self.channel = channel
        self.request_id = request_id
        self.temperature = temperature
        self.top_k = top_k
        self.max_steps = max_steps or _MAX_TEMPORAL_STEPS
        self.device = device

        self._paused = False
        self._temporal_step = 0

    def run(self):
        """Main event loop. Runs in a dedicated thread.

        Generates audio continuously until one of:
          - Channel is closed (client disconnected)
          - SHUTDOWN control signal received
          - max_steps reached (KV cache limit)
          - An unrecoverable error occurs
        """
        logger.info(
            "DuplexStageWorker %s: starting (max_steps=%d, temp=%.2f)",
            self.request_id, self.max_steps, self.temperature,
        )

        # Initialize per-request streaming state on the model
        self.model.init_streaming_state(
            request_id=self.request_id,
            device=self.device,
        )

        try:
            self._generation_loop()
        except Exception as e:
            logger.exception(
                "DuplexStageWorker %s: error at step %d",
                self.request_id, self._temporal_step,
            )
            # Push error frame to forward queue so session manager knows
            try:
                self.channel.put_output_sync(
                    OutputFrame(
                        audio_codes=[],
                        text_token=None,
                        temporal_step=self._temporal_step,
                    ),
                )
            except Exception:
                pass
        finally:
            self.model.clear_streaming_state(request_id=self.request_id)
            logger.info(
                "DuplexStageWorker %s: stopped after %d steps",
                self.request_id, self._temporal_step,
            )

    def _generation_loop(self):
        """Core generation loop — one iteration per 80ms temporal frame."""
        num_codebooks = self.channel.num_codebooks

        while (
            not self.channel.closed
            and self._temporal_step < self.max_steps
        ):
            # 1. Poll control signals
            action = self._handle_control()
            if action == "shutdown":
                break
            if action == "paused":
                time.sleep(_PAUSE_SLEEP)
                continue

            # 2. Read user audio codes for current temporal step
            user_codes = self._get_user_audio(num_codebooks)

            # 3. Run one temporal + depth step
            result = self.model.forward_dialogue_step(
                user_audio_codes=user_codes,
                temperature=self.temperature,
                top_k=self.top_k,
                request_id=self.request_id,
            )

            # 4. Emit output frame
            frame = OutputFrame(
                audio_codes=result["audio_codes"],
                text_token=result["text_token"],
                temporal_step=result["temporal_step"],
            )
            self.channel.put_output_sync(frame)

            # 5. Advance
            self._temporal_step += 1

        # Signal end of generation
        if not self.channel.closed:
            self.channel.put_output_sync(None)

    def _handle_control(self) -> str | None:
        """Process any pending control signals.

        Returns:
            "shutdown" if worker should stop, "paused" if paused,
            None to continue normally.
        """
        while True:
            signal = self.channel.poll_control()
            if signal is None:
                break

            if signal == ControlSignal.SHUTDOWN:
                logger.info(
                    "DuplexStageWorker %s: received SHUTDOWN",
                    self.request_id,
                )
                return "shutdown"

            if signal == ControlSignal.INTERRUPT:
                logger.info(
                    "DuplexStageWorker %s: received INTERRUPT at step %d",
                    self.request_id, self._temporal_step,
                )
                self._handle_interrupt()

            elif signal == ControlSignal.PAUSE:
                logger.info(
                    "DuplexStageWorker %s: paused", self.request_id,
                )
                self._paused = True

            elif signal == ControlSignal.RESUME:
                logger.info(
                    "DuplexStageWorker %s: resumed", self.request_id,
                )
                self._paused = False

        if self._paused:
            return "paused"
        return None

    def _handle_interrupt(self):
        """Reset generation state for a new turn.

        Clears KV cache and streaming state, resets temporal position.
        The model starts fresh as if the conversation began anew.
        """
        self.model.clear_streaming_state(request_id=self.request_id)
        self.model.init_streaming_state(
            request_id=self.request_id,
            device=self.device,
        )
        self._temporal_step = 0
        self._paused = False

    def _get_user_audio(self, num_codebooks: int) -> list[int] | None:
        """Get user audio codes for the current temporal step.

        If the user hasn't provided audio for this position yet, waits
        briefly then falls back to silence (None), which the model
        interprets as silence padding.
        """
        codes = self.channel.get_input(self._temporal_step)
        if codes is not None:
            return codes

        # Brief wait for input to arrive (user audio may lag slightly)
        wait_end = time.monotonic() + _INPUT_WAIT_MS / 1000.0
        while time.monotonic() < wait_end:
            codes = self.channel.get_input(self._temporal_step)
            if codes is not None:
                return codes
            time.sleep(0.005)  # 5ms poll interval

        # No user audio available — use silence
        return None
